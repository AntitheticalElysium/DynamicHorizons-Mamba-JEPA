"""Corrected EDA questions on freshly encoded M03 forks (no legacy probes).

All uncertainty resamples rollout seeds, retaining every root and sibling action.
Coordinates are compared only inside an arm; cross-arm comparisons use semantics.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..diagnostics import binary_auc
from .cache import memoized, PROBE_SETTINGS
from .gate import _bootstrap_mean_draws, _bootstrap_bounds


def decompose(values: Tensor) -> dict[str, Tensor]:
    x = values.double()
    grand = x.mean((0, 1), keepdim=True)
    root = x.mean(1, keepdim=True) - grand
    action = x.mean(0, keepdim=True) - grand
    return {"grand": grand, "root": root, "action": action,
            "interaction": x - grand - root - action}


def within_root_auc(scores: Tensor, truth: Tensor) -> Tensor:
    """NaN marks roots without both classes, never an invented chance score."""
    values = [binary_auc(s, y.bool()) if bool(y.any() and (~y.bool()).any()) else None
              for s, y in zip(scores, truth)]
    return torch.tensor([float("nan") if v is None else v for v in values])


@memoized()
def transfer_summary(observed, generated, floor, truth, clusters, settings):
    from .gate import _root_bootstrap

    obs, pred, prior = (within_root_auc(x, truth) for x in (observed, generated, floor))
    lethal = truth.sum(1)
    strata = {"all_opportunities": torch.isfinite(obs),
              "escape_rich": (lethal >= 1) & (lethal <= 2),
              "middle": (lethal >= 3) & (lethal <= 13),
              "trap_heavy": (lethal >= 14) & (lethal <= 16)}
    result = {"bootstrap_unit": "rollout_seed", "root_auc_observed": obs.nan_to_num(-1).tolist(),
              "root_auc_generated": pred.nan_to_num(-1).tolist(),
              "root_auc_action_only": prior.nan_to_num(-1).tolist(),
              "unsupported_root_sentinel": -1, "strata": {}}
    for name, mask in strata.items():
        mask = mask & torch.isfinite(obs)
        draws = _bootstrap_mean_draws(torch.stack((obs,pred,prior),1),mask,clusters,
                                     draws=settings.bootstrap_draws,seed=settings.seed+2700)
        a,p,f = draws.T
        recovery = np.full(len(a),np.nan)
        np.divide(p-.5,a-.5,out=recovery,where=a>.500001)
        sampled = {'observed_auc':a,'generated_auc':p,'generated_minus_action_only':p-f,
                   'generated_minus_observed':p-a,'recovery':recovery}
        def estimate(kind):
            def metric(rows):
                rows = rows[mask[rows]]
                if not len(rows):
                    return None
                a, p, f = (float(x[rows].mean()) for x in (obs, pred, prior))
                if kind == "recovery":
                    return (p - .5) / (a - .5) if a > .500001 else None
                return {"observed_auc": a, "generated_auc": p,
                        "generated_minus_action_only": p-f, "generated_minus_observed": p-a}[kind]
            point = metric(torch.arange(len(clusters)))
            ci = None
            if point is not None and len(clusters.unique()) >= 2:
                if kind == 'recovery' and np.any((a>.500001) & (a<.505)):
                    # Near chance, division amplifies float32 reduction error.
                    _,ci = _root_bootstrap(pred,clusters,metric,draws=settings.bootstrap_draws,seed=settings.seed+2700)
                else:
                    ci = _bootstrap_bounds(sampled[kind],settings.bootstrap_draws)
            return {"value": point, "interval": ci,
                    "status": "measured" if ci is not None else "insufficient_coverage"}
        result["strata"][name] = {"roots": int(mask.sum()), "seed_clusters": len(clusters[mask].unique()),
                                 **{k: estimate(k) for k in ("observed_auc", "generated_auc",
                                     "generated_minus_action_only", "generated_minus_observed", "recovery")}}
    return result


@memoized()
def geometry_report(observed: Tensor, generated: Tensor, clusters: Tensor, settings,
                    successors: Tensor) -> dict:
    from .gate import _root_bootstrap
    from scipy.optimize import linear_sum_assignment

    true, pred = observed.double(), generated.double()
    effect, hat = true-true.mean(1, keepdim=True), pred-pred.mean(1, keepdim=True)
    error = (pred-true).square().sum((1, 2))
    effect_error = (hat-effect).square().sum((1, 2))
    energy = effect.square().sum((1, 2))
    common_error = 17*(pred.mean(1)-true.mean(1)).square().sum(1)
    torch.testing.assert_close(error, common_error+effect_error)
    def ratio(numerator, denominator):
        def metric(rows):
            d = float(denominator[rows].sum())
            return float(numerator[rows].sum())/d if d > 1e-12 else None
        p, ci = _root_bootstrap(error, clusters, metric, draws=settings.bootstrap_draws, seed=settings.seed+2800)
        return {"value": p, "interval": ci}
    retrieval, chance, assignment, class_counts = [], [], [], []
    ranks, margins, geometry, same_distance, different_distance = [], [], [], [], []
    per_action = [[] for _ in range(17)]
    # Pixel equality is independent of the evaluated representation. Do not merge
    # actions merely because six coarse consequences or collapsed latents agree.
    for index, (t, p, pixels) in enumerate(zip(effect, hat, successors)):
        _, classes = torch.unique(pixels.flatten(1), dim=0, return_inverse=True)
        class_counts.append(len(classes.unique()))
        if float(energy[index]) <= 1e-12:
            retrieval.append(float("nan")); chance.append(float("nan")); assignment.append(float("nan"))
            continue
        distances = torch.cdist(p, t).square()
        nearest = distances.argmin(1)
        retrieval.append(float((classes[nearest] == classes).double().mean()))
        chance.append(float((classes[:, None] == classes[None]).double().mean()))
        row, col = linear_sum_assignment(distances.numpy())
        assignment.append(float((classes[row] == classes[col]).double().mean()))
        equal = classes[:, None] == classes[None]
        correct = distances.masked_fill(~equal, float("inf")).min(1).values
        wrong = distances.masked_fill(equal, float("inf")).min(1).values
        ranks.append(float((1+(distances < correct[:, None]).sum(1)).double().mean()))
        margins.append(float((wrong-correct).mean()) if len(classes.unique()) > 1 else float("nan"))
        first, second = torch.triu_indices(17, 17, 1)
        td, pd = torch.pdist(t).square(), torch.pdist(p).square()
        geometry.append(float(torch.corrcoef(torch.stack((td, pd)))[0, 1])
                        if float(td.std()) > 1e-12 and float(pd.std()) > 1e-12 else float("nan"))
        same = equal[first, second]
        same_distance.append(float(pd[same].mean()) if bool(same.any()) else float("nan"))
        different_distance.append(float(pd[~same].mean()) if bool((~same).any()) else float("nan"))
        for a in range(17):
            per_action[a].append(float(classes[nearest[a]] == classes[a]))
    def mean_report(values):
        values = torch.tensor(values)
        def metric(rows):
            kept = values[rows]; kept = kept[torch.isfinite(kept)]
            return float(kept.mean()) if len(kept) else None
        p, ci = _root_bootstrap(values, clusters, metric, draws=settings.bootstrap_draws, seed=settings.seed+2801)
        return {"value": p, "interval": ci}
    parts = {}
    for label, x in (("observed", true), ("generated", pred), ("error", pred-true)):
        d = decompose(x)
        total = float((x-d["grand"]).square().sum())
        energies = {k: float(v.expand_as(x).square().sum()) for k, v in d.items() if k != "grand"}
        if abs(sum(energies.values())-total) > 1e-6*max(total, 1):
            raise ValueError("m03_geometry: action decomposition is not orthogonal")
        parts[label] = {"energy": energies, "shares": {k: v/total if total > 1e-12 else None for k,v in energies.items()}}
    return {"scope": "within_arm_coordinates_descriptive_only", "equivalence": "exact_successor_pixels",
            "zero_effect_roots": int((energy <= 1e-12).sum()), "successor_classes_per_root": class_counts,
            "effect_nse_energy_weighted": ratio(effect_error, energy),
            "effect_share_of_error": ratio(effect_error, error),
            "common_share_of_error": ratio(common_error, error),
            "retrieval": mean_report(retrieval), "equivalence_chance": mean_report(chance),
            "hungarian_identity_free": mean_report(assignment), "decomposition": parts,
            # Geometry sublists below exclude zero-effect roots; preserve their
            # point values only rather than applying misaligned bootstrap IDs.
            "correct_class_rank": float(torch.tensor(ranks).mean()) if ranks else None,
            "correct_class_margin": _finite_mean(margins), "distance_geometry_correlation": _finite_mean(geometry),
            "predicted_squared_distance_equivalent": _finite_mean(same_distance),
            "predicted_squared_distance_distinct": _finite_mean(different_distance),
            "retrieval_by_action": [_finite_mean(v) for v in per_action]}


def _finite_mean(values):
    value = torch.tensor(values)
    value = value[torch.isfinite(value)]
    return float(value.mean()) if len(value) else None


def eda_report(train, dev, features, settings, *, device):
    """No-action successor transfer, delta/centred/residual probes, and controls."""
    from .gate import (OUTCOME_BINARY, _fit_probe_many, _one_hot_actions,
                                 _paired_binary_difference, _binary_metrics)

    names = OUTCOME_BINARY + ("damage_or_death",)
    def augmented(values):
        truth = values["outcomes"]
        combined = truth[..., OUTCOME_BINARY.index("damage")] | truth[..., OUTCOME_BINARY.index("death")]
        return torch.cat((truth, combined[..., None]), -1)
    labels = augmented(train).flatten(0, 1).float().to(device)
    truth, clusters = augmented(dev), dev["episode"]
    ta, da = (_one_hot_actions(len(x["outcomes"]), device=device) for x in (train, dev))
    report, saved = {"arms": {}, "paired_differences": {}}, {}
    for arm, x in features.items():
        tr, dr = x["train_projected"].float(), x["dev_projected"].float()
        to, ob, ge = (x[k].float() for k in ("train_observed_successor", "dev_observed_successor", "dev_generated_successor"))
        # FIT action marginal is frozen before DEV, independently for each arm.
        marginal = to.mean(0, keepdim=True)-to.mean((0, 1), keepdim=True)
        inputs = {"successor_only": (to, ob, ge),
                  "delta": (to-tr[:, None], ob-dr[:, None], ge-dr[:, None]),
                  "centred_effect": (to-to.mean(1, keepdim=True), ob-ob.mean(1, keepdim=True), ge-ge.mean(1, keepdim=True)),
                  "residualised": (to-marginal, ob-marginal, ge-marginal)}
        arm_report = {"geometry": geometry_report(ob, ge, clusters, settings, dev["successors"]), "probes": {}}
        saved[arm] = {}
        for hidden, family in ((False, "linear"), (True, "mlp")):
            floor = _fit_probe_many(ta, labels, {"dev": da}, settings, hidden=hidden, binary=True)["dev"].reshape_as(truth)
            fitted = {}
            for name, (fit, observed, generated) in inputs.items():
                evaluation = {"observed": observed.flatten(0, 1), "generated": generated.flatten(0, 1)}
                if name == "successor_only":
                    evaluation["persistence"] = dr.repeat_interleave(17, 0)
                    rng = torch.Generator().manual_seed(settings.seed+2900)
                    order = torch.stack([torch.randperm(17, generator=rng) for _ in ge])
                    evaluation["candidate_action_shuffled"] = ge.gather(1, order[..., None].expand_as(ge)).flatten(0, 1)
                if name == "delta":
                    # Oracle intervention, not a deployable correction. Every
                    # alpha uses the very same observed-delta fitted decoder.
                    for alpha in (.25, .5, .75):
                        evaluation[f"residual_restoration_{alpha}"] = (generated+alpha*(observed-generated)).flatten(0, 1)
                scores = _fit_probe_many(fit.flatten(0, 1).to(device), labels, evaluation,
                                         settings, hidden=hidden, binary=True)
                scores = {k: v.reshape_as(truth) for k,v in scores.items()}
                fitted[name] = {target: transfer_summary(scores["observed"][..., i], scores["generated"][..., i],
                                                        floor[..., i], truth[..., i], clusters, settings)
                                for i,target in enumerate(names)}
                if name == "successor_only":
                    saved[arm][family] = scores["generated"]
                    fitted[name]["controls"] = {k: _binary_metrics(v, truth, clusters, names, settings)
                                                for k,v in scores.items() if k not in ("observed", "generated")}
                if name == "delta":
                    fitted[name]["oracle_residual_restoration"] = {
                        k: {target: transfer_summary(scores["observed"][..., i], v[..., i], floor[..., i],
                                                    truth[..., i], clusters, settings)
                            for i,target in enumerate(names)}
                        for k,v in scores.items() if k.startswith("residual_restoration_")}
            root_only = _fit_probe_many(tr.repeat_interleave(17, 0).to(device), labels,
                                        {"dev": dr.repeat_interleave(17, 0)}, settings, hidden=hidden, binary=True)["dev"]
            fitted["state_only"] = _binary_metrics(root_only, truth, clusters, names, settings)
            constant = labels.mean(0).cpu().expand(*truth.shape)
            fitted["constant"] = _binary_metrics(constant, truth, clusters, names, settings)
            arm_report["probes"][family] = fitted
        report["arms"][arm] = arm_report
    for left, right in (("tc", "raw"), ("raw", "direct_attention"), ("raw", "direct_mamba"),
                        ("tc", "direct_attention"), ("tc", "direct_mamba")):
        if left in saved and right in saved:
            report["paired_differences"][f"{left}_minus_{right}"] = {
                family: _paired_binary_difference(saved[left][family], saved[right][family], truth, clusters,
                                                   names, settings) for family in ("linear", "mlp")}
    from .gate import _root_bootstrap
    report["paired_within_root_auc"] = {}
    for comparison in report["paired_differences"]:
        left, right = comparison.split("_minus_")
        report["paired_within_root_auc"][comparison] = {}
        for family in ("linear", "mlp"):
            target_report = {}
            for i,name in enumerate(names):
                delta = within_root_auc(saved[left][family][..., i], truth[..., i])-within_root_auc(saved[right][family][..., i], truth[..., i])
                def mean(rows):
                    value = delta[rows]; value = value[torch.isfinite(value)]
                    return float(value.mean()) if len(value) else None
                point, ci = _root_bootstrap(delta, clusters, mean, draws=settings.bootstrap_draws, seed=settings.seed+3100)
                target_report[name] = {"difference": point, "interval": ci, "bootstrap_unit": "rollout_seed"}
            report["paired_within_root_auc"][comparison][family] = target_report
    return report


def state_features(state):
    import jax
    position, board = np.asarray(state.player_position), np.asarray(state.map)
    visible = [float(getattr(state, f"player_{name}")) for name in ("health", "food", "drink", "energy")]
    visible += [float(state.is_sleeping), float(state.light_level)]
    visible += [float(int(state.player_direction) == i) for i in range(5)]
    visible += [float(getattr(state.inventory, k)) for k in (
        "wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe", "stone_pickaxe",
        "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")]
    for dx in range(-4, 5):
        for dy in range(-3, 4):
            x, y = position + (dx, dy)
            tile = int(board[x, y]) if 0 <= x < board.shape[0] and 0 <= y < board.shape[1] else 1
            visible += [float(tile == i) for i in range(17)]
    timing = [float(getattr(state, f"player_{k}")) for k in ("recover", "hunger", "thirst", "fatigue")]
    for name in ("zombies", "skeletons", "cows", "arrows"):
        mobs = getattr(state, name)
        offsets = np.asarray(mobs.position)-position
        inside = np.asarray(mobs.mask) & (np.abs(offsets[:, 0]) <= 4) & (np.abs(offsets[:, 1]) <= 3)
        order = np.argsort(np.where(inside, np.abs(offsets).sum(-1), 999), kind="stable")[:3]
        for i in order:
            live = bool(inside[i])
            visible += [float(live), float(offsets[i, 0])*live, float(offsets[i, 1])*live]
            timing.append(float(mobs.attack_cooldown[i])*live)
        for _ in range(3-len(order)):
            visible += [0., 0., 0.]; timing.append(0.)
    full = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in jax.tree.leaves(state)])
    return {"oracle_visible": torch.tensor(visible), "oracle_timing": torch.tensor(timing),
            "oracle_full_state": torch.from_numpy(full)}


@memoized(PROBE_SETTINGS)
def _fit_oracle_probe(train_roots, train_y, dev_roots, settings, *, device, hidden):
    """Same root+action probe, without materializing seventeen copies on CUDA."""
    from torch import nn
    train_roots = train_roots.float().to(device)
    dev_roots = dev_roots.float().to(device)
    train_y = train_y.flatten(0,1).float().to(device)
    mean = train_roots.mean(0,keepdim=True)
    scale = train_roots.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
    train_roots = (train_roots-mean)/scale
    dev_roots = (dev_roots-mean)/scale
    actions = torch.eye(17,device=device)
    actions = (actions-actions.mean(0,keepdim=True))/actions.std(0,unbiased=False,keepdim=True)
    seed = settings.seed + (19 if hidden else 7) + 100
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if str(device).startswith('cuda') else []):
        torch.manual_seed(seed)
        width = train_roots.shape[1]+17
        model = (nn.Sequential(nn.Linear(width,settings.probe_hidden),nn.GELU(),nn.Linear(settings.probe_hidden,train_y.shape[1]))
                 if hidden else nn.Linear(width,train_y.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings.probe_learning_rate,weight_decay=settings.probe_weight_decay)
    positive = train_y.sum(0)
    weight = ((len(train_y)-positive)/positive.clamp_min(1)).clamp(max=100.)
    rng = torch.Generator(device=device).manual_seed(seed+1)
    for _ in range(settings.probe_steps):
        index = torch.randint(len(train_y),(min(settings.probe_batch,len(train_y)),),device=device,generator=rng)
        x = torch.cat((train_roots[index//17],actions[index%17]),1)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x),train_y[index],pos_weight=weight)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    prediction = []
    with torch.no_grad():
        model.eval()
        for start in range(0,len(dev_roots)*17,settings.probe_batch):
            index = torch.arange(start,min(start+settings.probe_batch,len(dev_roots)*17),device=device)
            prediction.append(model(torch.cat((dev_roots[index//17],actions[index%17]),1)).cpu())
    return torch.cat(prediction)


def observability_report(train, dev, settings, *, device):
    from .gate import OUTCOME_BINARY, _binary_metrics, _root_bootstrap
    truth = dev['outcomes']
    output = {"scope": "privileged_pre_state_controls_only", "sources": {}}
    for name in ("pixels", "visible", "visible_timing", "full_simulator"):
        def inputs(values):
            if name == "pixels":
                return values["context"][:, -1].flatten(1).float()/255
            if name == "visible_timing":
                return torch.cat((values["oracle_visible"], values["oracle_timing"]), 1)
            return values["oracle_full_state" if name == "full_simulator" else "oracle_visible"]
        fit, test = inputs(train), inputs(dev)
        output["sources"][name] = {}
        for hidden, family in ((False, "linear"), (True, "mlp")):
            scores = _fit_oracle_probe(fit,train['outcomes'],test,settings,device=device,hidden=hidden).reshape_as(truth)
            metric = _binary_metrics(scores, truth, dev["episode"], OUTCOME_BINARY, settings)
            metric["within_root"] = {}
            for i,target in enumerate(OUTCOME_BINARY):
                auc = within_root_auc(scores[..., i], truth[..., i])
                def mean(rows):
                    valid = auc[rows]; valid = valid[torch.isfinite(valid)]
                    return float(valid.mean()) if len(valid) else None
                point, ci = _root_bootstrap(auc, dev["episode"], mean, draws=settings.bootstrap_draws, seed=settings.seed+3000)
                metric["within_root"][target] = {"auc": point, "interval": ci, "opportunity_roots": int(torch.isfinite(auc).sum())}
            output["sources"][name][family] = metric
    return output


# These are evaluation prefixes, not changes to TC's four-frame SIGReg window.
MEMORY_CONTEXTS = (1, 4, 16, 64)
MEMORY_CASES = ('c4', 'c1', 'c4_shuffled', 'c16', 'c16_shuffled', 'c64', 'c64_shuffled')


def shuffle_completed_pairs(z, actions, episodes, times, seed):
    """Permute completed pairs within each root; never touch its current frame."""
    from hashlib import sha256
    result, shuffled = z.clone(), actions.clone()
    n = actions.shape[1]
    for row, (episode, time) in enumerate(zip(episodes.tolist(), times.tolist())):
        if n < 2:
            continue
        key = int.from_bytes(sha256(f'm03-history:{seed}:{episode}:{time}:{n}'.encode()).digest()[:8], 'little')
        order = torch.randperm(n, generator=torch.Generator().manual_seed(key))
        if torch.equal(order, torch.arange(n)):
            order = order.roll(1)
        order = order.to(z.device)
        result[row, :n] = z[row, order]
        shuffled[row] = actions[row, order]
    return result, shuffled


@torch.inference_mode()
def encode_memory(bundle, values, settings):
    """Keep projected z and Mamba h before the fresh, untrained agent readout.

    h_next and carry are computed from (z_t, a_t), before either real or
    predicted z_next is accepted. Both successor branches share that exact h.
    Short BOS prefixes are grouped before encoding; padded pixels/actions never
    enter the model. Only completed pairs are shuffled, with a row-stable RNG.
    """
    pieces, positions = [], []
    lengths = values.get('context_length', torch.full((len(values['context']),), values['context'].shape[1]))
    for length in lengths.unique(sorted=True).tolist():
        members = torch.where(lengths == length)[0]
        if not 1 <= length <= values['context'].shape[1]:
            raise ValueError('m03_memory: invalid physical prefix length')
        for start in range(0, len(members), settings.encode_batch):
            ix = members[start:start+settings.encode_batch]
            frames = values['context'][ix, -length:].to(bundle.device)
            actions = values['past_actions'][ix, -(length-1):].to(bundle.device) if length > 1 else values['past_actions'][ix, :0].to(bundle.device)
            z = bundle.encoder(frames)
            count = len(ix)
            observed = bundle.encoder(values['successors'][ix].flatten(0, 1)[:, None].to(bundle.device))[:, 0, 0].reshape(count, 17, -1)
            row = {'root_z': z[:, -1, 0].cpu(), 'observed_z': observed.cpu(), 'length': lengths[ix]}
            for case in MEMORY_CASES:
                context = int(case.split('_')[0][1:])
                use = min(context, length)
                zz = z[:, -use:]
                aa = actions[:, -(use-1):] if use > 1 else actions[:, :0]
                if case.endswith('_shuffled'):
                    zz, aa = shuffle_completed_pairs(zz, aa, values['episode'][ix], values['time'][ix], settings.seed)
                state = bundle.prefill(zz, aa)
                branches = bundle.repeat_state(state, 17)
                candidate_actions = torch.arange(17, device=bundle.device).repeat(count)[:, None]
                predicted, _ = bundle.advance(branches, candidate_actions)
                row[f'{case}_root_h'] = state.history[:, 0].cpu()
                row[f'{case}_next_h'] = predicted.history[:, 0].reshape(count, 17, -1).cpu()
                row[f'{case}_generated_z'] = predicted.latent[:, 0, 0].reshape(count, 17, -1).cpu()
                # Summaries expose internal carry growth without fitting an arbitrary
                # flattened SSM-state decoder or treating a norm as semantic success.
                row[f'{case}_carry_rms'] = torch.stack([
                    torch.stack([c.conv.float().flatten(1).square().mean(1).sqrt(),
                                 c.ssm.float().flatten(1).square().mean(1).sqrt()], -1)
                    for c in state.memory], 1).cpu()
            if not all(torch.isfinite(v).all() for v in row.values()):
                raise ValueError('m03_memory: nonfinite state in context sweep')
            pieces.append(row); positions.append(ix)
    order = torch.argsort(torch.cat(positions))
    return {key: torch.cat([p[key] for p in pieces])[order] for key in pieces[0]}


def memory_view(features, case, representation, state):
    """Equal parameter count: z/h ablations occupy fixed slots in [z,h]."""
    z = features['root_z'] if state == 'root' else (features['observed_z'] if state == 'observed' else features[f'{case}_generated_z'])
    h = features[f'{case}_root_h' if state == 'root' else f'{case}_next_h']
    if representation not in ('z', 'h', 'joint'):
        raise ValueError('m03_memory: unknown representation')
    return torch.cat((z if representation != 'h' else torch.zeros_like(z),
                      h if representation != 'z' else torch.zeros_like(h)), -1)


def memory_report(train, dev, features, settings, *, device, progress=None):
    """TRAIN-c4 probes frozen across prefix interventions and successor branches.

    Longer-prefix results measure transfer of the same decoder, not a probe
    refitted on DEV or a search for the context with the highest score.
    """
    from .gate import (STATIC_BINARY, STATIC_CONTINUOUS, OUTCOME_BINARY, _fit_probe_many,
                      _binary_metrics, _regression_metrics, _paired_binary_difference,
                      _paired_regression_difference, _fit_label_coverage, _expanded_roots)
    outcome_names = OUTCOME_BINARY + ('damage_or_death',)
    def outcomes(data):
        y = data['outcomes']
        return torch.cat((y, (y[..., 0] | y[..., 1])[..., None]), -1)
    tasks = {
        'root_binary': ('root', train['root_binary'], dev['root_binary'], STATIC_BINARY, True),
        'root_continuous': ('root', train['root_continuous'], dev['root_continuous'], STATIC_CONTINUOUS, False),
        'outcomes': ('observed', outcomes(train), outcomes(dev), outcome_names, True),
        'successor_binary': ('observed', train['next_binary'], dev['next_binary'], STATIC_BINARY, True),
        'successor_continuous': ('observed', train['next_continuous'], dev['next_continuous'], STATIC_CONTINUOUS, False),
    }
    report = {'probe_fit_context': 4, 'contexts': MEMORY_CONTEXTS,
              'long_context_interpretation': 'frozen_c4_decoder_transfer_beyond_joint_training_span',
              'capacity': 'zero-padded [z,h] inputs; equal parameter counts and fixed hidden width',
              'alignment': 'h_next consumes current z and chosen action; shared by real and generated successors',
              'bootstrap_unit': 'episode_seed', 'arms': {}, 'paired': {}, 'fit_coverage': {},
              'controls': 'same probe under reset, within-prefix paired shuffle, and truncation; action-only and z-only baselines',
              'm4_authorized': False}
    # Retain small CPU prediction matrices for paired comparisons; no fitted
    # model, privileged target, or oracle enters the world-state extraction.
    predictions = {}
    for task, (state, train_y, truth, names, binary) in tasks.items():
        roots = dev['episode'] if state == 'root' else _expanded_roots(dev['episode'], truth[..., 0])
        truth = truth.reshape(-1, len(names))
        if binary:
            report['fit_coverage'][task] = _fit_label_coverage(train_y, train['episode'], names, settings)
        predictions[task] = {}
        from hashlib import sha256
        # Identical z-only views and shared h-only successors recur across
        # conditions. Reuse their exact statistics, including paired draws.
        metric_fn = _binary_metrics if binary else _regression_metrics
        paired_fn = _paired_binary_difference if binary else _paired_regression_difference
        metric_cache, paired_cache = {}, {}
        def tensor_key(value):
            return sha256(value.contiguous().numpy().tobytes()).digest()
        def metric(value, truth, roots, names, settings):
            key = tensor_key(value)
            if key not in metric_cache:
                metric_cache[key] = metric_fn(value, truth, roots, names, settings)
            return metric_cache[key]
        def paired(left, right, truth, roots, names, settings):
            key = (tensor_key(left), tensor_key(right), tensor_key(roots))
            if key not in paired_cache:
                paired_cache[key] = paired_fn(left, right, truth, roots, names, settings)
            return paired_cache[key]
        for arm, split in features.items():
            report['arms'].setdefault(arm, {})[task] = {}
            for representation in ('z', 'h', 'joint'):
                report['arms'][arm][task][representation] = {}
                x = memory_view(split['train'], 'c4', representation, state).flatten(0, 1) if state != 'root' else memory_view(split['train'], 'c4', representation, state)
                evaluation = {f'{case}:{target}': memory_view(split['dev'], case, representation, target).reshape(-1, x.shape[-1]).to(device)
                              for case in MEMORY_CASES for target in (('root',) if state == 'root' else ('observed', 'generated'))}
                for hidden, family in ((False, 'linear'), (True, 'mlp')):
                    score = _fit_probe_many(x.to(device), train_y.reshape(-1, len(names)).float().to(device), evaluation,
                                            settings, hidden=hidden, binary=binary)
                    key = f'{arm}:{representation}:{family}'
                    predictions[task][key] = score
                    report['arms'][arm][task][representation][family] = {
                        k: metric(v, truth, roots, names, settings) for k, v in score.items()}
        report['paired'][task] = {}
        for family in ('linear', 'mlp'):
            for arm in features:
                for representation in ('z', 'h', 'joint'):
                    scores = predictions[task][f'{arm}:{representation}:{family}']
                    target = 'root' if state == 'root' else 'generated'
                    contrasts = {f'c4_minus_{case}': (scores[f'c4:{target}'], scores[f'{case}:{target}'])
                                 for case in MEMORY_CASES if case != 'c4'}
                    contrasts.update({f'c{c}_minus_shuffled': (scores[f'c{c}:{target}'], scores[f'c{c}_shuffled:{target}']) for c in (4,16,64)})
                    if state != 'root':
                        contrasts.update({f'{case}_generated_minus_observed': (scores[f'{case}:generated'], scores[f'{case}:observed']) for case in MEMORY_CASES})
                    for name, (left, right) in contrasts.items():
                        report['paired'][task][f'{arm}:{representation}:{family}:{name}'] = paired(left, right, truth, roots, names, settings)
                joint = predictions[task][f'{arm}:joint:{family}']
                z_only = predictions[task][f'{arm}:z:{family}']
                for case in MEMORY_CASES:
                    for target in (('root',) if state == 'root' else ('observed', 'generated')):
                        name = f'{arm}:{family}:{case}:{target}:joint_minus_z'
                        report['paired'][task][name] = paired(joint[f'{case}:{target}'], z_only[f'{case}:{target}'], truth, roots, names, settings)
            if set(features) == {'raw', 'tc'}:
                for representation in ('z', 'h', 'joint'):
                    for case in MEMORY_CASES:
                        target = 'root' if state == 'root' else 'generated'
                        report['paired'][task][f'{family}:{representation}:{case}:tc_minus_raw'] = paired(
                            predictions[task][f'tc:{representation}:{family}'][f'{case}:{target}'],
                            predictions[task][f'raw:{representation}:{family}'][f'{case}:{target}'], truth, roots, names, settings)
        if task == 'outcomes':
            # A Mamba state can encode the candidate action. Require incremental
            # discrimination over an independently fitted action-only decoder.
            actions = torch.eye(17).repeat(len(train_y), 1).to(device)
            dev_actions = torch.eye(17).repeat(len(dev['outcomes']), 1).to(device)
            report['action_only'] = {}
            for hidden, family in ((False, 'linear'), (True, 'mlp')):
                action_score = _fit_probe_many(actions, train_y.flatten(0, 1).float().to(device), {'dev': dev_actions}, settings,
                                               hidden=hidden, binary=True)['dev']
                report['action_only'][family] = metric(action_score, truth, roots, names, settings)
                for arm in features:
                    scores = predictions[task][f'{arm}:joint:{family}']
                    for case in MEMORY_CASES:
                        generated = scores[f'{case}:generated']
                        observed = scores[f'{case}:observed']
                        report['paired'][task][f'{arm}:{family}:{case}:generated_minus_action'] = paired(generated, action_score, truth, roots, names, settings)
                        report['arms'][arm][task]['joint'][family][f'{case}:within_root'] = {
                            name: transfer_summary(observed[:, i].reshape(-1, 17), generated[:, i].reshape(-1, 17),
                                                   action_score[:, i].reshape(-1, 17), truth[:, i].reshape(-1, 17), dev['episode'], settings)
                            for i, name in enumerate(names)}
        del predictions[task]
        if progress is not None: progress(task,report)
    report['prefix_coverage'] = {arm: {split: {str(c): {'available': int((f['length'] >= c).sum()),
                                                                   'short_bos': int((f['length'] < c).sum())}
                                              for c in MEMORY_CONTEXTS} for split, f in rows.items()} for arm, rows in features.items()}
    report['carry_rms'] = {arm: {case: {'mean_per_layer_conv_ssm': rows['dev'][f'{case}_carry_rms'].mean(0).tolist(),
                                     'max': float(rows['dev'][f'{case}_carry_rms'].max())} for case in MEMORY_CASES}
                           for arm, rows in features.items()}
    return report
