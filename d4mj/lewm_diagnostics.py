"""Component-scoped M0-M3 gates. Failures never imply a verdict on the architecture."""

import copy
import importlib.util
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .data import JointSampler, audit_episodes
from .lewm import SIGReg, joint_loss
from .lewm_config import LeWMConfig
from .gates import ComponentGateError
from .mamba_recurrence import MambaCarry
from .sources import ROOT, lewm_source_manifest, tensor_state_digest
from .world_api import ModelBundle


def _assert_close(a, b):
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


# Sealed 2026-09-05 on RTX 3060, pinned Mamba + Triton 3.7.1. See the
# calibration records in docs/evidence/m0_m3. No tolerance is fit at gate time.
# SSM comparison is absolute-only: large relative errors near zero are unhelpful.
NUMERICAL_PROFILE = "rtx3060_mamba_f577286d_v2"
NUMERICAL_TOLERANCES = {
    "reference_fp32": {"output": (1e-5, 1e-4), "conv": (1e-5, 1e-4), "ssm": (1e-5, 0.0),
                       "gradient": (1e-5, 1e-3)},
    "triton_fp32": {"output": (5e-4, 1e-4), "conv": (1e-5, 1e-4), "ssm": (1e-4, 0.0),
                    "gradient": (2e-5, 1e-3)},
    # In a stack, later convolution inputs include preceding layers' SSD rounding.
    # Keep the single-mixer convolution check tighter; only this propagation gets 1e-4.
    "triton_world_fp32": {"output": (5e-4, 1e-4), "conv": (1e-4, 1e-4), "ssm": (1e-4, 0.0)},
    "triton_bf16": {"output": (1e-2, 5e-3), "conv": (1e-5, 0.0), "ssm": (1e-3, 0.0),
                    "gradient": (2e-3, 5e-3)},
    "triton_reference_fp32": {"output": (1e-3, 1e-4), "conv": (1e-5, 1e-4),
                              "ssm": (1e-3, 0.0), "gradient": (5e-5, 1e-3)},
    "triton_reference_bf16": {"output": (1e-2, 5e-3), "conv": (1e-5, 0.0),
                              "ssm": (1e-3, 0.0), "gradient": (2e-3, 5e-3)},
}


def numerical_check(left, right, profile, quantity):
    atol, rtol = NUMERICAL_TOLERANCES[profile][quantity]
    try:
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"{profile}/{quantity}: {error}") from error
    return float((left.detach().float()-right.detach().float()).abs().max())


def mixer_numerical_audit(mixer, precision: str) -> dict:
    """Independent equations, actual source operator and incoming-state gradients."""
    device = mixer.core.in_proj.weight.device
    backend = mixer.settings.backend
    if precision == "bf16" and backend != "triton":
        raise ComponentGateError("recurrence_precision", "BF16 reference backend has no sealed profile")
    profile = f"{backend}_{precision}"
    rows = []
    # The CUDA check includes a real chunk boundary, not only T < chunk_size.
    lengths = (2, 17, 65, mixer.settings.chunk_size+1) if backend == "triton" else (9, 65)
    for length in lengths:
        rng = torch.Generator(device=device).manual_seed(904+length)
        x = torch.randn(2, length, mixer.settings.width, device=device, generator=rng, requires_grad=True)
        empty = mixer.initial(2, device=device)
        incoming = MambaCarry(torch.randn(empty.conv.shape, device=device, generator=rng).requires_grad_(),
                              (.01*torch.randn(empty.ssm.shape,device=device,generator=rng)).requires_grad_())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            y, final = mixer.scan(x, incoming)
            carry, parts = incoming, []
            for t in range(length):
                part, carry = mixer.step(x[:, t:t+1], carry)
                parts.append(part)
            stepped = torch.cat(parts, 1)
            reference, ref_state = mixer.scan(x, incoming, backend="reference")
        row = {"length": length}
        for name, left, right in (("output",y,stepped),("conv",final.conv,carry.conv),("ssm",final.ssm,carry.ssm)):
            row[f"scan_step_{name}_max_abs"] = numerical_check(left,right,profile,name)
        variables = [x, incoming.conv, incoming.ssm, *mixer.parameters()]
        def gradients(output, state):
            objective = output.float().square().mean()+state.conv.float().square().mean()+state.ssm.square().mean()
            return torch.autograd.grad(objective,variables)
        full_grad, step_grad, ref_grad = gradients(y,final), gradients(stepped,carry), gradients(reference,ref_state)
        row["scan_step_gradient_max_abs"] = max(numerical_check(a,b,profile,"gradient")
                                                  for a,b in zip(full_grad,step_grad))
        if full_grad[1].abs().sum() == 0 or full_grad[2].abs().sum() == 0:
            raise AssertionError("gradient to incoming recurrence state was lost")
        comparison = f"triton_reference_{precision}" if backend == "triton" else profile
        row["reference_output_max_abs"] = numerical_check(y,reference,comparison,"output")
        row["reference_ssm_max_abs"] = numerical_check(final.ssm,ref_state.ssm,comparison,"ssm")
        row["reference_gradient_max_abs"] = max(numerical_check(a,b,comparison,"gradient")
                                                 for a,b in zip(full_grad,ref_grad))
        rows.append(row)
    return {"precision": precision, "profile": profile, "rows": rows, "incoming_state_gradients": True}


def objective_audit(config: LeWMConfig) -> dict:
    path = ROOT / "sources/lucas-maes__le-wm/module.py"
    spec = importlib.util.spec_from_file_location("_lewm_source_objective", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    device = config.runtime.device
    generator = torch.Generator(device=device).manual_seed(27)
    values = torch.randn(4, 8, 12, device=device, generator=generator, requires_grad=True)
    ours = SIGReg(17, 1024).to(device)
    reference = source.SIGReg(knots=17, num_proj=1024).to(device)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(321)
        expected = reference(values)
    actual = ours(values, torch.Generator(device=device).manual_seed(321))
    _assert_close(actual, expected)
    _assert_close(torch.autograd.grad(actual, values)[0], torch.autograd.grad(expected, values)[0])
    offset = torch.randn(1, 8, 12, device=device, generator=generator) * 3
    r = values-values.mean(0, keepdim=True)
    shifted = values+offset
    shifted = shifted-shifted.mean(0, keepdim=True)
    a = ours(r, torch.Generator(device=device).manual_seed(7))
    b = ours(shifted, torch.Generator(device=device).manual_seed(7))
    _assert_close(a, b)
    raw = ours(values+offset, torch.Generator(device=device).manual_seed(7))
    if torch.isclose(raw, a):
        raise AssertionError("the regularization treatment is not separated")
    return {"source_value": float(expected.detach()), "absolute_error": float((actual-expected).detach().abs()),
            "centering_offset_error": float((a-b).detach().abs())}


def recurrence_audit(bundle: ModelBundle) -> dict:
    """Longer-than-training context, incoming carry gradients, source and chunk checks."""
    bundle.eval()
    c, w = bundle.config, bundle.world
    device = next(w.parameters()).device
    rng = torch.Generator(device=device).manual_seed(500)
    profile = "triton_world_fp32" if c.dynamics.backend == "triton" else "reference_fp32"
    errors = []
    def compare_states(left, right):
        for i, (a, b) in enumerate(zip(bundle.state_tensors(left), bundle.state_tensors(right))):
            quantity = "ssm" if i >= 3 and i % 2 else "conv" if i >= 2 else "output"
            errors.append(numerical_check(a,b,profile,quantity))
    z = torch.randn(2, 18, 1, c.encoder.latent_dim, device=device, generator=rng)
    a = torch.randint(c.dynamics.n_actions, (2, 17), device=device, generator=rng)
    with torch.no_grad():
        full = w.teacher(z, a)
        state = bundle.start(z[:, :1])
        for i in range(a.shape[1]):
            state, _ = w.observe_latent(state, a[:, i:i+1], z[:, i+1:i+2])
        compare_states(full.state, state)
        prefix = w.teacher(z[:, :8], a[:, :7]).state
        chunked = w.teacher(z[:, 7:], a[:, 7:], state=prefix)
        compare_states(full.state, chunked.state)
        changed = a.clone()
        changed[:, 7] = (changed[:, 7]+1) % c.dynamics.n_actions
        causal = w.teacher(z, changed)
        _assert_close(full.features[:, :8], causal.features[:, :8])
        root = bundle.prefill(z[:, :8], a[:, :7])
        original = [v.clone() for v in bundle.state_tensors(root)]
        buffers = tensor_state_digest(w.state_dict())
        first, _ = bundle.advance(root, a[:, 7:8])
        bundle.advance(bundle.fork(root), changed[:, 7:8])
        again, _ = bundle.advance(root, a[:, 7:8])
        for before, after in zip(original, bundle.state_tensors(root)):
            _assert_close(before, after)
        for before, after in zip(bundle.state_tensors(first), bundle.state_tensors(again)):
            _assert_close(before, after)
        if tensor_state_digest(w.state_dict()) != buffers:
            raise AssertionError("branch changed world buffers")
        if first.step != root.step+1:
            raise AssertionError("accepted transition did not advance exactly once")
        perturbed = z.clone()
        perturbed[:, 0] += 1
        other = w.teacher(perturbed, a).state
        memory_effect = float((other.history-full.state.history).abs().max())
        if memory_effect <= 1e-8:
            raise AssertionError("old context has no measurable effect with current observation fixed")

    mixer = w.layers[0].mixer
    numerical = [mixer_numerical_audit(mixer, "fp32")]
    if c.runtime.precision != "fp32":
        numerical.append(mixer_numerical_audit(mixer, c.runtime.precision))
    x = torch.randn(2, 9, c.dynamics.width, device=device, generator=rng)
    # Independent oracle: execute upstream's actual inference step, using its own
    # provided PyTorch fallbacks on CPU. This does NOT validate the CUDA kernels.
    import mamba_ssm.modules.mamba2 as source
    from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
    core = copy.deepcopy(mixer.core)
    src_state = mixer.initial(2, device=device)
    with torch.no_grad():
        if device.type == "cpu":
            with patch.object(source, "selective_state_update", None), patch.object(source, "causal_conv1d_update", None), \
                 patch.object(core.norm, "forward", side_effect=lambda v, gate: rms_norm_ref(
                     v, core.norm.weight, None, z=gate, eps=core.norm.eps, norm_before_gate=False)):
                expected = []
                for t in range(x.shape[1]):
                    out, conv, ssm = core.step(x[:, t:t+1], src_state.conv, src_state.ssm)
                    src_state = MambaCarry(conv, ssm)
                    expected.append(out)
                expected = torch.cat(expected, 1)
        else:
            expected = core(x)
        actual, _ = mixer.scan(x)
        _assert_close(actual, expected)
    return {"context_frames": 18, "training_frames": c.joint.frames,
            "numerical_profile": NUMERICAL_PROFILE, "numerical_checks": numerical,
            "world_state_max_abs": max(errors),
            "state_gradient_checked": True, "source_step_error": float((actual-expected).abs().max()),
            "memory_effect": memory_effect, "cuda_kernels_checked": device.type == "cuda"}


def normalization_audit(bundle: ModelBundle) -> dict:
    bundle.eval()
    e = bundle.config.encoder
    device = next(bundle.encoder.parameters()).device
    frames = torch.randint(256, (2, 4, e.resolution, e.resolution, 3), device=device, dtype=torch.uint8,
                           generator=torch.Generator(device=device).manual_seed(77))
    before = tensor_state_digest(bundle.encoder.state_dict())
    with torch.no_grad():
        whole = bundle.encode(frames)
        for b in range(2):
            for t in range(4):
                _assert_close(whole[b:b+1, t:t+1], bundle.encode(frames[b:b+1, t:t+1]))
    if before != tensor_state_digest(bundle.encoder.state_dict()):
        raise AssertionError("eval encoding changed normalization buffers")
    return {"singleton_batch_parity": True, "buffers_immutable": True}


def resource_preflight(bundle, episodes) -> dict:
    """Actual declared batch and optimizer step, without changing a training model."""
    from .train import set_phase_mode, joint_optimizer, autocast_context

    c = bundle.config
    set_phase_mode(bundle, "joint")
    sampler = JointSampler(episodes, c, torch.Generator().manual_seed(c.seed + 19))
    optimizer = joint_optimizer(bundle)
    regularizer = SIGReg(c.joint.knots, c.joint.projections).to(c.runtime.device)
    rng = torch.Generator(device=c.runtime.device).manual_seed(c.seed + 20)
    cuda = c.runtime.device == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elapsed, encoder_gradient, predictor_gradient = [], [], []
    for _ in range(3):
        start = time.perf_counter()
        batch = sampler.sample().to(c.runtime.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(c):
            loss = joint_loss(bundle.encoder, bundle.world, batch.frames, batch.actions, regularizer, rng, c)
        loss.total.backward()
        encoder_gradient.append(float(bundle.encoder.projector[0].weight.grad.norm()))
        predictor_gradient.append(float(bundle.world.pair_projection.weight.grad.norm()))
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, c.joint.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
    if min(encoder_gradient) <= 0 or min(predictor_gradient) <= 0:
        raise AssertionError("joint encoder or dynamics received no gradient")
    allocated = torch.cuda.max_memory_allocated() if cuda else None
    reserved = torch.cuda.max_memory_reserved() if cuda else None
    if cuda and max(allocated, reserved) > c.runtime.memory_budget_bytes:
        raise ComponentGateError("joint_memory", f"peak {max(allocated,reserved)} exceeds {c.runtime.memory_budget_bytes}")
    return {"actual_batch": c.joint.batch, "frames": c.joint.frames,
            "precision": c.runtime.precision, "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved, "warm_step_seconds": sum(elapsed[1:])/2,
            "encoder_gradient": encoder_gradient[-1], "predictor_gradient": predictor_gradient[-1],
            "gpu_validated": cuda,
            "device_name": torch.cuda.get_device_name() if cuda else "cpu",
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
            "optimizer_steps": 3, "purpose": "resource_only_not_a_learning_result"}


def joint_checks(config: LeWMConfig, episodes, report: dict):
    """Family-specific probes; gates.py owns dependency handling and reporting."""
    from .gates import Gate

    bundle = None
    def sources():
        report["sources"] = lewm_source_manifest()
        return {"versions": report["sources"]["versions"]}
    def device():
        if config.runtime.device == "cuda" and not torch.cuda.is_available():
            raise ComponentGateError("device", "CUDA unavailable in this process")
        return config.runtime.device
    def construct():
        nonlocal bundle
        bundle = ModelBundle.create(config)
        return {"encoder_parameters": sum(p.numel() for p in bundle.encoder.parameters()),
                "world_parameters": sum(p.numel() for p in bundle.world.parameters())}
    return (
        Gate("source_identity", sources),
        Gate("dataset", lambda: {"split_counts": audit_episodes(episodes,config)["split_counts"]}),
        Gate("device", device),
        Gate("objective_source",lambda: objective_audit(config),("source_identity","device")),
        Gate("model_construction",construct,("source_identity","dataset","device")),
        Gate("recurrence",lambda: recurrence_audit(bundle),("model_construction",)),
        Gate("normalization",lambda: normalization_audit(bundle),("model_construction",)),
        Gate("joint_resource",lambda: resource_preflight(bundle,episodes),
             ("objective_source","recurrence","normalization")),
    )


def covariance_summary(values):
    """Finite empirical spectrum; report scale separately from effective rank."""
    x = values.detach().cpu().double().reshape(-1, values.shape[-1])
    if not bool(torch.isfinite(x).all()):
        raise ComponentGateError("screen_finiteness", "nonfinite exported features")
    centered = x-x.mean(0)
    spectrum = torch.linalg.eigvalsh(centered.T@centered/max(1,len(x)-1)).clamp_min(0).flip(0)
    total = spectrum.sum()
    probability = spectrum/total.clamp_min(1e-30)
    rank = torch.exp(-(probability*probability.clamp_min(1e-30).log()).sum()) if total > 0 else torch.tensor(0.)
    return {"coordinate_variance": float(total/x.shape[-1]), "mean_norm": float(x.mean(0).norm()),
            "effective_rank": float(rank), "eigenvalues": spectrum.tolist()}


@torch.no_grad()
def screen_features(bundle, windows, settings):
    """Frozen exported/CLS features and teacher predictions on one fixed sample."""
    bundle.eval()
    before = tensor_state_digest(bundle.encoder.state_dict())
    z_rows, cls_rows, prediction, marginal = [], [], [], []
    actions = windows["actions"]
    permuted = actions.roll(1, 0)
    for start in range(0, len(actions), settings.encode_batch):
        end = start+settings.encode_batch
        frames = windows["frames"][start:end].to(bundle.device)
        with torch.autocast(device_type=bundle.device.type, enabled=False):
            z, cls = bundle.encoder.projected_and_cls(frames)
            predicted = bundle.world.teacher(z, actions[start:end].to(bundle.device)).predicted
            shuffled = bundle.world.teacher(z, permuted[start:end].to(bundle.device)).predicted
        z_rows.append(z[:, :, 0].cpu()); cls_rows.append(cls.cpu())
        prediction.append(predicted[:, :, 0].cpu()); marginal.append(shuffled[:, :, 0].cpu())
    if tensor_state_digest(bundle.encoder.state_dict()) != before:
        raise ComponentGateError("normalization", "screen changed encoder buffers")
    z = torch.cat(z_rows)
    return {"projected": z, "cls": torch.cat(cls_rows), "prediction": torch.cat(prediction),
            "permuted_prediction": torch.cat(marginal), "persistence": z[:, :-1]}


def screen_prediction_report(features, clusters, settings, *, variance):
    target = features["projected"][:, 1:]
    errors = {name: (features[name]-target).square().mean((1,2))
              for name in ("prediction", "permuted_prediction", "persistence")}
    def difference(other):
        delta = errors["prediction"]-errors[other]
        values = torch.stack([delta[clusters == k].mean() for k in clusters.unique(sorted=True)])
        indices = torch.randint(len(values), (settings.bootstrap_draws, len(values)),
                                generator=torch.Generator().manual_seed(settings.seed+300))
        return {"difference": float(values.mean()),
                "interval": values[indices].mean(1).quantile(torch.tensor([.025,.975])).tolist(), "unit": "episode"}
    return {"normalized_prediction_mse": float(errors["prediction"].mean()/max(variance, settings.variance_floor)),
            "mse": {name: float(value.mean()) for name,value in errors.items()},
            "prediction_minus_persistence": difference("persistence"),
            "prediction_minus_permuted_actions": difference("permuted_prediction"),
            "scope": "logged-action association, not simulator action consequences"}, errors


def screen_retention(train_features, dev_features, train_windows, dev_windows, settings, device, n_actions):
    from .diagnostics import binary_auc, paired_auc_interval, fit_outcome_probe

    truth = dev_windows["labels"].flatten(0,1)
    valid = dev_windows["valid"].flatten(0,1)
    fit_truth = train_windows["labels"].flatten(0,1)
    fit_valid = train_windows["valid"].flatten(0,1)
    coverage = []
    for col, name in enumerate(train_windows["label_names"]):
        counts = {}
        for split, y, mask in (("train",fit_truth,fit_valid),("dev",truth,valid)):
            counts[split] = {"positive": int((y[:,col]&mask[:,col]).sum()),
                             "negative": int((~y[:,col]&mask[:,col]).sum())}
        supported = all(v["positive"] >= settings.minimum_positive and v["negative"] >= settings.minimum_negative
                        for v in counts.values())
        coverage.append({"label": name, "counts": counts, "supported": supported})
    selected = torch.tensor([row["supported"] for row in coverage])
    clusters = dev_windows["clusters"].repeat_interleave(dev_windows["actions"].shape[1])
    report = {"coverage": coverage, "probes": {}, "critical_semantic_retention": "not_evaluated",
              "scope": "short-future archive proxies conditioned on outgoing action"}
    outputs = {}
    for hidden, family in ((False,"linear"),(True,"mlp")):
        predictions = {}
        for name in ("cls","projected"):
            fit_x, dev_x = [torch.cat((features[name][:,:-1].flatten(0,1),
                          torch.nn.functional.one_hot(windows["actions"].flatten(), n_actions).float()), -1).to(device)
                          for features,windows in ((train_features,train_windows),(dev_features,dev_windows))]
            predictions[name] = fit_outcome_probe(fit_x, fit_truth.float().to(device), fit_valid.float().to(device),
                                                  dev_x, settings, hidden=hidden)
        outputs[family] = predictions
        if bool(selected.any()):
            comparison = paired_auc_interval(predictions["projected"][:,selected], predictions["cls"][:,selected],
                                             truth[:,selected], valid[:,selected], clusters,
                                             draws=settings.bootstrap_draws, seed=settings.seed+400)
        else:
            comparison = {"status": "insufficient_coverage", "difference": None, "interval": None}
        comparison["auc"] = {name: [binary_auc(scores[valid[:,i],i], truth[valid[:,i],i])
                                    for i in range(truth.shape[1])] for name,scores in predictions.items()}
        report["probes"][family] = comparison
    report["projection_stop"] = all(row["interval"] is not None and row["interval"][1] < -settings.auc_margin
                                     for row in report["probes"].values())
    return report, {"scores": outputs, "truth": truth, "valid": valid, "clusters": clusters}


def screen_joint_pair(runs, episodes, dataset_contract, settings, output):
    """Paired G1 with checkpoint identity, mechanics and component-local stops."""
    import json
    from .config import config_from_dict, recipe_dict, recipe_digest
    from .checkpoint import read_lewm_bundle
    from .data import atomic_manifest, screen_windows, _sha256
    from .gates import contract_digest
    from .world_api import load_bundle

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "screen.json"
    if report_path.exists():
        raise ValueError("screen_output: refusing to replace a sealed report")
    report = {"schema": "d4mj_joint_screen_report_v1", "settings": recipe_dict(settings),
              "settings_id": recipe_digest(settings), "dataset_id": contract_digest(dataset_contract),
              "sources": lewm_source_manifest(), "arms": {}, "components": {}, "decision": "stop_component",
              "architecture_verdict": "not_evaluated", "m4_authorized": False}
    payloads = {}
    try:
        pair_path = Path(runs["raw"]).parent/"pair.json"
        pair = json.loads(pair_path.read_text())
        if pair["screen_settings_id"] != recipe_digest(settings) or pair["dataset_id"] != contract_digest(dataset_contract):
            raise ComponentGateError("screen_settings", "screen differs from the pre-training seal")
        report["pair_manifest"] = {"path":str(pair_path.resolve()),"sha256":_sha256(pair_path)}
        for variant in ("raw", "tc"):
            run = Path(runs[variant]); config = config_from_dict(json.loads((run/"resolved_recipe.json").read_text()))
            checkpoint = run/"joint"/f"step-{config.joint.screen_step:06d}.pt"
            payloads[variant] = p = read_lewm_bundle(checkpoint)
            initial = read_lewm_bundle(run/"joint/step-000000.pt")
            if (p["config"] != recipe_dict(config) or config.variant != variant or p["step"] != config.joint.screen_step
                    or p["dataset"] != dataset_contract or initial["step"] != 0
                    or initial["config"] != p["config"] or initial["dataset"] != dataset_contract
                    or initial["initial_identity"] != p["initial_identity"]):
                raise ComponentGateError("pair_identity", "wrong recipe, dataset, step or initial checkpoint")
            for name in ("encoder", "world"):
                if tensor_state_digest(initial["modules"][name]) != p["initial_identity"][name]:
                    raise ComponentGateError("initial_identity", "saved initialization bytes do not match training")
            report["arms"][variant] = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
                                      "recipe_id": p["recipe_id"], "initial_identity": p["initial_identity"]}
        raw, tc = payloads["raw"], payloads["tc"]
        if ({k:v for k,v in raw["config"].items() if k != "variant"} !=
                {k:v for k,v in tc["config"].items() if k != "variant"}
                or raw["initial_identity"] != tc["initial_identity"]
                or not torch.equal(raw["sampler"]["generator"],tc["sampler"]["generator"])
                or not torch.equal(raw["projection_rng"],tc["projection_rng"])):
            raise ComponentGateError("pair_identity", "raw/TC differ beyond the declared objective")
        histories = [[json.loads(line) for line in (Path(runs[v])/"joint/metrics.jsonl").read_text().splitlines()]
                     for v in ("raw","tc")]
        length = raw["step"]
        if any([row["update"] for row in h] != list(range(1,length+1)) for h in histories):
            raise ComponentGateError("pair_history", "screen requires every update exactly once")
        if any(a["windows"] != b["windows"] or a["learning_rate"] != b["learning_rate"]
               for a,b in zip(*histories,strict=True)):
            raise ComponentGateError("pair_history", "actual windows or schedules differ")
        first_raw, first_tc = histories[0][0], histories[1][0]
        if (abs(first_raw["prediction"]-first_tc["prediction"]) > 1e-6
                or abs(first_raw["regularization"]-first_tc["regularization"]) <= 1e-6):
            raise ComponentGateError("objective_contrast", "first paired update lacks the intended isolated treatment")
        if not all(all(torch.isfinite(torch.tensor(float(value))) for key,value in row.items()
                       if isinstance(value,(int,float))) for h in histories for row in h):
            raise ComponentGateError("training_finiteness", "nonfinite training history")
        report["components"]["pair_identity"] = {"status": "pass"}
        config = config_from_dict(raw["config"])
        windows = {split: screen_windows(episodes, config, settings, split) for split in ("train","dev")}
        atomic_manifest(output/"windows.json", {split:{"episode_ids":w["episode_ids"],"starts":w["starts"].tolist()}
                                                for split,w in windows.items()})
        report["components"]["objective_contrast"] = {"status":"pass","detail":objective_audit(config)}
        for variant in ("raw","tc"):
            entry = report["arms"][variant]
            bundle, _ = load_bundle(entry["checkpoint"])
            bundle.world.requires_grad_(True)
            for name, check in (("normalization",normalization_audit),("recurrence",recurrence_audit)):
                try:
                    report["components"][f"{variant}_{name}"] = {"status":"pass","detail":check(bundle)}
                except Exception as error:
                    raise ComponentGateError(f"{variant}_{name}",str(error)) from error
            bundle.world.requires_grad_(False)
            features = {split: screen_features(bundle,w,settings) for split,w in windows.items()}
            z = features["dev"]["projected"]
            entry["spectra"] = {"raw":covariance_summary(z), "residual":covariance_summary(z-z.mean(1,keepdim=True)),
                                "persistent":covariance_summary(z.mean(1))}
            entry["temporal_power"] = torch.fft.rfft(z.double(),dim=1).abs().square().mean((0,2)).tolist()
            variance = covariance_summary(features["train"]["projected"])["coordinate_variance"]
            if variance < settings.variance_floor:
                raise ComponentGateError(f"{variant}_latent_variance", "numerical export collapse")
            entry["prediction"], errors = screen_prediction_report(features["dev"],windows["dev"]["clusters"],settings,variance=variance)
            entry["retention"], probe_rows = screen_retention(features["train"],features["dev"],windows["train"],windows["dev"],
                                                            settings,bundle.device,bundle.n_actions)
            torch.save({"features":features,"errors":errors,"probes":probe_rows},output/f"{variant}_rows.pt")
            del bundle
            initial_bundle, _ = load_bundle(Path(runs[variant])/"joint/step-000000.pt")
            initial_features = {split:screen_features(initial_bundle,w,settings) for split,w in windows.items()}
            initial_variance = covariance_summary(initial_features["train"]["projected"])["coordinate_variance"]
            entry["initial_prediction"], _ = screen_prediction_report(initial_features["dev"],windows["dev"]["clusters"],settings,variance=initial_variance)
            del initial_bundle, initial_features, features
            if entry["retention"]["projection_stop"]:
                raise ComponentGateError(f"{variant}_projection_retention", "both fixed probes exceed the loss margin with paired support")
            entry["learning_progress"] = entry["prediction"]["normalized_prediction_mse"] < entry["initial_prediction"]["normalized_prediction_mse"]
        report["decision"] = "continue_joint_budget" if all(a["learning_progress"] for a in report["arms"].values()) else "review_required"
    except Exception as error:
        component = getattr(error,"component","joint_screen_execution")
        report["components"][component] = {"status":"fail","reason":str(error)}
        report["blocked_component"] = component
    report["artifacts"] = {p.name:_sha256(p) for p in output.iterdir() if p.is_file() and p != report_path}
    report["artifact_root"] = str(output.resolve())
    report["report_id"] = contract_digest(report)
    atomic_manifest(report_path,report)
    return report
