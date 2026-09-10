"""Immutable historical root addresses, fresh simulator labels, common M03 scoring.

The 961/197 panel is the central regression panel. P13/104 are stress panels;
the 5,402 panel supplies breadth. Their overlapping roots are never pooled as
independent evidence. Old latent tensors and old fitted objects are not consumed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

PANELS = ("exact961", "policy104", "hazard5402", "legacy751")
PROTOCOL = "historical_addresses_exact_replay_native_prefix"


def seed_split(seed: int) -> str:
    # The FINAL fork-world split, not the unrelated all-action damage-classifier split.
    value = int.from_bytes(sha256(f"paired-seed:{seed}".encode()).digest()[:8], "little") % 10
    return "fit" if value < 7 else ("tune" if value < 8 else "test")


def _load(path):
    from .cache import resolve_payload
    return resolve_payload(torch.load(path, map_location="cpu", weights_only=False, mmap=True))


def source_paths(root: Path) -> list[Path]:
    eda = root / "artifacts/eda"
    paths = [eda / "fork_histories/branched_965.pt", eda / "fork_histories/policy_fork_104.pt",
             root / "artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt",
             root / "artifacts/stage_a_terminalfix/phase1a.pt",
             root / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"]
    for directory, pattern in (("fork_successors", "shard-*.pt"), ("root_frames", "shard-*.pt"),
                               ("branched_damage", "seed-*.pt")):
        found = sorted((eda / directory).glob(pattern))
        if not found:
            raise FileNotFoundError(f"m03_history: no sources in {eda / directory}")
        paths.extend(found)
    return paths


def addresses(root: Path):
    """Read only physical records and truth, discarding all archived encodings."""
    eda = root / "artifacts/eda"
    refs = {}
    for path in sorted((eda / "fork_successors").glob("shard-*.pt")):
        for row in _load(path):
            key = (int(row["seed"]), int(row["step"]))
            if key in refs:
                raise ValueError(f"m03_history: duplicate successor address {key}")
            refs[key] = {k: row[k] for k in ("successors", "terminated", "label")}
    panels = {"hazard5402": sorted(refs)}
    for name, filename in (("exact961", "branched_965"), ("policy104", "policy_fork_104")):
        keys = []
        for row in _load(eda / f"fork_histories/{filename}.pt"):
            key = (int(row["seed"]), int(row["step"]))
            if name == "exact961" and key not in refs:
                continue
            keys.append(key)
            refs.setdefault(key, {}).update({k: row[k] for k in ("frames", "led_to_action", "true_death", "trajectory_action")})
        panels[name] = sorted(keys)
    old = _load(root / "artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt")
    panels["legacy751"] = []
    for i, (seed, step) in enumerate(zip(old["seed"].tolist(), old["step"].tolist())):
        key = (seed, step)
        panels["legacy751"].append(key)
        ref = refs.setdefault(key, {})
        if "true_death" in ref and not torch.equal(ref["true_death"], old["true_death"][i]):
            raise ValueError(f"m03_history: historical tables disagree at {key}")
        ref.update(true_death=old["true_death"][i], true_reward=old["true_reward"][i])
    for path in sorted((eda / "root_frames").glob("shard-*.pt")):
        for row in _load(path):
            key = (int(row["seed"]), int(row["step"]))
            if key in refs and "frames" not in refs[key]:
                refs[key]["frames"] = row["frames"]
    for name, expected in zip(PANELS, (961, 104, 5402, 751)):
        if len(panels[name]) != expected or len(set(panels[name])) != expected:
            raise ValueError(f"m03_history: {name} has changed membership; expected {expected}")
    return panels, refs


def _policy(root: Path, device: str):
    """Only reaches old roots; strictly load the preserved pre-mixer architecture."""
    from ..agent import Heads
    from ..config import config_from_dict
    from ..representation import Encoder
    from .gate import _load_legacy_cpu
    from artifacts.eda.legacy import LegacyDirectWorld
    p1 = root / "artifacts/stage_a_terminalfix/phase1a.pt"
    p2 = root / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"
    base, config = (config_from_dict(_load(p)["config"]) for p in (p1, p2))
    encoder, world, heads = Encoder(base).to(device), LegacyDirectWorld(config).to(device), Heads(config).to(device)
    _load_legacy_cpu(p1, base, part0=encoder)
    _load_legacy_cpu(p2, config, part0=world, part1=heads)
    for module in (encoder, world, heads):
        module.eval().requires_grad_(False)
    return encoder, world, heads, replace(config, device=device)


def verify_reference(key, ref, frames, led, successors, death, reward, damage, chosen):
    """Fail before publishing any mismatched historical row. Pixels are exact here."""
    checks = {"frames": frames[-len(ref["frames"]):] if "frames" in ref else None,
              "led_to_action": led[-len(ref["led_to_action"]):] if "led_to_action" in ref else None,
              "successors": successors, "terminated": death, "true_death": death,
              "true_reward": reward, "label": damage.float()}
    for name, actual in checks.items():
        if name in ref and (actual is None or not torch.equal(ref[name], actual)):
            raise ValueError(f"m03_history_replay: {name} mismatch at {key}")
    if "trajectory_action" in ref and int(ref["trajectory_action"]) != chosen:
        raise ValueError(f"m03_history_replay: factual action mismatch at {key}")


@torch.inference_mode()
def replay_seed(root, seed, wanted, refs, settings, policy=None):
    from .. import env
    from ..data import patchify
    from ..transition import observe
    from .diagnostics import state_features
    from .gate import _load_replay, _state_scalars, _state_binary_labels, _action_outcomes
    replay = _load_replay()
    stream_path = root / f"artifacts/eda/branched_damage/seed-{seed:05d}.pt"
    # File names are collected from the real directory; tolerate unpadded seeds.
    if not stream_path.exists():
        stream_path = root / f"artifacts/eda/branched_damage/seed-{seed}.pt"
    stream = _load(stream_path)["led_to_action"].long() if stream_path.exists() else None
    if stream is None and policy is None:
        raise ValueError(f"m03_history: seed {seed} needs the archived policy")
    if stream is not None and (int(stream[0]) != 17 or len(stream) <= max(wanted)):
        raise ValueError(f"m03_history: incomplete incoming-action stream for {seed}")
    observation, state = env.reset(seed)
    frames, led, records = [], [17], []
    policy_state = None
    if policy is not None:
        encoder, world, heads, config = policy
        wrng = torch.Generator(device=config.device).manual_seed(seed+2**21)
        prng = torch.Generator(device=config.device).manual_seed(seed+2**20)
    for t in range(max(wanted)+1):
        frames.append(observation)
        if stream is None:
            incoming = torch.tensor([[led[-1]]], device=config.device)
            policy_state, agent = observe(world, encoder, policy_state, incoming,
                                          patchify(observation[None, None], config.patch).to(config.device), wrng, config)
            logits = heads(agent)["policy"][:, -1, 0]
            chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=prng))
        elif t+1 < len(stream):
            chosen = int(stream[t+1])
        else:
            # The last stored incoming stream may omit the terminal outgoing action.
            chosen = int(refs.get((seed, t), {}).get("trajectory_action", -1))
            if chosen < 0:
                raise ValueError(f"m03_history: missing final outgoing action at {(seed,t)}")
        if t in wanted:
            root_c, root_b = _state_scalars(state), _state_binary_labels(state)
            branches = [env.step(state, a, seed+t+1) for a in range(17)]
            successors = torch.stack([b[0] for b in branches])
            death = torch.tensor([b[3] for b in branches])
            reward = torch.tensor([b[2] for b in branches])
            damage = torch.tensor([float(b[1].player_health-state.player_health) <= -1 or b[3] for b in branches])
            verify_reference((seed,t), refs[(seed,t)], torch.stack(frames), torch.tensor(led),
                             successors, death, reward, damage, chosen)
            outcomes, continuous, binary, modes = [], [], [], []
            for a, b in enumerate(branches):
                c, binary_next = _state_scalars(b[1]), _state_binary_labels(b[1])
                continuous.append(c); binary.append(binary_next)
                outcomes.append(_action_outcomes(state, b[1], b[2], root_binary=root_b, successor_binary=binary_next, replay=replay))
                sampled = []
                for mode in range(settings.mode_samples):
                    # A disjoint, deterministic namespace; same draw for all actions.
                    key = int.from_bytes(sha256(f'm03-mode:{settings.seed}:{seed}:{t}:{mode}'.encode()).digest()[:4], 'little')
                    m = env.step(state, a, key)
                    sampled.append(_action_outcomes(state, m[1], m[2], root_binary=root_b,
                                                   successor_binary=_state_binary_labels(m[1]), replay=replay))
                modes.append(np.stack(sampled))
            length = min(t+1, settings.direct_context)
            padded = torch.zeros(settings.direct_context, *observation.shape, dtype=torch.uint8)
            padded[-length:] = torch.stack(frames[-length:])
            actions = torch.full((settings.direct_context-1,), 17, dtype=torch.long)
            if length > 1:
                actions[-length+1:] = torch.tensor(led[-length+1:])
            records.append({**state_features(state), "context": padded, "context_length": torch.tensor(length), "past_actions": actions,
                            "direct_first_action": torch.tensor(led[-length]), "successors": successors,
                            "root_continuous": torch.tensor(root_c), "root_binary": torch.tensor(root_b),
                            "next_continuous": torch.tensor(np.stack(continuous)), "next_binary": torch.tensor(np.stack(binary)),
                            "outcomes": torch.tensor(np.stack(outcomes)), "modes": torch.tensor(np.stack(modes)),
                            "episode": torch.tensor(seed), "time": torch.tensor(t), "factual_action": torch.tensor(chosen)})
        observation, state, _, dead, truncated = env.step(state, chosen, seed+t+1)
        led.append(chosen)
        if dead or truncated:
            break
    if len(records) != len(wanted):
        raise ValueError(f"m03_history: reached {len(records)} of {len(wanted)} roots for {seed}")
    return {k: torch.stack([r[k] for r in records]) for k in records[0]}


def split_indices(keys, panels, name):
    train_panel = panels["hazard5402"] if name == "hazard5402" else panels["exact961"]
    train_keys = {k for k in train_panel if seed_split(k[0]) == "fit"}
    dev_keys = set(panels[name]) if name in ("policy104", "legacy751") else {k for k in panels[name] if seed_split(k[0]) == "test"}
    if {s for s,t in train_keys} & {s for s,t in dev_keys}:
        raise ValueError("m03_history: probe fit/evaluation seed leakage")
    return torch.tensor([i for i,k in enumerate(keys) if k in train_keys]), torch.tensor([i for i,k in enumerate(keys) if k in dev_keys])


def run_history(output, settings, *, root, checkpoints, dataset_sha256, device, include_direct, smoke=False):
    from .gate import (_load_or_encode_features, _load_or_compute_stage, _sha256,
                                 _legacy_anchor, load_m03_bundle, _encode_lewm, _encode_legacy,
                                 _outcome_report, _legacy_native_parity_preflight, _write_status)
    from .diagnostics import eda_report
    from .diagnostics import observability_report
    panels, refs = addresses(root)
    wanted = defaultdict(set)
    for keys in panels.values():
        for seed,t in keys:
            wanted[seed].add(t)
    if smoke:
        # Exercise the central fit/test transfer and both short and long prefixes.
        selected = [next(k for k in panels["exact961"] if seed_split(k[0]) == part)
                    for part in ("fit", "test", "tune")]
        selected.append(panels["policy104"][0])
        seed = selected[-1][0]
        selected.extend(k for k in panels["legacy751"] if k == (seed, 0))
        wanted = defaultdict(set)
        for seed, t in selected:
            wanted[seed].add(t)
    directory = output / "historical"
    directory.mkdir(exist_ok=True)
    # Reuse the feature-cache transaction format for raw per-seed replay shards.
    # Hash the actual raw sources as well as the source code in the outer contract.
    contract_digest = _sha256(output / "run.json")
    run_contract = json.loads((output/'run.json').read_text())['contract']
    from .gate import _sha
    replay_token = _sha(run_contract['replay_dependencies'])
    manifests, summaries, keys = {}, [], []
    policy = None
    for seed in sorted(wanted):
        cache = directory / "features" / f"replay.{seed}.pt"
        if seed < 14000 and not cache.exists() and policy is None:
            policy = _policy(root, device)
        values, digest = _load_or_encode_features(directory, arm="replay", split=str(seed),
            identity={"protocol": PROTOCOL, "replay_dependencies": replay_token}, sidecar_sha256=replay_token,
            encode=lambda s=seed: replay_seed(root, s, wanted[s], refs, settings, policy))
        manifests[seed] = digest
        keys.extend((seed, int(t)) for t in values["time"])
        # Scoring needs only the current pixel frame, never the padded prefix.
        summaries.append({k: (v[:, -1:] if k == "context" else v) for k,v in values.items()
                          if k != "past_actions"})
        _write_status(output, "historical_replay", seed=seed, roots=len(keys))
    del policy, refs
    if device == "cuda":
        torch.cuda.empty_cache()
    data = {k: torch.cat([r[k] for r in summaries]) for k in summaries[0]}
    del summaries
    all_features = {}
    for arm in ("raw", "tc", "direct_attention", "direct_mamba") if include_direct else ("raw", "tc"):
        if arm in checkpoints:
            bundle, payload, source = load_m03_bundle(checkpoints[arm], device=device, dataset_sha256=dataset_sha256)
            identity = {"checkpoint": _sha256(checkpoints[arm]), "source": source}
            del payload
        else:
            bundle, identity = _legacy_anchor(arm, device=device)
        chunks = []
        for seed in sorted(wanted):
            values = _load(directory / "features" / f"replay.{seed}.pt")["features"]
            def encode():
                parts, indices = [], []
                for length in values["context_length"].unique(sorted=True).tolist():
                    ix = torch.where(values["context_length"] == length)[0]
                    group = {k:v[ix] for k,v in values.items()}
                    group["context"] = group["context"][:, -length:]
                    group["past_actions"] = group["past_actions"][:, -(length-1):] if length>1 else group["past_actions"][:, :0]
                    if arm.startswith("direct"):
                        proof = _load_or_compute_stage(directory, f"parity_{arm}_{seed}_{length}",
                            {"identity": identity, "replay": manifests[seed], "length": length},
                            lambda: _legacy_native_parity_preflight(bundle, group, settings))
                        if proof["status"] != "pass":
                            raise ValueError(f"m03_history: native parity failed {arm}/{seed}/{length}")
                    parts.append((_encode_legacy if arm.startswith("direct") else _encode_lewm)(bundle, group, settings))
                    indices.append(ix)
                order = torch.argsort(torch.cat(indices))
                return {k:torch.cat([p[k] for p in parts])[order] for k in parts[0]}
            features, _ = _load_or_encode_features(directory, arm=arm, split=str(seed), identity=identity,
                                                   sidecar_sha256=manifests[seed], encode=encode)
            chunks.append(features)
            _write_status(output, "historical_encoding", arm=arm, seed=seed)
        all_features[arm] = {k:torch.cat([p[k] for p in chunks]) for k in chunks[0]}
        del bundle, chunks
        if device == "cuda":
            torch.cuda.empty_cache()
    report = {"protocol": PROTOCOL, "mode": "structural_smoke_not_a_result" if smoke else "historical_regression",
              "split": "paired-seed SHA256, first 8 digest bytes little-endian modulo 10; fit<7, tune<8, test otherwise",
              "tune": "reserved_unused_fixed_probe_recipe", "bootstrap_unit": "rollout_seed",
              "comparison": "descriptive_trained_systems_not_architecture_attribution", "panels": {}}
    for name in PANELS:
        train_idx, dev_idx = split_indices(keys, panels, name)
        if not len(train_idx) or not len(dev_idx):
            if not smoke:
                raise ValueError(f"m03_history: empty split for {name}")
            report["panels"][name] = {"status": "smoke_panel_not_sampled"}
            continue
        train, dev = ({k:v[idx] for k,v in data.items()} for idx in (train_idx, dev_idx))
        features = {arm: {f"{split}_{k}":v[idx] for split,idx in (("train",train_idx),("dev",dev_idx)) for k,v in f.items()}
                    for arm,f in all_features.items()}
        metadata = {"run": contract_digest, "panel": name, "replay_manifests": manifests,
                    "fit_keys": [keys[i] for i in train_idx], "dev_keys": [keys[i] for i in dev_idx]}
        def score_panel():
            result = {}
            for component, compute in (
                ('all_action_one_step', lambda:_outcome_report(train,dev,features,settings,device=device)),
                ('eda', lambda:eda_report(train,dev,features,settings,device=device)),
                ('observability', lambda:observability_report(train,dev,settings,device=device))):
                _write_status(output,'historical_scoring',panel=name,component=component)
                result[component] = _load_or_compute_stage(directory,name+'_'+component,metadata,compute)
            return result
        result = _load_or_compute_stage(directory,name,metadata,score_panel)
        report["panels"][name] = {"fit_roots": len(train_idx), "evaluation_roots": len(dev_idx),
                                  "evaluation_seed_clusters": len(dev["episode"].unique()),
                                  "short_prefix_roots": int((dev["context_length"] < 64).sum()),
                                  "keys": metadata["dev_keys"], **result}
    report["overlap"] = {f"{a}:{b}":len(set(panels[a]) & set(panels[b])) for i,a in enumerate(PANELS) for b in PANELS[i+1:]}
    report["native_direct_parity"] = {p.stem: json.loads(p.read_text())["result"]
                                      for p in sorted((directory / "stages").glob("parity_*.json"))}
    report["m4_authorized"] = False
    return report


def run_memory(baseline, output, settings, *, device, smoke=False, allow_incomplete=False, checkpoints=None):
    """Supplement a sealed gate using its exact primary and historical raw rows.

    Input manifests and their tensor bytes are checked before adopting records.
    No legacy representation cache is used as a Mamba-state representation.
    Completed baseline results are embedded in the companion's final report.
    """
    from dataclasses import asdict
    from .gate import (ROOT, _input_identity, _sha, _sha256, _open_run, _atomic_json_save,
                      _load_or_encode_features, _load_or_compute_stage, _write_status, load_m03_bundle, atomic_manifest)
    from .diagnostics import encode_memory, memory_report, MEMORY_CONTEXTS
    import os
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    record = json.loads((baseline/'run.json').read_text())
    if record.get('contract_sha256') != _sha(record['contract']):
        raise ValueError('m03_memory: invalid baseline contract')
    parent = record['contract']
    checkpoints = checkpoints or {a: Path(parent[a+'_checkpoint']['path']) for a in ('raw','tc')}
    if not smoke and parent['execution']['structural_smoke']:
        raise ValueError('m03_memory: a smoke baseline cannot become a full result')
    if not (baseline/'report.json').exists() and not (smoke or allow_incomplete):
        raise ValueError('m03_memory: finish the baseline before the full supplement')
    if device != 'cuda' and not smoke:
        raise ValueError('m03_memory: full evaluation requires CUDA')
    # Evaluator relocation is explicit. Everything outside those retired gate
    # modules must still have the same bytes, including replay and simulator code.
    retired = {str(ROOT/f'd4mj/m03_{name}.py') for name in ('capability','history','diagnostics','observability')}
    retired.update(str(p) for p in (ROOT/'d4mj/m03').glob('*.py'))
    def verify_sources(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and value['path'] not in retired:
                if _sha256(Path(value['path'])) != value['sha256']:
                    raise ValueError(f"m03_memory: baseline source/input drift: {value['path']}")
            for v in value.values(): verify_sources(v)
        elif isinstance(value, list):
            for v in value: verify_sources(v)
    verify_sources(parent)
    manifest_path = baseline/'sidecar/manifest.json'
    side_manifest = json.loads(manifest_path.read_text())
    from .cache import REPLAY_SETTINGS, content, active
    if any(side_manifest['settings'][k] != parent['settings'][k] for k in REPLAY_SETTINGS) or side_manifest['dataset_sha256'] != parent['dataset']['sha256']:
        raise ValueError('m03_memory: primary replay settings or dataset mismatch')
    if not smoke and any(getattr(settings,k) != parent['settings'][k] for k in REPLAY_SETTINGS):
        raise ValueError('m03_memory: requested raw replay settings differ from baseline')
    side_path = baseline/'sidecar/sidecar.probe_only.pt'
    if _sha256(side_path) != side_manifest['sidecar']['sha256']:
        raise ValueError('m03_memory: primary replay bytes changed')
    shards = sorted((baseline/'historical/features').glob('replay.*.manifest.json'))
    if not shards and not smoke and parent.get('historical_panels_included'):
        raise ValueError('m03_memory: missing historical replay panels')
    if smoke and shards:
        # Sample all historical panel types, including true-BOS short prefixes.
        panels, _ = addresses(ROOT)
        selected = [next(k for k in panels['exact961'] if seed_split(k[0]) == part) for part in ('fit','test','tune')]
        selected.append(panels['policy104'][0])
        selected += [k for k in panels['legacy751'] if k == (selected[-1][0], 0)]
        selected = set(selected)
        shards = [p for p in shards if int(p.name.split('.')[1]) in {k[0] for k in selected}]
    contract = {'schema': 'd4mj_m03_memory_run', 'baseline': _input_identity(baseline/'run.json'),
                'baseline_report': _input_identity(baseline/'report.json') if (baseline/'report.json').exists() else None,
                'primary_manifest': _input_identity(manifest_path),
                'historical_replay_manifests': [_input_identity(p) for p in shards],
                'sources': [_input_identity(Path(__file__)), _input_identity(Path(__file__).with_name('diagnostics.py')),
                            _input_identity(Path(__file__).with_name('gate.py')),
                            _input_identity(Path(__file__).with_name('cache.py'))],
                'checkpoints': {a:_input_identity(p) for a,p in checkpoints.items()},
                'cache_compatibility': active().compatibility_identity if active() is not None else None,
                'settings': asdict(settings), 'contexts': list(MEMORY_CONTEXTS), 'device': device, 'smoke': smoke}
    digest = _open_run(output, contract)
    if (output/'report.json').exists():
        result = json.loads((output/'report.json').read_text())
        if result.get('run_contract_sha256') != digest:
            raise ValueError('m03_memory: report contract mismatch')
        return result
    _write_status(output, 'memory_inputs', baseline=str(baseline))
    sidecar = _load(side_path)
    raw_rows = {f'primary_{s}': v for s,v in sidecar['splits'].items()}
    if smoke:
        raw_rows = {k: {n:t[:8 if k.endswith('train') else 4] for n,t in v.items()} for k,v in raw_rows.items()}
    # Store only labels and current frame in RAM. Native 64-frame pixel shards
    # are read one seed at a time during encoding.
    data = {}; replay_ids = {}; keys = []
    for p in shards:
        m = json.loads(p.read_text()); tensor_path = p.with_name(p.name.replace('.manifest.json','.pt'))
        if _sha256(tensor_path) != m['sha256']:
            raise ValueError(f'm03_memory: replay hash mismatch: {tensor_path}')
        payload = _load(tensor_path)
        replay_identity = ({'protocol':PROTOCOL,'replay_dependencies':_sha(parent['replay_dependencies'])}
                           if 'replay_dependencies' in parent else {'protocol':PROTOCOL,'contract':_sha256(baseline/'run.json')})
        expected = {'arm':'replay','split':p.name.split('.')[1], 'identity':replay_identity,
                    'sidecar_sha256': replay_identity.get('replay_dependencies',replay_identity.get('contract'))}
        if payload['metadata'] != expected or m['metadata_sha256'] != _sha(expected):
            raise ValueError('m03_memory: historical replay provenance mismatch')
        values = payload['features']
        if smoke:
            ix = torch.tensor([i for i,(e,t) in enumerate(zip(values['episode'].tolist(),values['time'].tolist())) if (e,t) in selected])
            values = {k:v[ix] for k,v in values.items()}
        name = 'historical_'+p.name.split('.')[1]
        replay_ids[name] = {'replay': m.get('dependency_key', m['sha256']), 'rows': list(zip(values['episode'].tolist(),values['time'].tolist()))}
        data[name] = {k: (v[:, -1:] if k == 'context' else v) for k,v in values.items() if k not in ('past_actions', 'successors', 'oracle_full_state', 'oracle_visible', 'oracle_timing')}
        keys.extend(replay_ids[name]['rows'])
    manifests = {}
    def encode_names(names):
        for arm, checkpoint in checkpoints.items():
            bundle, payload, source = load_m03_bundle(checkpoint, device=device, dataset_sha256=side_manifest['dataset_sha256'])
            del payload
            identity = {'checkpoint': _input_identity(checkpoint), 'source': source}
            for name in names:
                def encode(name=name):
                    if name in raw_rows:
                        values = raw_rows[name]
                    else:
                        seed = name.split('_')[1]
                        values = _load(baseline/f'historical/features/replay.{seed}.pt')['features']
                        if smoke:
                            ix = torch.tensor([i for i,(e,t) in enumerate(zip(values['episode'].tolist(),values['time'].tolist())) if (e,t) in selected])
                            values = {k:v[ix] for k,v in values.items()}
                    return encode_memory(bundle, values, settings)
                _, manifests[f'{arm}:{name}'] = _load_or_encode_features(output, arm=arm, split=name, identity=identity,
                    sidecar_sha256=_sha(replay_ids[name] if name in replay_ids else {'data':side_manifest['sidecar']['sha256'],
                        'rows':list(zip(raw_rows[name]['episode'].tolist(),raw_rows[name]['time'].tolist()))}), encode=encode)
                _write_status(output, 'memory_encoding', arm=arm, shard=name)
            del bundle
            if device == 'cuda': torch.cuda.empty_cache()
    encode_names(list(raw_rows))
    metadata = {'run_contract': digest, 'features': manifests}
    # Avoid reopening each tensor file for every feature key.
    def features(names):
        result = {}
        for arm in ('raw','tc'):
            chunks = [_load(output/f'features/{arm}.{name}.pt')['features'] for name in names]
            result[arm] = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        return result
    primary = {s: features(['primary_'+s]) for s in ('train','dev')}
    primary_features = {a: {s:primary[s][a] for s in primary} for a in ('raw','tc')}
    def score(name, train, dev, rows):
        def progress(task, partial):
            _write_status(output,'memory_scoring',panel=name,completed_task=task)
            atomic_manifest(output/'partial.json',{'status':'partial','panel':name,'completed_task':task,
                'run_contract_sha256':digest,'result':partial,'m4_authorized':False})
        _write_status(output,'memory_scoring',panel=name)
        return memory_report(train,dev,rows,settings,device=device,progress=progress)
    results = {'primary': _load_or_compute_stage(output, 'memory_primary', dict(metadata),
        lambda: score('primary',raw_rows['primary_train'],raw_rows['primary_dev'],primary_features))}
    del primary, primary_features, sidecar
    raw_rows = {}
    _write_status(output, 'memory_scoring', panel='primary')
    if data:
        encode_names(list(data))
        panel_membership, _ = addresses(ROOT)
        labels = {k: torch.cat([v[k] for v in data.values()]) for k in next(iter(data.values()))}
        all_features = features(list(data))
        for name in PANELS:
            fit, test = split_indices(keys, panel_membership, name)
            if not len(fit) or not len(test):
                if not smoke: raise ValueError('m03_memory: missing historical split')
                results[name] = {'status': 'smoke_panel_not_sampled'}; continue
            train, dev = ({k:v[ix] for k,v in labels.items()} for ix in (fit,test))
            rows = {arm: {split:{k:v[ix] for k,v in f.items()} for split,ix in (('train',fit),('dev',test))}
                    for arm,f in all_features.items()}
            results[name] = _load_or_compute_stage(output, 'memory_'+name, metadata | {'fit': [keys[i] for i in fit], 'test':[keys[i] for i in test]},
                lambda: score(name,train,dev,rows))
            _write_status(output, 'memory_scoring', panel=name)
    report = {'schema': 'd4mj_m03_memory', 'run_contract_sha256': digest, 'panels': results,
              'baseline': json.loads((baseline/'report.json').read_text()) if (baseline/'report.json').exists() else contract['baseline'],
              'decision': {'m4_authorized': False, 'status': 'structural_smoke_not_a_result' if smoke else 'measured_pending_capability_review',
                           'suite_complete': bool(not smoke and len(results) == 5 and parent['execution']['direct_anchors_included'] and (baseline/'report.json').exists() and
                               all(contract['checkpoints'][a]['sha256'] == parent[a+'_checkpoint']['sha256'] for a in ('raw','tc'))),
                           'promotion': 'requires_review_of_coverage_and_semantic_results; no automatic scalar pass'}}
    _atomic_json_save(output/'report.json', report)
    _write_status(output, 'complete', report_sha256=_sha256(output/'report.json'))
    return report
