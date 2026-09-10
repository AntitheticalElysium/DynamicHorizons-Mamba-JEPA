"""The 5k safety gate for an alignment arm: is it broken, and is it doing anything?

Phase 1B has no arm-specific BC head, so observed-state BC quality cannot be measured
here -- that gate waits for this world's own Phase 2. What can be measured is whether the
alignment term has damaged ordinary prediction or collapsed the readout it pulls on, and
whether it is contributing at the share it was sized for.

Both worlds are scored on identical batches. The readout is the quantity under
suspicion: the one-way firewall means it cannot influence the dynamics features, so the
term could be satisfied by making it insensitive rather than by improving generated
states. Effective rank and variance are what would catch that.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataclasses import asdict

from d4mj.config import Config
from d4mj.data import load_episodes, sample_batch
from d4mj.state import WorldState
from d4mj.transition import World, advance, commit_inputs, transition_loss

CACHE = HERE / "latent_cache_64"
# Set per run: the 5k gate is meant to be answerable while the arm is still training, and
# a 6 GB card cannot host both. CPU is slower but does not contend.
DEVICE = "cuda"


def _to(batch, device):
    return type(batch)(**{k: (v.to(device) if torch.is_tensor(v) else v)
                          for k, v in vars(batch).items()})


def _effective_rank(readout: torch.Tensor) -> float:
    """exp of the entropy of the normalised singular spectrum (Roy & Vetterli). A
    readout that has collapsed onto a few directions reports a low rank even when its
    variance is unremarkable."""
    flat = readout.reshape(-1, readout.shape[-1]).float()
    flat = flat - flat.mean(0, keepdim=True)
    values = torch.linalg.svdvals(flat)
    share = values / values.sum().clamp_min(1e-12)
    share = share[share > 0]
    return float(torch.exp(-(share * share.log()).sum()))


@torch.no_grad()
def _measure(world: World, config: Config, episodes, batches: int, seed: int) -> dict:
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 99)
    sampler = torch.Generator().manual_seed(seed)
    rows: dict[str, list] = {k: [] for k in (
        "teacher", "first", "second", "first_cosine", "second_cosine",
        "first_nse", "second_nse", "readout_variance", "readout_rank",
        "align_cosine", "align_mse", "base_loss", "align_contribution")}
    steps = config.direct_rollout
    for index in range(batches):
        batch = _to(sample_batch(episodes, sampler, config, index, batches), DEVICE)
        committed, conditioning = commit_inputs(batch.latents, rng, config)
        features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
        predicted = world.predict(features[:, :-1], batch.led_to_action[:, 1:])
        rows["teacher"].append(float((predicted - batch.latents[:, 1:]).pow(2).mean()))
        rows["readout_variance"].append(float(agent.float().var()))
        rows["readout_rank"].append(_effective_rank(agent))

        length = batch.latents.shape[1]
        start = length - steps
        prefix, _, memory = world(None, batch.led_to_action[:, :start],
                                  committed[:, :start], conditioning[:, :start])
        state = WorldState(batch.latents[:, start - 1:start], memory, start, prefix[:, -1:])
        produced = []
        for depth, position in enumerate(range(start, length)):
            state, made = advance(
                world, state, batch.led_to_action[:, position:position + 1], rng, config)
            truth = batch.latents[:, position:position + 1]
            name = "first" if depth == 0 else "second"
            rows[name].append(float((state.latent - truth).pow(2).mean()))
            rows[f"{name}_cosine"].append(float(F.cosine_similarity(
                state.latent.flatten(1), truth.flatten(1), dim=1).mean()))
            rows[f"{name}_nse"].append(float(
                (state.latent - truth).pow(2).mean() / truth.var().clamp_min(1e-12)))
            produced.append(made)
        generated = torch.cat(produced, dim=1)
        rows["align_mse"].append(float((generated - agent[:, start:]).pow(2).mean()))
        rows["align_cosine"].append(float(F.cosine_similarity(
            generated.flatten(1), agent[:, start:].flatten(1), dim=1).mean()))

        plain = replace(config, align_weight=0.0)
        seeded = torch.Generator(device=config.device).manual_seed(index)
        base = float(transition_loss(world, batch, seeded, plain))
        seeded = torch.Generator(device=config.device).manual_seed(index)
        rows["base_loss"].append(base)
        rows["align_contribution"].append(
            float(transition_loss(world, batch, seeded, config)) - base)
    return {k: float(np.mean(v)) for k, v in rows.items() if v}


def _load_world(path: Path, base: Config) -> tuple[World, Config]:
    """`checkpoint.load` maps tensors to the device they were saved on, which would put
    a cuda checkpoint back on the card this gate is trying not to contend for. The
    config is still verified here, field by field, with `device` the only exemption."""
    payload = torch.load(path, weights_only=False, map_location=base.device)
    stored = payload["config"]
    config = replace(base, transition="direct", time_mixer=stored["time_mixer"],
                     align_weight=stored.get("align_weight", 0.0), seed=stored["seed"])
    wanted = asdict(config)
    differs = sorted(k for k, v in stored.items() if k != "device" and wanted.get(k) != v)
    if differs:
        raise ValueError(f"checkpoint config differs from the one requested: {differs}")
    world = World(config).to(base.device)
    world.load_state_dict(payload["modules"]["part0"])
    return world, config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path,
                        default=HERE / "v2_direct_attention_align" / "world_005000.pt")
    parser.add_argument("--control", type=Path,
                        default=HERE / "v2_direct_attention" / "world_005000.pt")
    parser.add_argument("--batches", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    globals()["DEVICE"] = args.device
    args.out = args.out or args.aligned.parent / "align_gate.json"

    base = replace(Config(), n_latents=64, d_bottleneck=16, device=args.device)
    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)

    measured = {}
    for name, path in (("control", args.control), ("aligned", args.aligned)):
        world, config = _load_world(path, base)
        measured[name] = _measure(world.eval(), config, episodes, args.batches, args.seed)
        measured[name]["align_weight"] = config.align_weight
        measured[name]["checkpoint"] = str(path)
        print(f"  measured {name} ({path.name}, align_weight={config.align_weight})", flush=True)
        del world
        if args.device == "cuda":
            torch.cuda.empty_cache()

    control, aligned = measured["control"], measured["aligned"]
    args.out.write_text(json.dumps(
        {"batches": args.batches, "seed": args.seed, "measured": measured}, indent=2))

    print(f"\n{'quantity':<22}{'control':>12}{'aligned':>12}{'change':>12}{'%':>9}")
    order = ["teacher", "first", "second", "first_nse", "second_nse",
             "first_cosine", "second_cosine", "align_mse", "align_cosine",
             "readout_variance", "readout_rank", "base_loss", "align_contribution"]
    for key in order:
        a, b = control[key], aligned[key]
        share = 100 * (b - a) / a if a else float("nan")
        print(f"{key:<22}{a:>12.5f}{b:>12.5f}{b - a:>+12.5f}{share:>+9.1f}")
    ratio = aligned["align_contribution"] / (aligned["base_loss"] + aligned["align_contribution"])
    print(f"\n  alignment share of the aligned objective: {ratio:.3f} (sized for ~0.21)")
    worse = [k for k in ("teacher", "first", "second", "first_nse", "second_nse")
             if aligned[k] > control[k] * 1.10]
    collapsed = (aligned["readout_rank"] < control["readout_rank"] * 0.75
                 or aligned["readout_variance"] < control["readout_variance"] * 0.5)
    print(f"  successor regression >10%: {worse or 'none'}")
    print(f"  readout collapse: {collapsed}")
    print(f"  VERDICT: {'STOP' if worse or collapsed else 'continue'}")


if __name__ == "__main__":
    main()
