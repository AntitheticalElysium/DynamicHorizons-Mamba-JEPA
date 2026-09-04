"""Compare two worlds on one shared trajectory: does a deeper rollout hold its semantics?

`check_generated_drift.py` drives each arm with its own BC, so two arms diverge before
the root and their curves are not comparable. `check_paired_ceiling.py` shares a
trajectory but deliberately assumes one world, and relaxing that assumption would score
every arm on the first arm's world. This is the third case: several worlds, one
trajectory.

One frozen BC picks every action. Each arm then carries its OWN observed state -- its own
world, config and head -- over exactly the same frames and actions, and spawns a generated
chain at every step. The primary quantity is within-arm: how far an arm's generated
readout has drifted from its OWN observed readout at the same index. Agreement with the
teacher is secondary, because the arms have different heads and a shared teacher flatters
whichever head resembles it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World, advance, observe

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


def _tokens(action: int) -> torch.Tensor:
    return torch.full((1, 1), action, dtype=torch.long, device=DEVICE)


def _policy(heads: Heads, agent: torch.Tensor) -> torch.Tensor:
    return heads(agent)["policy"][:, -1, 0].softmax(-1)[0]


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.clamp_min(1e-12), q.clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum())


def _load(folder: Path, base: Config) -> tuple[World, Heads, Config]:
    """Each arm brings its own architecture. `sequence` and `dynamics_context` belong to
    the world; the tokenizer is trained separately and must not inherit them."""
    trained = json.loads((folder / "training_report.json").read_text())
    world_base = base
    if "sequence" in trained:
        world_base = replace(base, sequence=trained["sequence"],
                             sequence_long=trained["sequence_long"],
                             dynamics_context=trained["dynamics_context"])
    config = replace(world_base, transition="direct",
                     time_mixer=trained.get("time_mixer", "attention"),
                     align_weight=trained.get("align_weight", 0.0),
                     direct_rollout=trained.get("direct_rollout", base.direct_rollout))
    world, heads = World(config).to(DEVICE), Heads(config).to(DEVICE)
    load(folder / "phase2_final.pt", config, part0=world, part1=heads)
    return world.eval(), heads.eval(), config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=HERE / "v2_phase2_attention")
    parser.add_argument("--arms", nargs="+",
                        default=["r2=v2_phase2_attention", "r16=v2_phase2_attention_r16"])
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/multiworld_drift")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    deepest, wanted = max(args.depths), set(args.depths)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    encoder.eval()

    teacher_world, teacher_heads, teacher_config = _load(args.teacher, base)
    arms = {}
    for entry in args.arms:
        name, folder = entry.split("=", 1)
        arms[name] = _load(HERE / folder, base)
    names = list(arms)
    print(f"teacher {args.teacher.name}; arms " + ", ".join(
        f"{n} (rollout {arms[n][2].direct_rollout}, sequence {arms[n][2].sequence})"
        for n in names), flush=True)
    # If the teacher is one of the arms, that arm's observed readout IS the teacher's, so
    # every teacher-referenced number collapses onto its own within-arm value and the
    # secondary comparison is not a comparison at all. The primary metric is unaffected,
    # because the teacher only chooses actions.
    degenerate = next((n for n, e in zip(names, args.arms)
                       if (HERE / e.split("=", 1)[1]).resolve() == args.teacher.resolve()), None)
    if degenerate:
        print(f"  NOTE: teacher is arm '{degenerate}'; secondary teacher metrics are "
              f"degenerate for it and are not reported", flush=True)

    rows = []
    for seed in range(args.seed_base, args.seed_base + args.episodes):
        rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**20)
        observation, env_state = reset(seed)
        opening = patchify(observation[None, None], base.patch).to(DEVICE)
        teacher_state, teacher_agent = observe(
            teacher_world, encoder, None, _tokens(base.n_actions), opening, rng, teacher_config)
        # Every arm ingests the same opening frame through its own world.
        seen = {n: observe(arms[n][0], encoder, None, _tokens(base.n_actions), opening,
                           rng, arms[n][2]) for n in names}
        chains = {n: [] for n in names}

        for index in range(args.limit):
            for name in names:
                chains[name].append({"start": index, "state": seen[name][0].world})
            action = int(torch.multinomial(
                _policy(teacher_heads, teacher_agent), 1, generator=policy_rng))
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, seed + index + 1)
            successor = patchify(observation[None, None], base.patch).to(DEVICE)

            teacher_state, teacher_agent = observe(
                teacher_world, encoder, teacher_state, _tokens(action), successor, rng,
                teacher_config)
            target = _policy(teacher_heads, teacher_agent)
            for name in names:
                seen[name] = observe(arms[name][0], encoder, seen[name][0], _tokens(action),
                                     successor, rng, arms[name][2])

            for name in names:
                world, heads, config = arms[name]
                surviving = []
                for chain in chains[name]:
                    chain["state"], made = advance(
                        world, chain["state"], _tokens(action), rng, config)
                    depth = index - chain["start"] + 1
                    if depth in wanted:
                        observed = _policy(heads, seen[name][1])
                        generated = _policy(heads, made)
                        rows.append({
                            "seed": seed, "root": chain["start"], "depth": depth,
                            "arm": name, "terminal": bool(terminated or truncated),
                            # Primary: the arm against itself on the same index.
                            "kl_observed_generated": _kl(observed, generated),
                            "top1_observed_generated": float(
                                int(observed.argmax()) == int(generated.argmax())),
                            # Secondary: a shared yardstick that favours whichever head
                            # happens to resemble the teacher.
                            "kl_teacher_observed": _kl(target, observed),
                            "kl_teacher_generated": _kl(target, generated),
                            "top1_teacher_generated": float(
                                int(target.argmax()) == int(generated.argmax())),
                        })
                    if depth < deepest:
                        surviving.append(chain)
                chains[name] = surviving
            if terminated or truncated:
                break
        print(f"  seed {seed}: {index + 1} steps", flush=True)

    (args.out / "multiworld_drift.json").write_text(json.dumps(
        {"teacher": str(args.teacher), "arms": {n: str(HERE / e.split("=", 1)[1])
                                                for n, e in zip(names, args.arms)},
         "episodes": args.episodes, "depths": args.depths, "rows": rows}, indent=2,
        default=float))
    _report(rows, names, args.depths, degenerate)


def _report(rows, names, depths, degenerate=None) -> None:
    picker = np.random.default_rng(23)

    def per_seed(subset, arm, key):
        """Roots averaged within a rollout seed before anything is compared."""
        seeds = sorted({r["seed"] for r in subset})
        out = {}
        for seed in seeds:
            values = [r[key] for r in subset if r["seed"] == seed and r["arm"] == arm]
            if values:
                out[seed] = float(np.mean(values))
        return out

    reported = [("kl_observed_generated", "KL(observed||generated), within arm"),
                ("top1_observed_generated", "top-1 observed vs generated, within arm")]
    if not degenerate:
        reported += [("kl_teacher_generated", "KL(teacher||generated), secondary"),
                     ("top1_teacher_generated", "top-1 teacher vs generated, secondary")]
    for key, label in reported:
        print(f"\n=== {label} ===")
        print(f"{'depth':<7}{'n seeds':>9}" + "".join(f"{n:>14}" for n in names)
              + f"{'paired ' + names[-1] + ' - ' + names[0]:>30}")
        for depth in depths:
            subset = [r for r in rows if r["depth"] == depth and not r["terminal"]]
            if not subset:
                continue
            tables = {n: per_seed(subset, n, key) for n in names}
            shared = sorted(set.intersection(*(set(t) for t in tables.values())))
            if not shared:
                continue
            cells = "".join(f"{np.mean([tables[n][s] for s in shared]):>14.4f}" for n in names)
            gap = np.array([tables[names[-1]][s] - tables[names[0]][s] for s in shared])
            draws = gap[picker.integers(0, len(gap), (4000, len(gap)))].mean(1)
            low, high = np.quantile(draws, .025), np.quantile(draws, .975)
            mark = "" if low <= 0 <= high else "  *"
            print(f"{depth:<7}{len(shared):>9}{cells}"
                  f"{gap.mean():>+18.4f} [{low:+.4f},{high:+.4f}]{mark}")


if __name__ == "__main__":
    main()
