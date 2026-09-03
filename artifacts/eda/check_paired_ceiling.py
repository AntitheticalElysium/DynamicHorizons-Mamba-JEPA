"""Does paired semantic supervision make a generated readout mean what an observed one means?

One frozen teacher -- the 10k `v2_phase2_attention` world and BC head -- generates the
entire evaluation trace. Neither candidate head ever chooses an action, so both arms see
identical seeds, roots, actions and simulator randomness, and a difference between them
cannot come from different state visitation.

At each real successor the teacher's own outgoing distribution q is the target for BOTH
candidate paths: the candidate's observed readout and its generated one. The policy target
is the NEXT outgoing decision at that successor, never the incoming action that produced
it, and terminal successors are dropped because no next decision exists.

This is a frozen-context diagnostic. It says whether the two candidate readout paths
reproduce established BC semantics; it is not executed-control evidence.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads, _distribution_loss
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.imagination import _expect
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World, advance, observe

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


def _tokens(action: int) -> torch.Tensor:
    return torch.full((1, 1), action, dtype=torch.long, device=DEVICE)


def _policy(heads: Heads, agent: torch.Tensor) -> torch.Tensor:
    return heads(agent)["policy"][:, -1, 0].softmax(-1)[0]


def _kl(target: torch.Tensor, other: torch.Tensor) -> float:
    target, other = target.clamp_min(1e-12), other.clamp_min(1e-12)
    return float((target * (target.log() - other.log())).sum())


def _js(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.clamp_min(1e-12), q.clamp_min(1e-12)
    mix = 0.5 * (p + q)
    return float(0.5 * (p * (p.log() - mix.log())).sum()
                 + 0.5 * (q * (q.log() - mix.log())).sum())


def _outcome(heads: Heads, agent: torch.Tensor, reward: float, alive: float) -> dict:
    readout = heads(agent)
    logits = readout["reward"][:, -1, 0]
    truth = torch.tensor([[reward]], device=DEVICE)
    keep = readout["continuation"][:, -1, 0]
    return {
        "reward_nll": float(_distribution_loss(logits, truth, heads.centers).mean()),
        "reward_mae": abs(float(_expect(logits, heads.centers)) - reward),
        "continuation_bce": float(F.binary_cross_entropy_with_logits(
            keep, torch.tensor([alive], device=DEVICE))),
        "continuation_probability": float(keep.sigmoid()),
    }


def _load(folder: Path, base: Config) -> tuple[World, Heads, Config]:
    trained = json.loads((folder / "training_report.json").read_text())
    config = replace(base, transition="direct",
                     time_mixer=trained.get("time_mixer", "attention"),
                     align_weight=trained.get("align_weight", 0.0))
    world, heads = World(config).to(DEVICE), Heads(config).to(DEVICE)
    load(folder / "phase2_final.pt", config, part0=world, part1=heads)
    return world.eval(), heads.eval(), config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=HERE / "v2_phase2_attention")
    parser.add_argument("--arms", nargs="+",
                        default=["control=v2_ceiling_control", "paired=v2_ceiling_paired"])
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/paired_ceiling")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    encoder.eval()

    teacher_world, teacher_heads, teacher_config = _load(args.teacher, base)
    arms, reports = {}, {}
    for entry in args.arms:
        name, folder = entry.split("=", 1)
        arms[name] = _load(HERE / folder, base)
        reports[name] = json.loads((HERE / folder / "training_report.json").read_text())
    names = list(arms)
    # A head ceiling only means anything if the arms differ in one thing. Checked here
    # rather than trusted: `freeze_world` is what makes the world identical, and the rest
    # is what makes the training matched.
    for name in names:
        assert reports[name].get("freeze_world"), f"{name} did not freeze its world"
    shared = ("source", "steps", "seed", "arm", "time_mixer", "align_weight",
              "counterfactual_roots", "counterfactual_mass", "direct_rollout")
    for key in shared:
        values = {reports[name].get(key) for name in names}
        assert len(values) == 1, f"arms differ on {key}: {values}"
    assert {reports[name].get("paired_semantic") for name in names} == {True, False}, (
        "the arms must differ exactly in paired_semantic")
    # The ceiling froze one Phase-1B world in both arms, so their worlds must be
    # identical and the readouts can be computed once. Asserted, not assumed: if they
    # ever diverge, the comparison stops being head-only and this must be revisited.
    first = arms[names[0]][0].state_dict()
    for name in names[1:]:
        for key, value in arms[name][0].state_dict().items():
            assert torch.equal(first[key], value), f"{name} world differs at {key}"
    candidate_world, candidate_config = arms[names[0]][0], arms[names[0]][2]
    print(f"teacher {args.teacher.name}; arms {names}; candidate worlds identical",
          flush=True)

    rows = []
    for seed in range(args.seed_base, args.seed_base + args.episodes):
        rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**20)
        observation, env_state = reset(seed)
        # The initial observation is ingested once, outside the loop. Each iteration then
        # observes exactly one successor: observing at both the bottom of a step and the
        # top of the next advanced the recurrent memory twice per simulator transition.
        opening = patchify(observation[None, None], base.patch).to(DEVICE)
        teacher_state, teacher_agent = observe(
            teacher_world, encoder, None, _tokens(base.n_actions), opening, rng, teacher_config)
        candidate_state, candidate_agent = observe(
            candidate_world, encoder, None, _tokens(base.n_actions), opening, rng,
            candidate_config)
        chains = []
        for index in range(args.limit):
            # Every state is a root: a chain starts here and is advanced by whatever the
            # teacher does next, so one episode yields many (root, depth) samples.
            chains.append({"start": index, "state": candidate_state.world})
            before = (teacher_state.world.step, candidate_state.world.step)

            action = int(torch.multinomial(
                _policy(teacher_heads, teacher_agent), 1, generator=policy_rng))
            observation, env_state, reward, terminated, truncated = env_step(
                env_state, action, seed + index + 1)
            alive = 0.0 if terminated else 1.0

            successor = patchify(observation[None, None], base.patch).to(DEVICE)
            teacher_state, teacher_agent = observe(
                teacher_world, encoder, teacher_state, _tokens(action), successor, rng,
                teacher_config)
            candidate_state, candidate_agent = observe(
                candidate_world, encoder, candidate_state, _tokens(action), successor, rng,
                candidate_config)
            after = (teacher_state.world.step, candidate_state.world.step)
            assert after == (before[0] + 1, before[1] + 1), (
                f"recurrent state advanced {after} from {before} on one simulator step")
            target = _policy(teacher_heads, teacher_agent)

            surviving = []
            for chain in chains:
                chain["state"], generated = advance(
                    candidate_world, chain["state"], _tokens(action), rng, candidate_config)
                depth = index - chain["start"] + 1
                row = {"seed": seed, "root": chain["start"], "depth": depth,
                       "terminated": bool(terminated), "truncated": bool(truncated),
                       # A truncated successor has no next decision either, so it is
                       # excluded from policy metrics; continuation still follows the
                       # training convention, where truncation is not death.
                       "decision_valid": not (terminated or truncated),
                       "reward": float(reward)}
                for name in names:
                    heads = arms[name][1]
                    seen, made = (_policy(heads, candidate_agent), _policy(heads, generated))
                    row[name] = {
                        "kl_teacher_observed": _kl(target, seen),
                        "kl_teacher_generated": _kl(target, made),
                        "top1_teacher_observed": float(int(target.argmax()) == int(seen.argmax())),
                        "top1_teacher_generated": float(int(target.argmax()) == int(made.argmax())),
                        "kl_observed_generated": _kl(seen, made),
                        "js_observed_generated": _js(seen, made),
                        "top1_observed_generated": float(int(seen.argmax()) == int(made.argmax())),
                        "observed": _outcome(heads, candidate_agent, float(reward), alive),
                        "generated": _outcome(heads, generated, float(reward), alive),
                    }
                row["teacher"] = _outcome(teacher_heads, teacher_agent, float(reward), alive)
                rows.append(row)
                if depth < args.depth:
                    surviving.append(chain)
            chains = surviving
            if terminated or truncated:
                break
        print(f"  seed {seed}: {index + 1} steps", flush=True)

    (args.out / "paired_ceiling.json").write_text(json.dumps(
        {"teacher": str(args.teacher / "phase2_final.pt"),
         "arms": {n: str(HERE / e.split("=", 1)[1] / "phase2_final.pt")
                  for n, e in zip(names, args.arms)},
         "seed_base": args.seed_base, "episodes": args.episodes, "depth": args.depth,
         "convention": "policy target is the teacher's outgoing distribution at the real "
                       "successor; terminal successors excluded from policy metrics; "
                       "continuation target is 0 if terminated else 1, truncation is not "
                       "death; reward scored at lead 0 against the simulator reward of "
                       "the incoming action",
         "rows": rows}, indent=2, default=float))
    _report(rows, names, args.depth)


def _report(rows, names, depth) -> None:
    picker = np.random.default_rng(19)

    def paired(selector, subset):
        """Average within rollout seed first, then bootstrap whole seeds."""
        seeds = sorted({row["seed"] for row in subset})
        per = []
        for seed in seeds:
            values = [selector(row) for row in subset if row["seed"] == seed]
            values = [v for v in values if v is not None]
            per.append(np.mean(values) if values else np.nan)
        per = np.array(per, dtype=float)
        per = per[~np.isnan(per)]
        if not len(per):
            return float("nan"), float("nan"), float("nan")
        draws = per[picker.integers(0, len(per), (4000, len(per)))].mean(1)
        return float(per.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))

    for level in range(1, depth + 1):
        alive = [r for r in rows if r["depth"] == level and r["decision_valid"]]
        every = [r for r in rows if r["depth"] == level]
        print(f"\n=== depth {level}: {len(alive)} transitions with a next decision, "
              f"{len({r['seed'] for r in every})} DEV seeds ===")
        print(f"{'metric':<34}" + "".join(f"{n:>26}" for n in names))
        for label, key in (("KL(teacher||observed)", "kl_teacher_observed"),
                           ("KL(teacher||generated)", "kl_teacher_generated"),
                           ("top-1 teacher vs observed", "top1_teacher_observed"),
                           ("top-1 teacher vs generated", "top1_teacher_generated"),
                           ("KL(observed||generated)", "kl_observed_generated"),
                           ("JS(observed,generated)", "js_observed_generated"),
                           ("top-1 observed vs generated", "top1_observed_generated")):
            cells = []
            for name in names:
                mean, low, high = paired(lambda r, n=name, k=key: r[n][k], alive)
                cells.append(f"{mean:>11.4f} [{low:.3f},{high:.3f}]")
            print(f"{label:<34}" + "".join(f"{c:>26}" for c in cells))
        for path in ("observed", "generated"):
            for label, key in (("reward NLL", "reward_nll"), ("reward MAE", "reward_mae"),
                               ("continuation BCE", "continuation_bce"),
                               ("continuation p", "continuation_probability")):
                cells = []
                for name in names:
                    mean, low, high = paired(
                        lambda r, n=name, p=path, k=key: r[n][p][k], every)
                    cells.append(f"{mean:>11.4f} [{low:.3f},{high:.3f}]")
                print(f"{path + ' ' + label:<34}" + "".join(f"{c:>26}" for c in cells))
        for label, key in (("teacher reward NLL", "reward_nll"),
                           ("teacher continuation BCE", "continuation_bce")):
            mean, low, high = paired(lambda r, k=key: r["teacher"][k], every)
            print(f"{label:<34}{mean:>11.4f} [{low:.3f},{high:.3f}]  (absolute reference)")

        if len(names) == 2:
            a, b = names
            print(f"\n  paired difference, {b} minus {a}, same observations:")
            for label, key, subset in (
                    ("KL(teacher||generated)", "kl_teacher_generated", alive),
                    ("top-1 teacher vs generated", "top1_teacher_generated", alive),
                    ("KL(observed||generated)", "kl_observed_generated", alive),
                    ("top-1 observed vs generated", "top1_observed_generated", alive),
                    ("KL(teacher||observed)", "kl_teacher_observed", alive)):
                mean, low, high = paired(
                    lambda r, k=key, x=a, y=b: r[y][k] - r[x][k], subset)
                print(f"    {label:<32}{mean:>+9.4f} [{low:+.4f},{high:+.4f}]")
            for path in ("observed", "generated"):
                for label, key in (("reward NLL", "reward_nll"), ("reward MAE", "reward_mae"),
                                   ("continuation BCE", "continuation_bce")):
                    mean, low, high = paired(
                        lambda r, p=path, k=key, x=a, y=b: r[y][p][k] - r[x][p][k], every)
                    print(f"    {path + ' ' + label:<32}{mean:>+9.4f} [{low:+.4f},{high:+.4f}]")


if __name__ == "__main__":
    main()
