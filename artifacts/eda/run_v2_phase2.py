"""Phase 2 carrying the full v2 contract: counterfactual dynamics AND outcome heads.

`train_agent` keeps training the world, on ordinary trajectories only. Without this the
v2 world gets 10,000 more steps with no same-state action contrast, and the reward and
continuation heads fit logged trajectories and terminal tails -- the coverage that made
death prediction fail. The corpus carries a reward and a termination for every action and
every surviving second step, so the heads can see conditional outcomes.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts"))

from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import load_episodes
from d4mj.train import train_agent
from d4mj.transition import World

DEVICE = "cuda"
N_ACTIONS = 17
CACHE = HERE / "latent_cache_64"


def load_v2():
    rows = []
    for path in sorted(glob.glob(str(HERE / "broad_latents_v2" / "shard-*.pt"))):
        rows += torch.load(path, weights_only=False)
    assert rows, "no v2 latents; run encode_broad_forks.py"
    pack = {
        "history": torch.stack([r["z_history"] for r in rows]).float(),
        "branch": torch.stack([r["z_branch"] for r in rows]).float(),
        "second": torch.stack([r["z_second"] for r in rows]).float(),
        "valid": torch.stack([r["second_valid"] for r in rows]).bool(),
        "led": torch.stack([r["led_to_action"] for r in rows]).long(),
        "reward": torch.stack([r["reward"] for r in rows]).float(),
        "terminated": torch.stack([r["terminated"] for r in rows]).bool(),
        "second_reward": torch.stack([r["second_reward"] for r in rows]).float(),
        "second_terminated": torch.stack([r["second_terminated"] for r in rows]).bool(),
    }
    assert pack["second"][~pack["valid"]].abs().sum() == 0, "an invalid second target is not zero"
    return pack


def sampler(pack, roots: int, seed: int):
    """One batch of roots per step, every action of each, on the device."""
    order = torch.randperm(len(pack["history"]), generator=torch.Generator().manual_seed(seed))

    def draw(step: int) -> dict:
        index = order[(step * roots + torch.arange(roots)) % len(order)]
        a = N_ACTIONS
        rows = index.repeat_interleave(a)
        actions = torch.arange(a).repeat(len(index))
        flat = (rows, actions)
        return {
            "history": pack["history"][index].to(DEVICE),
            "led": pack["led"][index].to(DEVICE),
            "actions": actions.view(len(index), a).to(DEVICE),
            "branch": pack["branch"][flat].to(DEVICE),
            "second": pack["second"][flat].to(DEVICE),
            "valid": pack["valid"][flat].float().to(DEVICE),
            "reward": pack["reward"][flat].to(DEVICE),
            "continuation": (~pack["terminated"][flat]).float().to(DEVICE),
            "second_reward": pack["second_reward"][flat].to(DEVICE),
            "second_continuation": (~pack["second_terminated"][flat]).float().to(DEVICE),
        }

    return draw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--roots", type=int, default=2, help="v2 roots per step")
    parser.add_argument("--mass", type=float, default=0.2)
    parser.add_argument("--expert", type=int, default=320)
    # An alignment arm is a separate world that needs its own Phase 2. Without explicit
    # routing it would either read the control's world or overwrite the control's output.
    parser.add_argument("--rollout-only", action="store_true",
                        help="score the heads only on the generated blocks, the ones "
                             "where the observed and generated readouts can differ")
    parser.add_argument("--freeze-world", action="store_true",
                        help="head-extraction ceiling: optimise the heads only, so a "
                             "difference between arms cannot come from the world")
    parser.add_argument("--paired-semantic", action="store_true",
                        help="score the observed readout alongside the generated one "
                             "against the same real targets, at half weight each")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    source = args.source or HERE / f"v2_direct_{args.arm}"
    out = args.out or HERE / f"v2_phase2_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    # The world arrives as a bare state dict, so nothing here would catch a config that
    # differs from the one it was trained under. Phase 2 keeps training the world for
    # another 10k steps, so inheriting the Phase-1B alignment weight is what stops this
    # phase from quietly undoing the intervention it is meant to carry.
    trained = json.loads((source / "training_report.json").read_text())
    config = replace(base, transition="direct", time_mixer=args.arm,
                     align_weight=trained.get("align_weight", 0.0))
    world = World(config).to(DEVICE)
    world.load_state_dict(torch.load(source / "world.pt", weights_only=False)["world"])

    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)
    pack = load_v2()
    print(f"phase 2 {args.arm}: {len(episodes)} episodes, {len(pack['history'])} v2 roots, "
          f"{args.roots * N_ACTIONS} counterfactual targets/step, mass {args.mass}", flush=True)

    heads = train_agent(episodes, world, args.steps, config,
                        checkpoint=out / "phase2.pt", world_steps=20_000,
                        counterfactual=sampler(pack, args.roots, config.seed + 11),
                        counterfactual_mass=args.mass,
                        paired_semantic=args.paired_semantic,
                        rollout_only=args.rollout_only,
                        freeze_world=args.freeze_world)
    save(out / "phase2_final.pt", config, part0=world, part1=heads)
    torch.save({"world": world.state_dict()}, out / "world.pt")
    # the evaluators read the mixer from here; without it they default to attention and
    # cannot load a mamba world
    (out / "training_report.json").write_text(json.dumps(
        {"phase": 2, "arm": args.arm, "time_mixer": args.arm, "steps": args.steps,
         "counterfactual_roots": args.roots, "counterfactual_mass": args.mass,
         "align_weight": config.align_weight, "direct_rollout": config.direct_rollout,
         "paired_semantic": args.paired_semantic, "freeze_world": args.freeze_world,
         "rollout_only": args.rollout_only,
         "source": str(source), "out": str(out), "seed": config.seed}, indent=2))
    print(f"phase 2 {args.arm} complete", flush=True)


if __name__ == "__main__":
    main()
