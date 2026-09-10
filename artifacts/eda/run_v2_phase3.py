"""Phase 3 on the raw Phase-2 heads, at Direct's trained horizon.

Loads `phase2_final.pt`, never `phase2_calibrated.pt` -- the affine calibrator failed its
cross-regime test and is not part of the model. Horizon is capped at `direct_rollout` per
S68: Direct trains two generated states and may not imagine past them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import load_episodes
from d4mj.train import train_actor
from d4mj.transition import World

DEVICE = "cuda"
CACHE = HERE / "latent_cache_64"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--steps", type=int, default=2500)
    # 10k is not an extension: train_actor puts total steps in its checkpoint
    # contract and the data schedule depends on the total, so a longer budget is
    # a fresh run from the same phase2_final in its own directory.
    parser.add_argument("--tag", default="")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    source = args.source or HERE / f"v2_phase2_{args.arm}"
    out = args.out or HERE / f"v2_phase3_{args.arm}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    trained = json.loads((source / "training_report.json").read_text())
    saved = replace(base, transition="direct", time_mixer=args.arm,
                    align_weight=trained.get("align_weight", 0.0))
    world, heads = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(source / "phase2_final.pt", saved, part0=world, part1=heads)
    config = replace(saved, horizon=saved.direct_rollout)

    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)
    print(f"phase 3 {args.arm}: {len(episodes)} episodes, horizon {config.horizon}, "
          f"{args.steps} steps", flush=True)

    actor = train_actor(episodes, world, heads, args.steps, config,
                        checkpoint=out / "phase3.pt")
    save(out / "phase3_final.pt", config, part0=world, part1=actor)
    (out / "training_report.json").write_text(json.dumps(
        {"phase": 3, "arm": args.arm, "time_mixer": args.arm, "steps": args.steps,
         "align_weight": saved.align_weight,
         "horizon": config.horizon, "source": str(source), "out": str(out),
         "seed": config.seed,
         "heads": "raw phase2_final, no calibrator"}, indent=2))
    print(f"phase 3 {args.arm} complete", flush=True)


if __name__ == "__main__":
    main()
