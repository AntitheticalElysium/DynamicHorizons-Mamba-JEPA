"""Do the control and treatment objectives actually differ? Run this before training.

The first paired arm failed this test after the fact: `_direct_loss` replaces only the
last `direct_rollout` readouts, so a head loss averaged over every block compared two
objectives that were the same tensor on 87.5% of short rows and 96.9% of long ones, and
the gradients agreed to a cosine of 0.99987. There was nothing to find. This measures the
contrast first, so a null result is a result rather than an artefact.
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

from d4mj.agent import Heads, head_loss, head_targets
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import load_episodes, sample_batch
from d4mj.transition import World, transition_loss

DEVICE = "cuda"
CACHE = HERE / "latent_cache_64"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, default=HERE / "v2_ceiling_control")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--rollout-only", action="store_true")
    args = parser.parse_args()

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    world, heads = World(config).to(DEVICE), Heads(config).to(DEVICE)
    load(args.arm / "phase2_final.pt", config, part0=world, part1=heads)
    for parameter in world.parameters():
        parameter.requires_grad_(False)

    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)
    sampler = torch.Generator().manual_seed(7)

    def gradient(loss):
        heads.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return torch.cat([p.grad.flatten() for p in heads.parameters() if p.grad is not None])

    cosines, ratios, gaps, blocks_seen = [], [], [], []
    for step in range(args.batches):
        batch = sample_batch(episodes, sampler, config, step, 2500, mixture=True)
        batch = type(batch)(**{k: (v.to(DEVICE) if torch.is_tensor(v) else v)
                               for k, v in vars(batch).items()})
        rng = torch.Generator(device=DEVICE).manual_seed(step)
        _, agent, observed = transition_loss(world, batch, rng, config,
                                             return_agent=True, return_observed=True)
        window = None
        if args.rollout_only:
            window = torch.zeros(agent.shape[1], device=DEVICE)
            window[-config.direct_rollout:] = 1.0
        blocks_seen.append(agent.shape[1])
        targets = head_targets(batch, config)
        control = head_loss(heads(agent) | {"centers": heads.centers}, targets, config, window)
        other = head_loss(heads(observed) | {"centers": heads.centers}, targets, config, window)
        gaps.append({k: float((other[k] - control[k]).abs()
                              / control[k].abs().clamp_min(1e-9).detach()) for k in control})
        a = gradient(sum(control.values()))
        b = gradient(sum(0.5 * (control[k] + other[k]) for k in control))
        cosines.append(float(F.cosine_similarity(a, b, dim=0)))
        ratios.append(float(b.norm() / a.norm()))

    scope = "rollout positions only" if args.rollout_only else "all positions"
    print(f"\n{args.arm.name}, {args.batches} real batches, {scope}, "
          f"blocks {sorted(set(blocks_seen))}")
    print(f"  gradient cosine(control, treatment)   {np.mean(cosines):.6f} "
          f"(min {min(cosines):.6f})")
    print(f"  gradient norm ratio treatment/control {np.mean(ratios):.4f}")
    print("  per-head |observed - generated| / |generated|:")
    for key in gaps[0]:
        print(f"    {key:<14}{np.mean([g[key] for g in gaps]):.4f}")
    separated = np.mean(cosines) < 0.99
    print(f"\n  VERDICT: {'discriminating' if separated else 'NOT DISCRIMINATING -- do not run'}")


if __name__ == "__main__":
    main()
