"""Where does the generated chain stop behaving like the observed one?

The hybrid arm showed that training on generated-state trajectories costs 2.13 of the
2.43 achievements Phase 3 gives up, while perfect rewards and continuations recover only
0.30. That indicts the generated path as a whole, not the value bootstrap specifically:
the critic bootstraps on generated states, the policy chooses from them, and both
networks are then deployed on observed ones.

This walks the observed and generated chains from identical roots under identical action
sequences and measures, at each depth: how far the latents and readouts have moved, what
the FROZEN BC policy does on each -- a trained semantic readout Phase 3 never touched --
what each Phase-3 critic values them at, and whether those values satisfy the Bellman
equation under true simulator outcomes. A final pass ranks all 17 one-step successors on
both paths to see whether the ordering survives at all.
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
sys.path.insert(0, str(ROOT / "artifacts"))

from d4mj.agent import Heads
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
CRITICS = {
    "oracle": ROOT / "artifacts/oracle_horizon_h2/progress.pt",
    "hybrid": ROOT / "artifacts/hybrid_context_h2/progress.pt",
    "modelled": ROOT / "artifacts/matched_context_h2/progress.pt",
}


def _tokens(action: int) -> torch.Tensor:
    return torch.full((1, 1), action, dtype=torch.long, device=DEVICE)


def _value(heads: Heads, agent: torch.Tensor) -> float:
    return float(_expect(heads(agent)["value"][:, -1], heads.centers))


def _divergence(p: torch.Tensor, q: torch.Tensor) -> tuple[float, float]:
    p, q = p.clamp_min(1e-12), q.clamp_min(1e-12)
    mix = 0.5 * (p + q)
    kl = float((p * (p.log() - q.log())).sum())
    js = float(0.5 * (p * (p.log() - mix.log())).sum() + 0.5 * (q * (q.log() - mix.log())).sum())
    return kl, js


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--ranking-roots", type=int, default=16)
    # Arm-specific by default: a shared path would overwrite the baseline's own report.
    parser.add_argument("--out", type=Path, default=None)
    # The arm supplies its own world, its own BC head and its own config. Reading
    # time_mixer and align_weight from its report is what keeps this from scoring one
    # arm's generated states against another arm's semantics.
    parser.add_argument("--arm-dir", type=Path, default=HERE / "v2_phase2_attention")
    # `name=path` pairs; pass the flag with no values to score drift without critics,
    # which is the case for an arm that has no Phase 3 yet.
    parser.add_argument("--critics", nargs="*", default=None)
    args = parser.parse_args()
    args.out = args.out or ROOT / "artifacts/generated_drift" / args.arm_dir.name
    args.out.mkdir(parents=True, exist_ok=True)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    trained = json.loads((args.arm_dir / "training_report.json").read_text())
    saved = replace(base, transition="direct",
                    time_mixer=trained.get("time_mixer", "attention"),
                    align_weight=trained.get("align_weight", 0.0))
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    world, prior = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(args.arm_dir / "phase2_final.pt", saved, part0=world, part1=prior)
    world.eval(), encoder.eval(), prior.eval()

    wanted = (CRITICS if args.critics is None else
              {pair.split("=", 1)[0]: Path(pair.split("=", 1)[1]) for pair in args.critics})
    critics = {}
    if wanted:
        import __main__
        from run_oracle_phase3 import OracleStream
        __main__.OracleStream = OracleStream
        for name, path in wanted.items():
            head = Heads(saved).to(DEVICE)
            head.load_state_dict(torch.load(path, weights_only=False)["heads"])
            critics[name] = head.eval()
    print(f"arm {args.arm_dir.name}: time_mixer={saved.time_mixer} "
          f"align_weight={saved.align_weight}, critics={list(critics) or 'none'}", flush=True)

    rng = torch.Generator(device=DEVICE).manual_seed(2**18)
    rows, rankings = [], []
    for seed in range(args.seed_base, args.seed_base + args.episodes):
        # One root per rollout seed, reservoir-sampled from the BC's own trajectory.
        walk = torch.Generator(device=DEVICE).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**20)
        picker = np.random.default_rng(seed)
        observation, env_state = reset(seed)
        state, action, seen, chosen = None, _tokens(base.n_actions), 0, None
        for index in range(base.horizon_eval):
            patches = patchify(observation[None, None], base.patch).to(DEVICE)
            state, agent = observe(world, encoder, state, action, patches, walk, saved)
            choice = int(torch.multinomial(
                prior(agent)["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
            seen += 1
            if picker.integers(seen) == 0:
                chosen = (index, env_state, state, agent, observation)
            observation, env_state, _, terminated, truncated = env_step(
                env_state, choice, seed + index + 1)
            action = _tokens(choice)
            if terminated or truncated:
                break
        if chosen is None:
            continue
        index, env_state, real, agent, frame = chosen

        # Identical action sequence on both chains, drawn from the BC at the shared root.
        sequence, seen_state, seen_agent, local = [], real, agent, env_state
        made_state, made_agent = real.world, agent
        record = {"seed": seed, "step": index, "depths": []}
        alive = True
        for depth in range(args.depth):
            act = int(torch.multinomial(
                prior(seen_agent)["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
            sequence.append(act)
            frame, local, reward, terminated, truncated = env_step(
                local, act, seed + index + depth + 1)
            seen_state, seen_agent = observe(
                world, encoder, seen_state, _tokens(act),
                patchify(frame[None, None], base.patch).to(DEVICE), rng, saved)
            made_state, made_agent = advance(world, made_state, _tokens(act), rng, saved)

            seen_policy = prior(seen_agent)["policy"][:, -1, 0].softmax(-1)[0]
            made_policy = prior(made_agent)["policy"][:, -1, 0].softmax(-1)[0]
            kl, js = _divergence(seen_policy, made_policy)
            entry = {
                "depth": depth + 1,
                "action": act,
                "true_reward": float(reward),
                "true_continuation": 0.0 if terminated else 1.0,
                "latent_mse": float(F.mse_loss(made_state.latent, seen_state.world.latent)),
                "latent_cosine": float(F.cosine_similarity(
                    made_state.latent.flatten(), seen_state.world.latent.flatten(), dim=0)),
                "readout_cosine": float(F.cosine_similarity(
                    made_agent.flatten(), seen_agent.flatten(), dim=0)),
                "bc_kl": kl,
                "bc_js": js,
                "bc_top1_agree": float(int(seen_policy.argmax()) == int(made_policy.argmax())),
                "bc_prob_of_observed_top": float(made_policy[int(seen_policy.argmax())]),
                "bc_top_prob_observed": float(seen_policy.max()),
            }
            for name, head in critics.items():
                entry[f"v_observed_{name}"] = _value(head, seen_agent)
                entry[f"v_generated_{name}"] = _value(head, made_agent)
            record["depths"].append(entry)
            if terminated or truncated:
                alive = False
                break
        record["alive"] = alive

        # Bellman residual at the root under true outcomes, on each path.
        if record["depths"]:
            first = record["depths"][0]
            for name, head in critics.items():
                record[f"bellman_observed_{name}"] = (
                    _value(head, agent) - (first["true_reward"] + base.gamma
                                           * first["true_continuation"] * first[f"v_observed_{name}"]))
                record[f"bellman_generated_{name}"] = (
                    _value(head, agent) - (first["true_reward"] + base.gamma
                                           * first["true_continuation"] * first[f"v_generated_{name}"]))
        rows.append(record)

        # Do the 17 one-step successors keep their ordering when generated?
        if len(rankings) < args.ranking_roots:
            seen_values, made_values = {n: [] for n in critics}, {n: [] for n in critics}
            for candidate in range(base.n_actions):
                frame_c, _, _, _, _ = env_step(env_state, candidate, seed + index + 1)
                _, seen_c = observe(world, encoder, real, _tokens(candidate),
                                    patchify(frame_c[None, None], base.patch).to(DEVICE), rng, saved)
                _, made_c = advance(world, real.world, _tokens(candidate), rng, saved)
                for name, head in critics.items():
                    seen_values[name].append(_value(head, seen_c))
                    made_values[name].append(_value(head, made_c))
            entry = {"seed": seed, "step": index}
            for name in critics:
                a, b = np.array(seen_values[name]), np.array(made_values[name])
                order_a, order_b = a.argsort().argsort(), b.argsort().argsort()
                entry[f"spearman_{name}"] = float(np.corrcoef(order_a, order_b)[0, 1])
                entry[f"top1_{name}"] = float(int(a.argmax()) == int(b.argmax()))
                entry[f"spread_observed_{name}"] = float(a.max() - a.min())
                entry[f"spread_generated_{name}"] = float(b.max() - b.min())
            rankings.append(entry)
        print(f"  seed {seed} root step {index}", flush=True)

    (args.out / "generated_drift.json").write_text(json.dumps(
        {"episodes": args.episodes, "depth": args.depth, "roots": len(rows),
         "arm_dir": str(args.arm_dir), "checkpoint": str(args.arm_dir / "phase2_final.pt"),
         "time_mixer": saved.time_mixer, "align_weight": saved.align_weight,
         "direct_rollout": saved.direct_rollout, "seed_base": args.seed_base,
         "critics": {name: str(path) for name, path in wanted.items()},
         "rows": rows, "rankings": rankings}, indent=2, default=float))

    print(f"\n{len(rows)} roots, one per DEV seed, identical action sequence on both chains")
    print(f"{'depth':<7}{'n':>4}{'latent MSE':>12}{'latent cos':>12}{'readout cos':>13}"
          f"{'BC KL':>9}{'BC JS':>9}{'top1 agree':>12}{'p(obs top)':>12}")
    for depth in range(1, args.depth + 1):
        at = [e for r in rows for e in r["depths"] if e["depth"] == depth]
        if not at:
            continue
        m = lambda k: float(np.mean([e[k] for e in at]))
        print(f"{depth:<7}{len(at):>4}{m('latent_mse'):>12.5f}{m('latent_cosine'):>12.4f}"
              f"{m('readout_cosine'):>13.4f}{m('bc_kl'):>9.4f}{m('bc_js'):>9.4f}"
              f"{m('bc_top1_agree'):>12.3f}{m('bc_prob_of_observed_top'):>12.4f}")
    print(f"   (BC's own top-action probability on the observed path: "
          f"{np.mean([e['bc_top_prob_observed'] for r in rows for e in r['depths']]):.4f})")

    print(f"\n{'critic':<11}{'depth':>6}{'V observed':>12}{'V generated':>13}{'drift':>10}"
          f"{'Bellman obs':>13}{'Bellman gen':>13}")
    for name in critics:
        for depth in range(1, args.depth + 1):
            at = [e for r in rows for e in r["depths"] if e["depth"] == depth]
            if not at:
                continue
            o = float(np.mean([e[f"v_observed_{name}"] for e in at]))
            g = float(np.mean([e[f"v_generated_{name}"] for e in at]))
            bo = float(np.mean([r[f"bellman_observed_{name}"] for r in rows
                                if f"bellman_observed_{name}" in r])) if depth == 1 else float("nan")
            bg = float(np.mean([r[f"bellman_generated_{name}"] for r in rows
                                if f"bellman_generated_{name}" in r])) if depth == 1 else float("nan")
            print(f"{name:<11}{depth:>6}{o:>12.4f}{g:>13.4f}{g - o:>10.4f}"
                  f"{bo:>13.4f}{bg:>13.4f}")

    if rankings:
        print(f"\n17-action successor ranking on {len(rankings)} roots "
              f"(observed vs generated, same critic)")
        print(f"{'critic':<11}{'spearman':>11}{'top-1 agree':>13}"
              f"{'spread obs':>12}{'spread gen':>12}")
        for name in critics:
            f = lambda k: float(np.mean([e[f"{k}_{name}"] for e in rankings]))
            print(f"{name:<11}{f('spearman'):>11.4f}{f('top1'):>13.3f}"
                  f"{f('spread_observed'):>12.4f}{f('spread_generated'):>12.4f}")


if __name__ == "__main__":
    main()
