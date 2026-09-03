"""Is SLEEP actually wrong under the actor's own policy, at the actor's own states?

Every earlier sleep measurement compared the wrong things. Branches followed the frozen
BC while the critic estimates returns under its actor, so it read Q_BC against V_actor;
the states were the BC's, not the actor's; and the reward column was read from the
pre-action state, which under the led-to convention (S22) is the reward of the action
that arrived, not the one being judged.

This samples awake, energy-low, not-yet-woken states from the actor's *own* deployed
trajectories -- including the ones where it refuses to sleep -- forks SLEEP against the
action it would actually take, continues both branches under that same actor with common
simulator randomness, and compares the empirical return with the aligned estimate
Q(s,a) = r(s') + gamma * c(s') * V(s'), formed from the observed and the generated
successor. One state per rollout seed, so episodes are independent.
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

from craftax.craftax_classic.constants import Achievement, Action

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
SLEEP = Action.SLEEP.value
WAKE_UP = [a.name for a in Achievement].index("WAKE_UP")
DRAWS = 4000


def _tokens(action: int) -> torch.Tensor:
    return torch.full((1, 1), action, dtype=torch.long, device=DEVICE)


def _quantities(heads: Heads, agent: torch.Tensor, config: Config) -> tuple[float, float, float]:
    """(r, c, V) read at the successor -- the led-to convention puts the action's own
    reward at lead 0 of the block it was committed into."""
    readout = heads(agent)
    return (float(_expect(readout["reward"][:, -1, 0], heads.centers)),
            float(readout["continuation"][:, -1, 0].sigmoid()),
            float(_expect(readout["value"][:, -1], heads.centers)))


@torch.no_grad()
def _opportunity(world, encoder, heads, seed, config):
    """One uniformly drawn awake / energy<9 / not-yet-woken state from this policy's own
    deployed trajectory, by reservoir sampling so no part of the episode is favoured."""
    rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**20)
    picker = np.random.default_rng(seed)
    observation, env_state = reset(seed)
    state, action, seen, chosen = None, _tokens(config.n_actions), 0, None
    tally = {"opportunities": 0, "sleep_actions": 0}
    for index in range(config.horizon_eval):
        patches = patchify(observation[None, None], config.patch).to(DEVICE)
        state, agent = observe(world, encoder, state, action, patches, rng, config)
        choice = int(torch.multinomial(
            heads(agent)["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
        if (not bool(env_state.is_sleeping) and int(env_state.player_energy) < 9
                and not bool(np.asarray(env_state.achievements)[WAKE_UP])):
            seen += 1
            tally["opportunities"] += 1
            tally["sleep_actions"] += choice == SLEEP
            if picker.integers(seen) == 0:
                chosen = (index, env_state, state, agent, choice)
        observation, env_state, _, terminated, truncated = env_step(
            env_state, choice, seed + index + 1)
        action = _tokens(choice)
        if terminated or truncated:
            break
    return chosen, tally


@torch.no_grad()
def _branch(world, encoder, heads, seed, index, env_state, state, action, rng, config, steps):
    """Execute `action`, then continue under `heads`. Common random numbers: both
    branches from a state share the policy stream and the simulator's seed schedule."""
    policy_rng = torch.Generator(device=DEVICE).manual_seed(seed * 7919 + index)
    observation, local, reward, terminated, truncated = env_step(
        env_state, action, seed + index + 1)
    total, discount, woke = reward, config.gamma, False
    carried = observe(world, encoder, state, _tokens(action),
                      patchify(observation[None, None], config.patch).to(DEVICE), rng, config)
    for offset in range(steps):
        if terminated or truncated:
            break
        choice = int(torch.multinomial(
            heads(carried[1])["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
        observation, local, reward, terminated, truncated = env_step(
            local, choice, seed + index + offset + 2)
        total += discount * reward
        discount *= config.gamma
        carried = observe(world, encoder, carried[0], _tokens(choice),
                          patchify(observation[None, None], config.patch).to(DEVICE), rng, config)
    return total, bool(np.asarray(local.achievements)[WAKE_UP]), bool(terminated)


def _interval(values: np.ndarray, cluster: np.ndarray) -> tuple[float, float, float]:
    keys = np.unique(cluster)
    per = np.array([values[cluster == key].mean() for key in keys])
    picker = np.random.default_rng(2**20)
    samples = per[picker.integers(0, len(per), (DRAWS, len(per)))].mean(1)
    return float(per.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--tag", default="_10k")
    # The oracle actor is stored as a bare state dict inside its progress file, not as a
    # d4mj checkpoint, so it is loaded by path rather than through the arm layout.
    parser.add_argument("--heads", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--branch-steps", type=int, default=300)
    args = parser.parse_args()

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    config = replace(saved, horizon=saved.direct_rollout)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    world = World(saved).to(DEVICE)
    load(HERE / f"v2_phase2_{args.arm}" / "phase2_final.pt", saved,
         part0=world, part1=Heads(saved).to(DEVICE))
    actor = Heads(saved).to(DEVICE)
    if args.heads is not None:
        # The progress file pickles its streams as `__main__.OracleStream`, so that name
        # has to resolve here before the payload will unpickle.
        import __main__
        sys.path.insert(0, str(ROOT / "artifacts"))
        from run_oracle_phase3 import OracleStream
        __main__.OracleStream = OracleStream
        stored_actor = torch.load(args.heads, weights_only=False)
        actor.load_state_dict(stored_actor["heads"])
        print(f"actor from {args.heads} at step {stored_actor['step']}", flush=True)
    else:
        load(HERE / f"v2_phase3_{args.arm}{args.tag}" / "phase3_final.pt", config,
             part0=World(saved).to(DEVICE), part1=actor)
    world.eval(), encoder.eval(), actor.eval()

    rng = torch.Generator(device=DEVICE).manual_seed(2**19)
    rows, totals = [], {"opportunities": 0, "sleep_actions": 0}
    for seed in range(args.seed_base, args.seed_base + args.episodes):
        chosen, tally = _opportunity(world, encoder, actor, seed, config)
        for key in totals:
            totals[key] += tally[key]
        if chosen is None:
            continue
        index, env_state, state, agent, sampled = chosen
        logits = actor(agent)["policy"][:, -1, 0]
        preferred = int(logits.argmax())
        if preferred == SLEEP:  # nothing to fork against
            continue
        record = {"seed": seed, "step": index, "sampled_action": Action(sampled).name,
                  "preferred_action": Action(preferred).name,
                  "probability_sleep": float(logits.softmax(-1)[0, SLEEP])}
        for name, action in (("sleep", SLEEP), ("preferred", preferred)):
            observation, _, _, _, _ = env_step(env_state, action, seed + index + 1)
            seen, seen_agent = observe(
                world, encoder, state, _tokens(action),
                patchify(observation[None, None], config.patch).to(DEVICE), rng, config)
            made, made_agent = advance(world, state.world, _tokens(action), rng, config)
            for kind, token in (("observed", seen_agent), ("generated", made_agent)):
                reward, keep, value = _quantities(actor, token, config)
                record[f"q_{kind}_{name}"] = reward + config.gamma * keep * value
                record[f"value_{kind}_{name}"] = value
                record[f"reward_{kind}_{name}"] = reward
                record[f"continuation_{kind}_{name}"] = keep
            total, woke, died = _branch(world, encoder, actor, seed, index, env_state,
                                        state, action, rng, config, args.branch_steps)
            record[f"return_{name}"] = total
            record[f"woke_{name}"] = woke
            record[f"terminated_{name}"] = died
        rows.append(record)
        print(f"  seed {seed} step {index} prefers {record['preferred_action']}", flush=True)

    out = args.out or HERE / f"v2_phase3_{args.arm}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "actor_sleep_value.json").write_text(json.dumps(
        {"arm": args.arm, "tag": args.tag, "episodes": args.episodes,
         "branch_steps": args.branch_steps, "deployed": totals,
         "states": len(rows), "rows": rows}, indent=2, default=float))

    print(f"\n{args.arm}{args.tag}: {len(rows)} states, one per rollout seed, "
          f"{args.branch_steps}-step branches under the actor itself")
    print(f"  deployed opportunities {totals['opportunities']}, "
          f"SLEEP taken {totals['sleep_actions']}")
    if not rows:
        return
    cluster = np.array([row["seed"] for row in rows])
    column = lambda key: np.array([row[key] for row in rows])
    show = lambda label, values: print(
        f"  {label:<34}{values[0]:>+9.4f} [{values[1]:+.4f},{values[2]:+.4f}]")
    show("empirical return SLEEP", _interval(column("return_sleep"), cluster))
    show("empirical return preferred", _interval(column("return_preferred"), cluster))
    show("empirical SLEEP - preferred",
         _interval(column("return_sleep") - column("return_preferred"), cluster))
    for kind in ("observed", "generated"):
        show(f"Q {kind} SLEEP - preferred",
             _interval(column(f"q_{kind}_sleep") - column(f"q_{kind}_preferred"), cluster))
    print(f"  P(sleep) under actor {column('probability_sleep').mean():.5f}   "
          f"WAKE_UP after SLEEP {column('woke_sleep').mean():.3f} "
          f"vs preferred {column('woke_preferred').mean():.3f}")
    print(f"  terminated: SLEEP {column('terminated_sleep').mean():.3f} "
          f"preferred {column('terminated_preferred').mean():.3f}")


if __name__ == "__main__":
    main()
