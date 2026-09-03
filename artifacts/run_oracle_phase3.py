"""Stage-0 positive control: train Phase 3 entirely on real Craftax transitions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from d4mj.actor_critic import actor_loss, critic_loss, lambda_returns
from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.execution import evaluate, run_episode
from d4mj.imagination import Trajectory, imagine
from d4mj.representation import Encoder
from d4mj.state import RealState
from d4mj.train import _balance, _update, optimizer
from d4mj.transition import World, observe


ROOT = Path(__file__).resolve().parent.parent
VERSION = "oracle-phase3-v3"


@dataclass
class OracleStream:
    slot: int
    generation: int
    seed: int
    index: int
    observation: Tensor
    env_state: object
    state: RealState
    agent: Tensor


def _expect(logits: Tensor, centers: Tensor) -> Tensor:
    mean = (logits.softmax(-1) * centers).sum(-1)
    return mean.sign() * torch.expm1(mean.abs())


@torch.no_grad()
def _reset_stream(
    slot: int,
    generation: int,
    world: World,
    encoder: Encoder,
    rng: torch.Generator,
    config: Config,
    seed_base: int,
) -> OracleStream:
    seed = seed_base + generation * config.actor_batch + slot
    observation, env_state = reset(seed)
    incoming = torch.full(
        (1, 1), config.n_actions, dtype=torch.long, device=config.device
    )
    patches = patchify(observation[None, None], config.patch).to(config.device)
    state, agent = observe(world, encoder, None, incoming, patches, rng, config)
    return OracleStream(slot, generation, seed, 0, observation, env_state, state, agent)


@torch.no_grad()
def _step_stream(
    stream: OracleStream,
    action: int,
    world: World,
    encoder: Encoder,
    rng: torch.Generator,
    config: Config,
    seed_base: int,
) -> tuple[OracleStream, float, float, bool]:
    observation, env_state, reward, terminated, truncated = env_step(
        stream.env_state, action, stream.seed + stream.index + 1
    )
    done = terminated or truncated
    if done:
        next_stream = _reset_stream(
            stream.slot,
            stream.generation + 1,
            world,
            encoder,
            rng,
            config,
            seed_base,
        )
    else:
        incoming = torch.tensor([[action]], dtype=torch.long, device=config.device)
        patches = patchify(observation[None, None], config.patch).to(config.device)
        state, agent = observe(
            world, encoder, stream.state, incoming, patches, rng, config
        )
        next_stream = OracleStream(
            stream.slot,
            stream.generation,
            stream.seed,
            stream.index + 1,
            observation,
            env_state,
            state,
            agent,
        )
    return next_stream, reward, 0.0 if terminated else 1.0, done


def oracle_trajectory(
    streams: list[OracleStream],
    world: World,
    encoder: Encoder,
    heads: Heads,
    rng: torch.Generator,
    policy_rng: torch.Generator,
    config: Config,
    seed_base: int,
) -> tuple[Trajectory, list[OracleStream], dict[str, float]]:
    """Roll PMPO through observed simulator successors and simulator outcomes."""
    agents = [torch.cat([stream.agent for stream in streams], dim=0).detach()]
    actions, logits, rewards, continuations = [], [], [], []
    resets = 0

    for _ in range(config.horizon):
        readout = heads(agents[-1])
        policy = readout["policy"][:, -1, 0]
        action = torch.multinomial(
            policy.softmax(-1), 1, generator=policy_rng
        ).squeeze(-1)
        actions.append(action)
        logits.append(policy)

        next_streams, step_rewards, step_continuations = [], [], []
        for stream, choice in zip(streams, action.tolist(), strict=True):
            next_stream, reward, continuation, done = _step_stream(
                stream, choice, world, encoder, rng, config, seed_base
            )
            next_streams.append(next_stream)
            step_rewards.append(reward)
            step_continuations.append(continuation)
            resets += int(done)
        streams = next_streams
        rewards.append(torch.tensor(step_rewards, device=config.device))
        continuations.append(torch.tensor(step_continuations, device=config.device))
        agents.append(torch.cat([stream.agent for stream in streams], dim=0).detach())

    joined = torch.cat(agents, dim=1)
    values = _expect(heads(joined)["value"], heads.centers)
    trajectory = Trajectory(
        action=torch.stack(actions, dim=1),
        logits=torch.stack(logits, dim=1),
        reward=torch.stack(rewards, dim=1),
        continuation=torch.stack(continuations, dim=1),
        value=values,
        agent=joined,
    )
    return trajectory, streams, {
        "reward": float(trajectory.reward.mean()),
        "nonzero_reward": float((trajectory.reward != 0).float().mean()),
        "terminal": resets / (config.actor_batch * config.horizon),
    }


def learned_trajectory(
    streams: list[OracleStream],
    world: World,
    heads: Heads,
    rng: torch.Generator,
    policy_rng: torch.Generator,
    config: Config,
) -> tuple[Trajectory, list[OracleStream], dict[str, float]]:
    """The matched-context control: the oracle's persistent BC contexts, but production
    `imagine` supplies the trajectory -- Direct successors with predicted reward and
    continuation. Holding the contexts fixed and swapping only the trajectory is what
    separates the offline context sampler from the modelled path.

    Streams carry different `step` values, so they cannot share one batched WorldState;
    each is imagined on its own and the rows are stacked. `terminal` here is predicted
    termination mass, not a count of real resets -- imagination has no simulator to end.
    """
    rolled = [
        imagine(world, heads, stream.state.world, stream.agent, rng, policy_rng, config)
        for stream in streams
    ]
    trajectory = Trajectory(
        **{
            field: torch.cat([getattr(row, field) for row in rolled], dim=0)
            for field in ("action", "logits", "reward", "continuation", "value", "agent")
        }
    )
    return trajectory, streams, {
        "reward": float(trajectory.reward.mean()),
        "nonzero_reward": float((trajectory.reward != 0).float().mean()),
        "terminal": float((1.0 - trajectory.continuation).mean()),
    }


@torch.no_grad()
def advance_contexts(
    streams: list[OracleStream],
    world: World,
    encoder: Encoder,
    prior: Heads,
    rng: torch.Generator,
    policy_rng: torch.Generator,
    config: Config,
    seed_base: int,
) -> tuple[list[OracleStream], float]:
    """Move the persistent starting contexts one step under the frozen BC prior."""
    agents = torch.cat([stream.agent for stream in streams], dim=0)
    logits = prior(agents)["policy"][:, -1, 0]
    actions = torch.multinomial(
        logits.softmax(-1), 1, generator=policy_rng
    ).squeeze(-1)
    advanced, resets = [], 0
    for stream, choice in zip(streams, actions.tolist(), strict=True):
        successor, _, _, done = _step_stream(
            stream, choice, world, encoder, rng, config, seed_base
        )
        advanced.append(successor)
        resets += int(done)
    return advanced, resets / config.actor_batch


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_progress(
    path: Path,
    step: int,
    heads: Heads,
    optimiser,
    balance: dict[str, float],
    streams: list[OracleStream],
    rng: torch.Generator,
    policy_rng: torch.Generator,
    context_policy_rng: torch.Generator,
    contract: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": VERSION,
            "contract": contract,
            "step": step,
            "heads": heads.state_dict(),
            "optimiser": optimiser.state_dict(),
            "balance": balance,
            "streams": streams,
            "rng": rng.get_state(),
            "policy_rng": policy_rng.get_state(),
            "context_policy_rng": context_policy_rng.get_state(),
        },
        temporary,
    )
    temporary.replace(path)


def _load_progress(
    path: Path,
    heads: Heads,
    optimiser,
    rng: torch.Generator,
    policy_rng: torch.Generator,
    context_policy_rng: torch.Generator,
    contract: dict,
) -> tuple[int, dict[str, float], list[OracleStream] | None]:
    if not path.exists():
        return 0, {}, None
    payload = torch.load(path, weights_only=False)
    if payload.get("version") != VERSION or payload.get("contract") != contract:
        raise ValueError("oracle Phase-3 resume contract changed")
    heads.load_state_dict(payload["heads"])
    optimiser.load_state_dict(payload["optimiser"])
    rng.set_state(payload["rng"])
    policy_rng.set_state(payload["policy_rng"])
    context_policy_rng.set_state(payload["context_policy_rng"])
    return int(payload["step"]), dict(payload["balance"]), payload["streams"]


def _report_evaluation(scores: dict) -> tuple[dict, dict]:
    summary = {
        name: {key: value for key, value in row.items() if key != "episodes"}
        for name, row in scores.items()
    }
    episodes = {
        name: [vars(result) for result in row["episodes"]]
        for name, row in scores.items()
    }
    return summary, episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1a", type=Path, default=Path("artifacts/stage_a_terminalfix/phase1a.pt")
    )
    parser.add_argument(
        "--phase2",
        type=Path,
        default=Path("artifacts/stage_a_s76_paired/direct-attention.2.pt"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/oracle_phase3_h2_bccontexts")
    )
    parser.add_argument("--steps", type=int, default=2500)
    # The pinned oracle checkpoints are 32-slot. `Config.n_latents` moved to 64 with the
    # v2 tokenizer, so the width has to be stated rather than inherited from the default.
    parser.add_argument("--latents", type=int, default=32)
    # The v2 encoder was saved under its own batch and seed, which are part of the
    # checkpoint's config and so have to be restored to load it.
    parser.add_argument("--encoder-report", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--trajectory", choices=("oracle", "learned"), default="oracle")
    parser.add_argument("--train-seed-base", type=int, default=20_000)
    parser.add_argument("--eval-seed-base", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=10_000)
    args = parser.parse_args()
    if min(args.steps, args.horizon, args.eval_episodes, args.eval_limit) < 1:
        parser.error("steps, horizon and evaluation sizes must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - started:8.1f}s] {message}"
        print(line, flush=True)
        with (args.out / "run.log").open("a") as stream:
            stream.write(line + "\n")

    base = replace(Config(), n_latents=args.latents)
    checkpoint_config = replace(base, transition="direct", time_mixer="attention")
    config = replace(checkpoint_config, horizon=args.horizon)
    if config.device != "cuda":
        raise RuntimeError("Stage 0 is declared on the CUDA training device")

    encoder = Encoder(base).to(config.device)
    world = World(checkpoint_config).to(config.device)
    heads = Heads(checkpoint_config).to(config.device)
    encoder_config = base
    if args.encoder_report is not None:
        saved = json.loads(args.encoder_report.read_text())
        encoder_config = replace(base, batch=saved["batch"], seed=saved["seed"])
    load(args.phase1a, encoder_config, part0=encoder)
    load(args.phase2, checkpoint_config, part0=world, part1=heads)
    encoder.eval()
    world.eval()
    for module in (encoder, world):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    prior = copy.deepcopy(heads).eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.actor_parameters():
        parameter.requires_grad_(True)
    optimiser = optimizer([heads], config)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 1003)
    policy_rng = torch.Generator(device=config.device).manual_seed(config.seed + 2**20)
    context_policy_rng = torch.Generator(device=config.device).manual_seed(
        config.seed + 2**19
    )
    contract = {
        "version": VERSION,
        "phase1a": _file_digest(args.phase1a),
        "phase2": _file_digest(args.phase2),
        "implementation": _file_digest(Path(__file__)),
        "steps": args.steps,
        "horizon": args.horizon,
        "actor_batch": config.actor_batch,
        "train_seed_base": args.train_seed_base,
        "trajectory": args.trajectory,
        "oracle": (
            "simulator successor pixels plus simulator reward and continuation"
            if args.trajectory == "oracle"
            else "learned Direct successors with predicted reward and continuation"
        ),
        "starting_contexts": "persistent frozen-BC streams; actor branches discarded",
        "world_advance_calls": (
            0 if args.trajectory == "oracle" else args.horizon * config.actor_batch
        ),
        "evaluation_gate": "paired mean achievement count; geometric score co-reported",
    }
    progress = args.out / "progress.pt"
    resume, balance, streams = _load_progress(
        progress, heads, optimiser, rng, policy_rng, context_policy_rng, contract
    )
    if streams is None:
        streams = [
            _reset_stream(
                slot, 0, world, encoder, rng, config, args.train_seed_base
            )
            for slot in range(config.actor_batch)
        ]
    log(
        f"oracle Phase 3 [{args.trajectory}]: direct-attention, h={config.horizon}, "
        f"batch={config.actor_batch}, steps={args.steps}, resume={resume}"
    )

    totals = {"reward": 0.0, "nonzero_reward": 0.0, "terminal": 0.0}
    for step_index in range(resume, args.steps):
        trajectory, _, metrics = (
            oracle_trajectory(
                streams, world, encoder, heads, rng, policy_rng, config, args.train_seed_base
            )
            if args.trajectory == "oracle"
            else learned_trajectory(streams, world, heads, rng, policy_rng, config)
        )
        returns = lambda_returns(trajectory, config)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        losses = {
            "actor": actor_loss(trajectory, returns, reference, config),
            "critic": critic_loss(
                heads(trajectory.agent[:, :-1])["value"], returns, heads.centers
            ),
        }
        loss = _balance(losses, balance, config)
        _update(optimiser, loss, [heads], config, step_index)
        streams, context_reset = advance_contexts(
            streams,
            world,
            encoder,
            prior,
            rng,
            context_policy_rng,
            config,
            args.train_seed_base,
        )
        metrics["context_reset"] = context_reset
        if "context_reset" not in totals:
            totals["context_reset"] = 0.0
        for name in totals:
            totals[name] += metrics[name]

        completed = step_index + 1
        if completed % 100 == 0 or completed == args.steps:
            scale = 100 if completed % 100 == 0 else completed % 100
            log(
                f"train {completed}/{args.steps} "
                f"reward={totals['reward'] / scale:.5f} "
                f"nonzero={totals['nonzero_reward'] / scale:.4f} "
                f"terminal={totals['terminal'] / scale:.4f} "
                f"context_reset={totals['context_reset'] / scale:.4f} "
                f"actor={float(losses['actor'].detach()):.4f} "
                f"critic={float(losses['critic'].detach()):.4f}"
            )
            totals = {name: 0.0 for name in totals}
        if completed % config.checkpoint_every == 0 or completed == args.steps:
            _save_progress(
                progress,
                completed,
                heads,
                optimiser,
                balance,
                streams,
                rng,
                policy_rng,
                context_policy_rng,
                contract,
            )

    heads.eval()
    seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes))
    log(
        f"paired DEV evaluation: {len(seeds)} seeds, native cap={args.eval_limit}"
    )
    scores = evaluate(
        {
            "actor": lambda seed: run_episode(
                world, encoder, heads, seed, config, limit=args.eval_limit
            ),
            "bc": lambda seed: run_episode(
                world, encoder, prior, seed, config, limit=args.eval_limit
            ),
        },
        seeds,
        config,
    )
    evaluation, episodes = _report_evaluation(scores)
    comparison = scores["actor"]["versus_bc"]
    passed = bool(comparison["achievements_beats"])
    report = {
        "contract": contract
        | {
            "eval_seed_base": args.eval_seed_base,
            "eval_episodes": args.eval_episodes,
            "eval_limit": args.eval_limit,
            "split": "fresh DEV positive-control seeds; FINAL untouched",
        },
        "evaluation": evaluation,
        "episodes": episodes,
        "gate": {
            "criterion": "paired 95% lower bound of mean-achievement actor-minus-BC > 0",
            "passed": passed,
            "achievements_gap": comparison["achievements_gap"],
            "achievements_interval": comparison["achievements_interval"],
            "geometric_score_gap": comparison["gap"],
            "geometric_score_interval": comparison["interval"],
            "decision": (
                "oracle_phase3_can_improve_bc"
                if passed
                else "oracle_phase3_did_not_beat_bc"
            ),
        },
    }
    temporary = args.out / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, default=float) + "\n")
    temporary.replace(args.out / "report.json")
    log(
        f"gate pass={passed} achievements actor={scores['actor']['achievements']:.3f} "
        f"bc={scores['bc']['achievements']:.3f} "
        f"gap={comparison['achievements_gap']:+.3f} "
        f"CI={tuple(round(value, 3) for value in comparison['achievements_interval'])} | "
        f"geometric actor={scores['actor']['score']:.3f} bc={scores['bc']['score']:.3f} "
        f"gap={comparison['gap']:+.3f} "
        f"CI={tuple(round(value, 3) for value in comparison['interval'])}"
    )


if __name__ == "__main__":
    main()
