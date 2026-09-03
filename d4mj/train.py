import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .actor_critic import actor_loss, critic_loss, lambda_returns
from .agent import Heads, _distribution_loss, head_loss, head_targets, paired_terminal_loss
from .config import Config
from .data import (
    FORMAT,
    STORE_FORMAT,
    Batch,
    Episode,
    EpisodeCorpus,
    atomic_manifest,
    load_episodes,
    patchify,
    sample_batch,
    sample_terminal_batch,
    save_episode_shard,
)
from .imagination import imagine
from .checkpoint import load, save
from .representation import Decoder, Encoder, pack, reconstruction_loss
from .sources import source_digests
from .state import WorldState
from .transition import advance, World, commit_inputs, transition_loss


def optimizer(modules: list[nn.Module], config: Config) -> torch.optim.AdamW:
    """The only place parameter groups are built, and the only place Mamba-2's
    no-weight-decay contract is honoured.

    Upstream marks A_log, D and dt_bias exempt. Decaying them is an asymmetry
    applied to exactly one arm of the single comparison the project exists to
    make, which is why it cannot live at a call site.
    """
    decayed, exempt = [], []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                (exempt if getattr(parameter, "_no_weight_decay", False) else decayed).append(parameter)
    groups = [{"params": decayed, "weight_decay": config.weight_decay}]
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def train_representation(
    episodes: list[Episode], steps: int, config: Config, checkpoint=None
) -> tuple[Encoder, Decoder, list[Episode]]:
    """Phase 1A, returning the frozen encoder, its decoder, and the latent cache.

    `batch.scored` decides which blocks the loss uses, per row: a block counts once
    it holds a full receptive field, and unconditionally in a window that starts at
    the episode start, where nothing earlier is missing. The cache is written under
    the same bounded window -- scanning whole episodes unbounded would give the same
    frame a different latent than deployment does.
    """
    import lpips

    device = config.device
    torch.manual_seed(config.seed)
    encoder, decoder = Encoder(config).to(device), Decoder(config).to(device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    optimiser = optimizer([encoder, decoder], config)
    sampler, rng = _generators(config, 0)
    balance: dict[str, float] = {}
    bundle, streams = [encoder, decoder, optimiser], {"sampler": sampler, "model": rng}
    resume = _checkpoint(checkpoint, config, bundle, balance, streams, contract=f"1A:{steps}")

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps), device)
        z, _, masked = encoder(batch.patches, p_mask=config.mae_p_max, rng=rng)
        predicted, _ = decoder(z)
        losses = reconstruction_loss(
            predicted, batch.patches, masked, batch.scored, perceptual, config
        )
        weights = {"lpips": config.lpips_weight}
        loss = _balance(losses, balance, config, weights)
        _update(optimiser, loss, [encoder, decoder], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, bundle, balance, streams, step + 1, f"1A:{steps}")

    encoder.eval()
    return encoder, decoder.eval(), cache_latents(encoder, episodes, config)


@torch.no_grad()
def _cache_episode(
    encoder: Encoder, episode: Episode, config: Config, digest: str
) -> Episode:
    if episode.observations is None:
        raise ValueError("raw observations are required to build a latent cache")
    frames = patchify(episode.observations[None], config.patch).to(config.device)
    latents, memory = [], None
    for start in range(0, frames.shape[1], config.sequence_long):
        chunk = frames[:, start : start + config.sequence_long]
        z, memory, _ = encoder(chunk, memory, offset=start)
        latents.append(z)
    packed = pack(torch.cat(latents, dim=1), config)[0].cpu()
    return replace(episode, latents=packed, latent_digest=digest)


@torch.no_grad()
def cache_latents(
    encoder: Encoder,
    episodes: list[Episode] | EpisodeCorpus,
    config: Config,
) -> list[Episode] | EpisodeCorpus:
    """Scan each episode once under the declared window, at mask probability zero.

    Chunked with memory carried, not one dense call: an episode runs to thousands
    of frames and a single scan builds an attention problem that size. The windowed
    mask makes chunked and fully recurrent encoding produce the same latent, which
    `scan_step_parity` asserts, so the cheap path is also the faithful one.
    """
    digest = _cache_digest(encoder, config)
    cached = [_cache_episode(encoder, episode, config, digest) for episode in episodes]
    if isinstance(episodes, EpisodeCorpus):
        return EpisodeCorpus(cached, source=episodes.source)
    return cached


@torch.no_grad()
def cache_latents_to_store(
    encoder: Encoder,
    episodes: list[Episode] | EpisodeCorpus,
    config: Config,
    out: Path,
    *,
    source_contract: dict,
    shard_episodes: int = 32,
) -> EpisodeCorpus:
    """Write a resumable mmap latent cache without retaining raw observations."""
    if shard_episodes < 1:
        raise ValueError("shard_episodes must be positive")
    digest = _cache_digest(encoder, config)
    manifest_path = out / "manifest.json"
    contract = {
        "format": STORE_FORMAT,
        "kind": "d4mj_latent_cache_v1",
        "cache_digest": digest,
        "source": source_contract,
        "planned_episodes": len(episodes),
        "shard_episodes": shard_episodes,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if {key: manifest.get(key) for key in contract} != contract:
            raise ValueError("latent-cache contract changed")
        if manifest.get("complete"):
            return load_episodes(out, digest=digest)
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest = contract | {
            "complete": False,
            "episodes": 0,
            "transitions": 0,
            "terminal_episodes": 0,
            "shards": [],
        }
        atomic_manifest(manifest_path, manifest)

    start = int(manifest["episodes"])
    if start > len(episodes):
        raise ValueError("latent cache contains more episodes than its source")
    for first in range(start, len(episodes), shard_episodes):
        last = min(first + shard_episodes, len(episodes))
        cached = []
        for episode in episodes[first:last]:
            value = _cache_episode(encoder, episode, config, digest)
            cached.append(replace(value, observations=None))
        shard_path = out / f"shard-{len(manifest['shards']):06d}.pt"
        if shard_path.exists():
            raise FileExistsError(f"unregistered latent-cache shard exists: {shard_path}")
        record = save_episode_shard(shard_path, cached)
        manifest["shards"].append(record)
        manifest["episodes"] += record["episodes"]
        manifest["transitions"] += record["transitions"]
        manifest["terminal_episodes"] += record["terminal_episodes"]
        atomic_manifest(manifest_path, manifest)
        print(f"latent cache: {manifest['episodes']}/{len(episodes)} episodes", flush=True)
    manifest["complete"] = True
    atomic_manifest(manifest_path, manifest)
    return load_episodes(out, digest=digest)


def train_dynamics(episodes: list[Episode], steps: int, config: Config, checkpoint=None) -> World:
    """Phase 1B, on the frozen cache. Agent slots are already present and masked,
    so no state shape changes at the Phase 2 boundary."""
    device = config.device
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(device)
    optimiser = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, rng = _generators(config, 1)
    streams = {"sampler": sampler, "model": rng}
    resume = _checkpoint(checkpoint, config, [world, optimiser], balance, streams, contract=f"1B:{steps}")

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps), device)
        dynamics = transition_loss(world, batch, rng, config, step=step)
        loss = _balance({"dynamics": dynamics}, balance, config)
        _update(optimiser, loss, [world], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, [world, optimiser], balance, streams, step + 1, f"1B:{steps}")
    return world


def _repeat_memory(memory, roots: int, actions: int):
    """Each root's memory repeated across its actions.

    Memory is (roots x tokens, ...) -- token-major -- so repeating dim 0 scrambles
    history without erroring. Unflatten first.
    """
    out = []
    for pair in memory:
        widened = []
        for tensor in pair:
            rest = tensor.shape[1:]
            widened.append(tensor.view(roots, -1, *rest)
                           .repeat_interleave(actions, dim=0).reshape(-1, *rest))
        out.append(tuple(widened))
    return tuple(out)


def _counterfactual_path(world: World, heads: Heads, batch: dict, rng, config: Config):
    """All 17 actions from one state, plus one further step where the branch survived.

    Dynamics on both generated latents, and reward/continuation heads on the readouts
    those generations produce -- supervising dynamics alone would leave the outcome heads
    on logged trajectories, the coverage that made death prediction fail. No policy loss:
    these actions were never chosen by a policy.
    """
    spatial, d = config.n_spatial, config.d_spatial
    z, actions = batch["history"], batch["actions"]
    n, t = z.shape[0], z.shape[1]
    committed, conditioning = commit_inputs(z.view(n, t, spatial, d), rng, config)
    features, _, memory = world(None, batch["led"], committed, conditioning)

    a = actions.shape[1]
    flat = actions.reshape(-1, 1)
    state = WorldState(z[:, -1:].view(n, 1, spatial, d).repeat_interleave(a, 0),
                       _repeat_memory(memory, n, a), t,
                       features[:, -1:].repeat_interleave(a, dim=0))
    first, agent = advance(world, state, flat, rng, config)
    dynamics = (first.latent.flatten(2)[:, 0] - batch["branch"]).pow(2).mean()

    readout = heads(agent)
    centers = heads.centers
    reward = _distribution_loss(readout["reward"][:, :, 0], batch["reward"][:, None],
                                centers).mean()
    continuation = F.binary_cross_entropy_with_logits(
        readout["continuation"][:, 0], batch["continuation"][:, None]).mean()

    keep = batch["valid"]
    if bool(keep.any()):
        noop = torch.zeros_like(flat)
        step_two, agent_two = advance(world, first, noop, rng, config)
        error = (step_two.latent.flatten(2)[:, 0] - batch["second"]).pow(2).mean(-1)
        # `_direct_loss` averages its two rollout terms; the second is scored only where
        # the first action left the branch alive, and the invalid targets are zeros that
        # must never enter the loss
        dynamics = dynamics + 0.5 * (error * keep).sum() / keep.sum()
        second_readout = heads(agent_two)
        reward = reward + 0.5 * (
            _distribution_loss(second_readout["reward"][:, :, 0],
                               batch["second_reward"][:, None], centers)[:, 0] * keep
        ).sum() / keep.sum()
        continuation = continuation + 0.5 * (
            F.binary_cross_entropy_with_logits(
                second_readout["continuation"][:, 0, 0],
                batch["second_continuation"], reduction="none") * keep
        ).sum() / keep.sum()
    return {"dynamics": dynamics, "reward": reward, "continuation": continuation}


def train_agent(
    episodes: list[Episode],
    world: World,
    steps: int,
    config: Config,
    checkpoint=None,
    world_steps: int = 0,
    terminal_dynamics_mass: float = 0.0,
    counterfactual=None,
    counterfactual_mass: float = 0.0,
) -> Heads:
    """Phase 2. The dynamics objective continues alongside the head losses, which
    keeps the world model from drifting while the heads fit it.

    Heads read the readout from the *same* diffusion-forced pass the transition
    loss uses, not a separately committed one: Dreamer 4 reuses the pretraining
    setting so the heads are fitted across the sampled signal range rather than at
    one uniform condition they will never see again.

    `world_steps` is how far Phase 1B already trained the world, so the shortcut
    bootstrap clock (S67) continues rather than restarting -- otherwise Phase 2
    would spend its first `bootstrap_start` steps back on the supervised objective.

    `terminal_dynamics_mass` is an explicit ablation. At zero, terminal tails only
    supervise continuation as before. Above zero, the same ordinary dynamics loss
    also scores those realized tail transitions; BC remains excluded.
    """
    if not 0.0 <= terminal_dynamics_mass < 1.0:
        raise ValueError("terminal_dynamics_mass must be in [0, 1)")
    if not 0.0 <= counterfactual_mass < 1.0:
        raise ValueError("counterfactual_mass must be in [0, 1)")
    if counterfactual is not None and counterfactual_mass == 0.0:
        raise ValueError("a counterfactual corpus with zero mass would never be read")
    device = config.device
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(device)
    optimiser = optimizer([world, heads], config)
    sampler, rng = _generators(config, 2)
    balance: dict[str, float] = {}
    bundle, streams = [world, heads, optimiser], {"sampler": sampler, "model": rng}
    contract = f"2:{world_steps}:{steps}"
    contract += ":continuation=paired-v2"
    if counterfactual is not None:
        contract += f":counterfactual={counterfactual_mass:.17g}"
    if terminal_dynamics_mass:
        contract += f":terminal_dynamics={terminal_dynamics_mass:.17g}"
    resume = _checkpoint(checkpoint, config, bundle, balance, streams, contract=contract)

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps, mixture=True), device)
        dynamics, agent = transition_loss(
            world, batch, rng, config, return_agent=True, step=world_steps + step
        )
        readout = heads(agent) | {"centers": heads.centers}
        losses = {"dynamics": dynamics} | head_loss(readout, head_targets(batch, config), config)

        terminal = _to(sample_terminal_batch(episodes, sampler, config, step, steps), device)
        terminal_dynamics, terminal_agent, terminal_observed = _terminal_path(
            world,
            terminal,
            rng,
            config,
            world_steps + step,
            score_dynamics=bool(terminal_dynamics_mass),
            return_observed=True,
        )
        if terminal_dynamics_mass:
            losses["dynamics"] = (
                (1.0 - terminal_dynamics_mass) * losses["dynamics"]
                + terminal_dynamics_mass * terminal_dynamics
            )
        terminal_readout = heads(terminal_agent) | {"centers": heads.centers}
        observed_readout = (
            heads(terminal_observed) | {"centers": heads.centers}
            if terminal_observed is not terminal_agent
            else terminal_readout
        )
        terminal_objective = paired_terminal_loss(
            terminal_readout, observed_readout, head_targets(terminal, config)
        )
        losses["continuation"] = (
            (1.0 - config.terminal_loss_mass) * losses["continuation"]
            + config.terminal_loss_mass
            * terminal_objective
        )
        if counterfactual is not None:
            # blended before `_balance`, the way `terminal_dynamics_mass` is: a hand-rolled
            # weighted sum would bypass the running-RMS normalisation and leave the arms
            # unmatched
            extra = _counterfactual_path(world, heads, counterfactual(step), rng, config)
            for name, value in extra.items():
                losses[name] = ((1.0 - counterfactual_mass) * losses[name]
                                + counterfactual_mass * value)
        _update(optimiser, _balance(losses, balance, config), [world, heads], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, bundle, balance, streams, step + 1, contract)
    return heads


def _terminal_path(
    world: World,
    batch: Batch,
    rng: torch.Generator,
    config: Config,
    step: int,
    *,
    score_dynamics: bool,
    return_observed: bool = False,
):
    """Evaluate a tail once; optionally admit it to the dynamics stratum."""
    routed = replace(batch, support=None) if score_dynamics else batch
    return transition_loss(
        world,
        routed,
        rng,
        config,
        return_agent=True,
        return_observed=return_observed,
        step=step,
    )


def train_actor(
    episodes: list[Episode],
    world: World,
    heads: Heads,
    steps: int,
    config: Config,
    checkpoint=None,
) -> Heads:
    """Phase 3. The world is frozen and the behaviour-cloned policy is copied and
    frozen as the prior, so the actor cannot improve by reshaping the model it is
    being scored inside. The reward and continuation body is frozen with it.

    Starting contexts use the §4.1 mixture. D4 applies it to "behavioral cloning,
    reward modeling, and reinforcement learning", so imagining only from uniformly
    drawn contexts starts RL away from the events that carry the sparse reward.

    S68 is enforced here rather than at each call site. Direct trains exactly
    `direct_rollout` generated states, so an actor imagining past that optimises against
    transitions and heads on inputs they never saw. Ten scripts were each re-deriving the
    cap by hand. It is checked here and not in `imagine`, which is a mechanism the tests
    legitimately exercise at other horizons."""
    if config.transition == "direct" and config.horizon > config.direct_rollout:
        raise ValueError(
            f"horizon {config.horizon} exceeds the {config.direct_rollout} generated "
            "states Direct trains (S68); raise direct_rollout and retrain the world first"
        )
    device = config.device
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    prior = copy.deepcopy(heads).eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.actor_parameters():
        parameter.requires_grad_(True)
    optimiser = optimizer([heads], config)
    sampler, rng = _generators(config, 3)
    policy_rng = torch.Generator(device=device).manual_seed(config.seed + 2**20)
    balance: dict[str, float] = {}
    streams = {"sampler": sampler, "model": rng, "policy": policy_rng}
    frozen = f"3:{steps}:{config.actor_batch}:{_identity(world, prior)}"
    resume = _checkpoint(checkpoint, config, [heads, optimiser], balance, streams, contract=frozen)
    sampling = replace(config, batch=config.actor_batch)

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, sampling, step, steps, mixture=True), device)
        with torch.no_grad():
            committed, conditioning = commit_inputs(batch.latents, rng, config)
            features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
        begin = WorldState(batch.latents[:, -1:], memory, batch.latents.shape[1], features[:, -1:])
        trajectory = imagine(world, heads, begin, agent[:, -1:], rng, policy_rng, config)

        returns = lambda_returns(trajectory, config)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        losses = {
            "actor": actor_loss(trajectory, returns, reference, config),
            "critic": critic_loss(heads(trajectory.agent[:, :-1])["value"], returns, heads.centers),
        }
        _update(optimiser, _balance(losses, balance, config), [heads], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(
                checkpoint, config, [heads, optimiser], balance, streams, step + 1, frozen
            )
    return heads


def _checkpoint(
    path, config: Config, modules: list, balance: dict, streams: dict, step=None, contract: str = ""
) -> int:
    """Both directions of a mid-phase resume: with `step` it saves, without it
    restores and returns the step to continue from.

    Optimizer state, the running-RMS normalisers and every generator stream travel;
    restoring weights alone is not a resume. `contract` fixes what `Config` cannot:
    the planned phase length, which sets the short/long schedule, and the frozen
    world and prior a phase is trained against.
    """
    named = {f"part{index}": module for index, module in enumerate(modules)}
    if step is not None:
        save(path, config, step=step, balance=balance, contract=contract,
             generators=generator_state(**streams), **named)
        return step
    if path is None or not path.exists():
        return 0
    stored = load(path, config, balance=balance, **named)["modules"]
    if stored.get("contract", "") != contract:
        raise ValueError(
            f"checkpoint contract {stored.get('contract', '')!r} does not match {contract!r}: "
            "the phase length or the frozen model it was trained against has changed"
        )
    for name, generator in streams.items():
        generator.set_state(stored["generators"][name])
    return int(stored["step"])


def _identity(*modules: nn.Module) -> str:
    """A digest of frozen modules a phase is scored against, so a resume cannot
    silently swap them. `Config` matching is not enough: two worlds with the same
    config are different learned environments."""
    import hashlib

    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def _share_initialisation(world: World, config: Config) -> World:
    """Same starting weights wherever the arms share a parameter (S45). `manual_seed`
    alone does not achieve it: the mixers consume different numbers of draws, so
    everything built after the first time layer would differ."""
    if config.time_mixer == "attention":
        return world
    torch.manual_seed(config.seed + 1)
    reference = World(replace(config, time_mixer="attention")).state_dict()
    current = world.state_dict()
    shared = {
        name: tensor
        for name, tensor in reference.items()
        if name in current and current[name].shape == tensor.shape
    }
    world.load_state_dict(shared, strict=False)
    return world


def _balance(
    losses: dict[str, torch.Tensor],
    state: dict[str, float],
    config: Config,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Running-RMS normalisation, so a coefficient is a relative weight rather than
    a scale accident. `weights` apply *after* it: folded in first they are divided
    straight back out, and 0.2 and 5.0 measure identically."""
    total = 0.0
    for name, value in losses.items():
        squared = float(value.detach().pow(2))
        state[name] = config.rms_decay * state.get(name, squared) + (1 - config.rms_decay) * squared
        scale = (weights or {}).get(name, 1.0)
        total = total + scale * value / max(state[name] ** 0.5, 1e-8)
    return total


def _cache_digest(encoder: Encoder, config: Config) -> str:
    """Identity of the latent cache: the whole latent function, not just its weights.

    Every field the encoder's forward pass depends on is included. Weights alone are
    not identity -- two encoders with identical parameters but different resolution
    and patch layout produced the same digest, and the cache from one would load
    against the other. The time mixer is excluded because the tokenizer is shared
    and always attention.
    """
    import hashlib

    shape = (
        config.patch,
        config.resolution,
        config.channels,
        config.n_patches,
        config.window,
        config.n_latents,
        config.d_bottleneck,
        config.packing,
        config.d_model_encoder,
        config.depth_encoder,
        config.n_heads_encoder,
        config.time_every,
        config.receptive_field,
        FORMAT,
    )
    weights = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        weights.update(name.encode())
        weights.update(tensor.detach().cpu().numpy().tobytes())
    visual = source_digests(replace(config, time_mixer="attention"))
    return hashlib.sha256(repr((shape, visual, weights.hexdigest())).encode()).hexdigest()[:16]




def _generators(config: Config, phase: int) -> tuple[torch.Generator, torch.Generator]:
    """A CPU generator for the sampler and a device generator for model noise, each
    seeded independently. Drawing one seed from the other fails outright when the
    device generator is CUDA, and couples two streams that should be separable."""
    sampler = torch.Generator().manual_seed(config.seed + phase)
    model = torch.Generator(device=config.device).manual_seed(config.seed + 1000 + phase)
    return sampler, model


def generator_state(**streams: torch.Generator) -> dict:
    """Every stream a phase draws from, in a form `checkpoint.save` stores.
    `torch.get_rng_state()` captures the global stream, which nothing here draws
    from -- saving it and calling that resumable would replay every window and every
    noise draw from step zero."""
    return {name: generator.get_state() for name, generator in streams.items()}


def _to(batch: Batch, device: str) -> Batch:
    moved = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in vars(batch).items()
    }
    return Batch(**moved)


def _update(optimiser, loss, modules, config: Config, step: int) -> None:
    for group in optimiser.param_groups:
        group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for module in modules for p in module.parameters()], config.grad_clip
    )
    optimiser.step()
