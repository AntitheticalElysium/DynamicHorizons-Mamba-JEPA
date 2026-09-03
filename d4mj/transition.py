import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import AGENT, REGISTER, SPATIAL, Backbone, Layout
from .config import Config
from .representation import Encoder, pack
from .state import Memory, RealState, WorldState


class World(nn.Module):
    """The action-conditioned dynamics. `forward` evaluates one block; `predict`
    turns its features into a latent and is the only thing the arms do differently.
    Direct uses an external action-token mixer over committed features (S34/S85),
    reading spatial and register slots only, which the dynamics mask keeps agent-free."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layout = Layout.dynamics(config)
        self.spatial = self.layout.span(SPATIAL)
        self.register = self.layout.span(REGISTER)
        self.agent = self.layout.span(AGENT)

        self.action_embed = nn.Embedding(config.n_actions + 1, config.d_model)
        self.agent_tokens = nn.Parameter(torch.randn(config.n_agent, config.d_model) * 0.02)
        self.project = nn.Linear(config.d_spatial, config.d_model)
        self.registers = nn.Parameter(torch.randn(config.n_register, config.d_model) * 0.02)
        self.backbone = Backbone(
            config,
            self.layout,
            "dynamics",
            config.d_model,
            config.n_heads,
            config.depth,
            config.dynamics_context,
        )

        if config.transition == "flow":
            self.signal_embed = nn.Embedding(config.n_signal_bins, config.d_model // 2)
            self.step_embed = nn.Embedding(config.n_step_bins, config.d_model // 2)
            self.readout = nn.Linear(config.d_model, config.d_spatial)
        else:
            self.condition_embed = nn.Embedding(1, config.d_model)
            width = config.n_spatial + config.n_register
            self.pool = nn.Linear(width, config.n_spatial)
            self.direct_mixer = nn.TransformerEncoderLayer(
                config.d_model,
                config.n_heads,
                4 * config.d_model,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.direct_norm = nn.LayerNorm(config.d_model)
            self.readout = nn.Linear(config.d_model, config.d_spatial)

    def condition(self, indices: Tensor) -> Tensor:
        if self.config.transition == "flow":
            halves = (self.signal_embed(indices[..., 0]), self.step_embed(indices[..., 1]))
            return torch.cat(halves, dim=-1)[:, :, None]
        return self.condition_embed(torch.zeros_like(indices[..., 0]))[:, :, None]

    def forward(
        self,
        memory: Memory | None,
        led_to_action: Tensor,
        latent: Tensor,
        conditioning: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, Memory]:
        b, t = led_to_action.shape
        blocks = torch.cat(
            [
                self.action_embed(led_to_action)[:, :, None],
                self.condition(conditioning),
                self.project(latent),
                self.registers.expand(b, t, -1, -1),
                self.agent_tokens.expand(b, t, -1, -1),
            ],
            dim=2,
        )
        out, memory = self.backbone(blocks, memory, offset)
        return out, out[:, :, self.agent], memory

    def predict(self, features: Tensor, action: Tensor | None = None) -> Tensor:
        """Flow: the clean latent of *this* block. Direct: the latent of the next
        block, given the action about to be taken. Direct's output is tanh-bounded
        here, so training and rollout share one codomain -- squashing at rollout
        only would decay a correct 0.900 to 0.431 over six recursive steps."""
        if self.config.transition == "flow":
            return self.readout(features[:, :, self.spatial])
        world = torch.cat([features[:, :, self.spatial], features[:, :, self.register]], dim=2)
        pooled = self.pool(world.transpose(2, 3)).transpose(2, 3)
        b, t, s, d = pooled.shape
        tokens = torch.cat([self.action_embed(action)[:, :, None], pooled], dim=2)
        mixed = self.direct_mixer(tokens.reshape(b * t, s + 1, d)).view(b, t, s + 1, d)
        return torch.tanh(self.readout(self.direct_norm(mixed[:, :, 1:])))


def flow_conditioning(
    rng: torch.Generator, shape: tuple[int, int], config: Config, device, bootstrap: bool = False
):
    """Sample empirical/self rows and return the positions active in the loss."""
    rows = shape[0]
    self_rows = max(0, min(rows - 1, round(config.self_fraction * rows)))
    order = torch.randperm(rows, generator=rng, device=device)
    self_index = order[:self_rows]

    step = torch.full(shape, config.step_index, device=device, dtype=torch.long)
    if self_rows:
        step[self_index] = torch.randint(
            config.n_step_bins - 1, (self_rows, shape[1]), generator=rng, device=device
        )
    rungs = 2**step
    index = (torch.rand(shape, generator=rng, device=device) * rungs).floor().long()
    conditioning = torch.stack([index * (config.k_max // rungs), step], dim=-1)
    scored = torch.ones(shape, device=device)
    if self_rows and not bootstrap:
        scored[self_index] = 0.0

    prefix_rows = int(config.commit_prefix_fraction * shape[0])
    if prefix_rows:
        available = scored.any(dim=1).nonzero().flatten()
        picked = available[
            torch.randperm(len(available), generator=rng, device=device)[:prefix_rows]
        ]
        conditioning[picked, :-1, 0] = config.tau_ctx_index
        conditioning[picked, :-1, 1] = config.step_index
        scored[picked, :-1] = 0.0
    return conditioning, scored


def signal_level(conditioning: Tensor, config: Config) -> Tensor:
    return conditioning[..., 0].float() / config.k_max


def initial(
    world: World,
    latent: Tensor,
    led_to_action: Tensor,
    rng: torch.Generator,
    config: Config,
    memory: Memory | None = None,
    offset: int = 0,
) -> tuple[WorldState, Tensor]:
    """Commit a known latent and return the state it produces.

    A rollout state cannot be advanced until its starting observation has been
    committed: the direct arm predicts from the committed block's features, and
    `advance` reads memory rather than latent, so starting from an uncommitted
    state silently predicts from nothing.
    """
    committed, conditioning = commit_inputs(latent, rng, config)
    features, agent, memory = world(memory, led_to_action, committed, conditioning, offset)
    return WorldState(latent, memory, offset + latent.shape[1], features), agent


def observe(
    world: World,
    encoder: Encoder,
    state: RealState | None,
    led_to_action: Tensor,
    patches: Tensor,
    rng: torch.Generator,
    config: Config,
) -> tuple[RealState, Tensor]:
    """The one path a real frame takes to become (e_t, z_t, m_t, h_t).

    The encoder's bounded-window memory and the dynamics memory are different
    objects with different lifetimes; carrying them in one field is how a rollout
    silently starts from zero memory.
    """
    encoder_memory = state.encoder_memory if state is not None else None
    world_memory = state.world.memory if state is not None else None
    step = state.world.step if state is not None else 0

    z, encoder_memory, _ = encoder(patches, encoder_memory, offset=step)
    latent = pack(z, config)
    world_state, agent = initial(world, latent, led_to_action, rng, config, world_memory, step)
    return RealState(encoder_memory, world_state), agent


def advance(
    world: World, state: WorldState, action: Tensor, rng: torch.Generator, config: Config
) -> tuple[WorldState, Tensor]:
    """One semantic transition: flow runs its read-only rungs then commits, direct
    predicts and commits once. Must span exactly one block. Incoming memory is
    detached to equalise the two time mixers (S55); gradient still flows through the
    accepted latent."""
    assert state.latent.shape[1] == 1, "advance steps one block; slice the state first"
    if config.transition == "flow":
        accepted = _flow_candidate(world, state, action, rng, config)
    else:
        accepted = world.predict(state.features, action)

    memory = None if state.memory is None else tuple(
        tuple(tensor.detach() for tensor in pair) for pair in state.memory
    )
    committed, conditioning = commit_inputs(accepted, rng, config)
    features, agent, memory = world(memory, action, committed, conditioning, state.step)
    return WorldState(accepted, memory, state.step + 1, features), agent


def transition_loss(
    world: World,
    batch,
    rng: torch.Generator,
    config: Config,
    return_agent: bool = False,
    return_observed: bool = False,
    step: int = 0,
):
    """Teacher-forced over the window, on real committed latents in both arms.

    Phase 2 asks for the agent readout from this same pass rather than running a
    second one: Dreamer 4 fits the heads in the pretraining setting, so they must
    see the signal range the transition loss sampled, not a uniform condition.

    `step` is the training step, which the flow arm needs: shortcut bootstrapping
    starts only at `bootstrap_start` (S67). It defaults to 0, so a caller that does
    not train -- a gate, a diagnostic -- gets the pre-bootstrap objective.
    """
    if return_observed and not return_agent:
        raise ValueError("return_observed requires return_agent")
    if config.transition == "direct":
        loss, agent, observed = _direct_loss(world, batch, rng, config)
    else:
        loss, agent = _shortcut_loss(world, batch, rng, config, step)
        observed = agent
    if return_observed:
        return loss, agent, observed
    return (loss, agent) if return_agent else loss


def commit_inputs(latent: Tensor, rng: torch.Generator, config: Config):
    """Flow commits at the signal its own conditioning bin names, so the tensor and
    its label cannot disagree; direct has no noise mechanism and commits clean."""
    shape = latent.shape[:2]
    label = torch.stack(
        [
            torch.full(shape, config.tau_ctx_index, device=latent.device),
            torch.full(shape, config.step_index, device=latent.device),
        ],
        dim=-1,
    )
    if config.transition == "direct":
        return latent, label
    signal = config.tau_ctx_signal
    noise = torch.randn(latent.shape, generator=rng, device=latent.device, dtype=latent.dtype)
    return signal * latent + (1.0 - signal) * noise, label


def _direct_loss(world: World, batch, rng: torch.Generator, config: Config):
    """Teacher forcing plus a generated-prefix rollout, after V-JEPA 2-AC (S55). Every
    generated state is committed through `advance`, and every readout replaces the real
    one at its own index. That is `direct_rollout` states, so imagination beyond it
    leaves the trained distribution and S68 caps the horizon there. The rollout terms are
    averaged, matching the source's `jloss + sloss` of two means; squared error is a
    declared deviation from its L1.

    The rollout consumes the last `direct_rollout` blocks of the row, so a row must carry
    one more block than it generates. `Config` asserts that against `sequence`, the
    shorter of the two schedules, rather than letting short batches quietly contribute
    teacher forcing alone.
    """
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
    taken = batch.led_to_action[:, 1:]
    predicted = world.predict(features[:, :-1], taken)
    teacher = (predicted - batch.latents[:, 1:]).pow(2).mean(dim=(1, 2, 3))

    length, steps = batch.latents.shape[1], config.direct_rollout
    if length < steps + 1:
        return _uniform_mean(teacher, batch), agent, agent

    start = length - steps
    prefix, _, memory = world(
        None, batch.led_to_action[:, :start], committed[:, :start], conditioning[:, :start]
    )
    state = WorldState(batch.latents[:, start - 1:start], memory, start, prefix[:, -1:])
    rollout, generated = torch.zeros_like(teacher), []
    for index in range(start, length):
        state, produced = advance(
            world, state, batch.led_to_action[:, index:index + 1], rng, config
        )
        rollout = rollout + (
            state.latent - batch.latents[:, index:index + 1]
        ).pow(2).mean(dim=(1, 2, 3))
        generated.append(produced)
    readout = torch.cat([agent[:, :start], *generated], dim=1)
    # `agent` is this same world reading the real latents at these positions, so the two
    # sides are the observed and generated readouts of one index. Detached on the
    # observed side, after Dreamer 3's stop-gradient prior/posterior alignment.
    aligned = (readout[:, start:] - agent[:, start:].detach()).pow(2).mean(dim=(1, 2, 3))
    combined = teacher + rollout / steps + config.align_mass * aligned
    return _uniform_mean(combined, batch), readout, agent


def _shortcut_loss(world: World, batch, rng: torch.Generator, config: Config, step: int = 0) -> Tensor:
    """Equation 7. At the finest step size the target is the clean latent; at every
    larger step it is the stop-gradient average of two half-steps, which is what
    teaches the step token to mean anything and what makes a four-rung sampler
    work. Without it the arm is the paper's own diffusion-forcing ablation.
    """
    target = batch.latents
    bootstrap = step >= config.bootstrap_start
    conditioning, scored = flow_conditioning(
        rng, target.shape[:2], config, target.device, bootstrap
    )
    tau = signal_level(conditioning, config)[..., None, None]
    noise = torch.randn(target.shape, generator=rng, device=target.device, dtype=target.dtype)
    corrupted = tau * target + (1.0 - tau) * noise
    features, agent, _ = world(None, batch.led_to_action, corrupted, conditioning)
    predicted = world.predict(features)

    finest = conditioning[..., 1] == config.step_index
    weight = 0.9 * tau + 0.1
    flow = (predicted - target).pow(2).mean(dim=(2, 3))

    self_loss = torch.zeros_like(flow)
    if bootstrap:
        with torch.no_grad():
            target_velocity = _bootstrap_target(world, batch, corrupted, conditioning, tau, config)
        velocity = (predicted - corrupted) / (1.0 - tau)
        self_loss = ((1.0 - tau) ** 2 * (velocity - target_velocity).pow(2)).mean(dim=(2, 3))

    combined = weight.squeeze(-1).squeeze(-1) * torch.where(finest, flow, self_loss) * scored
    per_row = combined.sum(dim=1) / scored.sum(dim=1).clamp(min=1.0)
    return _uniform_mean(per_row, batch), agent


def _uniform_mean(per_row: Tensor, batch) -> Tensor:
    """Every row during pretraining, the uniform half during finetuning, never a
    support row. D4 pretrains on the whole corpus and then restricts the continued
    dynamics loss to uniform rows "to avoid optimistic generations" (§4.1)."""
    mask = batch.rows("dynamics").to(per_row.device).float()
    return (per_row * mask).sum() / mask.sum().clamp(min=1.0)


def _bootstrap_target(world: World, batch, corrupted, conditioning, tau, config: Config) -> Tensor:
    """Two half-steps, stop-gradient, per Equation 7: b' from tau at d/2, then b''
    from the point b' reaches. Their average is what a single step of size d must
    reproduce."""
    half = torch.stack([conditioning[..., 0], (conditioning[..., 1] + 1).clamp(max=config.step_index)], -1)
    step = (2.0 ** -conditioning[..., 1].float())[..., None, None]

    first, _, _ = world(None, batch.led_to_action, corrupted, half)
    b_first = (world.predict(first) - corrupted) / (1.0 - tau)
    moved = corrupted + b_first * step / 2

    midpoint = ((tau + step / 2).squeeze(-1).squeeze(-1) * config.k_max).round().long()
    shifted = torch.stack([midpoint.clamp(max=config.k_max - 1), half[..., 1]], -1)
    second, _, _ = world(None, batch.led_to_action, moved, shifted)
    b_second = (world.predict(second) - moved) / (1.0 - tau - step / 2)
    return (b_first + b_second) / 2


def _flow_candidate(
    world: World, state: WorldState, action: Tensor, rng: torch.Generator, config: Config
) -> Tensor:
    """The shortcut ladder. At the final rung tau = 1 - d, so the Euler coefficient
    is exactly one and the iterate lands on the predicted clean latent."""
    shape = action.shape
    z = torch.randn(
        (*shape, config.n_spatial, config.d_spatial),
        generator=rng,
        device=action.device,
        dtype=state.latent.dtype,
    )
    scale, step = config.k_max // config.rungs, config.rungs.bit_length() - 1
    for rung in range(config.rungs):
        tau = rung / config.rungs
        indices = torch.stack(
            [
                torch.full(shape, rung * scale, device=z.device),
                torch.full(shape, step, device=z.device),
            ],
            dim=-1,
        )
        features, _, _ = world(state.memory, action, z, indices, state.step)
        z = z + (world.predict(features) - z) / (1.0 - tau) / config.rungs
    return z
