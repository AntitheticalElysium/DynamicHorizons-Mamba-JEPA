import time

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from .agent import Heads, head_targets
from .config import Config
from .data import Batch
from .transition import World, advance, commit_inputs, initial, transition_loss
from .world_api import ModelBundle


@torch.no_grad()
def rollout_predictions(bundle, latents, actions, context, rng=None, *, first_action=None):
    """Observed prefix followed by generated successors through the deployed API.

    Actions are outgoing: T latents have T-1 transitions. Legacy windows may
    supply their first incoming action; LeWM consumes only completed pairs.
    This is a mechanical diagnostic, not authorization for policy control.
    """
    if not 1 <= context < latents.shape[1]:
        raise ValueError("context must leave at least one successor")
    if actions.shape != (latents.shape[0], latents.shape[1] - 1):
        raise ValueError("rollout requires T latents and T-1 outgoing actions")
    state = bundle.prefill(latents[:, :context], actions[:, :context-1], rng,
                           first_action=first_action)
    predictions = []
    for step in range(context, latents.shape[1]):
        state, _ = bundle.advance(state, actions[:, step-1:step], rng)
        predictions.append(state.latent)
    return torch.cat(predictions, dim=1), state


@torch.no_grad()
def multistep_error(
    world: World,
    batch: Batch,
    rng: torch.Generator,
    config: Config,
    successors: Tensor | None = None,
    context: int | None = None,
) -> dict[str, list[float]]:
    """Per-step error under the runtime path, from a committed prefix of `context`
    real blocks -- the full dynamics context by default, since rolling from one
    block would select a horizon for a regime imagination never runs in (S54).

    Mean error alone cannot adjudicate the direct arm: the conditional-mean collapse
    minimises exactly this number. `successors` adds nearest-mode and mode-mean
    distances, which separate a predictor on a mode from one between modes.
    """
    blocks = batch.latents.shape[1]
    context = min(config.dynamics_context, blocks - 1) if context is None else context
    assert 1 <= context < blocks, f"context {context} leaves nothing to roll over {blocks} blocks"

    bundle = ModelBundle.from_models(config, None, world)
    predicted, state = rollout_predictions(bundle, batch.latents, batch.led_to_action[:, 1:],
                                          context, rng, first_action=batch.led_to_action[:, :1])
    report = {"mean_error": [float((predicted[:, i:i+1] - batch.latents[:, context+i:context+i+1])
                                  .pow(2).mean()) for i in range(blocks-context)]}

    if successors is not None:
        gap = (state.latent[:, 0, None] - successors).pow(2).flatten(2).mean(-1)
        report["nearest_mode"] = [float(gap.min(dim=1).values.mean())]
        report["mode_mean"] = [float((state.latent[:, 0] - successors.mean(1)).pow(2).mean())]
    return report


@torch.no_grad()
def latent_stats(world: World, batch: Batch, rng: torch.Generator, config: Config) -> dict[str, float]:
    """Range *and* scale, from a full committed context. A bounded readout fixes the
    range; it says nothing about contraction toward the conditional mean, which is
    the failure that looks like a working model in every one-step metric.

    The denominator is the matched next-state target, not the whole batch: sequence
    diversity there would make a sound prediction look contracted. Predicting from
    one block instead of a full context measures a regime deployment never runs in.
    """
    blocks = batch.latents.shape[1]
    context = min(config.dynamics_context, blocks - 1)
    bundle = ModelBundle.from_models(config, None, world)
    _, state = rollout_predictions(bundle, batch.latents[:, :context+1],
                                   batch.led_to_action[:, 1:context+1], context, rng,
                                   first_action=batch.led_to_action[:, :1])

    real, predicted = batch.latents[:, context : context + 1], state.latent
    return {
        "real_std": float(real.std()),
        "predicted_std": float(predicted.std()),
        "contraction": float(predicted.std() / real.std().clamp(min=1e-8)),
        "outside_unit": float((predicted.abs() > 1.0).float().mean()),
    }


@torch.no_grad()
def head_calibration(heads: Heads, agent: Tensor, batch: Batch, config: Config) -> dict[str, float]:
    """Reward and continuation at lead 0, the only lead deployment reads.

    Continuation is split by target, not reported as one mean: terminals are ~0.01%
    of transitions, so a constant "continue" head matches the global mean and looks
    calibrated. `continuation_separation` is what collapses to zero when it does,
    and `terminal_targets` says how many terminals the estimate rests on.
    """
    readout, targets = heads(agent), head_targets(batch, config)
    valid = targets["valid"][..., 0]
    # The reward head is scored on the rows its loss reads, or a support row that
    # never trained it would count against its calibration.
    rewarded = valid * targets["reward_rows"][..., 0]
    probability = readout["continuation"][..., 0].sigmoid()
    truth = targets["continuation"][..., 0]
    alive, dead = valid * truth, valid * (1 - truth)
    on_alive = float((probability * alive).sum() / alive.sum().clamp(min=1.0))
    on_dead = float((probability * dead).sum() / dead.sum().clamp(min=1.0))
    mean = (readout["reward"][..., 0, :].softmax(-1) * heads.centers).sum(-1)
    predicted = mean.sign() * torch.expm1(mean.abs())
    return {
        "reward_mae": float(
            ((predicted - targets["reward"][..., 0]).abs() * rewarded).sum() / rewarded.sum().clamp(min=1.0)
        ),
        "reward_mae_zero": float(
            (targets["reward"][..., 0].abs() * rewarded).sum() / rewarded.sum().clamp(min=1.0)
        ),
        "continuation_mean": float((probability * valid).sum() / valid.sum()),
        "continuation_target": float((truth * valid).sum() / valid.sum()),
        "continuation_on_continuing": on_alive,
        "continuation_on_terminal": on_dead,
        "continuation_separation": on_alive - on_dead if float(dead.sum()) else float("nan"),
        "terminal_targets": float(dead.sum()),
    }


def cost(modules: dict[str, nn.Module], world: World, config: Config) -> dict[str, float]:
    """Cost of the two arms: parameters, state sizes, memory and throughput.

    Caveats that change how the numbers read. `forward_backward_per_second` excludes
    the optimizer and transfer, so it is not a training step rate. `flops_per_step`
    counts dispatched aten ops, so a fused Mamba kernel is invisible to it.
    `memory_horizon_at_least` perturbs one history block with the present held
    fixed, so it measures reach, not trajectory divergence, and is a power-of-two
    lower bound. State size is read only after the cache saturates.
    """
    device, deployed = config.device, {"encoder", "world", "heads"}
    counts = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
    latent = torch.randn(1, 1, config.n_spatial, config.d_spatial, device=device)
    action = torch.zeros(1, 1, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(0)

    with torch.no_grad():
        state, _ = initial(world, latent, action, rng, config)
        for _ in range(config.dynamics_context + 4):
            state, _ = advance(world, state, action, rng, config)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(16):
            state, _ = advance(world, state, action, rng, config)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    peak = float(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        advance(world, state, action, rng, config)
    flops = counter.get_total_flops()

    probe = Batch(
        led_to_action=torch.zeros(config.batch, config.sequence, dtype=torch.long, device=device),
        reward=torch.zeros(config.batch, config.sequence, device=device),
        terminated=torch.zeros(config.batch, config.sequence, dtype=torch.bool, device=device),
        truncated=torch.zeros(config.batch, config.sequence, dtype=torch.bool, device=device),
        valid=torch.ones(config.batch, config.sequence, dtype=torch.bool, device=device),
        scored=torch.ones(config.batch, config.sequence, dtype=torch.bool, device=device),
        burn_in=0,
        latents=torch.randn(
            config.batch, config.sequence, config.n_spatial, config.d_spatial, device=device
        ).tanh(),
    )
    # A measurement must never abort a run. Mamba's Triton autotuner benchmarks
    # several backward kernels and has OOM'd here mid-experiment; the throughput
    # figure is then reported as absent rather than taking the training with it.
    frozen = [p for p in world.parameters() if not p.requires_grad]
    for parameter in frozen:
        parameter.requires_grad_(True)
    train_elapsed = float("nan")
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        for repeat in range(6):
            if repeat == 2:
                if device == "cuda":
                    torch.cuda.synchronize()
                train_start = time.perf_counter()
            world.zero_grad()
            transition_loss(world, probe, rng, config).backward()
        if device == "cuda":
            torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - train_start
    except (torch.OutOfMemoryError, RuntimeError):
        pass
    world.zero_grad(set_to_none=True)
    for parameter in frozen:
        parameter.requires_grad_(False)
    if device == "cuda":
        torch.cuda.empty_cache()

    horizon, distance = 0, 1
    with torch.no_grad():
        while distance <= 2 * config.dynamics_context:
            length = distance + 1
            base = torch.randn(1, length, config.n_spatial, config.d_spatial, device=device).tanh()
            actions = torch.zeros(1, length, dtype=torch.long, device=device)
            pair = []
            for head in (base[:, :1], torch.randn_like(base[:, :1]).tanh()):
                sequence = torch.cat([head, base[:, 1:]], dim=1)
                committed, conditioning = commit_inputs(
                    sequence, torch.Generator(device=device).manual_seed(0), config
                )
                features, _, _ = world(None, actions, committed, conditioning)
                pair.append(features[:, -1])
            if (pair[0] - pair[1]).abs().max() < 1e-6:
                break
            horizon, distance = distance, distance * 2

    encoder = modules.get("encoder")
    return {
        "memory_horizon_at_least": horizon,
        "deployed_parameters": sum(v for k, v in counts.items() if k in deployed),
        "training_only_parameters": sum(v for k, v in counts.items() if k not in deployed),
        "dynamics_state_elements": sum(t.numel() for pair in state.memory for t in pair),
        "encoder_state_elements": config.window
        * (config.n_latents + config.n_patches)
        * config.d_model_encoder
        * (config.depth_encoder // config.time_every)
        * 2
        if encoder is not None
        else 0,
        "peak_bytes": peak,
        "steps_per_second": 16.0 / elapsed,
        "forward_backward_per_second": 4.0 / train_elapsed,
        "flops_per_step": float(flops),
        "backbone_passes_per_step": config.rungs + 1 if config.transition == "flow" else 1,
    }
