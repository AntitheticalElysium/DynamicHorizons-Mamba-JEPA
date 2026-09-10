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


def binary_auc(scores: Tensor, truth: Tensor) -> float | None:
    """Mann-Whitney AUC with average ranks for ties; absent classes are not .5."""
    scores, truth = scores.detach().cpu().double(), truth.detach().cpu().bool()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("probe scores are nonfinite")
    positive = int(truth.sum()); negative = len(truth)-positive
    if not positive or not negative:
        return None
    order = scores.argsort(); sorted_scores = scores[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    end = counts.cumsum(0).double()
    ranks = torch.repeat_interleave(end-(counts.double()-1)/2, counts)
    return float((ranks[truth[order]].sum()-positive*(positive+1)/2)/(positive*negative))


def paired_auc_interval(left, right, truth, valid, clusters, *, draws: int, seed: int):
    """Macro-AUC difference, resampling whole paired episodes with fixed labels."""
    def delta(indices):
        values = []
        for col in range(truth.shape[1]):
            chosen = indices[valid[indices, col]]
            a, b = binary_auc(left[chosen, col], truth[chosen, col]), binary_auc(right[chosen, col], truth[chosen, col])
            if a is None or b is None:
                return None
            values.append(a-b)
        return sum(values)/len(values) if values else None
    groups = [torch.where(clusters == key)[0] for key in clusters.unique(sorted=True)]
    point = delta(torch.arange(len(truth)))
    samples = []
    rng = torch.Generator().manual_seed(seed)
    for _ in range(draws):
        indices = torch.cat([groups[i] for i in torch.randint(len(groups), (len(groups),), generator=rng)])
        value = delta(indices)
        if value is not None:
            samples.append(value)
    bounds = None if len(samples) < .95*draws else torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return {"difference": point, "interval": bounds, "valid_draws": len(samples), "draws": draws,
            "unit": "episode", "status": "measured" if bounds is not None else "insufficient_coverage"}


def fit_outcome_probe(train_x, train_y, train_valid, dev_x, settings, *, hidden: bool):
    """Fixed TRAIN-only normalization and optimization, no DEV selection."""
    from types import SimpleNamespace
    from .train import optimizer, optimizer_step

    device = train_x.device
    mean, scale = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x, dev_x = (train_x-mean)/scale, (dev_x-mean)/scale
    width, outputs = train_x.shape[1], train_y.shape[1]
    seed = settings.seed + (200 if hidden else 100)
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = (nn.Sequential(nn.Linear(width, settings.probe_hidden), nn.GELU(),
                               nn.Linear(settings.probe_hidden, outputs)) if hidden else nn.Linear(width, outputs)).to(device)
    config = SimpleNamespace(learning_rate=settings.probe_lr, weight_decay=settings.probe_decay)
    opt = optimizer([model], config)
    positive = (train_y*train_valid).sum(0)
    negative = ((1-train_y)*train_valid).sum(0)
    weight = negative/positive.clamp_min(1)
    rng = torch.Generator(device=device).manual_seed(seed+1)
    for _ in range(settings.probe_steps):
        index = torch.randint(len(train_x), (settings.probe_batch,), generator=rng, device=device)
        losses = nn.functional.binary_cross_entropy_with_logits(model(train_x[index]), train_y[index],
                                                                 pos_weight=weight, reduction="none")
        loss = (losses*train_valid[index]).sum()/train_valid[index].sum().clamp_min(1)
        optimizer_step(opt, loss, model.parameters(), learning_rate=settings.probe_lr,
                       grad_clip=1.0, strict=True)
    with torch.no_grad():
        return model.eval()(dev_x).cpu()
