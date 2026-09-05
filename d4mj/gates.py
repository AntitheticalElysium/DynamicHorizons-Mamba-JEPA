from dataclasses import dataclass
from collections.abc import Callable
import hashlib

import torch

from .backbone import AGENT, Backbone, Layout, space_mask
from .config import Config, canonical_json, recipe_digest
from .data import Episode, sample_batch
from .state import WorldState
from .transition import advance, initial


def scan_step_parity(config: Config, tolerance: float = 1e-3) -> None:
    """Scan must equal step-by-step, in outputs and in state. Runs past
    `dynamics_context` so the windowed branch executes. See GATES.md."""
    device = config.device
    torch.manual_seed(config.seed)
    layout = Layout.dynamics(config)
    backbone = Backbone(
        config, layout, "dynamics", config.d_model, config.n_heads, config.depth,
        config.dynamics_context,
    )
    backbone = backbone.to(device).eval()

    length = config.dynamics_context + 4
    frames = torch.randn(2, length, layout.size, config.d_model, device=device)
    with torch.no_grad():
        scanned, scanned_memory = backbone(frames)
        stepped, memory = [], None
        for index in range(frames.shape[1]):
            out, memory = backbone(frames[:, index : index + 1], memory, index)
            stepped.append(out)

    drift = (torch.cat(stepped, dim=1) - scanned).abs().max() / scanned.abs().max()
    assert drift < tolerance, f"scan/step drift {drift:.2e} exceeds {tolerance:.0e}"
    assert len(memory) == len(scanned_memory) == config.depth // config.time_every
    if config.time_mixer == "attention":
        assert all(pair[0].shape[2] <= config.dynamics_context for pair in scanned_memory), (
            "the scan ignores the window the cache enforces"
        )
    _state_parity(memory, scanned_memory)

    _world_parity(config, device, tolerance)
    _encoder_parity(config, device, tolerance)


def _state_parity(stepped, scanned, tolerance: float = 5e-3) -> None:
    """State values, not just shapes. Tolerance is looser than for outputs and
    separately evidenced: Mamba's state drift is 1e-3 but flat in length, so it is
    kernel numerics, not misalignment. See GATES.md."""
    assert len(stepped) == len(scanned), "different numbers of time-mixing states"
    for layer, (left, right) in enumerate(zip(stepped, scanned)):
        for slot, (one, other) in enumerate(zip(left, right)):
            assert one.shape == other.shape, (
                f"state {layer}.{slot} shape {tuple(one.shape)} vs {tuple(other.shape)}"
            )
            scale = other.abs().max().clamp(min=1e-8)
            drift = (one - other).abs().max() / scale
            assert drift < tolerance, f"state {layer}.{slot} drift {drift:.2e} exceeds {tolerance:.0e}"


def _world_parity(config: Config, device: str, tolerance: float) -> None:
    """The teacher-forced pass and the runtime path must agree. Only training runs
    the first and only imagination runs the second, so nothing else compares them --
    and a Direct arm whose loss never saw an observation passed every other gate."""
    from .transition import World, commit_inputs

    torch.manual_seed(config.seed)
    world = World(config).to(device).eval()
    latents = torch.randn(2, 4, config.n_spatial, config.d_spatial, device=device).tanh()
    actions = torch.randint(config.n_actions, (2, 4), device=device)

    with torch.no_grad():
        committed, conditioning = commit_inputs(
            latents, torch.Generator(device=device).manual_seed(1), config
        )
        forced, _, _ = world(None, actions, committed, conditioning)
        stepped, memory = [], None
        for index in range(4):
            block = slice(index, index + 1)
            features, _, memory = world(
                memory, actions[:, block], committed[:, block], conditioning[:, block], index
            )
            stepped.append(features)

    drift = (torch.cat(stepped, dim=1) - forced).abs().max() / forced.abs().max()
    assert drift < tolerance, f"teacher-forced vs stepped world drift {drift:.2e}"
    if config.time_mixer == "attention":
        assert all(
            pair[0].shape[2] <= config.dynamics_context for pair in memory
        ), "dynamics cache exceeds the declared context"


def _encoder_parity(config: Config, device: str, tolerance: float) -> None:
    """Scan, recurrence and the cache must agree, and nothing beyond the receptive
    field may reach a latent."""

    from .representation import Encoder

    torch.manual_seed(config.seed)
    encoder = Encoder(config).to(device).eval()
    reach = config.receptive_field
    frames = torch.rand(1, reach + 6, config.n_patches, config.patch_dim, device=device)

    with torch.no_grad():
        scanned, memory, _ = encoder(frames)
        stepped, carried = [], None
        for index in range(frames.shape[1]):
            z, carried, _ = encoder(frames[:, index : index + 1], carried, offset=index)
            stepped.append(z)
        distant = frames.clone()
        distant[:, : frames.shape[1] - reach] = torch.rand_like(distant[:, :-reach])
        bounded, _, _ = encoder(distant)

    drift = (torch.cat(stepped, dim=1) - scanned).abs().max() / scanned.abs().max()
    assert drift < tolerance, f"encoder scan/step drift {drift:.2e}"
    assert all(pair[0].shape[2] <= config.window for pair in memory), "encoder state exceeds the window"
    influence = (bounded[:, -1] - scanned[:, -1]).abs().max()
    assert influence == 0, f"a frame beyond the receptive field moved z by {influence:.2e}"

    inside = frames.clone()
    inside[:, -2] = torch.rand_like(inside[:, -2])
    with torch.no_grad():
        used, _, _ = encoder(inside)
    assert not torch.equal(used[:, -1], scanned[:, -1]), "the encoder ignores its history"


def alignment(config: Config) -> None:
    """The temporal contract, on episodes whose every value identifies its index."""
    episodes = [_probe(index, config) for index in range(4)]
    rng = torch.Generator().manual_seed(config.seed)
    batch = sample_batch(episodes, rng, config)

    for row in range(batch.led_to_action.shape[0]):
        start = _recover_start(batch, row, config)
        steps = torch.arange(start, start + batch.led_to_action.shape[1])
        exists = steps > 0
        source = (steps - 1).clamp(min=0)

        assert torch.equal(batch.valid[row], exists)
        assert torch.equal(batch.reward[row][exists], source[exists].float())
        assert torch.equal(batch.led_to_action[row][exists], source[exists] % config.n_actions)
        assert (batch.led_to_action[row][~exists] == config.n_actions).all()
        assert not batch.valid[row][~exists].any()

    assert batch.patches.shape[1:] == (
        config.burn_in + config.sequence,
        config.n_patches,
        config.patch_dim,
    )
    assert batch.relevant is None, "pretraining must not stratify the corpus"

    single = [_probe(0, config)]
    event = len(single[0]) // 2
    mixed = sample_batch(single, torch.Generator().manual_seed(config.seed), config, mixture=True)
    assert int(mixed.relevant.sum()) == config.batch // 2, "the mixture is not 50/50"
    blocks = mixed.led_to_action.shape[1]
    for row in range(config.batch):
        if bool(mixed.relevant[row]):
            start = _recover_start(mixed, row, config)
            assert start <= event, "the event's outgoing action is before the window"
            assert event + 1 <= start + blocks - 1, (
                "the window ends before the achievement arrives, so the BC target is absent"
            )
    _observation_dependence(config)


def _observation_dependence(config: Config) -> None:
    """The prediction must move when context moves, action and conditioning fixed.
    Comparing losses instead would pass for a model-free loss, since changing the
    latents also changes the target."""
    from .transition import World, commit_inputs

    device = config.device
    torch.manual_seed(config.seed)
    world = World(config).to(device).eval()
    action = torch.zeros(2, 3, dtype=torch.long, device=device)
    shape = (2, 3, config.n_spatial, config.d_spatial)

    from .data import Batch
    from .transition import transition_loss

    probe = Batch(
        led_to_action=torch.zeros(2, 4, dtype=torch.long, device=device),
        reward=torch.zeros(2, 4, device=device),
        terminated=torch.zeros(2, 4, dtype=torch.bool, device=device),
        truncated=torch.zeros(2, 4, dtype=torch.bool, device=device),
        valid=torch.ones(2, 4, dtype=torch.bool, device=device),
        scored=torch.ones(2, 4, dtype=torch.bool, device=device),
        burn_in=0,
        latents=torch.randn(2, 4, config.n_spatial, config.d_spatial, device=device).tanh(),
    )
    with torch.no_grad():
        loss = transition_loss(world, probe, torch.Generator(device=device).manual_seed(5), config)
    assert torch.isfinite(loss), f"transition loss is {loss}"

    predictions = []
    for seed in (1, 2):
        context = torch.randn(
            shape, generator=torch.Generator(device=device).manual_seed(seed), device=device
        ).tanh()
        with torch.no_grad():
            committed, conditioning = commit_inputs(
                context, torch.Generator(device=device).manual_seed(4), config
            )
            features, _, _ = world(None, action, committed, conditioning)
            predictions.append(world.predict(features, action))
    assert not torch.equal(*predictions), "the prediction ignores its context"


def _conditioning_coverage(config: Config) -> None:
    """Every conditioning row must be reachable by training (S10).

    Scored past `bootstrap_start`: before it, only the finest step trains, which is
    the pinned schedule's own warmup (S67) and not an unreachable row."""
    from .data import Batch
    from .transition import World, transition_loss

    device = config.device
    torch.manual_seed(config.seed)
    world = World(config).to(device)
    tables = (
        [world.signal_embed, world.step_embed]
        if config.transition == "flow"
        else [world.condition_embed]
    )
    touched = [torch.zeros(t.num_embeddings, device=device) for t in tables]

    for trial in range(40):
        generator = torch.Generator(device=device).manual_seed(trial)
        batch = Batch(
            led_to_action=torch.zeros(4, 6, dtype=torch.long, device=device),
            reward=torch.zeros(4, 6, device=device),
            terminated=torch.zeros(4, 6, dtype=torch.bool, device=device),
            truncated=torch.zeros(4, 6, dtype=torch.bool, device=device),
            valid=torch.ones(4, 6, dtype=torch.bool, device=device),
            scored=torch.ones(4, 6, dtype=torch.bool, device=device),
            burn_in=0,
            latents=torch.randn(4, 6, config.n_spatial, config.d_spatial, device=device).tanh(),
        )
        world.zero_grad()
        transition_loss(world, batch, generator, config, step=config.bootstrap_start).backward()
        for table, seen in zip(tables, touched):
            if table.weight.grad is not None:
                seen += table.weight.grad.abs().sum(-1)

    for table, seen in zip(tables, touched):
        missing = int((seen == 0).sum())
        assert missing == 0, f"{missing} of {table.num_embeddings} conditioning rows never train"


def reset_parity(config: Config) -> None:
    """An episode boundary erases the previous episode, asserted by running two and
    requiring the second to match the same episode run alone. `step` must restart
    too: it dates RoPE and the decode window."""
    device = config.device
    world = _world(config)
    action = torch.zeros(2, 1, dtype=torch.long, device=device)

    def episode(seed: int, after=None):
        rng = torch.Generator(device=device).manual_seed(seed)
        latent = torch.randn(
            2, 1, config.n_spatial, config.d_spatial,
            generator=torch.Generator(device=device).manual_seed(seed + 7), device=device,
        ).tanh()
        state, _ = initial(world, latent, action, rng, config)
        for _ in range(3):
            state, _ = advance(world, state, action, rng, config)
        return state

    with torch.no_grad():
        first = episode(11)
        after_reset = episode(23)
        from_scratch = episode(23)

    assert after_reset.step == from_scratch.step == 4, "the reset did not restart the step counter"
    assert torch.equal(after_reset.latent, from_scratch.latent), (
        "the second episode differs depending on whether one ran before it"
    )
    for carried, clean in zip(after_reset.memory, from_scratch.memory):
        for left, right in zip(carried, clean):
            assert torch.equal(left, right), "episode state survived the boundary"
    assert not torch.equal(first.latent, from_scratch.latent), "the two episodes are not distinct"


def firewall(config: Config) -> None:
    """Agent state reaches the world only through the chosen action. Asserted in
    both directions, because a mask that blocks only one is still broken and still
    passes a one-directional test."""
    layout = Layout.dynamics(config)
    kinds = layout.kinds
    agent = kinds == AGENT
    mask = space_mask(layout, "dynamics")
    assert not mask[~agent][:, agent].any(), "world reads agent keys"
    assert mask[agent].all(), "agent cannot read the world"
    assert not space_mask(layout, "dynamics", agent_active=False)[:, agent].any()


def branch_nonmutation(config: Config) -> None:
    """Candidate evaluations from one prefix must leave it untouched, so a planner
    comparing actions cannot corrupt the state it branched from. Mamba's step
    mutates in place, which is why the mixer clones rather than trusting callers."""
    device = config.device
    world = _world(config)
    rng = torch.Generator(device=device).manual_seed(0)
    latent = torch.randn(2, 1, config.n_spatial, config.d_spatial, device=device).tanh()
    with torch.no_grad():
        state, _ = initial(world, latent, torch.zeros(2, 1, dtype=torch.long, device=device), rng, config)
        state, _ = advance(world, state, torch.zeros(2, 1, dtype=torch.long, device=device), rng, config)
        before = tuple(tuple(t.clone() for t in pair) for pair in state.memory)
        for action in range(3):
            advance(world, state, torch.full((2, 1), action, device=device), torch.Generator(device=device).manual_seed(7), config)
    for pair, original in zip(state.memory, before):
        assert all(torch.equal(a, b) for a, b in zip(pair, original)), "branch mutated the prefix"


def recurrent_carry(config: Config) -> None:
    """The rollout must depend on history. Deep and shallow memory are compared
    against one identical committed tensor, so the difference is memory alone --
    reseeding is not enough, since flow's commit draw moves the output."""
    world = _world(config)
    committed, conditioning, action, memory = _isolated(world, config)

    with torch.no_grad():
        deep, _, _ = world(memory, action, committed, conditioning, 4)
        shallow, _, _ = world(None, action, committed, conditioning, 0)
    assert not torch.allclose(deep, shallow, atol=1e-6), "memory is inert"
    _conditioning_coverage(config)


def _isolated(world, config: Config):
    """One committed block plus the memory of a four-block prefix, so a caller can
    vary memory alone. Every history gate needs exactly this and none of them can
    build it from `advance`, which redraws corruption at each call."""
    from .transition import commit_inputs

    device = config.device
    rng = torch.Generator(device=device).manual_seed(0)
    prefix = torch.randn(2, 4, config.n_spatial, config.d_spatial, device=device).tanh()
    latent = torch.randn(2, 1, config.n_spatial, config.d_spatial, device=device).tanh()
    actions = torch.zeros(2, 4, dtype=torch.long, device=device)
    action = torch.zeros(2, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        history, history_conditioning = commit_inputs(prefix, rng, config)
        _, _, memory = world(None, actions, history, history_conditioning)
        committed, conditioning = commit_inputs(latent, rng, config)
    return committed, conditioning, action, memory


def _world(config: Config):
    """Seeded, because a gate that draws fresh weights each run is a gate whose
    pass or fail is a coin flip -- one that reported both on consecutive runs of
    the same commit."""
    from .transition import World

    torch.manual_seed(config.seed)
    return World(config).to(config.device).eval()


def _probe(index: int, config: Config) -> Episode:
    """An episode where actions_taken[t] = t mod n_actions and rewards[t] = t, so a
    one-step shift is not a plausible alternative reading of any array. One task
    event sits mid-episode, so the relevant sampler has something to centre on."""
    steps = config.burn_in + config.sequence + 8 + index
    shape = (steps + 1, config.resolution, config.resolution, config.channels)
    return Episode(
        observations=torch.zeros(shape, dtype=torch.uint8),
        actions_taken=torch.arange(steps) % config.n_actions,
        rewards=torch.arange(steps).float(),
        terminated=torch.zeros(steps, dtype=torch.bool),
        truncated=torch.zeros(steps, dtype=torch.bool),
        events=torch.arange(steps) == steps // 2,
    )


def _recover_start(batch, row: int, config: Config) -> int:
    """The window's episode offset, read back out of the reward it carries rather
    than trusted from the sampler that produced it."""
    if bool(batch.valid[row][0]):
        return int(batch.reward[row][0]) + 1
    return 0


class ComponentGateError(RuntimeError):
    def __init__(self, component: str, reason: str):
        self.component = component
        super().__init__(f"{component}: {reason}")


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(canonical_json(contract).encode()).hexdigest()


@dataclass(frozen=True)
class Gate:
    name: str
    check: Callable
    requires: tuple[str, ...] = ()


LEGACY_CHECKS = ("alignment", "scan_step_parity", "reset_parity", "firewall", "branch_nonmutation", "recurrent_carry")


def preflight(config, episodes=None, dataset_contract: dict | None = None) -> dict:
    """One gate runner for both families, with explicit prerequisite stops."""
    from .lewm_config import LeWMConfig

    if isinstance(config, LeWMConfig):
        from .lewm_diagnostics import joint_checks
        if episodes is None or dataset_contract is None:
            raise ComponentGateError("dataset", "joint preflight requires an audited raw corpus")
        report = {"schema": "d4mj_lewm_gates_v1", "recipe_id": recipe_digest(config),
                  "dataset_id": contract_digest(dataset_contract), "components": {},
                  "architecture_verdict": "not_evaluated",
                  "m4": {"status": "blocked", "reason": "M4/H16/actor/renderer require subsequent component gates"}}
        checks = joint_checks(config, episodes, report)
    else:
        report = {"schema": "d4mj_stage_a_gates_v1", "recipe_id": recipe_digest(config),
                  "family": f"{config.transition}-{config.time_mixer}", "components": {},
                  "architecture_verdict": "not_evaluated"}
        checks = [Gate(name, lambda name=name: globals()[name](config)) for name in LEGACY_CHECKS]
    for gate in checks:
        missing = [name for name in gate.requires if report["components"].get(name,{}).get("status") != "pass"]
        if missing:
            report["components"][gate.name] = {"status": "blocked", "reason": f"prerequisites did not pass: {missing}"}
            continue
        try:
            report["components"][gate.name] = {"status": "pass", "detail": gate.check()}
        except Exception as error:
            report["components"][gate.name] = {"status": "fail", "reason": f"{type(error).__name__}: {error}"}
    return report


def require_joint_gates(report: dict, config, dataset_contract: dict):
    from .sources import lewm_source_manifest
    if report.get("schema") != "d4mj_lewm_gates_v1" or report.get("recipe_id") != recipe_digest(config):
        raise ComponentGateError("gate_identity", "missing/stale recipe validation")
    if report.get("dataset_id") != contract_digest(dataset_contract):
        raise ComponentGateError("gate_identity", "dataset changed since preflight")
    for name in ("source_identity", "dataset", "device", "objective_source", "model_construction", "recurrence", "normalization", "joint_resource"):
        record = report.get("components", {}).get(name, {})
        if record.get("status") != "pass":
            raise ComponentGateError(name, record.get("reason", "gate has not run"))
    try:
        sources = lewm_source_manifest()
    except Exception as error:
        raise ComponentGateError("source_identity", str(error)) from error
    if report.get("sources") != sources:
        raise ComponentGateError("gate_identity", "source/dependency code changed since preflight")
    if config.runtime.device == "cuda":
        if not torch.cuda.is_available():
            raise ComponentGateError("device", "CUDA unavailable in the launch process")
        if report["components"]["joint_resource"]["detail"]["device_name"] != torch.cuda.get_device_name():
            raise ComponentGateError("device", "target GPU changed; run preflight on this device")
    if config.runtime.purpose == "research" and not report["components"]["joint_resource"]["detail"]["gpu_validated"]:
        raise ComponentGateError("joint_resource", "research launch requires target-GPU validation")
