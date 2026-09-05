from dataclasses import asdict, dataclass, fields
from typing import Literal
import hashlib
import json
from pathlib import Path

Transition = Literal["flow", "direct"]
TimeMixer = Literal["attention", "mamba"]


@dataclass(frozen=True)
class Config:
    """Every constant in the system. The four Stage-A arms are four Config values.

    Fields that define Z* -- patch, window, n_latents, d_bottleneck, packing --
    must be frozen before the final encoder trains: changing one changes every
    latent. Capacity was fixed by the 6 GB probe (S44), which moved `batch` and
    turned on checkpointing without touching any architecture field.
    """

    transition: Transition = "flow"
    time_mixer: TimeMixer = "attention"

    # Environment. Craftax-Classic renders 9x9 tiles of BLOCK_PIXEL_SIZE_AGENT=7.
    resolution: int = 63
    channels: int = 3
    patch: int = 7
    n_actions: int = 17

    # Z*. window is the bounded causal context: z_t = Z*(x_{t-window+1..t}).
    window: int = 16
    # The original 32-slot export discarded task-relevant encoder structure.
    # Sixty-four slots closed that export loss at both frozen-tokenizer seeds and
    # improved Direct successor fidelity at both seeds downstream (S84).
    n_latents: int = 64
    d_bottleneck: int = 16
    packing: int = 2
    mae_p_max: float = 0.9
    lpips_weight: float = 0.2

    d_model_encoder: int = 256
    depth_encoder: int = 8
    n_heads_encoder: int = 4

    d_model: int = 256
    depth: int = 8
    n_heads: int = 4
    mlp_ratio: float = 4.0
    time_every: int = 4
    n_register: int = 4
    n_agent: int = 2

    # d_state is the primary parameter-matching knob.
    mamba_d_state: int = 64
    mamba_headdim: int = 64
    mamba_expand: int = 1
    mamba_d_conv: int = 4

    # k_max is the training noise grid; rungs is the generation ladder length.
    k_max: int = 8
    rungs: int = 4
    tau_ctx_noise: float = 0.1

    # Shortcut scheduling, from the pinned mmbench2 defaults: a `self_fraction` of
    # *rows* bootstrap at coarser step sizes while the rest are supervised at d_min,
    # and no row bootstraps before `bootstrap_start`. Sampling the step size per
    # position instead inverts this -- 75% of positions chase targets produced by an
    # untrained model from the first update.
    self_fraction: float = 0.25
    bootstrap_start: int = 10_000

    mtp_leads: int = 8
    bins: int = 255
    symlog_limit: float = 20.0

    # A smoke default, not a settled value (S54). The final horizon is selected on
    # DEV under S63: the largest candidate whose rollout still beats the marginal
    # predictor, decided before any FINAL cell is inspected.
    horizon: int = 8
    horizon_candidates: tuple[int, ...] = (4, 8, 16, 32)
    # How many genuinely generated states `_direct_loss` trains. Deployment must not
    # imagine past it, or both transition and head inputs leave their trained
    # distribution -- S68 caps the direct arm's horizon here. Raising it is what a
    # longer actor horizon costs: the rollout eats the last `direct_rollout` blocks of
    # every row, so `sequence` bounds it and must be raised alongside.
    direct_rollout: int = 2

    # Dreamer 3 keeps its predicted prior and its observation-derived posterior aligned
    # with two stop-gradient KLs (`rssm.loss`: dyn = kl(sg(post)||prior), rep =
    # kl(post||sg(prior))). Direct has no equivalent -- it inherits only V-JEPA 2-AC's
    # recursive MSE -- and the measured failure is that generated states stay
    # geometrically close to real ones while their readouts stop meaning the same thing
    # to the agent. This weights a stop-gradient pull of the generated readout onto the
    # observed one at the already-matched rollout positions. An additive coefficient
    # on that term, not a share of a fixed budget. Zero until an arm asks.
    align_weight: float = 0.0

    # Evaluation (S52). The native Craftax horizon, not the collector's 2500 cap.
    horizon_eval: int = 10000
    eval_episodes: int = 64
    bootstrap: int = 2000
    parameter_tolerance: float = 0.005
    outcome_gate_seeds: tuple[int, ...] = tuple(range(12_000, 12_008))
    outcome_gate_steps: tuple[int, ...] = (0, 15, 40, 80, 110)
    outcome_gate_limit: int = 400
    outcome_gate_flow_samples: int = 4
    outcome_gate_min_opportunities: int = 3

    gamma: float = 0.997
    lam: float = 0.95
    pmpo_alpha: float = 0.5
    prior_beta: float = 0.3

    # Dreamer 4 reports C = 3*T_short and T_long = 4*T_short in all three of its
    # Appendix A configurations. A scaling rule it reports, not one it claims.
    sequence: int = 16
    sequence_long: int = 64
    dynamics_context: int = 48
    long_batch_every: int = 4
    long_only_fraction: float = 0.25
    commit_prefix_fraction: float = 0.25
    episode_start_fraction: float = 0.25
    terminal_batch: int = 1
    # Share of behaviour-cloning rows centred on a task event. The rest are ordinary
    # expert windows: D4's relevant sequences are task-conditioned, and once task
    # conditioning is dropped for one aggregate policy (S51) an event-only rule
    # starves BC of navigation, survival and positioning -- measured, it left 84.5%
    # of expert behaviour unreachable as a target.
    event_fraction: float = 0.5
    batch: int = 4
    # Phase 3 sizes its own batch. It imagines from cached latents and never runs
    # the tokenizer, so it is nowhere near Phase 1A's memory ceiling, and PMPO's
    # sign-of-advantage estimate is over starting contexts -- inheriting 4 of them
    # would be a memory limit from another phase deciding the actor's gradient.
    actor_batch: int = 16
    gradient_checkpointing: bool = True
    rms_decay: float = 0.99
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup: int = 1000
    checkpoint_every: int = 500
    ema_momentum: tuple[float, float] = (0.996, 1.0)

    seed: int = 20260731

    # One device for every arm, gates included. Whether a model uses Mamba is an
    # architecture choice; routing attention to CPU and Mamba to CUDA would make
    # throughput, memory and numerics incomparable across the only axis being
    # measured, and a gate suite on other hardware validates a pairing nothing runs.
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        assert self.resolution % self.patch == 0
        assert self.n_latents == self.n_spatial * self.packing
        assert self.depth % self.time_every == 0
        assert self.depth // self.time_every >= 2
        assert self.k_max >= 8 and self.k_max & (self.k_max - 1) == 0
        assert self.rungs <= self.k_max
        assert self.tau_ctx_index < self.k_max
        # The rollout needs one block to start from and one per generated state. Short
        # batches are the binding schedule, so this is checked against `sequence`, not
        # `sequence_long` -- otherwise three rows in four would train teacher forcing
        # alone and the configured depth would be silently fictional.
        assert self.direct_rollout < self.sequence
        assert self.dynamics_context == 3 * self.sequence
        assert self.sequence_long == 4 * self.sequence
        assert self.sequence_long > self.dynamics_context
        assert self.depth_encoder % self.time_every == 0
        assert self.d_model % self.n_heads == 0 and (self.d_model // self.n_heads) % 2 == 0
        assert self.d_model_encoder % self.n_heads_encoder == 0
        assert (self.d_model_encoder // self.n_heads_encoder) % 2 == 0
        assert self.rungs & (self.rungs - 1) == 0 and self.k_max % self.rungs == 0
        assert (self.mamba_expand * self.d_model) % self.mamba_headdim == 0
        assert 0.0 <= self.commit_prefix_fraction < 1.0
        assert 0.0 <= self.long_only_fraction <= 1.0
        assert self.terminal_batch > 0
        assert self.batch % 2 == 0, "the 50/50 mixture needs an even batch"
        assert self.actor_batch % 2 == 0, "the 50/50 mixture needs an even batch"

    @property
    def n_patches(self) -> int:
        return (self.resolution // self.patch) ** 2

    @property
    def terminal_loss_mass(self) -> float:
        """Equal per-sequence weight across the main and terminal strata."""
        return self.terminal_batch / (self.batch + self.terminal_batch)

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch * self.patch

    @property
    def n_spatial(self) -> int:
        return self.n_latents // self.packing

    @property
    def d_spatial(self) -> int:
        return self.d_bottleneck * self.packing

    @property
    def receptive_field(self) -> int:
        """`window` bounds each time layer's state. Stacked sliding windows overlap
        by one position, so L layers reach 1 + L(W-1) frames, not L*W -- measured as
        exactly zero influence at 1 + L(W-1) and nonzero one frame earlier."""
        return 1 + (self.depth_encoder // self.time_every) * (self.window - 1)

    @property
    def burn_in(self) -> int:
        return self.receptive_field - 1

    @property
    def n_signal_bins(self) -> int:
        """Exactly k_max: the top row of a k_max+1 table is unreachable by training."""
        return self.k_max

    @property
    def n_step_bins(self) -> int:
        return self.k_max.bit_length()

    @property
    def tau_ctx_index(self) -> int:
        """Signal bin of every committed block, clamped below the untrained top row."""
        return min(round((1.0 - self.tau_ctx_noise) * self.k_max), self.k_max - 1)

    @property
    def tau_ctx_signal(self) -> float:
        """The signal the committed tensor is actually mixed at. Derived from the bin
        so content and label cannot disagree: mixing at 0.9 while labelling bin 7/8
        mislabels every observation, commit and deployment prefix."""
        return self.tau_ctx_index / self.k_max

    @property
    def step_index(self) -> int:
        """Committed and observed blocks carry the finest step size d_min."""
        return self.n_step_bins - 1


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_dict(config) -> dict:
    return json.loads(canonical_json(asdict(config)))


def recipe_digest(config) -> str:
    return hashlib.sha256(canonical_json(recipe_dict(config)).encode()).hexdigest()


def config_from_dict(values: dict):
    """Explicit family dispatch; incompatible flat/nested settings never mingle."""
    if not isinstance(values, dict):
        raise ValueError("recipe must be an object")
    if values.get("family") == "lewm_mamba":
        from .lewm_config import config_from_dict as parse_joint
        return parse_joint(values)
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"unknown Config fields: {sorted(unknown)}")
    values = dict(values)
    for field in fields(Config):
        if field.name in values and isinstance(field.default, tuple):
            values[field.name] = tuple(values[field.name])
    return Config(**values)


def load_recipe(path: str | Path):
    return config_from_dict(json.loads(Path(path).read_text()))
