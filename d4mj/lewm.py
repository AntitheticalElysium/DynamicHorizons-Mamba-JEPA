"""Joint predictive encoder, source-scale SIGReg and persistent action-pair world."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .lewm_config import LeWMConfig, EncoderSettings
from .mamba_recurrence import FunctionalMamba2
from .state import PredictiveState


class LeWMProjector(nn.Sequential):
    def __init__(self, input_dim: int, settings: EncoderSettings):
        super().__init__(
            nn.Linear(input_dim, settings.projector_hidden),
            nn.BatchNorm1d(settings.projector_hidden, eps=settings.bn_eps,
                           momentum=settings.bn_momentum, affine=True, track_running_stats=True),
            nn.GELU(), nn.Linear(settings.projector_hidden, settings.latent_dim),
        )


class LeWMEncoder(nn.Module):
    """Framewise ViT CLS + retained projector. Export is B,T,1,D, unbounded."""

    def __init__(self, config: LeWMConfig):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        self.settings = e = config.encoder
        vit = ViTConfig(
            hidden_size=e.width, num_hidden_layers=e.depth, num_attention_heads=e.heads,
            intermediate_size=e.width * e.mlp_ratio, image_size=e.resolution,
            patch_size=e.patch, num_channels=e.channels, hidden_act="gelu",
            hidden_dropout_prob=e.dropout, attention_probs_dropout_prob=e.dropout,
            initializer_range=e.initializer_range, layer_norm_eps=e.layer_norm_eps,
            qkv_bias=e.qkv_bias, attn_implementation=e.attention_backend,
        )
        self.backbone = ViTModel(vit, add_pooling_layer=False, use_mask_token=False)
        if e.checkpoint_blocks:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.projector = LeWMProjector(e.width, e)
        self.register_buffer("pixel_mean", torch.tensor(e.pixel_mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(e.pixel_std).view(1, 3, 1, 1))
        self._frozen = False

    def train(self, mode: bool = True):
        return super().train(False if self._frozen else mode)

    def freeze(self):
        self._frozen = True
        self.requires_grad_(False)
        self.eval()
        return self

    def projected_and_cls(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        e = self.settings
        if frames.dtype != torch.uint8 or frames.ndim != 5 or tuple(frames.shape[2:]) != (e.resolution, e.resolution, 3):
            raise ValueError("encoder expects native uint8 B,T,H,W,3; preprocessing is internal")
        b, t = frames.shape[:2]
        pixels = frames.flatten(0, 1).permute(0, 3, 1, 2).contiguous().float() / 255.0
        pixels = (pixels - self.pixel_mean) / self.pixel_std
        cls = self.backbone(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        z = self.projector(cls)
        return z.reshape(b, t, 1, e.latent_dim), cls.reshape(b, t, e.width)

    def forward(self, frames: Tensor) -> Tensor:
        return self.projected_and_cls(frames)[0]


class SIGReg(nn.Module):
    """Exact reduction/weight scale of pinned le-wm/module.py, with explicit RNG.

    Input T,B,D. Statistical batch is B; time positions are averaged afterwards.
    The upstream trapezoid weights intentionally include its factor of two.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("invalid SIGReg integration geometry")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, values: Tensor, generator: torch.Generator | None = None,
                *, directions: Tensor | None = None) -> Tensor:
        if values.ndim != 3 or values.shape[1] < 2:
            raise ValueError("SIGReg needs T,B,D with actual B>=2")
        if directions is None and generator is None:
            raise ValueError("SIGReg requires a saved projection RNG")
        with torch.autocast(device_type=values.device.type, enabled=False):
            values = values.float()
            if directions is None:
                directions = torch.randn(values.shape[-1], self.num_proj, device=values.device,
                                         dtype=torch.float32, generator=generator)
                directions = directions / directions.norm(dim=0, keepdim=True)
            if directions.shape != (values.shape[-1], self.num_proj):
                raise ValueError("SIGReg projection shape mismatch")
            x = (values @ directions.float()).unsqueeze(-1) * self.t
            err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
            return ((err @ self.weights) * values.shape[1]).mean()


class _MambaBlock(nn.Module):
    def __init__(self, config: LeWMConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.dynamics.width, eps=config.dynamics.norm_eps)
        self.mixer = FunctionalMamba2(config.dynamics)

    def forward(self, x, memory=None, *, backend=None):
        out, carry = self.mixer(self.norm(x), memory, backend=backend)
        return x + out, carry


@dataclass(frozen=True)
class TeacherOutput:
    predicted: Tensor                  # B,T-1,1,D, aligned with observed z[:,1:]
    features: Tensor                   # B,T,1,width; outgoing action never enters h_t
    state: PredictiveState


class LeWMWorld(nn.Module):
    def __init__(self, config: LeWMConfig):
        super().__init__()
        self.config = config
        e, d = config.encoder, config.dynamics
        self.action_embedding = nn.Embedding(d.n_actions, d.action_dim)
        self.pair_projection = nn.Linear(e.latent_dim + d.action_dim, d.width)
        self.layers = nn.ModuleList([_MambaBlock(config) for _ in range(d.depth)])
        self.final_norm = nn.RMSNorm(d.width, eps=d.norm_eps)
        self.predictor_projector = LeWMProjector(d.width, e)
        self.agent_readout = nn.Sequential(nn.Linear(e.latent_dim + d.width, d.width),
                                          nn.LayerNorm(d.width, eps=d.norm_eps), nn.GELU())
        # The final state API exists in M2, but readout fitting belongs to M4.
        self.agent_readout.requires_grad_(False)

    def _latents(self, z: Tensor):
        if z.ndim != 4 or tuple(z.shape[2:]) != (1, self.config.encoder.latent_dim) or z.shape[1] < 1:
            raise ValueError("world latent must be B,T,1,latent_dim with T>=1")

    def _actions(self, a: Tensor, shape):
        if a.dtype != torch.long or a.shape != shape:
            raise ValueError("outgoing actions must be int64 B,T matching completed pairs")
        if a.numel() and (bool((a < 0).any()) or bool((a >= self.config.dynamics.n_actions).any())):
            raise ValueError("BOS/padding is not an outgoing policy action")

    def validate_state(self, state: PredictiveState):
        if not isinstance(state, PredictiveState):
            raise TypeError("LeWM requires PredictiveState, never legacy WorldState")
        self._latents(state.latent)
        if state.latent.shape[1] != 1 or state.history.shape != (state.latent.shape[0], 1, self.config.dynamics.width):
            raise ValueError("current latent/history shape mismatch")
        if len(state.memory) != len(self.layers) or type(state.step) is not int or state.step < 0:
            raise ValueError("invalid recurrence depth/step")

    def start(self, z: Tensor) -> PredictiveState:
        self._latents(z)
        if z.shape[1] != 1:
            raise ValueError("start takes exactly one initial frame")
        memory = tuple(layer.mixer.initial(z.shape[0], device=z.device, dtype=z.dtype) for layer in self.layers)
        return PredictiveState(z.clone(), memory, z.new_zeros(z.shape[0], 1, self.config.dynamics.width), 0)

    def readout(self, z: Tensor, history: Tensor) -> Tensor:
        return self.agent_readout(torch.cat((z[:, :, 0], history), -1)).unsqueeze(2)

    def features(self, state: PredictiveState) -> Tensor:
        self.validate_state(state)
        return self.readout(state.latent, state.history)

    def scan_pairs(self, z: Tensor, actions: Tensor, memory=None, *, backend=None):
        self._latents(z)
        self._actions(actions, z.shape[:2])
        if memory is not None and len(memory) != len(self.layers):
            raise ValueError("recurrence layer count mismatch")
        x = self.pair_projection(torch.cat((z[:, :, 0], self.action_embedding(actions)), -1))
        carried = []
        for i, layer in enumerate(self.layers):
            x, m = layer(x, None if memory is None else memory[i], backend=backend)
            carried.append(m)
        history = self.final_norm(x)
        predicted = self.predictor_projector(history.flatten(0, 1)).reshape(
            z.shape[0], z.shape[1], 1, self.config.encoder.latent_dim)
        return predicted, history, tuple(carried)

    def teacher(self, z: Tensor, actions: Tensor, *, state: PredictiveState | None = None,
                backend=None) -> TeacherOutput:
        self._latents(z)
        self._actions(actions, (z.shape[0], z.shape[1]-1))
        initial = self.start(z[:, :1]) if state is None else state
        self.validate_state(initial)
        if not torch.equal(initial.latent, z[:, :1]):
            raise ValueError("continuing teacher scan must begin at the state's current latent")
        if z.shape[1] == 1:
            return TeacherOutput(z[:, :0], self.features(initial), initial)
        prediction, history, memory = self.scan_pairs(z[:, :-1], actions, initial.memory, backend=backend)
        aligned = torch.cat((initial.history, history), dim=1)
        final = PredictiveState(z[:, -1:].clone(), memory, history[:, -1:].clone(), initial.step + actions.shape[1])
        return TeacherOutput(prediction, self.readout(z, aligned), final)

    def _streaming_mode(self):
        if any(isinstance(m, nn.BatchNorm1d) and m.training for m in self.predictor_projector.modules()):
            raise RuntimeError("predictor_normalization: streaming requires fixed BN statistics (eval mode)")

    def advance(self, state: PredictiveState, action: Tensor) -> tuple[PredictiveState, Tensor]:
        self._streaming_mode()
        self.validate_state(state)
        self._actions(action, (state.latent.shape[0], 1))
        z, history, memory = self.scan_pairs(state.latent, action, state.memory)
        result = PredictiveState(z, memory, history, state.step + 1)
        return result, self.features(result)

    def observe_latent(self, state: PredictiveState, action: Tensor, z_next: Tensor):
        # Same accepted pair update; replace only its predicted successor with truth.
        self._latents(z_next)
        if z_next.shape != state.latent.shape:
            raise ValueError("observed successor must match one current latent")
        predicted, _ = self.advance(state, action)
        result = PredictiveState(z_next.clone(), predicted.memory, predicted.history, predicted.step)
        return result, self.features(result)


@dataclass(frozen=True)
class JointLoss:
    total: Tensor
    prediction: Tensor
    regularization: Tensor
    latent: Tensor
    predicted: Tensor


def joint_loss(encoder: LeWMEncoder, world: LeWMWorld, frames: Tensor, actions: Tensor,
               regularizer: SIGReg, generator: torch.Generator, config: LeWMConfig) -> JointLoss:
    if frames.shape[:2] != (config.joint.batch, config.joint.frames):
        raise ValueError("statistical_batch: use the entire declared B,T, not accumulated microbatches")
    z = encoder(frames)
    prediction = world.teacher(z, actions).predicted
    with torch.autocast(device_type=z.device.type, enabled=False):
        pred_loss = (prediction.float() - z[:, 1:].float()).square().mean()
        values = z[:, :, 0].float()
        if config.variant == "tc":
            values = values - values.mean(1, keepdim=True)
        reg_loss = regularizer(values.transpose(0, 1), generator)
        total = pred_loss + config.joint.sigreg_weight * reg_loss
    return JointLoss(total, pred_loss, reg_loss, z, prediction)
