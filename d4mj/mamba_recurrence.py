"""Functional persistent Mamba-2 carry with the pinned upstream parameterization.

Equations follow state-spaces/mamba f577286d, modules/mamba2.py and
ops/triton/layernorm_gated.py (Copyright Tri Dao, Albert Gu; Apache-2.0).
The core *is* upstream Mamba2. This wrapper replaces mutation of its inference
cache with functional convolution/SSM carry, including gradients to initial state.
No observation prefix is stored or replayed by step().
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .lewm_config import DynamicsSettings


@dataclass(frozen=True)
class MambaCarry:
    conv: Tensor                 # B, inner+2*d_state, d_conv (newest on the right)
    ssm: Tensor                  # B, heads, headdim, d_state; persistent FP32


def clone_carry(state: MambaCarry) -> MambaCarry:
    return MambaCarry(state.conv.clone(), state.ssm.clone())


def detach_carry(state: MambaCarry) -> MambaCarry:
    # Own the storage as well: detached siblings must not alias mutable callers.
    return MambaCarry(state.conv.detach().clone(), state.ssm.detach().clone())


def repeat_carry(state: MambaCarry, count: int) -> MambaCarry:
    if count < 1:
        raise ValueError("repeat count must be positive")
    return MambaCarry(state.conv.repeat_interleave(count, 0), state.ssm.repeat_interleave(count, 0))


class FunctionalMamba2(nn.Module):
    def __init__(self, settings: DynamicsSettings):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.settings = settings
        self.core = Mamba2(
            d_model=settings.width, d_state=settings.d_state, d_conv=settings.d_conv,
            conv_init=None, expand=settings.expand, headdim=settings.headdim,
            d_ssm=None, ngroups=settings.ngroups, A_init_range=settings.A_init_range,
            D_has_hdim=False, rmsnorm=True, norm_before_gate=settings.norm_before_gate,
            dt_min=settings.dt_min, dt_max=settings.dt_max, dt_init_floor=settings.dt_init_floor,
            dt_limit=(0.0, float("inf")), bias=settings.bias, conv_bias=settings.conv_bias,
            chunk_size=settings.chunk_size, use_mem_eff_path=False, layer_idx=0,
            process_group=None, sequence_parallel=False,
        )

    def initial(self, batch: int, *, device=None, dtype=None) -> MambaCarry:
        c = self.core
        return MambaCarry(
            torch.zeros(batch, c.d_ssm + 2 * c.d_state, c.d_conv,
                        device=device or c.in_proj.weight.device, dtype=dtype or c.in_proj.weight.dtype),
            torch.zeros(batch, c.nheads, c.headdim, c.d_state,
                        device=device or c.in_proj.weight.device, dtype=torch.float32),
        )

    def _validate(self, x: Tensor, state: MambaCarry):
        c = self.core
        if x.ndim != 3 or x.shape[-1] != c.d_model or x.shape[1] < 1:
            raise ValueError("Mamba input must be B,T,width with T>=1")
        if state.conv.shape != (x.shape[0], c.d_ssm + 2*c.d_state, c.d_conv):
            raise ValueError("convolution carry shape mismatch")
        if state.ssm.shape != (x.shape[0], c.nheads, c.headdim, c.d_state):
            raise ValueError("SSM carry shape mismatch")
        if state.conv.device != x.device or state.ssm.device != x.device or state.ssm.dtype != torch.float32:
            raise ValueError("carry must be on input device with FP32 SSM state")

    def scan(self, inputs: Tensor, carry: MambaCarry | None = None,
             *, backend: str | None = None) -> tuple[Tensor, MambaCarry]:
        c = self.core
        if inputs.ndim != 3:
            raise ValueError("Mamba input must have three dimensions")
        carry = self.initial(inputs.shape[0], device=inputs.device, dtype=inputs.dtype) if carry is None else carry
        self._validate(inputs, carry)
        selected = backend or self.settings.backend
        if selected not in ("reference", "triton"):
            raise ValueError("unknown Mamba backend")
        if selected == "triton" and inputs.device.type != "cuda":
            raise RuntimeError("recurrence_cuda: Triton backend needs CUDA; no silent reference fallback")
        z, xbc, dt = torch.split(c.in_proj(inputs),
                                [c.d_ssm, c.d_ssm + 2*c.d_state, c.nheads], dim=-1)
        raw = xbc.transpose(1, 2)
        # K-1 previous inputs followed by this chunk, with no in-place cache updates.
        history = torch.cat((carry.conv[:, :, 1:].to(raw.dtype), raw), dim=-1)
        filtered = F.silu(F.conv1d(history, c.conv1d.weight, c.conv1d.bias,
                                  groups=c.conv1d.groups)).transpose(1, 2)
        final_conv = torch.cat((carry.conv.to(raw.dtype), raw), dim=-1)[:, :, -c.d_conv:].clone()
        x, b, cc = torch.split(filtered, [c.d_ssm, c.d_state, c.d_state], dim=-1)
        x = x.reshape(*x.shape[:2], c.nheads, c.headdim)
        if selected == "reference":
            y, final_ssm = self._reference_ssm(x, b, cc, dt, carry.ssm)
        else:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

            y, final_ssm = mamba_chunk_scan_combined(
                x, dt, -c.A_log.float().exp(), b.unsqueeze(2), cc.unsqueeze(2),
                chunk_size=c.chunk_size, D=c.D, dt_bias=c.dt_bias, dt_softplus=True,
                initial_states=carry.ssm, return_final_states=True, state_dtype=torch.float32,
            )
        # Source RMSNormGated(norm_before_gate=False): norm(y * silu(z)).
        y = y.flatten(2).to(z.dtype)
        gated = y.float() * F.silu(z.float())
        normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + c.norm.eps)
        normalized = (normalized * c.norm.weight.float()).to(y.dtype)
        return c.out_proj(normalized), MambaCarry(final_conv, final_ssm.float())

    def _reference_ssm(self, x: Tensor, b: Tensor, c: Tensor, dt: Tensor,
                       initial: Tensor) -> tuple[Tensor, Tensor]:
        core = self.core
        # Disable autocast for the persistent recurrence itself, not just its inputs.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, b, c = x.float(), b.float(), c.float()
            delta = F.softplus(dt.float() + core.dt_bias.float())
            a = -core.A_log.float().exp()
            state, outputs = initial.float(), []
            for t in range(x.shape[1]):
                decay = (delta[:, t] * a).exp()[:, :, None, None]
                update = delta[:, t, :, None, None] * x[:, t, :, :, None] * b[:, t, None, None, :]
                state = decay * state + update
                y = (state * c[:, t, None, None, :]).sum(-1) + core.D[None, :, None].float() * x[:, t]
                outputs.append(y)
            return torch.stack(outputs, 1), state

    def step(self, inputs: Tensor, carry: MambaCarry) -> tuple[Tensor, MambaCarry]:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("step consumes exactly one action-pair token")
        return self.scan(inputs, carry)

    def step_reference(self, inputs: Tensor, carry: MambaCarry) -> tuple[Tensor, MambaCarry]:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("step_reference consumes exactly one token")
        return self.scan(inputs, carry, backend="reference")

    forward = scan
