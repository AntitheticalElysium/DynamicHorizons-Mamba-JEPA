from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mamba_recurrence import MambaCarry

from torch import Tensor

Memory = tuple[tuple[Tensor, Tensor], ...]
"""Per temporal layer: (keys, values) for attention, (conv, ssm) for Mamba. Opaque
outside `time_mixer`, and never mutated in place."""


@dataclass(frozen=True)
class WorldState:
    """The imagined state S_t = (z_t, m_t). `latent` is the accepted clean latent;
    `memory` ingested whatever the committed block held, corrupted for flow, and
    covers the prefix through block t inclusive -- so `latent` is never ingested
    twice."""

    latent: Tensor
    memory: Memory
    step: int
    features: Tensor | None = None


@dataclass(frozen=True)
class RealState:
    """The deployed state, adding the tokenizer's own bounded-window memory, which
    imagination has no use for. A reset clears both."""

    encoder_memory: Memory
    world: WorldState


@dataclass(frozen=True)
class PredictiveState:
    """LeWM state after `step` completed (latent, outgoing-action) pairs.

    memory excludes the current latent and the not-yet-chosen outgoing action.
    history is the previous pair's top Mamba output; it is zero at a fresh start.
    Unlike WorldState, this state does NOT commit a current-frame block.
    Latent: B,1,1,D. History: B,1,width. step is relative to the supplied start.
    """

    latent: Tensor
    memory: tuple["MambaCarry", ...]
    history: Tensor
    step: int


def repeat_memory(memory, roots: int, actions: int):
    """Each root's memory repeated across its actions.

    Memory is (roots x tokens, ...) -- token-major -- so repeating dim 0 scrambles
    history without erroring. Unflatten first.
    """
    if actions < 1 or roots < 1:
        raise ValueError("repeat batch/count must be positive")
    if memory is None:
        return None
    out = []
    for pair in memory:
        widened = []
        for tensor in pair:
            rest = tensor.shape[1:]
            widened.append(tensor.view(roots, -1, *rest)
                           .repeat_interleave(actions, dim=0).reshape(-1, *rest))
        out.append(tuple(widened))
    return tuple(out)
