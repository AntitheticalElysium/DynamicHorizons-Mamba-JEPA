from dataclasses import dataclass

import torch
from torch import Tensor

from .agent import Heads
from .config import Config
from .state import WorldState
from .transition import World
from .world_api import ModelBundle


@dataclass(frozen=True)
class Trajectory:
    """`continuation` is the head's probability, not a boolean -- imagination has no
    ground-truth termination. `agent` covers every state including the start, so the
    prior and critic are evaluated where the actions were chosen."""

    action: Tensor
    logits: Tensor
    reward: Tensor
    continuation: Tensor
    value: Tensor
    agent: Tensor


def imagine(
    world: World,
    heads: Heads,
    state: WorldState,
    agent: Tensor,
    rng: torch.Generator,
    policy_rng: torch.Generator,
    config: Config,
) -> Trajectory:
    """One rollout per starting state. The caller supplies `state` and its readout;
    imagination encodes nothing.

    The action chosen at h_t is committed into the next block, so the reward it
    causes is read at lead 0 of the *next* readout (S22). Policy and world noise use
    separate generators, or flow's extra draws would desynchronise the arms.
    """
    bundle = ModelBundle.from_models(config, None, world)
    bundle.require_control()
    readout = heads(agent)
    actions, step_logits, rewards, continuations = [], [], [], []
    values = [_expect(readout["value"][:, -1], heads.centers)]
    readouts = [agent[:, -1]]

    for _ in range(config.horizon):
        logits = readout["policy"][:, -1, 0]
        action = torch.multinomial(logits.softmax(-1), 1, generator=policy_rng).squeeze(-1)
        state, agent = bundle.advance(state, action[:, None], rng)
        readout = heads(agent)

        actions.append(action)
        step_logits.append(logits)
        rewards.append(_expect(readout["reward"][:, -1, 0], heads.centers))
        continuations.append(readout["continuation"][:, -1, 0].sigmoid())
        values.append(_expect(readout["value"][:, -1], heads.centers))
        readouts.append(agent[:, -1])

    return Trajectory(
        action=torch.stack(actions, dim=1),
        logits=torch.stack(step_logits, dim=1),
        reward=torch.stack(rewards, dim=1),
        continuation=torch.stack(continuations, dim=1),
        value=torch.stack(values, dim=1),
        agent=torch.stack(readouts, dim=1),
    )


def _expect(logits: Tensor, centers: Tensor) -> Tensor:
    """Mean of the two-hot distribution, mapped out of symlog space."""
    mean = (logits.softmax(-1) * centers).sum(-1)
    return mean.sign() * torch.expm1(mean.abs())
