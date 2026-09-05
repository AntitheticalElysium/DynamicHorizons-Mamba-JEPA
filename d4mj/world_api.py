"""One execution boundary for the existing block worlds and joint LeWM world.

Adapters preserve distinct state timing and gradient contracts. Neither adapter
reimplements the other's transition, encoder, or stochastic commit mechanism.
"""

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .config import Config
from .lewm_config import LeWMConfig
from .lewm import LeWMEncoder, LeWMWorld
from .representation import Encoder, pack
from .transition import World
from .data import patchify
from .mamba_recurrence import clone_carry, detach_carry, repeat_carry
from .state import PredictiveState, WorldState, RealState, repeat_memory


@runtime_checkable
class WorldAPI(Protocol):
    def encode(self, frames: Tensor) -> Tensor: ...
    def start(self, z0: Tensor, generator=None, *, first_action=None): ...
    def prefill(self, z_context: Tensor, actions: Tensor, generator=None, *, first_action=None): ...
    def observe(self, state, action: Tensor | None, frame: Tensor, generator=None): ...
    def advance(self, state, action: Tensor, generator=None): ...
    def features(self, state) -> Tensor: ...
    def fork(self, state): ...
    def detach_state(self, state): ...
    def repeat_state(self, state, count: int): ...
    def state_tensors(self, state) -> tuple[Tensor, ...]: ...
    def world_state(self, state): ...


@dataclass
class ModelBundle:
    config: Config | LeWMConfig
    encoder: nn.Module | None
    world: nn.Module

    @classmethod
    def create(cls, config):
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            if isinstance(config, LeWMConfig):
                encoder, world = LeWMEncoder(config), LeWMWorld(config)
            else:
                encoder, world = Encoder(config), World(config)
        device = config.runtime.device if isinstance(config, LeWMConfig) else config.device
        return cls.from_models(config, encoder.to(device), world.to(device))

    @classmethod
    def from_models(cls, config, encoder, world):
        """Wrap existing trained modules without constructing weights or drawing RNG."""
        if isinstance(world, ModelBundle):
            if config != world.config or encoder is not None:
                raise ValueError("bundle contract differs from its supplied modules/config")
            return world
        if world.config != config:
            raise ValueError("world recipe differs from bundle recipe")
        if isinstance(config, LeWMConfig):
            if not isinstance(world, LeWMWorld) or not isinstance(encoder, LeWMEncoder):
                raise TypeError("LeWM bundle requires its encoder and world")
            return LeWMWorldAdapter(config, encoder, world)
        if not isinstance(world, World) or (encoder is not None and not isinstance(encoder, Encoder)):
            raise TypeError("legacy bundle requires the existing block world and MAE encoder")
        return LegacyWorldAdapter(config, encoder, world)

    @property
    def device(self):
        return next(self.world.parameters()).device

    @property
    def n_actions(self):
        return self.config.dynamics.n_actions if isinstance(self.config, LeWMConfig) else self.config.n_actions

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        self.world.eval()
        return self

    def require_control(self):
        if isinstance(self.config, LeWMConfig):
            raise RuntimeError("phase_gate: LeWM M0-M3 has no trained heads/readout or validated actor horizon")

    def world_state(self, state):
        # Validation happens through each adapter before unwrapping observation state.
        self.state_tensors(state)
        return state.world if isinstance(state, RealState) else state


class LegacyWorldAdapter(ModelBundle):
    """Current-frame-inclusive memory; retain the legacy S55 detach policy."""

    def _state(self, state):
        state = state.world if isinstance(state, RealState) else state
        if not isinstance(state, WorldState):
            raise TypeError("legacy worlds require WorldState or RealState")
        return state

    def _generator(self, generator):
        if generator is None and self.config.transition == "flow":
            raise ValueError("flow requires an explicit world RNG")
        return generator

    def _action(self, action, batch):
        if action.dtype != torch.long or action.shape != (batch,1):
            raise ValueError("outgoing action must be int64 B,1")
        if bool(((action < 0) | (action >= self.n_actions)).any()):
            raise ValueError("BOS is not an outgoing action")

    def encode(self, frames):
        if self.encoder is None or self.encoder.training:
            raise RuntimeError("observation_normalization: runtime encoding requires encoder.eval()")
        z, _, _ = self.encoder(patchify(frames, self.config.patch).to(self.device))
        return pack(z,self.config)

    def start(self, z0, generator=None, *, first_action=None):
        if z0.shape[1] != 1:
            raise ValueError("start takes exactly one initial latent")
        return self.prefill(z0,torch.empty(z0.shape[0],0,dtype=torch.long,device=z0.device),
                            generator,first_action=first_action)

    def prefill(self, z_context, actions, generator=None, *, first_action=None):
        from .transition import initial
        if actions.dtype != torch.long or actions.shape != (z_context.shape[0],z_context.shape[1]-1):
            raise ValueError("prefill requires T latents and T-1 outgoing actions")
        if bool(((actions < 0) | (actions >= self.n_actions)).any()):
            raise ValueError("BOS is not an outgoing action")
        first = (torch.full((z_context.shape[0],1),self.n_actions,dtype=torch.long,device=z_context.device)
                 if first_action is None else first_action)
        if first.shape != (z_context.shape[0],1) or first.dtype != torch.long:
            raise ValueError("first incoming action must be int64 B,1")
        if bool(((first < 0) | (first > self.n_actions)).any()):
            raise ValueError("invalid first incoming action")
        state, _ = initial(self.world,z_context,torch.cat((first,actions),1),
                           self._generator(generator),self.config)
        return replace(state,latent=state.latent[:,-1:].clone(),features=state.features[:,-1:].clone())

    def observe(self, state, action, frame, generator=None):
        from .transition import observe
        if self.encoder is None:
            raise ValueError("observing pixels requires an encoder")
        if state is None:
            if action is not None:
                raise ValueError("initial observation takes no outgoing action")
            action = torch.full((frame.shape[0],1),self.n_actions,dtype=torch.long,device=self.device)
        else:
            if not isinstance(state, RealState):
                raise TypeError("legacy pixel observation needs RealState with encoder memory")
            self._action(action,frame.shape[0])
        if self.encoder.training:
            raise RuntimeError("observation_normalization: runtime encoding requires encoder.eval()")
        return observe(self.world,self.encoder,state,action,patchify(frame,self.config.patch).to(self.device),
                       self._generator(generator),self.config)

    def advance(self, state, action, generator=None):
        from .transition import advance
        state = self._state(state)
        self._action(action,state.latent.shape[0])
        return advance(self.world,state,action,self._generator(generator),self.config)

    def features(self, state):
        state = self._state(state)
        if state.features is None:
            raise ValueError("legacy state has no committed features")
        return state.features[:,:,self.world.agent]

    def _map_state(self, state, tensor, memory):
        current = self._state(state)
        result = replace(current,latent=tensor(current.latent),memory=memory(current.memory),
                         features=None if current.features is None else tensor(current.features))
        return RealState(memory(state.encoder_memory),result) if isinstance(state,RealState) else result

    def fork(self, state):
        def memory(values):
            return None if values is None else tuple(tuple(t.clone() for t in pair) for pair in values)
        return self._map_state(state,lambda t:t.clone(),memory)

    def detach_state(self, state):
        def memory(values):
            return None if values is None else tuple(tuple(t.detach().clone() for t in pair) for pair in values)
        return self._map_state(state,lambda t:t.detach().clone(),memory)

    def repeat_state(self, state, count):
        if count < 1:
            raise ValueError("repeat count must be positive")
        batch = self._state(state).latent.shape[0]
        return self._map_state(state,lambda t:t.repeat_interleave(count,0),lambda m:repeat_memory(m,batch,count))

    def state_tensors(self, state):
        current = self._state(state)
        tensors = [current.latent]
        if current.features is not None:
            tensors.append(current.features)
        memories = [current.memory]
        if isinstance(state,RealState):
            memories.append(state.encoder_memory)
        for memory in memories:
            if memory is not None:
                tensors.extend(t for pair in memory for t in pair)
        return tuple(tensors)


class LeWMWorldAdapter(ModelBundle):
    """Completed-pair memory with differentiable conv and SSM carry."""

    def encode(self, frames: Tensor) -> Tensor:
        if self.encoder.training:
            raise RuntimeError("observation_normalization: runtime encoding requires encoder.eval()")
        return self.encoder(frames)

    def start(self, z0: Tensor, generator=None, *, first_action=None) -> PredictiveState:
        if first_action is not None:
            raise ValueError("LeWM start consumes no incoming action")
        return self.world.start(z0)

    def prefill(self, z_context: Tensor, actions: Tensor, generator=None, *, first_action=None) -> PredictiveState:
        if first_action is not None:
            raise ValueError("LeWM prefill consumes completed outgoing pairs only")
        self.world._streaming_mode()
        return self.world.teacher(z_context, actions).state

    def observe(self, state: PredictiveState | None, action: Tensor | None, frame: Tensor, generator=None):
        if state is None:
            if action is not None:
                raise ValueError("initial observation takes no outgoing action")
            state = self.start(self.encode(frame))
            return state, self.features(state)
        return self.world.observe_latent(state, action, self.encode(frame))

    def advance(self, state: PredictiveState, action: Tensor, generator=None):
        # Deterministic transition; policy/projection randomness never shares this path.
        return self.world.advance(state, action)

    def features(self, state: PredictiveState) -> Tensor:
        return self.world.features(state)

    def fork(self, state: PredictiveState) -> PredictiveState:
        self.world.validate_state(state)
        return PredictiveState(state.latent.clone(), tuple(clone_carry(m) for m in state.memory),
                               state.history.clone(), state.step)

    def detach_state(self, state: PredictiveState) -> PredictiveState:
        self.world.validate_state(state)
        return PredictiveState(state.latent.detach().clone(), tuple(detach_carry(m) for m in state.memory),
                               state.history.detach().clone(), state.step)

    def repeat_state(self, state: PredictiveState, count: int) -> PredictiveState:
        self.world.validate_state(state)
        if count < 1:
            raise ValueError("repeat count must be positive")
        return PredictiveState(state.latent.repeat_interleave(count, 0),
                               tuple(repeat_carry(m, count) for m in state.memory),
                               state.history.repeat_interleave(count, 0), state.step)

    def state_tensors(self, state: PredictiveState) -> tuple[Tensor, ...]:
        self.world.validate_state(state)
        return (state.latent, state.history, *(v for m in state.memory for v in (m.conv, m.ssm)))


def load_bundle(path, *, device: str | None = None, backend: str | None = None):
    """Inference load with explicit device/backend overrides, never a training resume."""
    from dataclasses import replace
    from .checkpoint import read_lewm_bundle
    from .config import config_from_dict

    payload = read_lewm_bundle(path)
    config = config_from_dict(payload["config"])
    if device is not None or backend is not None:
        config = replace(config, runtime=replace(config.runtime, device=device or config.runtime.device),
                         dynamics=replace(config.dynamics, backend=backend or config.dynamics.backend))
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"], strict=True)
    bundle.world.load_state_dict(payload["modules"]["world"], strict=True)
    bundle.encoder.freeze()
    bundle.world.requires_grad_(False)
    return bundle.eval(), payload
