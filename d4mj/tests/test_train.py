from dataclasses import replace

import pytest
import torch

from d4mj.checkpoint import load, save
from d4mj.train import (
    _checkpoint,
    _generators,
    _share_initialisation,
    _terminal_path,
    optimizer,
)
from d4mj.transition import World


def test_shared_initialisation_matches_every_common_tensor(config):
    """`manual_seed` alone does not: the mixers consume different numbers of draws
    at construction, so everything built after the first time layer diverged."""
    def build(mixer, share):
        arm = replace(config, time_mixer=mixer)
        torch.manual_seed(arm.seed + 1)
        world = World(arm)
        return (_share_initialisation(world, arm) if share else world).state_dict()

    for share, expect_all in ((False, False), (True, True)):
        attention, mamba = build("attention", share), build("mamba", share)
        common = [k for k in attention if k in mamba and attention[k].shape == mamba[k].shape]
        identical = [k for k in common if torch.equal(attention[k], mamba[k])]
        assert common
        assert (len(identical) == len(common)) is expect_all


def test_terminal_supervision_uses_the_phase_two_transition_path(config, monkeypatch):
    from d4mj import train
    from .conftest import latent_batch

    batch = latent_batch(config, 1, 6, relevant=[False], support=[True])
    sentinel = torch.randn(1, 6, config.n_agent, config.d_model)
    called = {}

    def transition(
        world,
        received,
        rng,
        received_config,
        return_agent=False,
        return_observed=False,
        step=0,
    ):
        called.update(
            batch=received,
            return_agent=return_agent,
            return_observed=return_observed,
            step=step,
        )
        return torch.zeros(()), sentinel, sentinel

    monkeypatch.setattr(train, "transition_loss", transition)
    _, got, _ = _terminal_path(
        World(config),
        batch,
        torch.Generator().manual_seed(0),
        config,
        step=123,
        score_dynamics=False,
        return_observed=True,
    )
    assert got is sentinel
    assert called == {
        "batch": batch,
        "return_agent": True,
        "return_observed": True,
        "step": 123,
    }


def test_terminal_dynamics_removes_only_the_support_mask(config, monkeypatch):
    from d4mj import train
    from .conftest import latent_batch

    batch = latent_batch(config, 1, 6, relevant=[False], support=[True])
    sentinel_loss = torch.tensor(3.0)
    sentinel_agent = torch.randn(1, 6, config.n_agent, config.d_model)
    called = {}

    def transition(
        world,
        received,
        rng,
        received_config,
        return_agent=False,
        return_observed=False,
        step=0,
    ):
        called.update(
            batch=received,
            return_agent=return_agent,
            return_observed=return_observed,
            step=step,
        )
        return sentinel_loss, sentinel_agent

    monkeypatch.setattr(train, "transition_loss", transition)
    loss, agent = _terminal_path(
        World(config),
        batch,
        torch.Generator().manual_seed(0),
        config,
        step=321,
        score_dynamics=True,
    )

    assert loss is sentinel_loss and agent is sentinel_agent
    assert called["batch"].support is None
    assert torch.equal(called["batch"].relevant, batch.relevant)
    assert torch.equal(called["batch"].led_to_action, batch.led_to_action)
    assert called["batch"].rows("dynamics").all()
    assert not called["batch"].rows("policy").any()
    assert called["return_agent"] is True
    assert called["return_observed"] is False
    assert called["step"] == 321


def test_resume_restores_optimizer_normalisers_and_streams(config, tmp_path):
    """Restoring weights alone restarts every normaliser and replays every window
    and noise draw from zero, which is not a resume."""
    path = tmp_path / "phase.pt"
    torch.manual_seed(config.seed + 1)
    world = World(config)
    optimiser = optimizer([world], config)
    balance = {"dynamics": 4.0}
    sampler, rng = _generators(config, 1)

    world.readout.weight.data.add_(0.5)
    for group in optimiser.param_groups:
        group["lr"] = 1e-9
    sampler.manual_seed(99)
    saved_sampler = sampler.get_state().clone()
    saved_weight = world.readout.weight.detach().clone()
    streams = {"sampler": sampler, "model": rng}
    _checkpoint(path, config, [world, optimiser], balance, streams, step=7)

    torch.manual_seed(config.seed + 1)
    other = World(config)
    other_optimiser = optimizer([other], config)
    other_balance: dict[str, float] = {}
    other_sampler, other_rng = _generators(config, 1)
    step = _checkpoint(
        path,
        config,
        [other, other_optimiser],
        other_balance,
        {"sampler": other_sampler, "model": other_rng},
    )

    assert step == 7
    assert other_balance == balance
    assert torch.equal(other.readout.weight.detach(), saved_weight)
    assert torch.equal(other_sampler.get_state(), saved_sampler)


def test_phase_three_resumes_its_policy_stream(config, tmp_path):
    """Phase 3 draws from three streams. Dropping the policy one resumes with the
    actor sampling a different action sequence, which no loss curve would reveal."""
    path = tmp_path / "actor.pt"
    torch.manual_seed(0)
    world = World(config)
    optimiser = optimizer([world], config)
    sampler, rng = _generators(config, 3)
    policy = torch.Generator().manual_seed(config.seed + 2**20)
    torch.randint(config.n_actions, (11,), generator=policy)
    saved_policy = policy.get_state().clone()
    streams = {"sampler": sampler, "model": rng, "policy": policy}
    _checkpoint(path, config, [world, optimiser], {}, streams, step=3)

    restored = torch.Generator().manual_seed(0)
    _checkpoint(
        path,
        config,
        [world, optimiser],
        {},
        {"sampler": sampler, "model": rng, "policy": restored},
    )
    assert torch.equal(restored.get_state(), saved_policy)


def test_resume_rejects_a_different_frozen_model(config, tmp_path):
    """Resuming an actor against a different world silently changes the environment
    it is scored inside; `Config` matching cannot see it."""
    from d4mj.train import _identity

    path = tmp_path / "actor.pt"
    torch.manual_seed(0)
    world = World(config)
    optimiser = optimizer([world], config)
    sampler, rng = _generators(config, 3)
    streams = {"sampler": sampler, "model": rng}
    _checkpoint(path, config, [world, optimiser], {}, streams, step=3, contract=_identity(world))

    torch.manual_seed(1)
    other = World(config)
    with pytest.raises(ValueError, match="does not match"):
        _checkpoint(path, config, [world, optimiser], {}, streams, contract=_identity(other))


def test_resume_rejects_a_different_phase_length(config, tmp_path):
    """The planned total sets the short/long schedule and the long-only tail, so
    resuming with a different one silently changes the data the phase sees."""
    path = tmp_path / "phase.pt"
    torch.manual_seed(0)
    world = World(config)
    optimiser = optimizer([world], config)
    sampler, rng = _generators(config, 1)
    streams = {"sampler": sampler, "model": rng}
    _checkpoint(path, config, [world, optimiser], {}, streams, step=3, contract="1B:1000")
    with pytest.raises(ValueError, match="does not match"):
        _checkpoint(path, config, [world, optimiser], {}, streams, contract="1B:2000")


def test_phase_two_resume_rejects_a_different_world_clock(config, tmp_path):
    path = tmp_path / "phase2.pt"
    world = World(config)
    optimiser = optimizer([world], config)
    sampler, rng = _generators(config, 2)
    streams = {"sampler": sampler, "model": rng}
    _checkpoint(path, config, [world, optimiser], {}, streams, step=3, contract="2:20000:10000")
    with pytest.raises(ValueError, match="does not match"):
        _checkpoint(path, config, [world, optimiser], {}, streams, contract="2:10000:10000")


def test_phase_checkpoints_its_final_step(config, tmp_path, episodes):
    """A phase whose length is not a multiple of `checkpoint_every` must still save
    what it trained. Firing only on the modulus loses the final model outright."""
    from d4mj.train import train_dynamics

    path = tmp_path / "phase.pt"
    steps = replace(config, checkpoint_every=500)
    cached = [
        replace_episode(e, config)
        for e in episodes
    ]
    train_dynamics(cached, 3, steps, checkpoint=path)
    assert path.exists()
    assert torch.load(path, weights_only=False)["modules"]["step"] == 3


def replace_episode(episode, config):
    """A cached episode, so `train_dynamics` runs without a tokenizer."""
    from dataclasses import replace as _replace

    latents = torch.zeros(len(episode) + 1, config.n_spatial, config.d_spatial)
    return _replace(episode, latents=latents, latent_digest="test")


def test_resume_from_absent_checkpoint_starts_at_zero(config, tmp_path):
    torch.manual_seed(0)
    world = World(config)
    sampler, rng = _generators(config, 1)
    streams = {"sampler": sampler, "model": rng}
    assert _checkpoint(tmp_path / "absent.pt", config, [world], {}, streams) == 0
    assert _checkpoint(None, config, [world], {}, streams) == 0


def test_checkpoint_predating_a_field_loads_at_its_default(config, tmp_path):
    """A field added to `Config` is absent from every checkpoint written before it, and
    a whole-dict comparison would reject all of them. The migration must accept those at
    the default and still refuse to load one as though it had trained with the feature --
    otherwise an unaligned world would silently pass as an aligned arm."""
    path = tmp_path / "old.pt"
    world = World(replace(config, transition="direct"))
    save(path, replace(config, transition="direct"), part0=world)
    payload = torch.load(path, weights_only=False)
    del payload["config"]["align_weight"]
    torch.save(payload, path)

    load(path, replace(config, transition="direct"), part0=World(replace(config, transition="direct")))
    with pytest.raises(ValueError, match="predates"):
        aligned = replace(config, transition="direct", align_weight=0.5)
        load(path, aligned, part0=World(aligned))
