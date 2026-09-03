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


def _paired_probe(config, monkeypatch, paired):
    """Run one `train_agent` step with the transition stubbed, counting how many times
    the head losses are scored and on which readout."""
    from d4mj import train
    from d4mj.agent import head_loss as real_head_loss

    rows, blocks = 2, config.sequence
    generated = torch.randn(rows, blocks, config.n_agent, config.d_model)
    observed = torch.randn(rows, blocks, config.n_agent, config.d_model)

    # The terminal stratum calls `transition_loss` too and scores its own observed
    # readout. Returning the same sentinels for both would make the terminal path look
    # like the main one, so it gets its own pair.
    from .conftest import latent_batch
    terminal_batch = latent_batch(config, rows, blocks, relevant=[True, False])
    tail = torch.randn(rows, blocks, config.n_agent, config.d_model)

    def transition(world, batch, rng, cfg, return_agent=False, return_observed=False, step=0):
        assert return_agent and return_observed, "paired supervision needs both readouts"
        zero = torch.zeros((), requires_grad=True)
        #  rebuilds the Batch, so the dataclass identity is gone by here; the
        # latents tensor survives it, because .to() returns self when already on device.
        if batch.latents is terminal_batch.latents:
            return zero, tail, tail
        return zero, generated, observed

    seen, targets_seen = [], []

    def head_loss(predictions, targets, cfg):
        targets_seen.append(id(targets))
        return {name: predictions["value"].float().mean()
                for name in ("policy", "reward", "continuation")}

    # Record what actually reaches the heads. Their output cannot discriminate here --
    # the value head is zero-initialised, so both readouts score identically at step 0.
    from d4mj.agent import Heads
    forward = Heads.forward
    monkeypatch.setattr(Heads, "forward",
                        lambda self, agent: (seen.append(agent), forward(self, agent))[1])

    # The conftest episodes carry no terminations, so the terminal stratum cannot be
    # sampled from them. It is stubbed out: this probe is about the main path.
    monkeypatch.setattr(train, "transition_loss", transition)
    monkeypatch.setattr(train, "head_loss", head_loss)
    monkeypatch.setattr(train, "sample_terminal_batch", lambda *a, **k: terminal_batch)
    monkeypatch.setattr(train, "paired_terminal_loss",
                        lambda *a, **k: torch.zeros((), requires_grad=True))
    return train, seen, targets_seen, generated, observed


def test_paired_semantic_scores_both_readouts_on_one_set_of_targets(config, monkeypatch, episodes):
    """Phase 1B's raw readout MSE was satisfiable by collapsing both sides into a shared
    subspace. Paired supervision judges each readout against real targets instead, so what
    must hold is that BOTH are scored and that they are scored on the SAME targets."""
    train, seen, targets_seen, generated, observed = _paired_probe(config, monkeypatch, True)
    train.train_agent(episodes, World(config), 1, config, paired_semantic=True)
    assert any(x is generated for x in seen), "the generated readout never reached the heads"
    assert any(x is observed for x in seen), "the observed readout never reached the heads"
    assert len(targets_seen) == 2, f"expected both readouts scored, saw {len(targets_seen)}"
    assert targets_seen[0] == targets_seen[1], "the two paths saw different targets"


def test_paired_semantic_off_scores_only_the_generated_readout(config, monkeypatch, episodes):
    train, seen, targets_seen, generated, observed = _paired_probe(config, monkeypatch, False)
    train.train_agent(episodes, World(config), 1, config, paired_semantic=False)
    assert len(targets_seen) == 1, f"the default must score one readout, saw {len(targets_seen)}"
    assert not any(x is observed for x in seen), "the observed readout was scored by default"


def test_frozen_world_updates_only_the_heads(config, monkeypatch, episodes):
    """The head ceiling is only a ceiling if the world is genuinely held fixed: a
    difference between two arms must not be attributable to the world or to its
    interaction with the heads."""
    train, seen, _, _, _ = _paired_probe(config, monkeypatch, True)
    world = World(config)
    before = {name: parameter.detach().clone() for name, parameter in world.named_parameters()}
    train.train_agent(episodes, world, 1, config, paired_semantic=True, freeze_world=True)
    assert all(torch.equal(before[name], parameter)
               for name, parameter in world.named_parameters()), "the world moved"
    assert not any(parameter.requires_grad for parameter in world.parameters())


def test_phase_two_resume_rejects_a_different_paired_setting(config, tmp_path):
    """`paired_semantic` and `freeze_world` change what the run optimises, so a
    checkpoint must not silently resume under the opposite one."""
    from d4mj import train

    contracts = set()
    original = train._checkpoint

    def record(path, cfg, bundle, balance, streams, step=None, contract=""):
        contracts.add(contract)
        return 1 if step is None else None

    train._checkpoint = record
    try:
        for paired in (False, True):
            for frozen in (False, True):
                train.train_agent([], World(config), 1, config,
                                  paired_semantic=paired, freeze_world=frozen)
    finally:
        train._checkpoint = original
    assert len(contracts) == 4, sorted(contracts)
