from dataclasses import replace

import pytest
import torch

import d4mj.transition as transition
from d4mj.agent import Heads, head_targets, paired_terminal_loss
from d4mj.transition import World, commit_inputs, flow_conditioning, transition_loss

from .conftest import latent_batch

ARMS = ("flow", "direct")


def world_for(config, transition, seed=0):
    torch.manual_seed(seed)
    return World(replace(config, transition=transition))


def loss_of(world, batch, config, seed=3):
    return transition_loss(world, batch, torch.Generator().manual_seed(seed), config)


@pytest.mark.parametrize("transition", ARMS)
def test_pretraining_scores_every_row(config, transition):
    """`relevant is None` is the pretraining regime: restricting it to uniform rows
    would discard half the corpus D4 pretrains on."""
    config = replace(config, transition=transition)
    world = world_for(config, transition)
    base = latent_batch(config, 4, 6)
    moved = latent_batch(config, 4, 6)
    moved.latents[:2] = torch.randn_like(moved.latents[:2]).tanh()
    assert not torch.equal(loss_of(world, base, config), loss_of(world, moved, config))


@pytest.mark.parametrize("transition", ARMS)
def test_finetuning_dynamics_ignores_the_relevant_half(config, transition):
    """§4.1 applies the continued dynamics loss to uniform rows only, so that
    imagination is not fitted to task-accomplishing play."""
    config = replace(config, transition=transition)
    world = world_for(config, transition)
    half = [True, True, False, False]
    base = latent_batch(config, 4, 6, relevant=half)
    touched_relevant = latent_batch(config, 4, 6, relevant=half)
    touched_relevant.latents[:2] = torch.randn_like(touched_relevant.latents[:2]).tanh()
    touched_uniform = latent_batch(config, 4, 6, relevant=half)
    touched_uniform.latents[2:] = torch.randn_like(touched_uniform.latents[2:]).tanh()

    assert torch.equal(loss_of(world, base, config), loss_of(world, touched_relevant, config))
    assert not torch.equal(loss_of(world, base, config), loss_of(world, touched_uniform, config))


def test_direct_commits_both_generated_states(config):
    """Both generated readouts must reach the heads at their own indices. Predicting
    the second latent without committing it leaves the heads trained on one
    generated state while Phase 3 reads them after every generated state."""
    config = replace(config, transition="direct")
    world = world_for(config, "direct").eval()
    blocks = 6
    batch = latent_batch(config, 2, blocks)
    with torch.no_grad():
        _, readout, observed = transition_loss(
            world,
            batch,
            torch.Generator().manual_seed(3),
            config,
            return_agent=True,
            return_observed=True,
        )
        committed, conditioning = commit_inputs(
            batch.latents, torch.Generator().manual_seed(3), config
        )
        real = world(None, batch.led_to_action, committed, conditioning)[1]
    assert torch.equal(observed, real)
    differs = [not torch.equal(readout[:, i], real[:, i]) for i in range(blocks)]
    assert differs == [i >= blocks - 2 for i in range(blocks)]


def test_direct_paired_terminal_gradient_reaches_predictor(config):
    config = replace(config, transition="direct", gradient_checkpointing=False)
    world = world_for(config, "direct")
    heads = Heads(config)
    batch = latent_batch(config, 1, 4, relevant=[False], support=[True])
    batch.terminated[:, -1] = True

    _, generated, observed = transition_loss(
        world,
        batch,
        torch.Generator().manual_seed(3),
        config,
        return_agent=True,
        return_observed=True,
    )
    loss = paired_terminal_loss(
        heads(generated), heads(observed), head_targets(batch, config)
    )
    gradients = torch.autograd.grad(
        loss, [world.readout.weight, world.direct_mixer.self_attn.in_proj_weight]
    )
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_direct_mixes_the_candidate_action_as_a_token(config):
    """The repaired Direct head must expose the candidate action before token
    interaction, not broadcast it after the spatial features have been mixed."""
    config = replace(config, transition="direct")
    world = world_for(config, "direct").eval()
    features = torch.randn(2, 3, world.layout.size, config.d_model)
    actions = torch.randint(config.n_actions, (2, 3))
    captured = []

    handle = world.direct_mixer.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    with torch.no_grad():
        predicted = world.predict(features, actions)
    handle.remove()

    tokens = captured[0].view(2, 3, config.n_spatial + 1, config.d_model)
    spatial_and_register = torch.cat(
        [features[:, :, world.spatial], features[:, :, world.register]], dim=2
    )
    pooled = world.pool(spatial_and_register.transpose(2, 3)).transpose(2, 3)
    assert torch.equal(tokens[:, :, 0], world.action_embed(actions))
    assert torch.equal(tokens[:, :, 1:], pooled)
    assert predicted.shape == (2, 3, config.n_spatial, config.d_spatial)
    assert (predicted.abs() <= 1).all()


def test_observed_readout_requires_agent_return(config):
    config = replace(config, transition="direct")
    with pytest.raises(ValueError, match="requires return_agent"):
        transition_loss(
            world_for(config, "direct"),
            latent_batch(config, 1, 4),
            torch.Generator().manual_seed(3),
            config,
            return_observed=True,
        )


def test_commit_prefix_does_not_reweight_the_step_grid(config):
    """A prefix row exists to give its last block a runtime-like history. Scoring
    the prefix too would shift the supervised share away from what S67 sets."""
    config = replace(config, transition="flow")
    finest = total = 0
    for seed in range(64):
        conditioning, scored = flow_conditioning(
            torch.Generator().manual_seed(seed), (8, 32), config, "cpu", bootstrap=True
        )
        keep = scored > 0
        total += int(keep.sum())
        finest += int(((conditioning[..., 1] == config.step_index) & keep).sum())
    assert abs(finest / total - (1 - config.self_fraction)) < 0.05


def test_shortcut_self_rows_wait_for_bootstrap_start(config):
    config = replace(config, transition="flow", commit_prefix_fraction=0.0)
    before, before_scored = flow_conditioning(
        torch.Generator().manual_seed(0), (8, 16), config, "cpu", False
    )
    after, after_scored = flow_conditioning(
        torch.Generator().manual_seed(0), (8, 16), config, "cpu", True
    )
    assert torch.equal(before, after), "warmup must not change which rows are self rows"
    self_rows = (before[..., 1] != config.step_index).any(dim=1)
    assert int(self_rows.sum()) == round(config.self_fraction * 8)
    assert not before_scored[self_rows].any(), "self targets are inactive during warmup"
    assert before_scored[~self_rows].all() and after_scored.all()


def test_shortcut_partition_is_independent_of_mixture_roles(config):
    config = replace(config, transition="flow", commit_prefix_fraction=0.0)
    dynamics_rows = torch.tensor([False, False, True, True])
    self_scored = total_scored = 0
    for seed in range(256):
        conditioning, scored = flow_conditioning(
            torch.Generator().manual_seed(seed), (4, 8), config, "cpu", True
        )
        self_rows = (conditioning[..., 1] != config.step_index).any(dim=1)
        self_scored += int((self_rows & dynamics_rows).sum())
        total_scored += int(dynamics_rows.sum())
        assert scored.all()
    assert abs(self_scored / total_scored - config.self_fraction) < 0.04


def test_shortcut_bootstrap_target_starts_at_the_exact_boundary(config, monkeypatch):
    config = replace(config, transition="flow", commit_prefix_fraction=0.0)
    world = world_for(config, "flow")
    batch = latent_batch(config, 4, 6)
    original, calls = transition._bootstrap_target, []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(transition, "_bootstrap_target", counted)
    loss_of(world, batch, config, seed=4)  # transition_loss defaults to step zero.
    assert not calls
    transition_loss(
        world,
        batch,
        torch.Generator().manual_seed(4),
        config,
        step=config.bootstrap_start,
    )
    assert len(calls) == 1


def test_commit_prefix_rows_carry_the_commit_condition(config):
    config = replace(config, transition="flow")
    conditioning, scored = flow_conditioning(
        torch.Generator().manual_seed(0), (8, 32), config, "cpu", bootstrap=True
    )
    prefix = (scored == 0).any(dim=1)
    assert prefix.any(), "no row carried the rollout prefix"
    for row in prefix.nonzero().flatten():
        assert (conditioning[row, :-1, 0] == config.tau_ctx_index).all()
        assert (conditioning[row, :-1, 1] == config.step_index).all()
        assert bool(scored[row, -1]), "the prefix row's target block must be scored"


@pytest.mark.parametrize("transition", ARMS)
def test_loss_is_finite(config, transition):
    config = replace(config, transition=transition)
    assert torch.isfinite(loss_of(world_for(config, transition), latent_batch(config, 4, 6), config))


def test_alignment_is_inert_at_zero_and_linear_in_its_weight(config):
    """`align_weight` is an additive coefficient, so the loss must move by exactly the
    weight times one fixed quantity. Linearity is what distinguishes that from a term
    that also perturbs the rollout or the teacher-forcing path."""
    config = replace(config, transition="direct")
    world = world_for(config, "direct").eval()
    batch = latent_batch(config, 4, 6)
    with torch.no_grad():
        # A list, not a dict: keying by weight collapsed the two zero-weight calls into
        # one entry and turned the reproducibility check into `x == x`.
        repeated = [float(loss_of(world, batch, replace(config, align_weight=0.0)))
                    for _ in range(2)]
        losses = {weight: float(loss_of(world, batch, replace(config, align_weight=weight)))
                  for weight in (0.0, 0.5, 1.0)}
    assert repeated[0] == repeated[1], "the zero-weight objective is not reproducible"
    assert repeated[0] == losses[0.0]
    assert losses[0.5] > losses[0.0], "alignment contributed nothing at a nonzero weight"
    assert abs((losses[1.0] - losses[0.0]) - 2 * (losses[0.5] - losses[0.0])) < 1e-6


def test_alignment_detaches_the_observed_side(config):
    """Dreamer 3 aligns its prior to a stopped posterior. If the observed readout were
    left attached, the world could satisfy the term by moving the observed side toward
    the generated one -- the opposite of the intent -- and the gradient would differ
    from the stop-gradient formulation."""
    config = replace(config, transition="direct", gradient_checkpointing=False)
    weight, world = 0.5, world_for(config, "direct")
    batch = latent_batch(config, 4, 6)

    def gradient(loss):
        world.zero_grad(set_to_none=True)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in world.parameters() if p.grad is not None])

    combined = gradient(loss_of(world, batch, replace(config, align_weight=weight)))

    plain, readout, observed = transition.transition_loss(
        world, batch, torch.Generator().manual_seed(3), replace(config, align_weight=0.0),
        return_agent=True, return_observed=True,
    )
    start = batch.latents.shape[1] - config.direct_rollout
    stopped = (readout[:, start:] - observed[:, start:].detach()).pow(2).mean(dim=(1, 2, 3))
    manual = gradient(plain + weight * transition._uniform_mean(stopped, batch))
    assert torch.allclose(combined, manual, atol=1e-6)


def test_alignment_gradient_reaches_the_world(config):
    """The term must train the generated side, not merely register in the scalar."""
    config = replace(config, transition="direct", gradient_checkpointing=False)
    world = world_for(config, "direct")
    batch = latent_batch(config, 4, 6)

    def gradient(weight):
        world.zero_grad(set_to_none=True)
        loss_of(world, batch, replace(config, align_weight=weight)).backward()
        return {name: p.grad.clone() for name, p in world.named_parameters()
                if p.grad is not None}

    plain, aligned = gradient(0.0), gradient(0.5)
    moved = [name for name in aligned if not torch.equal(aligned[name], plain[name])]
    assert moved, "alignment changed no gradient"
    # Named separately: an `any` over both groups passes on the backbone alone, which
    # would not show that the term reaches the parts that produce the generated readout.
    for group in ("readout", "direct_mixer", "pool", "backbone"):
        assert any(group in name for name in moved), (group, sorted(moved)[:8])
