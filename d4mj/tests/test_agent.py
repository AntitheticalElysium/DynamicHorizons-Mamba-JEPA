import pytest
import torch

from d4mj.agent import (
    Heads,
    head_loss,
    head_targets,
    paired_terminal_loss,
    twohot,
)

from .conftest import latent_batch


def readout_for(config, batch):
    """The reward and value heads ship zero-initialised, which makes their output
    uniform -- and a uniform distribution has cross-entropy log(bins) against *any*
    target. These tests are about which rows a loss reads, so they need a head whose
    predictions actually vary."""
    torch.manual_seed(0)
    heads = Heads(config)
    generator = torch.Generator().manual_seed(7)
    for head in (heads.reward, heads.value):
        head.weight.data.normal_(0.0, 0.02, generator=generator)
    agent = torch.randn(
        batch.led_to_action.shape[0],
        batch.led_to_action.shape[1],
        config.n_agent,
        config.d_model,
        generator=torch.Generator().manual_seed(1),
    )
    return heads, heads(agent) | {"centers": heads.centers}


def test_behaviour_cloning_uses_the_relevant_half_only(config):
    """§4.1 applies BC to the relevant fraction; reward and continuation see both."""
    half = [True, True, False, False]
    base = latent_batch(config, 4, 6, relevant=half)
    heads, readout = readout_for(config, base)

    moved_uniform = latent_batch(config, 4, 6, relevant=half)
    moved_uniform.led_to_action[2:] = 5
    moved_relevant = latent_batch(config, 4, 6, relevant=half)
    moved_relevant.led_to_action[:2] = 5

    reference = head_loss(readout, head_targets(base, config), config)
    uniform = head_loss(readout, head_targets(moved_uniform, config), config)
    relevant = head_loss(readout, head_targets(moved_relevant, config), config)

    assert torch.equal(reference["policy"], uniform["policy"])
    assert not torch.equal(reference["policy"], relevant["policy"])


def test_reward_head_uses_both_halves(config):
    """The paper restricts BC and dynamics by name and says the mixture amplifies
    signal *for* reward modelling, so halving that head's data would be an addition."""
    half = [True, True, False, False]
    base = latent_batch(config, 4, 6, relevant=half)
    heads, readout = readout_for(config, base)
    for rows in (slice(0, 2), slice(2, 4)):
        moved = latent_batch(config, 4, 6, relevant=half)
        moved.reward[rows] = 7.0
        assert not torch.equal(
            head_loss(readout, head_targets(base, config), config)["reward"],
            head_loss(readout, head_targets(moved, config), config)["reward"],
        )


def test_behaviour_cloning_requires_the_mixture(config):
    """Pretraining batches carry no roles, and BC must not silently score them all."""
    with pytest.raises(AssertionError, match="mixture"):
        head_targets(latent_batch(config, 4, 6), config)


def test_reward_lead_zero_is_the_arriving_reward(config):
    """The reward caused by the action at block t is lead 0 of block t+1, never
    lead 0 of block t."""
    batch = latent_batch(config, 2, 6, relevant=[True, False])
    batch.reward[:] = torch.arange(6).float()
    targets = head_targets(batch, config)
    assert torch.equal(targets["reward"][0, :, 0], torch.arange(6).float())


def test_policy_lead_zero_is_the_outgoing_action(config):
    """Led-to storage puts the outgoing action one block later."""
    batch = latent_batch(config, 2, 6, relevant=[True, False])
    batch.led_to_action[:] = torch.arange(6)
    targets = head_targets(batch, config)
    assert torch.equal(targets["action"][0, :-1, 0], torch.arange(1, 6).float())


def test_continuation_is_one_step(config):
    batch = latent_batch(config, 2, 6, relevant=[True, False])
    batch.terminated[0, 4] = True
    heads, predictions = readout_for(config, batch)
    targets = head_targets(batch, config)
    assert predictions["continuation"].shape == (2, 6, 1)
    assert targets["continuation"].shape == (2, 6, 1)
    assert targets["continuation"][0, 4, 0] == 0


def test_head_output_scales_match_the_pinned_config(config):
    """DreamerV3 ships `rewhead` and `value` at outscale 0.0, `policy` at 0.01 and
    `conhead` at 1.0. A value head starting at random emits random advantages on
    Phase 3's first steps, and PMPO reads only their sign."""
    torch.manual_seed(0)
    heads = Heads(config)
    for head in (heads.reward, heads.value):
        assert float(head.weight.detach().abs().max()) == 0.0
        assert float(head.bias.detach().abs().max()) == 0.0
    policy = float(heads.policy.weight.detach().abs().max())
    continuation = float(heads.continuation.weight.detach().abs().max())
    assert 0.0 < policy < continuation, "policy is scaled down, continuation is not"


def test_zero_initialised_heads_predict_a_flat_distribution(config):
    """The consequence worth stating: a uniform reward head has cross-entropy
    log(bins) against every target, so it starts with no preference at all."""
    import math

    torch.manual_seed(0)
    heads = Heads(config)
    agent = torch.randn(2, 3, config.n_agent, config.d_model, generator=torch.Generator().manual_seed(1))
    logits = heads(agent)["reward"]
    assert torch.allclose(logits, torch.zeros_like(logits))
    entropy = -torch.log_softmax(logits, -1).mean()
    assert abs(float(entropy.detach()) - math.log(config.bins)) < 1e-4


def test_terminal_pair_gives_both_arms_the_same_dead_class_share(config):
    """A tail-wide average would hand each arm a dead-class share of 1/T, so the arm
    training on longer windows is penalised far less for missing a death. Both arms
    use the paired rule, so the share is 1/2 at every window length."""
    for length in (config.sequence, config.sequence_long):
        batch = latent_batch(config, 1, length, relevant=[False], support=[True])
        batch.terminated[:, -1] = True
        targets = head_targets(batch, config)
        valid = targets["continuation_valid"]
        tail_share = float(((1 - targets["continuation"]) * valid).sum() / valid.sum())
        assert tail_share == pytest.approx(1.0 / length)

        selected = torch.zeros_like(valid)
        selected[:, -2:] = valid[:, -2:]
        paired_share = float(((1 - targets["continuation"]) * selected).sum() / selected.sum())
        assert paired_share == pytest.approx(0.5)
    assert config.terminal_loss_mass / 2 == pytest.approx(0.1)


def test_single_path_arm_scores_its_one_readout_once(config):
    """Flow has no generated-prefix readout, so it passes its single readout as both
    arguments. The average must then be that readout's own score, not a doubling."""
    batch = latent_batch(config, 2, 8, relevant=[False, False], support=[True, True])
    batch.terminated[:, -1] = True
    targets = head_targets(batch, config)
    only = {"continuation": torch.randn(targets["continuation"].shape)}
    other = {"continuation": torch.randn(targets["continuation"].shape)}

    alone = paired_terminal_loss(only, only, targets)
    assert torch.allclose(
        paired_terminal_loss(only, other, targets),
        (alone + paired_terminal_loss(other, other, targets)) / 2,
    )


def test_direct_terminal_pair_balances_path_and_label(config):
    batch = latent_batch(config, 2, 8, relevant=[False, False], support=[True, True])
    batch.terminated[:, -1] = True
    targets = head_targets(batch, config)
    shape = targets["continuation"].shape
    generated = {"continuation": torch.zeros(shape)}
    observed = {"continuation": torch.zeros(shape)}

    reference = paired_terminal_loss(generated, observed, targets)
    assert torch.allclose(reference, torch.tensor(2.0).log())
    assert config.terminal_loss_mass / 2 == pytest.approx(0.1)

    correct_generated = {"continuation": generated["continuation"].clone()}
    correct_observed = {"continuation": observed["continuation"].clone()}
    for predictions in (correct_generated, correct_observed):
        predictions["continuation"][:, -2] = 20.0
        predictions["continuation"][:, -1] = -20.0
    assert paired_terminal_loss(correct_generated, correct_observed, targets) < 1e-7

    ignored = {"continuation": generated["continuation"].clone()}
    ignored["continuation"][:, :-2] = 100.0
    assert torch.equal(reference, paired_terminal_loss(ignored, observed, targets))

    generated_only = paired_terminal_loss(correct_generated, observed, targets)
    observed_only = paired_terminal_loss(generated, correct_observed, targets)
    assert torch.allclose(generated_only, observed_only)
    assert generated_only < reference


def test_twohot_is_exact_between_centres(config):
    centers = torch.linspace(-2.0, 2.0, 9)
    values = torch.tensor([[-2.0, 0.0, 0.5, 2.0]])
    weights = twohot(values, centers)
    assert torch.allclose(weights.sum(-1), torch.ones_like(values))
    assert torch.allclose((weights * centers).sum(-1), values, atol=1e-6)


def _predictions(config, rows=2, blocks=6, seed=0):
    from d4mj.agent import Heads
    torch.manual_seed(seed)
    heads = Heads(config)
    agent = torch.randn(rows, blocks, config.n_agent, config.d_model)
    return heads(agent) | {"centers": heads.centers}


def test_position_mask_recombines_to_the_unmasked_loss(config):
    """Masking must only re-weight, never re-pair. Each block's masked loss, weighted by
    its share of the validity mass, has to sum back to the loss over all blocks."""
    from d4mj.agent import head_loss, head_targets
    from .conftest import latent_batch

    blocks = 6
    batch = latent_batch(config, 2, blocks, relevant=[True, False])
    targets = head_targets(batch, config)
    predictions = _predictions(config, 2, blocks)
    whole = head_loss(predictions, targets, config)

    mass = {"policy": targets["action_valid"] * targets["policy_rows"],
            "reward": targets["valid"] * targets["reward_rows"],
            "continuation": targets["continuation_valid"]}
    totals = {name: float(value.sum()) for name, value in mass.items()}
    rebuilt = dict.fromkeys(whole, 0.0)
    for block in range(blocks):
        window = torch.zeros(blocks)
        window[block] = 1.0
        part = head_loss(predictions, targets, config, window)
        for name in whole:
            share = float((mass[name] * window.view(1, -1, 1)).sum())
            if share:
                rebuilt[name] += float(part[name]) * share / totals[name]
    for name, value in whole.items():
        assert abs(rebuilt[name] - float(value)) < 1e-4, (name, rebuilt[name], float(value))


def test_position_mask_scores_each_block_against_its_own_target(config):
    """A masked block's loss must depend on that block's prediction and no other. If
    masking shifted a head's target by one -- the reward and policy leads are offset by
    the led-to convention -- perturbing a neighbour would move the wrong block's loss."""
    from d4mj.agent import head_loss, head_targets
    from .conftest import latent_batch

    blocks, chosen, other = 6, 2, 4
    batch = latent_batch(config, 2, blocks, relevant=[True, False])
    targets = head_targets(batch, config)
    predictions = _predictions(config, 2, blocks)
    window = torch.zeros(blocks)
    window[chosen] = 1.0
    before = head_loss(predictions, targets, config, window)

    for head in ("policy", "reward", "continuation"):
        moved = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in predictions.items()}
        # A single logit, not the whole block: policy and reward are scored through a
        # softmax, so shifting every logit equally leaves the loss untouched.
        moved[head][..., 0][:, other] = moved[head][..., 0][:, other] + 5.0
        after = head_loss(moved, targets, config, window)
        assert torch.allclose(after[head], before[head]), (
            f"masking block {chosen} was affected by a change at block {other} in {head}")

        touched = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in predictions.items()}
        touched[head][..., 0][:, chosen] = touched[head][..., 0][:, chosen] + 5.0
        assert not torch.allclose(head_loss(touched, targets, config, window)[head],
                                  before[head]), f"{head} ignored its own masked block"
