import pytest
import torch

from d4mj.m03_capability import (
    M03Settings,
    REPLAY_PIXEL_TOLERANCE,
    _binary_metrics,
    _direct_first_incoming_action,
    _fit_probe_many,
    _load_or_compute_stage,
    _mode_summary,
    _paired_binary_difference,
    _recorded_step_key,
    _ridge_predict,
)


def _settings():
    return M03Settings(train_roots=2, dev_roots=2, probe_hidden=4, probe_steps=2,
                       probe_batch=4, bootstrap_draws=8, minimum_positive=1,
                       minimum_negative=1)


def test_settings_reject_an_incompatible_direct_context():
    with pytest.raises(ValueError, match="context"):
        M03Settings(direct_context=16, lewm_context=4)
    assert REPLAY_PIXEL_TOLERANCE == 1


def test_recorded_step_key_uses_the_logged_outgoing_transition_key():
    class Replay:
        @staticmethod
        def _slot_keys(shard, slot):
            assert (shard, slot) == (7, 3)
            return "reset", ("step-0", "step-1", "step-2")

    assert _recorded_step_key(Replay(), {"shard": 7, "slot": 3, "t": 2}) == "step-2"


def test_direct_prefix_keeps_its_real_first_incoming_action():
    actions = torch.tensor([4, 2, 7]).numpy()
    assert _direct_first_incoming_action(actions, 0) == 17
    assert _direct_first_incoming_action(actions, 2) == 2


def test_ridge_control_returns_cpu_predictions_from_its_input_device():
    train_x = torch.tensor([[0.0], [1.0], [2.0]])
    train_y = torch.tensor([[0.0], [1.0], [2.0]])
    prediction = _ridge_predict(train_x, train_y, torch.tensor([[1.5]]), ridge=1.0)
    assert prediction.device.type == "cpu"
    assert prediction.shape == (1, 1)


def test_binary_metrics_bootstraps_one_root_and_all_action_targets_correctly():
    settings = _settings()
    # Static labels have just one value per root; all-action labels have 17.
    # Both must retain episode, rather than fork, as the resampling unit.
    static = _binary_metrics(torch.tensor([[3.0], [-3.0]]), torch.tensor([[True], [False]]),
                             torch.tensor([0, 1]), ("static",), settings)
    forks = _binary_metrics(torch.tensor([[[3.0]] * 17, [[-3.0]] * 17]),
                            torch.tensor([[[True]] * 17, [[False]] * 17]),
                            torch.tensor([0, 1]), ("fork",), settings)
    assert static["targets"]["static"]["auc"] == 1.0
    assert forks["targets"]["fork"]["auc"] == 1.0


def test_observed_generated_transfer_uses_one_fitted_train_decoder():
    settings = _settings()
    rng = torch.Generator().manual_seed(9)
    train = torch.randn(12, 3, generator=rng)
    labels = (train[:, :2] > 0).float()
    observed = torch.randn(5, 3, generator=rng)
    generated = torch.randn(5, 3, generator=rng)
    together = _fit_probe_many(train, labels, {"observed": observed, "generated": generated}, settings,
                               hidden=True, binary=True)
    separately = _fit_probe_many(train, labels, {"observed": observed}, settings, hidden=True, binary=True)
    torch.testing.assert_close(together["observed"], separately["observed"], atol=0, rtol=0)


def test_mode_summary_is_explicitly_advisory_for_deterministic_predictions():
    settings = _settings()
    logits = torch.zeros(2, 17, 6)
    truth = torch.zeros(2, 17, 6, dtype=torch.bool)
    modes = torch.zeros(2, 17, 2, 6, dtype=torch.bool)
    report = _mode_summary(logits, truth, modes, torch.tensor([0, 1]), settings)
    assert report["status"] == "advisory_only_deterministic_model"
    assert report["nearest_sampled_mode_mean_squared_score_distance"] == 0.25
    assert "brier" not in str(report).lower()


def test_paired_binary_difference_resamples_roots_not_individual_forks():
    settings = _settings()
    truth = torch.tensor([[[True]] * 17, [[False]] * 17])
    left = torch.tensor([[[3.0]] * 17, [[-3.0]] * 17])
    right = -left
    report = _paired_binary_difference(left, right, truth, torch.tensor([0, 1]), ("death",), settings)
    target = report["targets"]["death"]
    assert target["auc_difference"] == 1.0
    assert report["direction"] == "left_minus_right"


def test_stage_cache_reuses_the_same_verified_stage(tmp_path):
    calls = []

    def compute():
        calls.append(True)
        return {"answer": 7}

    metadata = {"feature_manifest": "sealed"}
    assert _load_or_compute_stage(tmp_path, "outcomes", metadata, compute) == {"answer": 7}
    assert _load_or_compute_stage(tmp_path, "outcomes", metadata, compute) == {"answer": 7}
    assert len(calls) == 1
