import pytest
import torch

from d4mj.m03_capability import (
    M03Settings,
    _binary_metrics,
    _fit_probe_many,
    _mode_summary,
)


def _settings():
    return M03Settings(train_roots=2, dev_roots=2, probe_hidden=4, probe_steps=2,
                       probe_batch=4, bootstrap_draws=8, minimum_positive=1,
                       minimum_negative=1)


def test_settings_reject_an_incompatible_legacy_context():
    with pytest.raises(ValueError, match="context"):
        M03Settings(legacy_context=3, lewm_context=4)


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
    assert report["nearest_sampled_mode_brier"] == 0.25
