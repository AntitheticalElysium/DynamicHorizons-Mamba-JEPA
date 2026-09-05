"""Cross-family contracts at the existing runtime and storage boundaries."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from d4mj import cache, execution, gates
from d4mj.config import Config, config_from_dict, recipe_dict
from d4mj.data import Episode
from d4mj.diagnostics import rollout_predictions
from d4mj.imagination import imagine
from d4mj.train import optimizer
from d4mj.transition import initial, observe, advance
from d4mj.world_api import ModelBundle, WorldAPI
from .test_lewm import small_config


def legacy_config(transition="direct", mixer="attention", device="cpu"):
    return Config(transition=transition, time_mixer=mixer, device=device,
                  d_model=32, d_model_encoder=32, n_latents=4, d_bottleneck=4,
                  depth=8, depth_encoder=8, mamba_d_state=8, mamba_headdim=16,
                  sequence=4, sequence_long=16, dynamics_context=12, window=2)


def assert_state(bundle, a, b):
    assert bundle.world_state(a).step == bundle.world_state(b).step
    for left, right in zip(bundle.state_tensors(a), bundle.state_tensors(b), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("transition", ["flow", "direct"])
@pytest.mark.parametrize("mixer", ["attention", "mamba"])
def test_legacy_adapter_preserves_observation_commit_rng_and_branch_memory(transition, mixer):
    if mixer == "mamba" and not torch.cuda.is_available():
        pytest.skip("legacy source Mamba requires CUDA")
    device = "cuda" if mixer == "mamba" else "cpu"
    c = legacy_config(transition, mixer, device)
    b = ModelBundle.create(c).eval()
    assert isinstance(b, WorldAPI)
    frames = torch.randint(256, (2, 6, 63, 63, 3), dtype=torch.uint8)
    actions = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=device)
    rng = torch.Generator(device=device).manual_seed(23)
    reference_rng = torch.Generator(device=device).manual_seed(23)
    actual = expected = None
    from d4mj.data import patchify
    with torch.no_grad():
        for t in range(6):
            outgoing = None if t == 0 else actions[:, t-1:t]
            incoming = torch.full((2, 1), c.n_actions, device=device, dtype=torch.long) if t == 0 else outgoing
            expected, ref_features = observe(b.world, b.encoder, expected, incoming,
                                              patchify(frames[:, t:t+1], c.patch).to(device), reference_rng, c)
            actual, features = b.observe(actual, outgoing, frames[:, t:t+1], rng)
            assert_state(b, actual, expected)
            torch.testing.assert_close(features, ref_features, atol=0, rtol=0)
        assert torch.equal(rng.get_state(), reference_rng.get_state())
        fork = b.fork(actual)
        assert_state(b, actual, fork)
        assert all(x.data_ptr() != y.data_ptr() for x, y in zip(b.state_tensors(actual), b.state_tensors(fork)))
        generated, _ = b.advance(actual, actions[:, :1], rng)
        reference, _ = advance(b.world, expected.world, actions[:, :1], reference_rng, c)
        assert_state(b, generated, reference)
        assert torch.equal(rng.get_state(), reference_rng.get_state())
        repeated = b.repeat_state(actual, 3)
        # Root-major repeat must preserve every spatial slot in both memory types.
        for old, new in zip(b.state_tensors(actual), b.state_tensors(repeated), strict=True):
            wanted = old.reshape(2, -1, *old.shape[1:]).repeat_interleave(3, 0).reshape(-1, *old.shape[1:])
            torch.testing.assert_close(new, wanted, atol=0, rtol=0)
        reset, _ = b.observe(None, None, frames[:, -1:], torch.Generator(device=device).manual_seed(23))
        assert (actual.world.latent - reset.world.latent).abs().max() > 1e-6


@pytest.mark.parametrize("family", ["flow", "direct", "lewm"])
def test_shared_rollout_uses_family_timing_and_outgoing_actions(family):
    c = small_config() if family == "lewm" else legacy_config(family)
    b = ModelBundle.create(c).eval()
    shape = (1, 12) if family == "lewm" else (c.n_spatial, c.d_spatial)
    z = torch.randn(2, 7, *shape)
    actions = torch.randint(17, (2, 6))
    rng = torch.Generator().manual_seed(41)
    direct_rng = torch.Generator().manual_seed(41)
    first = None if family == "lewm" else torch.tensor([[2], [3]])
    with torch.no_grad():
        if family == "lewm":
            root = b.world.teacher(z[:, :4], actions[:, :3]).state
        else:
            root, _ = initial(b.world, z[:, :4], torch.cat((first, actions[:, :3]), 1), direct_rng, c)
            root = replace(root, latent=root.latent[:, -1:], features=root.features[:, -1:])
        assert root.step == (3 if family == "lewm" else 4)
        wanted = []
        for t in range(3, 6):
            if family == "lewm":
                root, _ = b.world.advance(root, actions[:, t:t+1])
            else:
                root, _ = advance(b.world, root, actions[:, t:t+1], direct_rng, c)
            wanted.append(root.latent)
        predicted, state = rollout_predictions(b, z, actions, 4, rng, first_action=first)
        torch.testing.assert_close(predicted, torch.cat(wanted, 1), atol=0, rtol=0)
        assert_state(b, state, root)
        assert torch.equal(rng.get_state(), direct_rng.get_state())


def test_control_gate_precedes_environment_and_head_use(monkeypatch):
    b = ModelBundle.create(small_config()).eval()
    def forbidden(*args):
        pytest.fail("M0-M3 must fail before environment or policy execution")
    monkeypatch.setattr(execution, "reset", forbidden)
    with pytest.raises(RuntimeError, match="phase_gate"):
        execution.run_episode(b, None, forbidden, 0, b.config)
    with pytest.raises(RuntimeError, match="phase_gate"):
        imagine(b, forbidden, None, None, None, None, b.config)


@pytest.mark.parametrize("transition", ["flow", "direct"])
def test_existing_execution_calls_adapter_with_native_action_and_reset_timing(monkeypatch, transition):
    from d4mj.agent import Heads
    from d4mj.world_api import LegacyWorldAdapter
    c = legacy_config(transition); b = ModelBundle.create(c).eval()
    heads = Heads(c).eval()
    state = SimpleNamespace(achievements=np.zeros(22, dtype=bool))
    frame = torch.zeros((63, 63, 3), dtype=torch.uint8)
    monkeypatch.setattr(execution, "reset", lambda seed: (frame, state))
    choices = []; observed = []
    def step(env, action, seed):
        choices.append(action)
        return frame + len(choices), env, 1.0, len(choices) == 3, False
    monkeypatch.setattr(execution, "step", step)
    original = LegacyWorldAdapter.observe
    def observe_spy(self, current, action, frame, rng=None):
        observed.append((None if current is None else current.world.step,
                         None if action is None else int(action)))
        return original(self, current, action, frame, rng)
    monkeypatch.setattr(LegacyWorldAdapter, "observe", observe_spy)
    result = execution.run_episode(b.world, b.encoder, heads, 8, c, limit=5)
    assert result.steps == 3 and result.reward == 3 and result.terminated
    assert observed == [(None, None), (1, choices[0]), (2, choices[1])]


def test_shared_optimizer_preserves_explicit_family_decay_policies():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    model[0].weight._no_weight_decay = True
    for joint in (False, True):
        opt = optimizer([model], Config(), exclude_vectors=joint)
        groups = {id(p): group["weight_decay"] for group in opt.param_groups for p in group["params"]}
        assert groups[id(model[0].weight)] == 0
        assert groups[id(model[0].bias)] == (0 if joint else Config().weight_decay)
        assert groups[id(model[1].weight)] == (0 if joint else Config().weight_decay)


def test_shared_legacy_cache_resumes_verified_shards_and_preserves_identity(tmp_path, monkeypatch):
    c = legacy_config(); b = ModelBundle.create(c).eval()
    episodes = [Episode(torch.randint(256, (8, 63, 63, 3), dtype=torch.uint8),
                        torch.arange(7), torch.zeros(7), torch.zeros(7, dtype=torch.bool),
                        torch.zeros(7, dtype=torch.bool)) for _ in range(3)]
    original = cache._cache_episode
    calls = []
    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted export")
        return original(*args)
    out = tmp_path / "cache"
    monkeypatch.setattr(cache, "_cache_episode", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache.cache_latents_to_store(b.encoder, episodes, c, out, source_contract={}, shard_episodes=1)
    first = (out / "shard-000000.pt").read_bytes()
    assert json.loads((out / "manifest.json").read_text())["episodes"] == 1
    monkeypatch.setattr(cache, "_cache_episode", original)
    result = cache.cache_latents_to_store(b.encoder, episodes, c, out, source_contract={}, shard_episodes=1)
    assert len(result) == 3 and first == (out / "shard-000000.pt").read_bytes()
    expected = cache.cache_latents(b.encoder, episodes, c)
    for a, e in zip(result, expected):
        assert a.latent_digest == e.latent_digest
        torch.testing.assert_close(a.latents, e.latents, atol=0, rtol=0)
    (out / "shard-000000.pt").write_bytes(first + b"corrupted")
    with pytest.raises(ValueError, match="checksum|digest|hash"):
        cache.cache_latents_to_store(b.encoder, episodes, c, out, source_contract={}, shard_episodes=1)


def test_recipe_dispatch_and_shared_gate_dependencies(monkeypatch):
    for c in (legacy_config(), small_config()):
        assert config_from_dict(recipe_dict(c)) == c
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict({"family": "invented"})
    called = []
    for name in gates.LEGACY_CHECKS:
        monkeypatch.setattr(gates, name, lambda c, name=name: called.append(name))
    assert all(v["status"] == "pass" for v in gates.preflight(legacy_config())["components"].values())
    assert called == list(gates.LEGACY_CHECKS)
    from d4mj import lewm_diagnostics
    def fail():
        raise ValueError("source fixture")
    monkeypatch.setattr(lewm_diagnostics, "joint_checks", lambda *args: [
        gates.Gate("source", fail), gates.Gate("resource", lambda: pytest.fail("blocked probe ran"), ("source",))])
    report = gates.preflight(small_config(), [], {})
    assert report["components"]["resource"]["status"] == "blocked"
    assert report["architecture_verdict"] == "not_evaluated"


def test_package_cli_dispatches_legacy_recipe_and_preserves_no_argument_lattice(tmp_path, monkeypatch):
    from d4mj.__main__ import main
    seen = []
    for name in gates.LEGACY_CHECKS:
        monkeypatch.setattr(gates, name, lambda c, name=name: seen.append((c.transition, c.time_mixer, name)))
    assert main([]) == 0
    assert len(seen) == 4 * len(gates.LEGACY_CHECKS)
    recipe = tmp_path / "legacy.json"
    recipe.write_text(json.dumps(recipe_dict(legacy_config())))
    run = tmp_path / "run"
    assert main(["preflight", "--recipe", str(recipe), "--out", str(run)]) == 0
    report = json.loads((run / "gates.json").read_text())
    assert report["family"] == "direct-attention"
    assert main(["joint", "--run", str(run)]) == 1
    assert json.loads((run / "failure.json").read_text())["component"] == "recipe_family"
