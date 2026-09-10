from dataclasses import replace
import json

import pytest
import torch

from d4mj.data import Episode, save_episodes, _sha256
from d4mj.data import JointSampler, audit_episodes, load_joint_corpus
from d4mj.cache import cache_latents_to_store, encoder_digest, load_latent_cache
from d4mj.sources import tensor_state_digest
from d4mj.world_api import ModelBundle
from d4mj.tests.test_lewm import small_config


def raw_episodes(resolution=14):
    episodes = []
    for i, split in enumerate(("train", "train", "dev", "final")):
        frames = (torch.arange(9, dtype=torch.uint8) + i*20)[:, None, None, None]
        episodes.append(Episode(
            observations=frames.expand(9, resolution, resolution, 3).clone(),
            actions_taken=torch.arange(8), rewards=torch.arange(8).float(),
            terminated=torch.tensor([False]*7+[True]), truncated=torch.zeros(8,dtype=torch.bool),
            episode_id=f"episode{i}", split=split, uniform_eligible=True, bc_eligible=False))
    return episodes


def test_sampler_outgoing_alignment_and_exact_resume():
    c = small_config(); episodes = raw_episodes()
    sampler = JointSampler(episodes, c, torch.Generator().manual_seed(12))
    batch = sampler.sample()
    assert batch.frames.shape == (4, 4, 14, 14, 3) and batch.actions.shape == (4, 3)
    for row, (eid, start) in enumerate(zip(batch.episode_ids, batch.starts)):
        i = int(eid[-1]); assert i in (0, 1)
        assert torch.equal(batch.frames[row], episodes[i].observations[start:start+4])
        assert torch.equal(batch.actions[row], episodes[i].actions_taken[start:start+3])
    saved = sampler.state_dict(); expected = sampler.sample()
    resumed = JointSampler(episodes, c, torch.Generator().manual_seed(999))
    resumed.load_state_dict(saved); actual = resumed.sample()
    assert expected.episode_ids == actual.episode_ids and resumed.draws == sampler.draws == 8
    for name in ("frames", "actions", "starts"):
        assert torch.equal(getattr(expected, name), getattr(actual, name))
    # A length-three episode still contributes its single legal four-frame window.
    short = replace(episodes[0], observations=episodes[0].observations[:4],
                    actions_taken=episodes[0].actions_taken[:3], rewards=episodes[0].rewards[:3],
                    terminated=torch.zeros(3,dtype=torch.bool), truncated=torch.zeros(3,dtype=torch.bool))
    assert JointSampler([short], c, torch.Generator()).sample().starts.tolist() == [0]*4


@pytest.mark.parametrize("fault,match", [("split", "explicit ID and split"), ("duplicate", "duplicate episode"),
                                        ("boundary", "reset boundary"), ("action", "outgoing action"),
                                        ("pixels", "geometry/dtype"), ("train", "no eligible TRAIN")])
def test_dataset_rejects_silent_lineage_or_alignment_changes(fault, match):
    episodes = raw_episodes()
    if fault == "split": episodes[0] = replace(episodes[0], split=None)
    if fault == "duplicate": episodes[2] = replace(episodes[2], episode_id=episodes[0].episode_id)
    if fault == "boundary": episodes[0].truncated[2] = True
    if fault == "action": episodes[0].actions_taken[0] = 17
    if fault == "pixels": episodes[0] = replace(episodes[0], observations=episodes[0].observations.float())
    if fault == "train": episodes = episodes[2:]
    with pytest.raises(ValueError, match=match): audit_episodes(episodes, small_config())


def test_raw_bytes_and_collector_lineage_are_part_of_contract(tmp_path):
    c = small_config(); path = tmp_path / "raw.pt"; episodes = raw_episodes()
    save_episodes(path, episodes)
    _, first = load_joint_corpus(path, c)
    assert first["sha256"] == _sha256(path) and first["collector_training_access"] == "unknown"
    path.with_suffix(".pt.manifest.json").write_text(json.dumps({"collector_training_access": 1234}))
    _, second = load_joint_corpus(path, c)
    assert second != first and second["collector_training_access"] == 1234
    episodes[0].observations[0,0,0,0] += 1; save_episodes(path, episodes)
    assert load_joint_corpus(path,c)[1]["sha256"] != first["sha256"]


def test_cache_roundtrip_online_parity_and_encoder_only_identity(tmp_path):
    c = small_config(); bundle = ModelBundle.create(c).eval(); episodes = raw_episodes()
    before = tensor_state_digest(bundle.encoder.state_dict()); digest = encoder_digest(bundle.encoder)
    with torch.no_grad(): bundle.world.pair_projection.weight.add_(1)
    assert encoder_digest(bundle.encoder) == digest
    cached = cache_latents_to_store(bundle.encoder, episodes, c, tmp_path / "cache", source_contract={"test": "fixture"}, parent_checkpoint="parent-sha")
    manifest = json.loads((tmp_path / "cache/manifest.json").read_text())
    assert len(cached) == len(episodes)
    assert manifest["complete"] and manifest["cache"]["parent_checkpoint"] == "parent-sha"
    assert before == tensor_state_digest(bundle.encoder.state_dict())
    bundle.encoder.train(); assert not bundle.encoder.training
    cached = load_latent_cache(tmp_path / "cache", bundle.encoder)
    for raw, exported in zip(episodes, cached):
        assert exported.observations is None and exported.latents.shape == (9,1,12)
        assert exported.latents.dtype == torch.float32 and exported.episode_id == raw.episode_id
        assert torch.equal(exported.actions_taken, raw.actions_taken)
        torch.testing.assert_close(exported.latents, bundle.encode(raw.observations[None])[0], atol=1e-5,rtol=1e-4)
    with pytest.raises(ValueError,match="nonempty"):
        cache_latents_to_store(bundle.encoder, episodes, c, tmp_path / "cache", source_contract={}, parent_checkpoint="x")
    with torch.no_grad(): bundle.encoder.projector[1].running_mean.add_(.1)
    with pytest.raises(ValueError,match="cache_identity"): load_latent_cache(tmp_path / "cache",bundle.encoder)


@pytest.mark.parametrize("key,value", [("family","legacy"),("dtype","bfloat16"),("shape",[64,16])])
def test_cache_rejects_wrong_contract(tmp_path,key,value):
    c=small_config(); b=ModelBundle.create(c)
    cache_latents_to_store(b.encoder, raw_episodes(), c, tmp_path, source_contract={}, parent_checkpoint="test")
    path=tmp_path/"manifest.json"; manifest=json.loads(path.read_text())
    manifest["cache"][key]=value; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="cache_family|cache_identity"): load_latent_cache(tmp_path,b.encoder)
