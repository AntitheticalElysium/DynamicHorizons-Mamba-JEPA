import copy
from dataclasses import replace

import pytest
import torch

from d4mj.checkpoint import (save_lewm_bundle, read_lewm_bundle, restore_lewm_bundle,
                             publish_lewm_latest)
from d4mj.data import JointSampler
from d4mj.sources import lewm_source_manifest, tensor_state_digest
from d4mj.train import joint_optimizer, set_phase_mode
from d4mj.world_api import ModelBundle, load_bundle
from d4mj.tests.test_joint_data import raw_episodes
from d4mj.tests.test_lewm import small_config


@pytest.fixture
def snapshot(tmp_path):
    c = small_config(); b = ModelBundle.create(c); set_phase_mode(b, "joint")
    opt = joint_optimizer(b)
    # Real optimizer tensors and normalization buffers, not an empty state roundtrip.
    frames = torch.stack([e.observations[:4] for e in raw_episodes()])
    z = b.encoder(frames)
    b.world.teacher(z, torch.zeros(4,3,dtype=torch.long)).predicted.square().mean().backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    sampler = JointSampler(raw_episodes(),c,torch.Generator().manual_seed(5)); sampler.sample()
    rng = torch.Generator().manual_seed(6); torch.rand(3,generator=rng)
    path = tmp_path / "step-000001.pt"
    saved = save_lewm_bundle(path,b,step=1,dataset_contract={"bytes":"sealed"},
                            initial_identity={"fixture": True},optimizer=opt,sampler=sampler,
                            projection_rng=rng,gate_report={"test":True})
    return path, c, b, opt, sampler, rng, saved


def test_v2_roundtrip_restores_model_optimizer_normalization_and_random_streams(snapshot):
    path,c,b,opt,sampler,rng,saved = snapshot
    expected = {"encoder":tensor_state_digest(b.encoder.state_dict()),
                "world":tensor_state_digest(b.world.state_dict())}
    batch_expected = sampler.sample(); projection_expected = torch.rand(4,generator=rng)
    global_expected = torch.rand(4)
    restored = ModelBundle.create(c); set_phase_mode(restored,"joint")
    opt2 = joint_optimizer(restored)
    sampler2 = JointSampler(raw_episodes(),c,torch.Generator()); rng2 = torch.Generator()
    restore_lewm_bundle(path,restored,dataset_contract={"bytes":"sealed"},optimizer=opt2,
                        sampler=sampler2,projection_rng=rng2)
    assert tensor_state_digest(restored.encoder.state_dict()) == expected["encoder"]
    assert tensor_state_digest(restored.world.state_dict()) == expected["world"]
    assert len(opt2.state) == len(opt.state) > 0
    for p,q in zip(opt.state.values(),opt2.state.values()):
        for key in p: torch.testing.assert_close(p[key],q[key],atol=0,rtol=0)
    assert torch.equal(sampler2.sample().starts,batch_expected.starts)
    assert torch.equal(torch.rand(4,generator=rng2),projection_expected)
    assert torch.equal(torch.rand(4),global_expected)
    assert restored.encoder.training and restored.world.training
    assert all(not p.requires_grad for p in restored.world.agent_readout.parameters())
    inference,payload = load_bundle(path,device="cpu",backend="reference")
    assert payload["capabilities"]["m4_authorized"] is False
    assert tensor_state_digest(inference.world.state_dict()) == expected["world"]
    inference.encoder.train(); assert not inference.encoder.training
    assert not inference.world.training and all(not p.requires_grad for p in inference.encoder.parameters())


@pytest.mark.parametrize("fault,match", [("family","checkpoint_family"),("recipe","checkpoint_recipe"),
    ("scheduler","checkpoint_schedule"),("source","source_identity"),("capabilities","checkpoint_phase")])
def test_v2_rejects_corrupt_or_cross_family_payload(snapshot,tmp_path,fault,match):
    path,*_ = snapshot; payload = torch.load(path,weights_only=False)
    if fault == "family": payload["format"] = "d4mj_checkpoint_v1"
    if fault == "recipe": payload["config"]["variant"] = "raw"
    if fault == "scheduler": payload["scheduler"]["completed_updates"] += 1
    if fault == "source": payload["sources"]["references"]["vit_helper"] = "changed"
    if fault == "capabilities": payload["capabilities"]["validated_recursive_depth"] = 16
    broken=tmp_path/"broken.pt"; torch.save(payload,broken)
    with pytest.raises(ValueError,match=match): read_lewm_bundle(broken)


@pytest.mark.parametrize("fault,match", [("data","checkpoint_dataset"),("bn","checkpoint_recipe"),
    ("schedule","checkpoint_recipe"),("backend","checkpoint_recipe"),("frozen","checkpoint_phase")])
def test_resume_rejects_changed_contract_before_loading_weights(snapshot,tmp_path,fault,match):
    path,c,_,_,_,_,_ = snapshot; data={"bytes":"sealed"}
    if fault == "data": data={"bytes":"changed"}
    if fault == "bn": c=replace(c,encoder=replace(c.encoder,bn_eps=1e-4))
    if fault == "schedule": c=replace(c,joint=replace(c.joint,steps=5))
    if fault == "backend": c=replace(c,dynamics=replace(c.dynamics,backend="triton"))
    if fault == "frozen":
        payload=torch.load(path,weights_only=False);payload["encoder_frozen"]=True
        path=tmp_path/"frozen.pt";torch.save(payload,path)
    b=ModelBundle.create(c); opt=joint_optimizer(b)
    digest=tensor_state_digest(b.encoder.state_dict())
    with pytest.raises(ValueError,match=match):
        restore_lewm_bundle(path,b,dataset_contract=data,optimizer=opt,
            sampler=JointSampler(raw_episodes(),c,torch.Generator()),projection_rng=torch.Generator())
    assert tensor_state_digest(b.encoder.state_dict()) == digest


def test_snapshot_is_immutable_and_sources_have_recoverable_helper_pin(snapshot):
    path,c,b,opt,sampler,rng,_ = snapshot
    with pytest.raises(FileExistsError):
        save_lewm_bundle(path,b,step=1,dataset_contract={},initial_identity={},optimizer=opt,
                         sampler=sampler,projection_rng=rng,gate_report={})
    publish_lewm_latest(path)
    assert (path.parent/"latest.pt").resolve() == path.resolve()
    sources=lewm_source_manifest()
    assert sources["pins"]["galilai-group__stable-pretraining"]["commit"] == "9aa93f8b6153eebb73f57d4853ccf8a13d848310"
    assert "galilai-group__stable-pretraining/LICENSE" in sources["references"]
