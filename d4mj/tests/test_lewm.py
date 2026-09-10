import ast
from dataclasses import replace
import logging

import pytest
import torch
from transformers import ViTConfig, ViTModel

from d4mj.lewm import LeWMEncoder, SIGReg, joint_loss
from d4mj.lewm_config import LeWMConfig, EncoderSettings, DynamicsSettings, JointSettings, RuntimeSettings
from d4mj.config import config_from_dict, recipe_dict
from d4mj.sources import ROOT, tensor_state_digest
from d4mj.world_api import ModelBundle
from d4mj.lewm_diagnostics import objective_audit, normalization_audit, recurrence_audit


def small_config(**overrides):
    c = LeWMConfig(
        encoder=EncoderSettings(resolution=14, width=24, depth=1, heads=3, latent_dim=12,
                                projector_hidden=32, checkpoint_blocks=False),
        dynamics=DynamicsSettings(width=32, depth=2, headdim=16, d_state=8,
                                  action_dim=8, backend="reference"),
        joint=JointSettings(batch=4, projections=8, knots=5, steps=4, screen_step=2,
                            warmup=1, checkpoint_every=2),
        runtime=RuntimeSettings(device="cpu", precision="fp32", purpose="verification", cache_chunk=3),
    )
    return replace(c, **overrides)


def test_config_roundtrip_and_actual_statistical_batch():
    c = small_config()
    assert config_from_dict(recipe_dict(c)) == c
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict(recipe_dict(c) | {"guess": 123})
    with pytest.raises(ValueError, match="actual B128"):
        replace(c, runtime=replace(c.runtime, purpose="research"))
    b = ModelBundle.create(c)
    with pytest.raises(ValueError, match="statistical_batch"):
        joint_loss(b.encoder, b.world, torch.zeros(2,4,14,14,3,dtype=torch.uint8),
                   torch.zeros(2,3,dtype=torch.long), SIGReg(5,8), torch.Generator(), c)


def test_source_objective_and_normalization():
    assert objective_audit(small_config())["absolute_error"] < 1e-5
    assert normalization_audit(ModelBundle.create(small_config()))["buffers_immutable"]


def test_encoder_constructor_matches_pinned_tiny_helper():
    source = ast.parse((ROOT / "sources/galilai-group__stable-pretraining/stable_pretraining/backbone/utils.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "vit_hf")
    scope = {"nn": torch.nn, "ViTConfig": ViTConfig, "ViTModel": ViTModel,
             "_TRANSFORMERS_AVAILABLE": True, "logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_vit_helper", "exec"), scope)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        reference = scope["vit_hf"]("tiny", patch_size=7, image_size=63, use_mask_token=False)
        torch.manual_seed(9)
        ours = LeWMEncoder(LeWMConfig()).backbone
    assert tensor_state_digest(ours.state_dict()) == tensor_state_digest(reference.state_dict())
    assert ours.config.layer_norm_eps == reference.config.layer_norm_eps


def test_raw_tc_initialization_and_two_sided_joint_gradients():
    c = small_config(); tc = ModelBundle.create(c); raw = ModelBundle.create(replace(c, variant="raw"))
    assert tensor_state_digest(tc.encoder.state_dict()) == tensor_state_digest(raw.encoder.state_dict())
    assert tensor_state_digest(tc.world.state_dict()) == tensor_state_digest(raw.world.state_dict())
    frames = torch.randint(256,(4,4,14,14,3),dtype=torch.uint8); actions = torch.randint(17,(4,3))
    loss = joint_loss(tc.encoder,tc.world,frames,actions,SIGReg(5,8),torch.Generator().manual_seed(2),c)
    target_grad = torch.autograd.grad(loss.prediction,loss.latent,retain_graph=True)[0]
    assert target_grad[:,-1].abs().sum()>0 and target_grad[:,0].abs().sum()>0
    loss.total.backward()
    assert tc.encoder.projector[0].weight.grad.abs().sum()>0
    assert tc.world.pair_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in tc.world.agent_readout.parameters())
    assert tc.encoder.projector[1].num_batches_tracked.item()==1
    tc.encoder.freeze().train()
    assert not tc.encoder.training and not tc.encoder.projector[1].training


def test_checkpointed_encoder_bn_updates_once():
    c=small_config(); c=replace(c,encoder=replace(c.encoder,checkpoint_blocks=True))
    b=ModelBundle.create(c)
    loss=joint_loss(b.encoder,b.world,torch.randint(256,(4,4,14,14,3),dtype=torch.uint8),
                    torch.zeros(4,3,dtype=torch.long),SIGReg(5,8),torch.Generator(),c)
    loss.total.backward()
    assert b.encoder.projector[1].num_batches_tracked.item()==1
    assert b.world.predictor_projector[1].num_batches_tracked.item()==1


def test_source_recurrence_and_final_api():
    report=recurrence_audit(ModelBundle.create(small_config()))
    assert report['context_frames']>report['training_frames'] and report['state_gradient_checked']
    assert not report['cuda_kernels_checked']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='target CUDA kernels unavailable')
def test_cuda_source_and_differentiable_carry():
    c=small_config()
    c=replace(c,dynamics=replace(c.dynamics,backend='triton'),runtime=replace(c.runtime,device='cuda'))
    assert recurrence_audit(ModelBundle.create(c))['cuda_kernels_checked']
