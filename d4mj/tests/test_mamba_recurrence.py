import pytest
import torch

from d4mj.lewm_diagnostics import mixer_numerical_audit, numerical_check
from d4mj.world_api import ModelBundle
from d4mj.tests.test_lewm import small_config


def test_incoming_carry_gate_detects_a_dropped_state(monkeypatch):
    mixer = ModelBundle.create(small_config()).world.layers[0].mixer
    original = mixer.scan
    def broken(inputs, carry=None, **kwargs):
        return original(inputs, None, **kwargs)
    monkeypatch.setattr(mixer, "scan", broken)
    with pytest.raises((AssertionError,RuntimeError)):
        mixer_numerical_audit(mixer,"fp32")


def test_step_never_rescans_a_stored_prefix(monkeypatch):
    b = ModelBundle.create(small_config()).eval()
    state = b.prefill(torch.randn(2,9,1,12),torch.zeros(2,8,dtype=torch.long))
    mixer = b.world.layers[0].mixer; original = mixer.scan; lengths=[]
    def record(inputs,carry=None,**kwargs):
        lengths.append(inputs.shape[1]); return original(inputs,carry,**kwargs)
    monkeypatch.setattr(mixer,"forward",record)
    for _ in range(10): state,_ = b.advance(state,torch.zeros(2,1,dtype=torch.long))
    assert lengths == [1]*10 and state.step == 18


def test_absolute_carry_threshold_still_rejects_material_corruption():
    expected=torch.zeros(2,4,64,64); corrupted=expected.clone();corrupted[0,0,0,0]=.01
    with pytest.raises(AssertionError,match="triton_fp32/ssm"):
        numerical_check(corrupted,expected,"triton_fp32","ssm")
