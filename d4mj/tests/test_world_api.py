import pytest
import torch

from d4mj.state import WorldState
from d4mj.world_api import ModelBundle, WorldAPI
from .test_lewm import small_config


def test_persistent_state_fork_commit_observe_and_firewall():
    b=ModelBundle.create(small_config()).eval()
    assert isinstance(b,WorldAPI)
    pixels=torch.randint(256,(2,7,14,14,3),dtype=torch.uint8)
    z=b.encode(pixels); actions=torch.randint(17,(2,6))
    state=b.prefill(z[:,:6],actions[:,:5]); before=[x.clone() for x in b.state_tensors(state)]
    generated,_=b.advance(state,actions[:,5:])
    observed,_=b.observe(state,actions[:,5:],pixels[:,6:])
    assert generated.step==observed.step==6
    for a,c in zip(generated.memory,observed.memory):
        torch.testing.assert_close(a.conv,c.conv); torch.testing.assert_close(a.ssm,c.ssm)
    torch.testing.assert_close(observed.latent,z[:,6:])
    for a,c in zip(before,b.state_tensors(state)):assert torch.equal(a,c)
    repeated=b.repeat_state(state,3)
    branch,_=b.advance(repeated,actions[:,5:].repeat_interleave(3,0))
    torch.testing.assert_close(branch.latent,generated.latent.repeat_interleave(3,0))
    fork=b.fork(state)
    assert all(a.data_ptr()!=c.data_ptr() for a,c in zip(b.state_tensors(state),b.state_tensors(fork)))
    with torch.no_grad():
        for p in b.world.agent_readout.parameters():p.add_(10)
    changed,_=b.advance(state,actions[:,5:]); torch.testing.assert_close(changed.latent,generated.latent)
    with pytest.raises(TypeError,match='PredictiveState'):b.features(WorldState(z[:,:1],(),1))


def test_recursive_gradient_reaches_previous_memory_and_accepted_latent():
    b=ModelBundle.create(small_config()).eval(); initial=torch.randn(2,1,1,12,requires_grad=True)
    first,_=b.advance(b.start(initial),torch.zeros(2,1,dtype=torch.long))
    first.latent.retain_grad(); first.memory[0].ssm.retain_grad()
    final,_=b.advance(first,torch.ones(2,1,dtype=torch.long)); final.latent.square().mean().backward()
    assert initial.grad.abs().sum()>0
    assert first.latent.grad.abs().sum()>0 and first.memory[0].ssm.grad.abs().sum()>0
    assert all(not t.requires_grad for t in b.state_tensors(b.detach_state(first)))


def test_state_api_rejects_wrong_actions_and_training_bn():
    b=ModelBundle.create(small_config()); state=b.start(torch.randn(2,1,1,12))
    with pytest.raises(RuntimeError,match='normalization'):b.advance(state,torch.zeros(2,1,dtype=torch.long))
    b.eval()
    for a in (torch.full((2,1),17,dtype=torch.long),torch.zeros(2,2,dtype=torch.long),torch.zeros(2,1)):
        with pytest.raises(ValueError):b.advance(state,a)
    with pytest.raises(ValueError,match='completed pairs'):
        b.prefill(torch.randn(2,4,1,12),torch.zeros(2,4,dtype=torch.long))


def test_constant_state_size_over_long_context_and_reset():
    b=ModelBundle.create(small_config()).eval(); z=torch.randn(1,1,1,12); root=b.start(z)
    sizes=[t.numel() for t in b.state_tensors(root)]
    with torch.no_grad():
        state=root
        for _ in range(129):state,_=b.advance(state,torch.zeros(1,1,dtype=torch.long))
    assert state.step==129 and [t.numel() for t in b.state_tensors(state)]==sizes
    for a,c in zip(b.state_tensors(root),b.state_tensors(b.start(z))):assert torch.equal(a,c)
