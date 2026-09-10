import json, copy, time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import torch
from d4mj.lewm_config import DynamicsSettings
from d4mj.mamba_recurrence import FunctionalMamba2, MambaCarry

def err(a,b):
    a,b=a.detach().float(),b.detach().float()
    return {'max_abs':float((a-b).abs().max()),'rms':float((a-b).square().mean().sqrt()),'scale':float(b.abs().max())}
def measure(d,seed,T,precision):
    torch.manual_seed(seed)
    m=FunctionalMamba2(d).cuda()
    g=torch.Generator(device='cuda').manual_seed(seed+1)
    x=torch.randn(2,T,d.width,device='cuda',generator=g,requires_grad=True)
    empty=m.initial(2)
    initial=MambaCarry(torch.randn(empty.conv.shape,device='cuda',generator=g).requires_grad_(),(torch.randn(empty.ssm.shape,device='cuda',generator=g)*.01).requires_grad_())
    def run(backend,step=False):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            if not step:return m.scan(x,initial,backend=backend)
            s=initial; parts=[]
            for t in range(T):
                y,s=m.scan(x[:,t:t+1],s,backend=backend);parts.append(y)
            return torch.cat(parts,1),s
    y,s=run('triton'); yy,ss=run('triton',True); ref,rs=run('reference'); rr,rrs=run('reference',True)
    variables=[x,initial.conv,initial.ssm,*m.parameters()]
    def grads(y,s):return torch.autograd.grad(y.float().square().mean()+s.conv.float().square().mean()+s.ssm.square().mean(),variables)
    gy,gs,gr=grads(y,s),grads(yy,ss),grads(ref,rs)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        source=m.core(x); ours,_=m.scan(x)
    row={'width':d.width,'d_state':d.d_state,'seed':seed,'T':T,'precision':precision,'scan_step':{'out':err(y,yy),'conv':err(s.conv,ss.conv),'ssm':err(s.ssm,ss.ssm)},'triton_reference':{'out':err(y,ref),'ssm':err(s.ssm,rs.ssm)},'reference_self':{'out':err(ref,rr),'ssm':err(rs.ssm,rrs.ssm)},'source_full':err(ours,source),'scan_step_grad':[err(a,b) for a,b in zip(gy,gs)],'triton_reference_grad':[err(a,b) for a,b in zip(gy,gr)]}
    print(json.dumps(row),flush=True)
    return row
rows=[]
for precision in ('fp32','bf16'):
    for width in (32,256):
        d=DynamicsSettings(width=width,depth=2 if width==32 else 6,headdim=16 if width==32 else 64,d_state=8 if width==32 else 64)
        for seed in (123,500):
            for T in (2,17,65,257):rows.append(measure(d,seed,T,precision))
Path('/tmp/lewm-audit/calibration.json').write_text(json.dumps({'device':torch.cuda.get_device_name(),'torch':torch.__version__,'rows':rows},indent=2))
