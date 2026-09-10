import json
from pathlib import Path
from dataclasses import replace
import torch
from d4mj.world_api import ModelBundle
from d4mj.lewm_config import LeWMConfig
from d4mj.lewm_diagnostics import mixer_numerical_audit
b=ModelBundle.create(LeWMConfig()).eval();w=b.world
rows=[]
for seed in (500,1500):
 for length in (17,65,257):
  rng=torch.Generator(device='cuda').manual_seed(seed)
  z=torch.randn(2,length+1,1,192,device='cuda',generator=rng)
  actions=torch.randint(17,(2,length),device='cuda',generator=rng)
  with torch.no_grad():
   full=w.teacher(z,actions).state
   state=b.start(z[:,:1])
   for t in range(length):state,_=w.observe_latent(state,actions[:,t:t+1],z[:,t+1:t+2])
   prefix=w.teacher(z[:,:8],actions[:,:7]).state
   chunk=w.teacher(z[:,7:],actions[:,7:],state=prefix).state
  row={'seed':seed,'length':length}
  for name,s in [('step',state),('chunk',chunk)]:
   row[name]=[float((x-y).abs().max()) for x,y in zip(b.state_tensors(full),b.state_tensors(s))]
  rows.append(row);print(json.dumps(row),flush=True)
report={'rows':rows,'bf16':mixer_numerical_audit(w.layers[0].mixer,'bf16')}
Path('/tmp/lewm-audit/world_calibration.json').write_text(json.dumps(report,indent=2))
