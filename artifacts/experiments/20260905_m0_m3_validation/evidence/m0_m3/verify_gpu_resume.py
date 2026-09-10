import json
from pathlib import Path
import torch
from d4mj.lewm_config import load_recipe
from d4mj.joint_data import load_joint_corpus
from d4mj.train_lewm import train_joint
from d4mj.data import _sha256
from d4mj.sources import tensor_state_digest
root=Path('/tmp/lewm-audit/full_gpu_preflight_final');c=load_recipe(root/'resolved_recipe.json')
episodes,data=load_joint_corpus('/tmp/lewm-audit/real_shard',c)
gates=json.loads((root/'gates.json').read_text())
a,rows=train_joint(episodes,c,root/'resume_full',dataset_contract=data,gate_report=gates,stop_at=4)
expected={name:{key:value.detach().cpu() for key,value in getattr(a,name).state_dict().items()} for name in ('encoder','world')}
del a;torch.cuda.empty_cache()
b,first=train_joint(episodes,c,root/'resume_split',dataset_contract=data,gate_report=gates,stop_at=2)
parent=root/'resume_split/step-000002.pt';parent_sha=_sha256(parent)
del b;torch.cuda.empty_cache()
b,second=train_joint(episodes,c,root/'resume_split',dataset_contract=data,gate_report=gates,stop_at=4,resume=root/'resume_split/latest.pt')
errors={}
for name in ('encoder','world'):
 actual={key:value.detach().cpu() for key,value in getattr(b,name).state_dict().items()}
 errors[name]=max(float((actual[key].float()-value.float()).abs().max()) for key,value in expected[name].items())
 for key,value in expected[name].items():torch.testing.assert_close(actual[key],value,atol=1e-6,rtol=1e-5)
assert [r['windows'] for r in rows]==[r['windows'] for r in first+second]
assert parent_sha==_sha256(parent)
report={'scope':'full B128/F4/J1024 BF16 architecture, verification only; four updates of unchanged 10000-update schedule',
        'state_max_abs':errors,'metrics_exact':rows==first+second,'screen_sha256_preserved':parent_sha,
        'snapshots':json.loads((root/'resume_split/checkpoints.json').read_text()),
        'recipe_id':gates['recipe_id'],'cuda_device':torch.cuda.get_device_name(),'m4_authorized':False}
(root/'gpu_resume.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
