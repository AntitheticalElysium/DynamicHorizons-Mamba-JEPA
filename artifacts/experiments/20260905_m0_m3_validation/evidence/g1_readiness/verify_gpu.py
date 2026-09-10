from dataclasses import replace
import json
from pathlib import Path
from d4mj.config import load_recipe,recipe_dict
from d4mj.lewm_config import ScreenConfig
from d4mj.experiments import run_joint_pair
from d4mj.data import atomic_manifest
root=Path('artifacts/lewm_gates_20260906/gpu_verification')
configs={}
for variant in ('raw','tc'):
 c=load_recipe(f'd4mj/recipes/lewm_mamba_{variant}.json')
 configs[variant]=replace(c,joint=replace(c.joint,steps=4,screen_step=2,warmup=1,checkpoint_every=2),
                         runtime=replace(c.runtime,purpose='verification'))
s=ScreenConfig(train_episodes=8,dev_episodes=3,windows_per_episode=4,probe_steps=10,bootstrap_draws=40,
               minimum_positive=2,minimum_negative=2)
code=run_joint_pair(configs,s,Path('/tmp/d4mj-integration/gpu_final/raw_shard'),root,screen_only=True)
report=json.loads((root/'G1/screen.json').read_text())
summary={'scope':'G1 evaluator readiness on full B128/F4/J1024 BF16 models, two verification updates; not research screening',
         'decision':report['decision'],'components':{k:v['status'] for k,v in report['components'].items()},
         'blocked_component':report.get('blocked_component'),'code':code}
atomic_manifest(root/'verification_summary.json',summary);print(json.dumps(summary))
if report['decision']=='stop_component':raise RuntimeError(json.dumps(report['components']))
