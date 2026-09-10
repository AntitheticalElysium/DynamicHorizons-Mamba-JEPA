"""Fresh full-recipe GPU gates and pause/resume after the integration refactor.

PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
    d4mj/docs/evidence/integration/verify_gpu.py --work /tmp/d4mj-integration/gpu
Uses one intact, hash-verified shard as a technical fixture, never a research corpus.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil

import torch

from d4mj.config import load_recipe
from d4mj.data import atomic_manifest, load_joint_corpus, _sha256
from d4mj.experiments import main as run
from d4mj.train import train_joint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    evidence = Path(__file__).resolve().parent
    source = root / 'artifacts/craftax_support_v2/manifest.json'
    parent = json.loads(source.read_text())
    record = parent['shards'][0]
    shard = source.parent / record['file']
    assert _sha256(shard) == record['sha256']
    fixture = args.work / 'raw_shard'; fixture.mkdir()
    (fixture / record['file']).symlink_to(shard.resolve())
    manifest = {'format': parent['format'], 'complete': True, 'episodes': record['episodes'],
                'scope': 'resource verification only; first intact shard, not selected research corpus',
                'parent_manifest_sha256': _sha256(source), 'parent_manifest_path': str(source),
                'shards': [record]}
    for key in ('collector_training_access', 'expert_sha256', 'collector_sha256', 'limit', 'split_rule'):
        manifest[key] = parent.get(key, 'unknown')
    atomic_manifest(fixture / 'manifest.json', manifest)
    summary = {'scope': 'post-integration mechanics; no learning or control result',
               'device': torch.cuda.get_device_name(), 'gpu_preflight': {}, 'm4_authorized': False}
    for variant in ('raw', 'tc'):
        dest = args.work / variant
        code = run(['preflight', '--recipe', str(root / f'd4mj/recipes/lewm_mamba_{variant}.json'),
                    '--dataset', str(fixture), '--out', str(dest), '--verification'])
        if code:
            raise RuntimeError(f'{variant}: stopped at failed component; inspect {dest}/gates.json')
        report = json.loads((dest / 'gates.json').read_text())
        summary['gpu_preflight'][variant] = {
            'recipe_id': report['recipe_id'],
            'components': {k: v['status'] for k, v in report['components'].items()},
            'resource': report['components']['joint_resource']['detail']}
        (evidence / variant).mkdir(exist_ok=True)
        for path in dest.glob('*.json'):
            shutil.copyfile(path, evidence / variant / path.name)
        gc.collect(); torch.cuda.empty_cache()
    config = load_recipe(args.work / 'tc/resolved_recipe.json')
    episodes, data = load_joint_corpus(fixture, config)
    gates = json.loads((args.work / 'tc/gates.json').read_text())
    a, full = train_joint(episodes, config, args.work / 'full', dataset_contract=data, gate_report=gates, stop_at=4)
    expected = {name: {key: value.detach().cpu().clone() for key, value in getattr(a, name).state_dict().items()}
                for name in ('encoder', 'world')}
    del a; gc.collect(); torch.cuda.empty_cache()
    b, first = train_joint(episodes, config, args.work / 'split', dataset_contract=data, gate_report=gates, stop_at=2)
    parent_checkpoint = args.work / 'split/step-000002.pt'; parent_sha = _sha256(parent_checkpoint)
    del b; gc.collect(); torch.cuda.empty_cache()
    b, second = train_joint(episodes, config, args.work / 'split', dataset_contract=data, gate_report=gates,
                            stop_at=4, resume=args.work / 'split/latest.pt')
    assert full == first + second
    for name in expected:
        for key, value in getattr(b, name).state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[name][key], atol=0, rtol=0)
    assert _sha256(parent_checkpoint) == parent_sha
    summary['full_architecture_resume'] = {'updates': '4 versus 2+2', 'weights_and_bn': 'bit_exact',
        'metrics': 'exact', 'immutable_parent_sha256': parent_sha,
        'snapshots': json.loads((args.work / 'split/checkpoints.json').read_text()),
        'recipe_steps': config.joint.steps, 'batch': config.joint.batch, 'frames': config.joint.frames,
        'projections': config.joint.projections, 'precision': config.runtime.precision}
    atomic_manifest(evidence / 'gpu_summary.json', summary)
    print(json.dumps(summary))

if __name__ == '__main__': main()
