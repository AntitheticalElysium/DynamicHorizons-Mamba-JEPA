"""Compare shared callers against functions recovered from the pre-LeWM git commit.

Run from the repository root with PYTHONPATH=. and a CUDA-enabled environment.
The reference functions are read from git; no workspace source is replaced.
"""
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch

from d4mj import diagnostics, execution, imagination, train
from d4mj.agent import Heads
from d4mj.data import Episode
from d4mj.tests.conftest import latent_batch
from d4mj.tests.test_integration import legacy_config
from d4mj.world_api import ModelBundle

COMMIT = '162efd1'

def reference(name):
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:d4mj/{name}.py'], text=True)
    module = types.ModuleType(f'd4mj._integration_reference_{name}')
    module.__package__ = 'd4mj'
    sys.modules[module.__name__] = module
    exec(compile(source, f'git:{COMMIT}:d4mj/{name}.py', 'exec'), module.__dict__)
    return module


def rng(seed):
    return torch.Generator(device='cuda').manual_seed(seed)


def compare_tensors(a, b):
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def main():
    old = {name: reference(name) for name in ('diagnostics', 'execution', 'imagination', 'train')}
    report = {'reference_commit': subprocess.check_output(['git', 'rev-parse', COMMIT], text=True).strip(),
              'device': torch.cuda.get_device_name(), 'scope': 'exact legacy caller/cache/optimizer parity', 'arms': {}}
    for transition in ('flow', 'direct'):
        for mixer in ('attention', 'mamba'):
            c = legacy_config(transition, mixer, 'cuda')
            b = ModelBundle.create(c).eval()
            torch.manual_seed(91)
            heads = Heads(c).cuda().eval()
            with torch.no_grad():
                for p in heads.parameters():
                    p.normal_(std=.02)
            batch = latent_batch(c, 2, 7)
            batch = train._to(batch, 'cuda')
            successors = torch.randn(2, 3, c.n_spatial, c.d_spatial, device='cuda')
            for name, kw in [('multistep_error', {'context': 4, 'successors': successors}), ('latent_stats', {})]:
                r1, r2 = rng(23), rng(23)
                expected = getattr(old['diagnostics'], name)(b.world, batch, r1, c, **kw)
                actual = getattr(diagnostics, name)(b.world, batch, r2, c, **kw)
                assert expected == actual, (transition, mixer, name, expected, actual)
                compare_tensors(r1.get_state(), r2.get_state())
            with torch.no_grad():
                root = b.prefill(batch.latents[:, :4], batch.led_to_action[:, 1:4], rng(24),
                                 first_action=batch.led_to_action[:, :1])
                r1, r2, p1, p2 = rng(25), rng(25), rng(26), rng(26)
                expected = old['imagination'].imagine(b.world, heads, root, b.features(root), r1, p1, c)
                actual = imagination.imagine(b.world, heads, root, b.features(root), r2, p2, c)
                for field in fields(actual):
                    compare_tensors(getattr(expected, field.name), getattr(actual, field.name))
                compare_tensors(r1.get_state(), r2.get_state()); compare_tensors(p1.get_state(), p2.get_state())
            rows = []
            for module in (old['execution'], execution):
                choices = []
                frame = torch.randint(256, (63, 63, 3), generator=torch.Generator().manual_seed(99), dtype=torch.uint8)
                def reset(seed):
                    return frame, types.SimpleNamespace(achievements=np.zeros(22, dtype=bool))
                def step(state, action, seed):
                    choices.append(action)
                    return frame + len(choices), state, float(action), len(choices) == 6, False
                module.reset, module.step = reset, step
                result = module.run_episode(b.world, b.encoder, heads, 29, c, limit=8)
                rows.append((choices, result))
            assert rows[0][0] == rows[1][0]
            assert vars(rows[0][1]) == vars(rows[1][1])
            pixels = torch.randint(256, (20, 63, 63, 3), generator=torch.Generator().manual_seed(92), dtype=torch.uint8)
            episode = Episode(pixels, torch.arange(19) % 17, torch.zeros(19),
                              torch.zeros(19, dtype=torch.bool), torch.zeros(19, dtype=torch.bool))
            before_cache = old['train'].cache_latents(b.encoder, [episode], c)[0]
            after_cache = train.cache_latents(b.encoder, [episode], c)[0]
            assert before_cache.latent_digest == after_cache.latent_digest
            compare_tensors(before_cache.latents, after_cache.latents)
            # Compare grouping, order, schedule, decay and clipping through actual updates.
            torch.manual_seed(93); first = torch.nn.Linear(7, 5).cuda()
            second = torch.nn.Linear(7, 5).cuda(); second.load_state_dict(first.state_dict())
            first.weight._no_weight_decay = second.weight._no_weight_decay = True
            o1, o2 = old['train'].optimizer([first], c), train.optimizer([second], c)
            for i in range(3):
                inputs = torch.arange(21, device='cuda').float().reshape(3, 7)
                old['train']._update(o1, first(inputs).square().mean(), [first], c, i)
                train._update(o2, second(inputs).square().mean(), [second], c, i)
            for p, q in zip(first.parameters(), second.parameters()): compare_tensors(p, q)
            report['arms'][f'{transition}-{mixer}'] = {'diagnostics': 'bit_exact', 'imagination': 'bit_exact',
                'execution_actions_and_results': 'exact', 'rng_advancement': 'exact',
                'chunked_cache': 'bit_exact', 'cache_digest': before_cache.latent_digest, 'optimizer_updates': 'bit_exact'}
            del b, heads, first, second, o1, o2
            torch.cuda.empty_cache()
    path = Path(__file__).with_name('legacy_parity.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))

if __name__ == '__main__': main()
