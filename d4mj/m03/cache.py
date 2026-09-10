"""Content-addressed M03 dependencies; immutable run reports remain separate.

Small probe predictions/statistics live in one SQLite index. Existing large
replay/feature artifacts are referenced by verified byte hash, never recopied.
Changing a checkpoint invalidates only nodes that consume its feature values.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from functools import wraps
import hashlib
import inspect
import io
import json
from pathlib import Path
import sqlite3
import textwrap
import time

import torch

_ACTIVE = ContextVar('m03_dependency_cache', default=None)
PROBE_SETTINGS = ('seed','probe_hidden','probe_steps','probe_batch','probe_learning_rate','probe_weight_decay')
METRIC_SETTINGS = ('seed','bootstrap_draws','minimum_positive','minimum_negative')
REPLAY_SETTINGS = ('seed','train_roots','dev_roots','terminal_tail_fraction','direct_context','replay_checks','mode_samples')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def content(value):
    """Stable typed identities, including tensor bytes rather than filenames."""
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().contiguous()
        return {'tensor': hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                'shape': list(x.shape), 'dtype': str(x.dtype), 'device': str(value.device)}
    if is_dataclass(value): return content(asdict(value))
    if isinstance(value, dict):
        # A hash identifies the input; moving identical bytes is not a treatment.
        return {str(k):content(v) for k,v in sorted(value.items()) if not (k == 'path' and 'sha256' in value)}
    if isinstance(value, (list,tuple)): return [content(v) for v in value]
    if isinstance(value, Path): return str(value.resolve())
    if value is None or isinstance(value,(str,int,float,bool)): return value
    raise TypeError(f'unsupported cache dependency: {type(value)}')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def implementation(fn):
    """Transitive project-function code and referenced constants, not entire files.

    Thus a change to an oracle scorer does not invalidate encoder inference.
    Model/environment runtime files are additionally explicit feature dependencies.
    """
    seen = {}
    def visit(function):
        function = inspect.unwrap(function)
        name = function.__module__+'.'+function.__qualname__
        if name in seen: return
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        node = tree.body[0]
        node.decorator_list = []
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body.pop(0)
        seen[name] = {'ast': ast.dump(tree,include_attributes=False), 'constants': {}}
        for symbol in {n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}:
            obj = function.__globals__.get(symbol)
            if inspect.isfunction(inspect.unwrap(obj)) and obj.__module__.startswith(('d4mj.','artifacts.')):
                visit(obj)
            elif isinstance(obj,(str,int,float,bool,tuple)):
                try: seen[name]['constants'][symbol] = content(obj)
                except TypeError: pass
    visit(fn)
    return digest(seen)


def load_compatibility(path):
    document = json.loads(Path(path).read_text())
    if document.get('schema') != 'm03_bootstrap_cache_compatibility' or document.get('status') != 'pass':
        raise ValueError('m03_cache: missing validated compatibility proof')
    for name in ('reference_source','validation'):
        identity = document[name]
        if sha_file(identity['path']) != identity['sha256']:
            raise ValueError('m03_cache: compatibility evidence bytes changed')
    return document


class ArtifactCache:
    def __init__(self, path, *, imports=None, settings=None, device="cpu", compatibility=None, timing_path=None):
        self.path = Path(path).resolve(); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS nodes (key TEXT PRIMARY KEY, kind TEXT NOT NULL, dependencies TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB, external TEXT, selector TEXT)')
        self.db.commit()
        self.counts = Counter(); self.imports = imports or {}; self.settings = settings
        self._code = {}
        self.device = device
        self.compatibility = load_compatibility(compatibility) if compatibility else {}
        self.compatibility_identity = {'path':str(Path(compatibility).resolve()),'sha256':sha_file(compatibility)} if compatibility else None
        self.timing_path = Path(timing_path) if timing_path else None
        self.started = time.perf_counter(); self.cpu_started = time.process_time()
        self.started_unix = time.time(); self.last_timing_save = self.started
        self.timings = {}
        self.previous_sessions = []
        if self.timing_path is not None and self.timing_path.exists():
            prior = json.loads(self.timing_path.read_text())
            self.previous_sessions = prior.get('sessions',[prior])



    def code(self, function):
        fn = inspect.unwrap(function)
        if fn not in self._code: self._code[fn] = implementation(fn)
        return self._code[fn]

    def key(self, kind, dependencies):
        identity = {'schema':'m03_dependency_cache','kind':kind,'dependencies':content(dependencies),
                    'torch':torch.__version__, 'cuda':torch.version.cuda,
                    'float32_matmul':torch.get_float32_matmul_precision()}
        return digest(identity), json.dumps(identity,sort_keys=True,separators=(',',':'))

    def read(self, key):
        row = self.db.execute('SELECT sha256,payload,external,selector FROM nodes WHERE key=?',(key,)).fetchone()
        if row is None: raise KeyError(key)
        checksum, data, path, selector = row
        if path:
            if sha_file(path) != checksum: raise ValueError('m03_cache: imported artifact bytes changed: '+path)
            value = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        else:
            if hashlib.sha256(data).hexdigest() != checksum: raise ValueError('m03_cache: corrupt cached node '+key)
            value = torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        return value[selector] if selector else value

    def _timed(self, kind, outcome, started, compute_seconds=0., publish_seconds=0.):
        category = ':'.join(kind.split(':')[:2]) if kind.startswith('features:') else kind
        row = self.timings.setdefault(category, Counter())
        row[outcome+'_calls'] += 1
        row[outcome+'_wall_seconds'] += time.perf_counter()-started
        row['compute_seconds'] += compute_seconds; row['publish_seconds'] += publish_seconds
        if time.perf_counter()-self.last_timing_save >= 20: self.save_timings()

    def save_timings(self, *, closed=False):
        if self.timing_path is None or not self.timing_path.parent.exists(): return
        from ..data import atomic_manifest
        current = {'process_started_unix':self.started_unix,'updated_unix':time.time(),
            'active_seconds':time.perf_counter()-self.started,'cpu_seconds':time.process_time()-self.cpu_started,
            'operations':self.timings,'cache_counts':dict(self.counts),'state':'closed' if closed else 'running'}
        sessions = self.previous_sessions+[current]
        totals = {}
        for session in sessions:
            for kind,row in session['operations'].items():
                total = totals.setdefault(kind,Counter())
                for name,value in row.items(): total[name] += value
        atomic_manifest(self.timing_path,{'schema':'m03_runtime_timing','updated_unix':current['updated_unix'],
            'active_seconds':sum(s['active_seconds'] for s in sessions),
            'cpu_seconds':sum(s['cpu_seconds'] for s in sessions),'operations':totals,'sessions':sessions,
            'timing_scope':'sum of process sessions, excluding time stopped; operation timings may nest'})
        self.last_timing_save = time.perf_counter()

    def get(self, kind, dependencies, compute, *, external=None, compatible=()):
        started = time.perf_counter()
        key, encoded = self.key(kind,dependencies)
        row = self.db.execute('SELECT dependencies FROM nodes WHERE key=?',(key,)).fetchone()
        if row:
            if row[0] != encoded: raise ValueError('m03_cache: dependency identity collision')
            value = self.read(key); self.counts['hit:'+kind] += 1
            self._timed(kind,'hit',started)
            return value, key
        for prior in compatible:
            old_key, old_encoded = self.key(kind,prior)
            row = self.db.execute('SELECT dependencies FROM nodes WHERE key=?',(old_key,)).fetchone()
            if row:
                if row[0] != old_encoded: raise ValueError('m03_cache: compatibility identity collision')
                value = self.read(old_key)
                self.counts['compatible_hit:'+kind] += 1
                self._timed(kind,'compatible_hit',started)
                return value,old_key
        compute_seconds = 0.
        if external is not None:
            path, checksum, selector = external
            if sha_file(path) != checksum: raise ValueError('m03_cache: import hash mismatch')
            value = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            value = value[selector] if selector else value
            args = (key,kind,encoded,checksum,None,str(Path(path).resolve()),selector)
            self.counts['import:'+kind] += 1
        else:
            compute_started = time.perf_counter()
            value = compute()
            compute_seconds = time.perf_counter()-compute_started
            stream = io.BytesIO(); torch.save(value,stream); data = stream.getvalue()
            args = (key,kind,encoded,hashlib.sha256(data).hexdigest(),data,None,None)
            self.counts['compute:'+kind] += 1
        # Computation and publication are separate; a failed/OOM node is absent.
        publish_started = time.perf_counter()
        self.db.execute('INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,?)',args); self.db.commit()
        result = self.read(key)
        self._timed(kind,'import' if external is not None else 'compute',started,compute_seconds,time.perf_counter()-publish_started)
        return result,key

    def close(self):
        self.save_timings(closed=True)
        self.db.close()


@contextmanager
def use_cache(cache):
    token = _ACTIVE.set(cache)
    try: yield cache
    finally: _ACTIVE.reset(token)


def active(): return _ACTIVE.get()


def memoized(settings_fields=METRIC_SETTINGS):
    """Cache actual function inputs and outputs, including probe predictions."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            cache = active()
            if cache is None: return function(*args,**kwargs)
            bound = signature.bind(*args,**kwargs); bound.apply_defaults()
            inputs = dict(bound.arguments)
            if 'settings' in inputs:
                inputs['settings'] = {k:getattr(inputs['settings'],k) for k in settings_fields}
            dependencies = {'implementation':cache.code(function),'inputs':inputs}
            entry = cache.compatibility.get('functions',{}).get(function.__module__+'.'+function.__name__)
            compatible = []
            if entry:
                if entry['current'] != dependencies['implementation']:
                    raise ValueError('m03_cache: compatibility proof does not match current '+function.__name__)
                compatible = [dict(dependencies,implementation=entry['prior'])]
            return cache.get(function.__name__,dependencies,lambda:function(*args,**kwargs),compatible=compatible)[0]
        return wrapped
    return decorate


def resolve_payload(payload):
    if 'cache_ref' not in payload: return payload
    ref = payload['cache_ref']
    if not Path(ref['database']).is_file(): raise FileNotFoundError(ref['database'])
    store = ArtifactCache(ref['database'])
    try: return dict(payload,features=store.read(ref['key']))
    finally: store.close()


def prepare_imports(source, *, root, settings, raw_checkpoint, tc_checkpoint, dataset, compatibility=None):
    """One-time migration of the retired gate's exact replay/encoding artifacts.

    Runtime/data hashes and extraction ASTs must match. Completed outcome/EDA
    components can migrate; the changed oracle implementation is recomputed.
    Old reports did not retain fitted prediction arrays: they are not invented.
    """
    import importlib.util
    import os
    import zipfile
    from . import gate, history, diagnostics
    source, root = Path(source).resolve(), Path(root).resolve()
    stored = json.loads((source/'run.json').read_text()); parent = stored['contract']
    if stored['contract_sha256'] != gate._sha(parent): raise ValueError('m03_import: corrupt source contract')
    result = {'features':{}, 'stages':{}, 'source':str(source), 'source_contract_sha256':stored['contract_sha256']}
    if parent['dataset']['sha256'] != sha_file(dataset): return result
    if any(parent['settings'][k] != getattr(settings,k) for k in REPLAY_SETTINGS): return result
    if Path(parent['evaluator_source']['path']).name != 'm03_capability.py': return result
    archive_path = source/'source.zip'
    if not archive_path.exists():
        # New runs already populate the dependency cache; don't guess about
        # an older changed evaluator if no exact source snapshot is available.
        return result
    expected = {}
    def identities(v):
        if isinstance(v,dict):
            if 'path' in v and 'sha256' in v: expected[v['path']] = v['sha256']
            for x in v.values(): identities(x)
        elif isinstance(v,list):
            for x in v: identities(x)
    identities(parent)
    proof = load_compatibility(compatibility) if compatibility else {}
    equivalent_sources = {}
    if proof:
        with zipfile.ZipFile(proof['reference_source']['path']) as snapshot:
            for name in ('gate','diagnostics'):
                equivalent_sources[name] = ast.parse(snapshot.read(f'd4mj/m03/{name}.py').decode())
    sources = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in ('capability','history','diagnostics','observability'):
            path = f'd4mj/m03_{name}.py'; data = archive.read(path)
            if hashlib.sha256(data).hexdigest() != expected[str(root/path)]: raise ValueError('m03_import: source archive changed')
            sources[name] = ast.parse(data.decode())
    retired = {str(root/f'd4mj/m03_{n}.py') for n in sources}
    weights = {parent[a+'_checkpoint']['path'] for a in ('raw','tc')}
    for path, checksum in expected.items():
        if path in retired or path in weights: continue
        if sha_file(path) != checksum: return result
    mapping = {'d4mj.m03_capability':'d4mj.m03.gate', 'd4mj.m03_history':'d4mj.m03.history',
               'd4mj.m03_diagnostics':'d4mj.m03.diagnostics','d4mj.m03_observability':'d4mj.m03.diagnostics'}
    def normalize(node, package):
        import copy
        node = copy.deepcopy(node); node.decorator_list = []
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str): node.body.pop(0)
        for item in ast.walk(node):
            if isinstance(item,ast.ImportFrom):
                module = importlib.util.resolve_name('.'*item.level+(item.module or ''),package) if item.level else item.module
                item.module = mapping.get(module,module); item.level = 0
        return ast.dump(node,include_attributes=False)
    def same(group, names, module):
        old = {n.name:n for n in sources[group].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        for name in names:
            fn = inspect.unwrap(getattr(module,name))
            node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
            if normalize(old[name],'d4mj') != normalize(node,'d4mj.m03'):
                entry = proof.get('functions',{}).get(fn.__module__+'.'+name)
                reference = next((n for n in equivalent_sources.get(module.__name__.split('.')[-1],ast.Module(body=[],type_ignores=[])).body
                                  if isinstance(n,ast.FunctionDef) and n.name==name),None)
                if not entry or entry['current'] != implementation(fn) or reference is None or normalize(old[name],'d4mj') != normalize(reference,'d4mj.m03'):
                    return False
        return True
    replay_same = (same('capability',('_state_scalars','_state_binary_labels','_action_outcomes','_recorded_step_key'),gate)
                   and same('history',('addresses','seed_split','verify_reference','replay_seed'),history)
                   and same('observability',('state_features',),diagnostics))
    if not replay_same: return result
    result['sidecar'] = str(source/'sidecar')
    encoding_same = {
        'raw': same('capability',('_encode_lewm',),gate),
        'tc': same('capability',('_encode_lewm',),gate),
        'direct_attention': same('capability',('_encode_legacy','_legacy_native_parity_preflight','_legacy_anchor','_load_legacy_cpu'),gate),
        'direct_mamba': same('capability',('_encode_legacy','_legacy_native_parity_preflight','_legacy_anchor','_load_legacy_cpu'),gate)}
    encoding_same['raw'] &= sha_file(raw_checkpoint) == parent['raw_checkpoint']['sha256']
    encoding_same['tc'] &= sha_file(tc_checkpoint) == parent['tc_checkpoint']['sha256']
    for arm in ('raw','tc'):
        encoding_same[arm] &= all(parent['settings'][k] == getattr(settings,k) for k in ('lewm_context','encode_batch'))
    for arm in ('direct_attention','direct_mamba'):
        encoding_same[arm] &= all(parent['settings'][k] == getattr(settings,k) for k in ('direct_context','direct_encode_batch','direct_successor_batch'))
    for directory in (source/'features',source/'historical/features'):
        for manifest_path in directory.glob('*.manifest.json'):
            arm, split = manifest_path.name.split('.')[:2]
            if arm != 'replay' and not encoding_same.get(arm): continue
            tensor_path = manifest_path.with_name(arm+'.'+split+'.pt')
            manifest = json.loads(manifest_path.read_text())
            payload = torch.load(tensor_path,map_location='cpu',weights_only=False,mmap=True)
            metadata = payload['metadata']
            if manifest['metadata_sha256'] != gate._sha(metadata) or (metadata['arm'],metadata['split']) != (arm,split):
                raise ValueError('m03_import: feature metadata mismatch')
            if arm == 'replay':
                if metadata['identity'] != {'protocol':history.PROTOCOL,'contract':sha_file(source/'run.json')}:
                    raise ValueError('m03_import: replay belongs to another run')
            else:
                if directory.name == 'features' and directory.parent.name == 'historical':
                    replay_manifest = directory/f'replay.{split}.manifest.json'
                    if metadata['sidecar_sha256'] != sha_file(replay_manifest): raise ValueError('m03_import: wrong replay parent')
                else:
                    if metadata['sidecar_sha256'] != sha_file(source/'sidecar/sidecar.probe_only.pt'): raise ValueError('m03_import: wrong primary parent')
                identity = metadata['identity']
                if arm in ('raw','tc'):
                    if identity.get('checkpoint',identity.get('sha256')) != parent[arm+'_checkpoint']['sha256']:
                        raise ValueError('m03_import: wrong model checkpoint')
                else:
                    if (identity['encoder']['sha256'] != parent['direct_anchors']['encoder']['sha256'] or
                        identity['world']['sha256'] != parent['direct_anchors']['attention' if arm.endswith('attention') else 'mamba']['sha256']):
                        raise ValueError('m03_import: wrong Direct checkpoint')
            result['features'][(arm,split)] = (str(tensor_path),manifest['sha256'],'features')
    metrics_same = same('capability',('_fit_probe_many','_fit_probe','_standardize','_ridge_predict','_root_bootstrap',
        '_binary_metrics','_regression_metrics','_paired_binary_difference','_paired_regression_difference',
        '_fit_label_coverage','_static_report','_outcome_report','_paired_semantic_differences',
        '_choice_summary','_mode_summary','_equivalence_summary','_context_summary'),gate)
    eda_same = same('diagnostics',('decompose','within_root_auc','transfer_summary','geometry_report','eda_report'),diagnostics)
    all_same = all(encoding_same.values()) and parent['settings'] == asdict(settings)
    for scope,directory in (('primary',source/'stages'),('historical',source/'historical/stages')):
        for path in directory.glob('*.json'):
            name = path.stem
            if ('parity_direct' in name or name.startswith('native_direct_parity_')):
                arm = 'direct_attention' if 'direct_attention' in name else 'direct_mamba'
                if encoding_same[arm]: result['stages'][(scope,name)] = (str(path),sha_file(path),None)
            elif all_same and metrics_same:
                if scope == 'primary' and name in ('static_retention','all_action_one_step','corrected_eda') and (name != 'corrected_eda' or eda_same):
                    result['stages'][(scope,name)] = (str(path),sha_file(path),None)
                elif scope == 'historical' and name in history.PANELS:
                    for component in ('all_action_one_step','eda'):
                        if component != 'eda' or eda_same:
                            result['stages'][(scope,name+'_'+component)] = (str(path),sha_file(path),component)
    result['encoding_eligible'] = encoding_same
    return result
