import pytest
import torch

from d4mj.m03.gate import (
    M03Settings,
    REPLAY_PIXEL_TOLERANCE,
    _binary_metrics,
    _direct_first_incoming_action,
    _fit_probe_many,
    _load_or_compute_stage,
    _mode_summary,
    _paired_binary_difference,
    _recorded_step_key,
    _ridge_predict,
)


def _settings():
    return M03Settings(train_roots=2, dev_roots=2, probe_hidden=4, probe_steps=2,
                       probe_batch=4, bootstrap_draws=8, minimum_positive=1,
                       minimum_negative=1)


def test_settings_reject_an_incompatible_direct_context():
    with pytest.raises(ValueError, match="context"):
        M03Settings(direct_context=16, lewm_context=4)
    assert REPLAY_PIXEL_TOLERANCE == 1


def test_recorded_step_key_uses_the_logged_outgoing_transition_key():
    class Replay:
        @staticmethod
        def _slot_keys(shard, slot):
            assert (shard, slot) == (7, 3)
            return "reset", ("step-0", "step-1", "step-2")

    assert _recorded_step_key(Replay(), {"shard": 7, "slot": 3, "t": 2}) == "step-2"


def test_direct_prefix_keeps_its_real_first_incoming_action():
    actions = torch.tensor([4, 2, 7]).numpy()
    assert _direct_first_incoming_action(actions, 0) == 17
    assert _direct_first_incoming_action(actions, 2) == 2


def test_ridge_control_returns_cpu_predictions_from_its_input_device():
    train_x = torch.tensor([[0.0], [1.0], [2.0]])
    train_y = torch.tensor([[0.0], [1.0], [2.0]])
    prediction = _ridge_predict(train_x, train_y, torch.tensor([[1.5]]), ridge=1.0)
    assert prediction.device.type == "cpu"
    assert prediction.shape == (1, 1)


def test_binary_metrics_bootstraps_one_root_and_all_action_targets_correctly():
    settings = _settings()
    # Static labels have just one value per root; all-action labels have 17.
    # Both must retain episode, rather than fork, as the resampling unit.
    static = _binary_metrics(torch.tensor([[3.0], [-3.0]]), torch.tensor([[True], [False]]),
                             torch.tensor([0, 1]), ("static",), settings)
    forks = _binary_metrics(torch.tensor([[[3.0]] * 17, [[-3.0]] * 17]),
                            torch.tensor([[[True]] * 17, [[False]] * 17]),
                            torch.tensor([0, 1]), ("fork",), settings)
    assert static["targets"]["static"]["auc"] == 1.0
    assert forks["targets"]["fork"]["auc"] == 1.0


def test_observed_generated_transfer_uses_one_fitted_train_decoder():
    settings = _settings()
    rng = torch.Generator().manual_seed(9)
    train = torch.randn(12, 3, generator=rng)
    labels = (train[:, :2] > 0).float()
    observed = torch.randn(5, 3, generator=rng)
    generated = torch.randn(5, 3, generator=rng)
    together = _fit_probe_many(train, labels, {"observed": observed, "generated": generated}, settings,
                               hidden=True, binary=True)
    separately = _fit_probe_many(train, labels, {"observed": observed}, settings, hidden=True, binary=True)
    torch.testing.assert_close(together["observed"], separately["observed"], atol=0, rtol=0)


def test_mode_summary_is_explicitly_advisory_for_deterministic_predictions():
    settings = _settings()
    logits = torch.zeros(2, 17, 6)
    truth = torch.zeros(2, 17, 6, dtype=torch.bool)
    modes = torch.zeros(2, 17, 2, 6, dtype=torch.bool)
    report = _mode_summary(logits, truth, modes, torch.tensor([0, 1]), settings)
    assert report["status"] == "advisory_only_deterministic_model"
    assert report["nearest_sampled_mode_mean_squared_score_distance"] == 0.25
    assert "brier" not in str(report).lower()


def test_paired_binary_difference_resamples_roots_not_individual_forks():
    settings = _settings()
    truth = torch.tensor([[[True]] * 17, [[False]] * 17])
    left = torch.tensor([[[3.0]] * 17, [[-3.0]] * 17])
    right = -left
    report = _paired_binary_difference(left, right, truth, torch.tensor([0, 1]), ("death",), settings)
    target = report["targets"]["death"]
    assert target["auc_difference"] == 1.0
    assert report["direction"] == "left_minus_right"


def test_stage_cache_reuses_the_same_verified_stage(tmp_path):
    calls = []

    def compute():
        calls.append(True)
        return {"answer": 7}

    metadata = {"feature_manifest": "sealed"}
    assert _load_or_compute_stage(tmp_path, "outcomes", metadata, compute) == {"answer": 7}
    assert _load_or_compute_stage(tmp_path, "outcomes", metadata, compute) == {"answer": 7}
    assert len(calls) == 1


import pytest
import torch

from d4mj.m03.gate import (M03Settings, _broad_time_range, _encode_legacy,
                                 _fit_label_coverage, _root_bootstrap, _load_or_encode_features)
from d4mj.m03.diagnostics import decompose, geometry_report, transfer_summary, within_root_auc, eda_report
from d4mj.m03.history import seed_split, split_indices, verify_reference


def settings():
    return M03Settings(train_roots=2, dev_roots=2, bootstrap_draws=8,
                       minimum_positive=1, minimum_negative=1, probe_steps=2)


def test_short_terminal_episode_cannot_supply_a_broad_root():
    assert _broad_time_range(64, settings()) is None
    assert _broad_time_range(65, settings()) == (63, 64)


def test_historical_split_matches_corrected_runner():
    import hashlib
    for seed in range(14000, 14512):
        draw = int.from_bytes(hashlib.sha256(f"paired-seed:{seed}".encode()).digest()[:8], "little") % 10
        assert seed_split(seed) == ("fit" if draw < 7 else "tune" if draw < 8 else "test")


def test_stress_panel_cannot_leak_a_fit_seed():
    fit = next(s for s in range(14000, 14512) if seed_split(s) == "fit")
    panels = {"exact961": [(fit, 10)], "policy104": [(fit, 11)]}
    with pytest.raises(ValueError, match="leakage"):
        split_indices([(fit, 10), (fit, 11)], panels, "policy104")


def test_replay_rejects_mismatched_successor_or_action():
    pixels = torch.zeros(17, 2, 2, 3, dtype=torch.uint8)
    args = ((14000, 63), {"successors": pixels.clone()}, pixels[:1], torch.tensor([17]),
            pixels, torch.zeros(17, dtype=torch.bool), torch.zeros(17), torch.zeros(17, dtype=torch.bool), 0)
    verify_reference(*args)
    args[1]["successors"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="successors mismatch"):
        verify_reference(*args)


def test_decomposition_does_not_confuse_action_prior_and_interaction():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(9, 17, 4, generator=generator).double()
    parts = decompose(x)
    torch.testing.assert_close(sum(parts.values()), x)
    torch.testing.assert_close(parts["interaction"].mean(0), torch.zeros(17, 4, dtype=torch.double))
    torch.testing.assert_close(parts["interaction"].mean(1), torch.zeros(9, 4, dtype=torch.double))


def test_matching_respects_equivalent_pixels_and_rejects_collapsed_effects():
    z = torch.eye(17)[None].repeat(2, 1, 1)
    pixels = torch.arange(17, dtype=torch.uint8)[None, :, None].repeat(2, 1, 3)
    report = geometry_report(z, z, torch.tensor([1, 2]), settings(), pixels)
    assert report["retrieval"]["value"] == 1
    assert report["effect_nse_energy_weighted"]["value"] == 0
    collapsed = geometry_report(torch.zeros_like(z), torch.zeros_like(z), torch.tensor([1, 2]), settings(), pixels)
    assert collapsed["retrieval"]["value"] is None
    assert collapsed["zero_effect_roots"] == 2


def test_perfect_action_prior_does_not_count_as_state_conditional_lift():
    truth = torch.tensor([[True, False] * 8 + [False]] * 4)
    score = truth.float()
    result = transfer_summary(score, score, score, truth, torch.tensor([1, 1, 2, 2]), settings())
    group = result["strata"]["all_opportunities"]
    assert group["seed_clusters"] == 2
    assert group["generated_minus_action_only"]["value"] == 0
    assert group["recovery"]["value"] == 1
    assert torch.isnan(within_root_auc(torch.ones(1, 17), torch.ones(1, 17, dtype=torch.bool))).all()


def test_short_direct_prefix_requires_true_episode_start():
    class Bundle:
        pass
    values = {"context": torch.zeros(1, 4, 2, 2, 3), "past_actions": torch.zeros(1, 3),
              "successors": torch.zeros(1, 17, 2, 2, 3), "direct_first_action": torch.tensor([2]),
              "time": torch.tensor([50])}
    with pytest.raises(ValueError, match="true episode BOS"):
        _encode_legacy(Bundle(), values, settings())


def test_seventeen_actions_from_one_seed_do_not_establish_label_coverage():
    config = M03Settings(minimum_positive=2, minimum_negative=2)
    labels = torch.tensor([[[True]]*17, [[False]]*17])
    result = _fit_label_coverage(labels, torch.tensor([1, 2]), ("death",), config)
    assert result["death"]["status"] == "insufficient_coverage"
    assert result["death"]["positive_seed_clusters"] == 1
    point, ci = _root_bootstrap(torch.ones(4), torch.ones(4), lambda rows: 1., draws=8, seed=1)
    assert point == 1 and ci is None


def test_eda_panel_scores_compound_damage_and_frozen_restoration_without_heads():
    z = torch.eye(17)[None].repeat(2, 1, 1)
    outcome = torch.zeros(2, 17, 6, dtype=torch.bool)
    outcome[:, :8, 0] = True
    data = {"outcomes": outcome, "episode": torch.tensor([1, 2]),
            "successors": torch.arange(17, dtype=torch.uint8)[None, :, None].repeat(2, 1, 3)}
    feature = {"train_projected": z[:, 0], "dev_projected": z[:, 0],
               "train_observed_successor": z, "dev_observed_successor": z,
               "dev_generated_successor": z}
    result = eda_report(data, data, {"raw": feature, "tc": feature}, settings(), device="cpu")
    assert "damage_or_death" in result["arms"]["raw"]["probes"]["linear"]["delta"]
    assert len(result["arms"]["raw"]["probes"]["linear"]["delta"]["oracle_residual_restoration"]) == 3
    assert result["paired_within_root_auc"]["tc_minus_raw"]["linear"]["death"]["difference"] == 0


def test_replay_shard_resume_rejects_different_input_contract(tmp_path):
    kwargs = dict(arm="replay", split="14000", identity={"protocol": "fixed"},
                  sidecar_sha256="contract-one", encode=lambda: {"context": torch.zeros(1, 1)})
    _load_or_encode_features(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="identity or hash"):
        _load_or_encode_features(tmp_path, **(kwargs | {"sidecar_sha256": "contract-two"}))


def test_replay_shard_resume_rejects_mutated_tensor_bytes(tmp_path):
    kwargs = dict(arm="replay", split="14000", identity={"protocol": "fixed"},
                  sidecar_sha256="contract-one", encode=lambda: {"context": torch.zeros(1, 1)})
    _load_or_encode_features(tmp_path, **kwargs)
    path = tmp_path / "features/replay.14000.pt"
    payload = torch.load(path, weights_only=False)
    payload["features"]["context"].fill_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity or hash"):
        _load_or_encode_features(tmp_path, **kwargs)


from d4mj.m03.diagnostics import (MEMORY_CASES, encode_memory, memory_view,
                                     shuffle_completed_pairs, memory_report)


def test_memory_shuffle_is_paired_causal_and_batch_independent():
    z = torch.arange(2*5).reshape(2,5,1,1).float()
    actions = z[:, :4, 0, 0].long()
    episodes, times = torch.tensor([9,10]), torch.tensor([4,4])
    zz, aa = shuffle_completed_pairs(z, actions, episodes, times, 7)
    torch.testing.assert_close(zz[:, -1], z[:, -1], atol=0, rtol=0)
    torch.testing.assert_close(zz[:, :4, 0, 0], aa.float(), atol=0, rtol=0)
    assert not torch.equal(aa, actions)
    for i in range(2):
        one, act = shuffle_completed_pairs(z[i:i+1],actions[i:i+1],episodes[i:i+1],times[i:i+1],7)
        torch.testing.assert_close(one,zz[i:i+1],atol=0,rtol=0)
        torch.testing.assert_close(act,aa[i:i+1],atol=0,rtol=0)


def test_memory_features_match_real_observe_and_preserve_bos_padding():
    from d4mj.tests.test_lewm import small_config
    from d4mj.world_api import ModelBundle
    from d4mj.sources import tensor_state_digest
    from dataclasses import replace
    bundle = ModelBundle.create(small_config()).eval()
    bundle.encoder.freeze(); bundle.world.requires_grad_(False)
    values = {'context': torch.randint(256,(2,16,14,14,3),dtype=torch.uint8),
              'context_length': torch.tensor([1,16]), 'past_actions': torch.randint(17,(2,15)),
              'successors': torch.randint(256,(2,17,14,14,3),dtype=torch.uint8),
              'episode': torch.tensor([2,3]), 'time': torch.tensor([0,15])}
    values['past_actions'][0] = 17  # Invalid if BOS padding leaks into prefill.
    before = tensor_state_digest(bundle.world.state_dict())
    result = encode_memory(bundle, values, _settings())
    with torch.inference_mode():
        for row,length in enumerate((1,16)):
            for case in ('c1','c4','c16','c64'):
                n = min(int(case[1:]),length)
                z = bundle.encode(values['context'][row:row+1,-n:])
                actions = values['past_actions'][row:row+1,-(n-1):] if n>1 else values['past_actions'][row:row+1,:0]
                state = bundle.prefill(z, actions)
                state = bundle.repeat_state(state,17)
                action = torch.arange(17)[:,None]
                real, _ = bundle.observe(state,action,values['successors'][row,:,None])
                generated, _ = bundle.advance(state,action)
                torch.testing.assert_close(result[f'{case}_next_h'][row],real.history[:,0])
                torch.testing.assert_close(result[f'{case}_next_h'][row],generated.history[:,0])
                torch.testing.assert_close(result['observed_z'][row],real.latent[:,0,0])
                torch.testing.assert_close(result[f'{case}_generated_z'][row],generated.latent[:,0,0])
        z_only = memory_view(result,'c4','z','root')
        h_only = memory_view(result,'c4','h','root')
        joint = memory_view(result,'c4','joint','root')
        torch.testing.assert_close(z_only+h_only,joint)
    assert tensor_state_digest(bundle.world.state_dict()) == before
    changed = dict(values); changed['context'] = values['context'].clone(); changed['context'][0,:-1] = 0
    second = encode_memory(bundle,changed,replace(_settings(),encode_batch=1))
    for name in result: torch.testing.assert_close(result[name],second[name])
    assert result['c4_root_h'][0].count_nonzero() == 0


def test_memory_successors_share_history_but_keep_predicted_latent_distinct():
    data = {'root_z': torch.zeros(2,3), 'observed_z': torch.zeros(2,17,3),
            'c4_root_h': torch.ones(2,5), 'c4_next_h': torch.ones(2,17,5),
            'c4_generated_z': torch.full((2,17,3),2.)}
    assert torch.equal(memory_view(data,'c4','h','observed'),memory_view(data,'c4','h','generated'))
    assert not torch.equal(memory_view(data,'c4','joint','observed'),memory_view(data,'c4','joint','generated'))
    assert memory_view(data,'c4','joint','observed').shape == memory_view(data,'c4','z','observed').shape


def test_archived_resume_rejects_source_tampering(tmp_path):
    import json, zipfile
    from d4mj.m03.gate import ROOT, _sha, resume_archived_baseline
    contract = {'evaluator_source': {'path':str(ROOT/'d4mj/m03_capability.py'),'sha256':'wrong'}}
    (tmp_path/'run.json').write_text(json.dumps({'contract':contract,'contract_sha256':_sha(contract)}))
    with zipfile.ZipFile(tmp_path/'source.zip','w') as archive:
        for name in ('capability','diagnostics','observability','history'):
            archive.writestr(f'd4mj/m03_{name}.py','raise RuntimeError("must never execute")')
    with pytest.raises(ValueError,match='evaluator bytes'):
        resume_archived_baseline(tmp_path,validate_only=True)


@pytest.mark.parametrize('hidden', [False, True])
def test_streamed_oracle_matches_original_dense_probe(hidden):
    from d4mj.m03.diagnostics import _fit_oracle_probe
    rng = torch.Generator().manual_seed(4)
    train, dev = torch.randn(9,7,generator=rng), torch.randn(3,7,generator=rng)
    labels = torch.randint(2,(9,17,6),generator=rng).float()
    def dense(x):
        return torch.cat((x[:,None].expand(-1,17,-1),torch.eye(17)[None].expand(len(x),-1,-1)),2).flatten(0,1)
    expected = _fit_probe_many(dense(train),labels.flatten(0,1),{'dev':dense(dev)},_settings(),hidden=hidden,binary=True)['dev']
    actual = _fit_oracle_probe(train,labels,dev,_settings(),device='cpu',hidden=hidden)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


def test_dependency_cache_reuses_across_runs_and_invalidates_only_changed_arm(tmp_path):
    from dataclasses import replace
    from d4mj.m03.cache import ArtifactCache,use_cache
    from d4mj.m03.gate import _load_or_encode_features
    cache = ArtifactCache(tmp_path/'cache.sqlite3',settings=_settings())
    calls = []
    def encode():
        calls.append(True)
        return {'projected':torch.ones(2,3)}
    keys = {}
    with use_cache(cache):
        for run in ('first','second'):
            out = tmp_path/run; out.mkdir()
            for arm in ('raw','tc','direct_attention','direct_mamba','replay'):
                identity = {'checkpoint':'changed' if run == 'second' and arm == 'raw' else 'original'}
                _,keys[run,arm] = _load_or_encode_features(out,arm=arm,split='train',identity=identity,sidecar_sha256='rows',encode=encode)
    assert len(calls) == 6
    for arm in ('tc','direct_attention','direct_mamba','replay'): assert keys['first',arm] == keys['second',arm]
    assert keys['first','raw'] != keys['second','raw']
    cache.settings = replace(_settings(),probe_steps=3,bootstrap_draws=20)
    out = tmp_path/'third'; out.mkdir()
    with use_cache(cache):
        _,key = _load_or_encode_features(out,arm='tc',split='train',identity={'checkpoint':'original'},sidecar_sha256='rows',encode=encode)
    assert key == keys['first','tc'] and len(calls) == 6
    # Recovery between tensor and manifest publication retains the dependency key.
    (out/'features/tc.train.manifest.json').unlink()
    with use_cache(cache):
        _,again = _load_or_encode_features(out,arm='tc',split='train',identity={'checkpoint':'original'},sidecar_sha256='rows',encode=encode)
    assert again == key
    cache.close()


def test_prediction_cache_settings_and_tensor_invalidation(tmp_path):
    from dataclasses import replace
    from d4mj.m03.cache import ArtifactCache,use_cache
    x = torch.arange(12).float().reshape(6,2)
    y = (x[:,:1] > 4).float()
    cache = ArtifactCache(tmp_path/'cache.sqlite3')
    with use_cache(cache):
        expected = _fit_probe_many(x,y,{'dev':x},_settings(),hidden=False,binary=True)
        again = _fit_probe_many(x.clone(),y.clone(),{'dev':x.clone()},replace(_settings(),bootstrap_draws=24),hidden=False,binary=True)
        assert cache.counts['compute:_fit_probe_many'] == 1
        torch.testing.assert_close(again['dev'],expected['dev'],atol=0,rtol=0)
        _fit_probe_many(x,y,{'dev':x},replace(_settings(),probe_steps=3),hidden=False,binary=True)
        _fit_probe_many(x,y+1,{'dev':x},_settings(),hidden=False,binary=True)
        assert cache.counts['compute:_fit_probe_many'] == 3
    cache.close()
    cache = ArtifactCache(tmp_path/'cache.sqlite3')
    with use_cache(cache):
        _fit_probe_many(x,y,{'dev':x},_settings(),hidden=False,binary=True)
    assert cache.counts['hit:_fit_probe_many'] == 1
    key = cache.db.execute('SELECT key FROM nodes LIMIT 1').fetchone()[0]
    cache.db.execute('UPDATE nodes SET payload=? WHERE key=?',(b'corrupt',key)); cache.db.commit()
    with pytest.raises(ValueError,match='corrupt'): cache.read(key)
    cache.close()


def test_memory_first_cli_orders_work_and_seals_combined_result(tmp_path,monkeypatch):
    from d4mj.m03 import gate,history,cache
    from pathlib import Path
    events = []
    monkeypatch.setattr(cache,'prepare_imports',lambda *a,**kw:{})
    monkeypatch.setattr(gate,'_run_contract',lambda **kw:{'settings':{}})
    def baseline(*,output,**kw):
        events.append('baseline')
        result = {'decision':{'suite_complete':True}}
        gate._atomic_json_save(output/'report.json',result)
        return result
    def memory(source,output,settings,**kw):
        events.append('memory')
        assert kw['allow_incomplete'] and kw['checkpoints']['raw'] == Path('changed.pt')
        output.mkdir()
        result = {'decision':{'suite_complete':False,'m4_authorized':False},'panels':dict.fromkeys(range(5),{})}
        gate._atomic_json_save(output/'report.json',result)
        return result
    monkeypatch.setattr(gate,'run_gate',baseline)
    monkeypatch.setattr(history,'run_memory',memory)
    assert gate.main(['--reuse-from',str(tmp_path/'old'),'--memory-first','--out',str(tmp_path/'new'),
                      '--cache',str(tmp_path/'cache.sqlite3'),'--raw-checkpoint','changed.pt']) == 0
    assert events == ['memory','baseline']
    import json
    assert json.loads((tmp_path/'new/complete.json').read_text())['decision']['suite_complete']


def test_new_run_reuse_does_not_attempt_retired_archive_migration(tmp_path):
    import json,zipfile
    from dataclasses import asdict
    from d4mj.m03.cache import prepare_imports,sha_file
    from d4mj.m03.gate import _sha
    dataset = tmp_path/'data.json'; dataset.write_text('{}')
    parent = {'dataset':{'sha256':sha_file(dataset)},'settings':asdict(_settings()),
              'evaluator_source':{'path':str(tmp_path/'d4mj/m03/gate.py')}}
    (tmp_path/'run.json').write_text(json.dumps({'contract':parent,'contract_sha256':_sha(parent)}))
    with zipfile.ZipFile(tmp_path/'source.zip','w') as archive: archive.writestr('d4mj/m03/gate.py','')
    result = prepare_imports(tmp_path,root=tmp_path,settings=_settings(),dataset=dataset,
                             raw_checkpoint=tmp_path/'raw.pt',tc_checkpoint=tmp_path/'tc.pt')
    assert not result['features'] and not result['stages']


@pytest.mark.parametrize('seed', [7,43])
@pytest.mark.parametrize('case', ['ties','constant','rare','single_seed','unequal'])
def test_weighted_auc_bootstrap_matches_explicit_episode_resampling(seed,case):
    from d4mj.m03.gate import _auc_bootstrap,_root_bootstrap
    from d4mj.diagnostics import binary_auc
    g=torch.Generator().manual_seed(seed)
    roots=torch.tensor([8,2,8,4,2,8,9,9,9,4,2,9])
    y=torch.randint(2,(12,),generator=g).bool()
    a=torch.randint(4,(12,),generator=g).float();b=torch.randn(12,generator=g)
    if case=='constant': a.zero_()
    if case=='rare':y.zero_();y[0]=True
    if case=='single_seed':roots.zero_()
    if case=='unequal':roots[:8]=19
    for paired in (False,True):
        def metric(rows):
            x=binary_auc(a[rows],y[rows]);z=binary_auc(b[rows],y[rows])
            return None if x is None else x-z if paired else x
        expected=_root_bootstrap(a,roots,metric,draws=101,seed=seed)
        actual=_auc_bootstrap(a,y,roots,right=b if paired else None,draws=101,seed=seed)
        assert actual==expected


@pytest.mark.parametrize('case',['normal','large_offset','constant_resamples','tiny_variance','one_seed'])
def test_sufficient_statistics_regression_preserves_intervals_and_support(case):
    from d4mj.m03.gate import _regression_bootstrap,_root_bootstrap
    rng=torch.Generator().manual_seed(29)
    y=torch.randn(60,generator=rng);a=y+torch.randn(60,generator=rng);b=y+2*torch.randn(60,generator=rng)
    roots=torch.arange(60)//7
    if case=='large_offset':y+=1e6;a+=1e6;b+=1e6
    if case=='constant_resamples':y=(roots%2).float()
    if case=='tiny_variance':y*=1e-7;a*=1e-7;b*=1e-7
    if case=='one_seed':roots.zero_()
    for paired in (False,True):
        def metric(rows):
            v=y[rows];d=(v-v.mean()).square().sum().clamp_min(1e-12)
            if paired and float(d)<=1e-12:return None
            x=float(1-(a[rows]-v).square().sum()/d)
            return x-float(1-(b[rows]-v).square().sum()/d) if paired else x
        expected=_root_bootstrap(a,roots,metric,draws=101,seed=14)
        actual=_regression_bootstrap(y,a,roots,right=b if paired else None,draws=101,seed=14)
        assert actual[0]==expected[0]
        if expected[1] is None: assert actual[1] is None
        else: torch.testing.assert_close(torch.tensor(actual[1]),torch.tensor(expected[1]),rtol=2e-5,atol=2e-5)


def test_draw_counts_are_the_exact_old_rng_stream():
    from d4mj.m03.gate import _bootstrap_multiplicities
    rng=torch.Generator().manual_seed(19)
    expected=torch.stack([torch.bincount(torch.randint(13,(13,),generator=rng),minlength=13) for _ in range(127)])
    assert (expected.numpy()==_bootstrap_multiplicities(13,127,19)).all()


def test_compatible_statistics_reuse_is_explicit_hash_checked_and_timed(tmp_path):
    import inspect,json
    from d4mj.m03.cache import ArtifactCache,use_cache,implementation,METRIC_SETTINGS,sha_file
    from d4mj.m03.gate import _binary_metrics
    source=tmp_path/'reference.zip';source.write_bytes(b'fixture')
    validation=tmp_path/'validation.json';validation.write_text('{"status":"pass"}')
    proof={'schema':'m03_bootstrap_cache_compatibility','status':'pass',
           'reference_source':{'path':str(source),'sha256':sha_file(source)},
           'validation':{'path':str(validation),'sha256':sha_file(validation)},
           'functions':{'d4mj.m03.gate._binary_metrics':{'prior':'old-validated-implementation','current':implementation(_binary_metrics)}}}
    path=tmp_path/'compatibility.json';path.write_text(json.dumps(proof))
    cache=ArtifactCache(tmp_path/'cache.sqlite3',compatibility=path,timing_path=tmp_path/'timing.json')
    args=(torch.tensor([[1.],[-1.]]),torch.tensor([[True],[False]]),torch.tensor([1,2]),('label',),_settings())
    bound=inspect.signature(_binary_metrics).bind(*args);bound.apply_defaults();inputs=dict(bound.arguments)
    inputs['settings']={k:getattr(_settings(),k) for k in METRIC_SETTINGS}
    cache.get('_binary_metrics',{'implementation':'old-validated-implementation','inputs':inputs},lambda:{'saved':True})
    with use_cache(cache):assert _binary_metrics(*args)=={'saved':True}
    assert cache.counts['compatible_hit:_binary_metrics']==1
    assert cache.counts['compute:_binary_metrics']==1  # Only the synthetic preexisting node.
    cache.close()
    timing=json.loads((tmp_path/'timing.json').read_text())
    assert timing['operations']['_binary_metrics']['compatible_hit_calls']==1
    assert timing['operations']['_binary_metrics']['compute_seconds']>=0
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='evidence bytes changed'):
        ArtifactCache(tmp_path/'cache.sqlite3',compatibility=path)


@pytest.mark.parametrize('mask_kind',['all','sparse','empty'])
def test_batched_seed_means_preserve_variable_episode_grouping(mask_kind):
    from d4mj.m03.gate import _bootstrap_mean_draws,_bootstrap_bounds,_root_bootstrap
    roots=torch.tensor([9,4,9,4,4,8,8,8,8,2])
    values=torch.tensor([[i/17.,(9-i)/17.,.5] for i in range(10)])
    mask=torch.ones(10,dtype=torch.bool)
    if mask_kind=='sparse':mask[:8]=False
    if mask_kind=='empty':mask[:]=False
    actual=_bootstrap_mean_draws(values,mask,roots,draws=127,seed=19)
    for i in range(3):
        def mean(rows):
            rows=rows[mask[rows]]
            return float(values[rows,i].mean()) if len(rows) else None
        _,expected=_root_bootstrap(values,roots,mean,draws=127,seed=19)
        interval=_bootstrap_bounds(actual[:,i],127)
        if expected is None:assert interval is None
        else:torch.testing.assert_close(torch.tensor(interval),torch.tensor(expected),atol=1e-7,rtol=1e-6)


def test_timing_accumulates_sessions_without_counting_stopped_time(tmp_path,monkeypatch):
    import json
    from d4mj.m03 import cache as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    path=tmp_path/'timing.json';db=tmp_path/'cache.sqlite3'
    first=module.ArtifactCache(db,timing_path=path)
    first.get('fixture',{},lambda:torch.ones(1))
    clock[0]=5.;first.close()
    clock[0]=100.
    second=module.ArtifactCache(db,timing_path=path)
    def forbidden():raise AssertionError('recomputed completed node')
    second.get('fixture',{},forbidden)
    clock[0]=102.;second.close()
    timing=json.loads(path.read_text())
    assert timing['active_seconds']==7.
    assert len(timing['sessions'])==2
    assert timing['operations']['fixture']['compute_calls']==1
    assert timing['operations']['fixture']['hit_calls']==1
    assert all(s['state']=='closed' for s in timing['sessions'])


def test_imported_stage_verifies_its_string_origin_path(tmp_path):
    """prepare_imports records stage origins as str; the verifier must accept one."""
    import json
    from d4mj.m03.cache import ArtifactCache, sha_file, use_cache
    from d4mj.m03.gate import _load_or_compute_stage
    origin = tmp_path / 'native_direct_parity_direct_attention.json'
    origin.write_text(json.dumps({'result': {'status': 'pass'}}))
    name = 'native_direct_parity_direct_attention'
    cache = ArtifactCache(tmp_path / 'cache.sqlite3',
                          imports={'stages': {('primary', name): (str(origin), sha_file(origin), None)}})
    first = tmp_path / 'first'; first.mkdir()
    with use_cache(cache):
        result = _load_or_compute_stage(first, name, {'anchor': 1},
                                        lambda: pytest.fail('an imported stage must not recompute'))
    assert result == {'status': 'pass'} and cache.counts['import:stage'] == 1
    origin.write_text(json.dumps({'result': {'status': 'fail'}}))
    second = tmp_path / 'second'; second.mkdir()
    with use_cache(cache), pytest.raises(ValueError, match='stage bytes changed'):
        _load_or_compute_stage(second, name, {'anchor': 1}, lambda: {})
    cache.close()
