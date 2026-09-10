from dataclasses import replace
import copy
import json

import pytest
import torch

from d4mj.config import config_from_dict, recipe_dict
from d4mj.data import screen_windows, save_episodes, load_joint_corpus
from d4mj.diagnostics import binary_auc, paired_auc_interval, fit_outcome_probe
from d4mj.experiments import run_joint_pair
from d4mj.gates import ComponentGateError, require_joint_screen
from d4mj.lewm_config import ScreenConfig
from d4mj.lewm_diagnostics import covariance_summary
from .test_lewm import small_config
from .test_joint_data import raw_episodes


def settings():
    return ScreenConfig(train_episodes=2,dev_episodes=2,windows_per_episode=2,encode_batch=2,
                        probe_steps=3,probe_batch=8,probe_hidden=8,bootstrap_draws=40,
                        minimum_positive=1,minimum_negative=1)


def corpus():
    episodes = raw_episodes()
    episodes.append(replace(episodes[2],episode_id='another_dev',observations=episodes[2].observations+3))
    return episodes


def test_screen_selection_is_id_stable_episode_split_and_action_aligned():
    c=small_config();s=settings();episodes=corpus()
    for split in ('train','dev'):
        a=screen_windows(episodes,c,s,split);b=screen_windows(list(reversed(episodes)),c,s,split)
        assert a['episode_ids']==b['episode_ids'] and torch.equal(a['starts'],b['starts'])
        source={e.episode_id:e for e in episodes}
        for i,(eid,start) in enumerate(zip(a['episode_ids'],a['starts'])):
            e=source[eid];assert e.split==split
            assert torch.equal(a['frames'][i],e.observations[start:start+4])
            assert torch.equal(a['actions'][i],e.actions_taken[start:start+3])
            assert torch.equal(a['labels'][i,:,0],e.rewards[start:start+3]>0)
            assert torch.equal(a['labels'][i,:,3],e.terminated[start:start+3])
            assert not a['valid'][i,:,2].any()  # absent events are not negative labels
    with pytest.raises(ValueError,match='FINAL'):screen_windows(episodes,c,s,'final')
    with pytest.raises(ValueError,match='coverage'):screen_windows(episodes,c,replace(s,dev_episodes=3),'dev')
    assert config_from_dict(recipe_dict(s))==s
    with pytest.raises(ValueError,match='unknown'):config_from_dict(recipe_dict(s)|{'mystery':1})


def test_auc_ties_missing_classes_and_episode_cluster_bootstrap():
    y=torch.tensor([0,1,0,1],dtype=torch.bool)
    assert binary_auc(torch.tensor([0.,0.,1.,1.]),y)==.5
    assert binary_auc(y.float(),y)==1 and binary_auc(-y.float(),y)==0
    assert binary_auc(y.float(),torch.ones_like(y)) is None
    with pytest.raises(ValueError,match='nonfinite'):binary_auc(torch.full((4,),float('nan')),y)
    truth=y[:,None].repeat(8,1);scores=truth.float();clusters=torch.arange(8).repeat_interleave(4)
    report=paired_auc_interval(scores,-scores,truth,torch.ones_like(truth),clusters,draws=40,seed=5)
    assert report['difference']==1 and report['interval']==[1.,1.] and report['unit']=='episode'


def test_probe_is_deterministic_and_dev_is_never_used_for_fitting():
    rng=torch.Generator().manual_seed(8)
    x=torch.randn(32,6,generator=rng);y=(x[:,:2]>0).float();valid=torch.ones_like(y)
    dev=torch.randn(12,6,generator=rng)
    a=fit_outcome_probe(x,y,valid,dev,settings(),hidden=True)
    b=fit_outcome_probe(x,y,valid,torch.cat((dev,dev+100)),settings(),hidden=True)
    torch.testing.assert_close(a,b[:12],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(a,fit_outcome_probe(x,y,valid,dev,settings(),hidden=True),atol=0,rtol=0)
    assert covariance_summary(torch.ones(5,4))['coordinate_variance']==0
    with pytest.raises(ComponentGateError,match='finiteness'):covariance_summary(torch.full((5,4),float('inf')))


def test_paired_pipeline_saves_initialization_and_runs_component_screen(tmp_path):
    c=small_config();configs={'raw':replace(c,variant='raw'),'tc':c}
    dataset=tmp_path/'raw.pt';save_episodes(dataset,corpus());out=tmp_path/'pair'
    result=run_joint_pair(configs,settings(),dataset,out,screen_only=True)
    report=json.loads((out/'G1/screen.json').read_text())
    assert report['components']['pair_identity']['status']=='pass'
    assert report['components']['objective_contrast']['status']=='pass'
    assert report['decision'] in ('continue_joint_budget','review_required'),report
    assert result==(0 if report['decision']=='continue_joint_budget' else 1)
    assert report['arms']['raw']['initial_identity']==report['arms']['tc']['initial_identity']
    assert (out/'raw/joint/step-000000.pt').exists() and not report['m4_authorized']
    _,data=load_joint_corpus(dataset,c)
    parent=out/'tc/joint/step-000002.pt'
    if report['decision']=='continue_joint_budget':
        require_joint_screen(report,c,data,parent)
        from d4mj.train import train_joint
        episodes,_=load_joint_corpus(dataset,c)
        gates=json.loads((out/'tc/gates.json').read_text())
        train_joint(episodes,c,out/'tc/joint',dataset_contract=data,gate_report=gates,
                    stop_at=4,resume=parent,screen_report=report)
        require_joint_screen(report,c,data,out/'tc/joint/step-000004.pt')
    changed=copy.deepcopy(report);changed['decision']='changed'
    with pytest.raises(ComponentGateError,match='identity'):require_joint_screen(changed,c,data,parent)
    with pytest.raises(ComponentGateError,match='joint_screen'):require_joint_screen(None,c,data,parent)


def test_continuation_verifies_parent_evidence_and_records_screen_lineage(tmp_path):
    """Synthetic passing diagnostics exercise authorization, not scientific success."""
    from d4mj.gates import contract_digest
    from d4mj.train import train_joint
    c=small_config();dataset=tmp_path/'raw.pt';save_episodes(dataset,corpus());out=tmp_path/'pair'
    run_joint_pair({'raw':replace(c,variant='raw'),'tc':c},settings(),dataset,out,screen_only=True)
    report=json.loads((out/'G1/screen.json').read_text())
    assert all(v['status']=='pass' for v in report['components'].values()),report
    report['decision']='continue_joint_budget'
    for item in report['arms'].values():
        item['learning_progress']=True
        item['prediction']['normalized_prediction_mse']=.5
        item['initial_prediction']['normalized_prediction_mse']=1.
        item['retention']['projection_stop']=False
    report['report_id']=contract_digest({k:v for k,v in report.items() if k!='report_id'})
    episodes,data=load_joint_corpus(dataset,c);parent=out/'tc/joint/step-000002.pt'
    require_joint_screen(report,c,data,parent)
    with pytest.raises(ComponentGateError,match='parent'):
        require_joint_screen(report,c,data,out/'raw/joint/step-000002.pt')
    with pytest.raises(ComponentGateError,match='parent'):
        require_joint_screen(report,c,data,out/'tc/joint/step-000000.pt')
    gates=json.loads((out/'tc/gates.json').read_text())
    train_joint(episodes,c,out/'tc/joint',dataset_contract=data,gate_report=gates,
                stop_at=4,resume=parent,screen_report=report)
    require_joint_screen(report,c,data,out/'tc/joint/step-000004.pt')
    artifact=out/'G1/windows.json';original=artifact.read_bytes();artifact.write_bytes(original+b' ')
    with pytest.raises(ComponentGateError,match='identity'):
        require_joint_screen(report,c,data,parent)


def test_research_continuation_without_screen_stops_before_new_updates(tmp_path):
    from d4mj.train import train_joint
    from d4mj.gates import preflight
    # Full statistical dimensions, small model, CPU purpose is still not a GPU authorization.
    c=small_config();c=replace(c,joint=replace(c.joint,batch=128,projections=1024,knots=17),
                               runtime=replace(c.runtime,purpose='research'))
    dataset=tmp_path/'raw.pt';save_episodes(dataset,corpus());episodes,data=load_joint_corpus(dataset,c)
    report=preflight(c,episodes,data)
    with pytest.raises(ComponentGateError,match='joint_resource'):
        train_joint(episodes,c,tmp_path/'run',dataset_contract=data,gate_report=report,stop_at=4)
    assert not (tmp_path/'run').exists()
