import copy
from dataclasses import replace
import json

import pytest
import torch

from d4mj.data import save_episodes, _sha256
from d4mj.experiments import main
from d4mj.data import load_joint_corpus
from d4mj.gates import preflight, ComponentGateError
from d4mj.sources import tensor_state_digest
from d4mj.train import train_joint, learning_rate, set_phase_mode
from d4mj.world_api import ModelBundle
from d4mj.config import recipe_dict
from d4mj.tests.test_joint_data import raw_episodes
from d4mj.tests.test_lewm import small_config


@pytest.fixture
def ready(tmp_path):
    c=small_config(); path=tmp_path/"raw.pt"; save_episodes(path,raw_episodes())
    episodes,data=load_joint_corpus(path,c); gates=preflight(c,episodes,data)
    assert all(v["status"]=="pass" for v in gates["components"].values()), gates
    assert gates["m4"]["status"]=="blocked"
    return c,episodes,data,gates


def test_pause_resume_matches_full_run_and_preserves_screen_checkpoint(ready,tmp_path):
    c,episodes,data,gates=ready
    initial=ModelBundle.create(c)
    full,full_rows=train_joint(episodes,c,tmp_path/"full",dataset_contract=data,gate_report=gates)
    _,first=train_joint(episodes,c,tmp_path/"split",dataset_contract=data,gate_report=gates,stop_at=2)
    screen=tmp_path/"split/step-000002.pt"; before=_sha256(screen)
    resumed,second=train_joint(episodes,c,tmp_path/"split",dataset_contract=data,gate_report=gates,
                               stop_at=4,resume=tmp_path/"split/latest.pt")
    assert full_rows==first+second
    assert before==_sha256(screen)
    assert (tmp_path/"split/latest.pt").resolve()==(tmp_path/"split/step-000004.pt").resolve()
    index=json.loads((tmp_path/"split/checkpoints.json").read_text())
    assert index["snapshots"][screen.name]==before and len(index["snapshots"])==2
    for name in ("encoder","world"):
        expected=tensor_state_digest(getattr(full,name).state_dict())
        assert expected==tensor_state_digest(getattr(resumed,name).state_dict())
        assert expected!=tensor_state_digest(getattr(initial,name).state_dict())
    with pytest.raises(ValueError,match="resume_history"):
        train_joint(episodes,c,tmp_path/"split",dataset_contract=data,gate_report=gates,resume=screen)


def test_screen_snapshot_saved_even_off_regular_checkpoint_interval(ready,tmp_path):
    c,episodes,data,_=ready
    c=replace(c,joint=replace(c.joint,checkpoint_every=3))
    gates=preflight(c,episodes,data)
    train_joint(episodes,c,tmp_path/"run",dataset_contract=data,gate_report=gates)
    assert sorted(p.name for p in (tmp_path/"run").glob("step-*.pt")) == [
        "step-000002.pt","step-000003.pt","step-000004.pt"]


def test_failed_component_or_stale_contract_stops_before_training(ready,tmp_path):
    c,episodes,data,gates=ready
    broken=copy.deepcopy(gates); broken["components"]["recurrence"]={"status":"fail","reason":"fixture"}
    with pytest.raises(ComponentGateError,match="recurrence"):
        train_joint(episodes,c,tmp_path/"never",dataset_contract=data,gate_report=broken)
    assert not (tmp_path/"never").exists()
    with pytest.raises(ComponentGateError,match="gate_identity"):
        train_joint(episodes,c,tmp_path/"never",dataset_contract=data|{"changed":True},gate_report=gates)
    with pytest.raises(ComponentGateError,match="gate_identity"):
        train_joint(episodes,replace(c,variant="raw"),tmp_path/"never",dataset_contract=data,gate_report=gates)
    assert gates["architecture_verdict"]=="not_evaluated"


def test_later_stages_are_explicitly_blocked_and_schedule_is_fixed():
    c=small_config(); b=ModelBundle.create(c)
    for stage in ("bridge","actor","render-fit","play"):
        with pytest.raises(SystemExit) as e: main([stage])
        assert e.value.code==2
        with pytest.raises(ValueError,match="phase_gate"): set_phase_mode(b,stage)
    assert learning_rate(c,0)==c.joint.learning_rate
    assert learning_rate(c,c.joint.steps-1)==c.joint.min_learning_rate
    with pytest.raises(ValueError,match="sealed schedule"): learning_rate(c,c.joint.steps)


def test_cli_preflight_and_export_have_separate_artifact_contracts(tmp_path):
    c=small_config(); recipe=tmp_path/"recipe.json"; recipe.write_text(json.dumps(recipe_dict(c)))
    raw=tmp_path/"raw.pt"; save_episodes(raw,raw_episodes()); run=tmp_path/"run"
    argv=["preflight","--recipe",str(recipe),"--dataset",str(raw),"--out",str(run)]
    assert main(argv)==0
    assert (run/"dataset_audit.json").exists() and (run/"baseline_manifest.json").exists()
    assert main(argv)==1 and not (run/"failure.json").exists()  # existing run is untouched
    assert main(["joint","--run",str(run),"--stop-at","2"])==0
    checkpoint=run/"joint/step-000002.pt"
    export=["export","--run",str(run),"--checkpoint",str(checkpoint),"--out",str(tmp_path/"cache")]
    assert main(export)==1
    assert json.loads((run/"failure.json").read_text())["component"]=="joint_completion"
    assert main(export+["--diagnostic"])==0
    manifest=json.loads((tmp_path/"cache/manifest.json").read_text())
    assert manifest["cache"]["diagnostic_only"] and manifest["cache"]["joint_step"]==2
    assert manifest["cache"]["parent_checkpoint_path"]==str(checkpoint.resolve())
    assert manifest["cache"]["parent_checkpoint"]==_sha256(checkpoint)
