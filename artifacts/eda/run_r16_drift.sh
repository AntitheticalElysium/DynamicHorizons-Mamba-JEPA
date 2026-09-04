#!/usr/bin/env bash
# After Phase 2: semantic drift at depths 1..16 for the recursive arm and, as the matched
# baseline, for the existing rollout-2 arm. The baseline is measured past its trained
# depth on purpose -- that is the gap the recursive arm has to close.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] == $*"; }

say "waiting for r16 Phase 2"
while systemctl --user is-active --quiet r16-phase2; do sleep 60; done
[ -f v2_phase2_attention_r16/training_report.json ] || { say "FAILED: no Phase-2 report"; exit 1; }
say "Phase 2 present"

for arm in v2_phase2_attention v2_phase2_attention_r16; do
  say "drift depths 1-16: $arm"
  $PY check_generated_drift.py --arm-dir $PWD/$arm --episodes 48 --depth 16 \
      --ranking-roots 24 --critics || { say "FAILED: drift $arm"; exit 1; }
done
say "r16 drift comparison complete"
