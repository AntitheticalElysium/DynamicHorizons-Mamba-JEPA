#!/usr/bin/env bash
# The discriminating head ceiling. Both arms score the heads only on the generated
# blocks; the treatment additionally scores the observed readout there, averaged. One
# process at a time: each materialises the 12 GB latent cache.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] == $*"; }
for arm in control paired; do
  out=$PWD/v2_rollout_$arm
  [ -f "$out/training_report.json" ] && { say "$arm already complete"; continue; }
  flag=""; [ "$arm" = "paired" ] && flag="--paired-semantic"
  say "rollout ceiling $arm"
  $PY run_v2_phase2.py --arm attention --steps 2500 --freeze-world --rollout-only $flag \
      --source $PWD/v2_direct_attention --out "$out" \
    || { say "FAILED $arm"; exit 1; }
done
say "both rollout ceiling arms complete"
