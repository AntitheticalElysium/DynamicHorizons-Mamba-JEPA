#!/usr/bin/env bash
# One arm at a time. Each process materialises the 12 GB latent cache and peaks near
# 16 GB of host RAM; two at once exceeded this machine's 31 GiB with no swap and took
# the host down. Both arms resume from their surviving checkpoints rather than restart.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in control paired; do
  out=$PWD/v2_ceiling_$arm
  if [ -f "$out/training_report.json" ]; then echo "[$(date +%H:%M:%S)] == $arm already complete"; continue; fi
  flag=""; [ "$arm" = "paired" ] && flag="--paired-semantic"
  echo "[$(date +%H:%M:%S)] == ceiling $arm (resuming)"
  $PY run_v2_phase2.py --arm attention --steps 2500 --freeze-world $flag \
      --source $PWD/v2_direct_attention --out "$out" \
    || { echo "[$(date +%H:%M:%S)] == FAILED $arm"; exit 1; }
  echo "[$(date +%H:%M:%S)] == $arm done, peak RSS $(grep VmHWM /proc/self/status 2>/dev/null || echo n/a)"
done
echo "[$(date +%H:%M:%S)] == both ceiling arms complete"
