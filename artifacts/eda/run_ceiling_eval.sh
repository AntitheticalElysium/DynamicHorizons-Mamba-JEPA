#!/usr/bin/env bash
# Waits for the sequential ceiling unit to finish, smokes the evaluator, then runs it.
# The smoke is deliberate: a crash two minutes in is cheap, a crash after the full pass
# has re-simulated every DEV seed is not.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] == $*"; }

say "waiting for the ceiling arms"
while systemctl --user is-active --quiet ceiling; do sleep 60; done
for arm in control paired; do
  if [ ! -f "v2_ceiling_$arm/training_report.json" ]; then
    say "FAILED: v2_ceiling_$arm never completed"; exit 1; fi
done
say "both arms present"

# The evaluator drives the simulator and never loads the 12 GB latent cache, so it
# cannot repeat the host-RAM collision; it still runs only after training is done.
say "smoke, 2 seeds"
$PY check_paired_ceiling.py --episodes 2 --limit 40 \
    --out $PWD/../paired_ceiling_smoke > /dev/null 2>&1 \
  || { say "FAILED: evaluator smoke"; $PY check_paired_ceiling.py --episodes 2 --limit 40 \
       --out $PWD/../paired_ceiling_smoke 2>&1 | tail -20; exit 1; }
say "smoke clean"

say "full evaluation, 48 DEV seeds"
$PY check_paired_ceiling.py --episodes 48 --limit 300 || { say "FAILED: evaluation"; exit 1; }
say "ceiling evaluation complete"
