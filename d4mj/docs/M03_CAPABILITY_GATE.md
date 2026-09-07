# M03 capability gate

`python -m d4mj.m03_capability` is the one sealed, evaluation-only gate for
completed M0--M3 joint LeWM checkpoints.  It is deliberately a capability
report, not a scalar promotion rule and not an authorization for M4.

The gate creates a fresh `DEV`-only sidecar from exact `support-v2` simulator
replay.  It verifies rendered roots and sampled replay trajectories against the
stored pixels, then forks every root through all 17 Craftax actions using common
root randomness.  Root state matching uses a declared maximum pixel tolerance
of one: the fixed 384-root preflight found 383 exact renders and one one-pixel,
one-count renderer-rounding discrepancy; the recorded manifest exposes both
the tolerance and all nonzero pixel elements.  Labels are held in the sidecar,
never inferred from model outputs:

- visible scalar state: health, food, drink, energy, materials and tools;
- visible/local state and source-derived action prerequisites;
- fork consequences: death, damage, positive reward, achievement, inventory
  change, and tile change; and
- multiple simulator outcomes for descriptive deterministic-mode compatibility.

All linear and fixed-MLP decoders are fitted on the sidecar's `TRAIN` split.
The same observed-successor decoder (including its train-derived
standardization) is then applied to observed, generated, and prefix-reset
successor states for both the six coarse consequences and the full successor
state: vitals, materials/tools, local tile state, and action prerequisites.
The report includes action-only, shuffled-action, root-plus-action, timestep,
and raw-pixel controls.  Confidence intervals bootstrap replay roots/episodes;
the 17 sibling action forks are never treated as independent examples.

The report also gives predeclared paired root-bootstrap differences for
generated versus observed states within Raw and TC; Raw/TC generated states
versus Direct-A and Direct-M; and TC generated versus Raw generated.  It does
not report Brier scores: class-weighted BCE is a discrimination/ranking probe,
not a calibrated probability model.  Mode diagnostics are explicitly labelled
as squared score distances only.

For each Raw, TC, Direct-A, and Direct-M anchor, the report records static
retention and one-step all-action semantic transfer, fatal/safe and reward
ranking, action-effect equivalence, and a four-frame prefix ablation where the
LeWM state supports it.  Direct is a historical comparison anchor only: no old
head, actor, or policy score is transplanted to LeWM.

The sealed full gate requires CUDA and the IEEE Triton execution setting.  It
allows exactly one checkpoint/environment delta: the recorded
`TRITON_F32_DEFAULT=unset` becomes `ieee` for evaluation.  Any other source or
environment drift aborts loading.  CPU `--smoke --skip-direct` is only an
end-to-end structural test and writes `structural_smoke_not_a_result`; it is
not a scientific comparison.  Direct-M's archived Triton implementation also
requires CUDA.

Runs are automatically resumable with the same `--out` directory.  A sealed
`run.json` hashes settings, checkpoints, dataset, evaluator, replay source, and
anchors.  The sidecar publishes atomically; each arm/split feature cache and
each expensive probe stage is independently content-checked.  A changed input
or partially published cache fails closed rather than being mixed into a new
result.

Example full invocation on the recorded environment:

```bash
TRITON_F32_DEFAULT=ieee .venv/bin/python -m d4mj.m03_capability \
  --device cuda \
  --out artifacts/lewm_gates_20260906/m03_capability
```

The report always writes `m4_authorized: false`.  It cannot test learned
outcome heads, policy KL/top-1, actor-own-state forks, or recursive horizons:
those require M4's separately authorized bridge/outcome-head and actor work.
