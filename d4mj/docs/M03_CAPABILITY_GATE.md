# M03 capability gate

`python -m d4mj.m03_capability` is the one sealed, evaluation-only gate for
completed M0--M3 joint LeWM checkpoints.  It is deliberately a capability
report, not a scalar promotion rule and not an authorization for M4.

The gate creates a fresh probe-TRAIN / held-out-DEV sidecar from exact
`support-v2` simulator replay. Every record retains a common 64-frame physical
prefix and its first incoming action. Direct receives that complete native input
stream; Raw and TC receive its final four frames and three outgoing actions. This
is a trained-system comparison, not a claim that the two systems consume the
same amount of history.

It verifies rendered roots and sampled replay trajectories against the stored
pixels, then forks every root through all 17 Craftax actions using the recorded
episode step key. The logged action must reproduce its stored successor before a
row is published, while all action alternatives share that same key. Root state
matching uses a declared maximum pixel tolerance
of one: the fixed 384-root preflight found 383 exact renders and one one-pixel,
one-count renderer-rounding discrepancy; the recorded manifest exposes both
the tolerance and all nonzero pixel elements.  Labels are held in the sidecar,
never inferred from model outputs:

- visible scalar state: health, food, drink, energy, materials and tools;
- visible/local state and source-derived action prerequisites;
- fork consequences: death, damage, positive reward, achievement, inventory
  change, and tile change; and
- multiple fresh-key simulator outcomes for descriptive deterministic-mode
  compatibility. These are explicitly separate from the factual primary fork.

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

For each Raw, TC, Direct-A, and Direct-M arm, the report records static retention
and one-step all-action semantic transfer, fatal/safe and reward ranking, and
action-effect equivalence. Existing Direct-64 results may be cited only where
the frozen root panel, split, context, and scoring procedure are identical; rich
semantic rows are cached from one new Direct-64 inference pass. No old head,
actor, policy, latent cache, or fitted probe is transplanted to LeWM. A truncated
Direct context is an optional sensitivity experiment, never the Direct headline.
Before either Direct arm is encoded, the gate also computes one row through the
archived evaluator's explicit full-prefix encoder/commit/world route and checks
it against M03's adapter route at a maximum absolute error of `1e-6`.  This
native-parity record is atomically cached and included in the final report, so a
resumed run cannot silently swap the 64-frame, first-incoming-action protocol.

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
  --out artifacts/lewm_gates_20260906/m03_capability_v2_native64
```

The report always writes `m4_authorized: false`.  It cannot test learned
outcome heads, policy KL/top-1, actor-own-state forks, or recursive horizons:
those require M4's separately authorized bridge/outcome-head and actor work.
