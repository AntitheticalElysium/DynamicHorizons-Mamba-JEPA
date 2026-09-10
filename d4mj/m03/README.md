# M03 capability gate

`python -m d4mj.m03.gate` is the one sealed, evaluation-only gate for
completed M0--M3 joint LeWM checkpoints. The final combined report is
`<out>/complete.json`, which hash-links `<out>/memory/report.json` and
`<out>/report.json` (the baseline panels).
It is deliberately a capability
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

The corrected EDA annex adds successor-only probes (without an action-identity
input), observed-to-generated delta and root-centred action-effect probes,
FIT-marginal-residualised probes, within-root AUC and recovery, paired lift over
the action-only floor, escape-rich/middle/trap-heavy strata, persistence and
candidate-action-shuffle controls. Geometry is evaluated within each arm:
energy-weighted action-effect error, exact common/effect error decomposition,
root/action/interaction variance, and equivalence-aware nearest-successor and
Hungarian matching. Equivalence uses the actual successor pixels, independently
of the model's representation and independently of the six coarse outcome bits.
The EDA annex also retains the historical compound damage-or-death target,
per-action matching, class rank/margin, pairwise-distance geometry, paired
within-root AUC differences and an explicitly oracle-only residual restoration
curve. Restoration is read by the same frozen observed-delta decoder.

Fresh pre-state observability controls compare pixels, visible descriptors,
visible descriptors plus hidden counters/cooldowns, and full simulator state.
The descriptor follows the historical nearest-three-mobs-per-type design; its
failure alone does not prove the pixel observation is insufficient. These are
privileged diagnostic probes, never inputs to the evaluated world model.
Coverage counts distinct episodes/seeds carrying each class, in both probe FIT
and evaluation. Seventeen sibling action rows cannot substitute for seventeen
independent positive examples; a single seed cannot yield a confidence interval.

Four historical panels run through the same scoring functions by default:

| Panel | Probe fit | Evaluation |
| --- | --- | --- |
| `exact961` | 668 roots; 96 tune roots reserved | 197 roots / 88 rollout seeds |
| `policy104` | The `exact961` FIT partition | All 104 stress roots / 52 seeds |
| `hazard5402` | 3,820 FIT roots; 513 tune roots reserved | 1,069 test roots |
| `legacy751` | The `exact961` FIT partition | All 751 historical stress roots |

The original `paired-seed:` SHA256 split is preserved for the EDA fork-world
panels. Historical tune roots are unused because M03 uses a fixed probe recipe;
this is a new common rerun, not a reproduction of old fitted probes or scalar
scores. Panels overlap, and the report lists their intersections. They must not
be summed as independent evidence. All historical uncertainty resamples whole
rollout seeds, retaining their multiple roots and all 17 sibling actions.

Historical files supply addresses, pixels, incoming actions and simulator truth.
The evaluator replays the saved complete action streams when available; the
archived 32-slot policy is used only to reconstruct the old 13,000-series
trajectories that lack those streams. Every available root/history, successor
pixel, action, death, reward and damage reference must match before publication.
All rich labels and all encodings are regenerated. Historical replay shards and
four-model encodings are atomically cached per seed and shared across panels.

Episode-start prefixes retain their actual length, including the 25 short
prefixes in the 197-root test panel. Storage padding is removed before inference;
Direct consumes up to 64 real frames and LeWM up to four. Short Direct prefixes
must start at a true BOS. Native Direct parity is checked for every historical
seed/length group before its feature cache is published.

See the omission audit and experiment mapping below for the
complete mapping of the attached historical experiment families, including
diagnostics requiring further adapters and training/control work outside M03.

The sealed full gate requires CUDA and the IEEE Triton execution setting.  It
allows exactly one checkpoint/environment delta: the recorded
`TRITON_F32_DEFAULT=unset` becomes `ieee` for evaluation.  Any other source or
environment drift aborts loading.  CPU `--smoke --skip-direct --skip-history` is only an
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
TRITON_F32_DEFAULT=ieee .venv/bin/python -m d4mj.m03.gate \
  --device cuda \
  --out artifacts/lewm_gates_20260906/m03_capability_complete
```

Use a new output directory for this corrected contract. Old stopped directories
are preserved and cannot be resumed under changed code. `--skip-history` and
`--skip-direct` explicitly mark the suite incomplete. A bounded four-model GPU
smoke uses `--device cuda --smoke` and samples historical fit/test/tune and policy
roots; it retains `structural_smoke_not_a_result` regardless of metric values.

The report always writes `m4_authorized: false`.  It cannot test learned
outcome heads, policy KL/top-1, actor-own-state forks, or recursive horizons:
those require M4's separately authorized bridge/outcome-head and actor work.



## Mamba-state supplement

The frozen world is tested at its intended readout inputs: projected encoder
`z`, previous completed-pair Mamba output `h`, and `[z,h]`. The untrained agent
readout is not treated as an acquired capability. The `z` and `h` ablations use
zero-filled slots in the same concatenated input so the probe parameter count,
hidden width and optimization recipe remain identical.

Probes fit only the valid four-frame TRAIN/FIT condition. That single fitted
probe and its TRAIN normalization are applied to valid 1/4/16/64-frame prefixes
and shuffled completed-pair histories at 4/16/64. Shuffling permutes past
`(z_i,a_i)` pairs within a root, holds the current observation and candidate
action fixed, and is independent of batch ordering. It is an intervention,
not a valid simulator trajectory. Prefixes shorter than a requested context
retain their true BOS length; coverage reports the available lengths. Longer
prefix results are decoder-transfer diagnostics beyond the joint training span,
not evidence of training with persistent 64-step memory. The TC regularizer's
four-frame window is unchanged.

For successors, the world consumes `(z_t,a_t)` exactly once. Observed and
generated branches use the same resulting `h_next` and carry; only accepted
`z_next` changes. Tests compare the extraction to real `observe` and `advance`.
Static visible state, rich successor state, and the six consequence targets plus
compound damage-or-death are tested with linear and fixed-MLP probes. Reports
include paired seed-bootstrap differences for context interventions,
generated-minus-observed, joint-minus-z, TC-minus-Raw, and action-only controls
with within-root action ranking. Internal conv/SSM RMS values are numerical
diagnostics, not semantic pass criteria. Neither the supplementary report nor
baseline coverage automatically promotes TC or authorizes M4.

A companion reuses the baseline's hash-checked raw sidecar and historical replay
shards, and computes Mamba features. With `--memory-first`, it scores the primary
Mamba panel before historical Mamba encoding, then finishes historical Mamba
panels before the remaining baseline work. Four-frame conditions precede the
other prefix conditions. `memory/partial.json` publishes completed task results;
`memory/status.json` identifies current work. A partial result cannot declare the
suite complete or authorize M4.

```bash
# Optimized production run; keep the original source-locked run intact.
TRITON_F32_DEFAULT=ieee OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  JAX_PLATFORMS=cpu .venv/bin/python -m d4mj.m03.gate \
  --reuse-from artifacts/lewm_gates_20260906/m03_full --memory-first \
  --cache artifacts/lewm_gates_20260906/cache.sqlite3 \
  --cache-compatibility artifacts/lewm_gates_20260906/m03_bootstrap/compatibility.json \
  --out artifacts/lewm_gates_20260906/m03_bootstrap/evaluation --device cuda
```

The shared `--cache` defaults to `artifacts/lewm_gates_20260906/cache.sqlite3`.
Immutable run directories contain small feature references. Keep that database
and any imported source artifacts: cache reads verify their byte hashes.
Dependency keys cover input values, relevant settings, model/runtime identity
and transitive evaluator functions. A changed Raw checkpoint recomputes its
features and affected probes/comparisons; unchanged TC, Direct and raw replay
remain reusable. Probe-setting changes preserve encodings. Bootstrap-setting
changes preserve fitted prediction arrays. Cross-arm comparisons are recomputed
when either side changes. Local source-locked runs still require a new output
when code, inputs or settings change.

Migration of the original run requires its exact four-module `source.zip`,
matching extraction code, runtime/data hashes and per-arm checkpoint identities.
Compatible completed outcome/EDA stages are imported; the corrected oracle is
recomputed. Historical outcome, EDA and observability stages publish separately,
so an oracle failure no longer discards the two preceding components. The old
reports did not save fitted prediction arrays: a new comparison involving an
old aggregate-only result can require a one-time probe refit. New runs persist
those arrays and their statistics in the shared cache.

Bootstrap statistics use the original per-draw Torch RNG stream, stored as
per-seed multiplicities. AUC sorts once and handles tied-score weighted counts;
regression uses centered per-seed sufficient statistics with the original
calculation retained for degenerate or numerically sensitive cases. Within-root
transfer means use the same grouped draws, with a reference fallback near chance.
The 1,000 draws, coverage rules, seed pairing and quantile convention are unchanged.

The explicit compatibility proof hash-pins the archived reference, numerical
validation and five changed function identities. Only identical inputs/settings
can reuse their previous cached statistics. It never rewrites their payloads.
The remaining encodings and probe predictions retain their original keys.
Source/code changes outside that validated equivalence fail closed.

`<out>/timing.json` updates during execution and retains each process session.
It sums active wall/CPU time across resumes, excluding stopped time, and separates
computed operations, ordinary cache hits, compatible cache hits and publication
cost. Nested operation durations are not additive across categories. The paired
reference/optimized measurements and reproducible benchmark are under
`artifacts/lewm_gates_20260906/m03_bootstrap/`: 27.43s versus 2.21s on the selected
uncached statistical workload (12.4x), not an end-to-end gate ETA. AUC outputs
matched exactly; the largest regression/mean difference was 2.4e-7.

The oracle probe samples root/action pairs and evaluates minibatches without
materializing seventeen copies of every pixel feature on CUDA. Its objective,
sampling, normalization, seed and probe capacity are retained. The original
failed log/source archive remains under `artifacts/`; `--resume-baseline` is
retained for exact archival reproduction, including the original memory limit.


# M03 experiment omission audit and corrected contract

Audit date: 2026-09-08. Scope: the three user-supplied transcripts, the current
M03 runner, and the corresponding Direct/EDA collectors and evaluators. This is
an experiment-family mapping, not a claim to inventory every physical artifact.
Historical scalar results are not accepted as new LeWM results.

## Findings against the starting implementation

The two old replay criticisms were already repaired by commit `adf4b79`:
native Direct prefix 64 with the real first incoming action and native-route
parity; recorded support episode RNG with logged-successor verification.
The working tree also contained a broad-root time-range fix. It is preserved.

The material omissions were historical hard-state coverage and several different
experimental questions hidden under overly broad names:

1. `build_sidecar` only selected fresh support TRAIN/DEV episodes. No historical
   961/197, 104, 5,402 or 751 panel was loaded.
2. `_equivalence_summary` grouped six consequence bits and compared sigmoid
   score distances. It did not test successor identity, matching, geometry or
   state/action interaction. Two successors can share those bits but differ in
   tiles, inventory or pixels.
3. `_outcome_report` appended the requested action to every successor decoder.
   That is an action-conditioned semantic diagnostic. It does not reproduce
   the historical successor-only frozen decoder, delta transfer or within-root
   AUC/recovery question.
4. The existing shuffle trained a root-plus-shuffled-action probe. It did not
   shuffle the world's candidate successors against the requested action.
5. Static pixels/timestep, root+action and action-only controls existed; an
   explicit state-only probe, constant control and frozen-decoder persistence
   control did not. There were no escape-rich/trap-heavy transfer strata.
6. `_decision` only determined whether targets were measurable; it did not
   establish that retention or transition performance was adequate. The output
   remains a capability report, with no invented promotion thresholds.

## Verified physical panels and split traps

Counts below were read from the current tensor payloads, using the split
implemented in `artifacts/eda/train_phase1b_fork.py:seed_split`.

| Physical source | Verified inventory | Reuse contract |
| --- | --- | --- |
| `artifacts/eda/fork_histories/branched_965.pt` joined to `fork_successors/shard-*.pt` | 961 matches; FIT 668, TUNE 96, TEST 197; TEST has 88 seed clusters | Central regression panel; freshly re-encode; reserve TUNE |
| `artifacts/eda/fork_histories/policy_fork_104.pt` | 104 roots, 52 distinct seeds, all saved prefixes length 64 | Entire panel evaluation-only; fit probes on separate 961 FIT seeds |
| `artifacts/eda/fork_successors/shard-*.pt` | 5,402 unique addresses; FIT 3,820, TUNE 513, TEST 1,069 | Broad hazard panel; saved complete incoming-action streams in `branched_damage` |
| `artifacts/eda/forkset_s1_n64/shard-*.pt` | 5,402 rows before filtering; 5,146 with a full 32-frame history; filtered FIT 3,651, TUNE 473, TEST 1,022 | Explains conflicting transcript counts; old latent payload is not reused |
| `artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt` | 751 state rows with 17-action death/reward truth | Historical stress panel; freshly reconstruct physical states; score terminal opportunities as a subset |
| `artifacts/eda/multistep_forks/rollouts.pt` | 197 roots, 88 seeds; 17 branches, four depths, pixels and termination depth | Valid secondary pixel source; recursive evaluation is separately deferred |

The 961 panel has 25 TEST roots before timestep 63. Requiring a 64-frame tensor
without variable-length handling would silently change the panel. The new
adapter removes storage padding and uses actual episode-start histories.

The correct historical split is SHA256 of `paired-seed:<seed>`, first eight
digest bytes interpreted little-endian, modulo ten, FIT <7 / TUNE <8 / TEST
otherwise. The `all-action-seed:` split used by a different classifier family
must not be substituted. The 104 stress panel is not repartitioned under either
hash. Multiple roots from one seed remain together for confidence intervals.

## Every experiment family named in the attached runner transcript

“Implemented” means callable in the corrected M03 runner, not a completed full
scientific run. “Partial” identifies the exact retained question and remaining
work. Training interventions are not smuggled into an evaluation gate.

| Transcript family | Historical source | M03 disposition |
| --- | --- | --- |
| 3: fork/replay/ground truth | `run_action_conditioning_diagnostics.sh` | Implemented: 751 addresses, fresh exact replay and all-action labels; native contexts; old heads excluded |
| 5: S75/S76 Flow–Direct | `run_s76_paired_flow_s35.sh` | Paired observed/generated semantic methodology implemented; repeated simulator modes remain advisory. Flow sampling and trained continuation-head factorial deferred |
| 6: terminal dynamics/action conditioning | `run_terminal_dynamics_ablation.sh`, `action_shuffle_dev.py` | Candidate-action shuffle, action floor, persistence and semantic transfer implemented. Terminal-mass retraining is a new treatment, outside the evaluator |
| 7: transition localization | `localize_direct_transition_stages.py` | Observed/generated and delta/residual localization implemented. Internal predictor-tap/decoder-ceiling adapter remains additional work |
| 8: fatality identifiability/encoder fidelity | `measure_encoder_fatality_fidelity.py`, `diagnose_fatality_identifiability.py` | Static CLS/export and successor/delta transfer implemented. The full action-matched natural-terminal census and probe-capacity ladder are not reproduced |
| 10: counterfactual localization | `localize_counterfactual.py` | Same 751 stress population plus within-root fatal/safe readings implemented. Old leave-one-root folds and trained readout/head localization are not transplanted |
| 11: identifiability gate | `run_identifiability_gate_v2.py` | All 104 paired stress roots implemented; independent probe FIT population and whole-seed uncertainty. Original support-size/probe-seed scaling ladder deferred |
| 20: recursive generated outcome shaping | `run_generated_latent_outcome_shaping.py` | Its 104-root evaluation question reused. Optimizer treatments, generated outcome-head fitting and recursion excluded from M0–M3 |
| 22: consequence learnability | `benchmark_consequence_learnability.py`, `evaluate_consequence_learnability.py` | 104 and 961 all-action held-out consequence probes implemented. Full 3.1M-transition supervised retraining/pixel-classifier treatment deferred |
| 23: branched coverage | `run_branched_coverage_gate.py` | Central 961 physical join and 104 external stress panel implemented. Four missing successor addresses remain excluded with the explicit 965-to-961 join |
| 26: generated semantic drift | `run_r16_drift.sh`, `check_multiworld_drift.py` | One-step observed/generated semantics implemented. Policy KL/top-1, critic and recursive drift require the later control pipeline |
| 27: Raw versus TC | `lewm_gates_20260906/launch.json`, `d4mj.m03.gate` | Both frozen joint checkpoints and common Direct-A/Direct-M anchors supported; paired comparisons remain descriptive for cross-architecture rows |
| 30: damage/fatality probes | `run_triage.sh`, `run_round3.sh` | Frozen consequence transfer and action floor implemented. The separate 9,146-root TRAIN corpus/pixel-classifier/all-action retraining family is not a pristine DEV panel and is not added as one |
| 32: replay preparation | `artifacts/eda/replay.py`, `reproduce_fork_histories.py` | Fresh support replay retained; historical saved-action replay and strict preserved-policy reconstruction added; all representation-bound caches rejected |
| 33: terminal-tail audit | `audit_terminal_tails.py` | Recorded-key factual verification retained for support roots. The exhaustive 8,015-terminal census and corrected actionable training population are not claimed as covered by 256 sampled FIT roots |
| 34: observability/bottleneck | `probe_observability.py`, `probe_prebottleneck.py`, `probe_decoder_ceiling.py` | CLS/export, pixels, temporal reset, visible-state/hidden-timing/full-simulator oracles implemented. Predictor internal taps, equal-capacity decoder ceilings and memorisation controls still need explicit adapters |
| 40: action matching/interaction | `probe_action_matching.py`, `probe_action_interaction.py` | Exact-pixel equivalence, retrieval/Hungarian, ranks/margins/per-action matching, energy-weighted effect error, root/action/interaction decomposition, properly FIT-residualised transfer and frozen-decoder residual restoration implemented. Predictor internal taps remain annex work |
| 41: terminal-supervision evaluator | `evaluate_death_transfer.py` | Central 961/197 transfer, successor-only frozen decoder, action floor, within-root AUC/recovery and seed-clustered paired uncertainty implemented |
| 42: broad-regime evaluator | `run_seed2_matrix.sh` | Full 5,402 physical panel with original seed splits and fatal-action strata implemented; the 32-frame-filtered 3,651 FIT schedule is training provenance, not a new test split |
| 45: death-transfer family | `evaluate_death_transfer.py`, `evaluate_death_forks.py` | 961 and 104 remain separate evaluation panels. Do not resurrect the mixed 1,069-root random split or root-only bootstrap |
| 46: Direct-A/Direct-M anchors | `run_v2_pipeline.sh`, `run_v2_evaluation.sh` | Frozen existing 20k worlds, native context and independently refitted M03 probes; historical training directories retain their actual filenames |

## Scoring and interpretation contract

- Same raw root, candidate action, simulator key and true successor across all
  four arms. Direct consumes up to 64 frames; Raw/TC consume up to four. This
  compares trained systems, without claiming isolated architecture attribution.
- Fit standardisation and linear/fixed-MLP probes on FIT only. Historical TUNE
  is reserved; no new DEV-driven hyperparameter selection. Never load old probe
  weights, latent caches, continuation calibrators or historical scalar scores.
- Keep the original action-conditioned diagnostic and add successor-only,
  delta, root-centred effect and FIT-action-marginal-residualised transfer. A
  residualised feature is read only by a probe fitted in that same space.
- Report both global discrimination and within-root opportunity AUC. Recoveries
  with observed AUC at chance are unsupported, not evidence of perfect transfer.
  Compare generated scores directly against the action-only floor with paired
  seed uncertainty. Retain per-root scores and fatal-action strata.
- Include the historical compound damage-or-death target alongside M03's
  separate death and health-decrease targets. Coverage requires distinct
  episode/seed support for both classes on FIT and evaluation; fork rows do not
  multiply the number of independent positive examples. A one-seed sample has
  no confidence interval.
- Recompute visible, hidden-timing and full-simulator pre-state oracle features
  from replay. The visible descriptor retains the historical nearest-three-mob
  design and is not a proof of complete pixel observability. Oracle information
  never enters the world model. Residual-restored successor coordinates likewise
  diagnose error location; they are not a deployable correction.
- Action matching classes come from exact successor pixels. Pixel equivalence
  is not asserted to imply equality of hidden simulator state. Geometry remains
  representation-specific and descriptive; do not compare raw latent MSEs across
  different coordinate systems as an architecture ranking.
- Mode score distances are advisory. They are not calibrated probabilities,
  Brier scores, a learned stochastic generator evaluation or an S35 replication.
- `measured` means measurable, not passed. Adequate coverage is necessary but
  not sufficient. M4 authorization stays false. No numerical promotion margin
  is invented from these smoke readings.

## Execution and preservation

The corrected gate hashes its diagnostics/history source, raw historical inputs,
checkpoint bytes and settings in `run.json`. Per-seed replay and feature shards
are atomically published and hash-checked; panel probe stages are independently
resumable. All four panels use the same per-seed encoding caches, including their
overlap. A changed source or input requires a new output directory.

The existing stopped runs and pre-existing working-tree changes are preserved.
Gate schemas and new run names have no `_v2` suffix. Genuine historical source
paths such as `craftax_support_v2`, `broad_forks_v2` and `v2_direct_attention`
retain their names because renaming provenance sources would be misleading.

## Validation performed

Incremental runner checks: 52 CPU tests pass, including per-arm dependency
invalidation, persistent fitted predictions, corruption rejection, oracle
numerical parity and CLI ordering. The GPU oracle check uses synthetic roots
with the failing hazard panel dimensions (3820 FIT, 1069 evaluation, 11907 pixel
features, 17 actions), two optimizer steps and both probe families. Peak tensor
allocation is 588 MiB; tiny dense/streamed GPU parity differs by at most
2.4e-7. This validates execution, not a semantic result. Evidence and ordered
smoke outputs live in `artifacts/lewm_gates_20260906/m03_cache_smoke/`.

Earlier baseline smoke checks:

- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest
  d4mj/tests/test_m03.py -q`:
  **21 passed**. Tests cover short-history alignment, grouped coverage,
  successor mismatch rejection, seed leakage, equivalent successors, action-only
  shortcuts, frozen restoration, and changed/corrupted cache rejection.
- Bounded GPU smoke outside the sandbox on the RTX 3060 Laptop GPU:
  **completed**, with Raw, TC, Direct-A and Direct-M, fresh support replay,
  all four historical panels, EDA and observability reports. The smoke selects
  one FIT and one evaluation root per historical panel, plus a reserved TUNE
  root in the replay sample; its purpose is structural validation only.
- Both primary Direct parity checks and all **10 historical parity records**
  passed with maximum absolute error **0.0**, including an actual one-frame
  episode-start history.
- Repeating the identical GPU smoke command reused the sealed report. Final
  report: `artifacts/lewm_gates_20260906/m03_smoke_complete_20260908/report.json`.
  Its SHA256 is
  `6acd9933e71199abe39d81ca2a9344a16fd9af68c6cf48774345d85bf3b8fb37`.
- `git diff --check` passed. The pre-existing root-selection fix and all old
  stopped experiment directories remain intact.

No full scientific comparison was run. The smoke explicitly records
`structural_smoke_not_a_result`, `suite_complete: false` and
`m4_authorized: false`. Population-wide historical replay and full-scale probe
memory/runtime remain unvalidated until the full evaluation is executed.
