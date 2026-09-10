# G1 paired screening protocol

Sealed before inspecting trained raw/TC checkpoints, 2026-09-06. The existing M0–M3 implementation is committed as `0d49d7e`. This implements the update-2,000 gate in the evaluation plan; M4/H16/actor/renderer remain outside the authorized boundary.

## Data and scope

Use the entire existing `craftax_support_v2` corpus for the initial raw/TC pair, preserving its whole-episode TRAIN/DEV/FINAL assignments and uniform eligibility. All 420 shards (3,179,062 transitions) passed byte verification against the parent manifest. This is the existing exploratory expert-policy corpus, including epsilon 0.1/0.25/0.5/1.0; it is not pure expert footage or a new matched legacy run. It has no BC-eligible rows, which is acceptable for the joint objective and does not authorize BC training. Unknown upstream collector training access stays unknown. No FINAL features are encoded or inspected by G1.

The model recipes remain B128/F4/J1024, 10,000 total updates with the 500-update warmup. Both arms start from immutable update-zero checkpoints with equal encoder/world state digests and equal sampler/projection streams. The screen verifies actual sampled-window histories, source/data identity and schedule equality, not just seeds. At the first optimizer update it requires equal prediction loss (absolute1e-6) and different regularization loss (absolute gap greater than1e-6).

## Diagnostics and explicit local choices

The separate `joint_screen.json` recipe freezes evaluation-only settings. They are project choices, not paper claims or changes to the model recipe:

- Select 256 TRAIN and 128 DEV episodes by seeded stable ID hash, with four independently selected valid four-frame windows per episode. Keep equal counts per selected episode. Encode in FP32/eval, 32 windows per batch. No probe or scaler sees DEV during fitting.
- Record raw latent, temporal residual and per-window persistent-component covariance eigenvalues, effective rank, scale and means. Record the four-frame temporal Fourier power. An absolute total coordinate variance below `1e-10` is a numerical collapse stop; it is not a general semantic sufficiency criterion.
- Compare held-out prediction with persistence, and with actions permuted across rows. Report within-family normalized error, episode-cluster bootstrap differences and the same diagnostics at initialization. Permuted logged actions are an association control, not a simulator action-fork or a causal action-effect gate.
- Compare CLS and exported projected features using the archive's available **short-future outcome proxies**: positive reward, negative reward, aggregate achievement event and true termination, conditioned on the outgoing action. These labels do not cover inventory, local tiles, health state or all22 achievements. G2's critical semantic suite remains unpassed.
- Freeze two probe families: linear and one hidden GELU layer of width128, both with feature-plus-action-one-hot inputs standardized using TRAIN only. AdamW lr0.001, decay0.01, 200 minibatch updates of256 examples, fixed initialization/minibatch streams (screen seed+100/+101 for linear, +200/+201 for MLP), gradient clip1, TRAIN class-balancing weights. Features use a TRAIN standard-deviation floor of1e-6. No DEV checkpoint selection or capacity search.
- A label is reportable only with at least20 positives and20 negatives in both fit and DEV. Bootstrap DEV episodes (1,000 paired draws); require at least95% valid resamples for an interval; the margin is the already proposed `.03` macro-AUC. Both probes' upper95% bounds below −.03 are a stop at the projection/retention component. Sparse coverage or a weak early probe is explicitly inconclusive, not information destruction or a G2 pass.
- Recheck frozen normalization and the existing recurrence/objective gates at each trained checkpoint. Keep sealed numerical tolerances. A failure stops its component and saves the raw error; do not widen tolerances during the run.

Continue the original joint budget only if identities/mechanics pass, no finite/collapse/retention stop fires, and normalized DEV prediction improves over the saved initialization. This learning-progress criterion licenses more joint training only. If progress is unresolved, save a review-required report and stop before extra updates. If either arm fails, stop the pair for diagnosis; do not drop the weaker arm or change its recipe.

## Code/function plan

- `lewm_config.py`: standalone frozen `ScreenConfig` and strict evaluation-field validation. `config.py` dispatches its schema through the existing loader. Store all values in `recipes/joint_screen.json`.
- `data.py`: `screen_windows` selects ID-stable TRAIN/DEV windows with outgoing-action/outcome alignment and returns indices for episode-cluster uncertainty. It never constructs FINAL windows.
- `lewm_diagnostics.py`: shared feature extraction, covariance/frequency summaries, deterministic fixed probes, tied-score AUC, paired episode bootstrap, and `screen_joint_pair`. Save numeric rows sufficient to recompute results. Label unavailable semantic/control claims explicitly.
- `gates.py`: `require_joint_screen` validates a passing paired report against the exact source/data/recipes and immutable parent checkpoint before further joint updates.
- `train.py`: factor joint construction into one helper; `initialize_joint` saves a resumable immutable update-zero checkpoint. `train_joint` accepts a verified screen report for continuation, preserves it in later checkpoints, and retains the original schedule and all M4 capability blocks.
- `experiments.py`: `paired-run` seals recipes/data/initial checkpoints, runs both preflights, trains both arms to2,000, evaluates G1, and continues only the accepted original budget. Existing `joint` accepts an explicit screen report for equivalent manual resume. Failures produce component-local records. Never reinterpret G1 as M4 authorization.
- Tests: split/action/label sentinel alignment, no FINAL selection, deterministic probes/tied AUC, clustered bootstrap, insufficient coverage, objective/pair mismatch, report tampering/parent mismatch, bounded continuation, and tiny end-to-end paired orchestration using the same classes. Full B128 GPU gates precede research launch.
