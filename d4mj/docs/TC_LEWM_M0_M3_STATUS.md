# M0–M3 implementation and validation

Implemented in the working tree against `162efd1` on `craftax-clean-baseline`, 2026-09-05; integrated into shared infrastructure on 2026-09-06. See the [integration record](INTEGRATION_REFACTOR.md). This is the actual TC-LeWM–Mamba architecture and persistent state API. Small test fixtures instantiate these same classes with reduced dimensions; there is no disposable alternate world model.

## Active architecture

Native uint8 `63×63` RGB → source ImageNet preprocessing → HF ViT-Tiny (patch7, width192, depth12, heads3, CLS) → retained `192→2048→192` BN/GELU projector → unbounded `[B,T,1,192]` latent. Each completed `(z_t,a_t)` pair concatenates a learned 64-D action embedding and projects to width256. Six pre-RMS residual Mamba-2 blocks (`expand=1, state=64, head=64, conv=4`) and final RMSNorm feed the predictor's `256→2048→192` BN/GELU projector. Encoder plus world contain 6,175,872 + 2,503,496 parameters, including the untrained policy readout.

`PredictiveState` contains current latent `z_t`, constant-size conv/FP32 SSM carry for each layer, previous completed-pair output `u_(t-1)`, and an explicit step count. Its memory has consumed only pairs through `(z_(t-1),a_(t-1))`; `z_t` is unconsumed. `advance` consumes one new pair and keeps that resulting carry. `observe` performs the identical update and substitutes the observed successor latent. `prefill` follows those same semantics. `fork`, `repeat_state` and explicit `detach_state` own their tensor storage. No prefix is stored or replayed by a step. Both training and inference call the functional source chunk scan; streaming supplies T=1. Gradients reach incoming conv and SSM states unless explicitly detached.

Features are `GELU(LayerNorm(Linear(concat(z_t,u_(t-1)))))`, `[B,1,1,256]`. This readout is initialized with the bundle but frozen and untrained through M3. It never feeds the predictor. Its presence establishes the API, not policy competence. Runtime observation/streaming requires fixed BN statistics; full joint teacher training retains source flattened-batch BN behavior.

Joint training uses actual B128, four frames/three outgoing actions, both-sided next-latent MSE plus `0.09 × SIGReg`, 1,024 directions and 17 knots. The raw and TC arms change only temporal centering across the four frames. Initialization, sampler and projection RNG seeds are paired. The optimizer and 10,000-update scheduler are explicit in the two [recipes](../recipes/). Research execution stops at update2,000 pending G1; resumption does not shorten or restart the declared scheduler. No EMA, target stop-gradient, reconstruction loss or head fitting is active.

## Implemented file and API map

| File | Implemented surface |
|---|---|
| [config.py](../config.py), [lewm_config.py](../lewm_config.py) | Shared `canonical_json`, `recipe_dict`, `recipe_digest`, `config_from_dict`, `load_recipe`; separate frozen LeWM settings, `validate_recipe` and nested parsing. Unknown fields and microbatch substitutes fail. The original `Config` defaults remain intact. |
| [lewm.py](../lewm.py) | `LeWMProjector`; `LeWMEncoder` construction, `train/freeze/projected_and_cls/forward`; `SIGReg`; `_MambaBlock`; `TeacherOutput`, `JointLoss`, `joint_loss`; `LeWMWorld` shape/action/state validation, `start/readout/features/scan_pairs/teacher/advance/observe_latent` and fixed-BN streaming guard. |
| [mamba_recurrence.py](../mamba_recurrence.py) | `MambaCarry`; clone/detach/repeat helpers; `FunctionalMamba2` source construction, initial state, validation, functional `scan`, FP32 `_reference_ssm`, `step`, `step_reference`. No new CUDA kernel or mutating source inference cache. |
| [state.py](../state.py), [world_api.py](../world_api.py) | Additive `PredictiveState` and shared legacy `repeat_memory`; `WorldAPI`, `ModelBundle.create/from_models/eval/require_control`; concrete `LegacyWorldAdapter` and `LeWMWorldAdapter` implement encoding, start/prefill, observe/advance, features, fork/detach/repeat, tensor enumeration and world-state extraction. v2 `load_bundle` is inference-only; v1 loading retains its existing phase-specific contracts. |
| [data.py](../data.py) | Both `Batch` and `JointBatch`; `EpisodeCorpus.window_weights`; joint validation/audit/corpus loading and `JointSampler` with exact sampler resume. Existing role routing and joint TRAIN-only outgoing-action sampling remain distinct. |
| [cache.py](../cache.py) | Shared `encoder_digest`, `cache_latents`, `cache_latents_to_store`, `load_latent_cache` and one verified store writer. MAE carries temporal encoder memory and retains its old digest/schema; LeWM exports fixed-BN float32 latents with parent-checkpoint identity. Old `train.cache_latents*` names are imports of these functions. |
| [train.py](../train.py) | Existing phase trainers plus `train_joint`, `set_phase_mode`, `joint_optimizer`, fixed joint `learning_rate`, `autocast_context`, `freeze_encoder`. Shared `optimizer` grouping and `optimizer_step`; explicit legacy versus joint decay policies and schedules. Research screen and resume guards remain active. |
| [checkpoint.py](../checkpoint.py) | Additive v2 `save_lewm_bundle`, `publish_lewm_latest`, `read_lewm_bundle`, `restore_lewm_bundle`. Full model/BN/optimizer/sampler/projection/global RNG state, module modes and gradient ownership; recipe/source/data/schedule/capability rejection. Immutable numbered snapshots, movable latest link and SHA256 index. v1 functions remain intact. |
| [sources.py](../sources.py) | `lewm_source_manifest`, `verify_lewm_sources`, `tensor_state_digest`. Canonical pins/licenses, installed Mamba byte parity, HF runtime code, dependency versions, math settings and shared runtime closure. Legacy source/cache contract is unchanged. |
| [gates.py](../gates.py), [lewm_diagnostics.py](../lewm_diagnostics.py) | One `Gate` dependency runner, `preflight`, `ComponentGateError`, `contract_digest`, `require_joint_gates`; six legacy gates remain available. LeWM diagnostics supply source/objective/recurrence/normalization/resource probes and sealed numerical tolerances. Failures name their component, with `architecture_verdict=not_evaluated`. |
| [execution.py](../execution.py), [imagination.py](../imagination.py), [diagnostics.py](../diagnostics.py) | Existing execution and imagination use the adapters. Both reject LeWM control before touching the environment or policy. Shared `rollout_predictions` drives legacy multistep diagnostics and supports LeWM mechanical rollouts. Other legacy diagnostics and training targets retain their explicit family scope. |
| [__main__.py](../__main__.py), [experiments.py](../experiments.py) | Unified `python -m d4mj` CLI. No arguments or `gates` retains the four-arm lattice; recipe preflight dispatches by family; joint/resume/export uses shared modules. Later LeWM stages remain blocked. |

The original M0–M8 map included future adapters and methods. `LegacyWorldAdapter`, shared actor/execution callers, bridge/heads/actor/renderer settings, semantic screening and control aggregation remain M4+ work. Legacy callers keep their existing interfaces; the new family never masquerades as a legacy `WorldState` or v1 checkpoint.

## Audit findings and resolutions

1. **CUDA recurrence failure reproduced exactly.** The old universal `atol=1e-5, rtol=1e-4` failed on one of 512 SSM entries at `1.592934e-5`. Source-versus-wrapper, differentiable reference and gradient checks support a numerical explanation on this GPU. The source Triton dot operations have a different precision/accumulation path from the direct FP32 recurrence; calling the whole effect only temporal error accumulation would be inaccurate.
2. **Screen checkpoint overwrite confirmed and fixed.** Checkpoints are `step-NNNNNN.pt`, atomically published without overwrite. `latest.pt` is a relative symlink. `checkpoints.json` maps immutable filenames to SHA256. The screen is saved even when it is off cadence. Resuming older history requires a fresh destination; export records the immutable resolved parent path and digest.
3. **Missing helper pin confirmed and fixed.** `SOURCES.lock` now records `galilai-group__stable-pretraining` at `9aa93f8b6153eebb73f57d4853ccf8a13d848310`, MIT. Source closure records its license, constructor and pixel-statistics files, alongside installed HF code. TC-29 removes two unnecessary earlier proposals: mean/std0.5 and ViT epsilon `1e-6`; source ImageNet normalization and `1e-12` are active.
4. **Missing data/checkpoint tests confirmed and filled.** Added sampler/cache lineage and identity tests, v2 round-trip and rejection tests, full pause/resume tests and CLI artifact/phase guards. Existing legacy tests are retained.

The original ledger had 17 adaptation-tagged entries grouped into 12 substantive departures. Groups1–8 enter M0–M3; groups9–12 remain deferred. TC-29–32 additionally document source resolutions and engineering safeguards. These counts are not independent failure probabilities. Tests settle bounded implementation contracts; they do not establish that the 192-D state retains Craftax mechanics or that imagined control will work.

## Numerical evidence and tolerances

[Single-mixer calibration](evidence/m0_m3/recurrence_calibration.json) uses widths32/256, states8/64, seeds123/500, lengths2/17/65/257, FP32/BF16, nonzero incoming conv and SSM carry, and derivatives with respect to inputs, incoming states and all parameters. [Full-world calibration](evidence/m0_m3/world_calibration.json) additionally measures the six-layer final state and chunks at T17/65/257. The gate uses different numerical inputs (seed904+length) for the mixer checks.

FP32 single-mixer scan/step maxima were `1.75e-4` output and `4.55e-5` SSM. Triton/reference maxima were `5.19e-4` output and `3.51e-4` SSM; the actual upstream full forward agreed with our wrapper to `5.96e-7`. The reference's own scan/step maxima were `1.19e-6` output and `1.49e-7` SSM. Full-stack scan/step conv and SSM maxima were `6.17e-5` and `6.35e-5`: later conv carries inherit earlier layers' SSD rounding. BF16 single-mixer output maxima were `7.8125e-3`, with SSM differences below `7.46e-4` across these comparisons. Gradient maxima and individual measurements are in the raw records.

The sealed `rtx3060_mamba_f577286d_v2` profile keeps reference FP32 output/SSM at `1e-5`; Triton scan/step FP32 output at `5e-4`, SSM at `1e-4`; Triton/reference FP32 at `1e-3`; BF16 output at `1e-2`, SSM at `1e-3`. SSM checks use **rtol=0**. Single-layer conv remains `1e-5`; full-stack conv gets `1e-4`. Gradient absolute tolerances are `1e-5` reference, `2e-5` Triton scan/step FP32, `5e-5` Triton/reference FP32, and `2e-3` BF16, with explicit relative budgets in code. No test widens a tolerance dynamically. A material carry corruption is still rejected. These measured finite-input bounds are not guarantees for arbitrary weights, hardware or long learned trajectories.

## Hardware and data scope

GPU execution requires access outside this sandbox; CPU skips do not certify CUDA. The full architecture's BF16 preflight used the RTX3060 Laptop GPU (6,076,104,704 device bytes), B128/F4/J1024, and three real optimizer updates. One recorded pass used 952,169,984 peak allocated bytes (~0.887GiB), 1,193,279,488 reserved bytes (~1.111GiB), and about 0.302 seconds per warm TC step (0.348 seconds for raw). This measures local resource feasibility, not convergence or total research runtime.

The resource fixture is the first intact, hash-verified shard of `artifacts/craftax_support_v2`: 24 whole episodes, with 18 TRAIN / 3 DEV / 3 FINAL. Only TRAIN windows enter optimization. The parent manifest, shard, collector and expert checkpoint hashes are retained. Collector training access remains **unknown**, and the recorded collector episode limit is2500. This is a technical fixture, not a selected research corpus or matched D3 protocol. No full budget, G1 scientific screening or control evaluation was run.

The pre-integration CPU regression suite reported **187 passed, 5 skipped**. Both **raw and TC full-architecture GPU preflights passed** in that initial snapshot. Fresh post-integration results and source manifests are in [the integration record](INTEGRATION_REFACTOR.md) and [its evidence](evidence/integration/). [Machine-readable summary](evidence/m0_m3/summary.json), validation logs, final gate/source/data records and the full-architecture GPU pause/resume result are retained under [evidence/m0_m3](evidence/m0_m3/). An uninterrupted four-update GPU run and a two-plus-two resumed run produced identical model parameters, BN buffers and metrics; the update2 immutable file retained its hash. Verification checkpoints remain local test artifacts, not research results.

## Commands and hard stops

From the repository root, using the measured environment in [requirements-lewm-rtx3060.lock.txt](../../requirements-lewm-rtx3060.lock.txt):

```bash
.venv/bin/python -m d4mj preflight \
  --recipe d4mj/recipes/lewm_mamba_tc.json \
  --dataset <verified-raw-store-directory-or-manifest.json> --out <fresh-run-directory>
.venv/bin/python -m d4mj joint --run <run-directory> --stop-at 2000
.venv/bin/python -m d4mj export --run <run-directory> \
  --checkpoint <run-directory>/joint/step-002000.pt --out <fresh-cache-directory> --diagnostic
```

Create a separate raw run with the raw recipe and the same data/seed. Source or backend-setting edits invalidate preflight; data changes invalidate the dataset identity. Ordinary joint research calls default to the screen checkpoint. G1 screening must be implemented and passed before enabling continuation beyond it. `--verification` labels technical tests and cannot be reported as a research result. Partial exports require `--diagnostic` and remain explicitly incomplete.

M4/H16/actor/renderer remain blocked. Checkpoint capabilities record `trained_recursive_depth=0`, `validated_recursive_depth=0`, `readout_trained=false` and `m4_authorized=false`. Persistent software recurrence and a passed numerical gate do not confer a learned recursive horizon.

## Source recovery

The canonical base LeWM, stable-pretraining helper and Mamba trees are locally available under `third_party/sources/`. Recreate each missing clone with its URL in `third_party/SOURCES.lock`, then `git checkout --detach <recorded-commit>`. Keep each checkout's LICENSE; source preflight checks HEAD against the lock. Install the recorded dependency versions and run preflight to verify all installed Mamba Triton Python files against the pin and record imported HF code. The TC v3 paper is pinned in `PAPERS.lock`; no released canonical TC code was available at the source review. The implemented TC centering therefore follows the paper over the canonical base objective.
