# Integrating M0–M3 into the existing system

Implemented 2026-09-06 against `162efd1`. The initial implementation created too much parallel infrastructure. Joint training now uses the existing package's data, training, checkpoint, gate and execution infrastructure. The existing Flow/Direct × Attention/Mamba baselines remain active callers of that infrastructure.

## Ownership and boundaries

| Responsibility | Final owner and integration |
|---|---|
| Episodes, validation and sampling | `data.py` owns `Batch`, `JointBatch`, corpus lineage and both samplers. `EpisodeCorpus.window_weights` supplies valid-window counts. Joint sampling retains outgoing actions and TRAIN-only eligibility; existing phase sampling retains led-to actions, burn-in and role routing. `joint_data.py` was removed. |
| Latent caches | New shared `cache.py` owns identities, family-specific encoding adapters, loading and one verified archive writer. Moving caches out of `train.py` gives both trainers/exporters the same storage and recovery path. `train.cache_latents`, `train.cache_latents_to_store` and `train._cache_digest` remain compatibility imports of these functions. MAE exports remain resumable; joint exports retain the fresh-directory and parent-checkpoint requirements. |
| Training | `train.py` owns the original phase trainers and `train_joint`. `optimizer` and `optimizer_step` share AdamW grouping/backward/clipping/update mechanics. `train_lewm.py` was removed. Different objectives, gradient ownership and schedules remain explicit phase code. |
| Recipe handling | `config.py` owns canonical serialization, hashing, JSON loading and family dispatch. The original flat `Config` defaults remain intact. Separate nested `LeWMConfig` settings prevent invalid combinations of legacy and joint fields. |
| Gates | `gates.py` owns `Gate`, dependency handling, component reports, `preflight`, `ComponentGateError` and `require_joint_gates`. It runs the original six gates and the new family's probes. A failed prerequisite blocks dependent components; the architecture verdict remains unevaluated. |
| Runtime state | `world_api.py` contains concrete legacy and LeWM adapters, constructed through `ModelBundle.create` or `from_models`. No adapter duplicates transition mathematics. `state.py` owns both state types and the spatially correct legacy memory repeat operation. |
| Existing callers | `execution.run_episode` and `imagination.imagine` advance through the adapter. `diagnostics.rollout_predictions` provides a shared observed-prefix/generated-successor path, used by the existing `multistep_error` and `latent_stats`. Existing raw-world call signatures remain accepted. |
| Commands | `python -m d4mj` exposes recipe preflight, joint/resume and export. No arguments or `gates` retains the legacy four-arm lattice. `experiments.py` is orchestration over the shared modules, not a separate training implementation. |
| Checkpoints | Existing `checkpoint.py` owns both formats. Legacy v1 phase checkpoints and named v2 joint bundles remain distinct serialization contracts. Immutable numbered joint snapshots and parent hashes are preserved. |

`lewm.py`, `lewm_config.py`, `mamba_recurrence.py` and `lewm_diagnostics.py` remain separate for concrete reasons: the framewise encoder/objective differs from MAE; nested recipes have different validity conditions; functional Mamba carry has different gradient semantics from the legacy source inference cache; and source/objective/numerical probes are model-specific. The existing `representation.py`, `transition.py`, `time_mixer.py`, head/target code and actor-critic mathematics continue to serve the baseline family. Their specialized training losses and diagnostic targets are not forced through a generic signature.

## Preserved contracts and deliberate differences

- Legacy memory includes the current committed frame and carries separate temporal encoder memory. LeWM memory includes only completed `(z, outgoing action)` pairs; its current latent remains unconsumed. After a C-frame prefill, their relative step counts are C and C−1 respectively.
- Legacy Flow retains explicit world RNG, corruption and stochastic commits. Direct draws no world noise. LeWM remains deterministic. Policy and world generators stay independent.
- Legacy `advance` retains S55's detached incoming memory. LeWM retains differentiable conv/SSM carry and accepted latents. Fork/detach/repeat preserve storage ownership; legacy repeats unflatten batch and spatial slots before branching.
- MAE cache identity, bounded temporal encoding and cache schema are preserved. LeWM cache identity still includes projector parameters, BN buffers, preprocessing and imported ViT implementation; it stores unbounded float32 `[T,1,192]` latents. The shared writer validates registered shard bytes before resuming and refuses to overwrite orphan shards.
- Both optimizers honor Mamba's `_no_weight_decay`. Legacy phases continue to decay ordinary vectors/biases; joint training explicitly excludes them. Joint warmup/cosine and legacy warmup schedules remain different. No research hyperparameter, source tolerance, statistical batch or adaptation group changed in this refactor (TC-33).
- Shared execution does not enable LeWM control: its adapter rejects execution/imagination before touching the environment or heads. G1 was closed at the integration snapshot; it was subsequently implemented and passed by the [first research pair](TC_LEWM_PAIRED_RESULTS.md). M4, H16, actor and renderer gates remain closed. Mechanical rollout support is not evidence of a learned horizon.

## Verification

The retained [evidence](evidence/integration/) includes:

- **Legacy git-reference parity:** all four Flow/Direct × Attention/Mamba arms on the RTX3060. Existing diagnostic outputs, imagined trajectories, RNG advancement, chunked latent caches/cache identities and optimizer updates match exactly. A deterministic simulated environment produces identical execution actions and episode results. The reference caller functions are loaded directly from git commit `162efd178ef72a8eca05730fe3996358162054ad`; this does not measure live Craftax policy quality.
- **Original joint-trainer parity:** four CPU verification updates using the original optimizer, sampler and update functions, with only moved imports redirected. Encoder/world weights, BN buffers, metrics and sampled windows are exact. The record identifies the compared original source bytes. This small test checks the refactor; it is not a replacement architecture or resource gate.
- **Full architecture GPU gates:** both raw and TC use B128/F4/J1024 BF16, all six 256-wide recurrent layers and the native 63-pixel encoder. All eight component gates pass on the RTX3060 Laptop GPU. Peak allocated/reserved memory remains 952,169,984 / 1,193,279,488 bytes. Only TRAIN windows from the first hash-verified support shard enter the resource probe; it is a technical fixture, not a selected research corpus.
- **Full architecture GPU resume:** four updates versus two plus two resumed updates produce bit-exact weights/BN and exact metrics under the unchanged 10,000-update schedule. The earlier immutable checkpoint retains its SHA256 and remains indexed.
- **Regression tests:** the full CPU suite and targeted CUDA tests cover sampler/cache lineage, v1/v2 rejection and resume, state timing, observation memory, spatial branch order, recurrence gradients, source parity, existing execution and CLI family dispatch. **199 passed, 7 skipped** in the full CPU suite; **21 passed, 4 skipped** in the targeted CUDA invocation. The latter skips are old tests whose fixture explicitly selects CPU. Commands and logs are retained in the evidence README.

Temporary before-refactor legacy traces were unavailable after an environment refresh, so the legacy parity check uses git-recovered reference functions instead. Original joint trainer/data copies survived long enough for the recorded comparison; their source hashes are retained, while temporary training artifacts are not research checkpoints.

## Provenance and remaining scope

The joint source closure now hashes the shared runtime files. Old M0–M3 reports remain historical under `evidence/m0_m3`; fresh reports are under `evidence/integration`. A pre-refactor joint gate/checkpoint cannot silently resume against changed source files. Re-run preflight for the final recipe/data/runtime before a new run. Legacy v1 source and encoder-cache contracts remain unchanged.

M4 will add the new family's recursive training and outcome targets to the existing trainers and use this shared runtime. The current refactor does not implement those targets, train the readout or establish Craftax retention/control performance. A negative component gate still stops at that component.
