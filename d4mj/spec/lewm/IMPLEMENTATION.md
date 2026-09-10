# Implementation plan and code-change map

Status: M0–M3 runtime is implemented. [Actual files, validation and remaining boundaries](STATUS.md) are authoritative for implementation status; this document retains the end-state M0–M8 map. M4–M8 and their associated tests/CLI commands remain future work. History baseline: `162efd1`.

## 1. Implementation boundaries

Keep `Config`, the legacy encoder/decoder, `World`, and their checkpoint format independently usable. Add a new configuration and world implementation, with a small common execution interface. Reuse verified loss and metric functions through explicit settings/target adapters. Do not produce a new-family configuration by pretending that `n_latents=1, packing=1, window=1` is a valid legacy model: its assertions, layout, cache digest, burn-in, and state semantics encode another architecture.

The required outcome is a reproducible path from raw episodes to a jointly trained encoder/world, cache, recursively trained agent, real evaluation, and a simulator-free dream session. It includes a raw-SIGReg control. It does not include implementing every deferred representation family, rebuilding historical experiment scripts, or running expensive training as part of this documentation task.

## 2. Ordered milestones and acceptance artifacts

| Milestone | Work | Required evidence before next stage |
|---|---|---|
| M0 — source and experiment contract | Resolve source/dependency pins, dataset/split identity, benchmark scope, raw/TC recipes; inventory legacy compatibility | `resolved_recipe.json`, `source_manifest.json`, `dataset_audit.json`, `baseline_manifest.json`; every required field resolved |
| M1 — representation and objective | Implement framewise encoder, both projectors, source-scale SIGReg, raw/TC loss, raw sequence sampler | Objective/source parity, two-sided gradients, exact action alignment, eval BN/cache parity; no training result claimed |
| M2 — recurrent world | Implement action-pair state, teacher scan, prefill, observe, advance, safe forks and differentiable carry | Full scan/step/prefill/gradient agreement, no action leakage, reset and branch invariants on CPU reference and target CUDA |
| M3 — joint trainer and export | One optimizer over encoder/world, resume, actual B128 resource probe, frozen encoder cache | Split-run resume agrees with uninterrupted run; peak memory/throughput measured; cache cannot mix families or buffers |
| M4 — bridge and actor shell | Continue joint world, H2/H16 loss and head paths, truthful terminals, frozen-world actor training | Generated suffix really differs, reaches stated depth, has recursive gradients; role routing and outcome offsets pass; world buffers immutable in Phase 3 |
| M5 — diagnostics and paired screen | Shared raw traces, family-specific encoders, semantic/action/outcome diagnostics, matched raw/TC screen | DEV report classifies failure or licenses the next finite training stage; no automatic continuation after failure |
| M6 — visible imagination | Fit independent renderer, headless dream session, optional local viewer, replay | Real-latent display ceiling, generated rollout diagnostics, zero simulator calls after initialization, measured latency |
| M7 — controlled evaluation | Replicate full raw/TC pair, actor versus own BC, legacy references; optionally implement verified D3 adapter | Per-training-seed metrics and uncertainty, complete access/compute accounting; claim-specific gates passed |
| M8 — longer playable worlds, conditional | A separately sealed longer recursive curriculum and validation, only after M7 or a compelling display-specific result | Validated64/256-step play with no refresh; no implicit 10,000-step claim |

M1 and the renderer's standalone shape implementation can be developed independently, but decoder training waits for a frozen export. M4 integration does not license Phase 3 performance training before M5's relevant gates. First screen one raw/TC seed; replicate only according to the [stop rules](EVALUATION.md).

## 3. Runtime files and public functions

This table describes the end state; the [integration decision](INTEGRATION.md) consolidates infrastructure in the existing modules. M4 and later methods listed here remain deferred. The [implemented M0–M3 function map](STATUS.md#implemented-file-and-api-map) resolves renamed/split helpers and explicitly identifies deferred methods. It is the approved scope update for the current implementation.

| Path | Types/functions and responsibilities |
|---|---|
| `d4mj/lewm_config.py` | Frozen `LeWMConfig` with `EncoderSettings`, `DynamicsSettings`, `JointSettings`, `BridgeSettings`, `HeadSettings`, `ActorSettings`, `RendererSettings`, `DataSettings`, `EvaluationSettings`; family validation and nested parsing. Shared loading, serialization and recipe digests live in `config.py`. Validate shape divisibility, full statistical batch, losses, phase lengths, normalization modes, source/data fields, and trained/validated horizon metadata. Reject unknown fields. Serialize all inherited library defaults. |
| `d4mj/lewm.py` | `LeWMEncoder.forward(frames) -> z`, `projected_and_cls(frames)` for offline probes; `LeWMProjector`; `SIGReg.forward(values, generator)`; `joint_loss(encoder, world, joint_batch, recipe, generators)`; `LeWMWorld.teacher(z, outgoing_actions, memory=None)`, `step_pair(z, action, memory)`, `readout(z, previous_output)`. Keep prediction separate from policy readout; initialize the final API readout but do not train or use it in the Phase 1 loss. Return per-term losses and diagnostics without retaining entire graphs in logs. |
| `d4mj/mamba_recurrence.py` | `MambaCarry`, `FunctionalMamba2`, `scan(inputs, carry=None)`, `step(input, carry)`, `step_reference(...)`, `detach_carry`, `clone_carry`, `repeat_carry`. Preserve source parameterization and named state; wrap source scan kernels with functional convolution history. Expose separate gradient-safe and no-grad paths. Check both initial-state and final-state derivatives; no custom kernel. |
| `d4mj/world_api.py` | `WorldAPI` protocol, `LegacyWorldAdapter`, `LeWMWorldAdapter`, `ModelBundle`, `load_bundle`. Common methods: `encode(frames)`, `start(z0)`, `prefill(z_context, actions, ...)`, `observe(previous_state, action, next_frame)`, `advance(state, action, generator)`, `features(state)`, `fork(state)`, `repeat_state(state,n)`, `state_tensors(state)`. Each adapter owns its configuration, encoder and state semantics. Generic callers must not inspect `layout`, `n_spatial`, or slot-flattened caches. |
| `d4mj/data.py`, `d4mj/cache.py` | `JointBatch`, `BridgeBatch`, `sample_joint_batch`, `sample_bridge_batch`, `sample_bridge_terminals`, `to_head_batch`, `validate_batch`; shared `cache_latents_to_store`, `load_latent_cache`, `encoder_digest` in `cache.py`. Joint batches hold four raw frames and three explicit outgoing actions. Bridge batches carry separate burn-in/main arrays, valid lengths, absolute episode/frame identifiers, roles, actions, rewards and both termination flags. Convert to the current head-target convention once in a tested adapter. |
| `d4mj/train.py` | `train_joint`, `freeze_encoder`, `train_bridge`, `train_actor_lewm`, `train_renderer`, `set_phase_mode`, `joint_optimizer`, `phase_optimizer`, `restore_training_state`. `train_joint` returns both encoder and trained predictor. Initialize the API readout with the bundle but leave it frozen/untrained until M4; build heads at Phase 2 and preserve predictor weights, initialize its fresh optimizer explicitly. Actor trainer calls shared imagination/actor-critic math through the world API. Record per-phase schedules, modes, gradients and gates. |
| `d4mj/lewm_diagnostics.py` | `objective_audit`, `representation_retention`, `temporal_variance_report`, `shared_trace_report`, `action_effect_report`, `generated_semantics`, `actor_fork_report`, `renderer_report`, `resource_preflight`, `evaluate_bundle_set`, `aggregate_training_seeds`. Use the evaluation units, controls and uncertainty in the evaluation document; persist raw rows sufficient to recompute reports. |
| `d4mj/renderer.py` | `LatentRenderer.forward(z) -> pixels`; `renderer_loss`; independent decoder identity. Use frozen post-projector latents only, native patch geometry, explicit pixel range, no encoder feature side channel. |
| `d4mj/play.py` | `DreamSession`, `DreamStep`, `seed_session`, `replay_session`, `run_viewer`. Initialization is separate from the environment-free session core. `step` accepts one integer action, advances once, renders once, records predicted outcomes and horizon status. Lazy-import pygame only for the optional UI. |
| `d4mj/experiments.py` | `main`, `preflight`, `joint`, `export`, `bridge`, `actor`, `render_fit`, `evaluate`, `play`, `report`. Parse recipes and checkpoint manifests; never infer training length, encoder settings or rollout depth from directory names. Gate-dependent commands verify gate artifacts and parent identities before launch. |

Protocol conventions: a context of `C` frames has `C-1` outgoing actions; policy features exist at all `C` frames. The new state's `step` counts completed pairs since the supplied context start. Absolute archive frame indices belong in metadata, not an assumed full-episode hidden state. `observe` consumes a previous state and real action once; `start` consumes no action. Return types include the new state and current readout, not a mutable shared cache.

## 4. Existing module changes, function by function

| Existing path | Required change / intentionally retained behavior |
|---|---|
| `d4mj/config.py` | Keep `Config` and all legacy assertions/properties unchanged. Add no new family fields to its serialized dictionary. Shared consumers accept structural settings interfaces or the explicit new `HeadSettings`/`ActorSettings`; no fake legacy model configuration. |
| `d4mj/representation.py` | Leave `Encoder`, `Decoder`, `Projector`, `pack`, MAE losses, `representation_loss` stub and `update_target` untouched for legacy behavior. New joint representation lives in `lewm.py`; the existing train-only `Projector` is not reused as the exported LeWM projector. |
| `d4mj/backbone.py` | Keep `Layout`, `Backbone`, `Block`, `Attention`, masks and `rope` for old worlds. Reuse a simple mathematical helper only if its exact normalization/initialization matches the new recipe. Do not add a one-token special case throughout the legacy layout. |
| `d4mj/time_mixer.py` | Keep `TimeAttention`, `TimeMamba`, `time_mixer` behavior unchanged for old checkpoints. New functional training carry lives in `mamba_recurrence.py`; compare it against this module and pinned source but do not assume existing inference caches support gradients. |
| `d4mj/state.py` | Retain `WorldState`, `RealState`, and `Memory` semantics. Add frozen `PredictiveState(latent, memory, history, step)` and a new real-state wrapper if needed by adapters. Document the completed-pairs invariant. Memory tensor batching is defined by the adapter, not by flattening spatial slots. |
| `d4mj/transition.py` | Keep `World`, `initial`, `observe`, `advance`, `commit_inputs`, `transition_loss`, `_direct_loss`, shortcut/flow routines intact. `LegacyWorldAdapter` calls them. New family must never pass through tanh readout, flow corruption/conditioning, `_direct_loss` detaches, or Direct's external action mixer. |
| `d4mj/data.py` | Keep `Episode`, existing archive readers, role semantics and old sampling defaults compatible. Extend cache metadata validation to recognize versioned family/encoder identities and declared latent shape/dtype. `EpisodeCorpus.pools` for new batches must use explicit valid window lengths/roles, not legacy encoder receptive field. Joint sampling resides here alongside the existing sampler; `patchify`/`unpatchify` may accept a minimal geometry interface without changing their math. |
| `d4mj/train.py` | Preserve legacy `train_representation`/`train_dynamics`/`train_agent` entry points. Factor reusable optimizer grouping and RMS math only with legacy parity tests. Fix `train_actor` to freeze world mode as well as gradients when it is adapted to the common API; use adapter prefill instead of manual `WorldState` construction for generic execution. Keep legacy cache digests/checkpoint contracts valid; family-specific encoder hashing/encoding use the shared `cache.py` store writer. Do not route new states through `_repeat_memory`, `_counterfactual_path` or `_terminal_path` without the adapter and explicit new target construction. |
| `d4mj/agent.py` | `Heads`, `_centers`, `head_targets`, `head_loss` take the narrow settings interface they actually use. Existing one-agent-token pooling works without architectural changes. Keep separate actor/critic/model bodies and output initialization. Reuse `paired_terminal_loss`; new trainer supplies correct paired positions and support rows. Add validation that BOS/padding cannot become BC classes and masks have nonzero declared denominators. No new task embedding in this first arm. |
| `d4mj/imagination.py` | Generalize `imagine` to advance via `WorldAPI`, preserving a legacy adapter entry path for current callers. Keep `Trajectory` and successor indexing. Obtain start readout from adapter state; use a caller-specified frozen world, actor horizon and independent policy RNG. Do not re-encode generated latents or detach/refresh them with simulator frames. |
| `d4mj/actor_critic.py` | Preserve `lambda_returns`, `actor_loss`, `critic_loss`, `_masked_mean` formulas. Narrow their configuration annotations; add only contract assertions if necessary. Numerical policy/value logic is not an architecture ablation in this plan. |
| `d4mj/checkpoint.py` | Preserve `FORMAT=d4mj_checkpoint_v1` and `save`/`load` legacy behavior. Add named v2 bundle save/load functions with family/schema dispatch, strict source/data/recipe checks and phase-specific contents. Do not globally increment the legacy constant and invalidate old checkpoints. Reject cross-family loading and incompatible BN/preprocessing/cache contracts. |
| `d4mj/sources.py` | Keep `PINNED`, `source_digests`, `verify_sources` legacy closure unchanged. Add `lewm_source_digests` and a versioned manifest verifier for P-TC, C-LWM files, actual ViT constructor/config, C-M2 module+used ops, new runtime code and dependency lock. Record local modifications separately from upstream commit IDs. Renderer dependencies have a separate closure. |
| `d4mj/env.py` | Preserve `reset`, `step`, `_timed_out`, `_frame` math. Expose/read a frozen environment contract for run manifests: exact ID, version, parameters, pixel conversion, action repeat, max steps, reward and terminal semantics. Simulator forks remain an evaluation facility, never a dependency of `DreamSession.step`. |
| `d4mj/execution.py` | Generalize `run_episode` and `evaluate` to `ModelBundle`/adapter while keeping legacy calls accepted. Each family uses its own encoder, normalizer and world state. Preserve `Result`, `_result`, `score`, native termination, categorical evaluation and real achievement truth. Extend output metadata to training seed/family/protocol. Move cross-training-seed aggregation to diagnostics rather than treating episodes as independent trained models. |
| `d4mj/counterfactual.py` | Adapt `collect_outcome_forks`, `_predict_actions`, `_predict_observed_actions`, `outcome_metrics` and `actor_safety_metrics` to bundles. Fork raw simulator states with common randomness, then encode separately per model. Batch candidates using `repeat_state`, not old `B×spatial` assumptions. Preserve actual action equivalences; compute successor-aligned reward/value. Include actor-own-state contexts in addition to archive contexts. |
| `d4mj/diagnostics.py` | Keep legacy diagnostics available. Use world adapters for shared entry points or dispatch explicitly to `lewm_diagnostics`. `multistep_error`/`latent_stats` must not construct old state for the new family. An `outside_unit` statistic is not a validity gate for an unbounded latent. `cost` must include actual optimizer/transfer timings, state bytes, encoder/projector and optional decoder costs, and label fused-kernel FLOP undercounting. |
| `d4mj/gates.py` | Preserve six legacy gates. Add a new-family gate registry using adapter methods and `lewm_diagnostics`; do not make a skipped CUDA/Mamba gate count as pass. Separate structural contracts from empirical semantic/control gates. |
| `d4mj/expert.py` | Retain collection/archive loading behavior. Audit and propagate collector checkpoint, upstream training steps, epsilon/action randomness, archive split and observation modality into the run manifest. Do not add a new collector or silently promote an archive's role for this first offline arm. |
| `d4mj/__main__.py` | Preserve no-argument legacy gate behavior; optionally dispatch an explicit `experiment` subcommand to `experiments.main`. Document `python -m d4mj ...` as the primary new-family interface. |
| `d4mj/__init__.py` | Export only stable bundle/config entry points if needed; importing the package must not construct models, import pygame, allocate CUDA, or start JAX environments. |

Existing `artifacts/eda/check_multiworld_drift.py` assumes a shared tokenizer; do not apply its latent comparison unchanged across families. Port the shared **raw-trajectory** concept into `lewm_diagnostics.shared_trace_report`, with separate encoders and caches. Preserve historical scripts/results; if a user is editing that script concurrently, do not overwrite their work. Other artifact scripts with manual `Config`, `WorldState`, `train_dynamics`, shared encoder or `_repeat_memory` construction are legacy-only until explicitly migrated.

## 5. Training implementation details that need tests

### Joint trainer

Load the sealed TRAIN corpus and source manifest, construct one base model initialization, and persist its digest. Raw and TC runs load that same initial state and replay identical window/projection streams. A dedicated generator drives SIGReg; initialization, sampler and policy generators must not be coupled to loss-dependent code paths.

Use one optimizer over encoder, encoder projector, action embedding, Mamba and predictor projector. Mixed precision must not downcast the regularizer. Log prediction and regularization magnitudes, encoder/predictor gradient norms, latent/residual covariance spectra, window overlap, actual B, BN counters and learning rate. Log numerical summaries, not tensors retaining the training graph.

`set_phase_mode` defines both module mode and gradient ownership explicitly. Activation checkpointing cannot replay a BN update. During export, hash the encoder before and after all inference and verify buffers are unchanged. The cache stores T+1 latents for each T-action episode, not one latent per action. Full-sequence BN training is allowed by the source objective; causal inference checks run with fixed normalization statistics.

### Bridge trainer

Load the joint world exactly. Fresh Phase 2 readout/head initialization and optimizer are recorded separately; world-state hashes prove it was not replaced. Predictor BN is eval-mode while its affine parameters may train. Complete a functional burn-in and detach only once before scored training history. A single shared prefix can seed teacher and recursive branches, but branch state cannot mutate either sibling.

Construct suffix mask by **positions**, not sliced/reindexed targets. Preserve led-to reward and outgoing-action conventions through `to_head_batch`. Explicit teacher/recursive losses use uniform rows, policy uses relevant rows, reward uses non-support rows, continuation includes support. Pair observed/generated suffix losses 0.5/0.5; outside the suffix use observed-only loss. Reduce by valid counts within each disjoint stratum, then combine prefix/suffix means with fixed 0.5/0.5 weights. Renormalize over nonempty strata only and report missing coverage. This is a declared weighting adaptation; report suffix-only metrics separately. Mix continuation main/paired-terminal objectives 0.8/0.2 before RMS balancing; retain the existing paired path's separate alive/dead balance.

Maintain effective recursive depth per row, scheduled depth, successful training depth, and independently validated depth. Missing or shortened windows do not count as full-depth training. Fatal shortage of valid examples must stop preflight, not substitute repeated frames. Record every stage-boundary checkpoint and its DEV gate hash.

### Actor trainer

Snapshot BC; freeze world, readout, reward and continuation parameters **and buffers**. Initialize/retain critic according to the existing phase convention and record it. Prefill cached real contexts with the new API. Generate policy actions and successor states through the same path used in deployment/play. Call existing lambda/PMPO/two-hot functions. Save actor/critic optimizer/RMS state and frozen environment/prior identities. Assert after an update that reward/continuation/world outputs and state dictionaries have not moved.

The 500-update actor screen and 5,000 total are proposed budgets, not a claim that the critic converges within them. If actor improvement fails, use the actor-own-state/oracle diagnostic before extending the run. No online replay or ground-truth reward is introduced into Phase 3 without a new protocol.

## 6. Checkpoints, caches and artifact schemas

The new v2 bundle includes:

- `format`, schema version, family and variant; complete resolved recipe and digest; upstream commits and exact source/dependency hashes; current runtime code digest and dirty patch digest.
- Exact dataset manifest and split IDs, collector access/provenance, environment contract, training/evaluation seed manifests, recipe parent and phase.
- Named encoder, predictor, readout, heads, frozen BC prior and optional renderer state dictionaries; BN running means/variances/counters; module mode/gradient-ownership contract. Missing phase-inapplicable modules are explicit, not guessed from a filename.
- Optimizer, scheduler, AMP scaler if used, loss RMS state, completed optimizer step, total scheduled updates, horizon stage, effective/trained/validated depths and gate artifact identities.
- CPU global RNG, all CUDA global RNG states used, dedicated sampler/model/projection/policy streams, sampler cursor/window IDs and data-loader resume semantics. First implementation may use deterministic synchronous sampling; worker prefetch must not invalidate resume.

Hash tensors with names, shapes, dtypes and stable raw bytes; support BF16 without silently converting the identity to another dtype. Save atomic immutable `step-NNNNNN.pt` files plus a movable `latest.pt` link and `checkpoints.json` hash index; preserve the screen step independently of cadence. Distinguish resumable training bundles from smaller inference exports, and verify frozen parent identities on load. Resume of the same schedule and a budget extension are separate operations.

New cache metadata includes exported family, latent shape/dtype, complete preprocessing, encoder weight **and buffer** digest, architecture/source identity, episode/observation IDs and raw data identity. The predictor is not part of encoder-cache identity; it is part of Phase 2/3 environment identity. A renderer cache is keyed additionally by renderer identity. The old MAE cache schema remains readable only by its matching old encoder.

New experiment outputs use a fresh `artifacts/tc_lewm/<recipe-id>/<seed>/<phase>/` root with `resolved_recipe.json`, `manifest.json`, `metrics.jsonl`, checkpoints and gate reports. Reports cite immutable parent files by digest. Do not reuse `artifacts/generated_drift/...` as a training destination or treat an existing directory name as a successful run.

## 7. Tests and validation code to add

| Test file | Meaningful checks |
|---|---|
| New `d4mj/tests/test_lewm.py` | Compare SIGReg values/gradients to pinned module with fixed projections; constant temporal-offset invariance of TC loss; raw/TC objective separation; both MSE branches receive gradient; exported projector is really used; BN buffers freeze and cache/singleton eval parity. |
| New `d4mj/tests/test_mamba_recurrence.py` | Reference versus full scan versus chunked/step outputs, input/parameter/initial-state gradients, functional convolution carry, long carry, short-sequence edge cases, reset and immutable branching. CPU reference is not labeled production CUDA validation. |
| New `d4mj/tests/test_world_api.py` | Both legacy adapters preserve outputs; new observe/advance each consume one pair; prefill excludes unchosen action; policy features invariant to a future action; selected fork commits exactly once; mixed-state-family rejection. |
| New `d4mj/tests/test_joint_data.py` | Sentinel episode with distinguishable actions/rewards verifies T+1 alignment; no episode/reset crossing; burn-in versus main masks; true-start versus arbitrary-context semantics; terminal/truncation cases; per-loss role routing; cache shape/family/BN mismatch rejects. |
| New `d4mj/tests/test_train_lewm.py` | Tiny end-to-end backward and split-run resume; both encoder/world weights move in Phase 1; Phase 2 starts from those exact world weights; recursive gradients reach early accepted states/carry; suffix masks isolate distinct paths; predictor BN freezes; Phase 3 changes only actor/critic. |
| New `d4mj/tests/test_bundle_checkpoint.py` | v1 still loads; v2 round trip/resume; stale source, data, projector buffers, preprocessing, latent dtype, schedule and parent checkpoint reject; raw/TC mismatch rejects; renderer cannot be paired with wrong encoder. |
| New `d4mj/tests/test_play.py` | Deterministic session replay; every action advances once; all environment calls patched to raise after seeding; no encoder called after seeding; renderer receives only exported latent; user/autoplay switch, cutoff and reset-to-seed are explicit. |
| New `d4mj/tests/test_lewm_diagnostics.py` | Per-family encoding of identical raw traces; known synthetic action-effect equivalences; successor reward/value indexing; no cross-family raw-MSE ranking; hierarchical seed aggregation and geometric score recomputation; stable report parent identities. |

Extend existing `test_agent.py`, `test_imagination.py`, `test_actor_critic.py`, `test_execution.py`, `test_counterfactual.py`, `test_data.py`, `test_train.py`, `test_sources.py`, `test_gates.py` and `test_experiment_diagnostics.py` only where a shared interface changes. Existing `test_backbone.py`, `test_time_mixer.py`, `test_representation.py`, `test_transition.py` provide legacy regression coverage. `tests/conftest.py` can add tiny CPU fixtures and explicit CUDA marks; do not silently replace Mamba with attention to pass tests. `tests/__init__.py` needs no change.

TC-30 supersedes the provisional uniform FP32/BF16 tolerances with measured backend/quantity-specific profiles in `lewm_diagnostics.py`. The RTX 3060 source Triton kernels have a different rounding envelope from the reference recurrence; full-stack convolution carries additionally include upstream layer errors. Record maximum/relative error and gradient checks separately; ill-conditioned near-zero gradients need absolute error inspection. These are proposed engineering tolerances, not source guarantees. A long-horizon divergence cannot be excused just because a short numerical test passed.

## 8. Recipes, launch surface, supporting files

M0–M3 includes `d4mj/recipes/lewm_mamba_raw.json` and `d4mj/recipes/lewm_mamba_tc.json`. A sealed `evaluation_seeds.json` is still required before control evaluation; schema/validation lives in `lewm_config.py`. Store all resolved nested values, not an undocumented pile of CLI overrides. M0–M3 records `requirements-lewm-rtx3060.lock.txt` after source/environment preflight and optional `requirements-play.txt` for the viewer; keep the existing requirements and audited legacy lock usable. Update `third_party/SOURCES.lock` only for newly vendored source actually used, preserving existing pins/licenses. The TC v3 PDF is already local and recorded in `third_party/PAPERS.lock`; an unavailable canonical TC repository is not invented.

End-state CLI below mixes implemented and future commands. Use the [M0–M3 commands](STATUS.md) for the implemented surface; `evaluate`, bridge, actor and renderer commands here are not available research stages:

```text
python -m d4mj preflight --recipe <recipe.json> --dataset <manifest.json>
python -m d4mj joint --run <resolved-run-dir> --stop-at 2000
python -m d4mj evaluate --run <resolved-run-dir> --stage joint-screen --split dev
python -m d4mj joint --run <resolved-run-dir> --resume <checkpoint>
python -m d4mj export --run <resolved-run-dir> --checkpoint <joint-checkpoint>
python -m d4mj bridge --run <resolved-run-dir> --stop-after-stage h2
python -m d4mj evaluate --run <resolved-run-dir> --stage bridge-h2 --split dev
python -m d4mj bridge --run <resolved-run-dir> --resume <bridge-checkpoint>
python -m d4mj actor --run <resolved-run-dir> --stop-at 500
python -m d4mj evaluate --run <resolved-run-dir> --stage actor-screen --split dev
python -m d4mj render-fit --run <resolved-run-dir>
python -m d4mj play --bundle <inference-bundle> --context <seed-context>
python -m d4mj report --manifest <comparison-manifest>
```

`--stop-at` pauses an unchanged full schedule; it does not redefine cosine length. Gate artifacts determine whether subsequent launch commands are valid. FINAL evaluation is a distinct explicitly named command stage with preselected checkpoints and immutable seeds; generic DEV monitoring cannot read it.

At implementation time update repository `README.md`, `d4mj/spec/ARCHITECTURE.md`, `DECISIONS.md`, and `GATES.md` with scoped new-family links and actual implemented status. Do not replace the old decision history with this plan or mark milestones complete before their evidence exists.

## 9. Optional D3 comparator implementation, separated from the first experiment

The first raw/TC pair does not depend on fitting a canonical-size D3 model. To make a direct D3 Craftax comparison later, add `d4mj/baselines/dreamerv3_craftax.py`, `d4mj/baselines/__init__.py`, and `d4mj/recipes/dreamerv3_craftax.json`, plus `d4mj/tests/test_dreamerv3_craftax.py`. Keep upstream `third_party/sources/danijar__dreamerv3` unmodified where possible.

The adapter must expose the pinned Dreamer/embodied environment protocol, native pixel shape and action space, correct `is_first/is_last/is_terminal`, timeout discount, total reward and all22 achievement metrics. Test reset/step/terminal/timeout against `d4mj.env` on identical simulator seeds before training. No symbolic inventory or privileged state may reach the policy unless the comparison declares that modality.

Record exact upstream config/model size, replay/update ratios, action repeat, optimizer and exploration recipe, unique environment steps, wall-clock/GPU usage and all pretraining. An offline-replay adaptation of DreamerV3 is a separate **D3-derived offline control** and must say so; a canonical online result requires an online access protocol for both methods. A reduced D3 model for the local memory budget is a useful size-controlled baseline, not a reproduction of a published 201M run. Integrate its results through the same metric/seed manifest, without forcing its latent state into `WorldAPI`.
