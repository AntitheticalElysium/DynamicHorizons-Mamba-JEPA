# TC-LeWM–Mamba: architecture, implementation and gates

Status: **M0–M3 implemented in the working tree; validation and scope are recorded in [M0–M3 status](TC_LEWM_M0_M3_STATUS.md). M4–M8 remain proposed and blocked.** Design and history reviewed through `162efd1` on `craftax-clean-baseline`, 2026-09-05. No learned-control or architecture-success result is claimed.

## Recommendation and project goal

**Current scope:** the user authorized M0–M3 of the actual architecture, including persistent recurrence and the final state API, and explicitly rejected a disposable prototype. The [risk review](TC_LEWM_RISK_REVIEW.md) retains 12 substantive adaptation groups for the full roadmap; eight concern M0–M3. Component tests reduce engineering uncertainty, not the unresolved scientific risks.

Proceed with a bounded **jointly trained LeWorldModel encoder and action-conditioned Mamba world**, comparing ordinary SIGReg with temporally centered SIGReg. Retain the existing D4-inspired heads and imagination actor training after a separate recursive training bridge. Add a frozen-representation decoder for interactive imagined play.

This preserves the question in the repository [README](../../README.md): can JEPA-style prediction and Mamba memory retain the mechanics needed for control at modest compute? It also serves the goal of taking actions inside a learned game. It changes the original controlled study: a jointly trained encoder/world is a new system family, not a replacement encoder in the shared-tokenizer `Flow|Direct × Attention|Mamba` experiment. Keep that experiment independently runnable and describe the change of scope explicitly.

Direct deliberately replaces D4's generator. It was not intended to reproduce it. Local Flow is also a small, adapted implementation. Neither establishes parity with the published Dreamer 4 system. A well-specified external comparison remains possible, but D4 parity need not be the project's success criterion.

The proposed contribution is **an action-conditioned recurrent JEPA world that supports useful imagined policy learning and measurable interactive simulation on Craftax-Classic under a declared resource budget**. TC-SIGReg, Mamba, and imagination learning are existing ideas; their composition is a research hypothesis, not an established novelty claim. Claim a contribution through the resulting method, analysis, and control evidence, after a related-work check before publication.

## Why this is worth testing, and what it does not solve

Joint learning removes the requirement that dynamics predict a representation optimized independently for reconstruction. Temporal centering changes what the anti-collapse objective demands: variation within a short temporal window must carry structure instead of allowing persistent scene information to dominate. Those mechanisms are relevant to our generated-state/readout mismatch.

They do not prove that the retained variation is controllable, that health and inventory survive a global bottleneck, that a deterministic predictor models stochastic branches, or that the critic values actions correctly. Predictable but irrelevant motion can satisfy the objective. Short-window centering can also underprotect slowly changing information needed for long-term decisions. A framewise 192-dimensional latent may be a new information bottleneck. These are reasons for discriminating tests, not assumptions to hide behind a successful loss curve.

TC-LeWM's reported setting is robot policy learning; it does not establish success for our recurrent imagination actor. Its results and theory do not guarantee Craftax transfer. The source is the [TC-LeWM v3 paper](https://arxiv.org/abs/2607.26924v3), with the [pinned base implementation](../../third_party/sources/lucas-maes__le-wm/train.py) supplying executable objective details. No canonical TC implementation was available on the [project page](https://ryuuchou17.github.io/tclewm/) at this review; it said “Code coming soon.”

History supplies motivation rather than a diagnosis of one guilty component:

- The `64×16` export change improved measured retention. Its existence makes replacing it with one 192-dimensional vector a decision requiring evidence, not a harmless shape cleanup.
- `6350672` did not isolate generated readouts; `2b940f5` corrected the comparison. The corrected frozen-world ceiling failed in `f1d4ebb`. This does not prove every head correction is impossible.
- `4bea608` explicitly increased recursive supervision from two to sixteen steps. `21cd48f` and `586b888` fixed evaluator/configuration lineage. `b545e3a` added shared-trajectory comparison. The new `162efd1` result is mixed: the saved shared-trace rows give depth-16 within-arm policy KL `1.264→0.877`, but top-action agreement `0.474→0.346`. These are 48 evaluation seeds, not independent training replications, and the arm also changes sequence length and terminal sampling. Lower KL does not establish improved decisions or real policy performance.
- Prior outcome and oracle diagnostics implicate both generated-state semantics and value estimates. Good latent similarity, noncollapse, or reward calibration alone is insufficient evidence for actor improvement.

## Benchmark and training-access contract

The requested environment is **Craftax-Classic**, initially `Craftax-Classic-Pixels-v1`, native `63×63` pixels and 17 actions. It is neither Crafter nor Craftax-full. Offline training remains the working assumption: all downstream training uses a frozen, fully identified episode archive. Changing to online collection is a separate protocol decision.

Compare against each arm's own BC policy and the existing MAE systems on matched real evaluation seeds. Published DreamerV3 results are external references until environment, data access, model size, training budget, and metric match. Expert collector training is part of data provenance; an offline archive does not make that upstream experience disappear.

A concrete trap: [Dedieu et al., Table 1](https://arxiv.org/html/2502.01591v3) marks the familiar DreamerV3 score `14.5 ± 1.6` and reward `53.2 ± 8` as **Crafter** results. Its own Craftax-Classic DreamerV3 run reports reward `47.18 ± 3.88`, with score unavailable. Do not turn 14.5 into our Craftax pass threshold. Beating an older score is a useful target; it is not by itself SOTA or a matched sample-efficiency result.

## Reading order

Start with the [shared-runtime integration record](INTEGRATION_REFACTOR.md), [implementation status](TC_LEWM_M0_M3_STATUS.md) and [risk review](TC_LEWM_RISK_REVIEW.md). The remainder describes the full roadmap, with M4 onward still deferred.

1. [Architecture](TC_LEWM_ARCHITECTURE.md): tensor contracts, recurrent state, gradients, phase boundaries, and imagined play.
2. [Decisions and hyperparameters](TC_LEWM_DECISIONS.md): source facts versus proposed settings, explicit deviations, unresolved items, and conditions for changing the recipe.
3. [Implementation plan](TC_LEWM_IMPLEMENTATION_PLAN.md): dependency order, file/function changes, tests, checkpoints, CLI, and artifacts.
4. [Evaluation and stop rules](TC_LEWM_EVALUATION.md): the smallest discriminating experiment, fair comparisons, real control, and validated dream horizons.

The existing [architecture](../spec/ARCHITECTURE.md), [decisions](../spec/DECISIONS.md), and [gates](../spec/GATES.md) remain the specification of the existing family. During implementation, add a scoped cross-reference and a new-family section; do not retroactively rewrite its shapes, state semantics, or experiment results. The old decision identifier `S84` is duplicated; refer to its heading/commit until separately repaired. This proposal uses unique `TC-*` identifiers.

## What is settled versus proposed

Settled by the request: preserve the broad JEPA/Mamba/control goal; target Craftax; include imagined play; document deviations before implementation. Settled by the existing system: action/target alignment, truthful termination handling, independent evaluation, and immutable model/data lineage.

**M0–M3 settings are explicit in the checked-in raw/TC recipes.** TC-29–33 record implementation resolutions, numerical tolerances and launch boundaries. Later-phase numbers remain proposals. Each run seals its resolved recipe, source and data identities before training. The first expensive run must not precede the source, memory, normalization, retention, and resource checks. No architecture can make the requested outcome “extremely likely” or guaranteed from the evidence currently available.
