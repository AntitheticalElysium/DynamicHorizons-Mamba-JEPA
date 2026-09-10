# Evaluation, experiment order, and stop rules

Status: proposed protocol. Read the [architecture](ARCHITECTURE.md) and [hyperparameter ledger](DECISIONS.md). Thresholds below are explicit project tolerances, not numbers certified by a paper. Seal them with source, dataset and seed manifests before inspecting new results.

## 1. Questions and permitted claims

| Question | Evidence required | Evidence that is insufficient |
|---|---|---|
| Does joint TC training retain decision-relevant information? | Frozen probes on held-out episodes, action consequences, observed-path BC, comparison with raw joint training and legacy export | SIGReg loss, variance, cosine similarity, a readable easy frame |
| Does centering help this world? | Raw/TC pair differing only in centering, same initialization/data/updates, replicated generated-semantic and control outcomes | TC compared only with MAE; a better encoder probe with worse generated control |
| Does this system train useful policies in imagination? | Actor improves over its own frozen BC in the real game, without online training access; outcomes and critic checked on actor-own-state forks | Higher imagined return, lower validation MSE, actor beating random |
| Can someone play inside the learned game? | Actions produce coherent decoded futures without simulator refreshes; semantic and latency tests over a stated horizon | An actor rollout hidden in latent space; teacher-forced video reconstruction; occasional true-frame resets |
| Does Mamba cause an advantage? | Later matched temporal-backend comparison at declared capacity/compute, with identical new representation objective | The raw/TC Mamba pair by itself |
| Does joint learning cause the gain? | Later matched joint-versus-frozen representation/world training control | New joint system versus the old MAE system with many other differences |
| Does the system beat D3 on Craftax? | Identified Craftax D3 reference/run, same metric, explicit data/compute/access comparison and uncertainty | Crafter's 14.5 score; offline expert-data agent described as an online 1M-step result |
| Is it SOTA or guaranteed? | Broader current benchmark/related-work review for a SOTA claim; no empirical setup guarantees transfer | Beating one older baseline or satisfying assumptions of an unrelated population theorem |

The near-term success target is the first four questions. A publishable causal account may need the later controls; defer their expense until a promising system exists. Do not require an EMA sweep before testing this direction, and do not claim that omitting it establishes superiority over EMA.

## 2. Frozen data and benchmark contract

Use one identified Craftax-Classic pixel corpus for both new arms. Preserve raw episodes and metadata: environment version/parameters, collector implementation and checkpoint, collector's upstream experience, action randomness/epsilon, native observation conversion, repeat, terminal/truncation flags, all role eligibility flags, episode IDs and split assignments. Archive paths must resolve to checksummed manifests. Count unique transitions, repeated training exposures and upstream collector training separately.

Split by episode before extracting windows. No DEV or FINAL frames enter encoder BN calibration, decoder fit, probe fitting, world/actor training, or cache-statistic estimation. Offline archive DEV and online real-environment DEV are different datasets and must be named separately. Keep all22 achievements, rare event opportunities and terminal causes represented in diagnostic sampling; report denominators rather than interpreting a class with zero examples.

Primary environment: pinned `Craftax-Classic-Pixels-v1`, native 63-pixel images, 17 actions, one simulator step per action, native maximum 10,000 steps. Report any collector cap separately from real evaluation length. Preserve genuine termination versus timeout. Symbolic state is permitted for evaluation labels and oracle diagnostics, never as an unreported actor/world input.

There are three baseline categories:

1. **Causal raw control:** same new family, ordinary SIGReg. This is the first mandatory comparison.
2. **Internal system references:** frozen MAE `64×16` Direct/Flow checkpoints with exact tokenizer/world/head identities and trained rollout depths. Evaluate on the same real seeds/raw traces. They establish where the new system stands; unmatched existing training budgets do not isolate the encoder, joint learning or Mamba. Retrain a budget-matched reference later if the claim needs it.
3. **External benchmark references:** a cited Craftax D3 run and other relevant Craftax methods, with environment, access, model size, metric and uncertainty stated. A local D3-derived offline replay adaptation is labeled as such. For a matched online claim, both approaches need an explicit online protocol counting all unique training interactions and pretraining access.

The current choice is offline, not a settled promise of online comparability. Beating a published number under more privileged offline data can still be reported as a **score under that data regime**, with the access difference beside the number. It cannot establish better sample efficiency. Resource-limited size comparisons and published canonical-size comparisons are different rows, not interchangeable baselines.

## 3. Metrics and uncertainty

For each trained model, evaluate the same sealed set of 64 environment seeds, with independently recorded policy RNG streams and categorical temperature 1. Greedy evaluation is secondary. No checkpoint is chosen on FINAL. Save complete per-episode achievements, returns, length and cause of termination.

For achievement `i`, let `s_i` be the percentage of episodes unlocking it at least once. Report:

```text
score = exp(mean_i(log(1 + s_i))) - 1       # s_i in [0,100], all 22 achievements
mean_achievements = mean_episode(number unlocked)
achievement_reward_percent = 100 * mean_achievements / 22
```

Raw shaped environment return is another metric; do not equate it exactly with achievement count or the above normalized reward. Report all22 success percentages, not only a scalar hiding missing advanced achievements. For external reward comparisons, verify the exact health shaping and normalization of the cited protocol.

Train the raw/TC pair with matching seeds `20260731, 20260732, 20260733` after the initial screen. Compute metrics separately for each trained model; report their mean and per-seed values. Bootstrap **training-seed pairs** first and environment-seed pairs within each sampled training pair, recomputing the nonlinear score for each sampled model before averaging. Use 2,000 draws initially. With only three trained pairs, uncertainty remains weak: show the paired differences and extend to five if the finalist conclusion remains uncertain. Do not present an episode-only interval as training robustness.

For action-fork diagnostics, the unit is the original environment seed/trajectory or fork root, not each of 17 actions as if independently collected. Use identical simulator randomness within paired action comparisons where appropriate. Report effect sizes, uncertainty, opportunity counts and action-effect equivalence classes. Paired environment seeds do not make diverging learned policies visit the same states; that is why the shared-trace and actor-own-state evaluations are separate.

Proposed retention tolerances are `.03` macro-AUC and `.5` mean achievements; they are practical budgets chosen before the run. Use lower 95% paired bounds against these noninferiority margins. If a diagnostic has too few labeled positives to establish a bound, its result is **insufficient coverage**, not pass. Fix one probe family/training budget and label set before the pair: standardized TRAIN features, a linear readout and a fixed-width small MLP diagnostic, identical optimization protocol across arms. Fit scalers on TRAIN only; store probe hyperparameters and selection history in the evaluation recipe. Do not tune probe capacity separately until a preferred encoder wins.

## 4. Gate sequence

### G0 — source, shape, normalization and resource preflight

Before any long run, require:

- Loss/output/gradient parity with C-LWM's executable SIGReg under fixed directions, including source integration scale and batch axis. TC regularizer is invariant to adding a time-constant offset per window; raw regularizer generally is not. This tests the regularizer, not invariance of the whole predictive model.
- Nonzero encoder gradient from both prediction branches, no EMA/stopped target, no decoder gradient, and exact sentinel action/reward alignment. Perturbing a future action cannot change earlier eval-mode predictions or the policy readout choosing that action.
- Encoder BN train/eval behavior documented; eval batch/cached/single-frame parity; immutable export buffers; full objective evaluated at actual B128. A checkpointed BN double update is a failure.
- New recurrence scan/step/prefill/reset/branch and gradient tests on the actual execution path. Inference fast-step success does not count as recursive-training gradient success. Test module buffers and caller state, not just predicted tensors.
- Target-GPU peak memory for a real joint optimizer step, bridge H16 backward, and actor rollout; step throughput including optimizer/transfer; parameter/state/cache bytes. Repeat after kernel warmup. A CPU pass does not validate CUDA kernels.

If B128 does not fit, first use valid pure-block activation checkpointing and profile the source loss. Do not silently shrink the representation, pool time into the statistical batch, or accumulate approximate BN/SIGReg microbatches. A memory-saving implementation must pass full-batch value/gradient/BN parity on a device where a reference fits. Otherwise stop and explicitly revise the statistical batch/recipe for **both** arms before training.

### G1 — one-seed joint screen at update 2,000

Run the matched raw/TC pair from the saved identical initialization, using the same TRAIN windows and projection draws. The full schedule is already 10,000 updates; the screen pauses it.

Inspect raw/residual/persistent-component covariance spectra, feature means/scales, temporal frequencies, action-conditioned prediction versus persistence, and export-versus-CLS retention. Residual variance rising is not success if it encodes unpredictable noise. Mean drift is also relevant because TC does not constrain a time-constant offset through its residual regularizer.

Stop for implementation/normalization failure, nonfinite training, verified information destruction, or failure of the intended objective contrast. A weak early probe is not proof of final failure: if mechanics work and learning is progressing, complete the predeclared 10,000 updates on both arms. Do not extend beyond that budget merely because no desired result has appeared.

### G2 — frozen representation and observed control

At the completed joint checkpoint, freeze/export and compare:

- CLS versus projected latent within each encoder; TC versus raw at the same coordinate width; each versus the old export on identical held-out raw frames. Probe health, inventory/resources, selected local tiles and action-relevant prerequisites, with explicit labels/support. Audit the projection boundary before choosing a larger dimension.
- Current-state and short-future consequence probes, not only static scene identity. Use TRAIN-only fit and held-out episode evaluation; report whether failure comes from visual extraction, projection, temporal state, or readout.
- Observed-path BC after the common Phase 2 H2 startup, using its own complete recurrent state. Compare TC/raw and old BC with the `.5`-achievement noninferiority margin. This comparison is unavailable before heads have actually trained; do not invent a BC result at export.

For retention, the projected representation should meet the `.03` macro-AUC noninferiority margin relative to its own CLS and the preselected reference on the critical probe suite. A failed classifier on a sparse class does not prove irreversible loss; use the second fixed probe to separate a poor linear interface from lost information. Both failures with adequate support are a stop signal for this export.

If static retention fails, do not add EMA, a reconstruction loss, a larger vector and new crops simultaneously. Diagnose the boundary; propose at most one evidence-based repair and a new paired recipe. If static retention passes but generated control fails, increasing latent width is not the default remedy.

### G3 — action consequences, generated semantics and recursive H16 bridge

The latest legacy result makes this distinction concrete. Recomputed from `artifacts/multiworld_drift/multiworld_drift.json` (SHA-256 `2bc56a45c68a89e4c72d752e7b233a38ce534d5667c5e62a953b3ce151819b66`): on nonterminal shared traces, first averaged within each of 48 environment seeds, H16 versus H2 training changes depth-16 within-arm observed/generated policy KL from `1.2638` to `0.8775`, while top-action agreement drops from `0.4745` to `0.3458`. The comparison also changes sequence length; commit `162efd1` records different terminal-root counts. This is not an isolated horizon effect or a real actor return result. The saved rows inspected here do not include the entropy fields added by that commit; its further entropy interpretation is not used as verified evidence in this plan.

After H2 bridge, use common raw traces and simulator forks, encoded separately for each family. Evaluate at depths1/2; after the gated longer bridge evaluate1/2/4/8/16. Use recorded actions for shared-trace comparisons and separately sample actor actions for actor-distribution tests.

Measure:

- Action-effect predictions against actual next-state changes, persistence and marginal-action baselines. Use semantic labels plus within-family normalized latent errors. Do not rank a 192-D unbounded latent against 1,024 bounded coordinates by raw MSE. Check no-op/effect-equivalent actions and action sensitivity where the environment really has consequences.
- Frozen observed-path policy distributions and task probes applied to generated states; compare with the same world's observed successor readouts. Report KL/agreement, policy entropy, margins and consequent action regret, not just latent cosine. Entropy alone does not determine KL or prove why an argmax changed. Cross-world shared traces do not share an encoder or feature coordinate system.
- Reward error against a zero/marginal predictor, continuation Brier/BCE and calibration split by live/dead, and action ranking on opportunity strata. An always-continue head must fail the balanced terminal diagnostic.
- Stochastic successor mode fidelity: compare predicted semantics/decoded tiles with possible real successor modes using repeated simulator seeds. A deterministic conditional mean can minimize MSE while being an invalid game state. Do not represent deterministic rollout diversity as uncertainty calibration.
- Persistent memory utility with the current observation held fixed and earlier context varied, plus context-reset/truncated-prefix sensitivity. A nonzero memory influence is a structural property; improved consequence/control prediction is the useful result.

Permit the H16 bridge only when H2 meets source contracts, retention, valid action-effect improvement over persistence/marginal controls, and outcome calibration better than the declared trivial baselines with paired uncertainty. At H16, require those conditions at the candidate horizon and no retention/observed-BC violation. For balanced terminal BCE, the constant-balanced reference is `log(2)`; compare both classwise and aggregate losses. Where sparse opportunities make superiority unresolved, collect **evaluation** coverage before another training intervention and record its selection.

For raw versus TC, prefer TC only if its generated-semantic/control improvement is supported without sacrificing retention. If they tie, retain raw as the simpler supported objective and report that centering did not earn its complexity here; the joint-Mamba family can still be useful. If both fail similarly, do not turn the raw/TC experiment into evidence that Mamba, EMA, or all JEPA models are impossible.

### G4 — actor learning in the real game

Freeze the accepted world/BC checkpoint, including all buffers. Run the 500-update actor screen, then continue to the sealed 5,000 budget only if the learned environment remains valid and the critic/action diagnostic does not reveal exploitation or a broken treatment.

Primary control success: positive actor-minus-own-BC mean-achievement improvement with a lower 95% paired bound above zero, no point-estimate decrease in official score, and no hidden collapse on safety/terminal opportunity strata. Report score uncertainty independently; a claim of score improvement specifically requires support for that score difference. The system-level target of beating D3 uses the separately identified benchmark protocol, not this internal gate.

If imagined return rises while real performance falls, fork states from the actor's **own** real trajectories. Compare candidate actions using common environment randomness, successor-aligned reward/continuation and value, with true-successor and oracle-return substitutions. Include SLEEP, survival/resource tradeoffs and ordinary navigation. This separates transition error, outcome error, and critic error. A good observed-successor world can still be undermined by poor values.

Record generated state-action positions per optimizer update. H16 versus H2 increases samples as well as horizon; a horizon-effect claim requires matched generated-position budget or reports both budgets plainly. Do not extend the actor past trained/validated world depth or add simulator rewards into its objective to rescue a failing result without a new protocol.

### G5 — actually playing in imagination

Two distinct tests are required:

1. **Renderer ceiling:** reconstruct held-out real exported latents. Report native pixel error and semantic/HUD readability against real images. Decoder underfit and exported-information loss must be separated before blaming the world. A higher-capacity renderer, if needed, is a declared display-only change and must be used consistently on real/generated latents.
2. **Generated session:** initialize once, then execute actions without any simulator reset/step, encoder call, true reward, oracle inventory update, nearest-neighbor real-frame substitution or other hidden refresh. An automated test makes those calls raise after seeding. Record every generated state and chosen action; replay must reproduce the session within numerical tolerance.

Evaluate action-controlled events, persistent inventory/health consistency, death/continuation, invalid-action equivalence and stochastic-mode coherence over a stated horizon. A separate evaluator may run true simulator forks for comparison, but no truth enters the session being measured. Human readability is reported alongside semantic measurements, not substituted for them. Do not invent a learned score from simulator achievements unavailable inside the dream; all in-session rewards are labeled predictions.

First establish 16-step play fidelity and p95 action-to-frame latency ≤500ms on the stated target hardware. Longer64/256-step sessions require a new longer generated-prefix training recipe and the same semantic tests at their endpoints. The UI may allow extrapolation beyond validated depth with an explicit indicator, but that is not passed validation. Full native 10,000-step play, an unconditional reset generator, and calibrated stochastic branching remain separate future capabilities.

## 5. Minimal run sequence and cost controls

1. Complete G0 with synthetic/small real fixtures and measured full-batch steps. No architecture comparison before source/gradient/mode correctness.
2. Train one matched raw/TC seed to update2,000; G1. Resume the original schedule to10,000 only if its defined conditions hold.
3. Export; fit fixed retention probes; train both H2 bridge startups to obtain comparable observed BC/outcomes; G2/G3. Stop or complete both H16 bridges according to the sealed criteria.
4. If a useful system emerges, run the two additional seed pairs. Keep the initial seed in all aggregate reports, including if it was unfavorable. Select a finalist on DEV with all selection history retained.
5. Train actor screens and complete the finite accepted budget, evaluate actor versus own BC, then perform the single predeclared FINAL comparison. A failed critic diagnostic is investigated before any longer run.
6. Fit/evaluate the display decoder from the frozen export and demonstrate no-refresh play. This can proceed alongside accepted cached-latent work; it cannot change the representation. Extend dream horizon only through a separately budgeted curriculum.
7. Fund the D3 adapter/online protocol or extra attribution controls only after these evidence gates justify them. Recheck current relevant baselines before a publication-level novelty/SOTA claim.

This is not a promise that all stages will run. Every failed gate produces a report specifying the observable, affected component, source/recipe/checkpoint identities, effect size and uncertainty, smallest discriminating next diagnostic, and whether the first recipe stops. There is no automatic search over latent size, centering window, EMA, prediction loss, actor horizon and decoder at once.

## 6. Required final report

Deliver one comparison manifest linking raw rows, parent checkpoints and source/data recipes; per-seed training/access/compute budgets; official score and achievement distributions; actor-versus-own-BC differences; real/generated outcome and value diagnostics; maximum validated dream horizon and session logs; renderer real-latent ceiling and generated examples; and a list of failed or untested claims.

Separate observation from interpretation. A positive result supports this tested configuration and protocol. A negative result identifies which proposed mechanism failed under the measured conditions. Neither licenses a retrospective rewrite of the old `64×16` decision or a guarantee about another architecture.
