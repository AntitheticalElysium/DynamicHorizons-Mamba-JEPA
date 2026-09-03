# Decisions and function plan

Companion to `ARCHITECTURE.md`. That file says what the system is; this one says
what is settled, what is open, and exactly which functions exist.

The contract: **no new files, and no function that is not listed here.** Private
stateless tensor helpers are permitted and are listed like everything else — the
defence is planning every signature in advance, not forcing arithmetic to be
duplicated inside oversized functions. Anything absent below is a design change,
not an implementation detail. If the plan is wrong, fix the plan.

## Settled

| # | Decision | Basis |
|---|---|---|
| S1 | Patch size 7, no resize, 81 tokens — one token per rendered tile | Craftax-Classic obs is 63×63 with `BLOCK_PIXEL_SIZE_AGENT = 7`; 63 = 9×7. Divisors of 63 are {1,3,7,9,21,63}, of which only {7,21} are tile-aligned and non-degenerate. 7 is the finest. `[R0-CHOICE]` — no genericity claim is made |
| S2 | `Z*` = D4 `tanh` bottleneck, **unnormalized**; direct readout is `tanh`-bounded | Keeps the flow anchor paper-faithful and Stage A genuinely fixed-representation. Matches the readout's codomain to its target's, which is the principle behind V-JEPA's `normalize_reps` for unbounded features. Bounds range only -- it cannot prevent conditional-mean collapse when the fixed-context target is genuinely multimodal (S35) |
| S3 | Policy/reward MTP `L = 8`, leads `n = 0..7`; policy lead 0 = outgoing `a_t`, reward lead 0 = incoming `r_t` | §3.3 states `L = 8` for policy and reward only. Module defaults agree in both reproductions, but edwhu's eval script runs `L: int = 2` and `nicklashansen/dreamer4` has no MTP heads at all, so the paper is the basis and the reproductions only fail to contradict it. Eq. (9)'s inclusive `Σ_{n=0}^{L}` reads as nine terms and is recorded as ambiguous. Continuation is separate under S75 |
| S4 | Reward and value: 255 bins, symlog support ±20 | D4 §3.3 "Following Dreamer 3"; D3 `rewhead`/`value` `bins: 255` with centres `symexp(linspace(-20,0))` mirrored. Two-hot is linear interpolation between neighbouring centres, so bin count sets grid density, not a quantization floor; the wide support is miscalibration headroom, not wasted resolution |
| S5 | `time_every = 4` | D4 §3.4. Reproductions ship 2 and 1 — not inherited |
| S6 | λ-returns in next-index form, `G_t = r_{t+1} + γc_{t+1}[(1−λ)v_{t+1} + λG_{t+1}]` | Eq. (10) prints same-index reward, continuation and value, which is not self-consistent with §3.3's shared-index annotation. `dreamerv3/agent.py:487` indexes `rew[:, 1:]` and `boot[:, 1:]`, expanding to exactly this form |
| S7 | λ = 0.95 | `[DESIGN]`. D4 is silent; DreamerV3 `lam: 0.95` is implementation precedent only |
| S8 | Temporal state is a per-layer, per-slot pair; the conditioning slot is retained in every arm | `ARCHITECTURE.md` Boxes 3 and 4 |
| S9 | One `World.forward` evaluates any block; `World.predict` turns its features into a latent | The occupying latent and its conditioning are per-call arguments, not state. Flow runs N candidate rungs then one commit; direct runs a predictor head then one commit (S34) |
| S10 | The signal table has exactly `k_max` rows; every context index is clamped to `k_max−1` | Eq. (4)'s grid tops out at `1−d`, so τ = 1 is never trained. MMBench2 sizes `signal_embed` at `k_max+1` and labels uncorrupted context with index `k_max` — a row its own sampler cannot reach; `plan_cem.py:189,545` pass `tau_ctx=0.0`, so its planner runs entirely on that untrained row, and `round(0.9·k_max) = k_max` also fires at `k_max = 4`. Same pathology as the predecessor's unreachable shortcut rows, in the upstream source |
| S16 | τ_ctx = **0.1 noise / 0.9 signal**, snapped to the nearest trained bin | `train_dynamics.py:576` states it verbatim: "tau_ctx is the noise fraction: 0 = fully clean, 1 = fully noisy". The paper's "signal level τ_ctx = 0.1" is a slip — 0.1 *signal* contradicts its own word "slightly". No longer open |
| ~~S17~~ | **Superseded by S34.** The in-block query is untrainable in one pass, which S17 did not see: a position cannot hold the real latent for later blocks to attend to *and* the query to be predicted from. Direct now predicts from committed world features. The firewall objection S17 raised against that is answered by reading spatial and register outputs, which the dynamics mask makes agent-free | |
| S18 | Invariant 5 is `[DESIGN]`, not `[D4-ENTAILED]` | No audited source persists dynamics state across accepted frames — MMBench2 re-draws context noise per frame (`z0_ctx = torch.randn_like(...)` inside the per-frame call) and re-prefills. Persistence is our deviation and the `[MAMBA]` efficiency hypothesis rests on it |
| S11 | Context corruption is applied once at commit, never at read | Fresh noise per read changes the prefix every frame, invalidating any KV cache or SSM state and contradicting invariant 5 |
| S12 | Agent slots exist from Phase 1B, last in the layout, masked both ways until Phase 2 | Fixes `S`, mask shape, stream count and state shapes once, so no Phase-1B gate is invalidated when the agent modality activates |
| S13 | `depth % time_every == 0` and at least two time layers | At `depth = 4` the entire Mamba blast radius would be one module and parameter matching would be dominated by it |
| S14 | Anchor drops GQA and attention-logit soft capping, keeps RoPE and QKNorm | Table 2 shows GQA at FVD 70 → 71, adopted for KV bandwidth at 2B parameters — no benefit at our scale. RoPE and QKNorm have executable precedent in pinned MMBench2 attention; GQA and capping have none anywhere. Anchor is described as paper-constrained *modulo declared omissions* |
| S19 | Flow is **5 backbone passes/frame**, direct is **1** plus a predictor head | Flow's commit is a real extra pass: S11 puts corruption at commit, and the final rung's input is the running Euler iterate, not a fresh corruption of the accepted latent. Direct needs no candidate pass at all under S34. 5-vs-1 is an *evaluation-count* ratio; `diagnostics.cost` measures whether it is a throughput ratio |
| S20 | Causal temporal tokenizer, as D4 §3.1 | The primary model follows Dreamer 4. Both arms share one encoder, so the T-vs-M comparison stays fair even though the encoder carries history; a frame-only encoder is a later ablation, not a fork |
| S21 | `k_max ≥ 8`, declared in `Config` alongside `K = 4` | `round(0.9·k_max)` hits the untrained top row exactly at `k_max = 4`, and S10's clamp then drops τ_ctx to 0.75. Both failures vanish at `k_max ≥ 8` (τ_ctx = 0.875). k_max sets the training noise grid; K is the generation rung count — they are independent and neither was registered |
| S22 | The reward and continuation caused by `a_t` are read at lead 0 of `h_{t+1}`, never `h_t` | S3 defines reward lead 0 as the reward *arriving*. `a_t` is chosen at `h_t`; its consequence arrives with `o_{t+1}`. Reading lead 0 at `h_t` returns the previous action's reward and shifts every return by one step — the predecessor's `reward_logits[:, 0, 0]` is correct only under this reading. Realised in `imagination.imagine`, which reads both from the readout `advance` returns; **no gate asserts it** -- doing so needs the gate to run a rollout, which it does not |
| S23 | `Z*` is defined with representation corruption off | MAE replacement probability, EMA context masking, and LeVJEPA token dropping are Phase-1A mechanisms only. Cached targets and deployment use the declared dense export |
| ~~S24~~ | **Superseded by S34.** With no candidate pass there is nothing for a `{candidate, commit}` table to distinguish, so direct's conditioning slot carries a single reachable embedding | |
| S25 | Action table has `n_actions + 1` rows, the extra being BOS. No query token exists (S34) | Reachable at every true episode start; the predecessor shipped 18 rows undocumented. The query-shape half is void under S34, which has no query |
| S26 | Committed and observed blocks carry the finest step index `d_min` | The signal bin is fixed by S16; nothing fixed `d`. MMBench2 labels context blocks with `d_min`; training samples `d` per block, so a committed block needs a declared value rather than an inherited one |
| S38 | `long_only_fraction = 0.25` and `rms_decay = 0.99` are `[DESIGN]` | D4 specifies a final long-only finetune and running-RMS normalisation but neither number. Registered so they are not read as paper values |
| S39 | Separate-image training is **not implemented**, and the claim is withdrawn | D4 uses it to generate start frames without context; every rollout here begins from a committed dataset context, so the mechanism it trains is never exercised. The earlier `_isolate_rows` cleared target flags without changing the temporal context at all -- measured, the loss was bit-identical with and without it, which is worse than an honest omission |
| S40 | MSE on masked patches, LPIPS on the whole predicted frame | Scoring visible patches under MSE rewards copying, which masked autoencoding exists to avoid. That leaves `p = 0` images carrying perceptual signal alone, which is why LPIPS must not be composited. Eq. 5 masks neither term; both halves are declared |
| S41 | A quarter of flow training rows carry the *rollout* prefix: every block but the last at the commit condition | Independent per-block draws make that joint prefix vanishingly rare -- measured, a 48-block prefix uniformly at the commit condition has probability 0.000000, while at rollout it is every prefix. Per block the condition is in distribution; the prefix never was. `commit_prefix_fraction` is `[DESIGN]`: it must be nonzero and small enough not to displace diffusion forcing |
| S42 | The attention temperature is clamped, as the pinned source clamps it | S14 dropped the paper's logit soft capping alone; without the clamp too, nothing bounds the logits at all, and a runaway temperature saturates attention to one-hot with no diagnostic. The two decisions belong together |
| S43 | The 50/50 mixture belongs to **Phases 2 and 3**. Phases 1A and 1B sample the whole corpus and score every row. In Phases 2 and 3, half the rows are drawn uniformly and half are drawn so the window *contains* a task event -- §4.1 names "behavioral cloning, reward modeling, and reinforcement learning", so imagination RL starts from mixture contexts too, not the pretraining distribution; BC scores the relevant half, the continued dynamics loss the uniform half, reward and continuation everything. `relevant` is a **sampling role**, `None` in pretraining. Nothing falls back to the other pool | Applying this to pretraining was a real error, now fixed: D4 pretrains the world model on the full corpus (§3.3, Algorithm 1) and only then finetunes with the mixture. Restricting pretraining to uniform rows discards half the corpus. §4.1, verbatim: "we use data mixture of 50% uniform sequences and 50% relevant sequences that accomplish one of the tasks. The behavioral cloning loss is applied only on the relevant fraction, while the dynamics loss is applied only on the uniform sequences **to avoid optimistic generations**." A dynamics model fitted only on successful play learns that things tend to go well, and imagination inherits it. Reward/continuation are not restricted because the paper restricts the other two by name and says the mixture amplifies signal *for* reward modelling; halving that head's data would be an addition, not a reading. Relevance is per step (`Episode.events`), not per episode: a random window from a successful 2500-step episode mostly shows walking. Verified: pretraining dynamics responds to every row; finetuning dynamics is bitwise unchanged when relevant rows are perturbed, and BC is the exact mirror |
| S44 | Blocks are gradient-checkpointed, and `batch` is **4** | The 6 GB probe, run per-configuration in fresh processes because in-process trials do not release memory and gave non-monotonic garbage. Phase 1A binds. Without checkpointing the declared architecture fits **only batch 1 at the short length** (2.69 GiB) and the long batch never fits -- the previous `batch: 8` was unreachable by 8x, so every phase-1A number to date would have come from a config that cannot run. With checkpointing: 3.00 GiB at batch 8 short, 3.07 GiB at batch 4 long. Batch 4 is the largest that runs *both* lengths, which D4's alternating short/long schedule requires. Checkpointing is exact -- measured bitwise-identical loss and gradients on attention, 1e-10 relative on Mamba from kernel nondeterminism -- so *it* moves cost, not results. **The batch change does not**: halving the batch halves the examples per update and raises gradient variance, so it is an optimization change and the effective batch/update budget must be registered with any result produced under it. No Z*-defining field changed. Confirmed end to end: all four arms, all four phases, 3.20 GiB peak |
| S45 | Both time mixers start from the same weights wherever they share a parameter | `manual_seed` alone does not achieve this. Construction draws in order and the two mixers consume different numbers of values, so every parameter built after the first time layer diverges. Measured: 36 of 99 shared tensors differed between the arms -- everything from block 4 onward -- so the Mamba-vs-attention comparison was also a different-random-init comparison, on the one axis the project exists to measure. Shared weights are now copied from the attention arm by name and shape; 99/99 identical |
| S47 | Where Phase-1A memory goes, measured, and why checkpointing was the right lever rather than a workaround | Profiled per submodule rather than assumed. Nothing is unvectorized and attention is not the problem: `F.scaled_dot_product_attention` already selects a backend costing 36.4 MiB for our shapes, *better* than forced mem-efficient (52.7) or forced math (77.5), and flash is unavailable only because the model runs fp32. The cost is the ordinary transformer intermediates against a 5.1 MiB activation: SwiGLU's gated 4x expansion saves 40.6 + 20.3 + 20.3 + 5.1 = **86.3 MiB per layer, matching the measured 86.3 exactly**, and space attention 52.1. Eight layers of that is the 1.4 GiB encoder forward. So there is no hidden waste to remove -- recomputation is the only lever that does not change the architecture |
| S48 | `Batch.scored` is per block and per row; **S31's single burn-in integer is withdrawn** | A block is faithful once it holds a full receptive field, and unconditionally in a window starting at the episode start, where nothing earlier is missing. One integer masked the first `receptive_field - 1` blocks out of *every* row -- so the states deployment actually begins from were never supervised, while still costing full activation memory. Measured: 65% of encoder memory was spent on blocks producing no loss term; the scored fraction rises from 34.8% to 51.1%. A quarter of uniform rows are drawn at the episode start, because at a uniform start those windows arrive with probability 1/span (~0.5%) and the fix would be inert |
| S49 | The archived replay is loaded, not regenerated, and `train_expert` is withdrawn in favour of `expert.load_archive` | There is no expert to retrain: the archive's manifest carries `params_sha256` and `replay_sha256`, and the corpus supplies both §4.1 sampling roles (S46), so nothing is "missing" in the mixture sense. Broader collection is a *support-coverage* decision -- terminal exposure (S50) and behavioural diversity -- not a requirement the paper imposes. Conversion is exact and lazy -- crop and permute are views, so 320 episodes cost 0.86 GiB RSS against an 8.6 GiB file, and the 696,746 transitions match the manifest. `events` is reconstructed from the per-frame cumulative achievements the archive already stores. The previous pipeline was right where this one had regressed: its `_dead` reads `in_lava | (player_health <= 0)` directly, exactly the fix applied to `env.step` this session |
| S50 | The archive's 2500 cap is **ours, not the environment's**, and its cost is terminal scarcity rather than truncated windows | Craftax's native horizon is 10000 and no archived episode reaches it, so all 252 of 320 cap endings are our own truncation and only 68 episodes died. Against this architecture the cap is harmless for window sampling -- the mean episode is 2177 steps, 34x `sequence_long` and 45x `dynamics_context`, and only 3 episodes are too short for a long window. What it does cost is terminations: 68 terminal transitions in 696,746 is 0.0098%. The per-step figure must be computed under the *actual* sampler -- episode-uniform, then start-uniform -- not by multiplying the global frequency by the block count, which assumes block-uniform sampling and is optimistic. Measured over the real archive: **0.00209 terminal blocks per short batch (one every 479 steps)** and 0.00563 per long batch (one every 178). An earlier figure of "one every 160" used the block-uniform shortcut and is corrected here. Task events are not scarce by comparison (0.95%, 0.61 per step). Terminal exposure is therefore a collection problem for the uniform half, not an argument about the cap, and raising the cap would make it worse by lengthening the surviving episodes |
| S51 | **Revised by S69, S80 and S89. Active task:** one aggregate Craftax-Classic task; "any first achievement" is the relevance event | No task tokens exist in the active architecture, so the policy and value optimize aggregate reward. Mean unique-achievement count remains its causal metric; S89 specifies, but does not activate, the task-conditioned successor |
| S52 | **Evaluation protocol; metric gate revised by S80** | Native **10000**-step horizon, categorical sampling at temperature 1, paired seeds and retained raw rows remain unchanged. Controls are the actor's own frozen BC prior and random. Mean achievement count, official geometric score, raw return, per-achievement rates, termination and length are all reported. Count, reward and geometric gaps each receive their own paired percentile interval; no metric substitutes for another. Under the active aggregate task, an arm passes the causal actor gate only when the lower bound of its mean-count advantage over both controls exceeds zero. DEV and FINAL remain disjoint and FINAL is opened once |
| S53 | **Parameter matching**: at most **0.5%** deployed-parameter residual, shared dimensions held fixed | `d_state` is the single matching knob (S28). The measured residual at `d_state = 64` is -0.316%, which passes, so 64 is settled by a declared rule rather than by being the value that happened to be there. Tolerance and rule are fixed before any result is read; a later arm that misses it must move `d_state`, not the tolerance |
| S54 | **Imagination horizon is not settled at 8.** `horizon` is a smoke default; the final value is selected on DEV from `horizon_candidates` | Blessing a default because it ran is how an arbitrary constant becomes a result. Selection uses `diagnostics.multistep_error` under a *full* committed context, which is why that diagnostic was corrected from its one-block start -- choosing a horizon from a model with almost no history selects for the wrong regime. Selection happens on DEV, never against executed-control FINAL numbers |
| S55 | **Generated-prefix contract**, closing the open register row | One-step teacher forcing plus a two-step autoregressive rollout through the real `advance` path; the two rollout terms **averaged**, following the pinned V-JEPA 2-AC source, which computes `jloss + sloss` with each a mean. Squared error rather than the source's L1 (`loss_exp: 1.0`), because S35's conditional-mean analysis is stated for squared loss -- a declared deviation. **Incoming recurrent memory is detached in both arms.** Official Mamba-2 mutates its `InferenceParams` cache in place, so its step is not differentiable in the state it receives while attention's is: measured, an attention prefix took gradient 147.92 through carried memory and a Mamba prefix took `None`. Truncating both is a *matching* choice, and it is registered here rather than only explained at the call site because it changes the objective both arms optimise |
| S56 | Uniform sequences are drawn uniformly over eligible **(episode, start)** pairs, with an explicit episode-start stratum | Picking an episode and then a start weights every episode equally regardless of how many windows it contains: measured on this archive, a single window from the shortest eligible episode was hundreds of times likelier than one from a 2500-step episode. `episode_start_fraction` of both uniform and BC rows begin at the episode start. Terminal supervision no longer replaces a main-mixture row; S72 gives it a separate batch |
| S57 | `Episode` carries **three independent facts**: `events`, `uniform_eligible`, `bc_eligible` | Relevance-for-BC was previously inferred from `events.any()`, so any rollout that happened to unlock one achievement became eligible for behaviour cloning. Deliberately degraded support data would therefore have entered BC, which is the one thing it must never do. The archive is `bc_eligible=True`; the support corpus is `uniform_eligible=True, bc_eligible=False` while keeping its true events and rewards |
| S58 | A **support corpus** is collected: archived PPO with epsilon-greedy noise at 0.1, 0.25, 0.5 and 1.0, every rollout kept, until ~320 genuine terminal episodes | The archive gives the continuation head 68 terminals in 696,746 transitions. Pure random noise is too narrow a failure distribution, so the epsilon ladder supplies failures near competent play as well as early ones. These episodes remain uniform-eligible and BC-ineligible. The old one-row-every-eight routing was insufficient and is superseded by S72 |
| S59 | Continuation is diagnosed **split by target**, not by its global mean | A constant "continue" head matches the global mean on this corpus. `continuation_separation` and terminal-conditional probability are therefore pooled from explicit DEV terminal tails, while S73 tests the quantity deployment needs: calibrated ranking over counterfactual fatal and nonfatal actions |
| S60 | The critic gets its own trunk, and Phase 3 sizes its own batch | Policy and value shared `actor_body`, so a value-only backward put gradient 17654 into the body the policy reads: the critic reshaped policy features outside PMPO and outside the prior KL meant to bound how far the actor may move. Measured `None` after the split. D4 calls it "an additional value head" and does not ask for a shared trunk. Phase 3 also inherited Phase 1A's batch of 4 -- a memory ceiling from a phase that runs the tokenizer, imposed on one that never does -- while PMPO's sign-of-advantage estimate is over starting *contexts*; `actor_batch` is 16. Measured together on flow-attention, actor-minus-BC moved from **-1.59 to +0.37** |
| ~~S61~~ | **Withdrawn.** The claim that flow's reward models carry no information rested on a zero baseline measured on the wrong windows: it used *uncached* DEV episodes (burn-in 30, short/long mix of its own) while the model MAE came from the *cached* DEV batches, and it ignored `reward_rows`. On identical cached windows with the correct mask both flow arms beat zero. The baseline is now computed inside `head_calibration` on the same rows it scores, so the two can no longer disagree. What survives is narrower and belongs to S64 | Original text: Phase 3 gated on the reward model beating a zero predictor | Measured on DEV, the zero-predictor MAE is 0.0795. Both flow arms score worse -- 0.098 and 0.086 -- so PMPO was optimising a reward signal carrying no information, while direct scored 0.043-0.049. D4 assumes the learned reward model is useful because Phase 3 optimises it directly; that assumption has to be checked rather than inherited |
| S62 | The S54 horizon rule as first written degenerates and must be restated before it decides anything | "Largest candidate within `horizon_tolerance` of the one-step error, else the smallest" selected 32 for flow-attention and 4 for the other three -- but in those three *no* candidate met the tolerance, so 4 was a fallback rather than a selection. A tolerance relative to the one-step error is tighter in absolute terms for a more accurate arm, which is backwards, and it cannot distinguish "4 is good" from "nothing is good". The mechanism (DEV, declared candidates, full-context roll) stands; the criterion does not |
| S63 | The imagination horizon is the largest declared candidate at which the rollout still beats the **marginal predictor** -- the constant mean latent | S62 rejected the previous criterion; this replaces it. The line is not a tuned threshold: past it the rollout carries less about the future than knowing nothing, so imagining further cannot inform the actor. It is scale-free, which a tolerance relative to the one-step error is not -- that rule was *tighter* for a more accurate arm and degenerated to "no candidate qualified" on three arms of four. It doubles as a collapse test, since a predictor collapsed to the conditional mean scores at the marginal by construction. **The Lemma 1 reading is withdrawn.** The reported statistic is the per-step ratio of rolled MSE, which measures error *accumulation*; Lemma 1's hypothesis is about perturbation *sensitivity*, and the two differ -- an error recursion `e -> 0.5e + 1` has sensitivity 0.5, converges to a fixed point, and still gives a ratio above 1 at every horizon, so the statistic would call a satisfied hypothesis vacuous. Lemma 1 also assumes deterministic dynamics, which a sampled flow transition is not. The number is kept as a descriptive accumulation rate with no bound attached; the marginal-predictor criterion stands on its own |
| S64 | Head output scales follow the pinned DreamerV3 config: `reward` and `value` at 0.0, `policy` at 0.01, `continuation` at 1.0 | `configs.yaml` sets `rewhead: outscale 0.0`, `value: outscale 0.0`, `policy: outscale 0.01`, `conhead: outscale 1.0`. We shipped PyTorch defaults on all four. A value head starting at random emits random advantages on Phase 3's first updates and PMPO reads only their *sign*, so the actor's first moves are noise -- a plausible contributor to the measured post-RL regression. The consequence is worth stating: a zero-initialised reward head is uniform, and a uniform distribution has cross-entropy `log(bins)` against every target, so it begins with no preference at all |
| ~~S65~~ | **Closed by S69.** Event-only BC made 84.5% of expert transitions unreachable as targets | The measurement remains the reason S69 widened relevant sampling; it is no longer the live sampler contract |
| S66 | **Corrected.** Phase 2 degrades the world model's rollout in every arm, but the cause is the shortcut objective, not the loss balance | Measured on identical DEV batches, mean rolled error over 32 steps before and after Phase 2: flow-attention 0.648 -> **1.347**, flow-mamba 0.683 -> 0.828, direct-attention 0.323 -> 0.327, direct-mamba 0.329 -> 0.373. Phase 2 continues the dynamics loss beside three head losses, all normalised to unit RMS by `_balance`, so dynamics carries about a quarter of the weight it held in Phase 1B while head gradients reshape the world through the agent tokens' inputs. My original explanation -- dynamics reduced to a quarter weight, head gradients reshaping the world -- is **withdrawn**. Replaying Phase 2 from the flow-attention Phase-1B checkpoint with *only* the dynamics loss gives 0.648 -> 1.026, against 1.347 for the full phase: continued dynamics training alone accounts for **54%** of the degradation. The unstable shortcut objective (S67) is the primary cause and head training compounds it. Both flow arms are already worse than the marginal after Phase 1B alone |
| S67 | Shortcut training partitions **rows**, not positions, and bootstraps only after `bootstrap_start`; corrected further by S71 | The pinned mmbench2 source is explicit -- "empirical rows are finest (d_min), self rows sample coarser" -- with `self_fraction=0.25` and `bootstrap_start=10_000`. The former per-position implementation bootstrapped untrained predictions immediately. Phase 2 continues the Phase-1B clock |
| S68 | Imagination may not exceed the rollout length the arm's loss trains | `_direct_loss` commits exactly `direct_rollout` = 2 generated states, and the run deployed a horizon of 8: from the third imagined step both the transition and the head inputs are outside anything training visited. The docstring claiming "the heads see every state Phase 3 will read them at" was false and is corrected. The horizon is now capped at `direct_rollout` for the direct arm; lifting the cap means training the rollout through the horizon, not raising the number |
| S69 | **S51 revised.** Behaviour cloning reads ordinary expert behaviour, with task events *oversampled* rather than exclusive | D4's relevant sequences are task-conditioned. S51 dropped task conditioning for one aggregate Craftax policy but kept an event-local relevance rule, and the two are incompatible: for an aggregate imitation policy, navigation, survival, recovery and positioning are all expert behaviour worth cloning. Measured, event-only windows made **84.5%** of expert transitions unreachable as a BC target *at any training length*, and only **3.44%** of BC windows began at an episode start, because `episode_start_fraction` applied to the uniform half that BC never reads. `event_fraction` of BC rows are now event-centred and the rest are ordinary windows drawn uniformly over eligible starts, with the episode-start stratum applying to both. On the same 2500-step schedule, coverage goes 15.5% -> 41.8% and episode starts 3.44% -> 13.78%, and what remains uncovered is merely unsampled rather than unreachable |
| S70 | **Phase-2 outcome supervision is the first universal collapse in the 20k DEV run** | An environment fork executed all 17 actions at 36 identical real states. Encoded action effects remain nonzero and matched-action successor MSE beats off-action MSE in every arm, but true death risk under the actors is 13.1–14.4% while the continuation heads predict only 0.046–0.070% death even on true encoded successors. Logged-action calibration therefore gated the wrong distribution. Raw measurements are preserved in `artifacts/stage_a_olddesign/counterfactual_forks_preterminalfix.json`; S72 and S73 implement the response |
| S71 | **Shortcut warmup and mixture routing now match the pinned objective** | The source reserves `self_fraction` rows at coarser steps from the start but gives them zero loss until step 10,000; ours incorrectly turned them into additional finest-step empirical rows. It also always put the self row last, while our mixture always put dynamics rows last, so 46.7% rather than 25% of Phase-2 dynamics rows bootstrapped. Self rows are now randomly assigned independently of semantic roles, inactive before the exact boundary, and active afterwards. The Phase-2 checkpoint contract includes the inherited world-step clock, and the runner refuses a Phase 1B that never crosses the boundary |
| S72 | **Terminal supervision is a stratified Bernoulli likelihood, not a positive-class weight; corrected by S75/S76** | D4 names nonterminal `c_t` but does not specify its estimator. The closest pinned implementation is Dreamer 3's ordinary binary continuation likelihood. We retain it on the main mixture and add one tail-aligned terminal sequence per Phase-2 update. It never displaces BC or dynamics rows and cannot enter their losses. Tail stratification is a declared Craftax adaptation, not attributed to D4. S75 repaired the path; S76 repairs the Direct path/label confound found by the matched-fork diagnostic |
| S73 | **Phase 3 is conditional on a real all-actions outcome gate** | On separate DEV environment seeds 12000–12007, the BC prior visits scheduled and final pre-death states and the simulator executes all 17 actions from each identical state. Generated-successor reward choice regret and terminal BCE must beat the best state-blind **action marginals** on those forks, terminal AUC must exceed 0.5, and each outcome needs at least three varying states. This rejects a model that only learns globally safe or rewarding actions without reading state. Raw forks are saved. After Phase 3, the actor is marked failed if its exact policy-weighted one-step death exceeds its BC prior. Craftax is pinned to CPU so JAX cannot reserve the PyTorch training GPU. The known-bad Direct-A checkpoint is rejected under this gate; S70 preserves its raw measurements |
| S74 | **Corrected DEV budget: all 320 expert episodes; 20k world, 10k agent, 2.5k actor updates per arm** | A 4k world run never reached the source's 10k shortcut bootstrap; the 20k run then showed that 2.5k head updates left outcomes unidentified. The earlier `--expert 96` also discarded 224 available expert episodes and left only 76 in training. The restarted run uses the full archive, refuses `dynamics_steps <= bootstrap_start`, and writes a fresh directory, so no incompatible checkpoint is migrated |
| S75 | **Continuation is one-step and terminal tails use the ordinary arm-specific Phase-2 path; Direct corrected by S76** | D4 specifies MTP only for policy and reward; extending it to continuation was an unsourced local choice, while the pinned Dreamer 3 continuation likelihood is one binary prediction per state. More importantly, the S72 branch called `world` directly on real latents: it bypassed Flow's normal sampled corruption and Direct's generated-prefix readouts, repeating the predecessor's D043 real-versus-imagined head mismatch. The tail therefore obtains its readout from `transition_loss`. Its dynamics loss remains masked out by the support role, so this changes neither the §4.1 mixture nor the dynamics objective. S76 supersedes only the Direct continuation objective |
| S76 | **Terminal tails pair path and label: observed/generated x alive/dead, in both arms** | The ordinary tail has many observed nonterminal states but its only terminal is one of Direct's two generated-prefix states. It therefore confounded representation domain with the label: observed supplied no death example, and generated supplied only one alive example beside the death. Both arms now score exactly the final alive and terminal state, Direct in both its teacher-forced and generated-prefix readouts and Flow in the single readout it has, passed as both arguments so the average is that readout's own score. This is a balanced auxiliary likelihood; at `terminal_loss_mass=0.2` terminal labels carry 10% of continuation loss, far below the predecessor's failed 61.5% positive mass. **Applying it to Direct alone was itself a defect** -- the tail-wide average it replaced gives a dead-class share of `1/T`, so the arms would have been penalised 8x (T=16) to 31x (T=64) differently for missing a death, confounding the one comparison the project exists to make; measured, `terminal_loss` on an all-alive head scored 0.3775 and 0.0962 against the paired rule's 3.0025. D4 leaves continuation estimation unspecified, so this is a declared Craftax repair, not a paper claim. The main mixture likelihood, BC routing, reward loss, and dynamics routing are unchanged |
| S77 | **Support-v2 targets 10,000 genuine terminal trajectories without changing the competence ladder** | Collection retains every rollout from the same PPO expert with epsilon in `{0.1, 0.25, 0.5, 1.0}` cycled by complete environment batch; support remains `uniform_eligible=True`, `bc_eligible=False`. Each episode records epsilon, immutable identity and a hash-assigned 80/10/10 TRAIN/DEV/FINAL split. The new artifact is sharded and hash-manifested; raw frames and the encoder-digest-bound latent cache are independently mmap-backed, so neither corpus scale nor caching requires resident observations. `craftax_support_v1.pt` remains untouched and every earlier result remains bound to its digest |
| S78 | **Terminal-diversity scaling holds exposure fixed and must return a saturation verdict on conditional consequence learning** | Nested subsets are stratified by source, epsilon and fatal action, with exactly 20k tail draws per cell. S79's cheap gate made the lower rungs unnecessary, so the launched ladder is 300/900/3800/full TRAIN, with two subset replicates and one full endpoint. Primary endpoints are held-out fatal-minus-matched-safe predicted movement and within-state predicted-vs-true slope at 5k/10k/20k; AUC is secondary. Saturation is declared only if both final paired 95% intervals lie inside a two-sided practical-equivalence band of +/-5% of the true held-out conditional movement; otherwise the result is `not_saturated` or `inconclusive`, never an open-ended "more helped" claim | **Answered 2026-08-14: saturated by 900 unique episodes.** Paired increments on the primary endpoint are 300->900 -0.0076 [-0.0114, -0.0037], 900->3800 -0.0008 [-0.0045, 0.0029], 3800->7501 -0.0009 [-0.0041, 0.0022], against a minimum meaningful increment of 0.0225; the final two intervals lie inside the equivalence band and the whole-ladder trend is -0.0026 per log unique episode. Held-out fresh-probe AUC (0.642-0.649) and MSE (0.116-0.127) are flat across all twelve cells. The null is decisive rather than underpowered, which is what the corpus bought |
| S79 | **Supervised identifiability gates S78 before world-model training** | The v1 logged-data probe learned state risk but correct action pairing did not beat the shuffled-action control on fixed forks (0.521 versus 0.518 within-state AUC). Support-v2 must therefore first be tested at increasing unique-episode rungs with fixed optimizer exposure and fixed DEV/fork evaluation. A separate fork-trained ladder holds complete trajectory pairs out while comparing simulator state, current pixels, the encoder's pre-bottleneck tokens and frozen z, each through an action-indexed outcome probe. This separates missing logged conditional coverage from information lost before Direct. The expensive S78 ladder remains stopped until these two reports are interpreted | **Answered 2026-08-13, and it corrects the v1 reading.** With a per-action output head rather than a concatenated action, within-state-centered AUC is 0.611 for raw observation, 0.592 for frozen z and 0.568 pre-bottleneck, with the action-only control at its expected 0.493: the stack loses little, the ordering is non-monotone, and the bottleneck is cleared. The simulator-state row returns 0.515 -- at chance and below the observation rendered from it, which is impossible as a statement about information and is recorded as a featurisation defect, not a finding. Scaling over support-v2 at fixed exposure, state discrimination climbs 0.739->0.882 and flattens by ~3800 while the fork pairing margin over the shuffled control grows 0.002->0.058; the gap between 0.882 on the collection distribution and 0.608 on policy-fork states exceeds both, and 0.608 is near what raw pixels support |
| S80 | **Aggregate-task actor gate: mean achievement count primary, official geometric score mandatory beside it** | Phase 3 maximizes scalar discounted environment reward, not the nonlinear population-level geometric score. On 512 paired DEV episodes, the h=2 fully-oracle actor improves mean achievements by **+0.826 [0.537, 1.119]** yet changes geometric score by only **+0.092 [-1.011, 1.143]**; all actor and BC episodes terminate. The old learned-imagination run also fails under count, so this correction does not rehabilitate it: every arm is below its BC prior, with flow-attention at -1.125 [-2.125, -0.188] and direct-mamba at -0.812 [-1.500, -0.125]. The oracle result establishes a live but weak local actor path, not satisfactory control. The next positive control changes only oracle horizon 2->16 to test delayed survival credit; learned-world horizon results cannot answer that question. If longer truthful rollouts improve survival without broader achievements, the next architecture step is Dreamer 4-style task conditioning of policy, reward and value, with a fixed prompt schedule; ad-hoc reward weighting is not adopted. **Reading stored reports:** artifacts written before this decision carry `contract.version` `oracle-phase3-v2` and a `gate.criterion` of "official-score actor-minus-BC" -- the *superseded* metric -- so their `gate.passed: false` does not mean the run failed the current gate. `oracle_phase3_h2_bccontexts` is one such report: its stored gate reads false on geometric score while the h=2 oracle **passes** the primary achievement-count gate at +0.826 [0.537, 1.119]. Check `contract.version` is `oracle-phase3-v3` before reading any `gate` field; v2 reports also store no achievement interval and cannot be evaluated against this criterion from the file alone. See `artifacts/oracle_phase3_h2_bccontexts/GATE_NOTE.md` |
| S81 | **Before the h=16 actor control, attribute Direct predictor topology and transfer S78's diversity test to Flow** | Dreamer 4 constrains action/latent interleaving inside its generative backbone but publishes no standalone deterministic Direct head. The pinned V-JEPA 2-AC paper and source instead project feature and action inputs separately, insert action before a deep token transformer, and predict feature tokens; official DINO-WM independently inserts encoded actions before its transformer predictor. They do **not** source our exact `pool -> concatenate broadcast action -> 2-layer MLP` head. The Direct test therefore holds the frozen representation, world backbone, squared loss, two-step rollout, data, terminal exposure, optimizer, streams and every shared initial tensor fixed while comparing: the current 0.140M head; a 3.372M deeper pooled MLP capacity control; and a 3.299M four-block token transformer with separate feature/action projections and action present before attention. The latter is source-shaped, not called a V-JEPA reproduction: its per-block interface deliberately preserves S34 because D4 world features already contain causal history. Separately, S78 answered diversity only for Direct. Flow-Attention now receives the same nested 300-versus-7501 terminal subsets with 20k tail draws, common random numbers and otherwise identical Phase-1B training. Both questions use the fixed support DEV pairs and fixed policy forks; fatal-minus-safe movement along the fixed TRAIN direction is primary, ordinary latent MSE and fatality AUC are secondary. A rescue requires a paired lower 95% bound above 5% of the true contrast on both distributions; equivalence requires the full interval inside that two-sided band. No actor or Phase-2 change is mixed into either test | **Answered 2026-08-15: both questions null, and the evaluation split them apart.** Topology is not the cause -- the source-shaped token transformer is equivalent to the current 0.140M head on both distributions (archive -0.0012 [-0.0059, 0.0037], forks -0.0046 [-0.0161, 0.0066]). The 3.372M capacity control gains +0.0183 [+0.0128, +0.0238] on archive, below the 0.0225 materiality threshold, and is equivalent on forks; it also *beats* the source-shaped head on archive (-0.0195 [-0.0258, -0.0134]), so what little moves is parameters, not interleaving. Flow does not respond to terminal diversity: 300 -> 7501 unique episodes moves recovered fraction +0.0023 to +0.0038 on archive and -0.0016 to -0.0036 on forks, every interval inside the band, replicating S78's Direct-only saturation on the other arm. The two-rung Flow ladder establishes no effect between the endpoints and cannot locate a saturation point, which S78's four rungs could. What the grid actually exposed is registered as S82 |
| S82 | **The policy-fork collapse is real, but matched branch coverage is not its demonstrated cause** | S81's grid separated two distributions that every earlier measurement had averaged over. On held-out logged DEV transitions the worlds recover **5-9%** of the true fatal contrast (direct_current 0.053, deep_mlp 0.094, token_transformer 0.050, flow 0.076-0.081) at fresh-probe AUC 0.61-0.65. On the 104 policy forks **every cell recovers zero or less** (-0.006 to -0.037) at probe AUC 0.507-0.579, while a supervised readout can extract consequence information from the same starting latents. Data volume (S78), Flow diversity (S81), predictor capacity (S81), predictor topology (S81), whitening, training length and outcome gradients all failed to repair that fork result. A plausible remaining data hypothesis was the missing joint of off-policy action at a competent-policy state: support-v2 contains 2520 epsilon=1.0 episodes, but random actions are not necessarily paired with states reached by competent behavior. **Tested with a gate before world training:** all 17 one-step outcomes were collected at every state reached by the frozen BC policy on 512 TRAIN seeds disjoint from both fixed fork sets. The resulting immutable collection has 67,365 states and 965 action-dependent survival roots; 781 roots fit the probe and 184 whole-seed-held-out roots tuned it. The positive control used oracle-selected opportunity roots, both a linear per-action probe and the existing small 512->64->17 fork-probe topology, and selected the small probe from TRAIN tune only. On the unchanged 104 DEV forks, branched state+action reaches mean per-state AUC **0.635** and beats its outcome-preserving shuffled-action control by **+0.066 [+0.015, +0.122]**, so the collection contains genuine conditional signal. But it does not establish the proposed rescue: its gain over action-only is **+0.039 [-0.037, +0.116]**, and against the same probe trained for the same 2000 updates on existing logged support it is **-0.023 [-0.079, +0.034]**; logged support reaches **0.658**. Therefore the claim that the corpus lacks the necessary joint is not supported, and Phase-1B branch-window training is not authorized. The existing `collect_paired_trajectory_forks.py` remains evaluation-only; no oracle-filtered data entered production replay. This result restores priority to the deferred objective test: consequence labels must shape the generated latent itself, with a stopped-head control, because a head-only shortcut would not repair imagination. It does not guarantee that objective will work, and the earlier harmful outcome-gradient result remains a required control. **The h=16 oracle is a separate Phase-3 question and is not evidence about the world model.** |
| S83 | **Generated-latent terminal shaping is the final preregistered objective test, not a claimed Dreamer 4 component** | Dreamer 4 jointly continues its video objective while reward gradients enter the shared agent-token transformer, but publishes neither a deterministic Direct predictor nor a continuation estimator. Dreamer 3, DRAMA, NE-Dreamer and Dreamer-CDP jointly let reward/continuation likelihoods shape their world features; V-JEPA 2-AC and DINO-WM instead fit frozen targets with latent prediction alone. Attaching a continuation likelihood directly to Direct's predicted packed latent is therefore a source-motivated test at an unsourced boundary, not paper fidelity. Two Direct-Attention cells retain S78's full 7,501-terminal schedule, ordinary/terminal MSE mixture, initialization, batches, optimizer, streams and 5k/10k/20k endpoints. A shared small latent head sees the same final alive/dead labels on the matching observed and recursively generated successors. The only intervention is whether the generated-latent path is attached or stopped; observed targets never update the frozen tokenizer. The stopped world's update must reproduce S78's full-diversity checkpoint bitwise at every milestone. Head AUC is a validity check, never the endpoint: the primary outcome is paired fatal-minus-safe movement of the generated latent along the fixed TRAIN direction on both unchanged logged DEV transitions and the 104 policy forks, with fresh frozen-probe AUC and MSE secondary. At 20k, a rescue requires the allowed-minus-stopped lower paired 95% bound to exceed 5% of the true contrast on both distributions; equivalence requires both intervals inside the corresponding two-sided bands. The earlier agent-readout consequence gradient, which worsened Direct, and MMBench2's reported reward-gradient null remain adverse controls. A negative result closes this particular terminal-shaping mechanism, not every possible world model. |
| S84 | **The exported representation uses 64 latent slots** | The 32-slot export scored 0.591 against 0.674 before its bottleneck; 64 slots closed that loss at both tokenizer seeds. Under matched Direct training, 64 versus 32 improved paired action-effect NSE by -0.521 and -0.463 and crossed below the no-action-effect null at both seeds. This changes `Z*`, so old latent caches and checkpoints are intentionally incompatible. |
| S85 | **Direct uses one candidate-action token interaction block after pooling** | Against the old broadcast-action MLP, the one-block mixer improved `R_delta` by +0.053 and +0.066, NSE by -0.332 and -0.416, and cosine by +0.071 and +0.104 across two world seeds; all paired intervals excluded zero. The gain arose through joint training reshaping upstream world features, not proof that a specific internal probe drop was causal. The final `tanh` remains: removing it worsened geometric fidelity at both seeds and left the encoder codomain. |
| S86 | **Stage B has three named representation families:** `mae`, `ema_jepa`, `levjepa` | EMA-JEPA-R is V-JEPA-style masked prediction with an explicit predictor and stopped EMA targets. LeVJEPA-R is one shared encoder with global/local views and SIGReg; SIGReg is not an architecture name. Both are causal-D4 adaptations and keep JEPA-R separate from Direct JEPA-D |
| S87 | **LeVJEPA-R uses its published loss interface** | One global and multiple local spatial/photometric views share one temporal interval. Their final clip readouts use global/local MSE with gradients through both sides plus per-view, full-global-batch SIGReg at `lambda=0.02`. The projector is training-only; no objective target, predictor, or stop-gradient exists |
| S88 | **LeVJEPA-R preserves fixed recurrent slots before optimizing compaction** | The published uniform random spatiotemporal keep-set is represented by true-position attention exclusion: excluded slots cannot be read and their outputs are discarded. This preserves the causal tokenizer's slot identity but not LeVJEPA's compute saving. Physical compaction is eligible only after dense-vs-compact output, gradient, and recurrent-state parity |
| S89 | **The final control successor is D4-style achievement conditioning, not reward reweighting** | The active aggregate agent stays as control. An eligible TRAIN-supported subset of recorded achievements supplies one-hot `q` and sparse binary task rewards; only one-way agent tokens, policy, reward, and value receive `q`. The world remains task-independent, and the task set plus evaluation prompt freeze before results. D4 supplies this interface; the Craftax task vocabulary and prompt are local choices |
| S90 | **Each representation declares one dense export and one decoder boundary** | MAE exports its trained encoder and retains its jointly trained reconstruction decoder. EMA-JEPA-R exports the EMA target used by the pinned frozen evaluations; LeVJEPA-R exports its evaluation-only Polyak encoder. JEPA decoders fit only after freeze. Every export is `64x16`, uncorrupted and projector/readout-free downstream; policy or human play uses recursive `Advance`, never decoded pixels as state |
| S84 | **S83's original evaluator tested transfer across rollout paths, not the path it trained** | The 20k run and its causal controls are valid: identical initialization, batches and streams; nonzero gradient only in the attached cell; and the stopped world reproduces S78 bitwise at 5k, 10k and 20k. But a post-run function-level audit found a material endpoint mismatch. `terminal_objective` labels death only on the *second recursive* successor, after the penultimate state has itself been generated; both archive `reset16` and the fixed policy forks instead score one step from an *observed* predecessor. Their null is a valid transfer result, not a test of whether the supervised recursive path improved. The exact held-out archive path check changes the picture: at 20k, stopped recursion recovers **31.5%** of the true fatal-safe movement (0.142 [0.124, 0.159]) and attached recursion recovers **44.5%** (0.200 [0.183, 0.216]), while attached one-step remains at 3.46%. Before inspecting the still-unseen recursive fork result, its correction is fixed: reconstruct the preceding frozen-policy state, generate the fork state with the actually executed incoming action, then apply each of the same 17 fork actions; compare attached minus stopped at 20k with the existing paired whole-state bootstrap and 5%-of-true threshold. This is a post-hoc alignment correction, not a new training result or a production change. It will decide whether the recursive archive gain transfers to policy states; it cannot establish Dreamer 4 fidelity or repair reward prediction by itself. |
| S46 | The archived Craftax replay is **losslessly convertible and usable for both §4.1 sampling roles**; what it lacks is behavioural breadth, not a "uniform half" | Everything below verified by reading the artifact, not from its manifest alone. `artifacts/expert/craftax_expert_v1.pt` (8.6 GB) holds 320 episodes / 696,746 transitions at `mean_achievements` 20.62 of 22, with all 320 flagged deep-achievement. Conversion is exact: `obs` is `(2501, 3, 64, 64)` uint8 channels-first, zero-padded from 63x63 (row 63 and col 63 measured all-zero), so `[:, :, :63, :63]` then permute to HWC loses nothing. It stores only `continues`, so `terminated = continues == 0` and `truncated` is the 2500 cap. **Correction, 2026-08-01**: the claim that this makes the corpus "100% relevant and 0% uniform" was wrong and is withdrawn. S43 defines `relevant` as a *sampling role*, not a property of an episode, and §4.1's language is sequence-level: a 2500-step successful episode contains many ordinary windows holding no achievement event, so this corpus supplies both roles. The generator also never acceptance-filtered by achievement -- it records every completed rollout and counts achievements afterwards -- so it is already unfiltered *within one PPO policy's behaviour distribution*. The fallback that sentence described no longer exists in the code either. What remains true is narrower and is the actual limitation: one strong policy over 320 episodes is far less diverse than 2541 hours of contractor play, its failure support is tiny (S50), and its expert lacks vendored source lineage. Two further defects compound it: only 68 of 320 episodes terminate (252 hit the cap), so the continuation head sees almost no real terminations, and its expert has no byte-level provenance (`Craftax_Baselines@7ce36fa` is not pinned in `third_party/`). Per S29 it therefore stays a smoke-test corpus. What is missing is not more expert play -- 697k expert transitions is already ample for the relevant half -- but an equal mass of unfiltered rollouts, which also supplies the terminations |
| S27 | Bounded encoder context `W`, part of `C*` — `z_t = Z*(x_{t−W+1..t})` everywhere | Phase 1A windows carry a `W−1` burn-in that is encoded but not scored; once frozen, each episode is scanned once and cached **under the same `W` limit** — an unbounded full-episode scan would produce a different `Z*` from deployment. `W` is not a capacity number: it defines the representation, so changing it changes every `z_t` and it must be frozen before the final encoder trains |
| S28 | Match total **deployed** parameters within a declared tolerance while holding `d_model`, depth, token layout and shared interfaces fixed | Primary knob is Mamba's `d_state`, the only one that moves M-arm parameters without touching the shared backbone. Parameter counts move discretely, so `d_state` alone may not reach tolerance at a sane state size — any additional knob must be declared before training. Report the unmatched residual, FLOPs, memory, recurrent-state size and measured throughput regardless. The predecessor called arms matched in a comment while the temporal module differed by 29.6% |
| S29 | Archived replay is for debugging and smoke tests only; regenerated replay backs every reported number | Its expert has no byte-level provenance and its terminal-window support is 58 windows. Keeping both uses named stops the old set from quietly becoming the final dataset, and keeps `expert.py` exercised |
| S30 | The frozen latent cache lives on `Episode` as `latents` + `latent_digest`; `train_representation` writes it at the Phase-1A boundary and `load_episodes` verifies it | The digest covers exactly `C*`: representation family, encoder checkpoint, exported copy, `W`, bottleneck and packing, all training corruption off, and patch size. Without it a cache built under another export is silently reusable |
| S31 | `Batch` names its regions explicitly: `burn_in` (int prefix), `valid` (per-position target validity). The MAE mask is **not** a loader output — the encoder generates it | "masks" was ambiguous across five different things. Burn-in is always a prefix, so an integer beats a mask. Burn-in frames update encoder memory and score no loss; scored frames follow immediately under that memory. `patches` and `latents` are phase-determined: Phase 1A carries pixels, Phase 1B onward carries cached latents |
| S32 | `Encoder.forward(patches, memory, corruption, rng) -> (z, readout, memory, excluded)` | `corruption` selects MAE replacement, EMA masking, LeVJEPA exclusion, or none. `readout` exists only where the representation objective needs it; `z` is always the dense candidate export. Every `Z*` call uses `none` per S23 |
| S33 | `scan_step_parity` covers the encoder; `representation_export` asserts `C*` | Batched `W`-context scan ≡ recurrence ≡ cached `Z*`; frames older than `W` cannot change `z_t`; reset clears encoder memory. Export also checks family, copy, dense shape, and corruption-off identity |
| S15 | Windows never cross an episode boundary; `evaluate` takes no reset mask | Keeps a reset a fresh construction. Asserted by `gates.alignment`; relaxing it changes the signature |

## Design decisions taken on evidence

Three questions that shape the experiment rather than the code. Each was settled
against `third_party/`, not by preference.

### S34 — Direct predicts from an external head over committed world features

**Rejected:** an in-block query trained in one teacher-forced pass. Position `t`
cannot simultaneously hold the real latent, so later positions can attend to it,
*and* hold the query, so it can be predicted from. Filling every position with a
query resolves that by deleting the task: measured, the prediction is bitwise
identical for two different latent sequences, so the model is asked to predict
`z_t` from the action history alone.

**Also rejected:** a two-pass teacher-forced loss (commit pass to build memory,
query pass against it). Correct, but doubles Direct's training cost for a
mechanism no source uses.

**Taken:** `ẑ_{t+1} = P(world features of committed block t, a_t)`, one pass, the
predictor reading spatial and register outputs. This is what V-JEPA 2-AC does --
its predictor is a separate module over the interleaved `(a_k, s_k, z_k)`
sequence -- and what DINO-WM does. Box 4 rejected it over the §3.3 firewall, but
that objection only bites if the predictor pools the *agent* slot; spatial and
register outputs are agent-free by induction over depth, which `gates.firewall`
already asserts.

Dreamer 4 can share its backbone because a corrupted latent at `τ > 0` still
carries signal, so one block is both context and query. A query token carries
none, so the same trick does not transfer.

### S35 — Test fixed-pair multimodality before enabling `K > 1`

Craftax is stochastic, but action-dependent death under one simulator draw does
not prove that a fixed `(state, action)` has multiple successor modes, and low
one-target MSE does not prove Direct lies between them. Those were previously
conflated. The diagnostic must hold both state and action fixed, vary only the
environment RNG, encode every realized successor under identical encoder history,
and compare Direct with the empirical mean and nearest realized mode. Flow is
evaluated on the same contexts by sample-to-mode precision and mode coverage.

MoP-JEPA (`2607.05238`, Prop. 1) proves the consequence: under squared loss the
optimal single predictor is `E[z'|c] = Σ w_m μ_m`, error lower-bounded by the
between-mode variance, and for separated modes the optimum lies far from *every*
mode. Under cosine loss with normalised targets the same holds. Prop. 2 shows a
gated weighted-sum mixture does not escape it -- it still emits one vector.

The fix is Prop. 3: best-of-K regression, `L = E[min_k ‖g_k(c) − z'‖²]`, which is
the per-context K-means distortion, so every optimum assigns a head per mode.
Plus a router trained on the winning index and a load-balance term.

At `K = 1` the MoP loss reduces exactly to dense regression, but `K > 1` still
changes parameters, routing and rollout semantics and remains an ablation. It is
eligible only if terminal-critical fixed pairs are demonstrably multimodal,
their empirical mean lies away from realized modes, Direct lies nearer that mean
than every mode, and Flow handles the same modes materially better.

`diagnostics.multistep_error` remains useful but cannot answer this question:
squared error is minimized by the conditional mean. The fixed-pair probe reports
the needed geometry and raw rows, but makes no automatic MoP decision.

**Measured 2026-08-11 — `K > 1` is not eligible for the terminal-prediction
failure.** `artifacts/diagnose_s35_multimodality.py` on the 100 saved
terminal-opportunity states: 1700 `(state, action)` pairs, 64 draws each, state
and action held fixed and only simulator RNG varied, successors encoded under
identical encoder history, Phase-1B worlds so no head is involved. Raw rows in
`artifacts/stage_a_s76_paired/s35_multimodality.json`.

Of the four conditions above, one holds and three fail:

| Condition | Verdict | Measurement |
|---|---|---|
| Fixed pairs are multimodal | **holds** | 48.1 distinct successors per pair; 1610 of 1700 branch |
| Their mean lies away from realized modes | **fails** | mean-to-nearest-mode MSE 0.00032 (median 0.000145) against an inter-mode spread of 0.00126 — the mean sits *inside* the cloud, beside a real mode |
| Direct lies nearer that mean than every mode | **fails** | closer-to-mean on 25.8% of pairs; mean advantage −0.00026, i.e. the nearest mode is usually the closer of the two |
| Flow handles the same modes materially better | **fails** | Flow sample-to-mode precision 0.0268 against Direct's 0.0232; coverage 0.0169 |

Prop. 1's harm needs *separated* modes. These are not separated on the scale that
matters: Direct's distance to the nearest realized mode is 0.0232, **18× the
entire spread of the successor cloud** and 73× the mean's own distance to a mode.
The prediction is not sitting between modes; it is outside a tight cluster.
Assigning a head per mode cannot recover a distinction that is not there.

Death is also near-deterministic here, which is the more direct reason: it varies
with RNG in **16 of 1700 pairs (0.94%)**, and reference-fatal pairs die on 99.1%
of draws. Fatality is a function of `(state, action)` at these states, so there is
no fatal/safe bimodality for `K > 1` to separate.

**Scope — this retires the lever for one failure, not in general.** It says
nothing about reward prediction, long-horizon rollout drift, or Stage B, and it
does not contradict MoP-JEPA; it finds its hypothesis unmet here. Condition 1
*holds*, so the environment-level branching S35 was built on stands and is
confirmed. Two further limits are recorded rather than smoothed over: where death
genuinely is stochastic, those 16 pairs show a closer-to-mean fraction of 0.75 --
the mechanism does appear exactly where multimodality is real, on under 1% of
pairs and still with inter-mode spread 4× below the prediction error (0.00485
against 0.01963, on the same pairwise measure used above) -- and only the two
attention arms at one DEV seed were measured, with no Mamba arm. Reopen this if a
failure appears where fixed-pair modes are separated at the scale of the
prediction error.

The residual gap is not distributional structure: prediction error is 0.152 RMS
against the DEV latent std of 0.785 (19%), while the successors it must distinguish
span 0.036 RMS (4.5%). **Calling that gap "accuracy" is withdrawn** -- training four
times longer improved aggregate accuracy and made fatality discrimination worse. The
question is a different one with different levers, and it is not S35's; it is
recorded under "Phase-1B consequence investigation" below.

### S36 — Dynamics context `C = 3·T_short`, with `T_long = 4·T_short`

Neither cache is currently bounded, and `Config` had no dynamics-context field at
all. Appendix A gives all three of D4's configurations: Minecraft `C=192,
T1=64, T2=256`; SOAR and Epic Kitchens `C=96, T1=32, T2=128`. Every one satisfies
`C = 3·T_short` and `T_long = 4·T_short` -- an invariant across every
configuration the paper reports, not a single data point.

Taking `T_short = 16` gives `C = 48`, `T_long = 64`. This also satisfies §3.4's
requirement that the long batch exceed the context.

## Open, with the functions each one constrains

Nothing here blocks writing the listed signatures. Each blocks a *body* or a
config value, and each must be closed before the phase named.

| Question | Close before | Constrains |
|---|---|---|
| **EMA-JEPA-R adaptation** — exact context/target masks, predictor size, loss norm, and momentum schedule; all must stay causal and preserve the dense export | EMA Stage B | `data.representation_views`, `representation.RepresentationPredictor`, `representation.representation_loss`, `representation.update_target` |
| **LeVJEPA-R Craftax adaptation** — local-view count/crops, safe photometric transforms, token-drop rate, and the SIGReg sample count. The paper's 0.95 drop and large global batch are source defaults, not Craftax optima; gradient accumulation counts only if the loss sees the joint embeddings | LeVJEPA Stage B | `Config`, `data.representation_views`, `representation.representation_loss` |
| **Task set, prompt, and switch state** — eligible achievements need TRAIN support, sparse rewards, a fixed prompt, a task-neutral continuation readout, and a declared carry/reset rule for agent-only memory. World streams never reset on a prompt change | Task stage | `agent.TaskSpec`, `agent.task_targets`, `agent.select_task`, `agent.Heads`, `state.WorldState` |
| ~~**Generated-prefix contract**~~ — **closed by S55.** Settled: one-step teacher forcing plus a two-step autoregressive rollout through the real `advance` path, the two rollout terms averaged, squared error, and incoming recurrent memory detached in both arms | ~~Before Stage-A direct training~~ | `transition._direct_loss` |
| **Go/no-go threshold *numbers*** — formulas exist now; scales come from the anchor or a pilot; numbers freeze **before any experimental cell is inspected**. Choosing them after all four cells train is not preregistration, whatever the intent | After the anchor, before inspecting Direct/Mamba cells | `Config` |
| ~~**Capacity**~~ — **closed for MAE by S44**. EMA adds a target and predictor; LeVJEPA adds multi-view activations and global-batch SIGReg. Each gets a fresh worst-case probe before Stage B | Each Stage-B arm | `Config` |
| ~~**Imagination horizon**~~ — **closed by S54.** The selection *rule* is now fixed: DEV only, from `horizon_candidates`, via `multistep_error` under a full committed context. The resulting number is not yet chosen, and choosing it is a run, not a decision | ~~Phase 3~~ | `Config.horizon` |
| ~~**Matching tolerance and final Mamba dimensions**~~ — **closed by S53.** 0.5% deployed residual, shared dimensions fixed; `d_state = 64` measured at -0.316% and passes | ~~Before building the Stage-A models~~ | `diagnostics.cost`, `Config` |
| ~~**Executed-control metric definition**~~ — **revised by S80** after the oracle positive control exposed objective/evaluation disagreement. Aggregate-task mean achievement count is the causal gate; official geometric score is mandatory beside it. Both use paired intervals against BC and random, with native horizon and raw rows retained | ~~Before Stage A~~ | `execution.run_episode`, `evaluate`, `score` |
| **Behavioural support of the corpus** — closed operationally by S77, with causal adequacy still gated by S78. The archive serves both §4.1 roles (S46), so this is not about a "uniform half". V1 deaths are broad under coarse checks -- 0-22 achievements at a median of 12, only 15.7% inside 100 steps -- but those checks do not establish state-by-action consequence coverage. S77 preserves that policy/epsilon ladder and raises unique terminal count without relabelling support as expert data | Before reported Stage-A training | `expert.collect` |

## Function plan

Modules in architecture order. `Type` means a dataclass or `nn.Module`;
everything else is a function. Every entry maps to a box in `ARCHITECTURE.md`.

### Contracts

**`config.py`** — `Config` (Type, frozen). Every constant, plus
`representation ∈ {mae, ema_jepa, levjepa}`, `transition ∈ {flow, direct}`,
`time_mixer ∈ {attention, mamba}`, and `task_mode ∈ {aggregate, achievement}`.
Experimental arms are `Config` values; there is no arm factory.

**`sources.py`** — `source_digests(config)`, `verify_sources(recorded, config)`.

**`state.py`** — `WorldState` (Type: latent, memory, step), `RealState` (Type:
encoder_memory, world). No functions: a reset is a fresh construction, and
non-mutation is a property of the time mixer rather than a caller duty.

### Environment and data — Box 1

**`env.py`** — `reset(key)`, `step(state, action)`. `step` returns `terminated`
and `truncated` as separate raw fields; continuation is derived in `agent.py`.

**`expert.py`** — `load_archive(path, config, limit)`, `collect(policy, count, config, limit)`.

**`data.py`** — `Episode` (Type, unshifted storage: `observations`,
`actions_taken`, `rewards`, `terminated`, `truncated`, `events`, eligibility,
epsilon and declared split), `EpisodeCorpus` (Type, mmap-backed indexed sequence),
`Batch` (Type, block arrays, task id, plus sampling/support roles),
`RepresentationViews` (Type: source indices, true token positions, global/local views),
`patchify(frames, patch)`, `unpatchify(patches, config)`, `episode_splits(n, seed)`,
`sample_batch(episodes, rng, config)`, `sample_terminal_batch(episodes, rng, config)`,
`representation_views(batch, rng, config)`,
`save_episodes(path, episodes)`, `save_episode_shard(path, episodes)`,
`load_episode_store(path)`, `load_episodes(path)`.

`relevant` is the S43 sampling role. `Batch.rows` is public because behaviour
cloning and dynamics select complementary halves and neither owns the rule.

`Episode` and `Batch` are the two index conventions, deliberately named apart.

### Representation — Box 2

**`representation.py`** — `Encoder` (Type), `Decoder` (Type), `Projector` (Type),
`RepresentationPredictor` (Type),
`pack(z, n_spatial, k)`, `reconstruction_loss(pred, target, mask)`,
`representation_loss(online, target, predictor, projector, views, config)`,
`update_target(online, target, momentum)`.

`Encoder.forward(patches, memory, corruption, rng) -> (z, readout, memory,
excluded)` is all of `C* ∘ E*`. EMA requires `target` and `predictor`; LeVJEPA
requires neither and reads its one-way final-time readout through `projector`.
`update_target` is objective EMA in the former and checkpoint-only Polyak in the
latter. `Decoder` is diagnostic-only after freeze.

### Backbone and time mixer — Box 3

**`backbone.py`** — `Layout` (Type: segment order, slices, modality ids),
`space_mask(layout, mode)`, `rope(x, positions)`, `Attention` (Type),
`SwiGLU` (Type), `Block` (Type), `Backbone` (Type).

`mode ∈ {encoder, decoder, dynamics}` is the only place the three masks differ.
`Block` is pre-RMSNorm → space → optional time → MLP, matching the source layer
exactly. `Backbone.forward(x, memory) -> (x, memory)`.

**`time_mixer.py`** — `TimeAttention` (Type), `TimeMamba` (Type),
`time_mixer(config)`.

Both types share `forward(x, memory) -> (y, memory)` and never mutate `memory`
in place. This module is the entire Mamba blast radius.

### Transition — Box 4

**`transition.py`** — `World` (Type), `flow_conditioning(rng, shape, config)`,
`observe(world, encoder, state, led_to_action, patches, q, rng, config)`,
`advance(world, state, led_to_action, q, rng, config)`,
`transition_loss(world, batch, rng, config)`.

`World.forward(state, led_to_action, latent, conditioning, q) -> (latent_out,
agent_out, state_out)` is the generic `evaluate`. `advance` is *N* read-only
candidate evaluations plus exactly one commit evaluation — flow N=4, direct N=0
per S34 — and always reads `h` from the commit pass. `q=None` is the aggregate
agent and every Phase-1B call. `observe` is the one place a real frame becomes `(e_t, z_t, m_t, h_t)`. It takes
`rng` because the flow arm corrupts the committed latent at τ_ctx while the direct
arm commits it clean; it exists because training,
imagination context construction and execution would otherwise each inline
encode-then-commit, which is the accretion this contract forbids.
`transition_loss` covers shortcut-flow and direct feature prediction, including
the rollout schedule, selected by `config.transition`.

### Agent — Box 5

**`agent.py`** — `TaskSpec` (Type: eligible achievements and fixed prompt),
`TaskEmbedder` (Type), `Heads` (Type: policy, reward, continuation, value),
`task_targets(batch, config)`, `select_task(events, task_spec)`,
`twohot(values, centers)`, `head_targets(batch, config)`,
`head_loss(predictions, targets, config)`,
`paired_terminal_loss(generated, observed, targets)`.

`value` exists from construction but enters no optimizer before Phase 3.
`head_targets` is where MTP lead alignment and the terminal/truncation split are
realised. Achievement mode replaces scalar reward targets with the selected
event's sparse reward and supplies `q` only to agent-side paths. Continuation
must read a task-neutral feature; that exact readout remains open.

### Imagination and improvement — Boxes 6, 7

**`imagination.py`** — `Trajectory` (Type),
`imagine(world, heads, state, agent, q, rng, policy_rng, config)`.

The caller — `train_actor` — builds the starting state by repeated `observe`;
`imagine` receives it complete and owns no encoder. Box 6's "encode and scan the
context" describes the caller's work, and that boundary is where the
predecessor's context re-slice bug lived.

`Trajectory` carries **soft** `continuation` — the head's probability, not a
boolean — because imagination has no ground-truth termination. DreamerV3 passes
`self.con(inp, 2).prob(1)` and sets `term = 1 - con`, so the probability enters
the discount directly (`agent.py:206,404`). It carries no boundary mask: that
repo sets `last = jnp.zeros_like(con)` for imagination, and the horizon end is
the recursion's initial condition `R_T = v_T`, not a mask. Hard `terminated` and
`truncated` stay in `Batch`, where they are the continuation head's targets.

**`actor_critic.py`** — `lambda_returns(trajectory, config)`,
`actor_loss(logits, actions, returns, values, prior_logits, config)`,
`critic_loss(logits, returns, centers)`.

`actor_loss` is PMPO plus the reverse KL to the frozen prior.

### Execution — Box 8

**`execution.py`** — `Result` (Type: seed, steps, reward, terminated, truncated, achievements), `score(results)`, `evaluate(policies, seeds, config)`, `run_random(seed, config, limit)`,
`run_episode(world, encoder, heads, seed, config, task_spec=None)`.

**`counterfactual.py`** — `OutcomeForks` (Type),
`collect_outcome_forks(world, encoder, prior, config)`,
`outcome_metrics(forks, policy, config)`, `actor_safety_metrics(before, after)`.

It executes every action from immutable real states, compares learned outcomes
against action-only marginals, and gates both entry to and exit from Phase 3.

### Orchestration

**`train.py`** — `optimizer(modules, config)`, `train_representation(config)`,
`cache_latents(encoder, episodes, config)`,
`cache_latents_to_store(encoder, episodes, config, out)`,
`train_dynamics(config)`, `train_agent(config)`, `train_actor(config)`.

One driver per phase, in phase order. `optimizer` is the only place parameter
groups are built, and the only place upstream `_no_weight_decay` is honoured.

**`checkpoint.py`** — `save(path, config, **state)`, `load(path, config)`.

**`__main__.py`** — `main()`. **`__init__.py`** — exports only.

### Validation

**`gates.py`** — `alignment(config)`, `scan_step_parity(config)`,
`reset_parity(config)`, `firewall(config)`, `branch_nonmutation(config)`,
`recurrent_carry(config)`, `representation_export(config)`,
`representation_retention(config)`.

`scan_step_parity` covers four things: scan versus recurrent step, teacher-forced
forward versus the equivalent reconstructed `evaluate`, encoder scan versus
recurrence, and the windowed branch -- its sequence runs past `dynamics_context`
on purpose, since at a shorter length that branch never executes.
`recurrent_carry` additionally asserts every conditioning row receives gradient
(S10's pathology) and is the only place the flow arm's *history* dependence is
tested, `_observation_dependence` being satisfiable there by the current block
alone. Nothing else exercises `advance`, so without the second half the
most bug-prone function in the system ships untested until Stage-A results are
already contaminated.

`alignment` carries the Box-1 fixtures: length invariants, no-action-leak,
future-observation leakage (with MAE masking disabled or seeded), reward shift,
window-start action identity, the separate-image fraction, that the transition
loss is finite, and that the prediction moves when only its context does. It does
*not* cover S22's imagination read index, which would need a rollout.

**`diagnostics.py`** — `multistep_error(world, batch, config)`,
`latent_stats(world, batch, config)`, `head_calibration(heads, agent, batch, config)`,
`cost(modules, config)`.

`latent_stats` reports range *and* scale, so S2's residual — contraction toward
the conditional mean — is measured rather than assumed away. `cost` reports
deployed parameters, training-only parameters, FLOPs, memory, throughput, and
`e_t`/`m_t` state sizes separately, per invariant 5, which is why it takes the
whole module set rather than the world alone.

## Totals

Planned after the Stage-B/task additions above: 25 types and 71 public functions
across the same 20 modules. The current implementation remains at 21/66 plus 51
private helpers until those stages begin.

The private count has grown from the planned 18. That is drift the contract exists
to catch, and it is recorded rather than rounded away: the growth is real, most of
it in `gates.py` and `train.py`, and it should be pushed back down before Stage B
adds more. This sweep moved it in the right direction for the first time -- two
`_device` copies deleted and a duplicated mixture fallback consolidated into one
`data.mixture_weight` -- but consolidating the rest is outstanding work.

## Signatures corrected during implementation

Each was forced by a contract already in this document, and each is a defect the
plan would otherwise have shipped.

| Signature | Why |
|---|---|
| `observe` takes and returns `RealState`, not `WorldState` | It has to carry the encoder's bounded-window memory, which `WorldState` does not hold. Merging them is how a rollout starts from zero memory while looking correct |
| `imagine` takes the starting `agent` readout | Recomputing it would ingest the same latent twice, against the rule that `m_t` already covers block `t` |
| `Trajectory` carries `agent` | The frozen prior and the critic must be evaluated where the actions were chosen, not at the first state alone |
| `diagnostics.cost` takes the module set and the world | It cannot separate deployed from training-only parameters given the world alone |
| `actor_loss(trajectory, returns, prior_logits, config)` | It needs the logits, actions and values together and they already travel as one `Trajectory`; passing them apart invites a mismatched slice |
| `World.forward` returns memory, not a `WorldState` | `WorldState.latent` is defined as the *accepted* latent, which only a commit site can supply; a candidate has no accepted latent to put there |

`representation_loss` raises `NotImplementedError` naming the remaining Stage-B
choices rather than guessing defaults. `expert.train_expert` did too; S49
withdrew it.

## S37 — `window` bounds state, not receptive field

Measured: with two encoder time layers at `window = 4`, perturbing frames more
than 4 back still moves `z` by 1.8e-3, and beyond 8 back moves it by **exactly
zero**. Influence travels one window per time layer, so the reach is their
product while each layer's cache stays bounded to `window`.

The state bound is the load-bearing half -- it is what makes the cache and the
deployed rollout produce the same latent for the same frame. `Config.receptive_field`
names the other half so no claim confuses them.

Also settled by execution: the tokenizer always mixes time with attention, in
every arm. Mamba's state summarises all history rather than a window, so a Mamba
encoder cannot honour the bound that makes `Z*` well defined -- and keeping the
encoder common is what confines the substitution to the dynamics.

## Phase-1B consequence investigation — measurements only, nothing decided

**No decision is recorded here and no remedy is adopted.** This section states what
was measured between 2026-08-10 and 2026-08-13, so the next choice is made against
evidence rather than memory. Every figure is on the same 100 matched
terminal-opportunity DEV states unless stated, dead-vs-safe compared within the
same action, with `action_identity_only` at exactly 0.500 as the control. AUCs are
pooled-pair unless named macro; the two differ by about 0.02 and the register was
previously quoting both without saying which.

**The failure.** A linear probe reads fatality from the real encoded successor at
**0.923**. From the model's generated successor it reads **0.646** (Direct) and
**0.649 / 0.680** (Flow, one sample / eight averaged). Imagination therefore
presents an actor with a world where death is nearly invisible: modelled death
under policy 0.0009-0.0019 against a true 0.12-0.13.

**Where it is not.** Each of these was a live hypothesis, and each is closed by
measurement rather than argument:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| The encoder | no | observed-latent probe pinned at 0.916 macro across every stage and arm; death transitions move 4.40x in pixels and 3.30x in latents (compression 0.749), and 7.91x along the fatality direction |
| Action conditioning | no | shuffling the outgoing action raises MSE 2.44x |
| The continuation head | no | frozen world, 10k steps, trained *only* on generated states, caps at 0.527 |
| Terminal data volume in Phase 2 | no | terminal-dynamics mass 1/3 moved the generated latent 0.686 -> 0.676 macro |
| Multimodality / MoP-JEPA `K > 1` | no | S35, measured: modes unseparated, death RNG-stochastic in 16 of 1700 pairs |
| Undertraining | **inverted** | 20k -> 80k: latent error 0.0242 -> 0.0186 while fatality AUC 0.666 -> 0.572 macro, fatal/safe error ratio 0.843 -> 1.079 |
| The deterministic JEPA substitution | no | Flow loses the same at matched accuracy (0.0241 vs 0.0242 latent error) |
| Outcome gradients into the world | no, harmful | 0.604 with, 0.644 with the gradient stopped, matched 20k |
| Offline action coverage | no | logged-action fatality 0.497-0.502 across four worlds on 52 paired trajectories |
| Loss reweighting (whitening) | no | best cell 1.062 direction error at 20k, still worse than a constant; 0.895 at 5k decays away |

**What the failure is.** Under-dispersion, not a sign error and not silence. On
fatal transitions the true latent moves **+0.456** along the fitted direction; the
model predicts **+0.033 to +0.148**, 7% to 32%, with the sign right 66-85% of the
time. `fatal_failure_mode` is `tracks_true_direction` in 14 of 16 cells. An earlier
reading of the prediction as *inverted* is withdrawn: that was the starting-state
offset, since fatal states begin lower on this axis and the model barely moves.

The geometry is consistent with rarity rather than compression. The direction holds
0.000356 of latent variance (0.182x isotropic) in a latent of effective rank 45.8
of 512 -- but deaths excite it 7.91x, so the small variance share follows from 400
terminal events in 753k transitions, not from the encoder discarding it.

**Registered as method, not as a decision.** A scaling experiment on this quantity
must score conditional fatal-vs-safe predicted movement, or slope, and keep AUC
secondary: on the same eight cells the delta statistic has 95% intervals 0.040-0.093
wide and separates cells with disjoint intervals, while predicted-delta AUC
intervals are 0.227-0.284 wide and all eight overlap. Early checkpoints must be kept,
because the signal attenuates with training in three of four factorial cells.

**Closed 2026-08-14: it is not the data.** The question below was answered by S77
and S78. Support-v2 supplies 10,011 terminal episodes against v1's 400, and varying
unique terminals from 300 to 7,501 at fixed exposure leaves conditional fatal
movement flat-to-worse, saturating by 900. Over the same range a supervised probe
on the same corpus improves its action-conditional margin from 0.002 to 0.058. So
diversity makes consequence *more extractable* and the unsupervised latent
objective extracts none of it; the model still moves about 6-7% of the true fatal
delta. Data volume and coverage are no longer candidate explanations.

**Open, as originally posed.** Whether more unique terminal episodes recover the
magnitude. 320 TRAIN
terminals repeated about 62x at 20k exposure is the memorisation regime, and
terminal-enriched sampling is the one intervention with a clean positive effect
(0.069 [0.048, 0.093] against 0.033 [0.012, 0.056] at 20k). S77 removes RAM as a
constraint and S78 fixes the stopping rule before the larger result exists.

The cheap supervised identifiability control strengthens the coverage hypothesis
without proving it. A state+action MLP trained on whole-episode TRAIN logged data
scores 0.920 natural DEV AUC, but only 0.510 on executed policy actions and 0.521
within-state AUC across all-action forks. State-only is 0.489 within-state and the
state+within-label-shuffled-action control is 0.518: correct pairing adds no robust
conditional signal on the fork distribution. An action-only model trained on the
logged corpus is 0.510 on forks; the historical 0.927 action-only result was a
cross-validated oracle fitted on the fork label distribution itself, not evidence
that the offline corpus learned that marginal. Also open, and untouched by all of
the above: no Mamba arm has been through any of this, and no experiment has shown
that repairing fatality prediction would lift an actor above its BC prior.

## Verified defects, in fix order

Two independent audits; every entry below reproduced by execution before being
accepted. Nothing here reopens the architecture -- these are the implementation
failing to be what this document already says.

**Before any training run:**

| # | Defect | Evidence |
|---|---|---|
| 1 | Direct's transition loss is observation-free | Prediction bitwise identical for two different latent sequences. Fixed by S34 |
| 2 | Flow is diffusion forcing, not shortcut forcing | Eq. (7)'s bootstrap branch and its two half-step evaluations are absent, so the step token is inert and the arm is the paper's own ablation: Table 2 puts it at FVD 875 against 329 |
| 3 | `tanh` at inference but not in training | `_direct_candidate` squashes, `transition_loss` does not. A unit that learned 0.900 emits 0.716, decaying to 0.431 over six recursive steps toward a fixed point of 0 |
| 4 | Encoder window not enforced | `‖z(t=19 \| 20 frames) − z(t=19 \| 4)‖ = 1.7e-2`. The declared bound is documentation only |
| 5 | Phase 1A never writes the latent cache | `train_dynamics` fails on `batch.latents is None` |
| 6 | Multi-block decode against a cache is non-causal | New blocks attend bidirectionally; batched-vs-sequential differs by 9.2e-2. Attention corrupts silently, Mamba crashes -- divergent failure across the compared axis |
| 7 | Committed content and its label disagree | Mixes at 0.9 signal, labels bin 7/8 = 0.875 |
| 8 | No gate reaches `transition_loss` or `advance` | All six gates pass on the arm defeated by defect 1 |

**All fixed.** Plus, from the same pass: LPIPS at weight 0.2 with `p ~ U(0, 0.9)`
sampled per image; running-RMS loss balancing; the short/long batch curriculum;
seeded construction and device-matched generators; learned agent tokens;
diagnostics that commit before predicting.

Two things the fixes added rather than repaired. `transition.initial` commits a
known latent and returns the state it produces -- gates, diagnostics and Phase 3
all needed it, and inlining it three times is how the uncommitted-start defect
appeared in the first place. `data.unpatchify` is required by the perceptual
term. Both are in the inventory.

Still deferred: the Stage-B representation bodies, which raise rather than guess.
`expert.train_expert` is withdrawn -- see S49; there is no expert to retrain.

**Sweep of 2026-08-01.** Four more, each reproduced before being fixed:

1. **The configured batch could not run.** Phase 1A at `batch: 8` OOMs on the 6 GB
   card at every length; only batch 1 short fits. Every capacity claim to date came
   from a config that cannot execute. Fixed by S44 (checkpointing, `batch: 4`), not
   by shrinking the architecture. The first probe was itself wrong -- run in one
   process it reported batch 2 fitting where batch 1 OOMed, because memory is not
   released between trials; per-configuration subprocesses gave monotonic numbers.
2. **The two arms trained from different initialisations** -- 36 of 99 shared
   tensors, everything after the first time layer. Fixed by S45, measured 99/99.
3. **The dynamics loss saw only expert play**, because no mixture existed. Fixed by
   S43, verified bitwise: dynamics ignores relevant rows, BC ignores uniform rows.
4. **`_device` was still duplicated** across `train.py` and `gates.py` -- the exact
   duplication that caused an earlier defect when one copy was fixed and the other
   missed. Both were by then identical passthroughs, so both are deleted and call
   sites read `config.device`; the rationale moved onto the field itself.

Not a defect, and recorded so it is not re-raised: no gate runs in a deployment
dtype other than FP32 **because no other dtype exists** -- there is no autocast,
bf16 or fp16 path anywhere in the system. A dtype gate would be gating a path that
does not exist. It becomes real only if mixed precision is ever introduced.
