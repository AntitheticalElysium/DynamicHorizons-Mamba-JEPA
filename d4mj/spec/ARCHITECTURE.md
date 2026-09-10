# Dreamer 4 + JEPA + Mamba — Global Architecture Draft

The independent TC-LeWM–Mamba family now has M0–M3 implemented: jointly trained framewise encoder/world, functional completed-pair recurrence and its final state API. See [actual implementation and gates](lewm/STATUS.md) and [new-family architecture](lewm/ARCHITECTURE.md). Both families now use [shared infrastructure and runtime adapters](lewm/INTEGRATION.md). Its unbounded one-token latent and predictive state are separate from the legacy contracts below; M4/H16/actor/renderer remain blocked.

**Status:** System map for review, not a component specification.

This draft answers only:

1. what the complete system is;
2. where JEPA enters;
3. where Mamba enters;
4. which Dreamer 4 mechanisms remain around them.

Shapes, layer choices, exact masks, loss formulas, and temporal indexing belong
in the later component expansions.

## Labels

- `[D4]`: retained from the Dreamer 4 paper.
- `[D4-ENTAILED]`: not stated verbatim, but required for consistency between
  explicit Dreamer 4 statements.
- `[JEPA-R]`: JEPA change to visual representation learning.
- `[JEPA-D]`: JEPA change to action-conditioned world dynamics.
- `[MAMBA]`: Mamba change to temporal sequence processing.
- `[DESIGN]`: our integration choice, not a Dreamer 4 claim.
- `[D4-UNKNOWN]`: required boundary that the paper does not specify.

## One-sentence architecture

> Keep Dreamer 4's offline training phases, causal spatial world model,
> one-way task/agent interface, behavior cloning, reward learning, and
> imagination RL; learn its visual state with JEPA, predict the next state with
> action-conditioned JEPA, and replace temporal attention in the dynamics with
> Mamba.

`[DESIGN]` Assembly is not success: at matched active parameters, the combined
arm must match the paper-constrained anchor on paired executed control, and any
claimed JEPA or Mamba advantage must win its own preregistered metric.

Representation JEPA and dynamics JEPA are separate interventions:

- `[JEPA-R]` learns the observation latent \(z_t\);
- `[JEPA-D]` predicts \(z_{t+1}\) from world history and action;
- `[MAMBA]` carries the dynamics history used for that prediction.

EMA-target JEPA and LeVJEPA belong only to Phase 1A (`[JEPA-R]`). SIGReg is
LeVJEPA's regularizer, not a separate architecture. Once the encoder is frozen,
Phase 1B (`[JEPA-D]`) predicts fixed \(Z^*\) targets and uses none of their
training-only machinery.

The encoder state, latent, and dynamics memory are not the same object:

```text
e_t = causal tokenizer state/cache used while encoding real observations
z_t = backend-independent observation latent
m_t = dynamics temporal memory (Transformer cache or Mamba state)
h_t = one-way agent readout computed from the current world state
Z*  = C* ∘ E* = declared frozen latent function for one representation arm

imagined state       S_t      = (z_t, m_t)
real-execution state S_t^real = (e_t, z_t, m_t)

Observe(reset_t, e_{t-1}, m_{t-1}, a_{t-1}, o_t, optional q_t)
    -> S_t^real = (e_t, z_t, m_t), h_t

evaluate(S_t, led_to_action, latent, conditioning)
    -> latent_out, agent_out, S_out

Advance = one or more evaluate calls; the caller keeps exactly one S_out
```

One generic call covers every path, because the occupying latent and its
conditioning are per-call arguments rather than state:

| arm | path | `latent` | `conditioning` | writes state |
|---|---|---|---|---|
| flow | `Observe` | \(Z^*(o_t)\), τ_ctx-corrupted | τ_ctx bin | yes |
| flow | rung *i* | current candidate \(\tilde z\) | \((\tau_i, d)\) | no |
| flow | commit | \(\hat z\), τ_ctx-corrupted | τ_ctx bin | yes |
| direct | `Observe` | \(Z^*(o_t)\), clean | family vector | yes |
| direct | predict | *(no block — a head over block t's features)* | — | no |
| direct | commit | \(\hat z\) | family vector | yes |

The flow arm corrupts every committed latent because training never presents an
uncorrupted one (eq. 4's grid tops out at \(1-d\)). The direct arm has no noise
mechanism anywhere, so it commits clean latents and carries no signal bin — which
is the same deletion Box 5 records: it has no channel for "this context entry is
imperfect", and only generated-prefix training replaces it.

`Advance` = the arm's prediction step + exactly one commit evaluation. Flow
predicts with four read-only rungs; direct predicts with a head over the last
committed block's features, costing no block at all. \(h\) always comes from the
commit pass, and a candidate's `S_out` is always discarded, so the committed
memory only ever ingests a latent the model will actually condition on later.

`Observe` encodes and commits a real observation. \(m_t\) is the committed prefix
**through block \(t\) inclusive**, so \(z_t\) is carried alongside only for loss,
decoding and bookkeeping — no call may ingest it as a second temporal block.
\(h_t\) is produced before \(a_t\) becomes visible. `evaluate` may see \(a_t\)
and its outputs are *ephemeral* — a candidate block for \(t{+}1\) contains
\(a_t\), and \(h_{t+1}\) predicts \(a_{t+1}\), so there is no leak — but it never
mutates the prefix it was given, and rejected candidates' outputs are discarded.
A candidate's `S_out` is **always** discarded; \(m_{t+1}\) comes from the commit
evaluation alone, in every arm. The two halves of `S_out` carry different things:
`latent` is the accepted **clean** \(\hat z\), always in \(Z^*\) space, while
`memory` ingests whatever the commit block actually held — for flow, the
τ_ctx-corrupted copy. Losses, decoding and diagnostics read `latent`; nothing
reads the corrupted copy back out. Tasks remain agent-side only.

\(m_t\) spans **every** token slot, agent slots included: temporal mixing is
per-slot, so agent streams carry their own recurrent summary and world streams
can never read them. The firewall therefore needs no partition of \(m_t\) — it
is a property of per-slot time mixing plus the spatial mask. `[DESIGN]` Agent
slots are present **from Phase 1B**, placed last in the layout and masked out
both ways until Phase 2 activates them, so \(S\), the space mask shape, the
stream count and every state shape are fixed once for all phases and no Phase-1B
gate is invalidated later.

## Whole-system flow

Phase 1A and Phase 1B below are the tokenizer and dynamics subphases of Dreamer
4 Phase 1, not separate replacements for its three-phase lifecycle.

```text
PHASE 1A — REPRESENTATION

                         ┌─ D4 control: causal MAE encoder + decoder
offline video ──────────┤
                         └─ JEPA-R: causal encoder
                              ├─ V-JEPA-style masked EMA-target arm
                              └─ LeVJEPA-style global/local + SIGReg arm
                                      │
                                      ▼
                              frozen latent function Z*
                                      │
                                      ├─► canonical latent z
                                      └─► diagnostic decoder ─► pixels

PHASE 1B — WORLD MODEL

(z history, actions, temporal memory)
                  │
                  ▼
       D4 spatial/modal backbone
                  │
                  ▼
       Transformer time mixer  OR  Mamba time mixer
                  │
                  ├─► D4 shortcut-flow transition
                  └─► JEPA-D direct latent transition

PHASE 2 — AGENT ADAPTATION

world backbone + in-backbone agent/task modality
                  │
                  ├─► policy at state S_t ─► action a_t
                  └─► reward/continuation at resulting state S_t+1

The agent modality reads the world; the world never reads agent/task state.

PHASE 3 — IMAGINATION RL

dataset context ─► initialize S_t
                  │
                  ▼
policy ─► action ─► world transition ─► next state/reward/continuation
  ▲                                                       │
  └──────────────── imagined rollout ◄────────────────────┘
                              │
                              ▼
                  value learning + PMPO
                  + frozen BC policy prior

DEPLOYMENT

real observation ─► Z* ─► update S_t^real ─► agent policy ─► environment action
```

## Box 1 — Offline data and sequence construction

- **Input → operation → output:** Episodic videos, optional actions, rewards,
  optional tasks, and terminal facts → retain unshifted episodes and construct
  episode-safe causal sequences with the preceding action → world-pretraining
  and agent-adaptation batches. `[DESIGN]` Only true episode starts receive
  BOS; noninitial windows receive the real preceding action. `[DESIGN]` Storage
  and block arrays carry different indices and must be named differently
  (`action_taken[t]` vs `led_to_action[t]`); every offset statement names its
  array.
- **What stays from D4 — `[D4]`:** Offline video pretraining, optional
  action-labeled data, task/reward-labeled agent data, and alternating short and
  long training sequences. Appendix A gives \(C=192, T_1=64, T_2=256\) for
  Minecraft and \(C=96, T_1=32, T_2=128\) elsewhere, so the short batch is
  *shorter* than context; §3.4's "longer than the context length" describes the
  long batch, which the final finetune uses alone. `[D4]` §4 treats 30% of each batch
  as separate images to generate start frames; **not implemented** (S39), because
  every rollout here begins from a committed dataset context and that capability
  is never used.
  `[D4-ENTAILED]` Block \(t\) exposes \(a_{t-1}\), while
  \(h_t\) predicts the outgoing \(a_t\); exposing \(a_t\) in the same block
  would leak the behavior-cloning target.
- **`[DESIGN]` Windows never cross an episode boundary**, so `evaluate` needs no
  reset mask and a reset stays a fresh state construction. `gates.alignment`
  asserts it; if it is ever relaxed, the signature changes.
- **What JEPA changes — `[JEPA-R]` / `[JEPA-D]`:** Phase 1A uses masked
  prediction for EMA-JEPA-R or same-window global/local views for LeVJEPA-R.
  Phase 1B separately forms action-conditioned future-\(Z^*\) targets. Neither
  introduces external labels.
- **What Mamba changes — `[MAMBA]`:** The data do not change; sequence batches
  must carry enough boundary and prefix information to initialize \(m_t\).
- **`[D4]` Encoder prefix is *not* a Mamba concern.** §3.1 makes the tokenizer
  causal, so \(z_t = Z^*(o_{\le t})\) and a random crop's first frames are encoded
  with no history under **every** arm — the MAE-Flow-T anchor included. Burn-in,
  or a declared bounded encoder context, is a representation requirement binding
  all four Stage-A cells equally, which is why it is fair: they share one encoder.
  A frame-only encoder is a later ablation, not a design fork.
- **`[DESIGN]` Phase-2 support does not alter the §4.1 mixture.** The main batch
  remains half ordinary and half BC-eligible sequences. A separate tail-aligned
  terminal sequence supplies the rare positive class to continuation only; it
  cannot displace or enter dynamics, reward, or policy losses. Direct pairs its
  final alive/terminal labels across observed and generated readouts (S72/S76).
- **`[DESIGN]` Rare-outcome support is a versioned, sharded episode store.**
  Shards are memory-mapped and independently hashed; latent caches omit pixels
  and bind themselves to (C^*), the source manifest, and the cache code. Each
  rollout records its collection epsilon and a stable episode-level split.
  Support remains uniform-eligible and BC-ineligible. S77/S78 vary unique
  terminal trajectories at fixed tail exposure before this corpus can justify a
  production-data change.

## Box 2 — Visual representation system

- **Input → operation → output:** Causal video context → train one
  representation arm, declare its exported latent function, and freeze it
  before dynamics training → canonical \(z_t\), deployment function
  \(Z^*=C^*\!\circ E^*\), and optional post-hoc pixel diagnostics.
- **What stays from D4 — `[D4]`:** The paper-constrained control keeps the
  causal MAE tokenizer, \(L_\mathrm{MSE}+0.2L_\mathrm{LPIPS}\), `tanh`
  bottleneck, and encoder freeze before dynamics training.
- **What JEPA changes — `[JEPA-R]`:** Replace the defining pixel objective, not
  the exported latent interface. EMA-JEPA-R uses an online encoder and predictor
  against stopped targets from an EMA encoder. LeVJEPA-R uses one shared encoder,
  one global and multiple local spatial/photometric views of the same temporal
  interval, global/local MSE with gradients through both sides, and per-view
  projector-space SIGReg with
  \(\lambda=0.02\); it has no objective target, predictor, or stop-gradient. Its
  Polyak encoder is evaluation-only. Both are causal-D4 adaptations, not exact
  reproductions of their ViT backbones.
- **`[DESIGN]` LeVJEPA readout:** Append a one-way readout stream after the
  existing encoder slots and apply the loss to its final-time output through the
  paper's disposable projector. Latent/image slots cannot read that stream, and
  only the existing dense \(64\times16\) `tanh` latents enter \(Z^*\).
- **`[DESIGN]` LeVJEPA dropping:** Sample the paper's uniform random
  spatiotemporal keep-set, but preserve the encoder's fixed slot identities with
  true-position masks. Padded dropped slots are unreadable and unscored. This is
  semantically source-shaped but claims none of physical compaction's speedup;
  a compact implementation must first prove parity and recurrent-state identity.
  Token dropping, MAE replacement, and EMA masking are distinct mechanisms.
  JEPA decoders train only after the encoder freezes.
- **What Mamba changes — `[DESIGN]`:** Nothing in the primary thesis path.
  Keeping the encoder common isolates Mamba to dynamics; encoder-Mamba would
  be a separate experiment.

## Box 3 — Causal world backbone

- **Input → operation → output:** Latent history, actions, temporal memory, and
  optional in-backbone agent/task tokens → causal spatial/modal processing plus
  a temporal mixer → world features, updated memory, and agent readout \(h_t\).
- **What stays from D4 — `[D4]`:** All arms retain interleaved latent, action,
  and register tokens; separate space-only and time-only layers; temporal
  mixing every four layers (reproductions ship 2 and 1 — verify the anchor,
  do not inherit it), with `depth % time_every == 0` and at least two time layers
  so the Mamba substitution is not a single module; and the one-way agent
  firewall. Packing is D4's own arithmetic: \((N_b{=}512)\times(D_b{=}16)\)
  reshaped to \((N_z{=}256)\times32\), i.e. \(k=2\) (Appendix A).
  The Transformer anchor retains pre-RMSNorm, SwiGLU, RoPE and QKNorm — both
  have executable precedent in the pinned MMBench2 attention. `[DESIGN]` **GQA
  and attention-logit soft capping are dropped and registered**: Table 2 shows
  GQA moving FVD 70 → 71, adopted purely for KV-cache bandwidth at 2B parameters,
  and no pinned source implements either. The anchor is therefore
  *paper-constrained modulo declared scale-driven omissions*, not paper-faithful.
  Flow arms also retain the signal/step token.
- **What JEPA changes — `[JEPA-R]` / `[JEPA-D]`:** The input latent may come
  from \(Z^*\). Direct JEPA-D removes flow noise and signal-level/step
  conditioning; causality and the task firewall are untouched. `[DESIGN]` The
  conditioning *slot* is retained in every arm — flow puts its signal/step
  embedding there, Direct one reachable embedding. Deleting the slot would change block
  width, every later segment's spatial RoPE index, attention cost, stream count
  and state size at once, so slot-wise comparability and shared init across arms
  would be lost for a 1-of-\(S\) saving.
- **What Mamba changes — `[MAMBA]` / `[DESIGN]`:** Replace each dynamics
  time-attention sublayer with complete `Mamba2`, retaining its projections,
  causal convolution, SSD, gate/internal norm and D4's outer block, MLP, space
  layers, and cadence. One stream per fixed token slot preserves shape and
  causal isolation, not functional equivalence. Mamba time layers drop temporal
  RoPE, QKNorm, logit capping, and GQA; spatial attention keeps them. Fixed-size
  dynamics state and speed are measured hypotheses.

## Box 4 — Action-conditioned next-state model

- **Input → operation → output:** Current imagined state \(S_t=(z_t,m_t)\),
  chosen action \(a_t\), and optional noise → apply one transactional
  `Advance` → \(S_{t+1}=(z_{t+1},m_{t+1})\) and \(h_{t+1}\).
- **What stays from D4 — `[D4]`:** An autoregressive action-conditioned world
  model and stochastic shortcut flow with four denoising evaluations.
  `[D4-UNKNOWN]` The paper calls signal
  \(\tau_{\mathrm{ctx}}=0.1\) a slight corruption despite defining
  \(\tau=0\) as noise. Settled at 0.1 *noise* / 0.9 signal: the source states
  verbatim that "tau_ctx is the noise fraction". `[DESIGN]` Two constraints
  further pin it: the signal
  level is a discrete lookup, and eq. (4)'s grid tops out at \(1-d\), so
  \(\tau_{\mathrm{ctx}}\) **must be a trained grid bin**. That rules out the
  literal 0.1-signal reading and rules out MMBench2's uncorrupted path, which
  labels context with index `k_max` — a row its own sampler can never train.
  Our signal table therefore has exactly `k_max` rows.
  `[DESIGN]` The corruption is applied **once at commit**, not at every read:
  re-corrupting a prefix with fresh noise changes it every frame, which would
  invalidate any KV cache or SSM state and contradict invariant 5.
  `[DESIGN]` D4 states four denoising forwards but never says which latent
  condition supplies the persistent prefix and agent readout. Commit-time
  corruption settles it: the commit ingests a **fresh** corruption of the accepted
  \(\hat z\), whereas the final rung ingests the running Euler iterate. Those are
  different tensors even where their signal indices coincide, so the frame is
  **4 rungs + 1 commit = 5 backbone passes**. Direct is one commit plus a
  predictor head, so 1. That is a 5x *evaluation-count* ratio; whether it is a
  throughput ratio is for `diagnostics.cost` to measure.
  `[DESIGN]` At the final rung \(\tau = 1-d\), so the Euler step
  \(z \mathrel{+}= (\hat x_1 - z)\,d/(1-\tau)\) has coefficient exactly 1 and
  returns \(\hat x_1\) itself: under B, \(\hat z_{t+1}\) and \(h_{t+1}\) leave the
  *same* forward pass, which is self-consistent under rescan. The inconsistency
  appears only once \(m_t\) persists, because the stored features then come from
  that rung's noisy input while the stored latent is \(\hat z\).
- **What JEPA changes — `[JEPA-D]`:** The thesis branch directly predicts the
  frozen \(Z^*\) target. Its prediction readout must see \(a_t\) without
  exposing it to \(h_t\), so V-JEPA 2-AC's same-slot readout cannot be copied
  unchanged. `[DESIGN]` **An external predictor over committed world features**,
  \(\hat z_{t+1} = P(f_t, a_t)\), as V-JEPA 2-AC and DINO-WM do: one backbone pass
  per transition plus a head. `[DESIGN]` The validated Direct head pools spatial
  and register features to the latent grid, prepends the candidate action as a
  token, applies one pre-norm self-attention block, then projects each spatial
  token through `LayerNorm -> Linear -> tanh`. This is source-shaped action-token
  interaction, not a Dreamer 4 component or reproduction. The rejected in-block query cannot be trained at all
  in one pass — a position cannot hold the real latent for later blocks to attend
  to *and* the query to be predicted from — and filling every position with a
  query leaves the prediction a function of the action history alone. The head
  reads spatial and register features only; pooling the agent slot would route
  task state into world prediction against §3.3. Short generated-prefix training replaces flow's explicit
  corruption-conditioned robustness path; two steps are source-backed
  (`auto_steps: 2`). The branch is deterministic unless stochasticity is added.
  `[DESIGN]` \(Z^*\) stays the unnormalized `tanh` bottleneck for every arm, and
  the direct readout is `tanh`-bounded so its codomain matches its target's. The
  rejected alternative was V-JEPA 2-AC's `normalize_reps`, which runs its
  predictor entirely in a LayerNormed space; adopting it would make normalization
  part of \(Z^*\), and since LayerNorm discards per-token mean and scale a
  normalizing Direct arm and a non-normalizing Flow arm would not share a
  representation, failing Stage A's premise.
- **What Mamba changes — `[MAMBA]` / `[DESIGN]`:** `Advance` consumes and
  returns per-layer `(conv_state, ssm_state)` and obeys the read-only
  evaluate/one-commit transaction — stricter here than for a cache, since
  `step()` mutates state in place, so rungs must snapshot and restore where a
  cache merely declines to append. The commit contract is fixed and identical for
  both backends, so no execution count is attributed to one. Count every execution.

## Box 5 — Agent adaptation and heads

- **Input → operation → output:** Pretrained dynamics plus labeled
  action/reward, optional task, and boundary sequences → continue the dynamics
  objective while fitting the one-way agent readout and heads → adapted world
  model, BC policy, reward model, and continuation estimate; copy and freeze
  the BC policy at the Phase-3 boundary.
- **What stays from D4 — `[D4]`:** A distinct adaptation phase,
  agent tokens, continued world training, and one policy/reward head per MTP
  distance. Policy is categorical or vectorized-binary; reward is
  symexp-twohot; value starts in Phase 3. `[D4-UNKNOWN]` The source of \(c_t\)
  is unspecified. The closest pinned precedent is Dreamer 3's ordinary binary
  continuation likelihood. `[DESIGN]` Add environmental nontermination:
  terminal=0; truncation resets runtime but bootstraps. Continuation is one-step,
  as in the pinned Dreamer 3 precedent; D4's MTP statement applies only to policy
  and reward. The same likelihood is retained on the main mixture. One tail-aligned
  sequence joins four main sequences with equal per-sequence weight. Flow uses
  its ordinary sampled noisy pass. Direct scores the final alive and terminal
  positions on both teacher-forced and generated-prefix readouts, removing a
  path/label confound without entering the dynamics objective (S72/S75/S76). At \(h_t\), policy lead 0 is
  outgoing \(a_t\), reward lead 0 incoming \(r_t\). `[D4]` Adaptation reuses the
  pretraining setting, so flow-arm heads are deliberately trained on *noisy*
  representations across the sampled signal range — that is what makes them
  usable on generated latents at imagination time.
- **`[D4]` / `[DESIGN]` Task-conditioned successor:** D4 feeds one-hot task
  embeddings only to one-way agent tokens and conditions policy, sparse task
  reward, and value on them. We derive candidate tasks from recorded Craftax
  achievement events, admit only TRAIN-supported tasks, and freeze the prompt
  sequence before evaluation. The active aggregate-reward agent remains a
  reference; world prediction stays task-independent.
- **What JEPA changes — `[JEPA-R]` / `[JEPA-D]`:** Keep \(Z^*\) frozen and
  continue its transition objective. Heads stay interface-compatible;
  generated-prefix robustness is gated without changing target semantics.
  `[JEPA-D]` Deleting the signal level also deletes the channel that spread head
  training over imperfect latents, so each arm must declare the latent condition
  its heads are trained on and show it matches the condition they are read under
  in imagination.
- **What Mamba changes — `[MAMBA]`:** Head semantics stay fixed; scan and
  recurrent step must preserve world outputs, readouts, final states, and the
  firewall within registered tolerances.

## Box 6 — Imagination engine

- **Input → operation → output:** Dataset context, frozen \(Z^*\), world,
  reward and continuation models, current policy/value heads, and optional
  task → encode and scan the context to initialize \(S_t\), then alternate
  policy actions and `Advance` calls → imagined trajectories with states,
  actions, rewards, continuations, and values.
- **What stays from D4 — `[D4]`:** Start one rollout per diverse dataset
  context, sample policy actions and latent futures, and keep the world model
  frozen during the default imagination-RL phase.
- **What JEPA changes — `[JEPA-D]`:** Direct predicted latents replace flow
  samples in the thesis branch and are fed back recursively. The diagnostic
  decoder is not part of the control loop.
- **`[DESIGN]` Targeted and interactive use:** A task-conditioned imagination
  rollout holds one task \(q\) fixed while training its policy/value. A real
  prompt scheduler advances only from recorded environment achievements.
  Human or counterfactual play replaces the policy action source but calls the
  same recursive `Advance`; pixels are optional decoder diagnostics, never state.
  `[D4-UNKNOWN]` D4 does not state whether agent temporal memory resets when a
  prompt changes; that choice freezes before the task stage and never resets
  world streams.
- **What Mamba changes — `[MAMBA]`:** The context scan initializes every
  temporal-layer state pair. Each rollout owns its branch of that state, and
  `Advance` and `Observe` obey the same state-branch, commit, and reset
  semantics.

## Box 7 — Value and policy improvement

- **Input → operation → output:** Imagined trajectories and frozen BC-policy
  prior → compute returns, fit value, and improve policy → trained policy and
  value heads.
- **What stays from D4 — `[D4]`:** Symexp-twohot value learning,
  TD-\(\lambda\), sign-based PMPO, reverse KL to the frozen behavioral prior,
  and a frozen world/reward model. The paper fixes \(\gamma=0.997\),
  PMPO \(\alpha=0.5\), and prior weight \(\beta=0.3\), but not \(\lambda\).
  `[DESIGN]` \(G_t=r_{t+1}+\gamma c_{t+1}[(1-\lambda)v_{t+1}
  +\lambda G_{t+1}]\). `[D4-UNKNOWN]` Equation 10 instead prints same-index
  reward, continuation, and value; do not implement it literally before this
  discrepancy is resolved.
- **`[DESIGN]` Phase 3 is gated on outcomes, not logged-action calibration.** On
  held-out simulator states, all actions are executed from the same immutable
  state. Generated-successor reward choice and death probability must beat
  state-blind action marginals before the learned model may train an
  actor; the actor may not increase exact one-step death against its BC prior
  on those forks (S73).
- **What JEPA changes — `[JEPA-R]` / `[JEPA-D]`:** No RL algorithm change;
  executed control tests whether the learned representation and transition are
  sufficient.
- **What Mamba changes — `[MAMBA]`:** No RL objective change; policy and value
  consume Mamba-conditioned agent readouts without gradient updates to
  world-model parameters. Recurrent state still advances inside each rollout
  and is discarded or reset at its boundary.

## Box 8 — Real-environment execution

- **Input → operation → output:** Real observation \(o_t\), previous action
  \(a_{t-1}\), optional task, encoder state \(e_{t-1}\), dynamics memory
  \(m_{t-1}\), and reset signal → clear \(e,m\) on an actual reset, `Observe`,
  read \(h_t\), choose \(a_t\), and step the environment → next real
  transition and carried states.
- **What stays from D4 — `[D4]`:** Causal visual processing,
  task-conditioned agent readout, and direct low-level policy actions.
- **What JEPA changes — `[JEPA-R]` / `[JEPA-D]`:** Only the selected frozen
  deployment function \(Z^*\) is active on real observations; training-only
  predictors, projectors, SIGReg, representation views, transition losses, and
  the decoder are absent. `[DESIGN]` One
  training-time mechanism does survive in the flow arm: it commits
  τ_ctx-corrupted real latents at deployment too, so executed control carries a
  third randomness source beyond environment seed and policy sampling, and the
  two arms deploy under different perception conditions. The executed-control
  metric must control that draw and say so.
- **What Mamba changes — `[MAMBA]`:** Carry and reset Mamba memory under the
  same declared episode semantics used for training and imagination. Resetting
  runtime state remains separate from deciding whether a time-limit transition
  bootstraps its value target.

## Global invariants

1. `[DESIGN]` Each arm exports exactly one \(Z^*=C^*\!\circ E^*\), where
   \(C^*\) fixes copy, output, normalization/bottleneck, and packing. Real
   latents, targets, and diagnostics share this \(z\)-space; \(e_t,m_t\) do not.
   Because the tokenizer is causal, \(z_t=Z^*(o_{\le t})\), so \(C^*\) must also
   fix the causal mask, maximum encoder context, reset policy and prefix/burn-in
   rule: the same frame encoded from a bare crop, from a burn-in prefix, or from
   a whole episode yields three different latents under identical weights. Every
   cached target, JEPA target, diagnostic and deployment latent uses that one
   contract, and no target encoder may see frames unavailable to the deployed
   encoder. \(Z^*\) is defined with **all training corruption off**: MAE
   replacement probability 0, EMA masking off, and LeVJEPA token dropping off.
   It is dense and deterministic under the declared exported copy.
   Representation arms that differ in \(C^*\)'s geometry change objective *and*
   latent space at once; either hold geometry fixed across arms or label the
   comparison compound.
2. `[D4]` Agent/task state reaches the world only through the selected action.
   `[DESIGN]` Active aggregate-task runs instantiate no task projection. The
   successor conditions only agent-side policy, sparse reward, and value paths
   on an eligible one-hot achievement task; world latents and transition outputs
   must be invariant to \(q\). Continuation remains a physical, task-independent
   prediction; its neutral readout is fixed before that stage.
3. `[DESIGN]` Policy reads the pre-action state; reward/continuation describe
   its result. Candidate branches are read-only, one edge commits, and every
   actual reset clears \(e_t,m_t\) regardless of bootstrap.
4. `[DESIGN]` EMA-JEPA-R and LeVJEPA-R share one deployed interface, while their
   EMA copies have different roles; Mamba changes dynamics time mixing only;
   JEPA-R, JEPA-D, and Mamba remain separately measurable.
   `[D4]` Concurrent losses use running-RMS normalization, with new composite
   objectives declared explicitly.
   Phase 3 additionally requires the trained reward/continuation model to pass
   the real all-actions fork gate; finite losses and logged-action calibration
   are not substitutes for counterfactual outcome support.
5. `[DESIGN]` The Transformer anchor carries a persistent dynamics KV cache
   across accepted frames; full-prefix rescanning is not an efficiency control.
   §3.4 cites "the memory bandwidth needed to access the KV cache of a long
   context" and Table 1 reports 21 FPS at a 192-frame context, which persistence
   would explain — but **no audited source does it**: MMBench2 re-draws the
   context noise every frame and re-prefills, so its cache lives inside one
   frame. This is therefore our deviation, and it is the one the `[MAMBA]`
   efficiency hypothesis rests on: read-time corruption re-randomises the prefix,
   which would force an O(t) rebuild per frame in *both* backends and evaporate
   the claim. Commit-time corruption (S11) is also the more training-faithful
   choice, since training draws one noise realisation per frame and processes the
   sequence once. `[DESIGN]` Report \(e_t\) and \(m_t\) separately: Mamba fixes only
   \(m_t\); \(e_t\) scales with encoder context. Match deployed parameters and
   separately report training-only parameters, FLOPs, memory, throughput, and
   state size.

## Staged attribution plan

Stage A uses one frozen D4-MAE encoder for a complete dynamics \(2\times2\):

| Arm | Representation | Transition | Time mixer | Question |
|---|---|---|---|---|
| MAE-Flow-T | D4 MAE | D4 flow | Transformer | Paper-constrained anchor |
| MAE-Flow-M | Same frozen MAE | D4 flow | Mamba | Mamba under flow |
| MAE-Direct-T | Same frozen MAE | Direct JEPA-D | Transformer | Direct vs flow under T |
| MAE-Direct-M | Same frozen MAE | Direct JEPA-D | Mamba | Completes the factorial |

Stage A estimates Mamba under each transition family, direct versus flow under
each mixer, and their interaction; flow versus direct moves multiple mechanisms
and does not isolate stochasticity by itself. One of those mechanisms is the
head-input distribution: flow heads are fit on τ-grid latents while direct heads
see clean or generated ones, so "same frozen MAE" does not make the cells differ
only in transition and mixer.

Before Mamba training, gate scan/step parity for outputs and states, selective
reset parity, firewall counterfactuals, branch nonmutation, and actual recurrent
carry in FP32 and deployment dtype.

Stage B holds the valid Stage-A transition and Mamba mixer fixed and compares
`MAE`, `EMA-JEPA-R`, and `LeVJEPA-R` exports. The two JEPA-R arms are independent;
LeVJEPA is not gated on EMA. If Direct is invalid but Flow is valid, the clean
representation comparison runs under Flow. Each export must first pass the
dense-retention gate; otherwise no world-model result is interpreted.

Stage C compares Direct and Flow within each retained frozen representation. It
measures the transition-family system effect, not stochasticity alone. If Direct
never passes, the surviving model is JEPA-R + Flow + Mamba and makes no JEPA-D
success claim.

Stage D adds D4-style achievement task conditioning only to the selected
representation/transition/mixer. It keeps the aggregate agent as a reference,
freezes the eligible task set and prompt from TRAIN support, and reports both
per-task success and end-to-end prompt progress. Each imagined policy is gated
against its own frozen task-conditioned BC prior; the aggregate agent is a
separate system reference. Interactive imagination uses that same frozen world
and cannot claim a horizon beyond trained recursive rollouts.

Each stage passes source-fidelity, recurrence, parameter-match, and executed-
control gates before the next stage. Matched arms share data/splits and
phase-reseeded construction with initialization digests. Budget every matched
cell plus BC, actor, and executed evaluation; never start a partial factorial.

## Decisions

`spec/DECISIONS.md` is the single source of truth for what is settled and what is
open, and carries the function inventory. Nothing is reserved here; if this file
and that one disagree, that one wins and this one is stale.

## Source boundary

- Dreamer 4: `third_party/papers/2509.24527v1.pdf`, Algorithm 1, Figure 2,
  Sections 3.1–3.4, and Equations 5–11.
- EMA representation JEPA and action-conditioned latent prediction:
  `third_party/papers/2506.09985v1.pdf`, Sections 2.1 and 3.1, Equations 1–4;
  `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812`.
- SIGReg: `third_party/papers/2511.08544v2-lejepa.pdf`, Sections 4–5.1 and
  Algorithms 1–2;
  `rbalestr-lab/lejepa@c293d291ca87cd4fddee9d3fffe4e914c7272052`.
- LeVJEPA video recipe: `third_party/papers/2608.27395v1-levjepa.pdf`, Section 3
  and Appendices A–B;
  `MLO-lab/LeVJEPA@941526c428aa513a5bdfa38697724fda794d8496`.
- EMA momentum schedule and target-encoder-for-evaluation precedent:
  `third_party/papers/2301.08243v3.pdf`;
  `facebookresearch/ijepa@52c1ae95d05f743e000e8f10a1f3a79b10cff048`.
- Mamba-2: `third_party/papers/2405.21060v1.pdf`;
  `state-spaces/mamba@f577286d052741c35d39cd43bdc3fad27120f22c`.

No official Dreamer 4 implementation is available in the audited source set.
`nicklashansen/dreamer4@b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6`
and
`edwhu/dreamer4-jax@8144b940d801971f12ec5633553b95001e555949`
are corroboration only, never canonical evidence.
`nicklashansen/mmbench2@3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`
corroborates one temporal stream per D4 token slot. Replacing those attention
streams with official Mamba-2 is our integration; MMBench2 defines neither
Mamba nor canonical Dreamer 4.
`danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe`
is implementation precedent only for return and boundary semantics.
