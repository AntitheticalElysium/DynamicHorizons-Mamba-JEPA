# Architecture

> Scope: this is the legacy `d4_mamba_jepa` implementation ledger. The active
> successor design — MAE/EMA-JEPA-R/LeVJEPA-R, Direct/Flow, persistent T/M state,
> and achievement-conditioned imagination — lives in
> [`d4mj/spec/ARCHITECTURE.md`](../../d4mj/spec/ARCHITECTURE.md), with its
> decisions and gates alongside it. Planned components are not backported here
> until this package implements them.

What the system IS, right now. Present tense, overwritten in place, no history.
One block per component in dataflow order. Shapes are for the live
`craftax_jepa_config()`.

- `diff` — departure from the pinned source. Omitted means faithful.
- `inherit` — where each knob's VALUE came from. Omitted means `craftax`.
  - `craftax` chosen or measured on Craftax
  - `cartpole` selected on CartPole (2 actions, 500-step cap) and carried over
  - `default` `D4LiteConfig` default; never selected on any task
  - `source` fixed by a pinned paper or repo
- `tested` — a Craftax run has since measured this knob. `ABLATIONS.md` row.
  Origin, tested and selected are three different questions: `inherit` answers
  only the first, and no `tested` row has yet CHANGED a live value.

`cartpole` and `default` mark where a value came from, not that it is unmeasured:
every such value was set against a task with two actions and three informative
scalars, and the `tested` rows say which have since been probed on Craftax.

Rule: if a line would need a date, a result, or the word "corrected", it belongs
in `ABLATIONS.md`, not here.

---

### 1. Environment — `craftax_env.py`
out    `[3,64,64]` uint8 CHW
knobs  Craftax-Classic pixels · native 63x63 (9x9 tiles at 7px) zero-padded to 64 · 17 actions · 22 achievements
src    `craftax` 1.6.1; `game_logic.py`/`constants.py`/`renderer.py` digest-pinned in `source.py`. No source checkout: pinned by installed version + file digests
diff   `continues` uses Crafter's `1 - dead`: timeout truncation keeps `continues=1`. `dead` is read from the state by `is_dead` (lava OR `player_health<=0`, the two absorbing disjuncts of `game_logic.py:is_game_over`), not inferred as `done and not timeout` — the two disagree when a death lands on the native 10,000-step horizon

### 2. Replay — `expert/generate.py`, `data.py`
out    episode-bounded `EpisodeReplay`
knobs  320 episodes / 696,746 transitions · PPO expert · 80/10/10 whole-episode split, seed 20260727 (computed by the runner; not stored in the artifact)
src    `Craftax_Baselines@7ce36fa` `ppo_rnn.py` — **NOT vendored**: absent from `third_party/sources/` and `SOURCES.lock`, so `expert/ppo_expert.py` cannot be source-diffed. It is a re-implementation (distrax→native categorical, chex→`flax.struct`, orbax→`flax.serialization`, wandb/logz dropped, target env `Craftax-Classic-Symbolic-v1`)
diff   episodes capped at `max_steps=2500`; 252/320 hit that cap, so only 68 episodes end absorbing (58 after the train split)
diff   the artifact pins the replay and policy bytes by digest but records no PPO training lineage: no budget, optimizer schedule, minibatching, layer size, `DeathPenaltyWrapper`/`DEATH_PENALTY` setting, or trainer digest
diff   post-`done` slots are not masked out of the vectorized scan — they keep stepping and are discarded by first-`done` slicing. `truncated_episodes` counts slots that never reached `done` inside the local cap; `timed_out_episodes` counts native-horizon endings (disjoint)

### 3. Sampler — `common.py:sample_sequences`
out    `SequenceBatch`: `[B,T,3,64,64]` + led-to actions/rewards/continues/valid
knobs  `sequence_length=16` · `terminal_fraction=0.5` · batch 8
inherit  code CartPole-era but env-agnostic · `sequence_length` default (CartPole ran 12) · `terminal_fraction` cartpole · batch cartpole
tested `terminal_fraction` rows 12, 13, 15
diff   `round(B*fraction)` rows are forced to the LAST window of a terminal episode; the remainder is episode-uniform, not transition-uniform
diff   the forced-terminal support is 58 windows (see §2), so 4 of every 8 rows are drawn from 58 fixed 16-frame windows: ~1,380 repeats each over 20,000 updates

### 4. Encoder — `model.py:build_tokenizer`, called at `model.py:243`
in     `[B,16,3,64,64]` uint8 -> `temporal_patchify(8)` -> `[B,16,64,192]`
out    bottleneck `[B,16,16,16]`, tanh
knobs  patch 8 (64 patches) · d_model 64 · depth 4 · heads 4 · time_every 2 · n_latents 16 · d_bottleneck 16 · **MAE OFF**
src    MMBench2 `Encoder` class, unmodified — but its TRAINING regime is local: no MAE, no decoder, no upstream tokenizer pretraining
inherit  every knob `default` — no encoder geometry has ever been selected for Craftax
tested `d_bottleneck` row 4 · `n_latents` row 5 · encoder LR rows 16, 17 (none changed the live value)
diff   `training_mask=False` forces `mae_p_min=mae_p_max=0`, so `MAEReplacer` takes its no-op path; the config's `mae_p_max=0.9` is unused by the world, and `encoder.mae.mask_token` (64 params) never receives gradient
diff   trained from random init at full LR unless `encoder_learning_rate` is passed; Dreamer 4 and V-JEPA 2-AC freeze a pretrained encoder, Dreamer-CDP uses `enc_lr 6e-6` vs `dyn_lr 4e-4` (`configs.yaml:88-89`)
diff   `n_latents=16` vs the Dreamer 4 PAPER's `N_b=512` (`appendix.txt:14`); the cited JAX code defaults to `enc_n_latents=16` (`train_policy.py:109`), which we match. `d_bottleneck=16` matches the paper's `D_b=16` and differs from that code's 32

### 5. Packing — `model.py:encode_frames`
in     `[B,16,16,16]`
out    `[B,16,4,64]` (`n_spatial=4` x `d_spatial=64`) = 256 floats/frame
knobs  `packing_factor=4`
src    MMBench2 `pack_bottleneck_to_spatial`, unmodified
inherit  default

### 6. Dynamics — `model.py:forward_dynamics` (MMBench2 `Dynamics`)
in     packed `[B,T,4,64]` + led-to actions
out    tokens, agent tokens `[B,T,2,64]`
knobs  d_model 64 · depth 4 · heads 4 · time_every 2 (2 temporal modules) · n_register 2 · n_agent 2 · k_max 4
src    MMBench2 `Dynamics`, unmodified class
inherit  every knob `default`
diff   `DiscreteActionEncoder` replaces the continuous 16-D action MLP in the same token slot. It holds **18** embeddings, not 17: ids `0..16` plus a dedicated slot for the `-1` start/unlabelled action
diff   built with `lang_dim=0`, so upstream's `task_proj` is `None` and the agent tokens initialize to **zeros** on every forward. The task-conditioning pathway those tokens exist for upstream is disabled
diff   `flow_x_head` (zero-initialized upstream, 4,160 params) receives NO gradient in this arm: nothing consumes the spatial output, because `jepa_predictor_context="pooled_agent"` discards it. Its output is identically zero at every step
diff   the shortcut conditioning is degenerate: every JEPA path passes constant `step=max_step_index` and `signal=k_max`, so 1 of 3 `step_embed` rows and 1 of 5 `signal_embed` rows are ever indexed. The two live rows act as a constant bias; the other six receive no loss gradient (they still move under AdamW weight decay, being rows of a densely-grad tensor)

### 7. Temporal operator — `temporal.py:replace_dynamics_time_attention`
knobs  T arm: upstream `TimeSelfAttention` · M arm: official `Mamba2`, `d_state=64 · headdim=64 · expand=1 · d_conv=4`
src    `state-spaces/mamba`, digest-pinned
inherit  `d_state`/`headdim` cartpole (selected there over a parameter-matched 16/32) · `expand`/`d_conv` default
diff   the single research axis; the swap runs LAST in `D4LiteWorld.__init__`, after `_build_jepa`, so both arms draw identical shared init (verified: 227 shared state-dict tensors, all bit-identical at equal seed)
diff   `d_inner = d_model*expand = 64` with `headdim=64` gives the M arm exactly one head
diff   arms are not parameter-matched: 21,571 params per Mamba temporal module vs 16,644 per attention module (+29.6%), 43,142 vs 33,288 total; worlds 996,202 vs 986,348. `config.py`'s parameter-matching comment describes the REJECTED 16/32 defaults, not this
diff   `use_mem_eff_path=False`; official Mamba-2 defaults it to `True` (`mamba2.py:59`). Still unmodified official code — only the kernel path is selected locally, so numerics differ from the fused path
diff   the M arm's `dt_bias`/`A_log`/`D` are marked `_no_weight_decay=True` upstream (`mamba2.py:130,136,140`). They are now placed in a `weight_decay=0.0` group; a T world has no such tensor, so its optimizer is unchanged. Runs before this fix decayed them, i.e. applied a hidden second difference to one arm of the research axis
diff   `Mamba2.step`/`allocate_inference_cache` and the whole `MambaTemporalState` path are unreachable from `craftax_jepa_config()`: only the generative arms consume a cache (see §13)

### 8. Predictor — `model.py:CDPPredictor`
in     mean over 2 agent tokens (64) + next action token (64)
out    `[B,1,4,64]` — **this is the entire rollout**; no denoiser, decoder, or MAE
knobs  `jepa_predictor_context="pooled_agent"` · `hidden_ratio=1.0`
src    shaped after Dreamer-CDP's predictor
inherit  default
tested predictor context row 14 (`spatial_agent` rejected)
diff   Dreamer-CDP predicts from an 8192-d deterministic RSSM state (`configs.yaml:93`, consumed at `rssm.py:140`); ours is a pooled 64-d channel
diff   Dreamer-CDP's action conditioning has already entered the RSSM core (`rssm.py:84-100`); ours concatenates an explicit next-action token
diff   Dreamer-CDP's `dyn_deter` cosine is a SINGLE GLOBAL cosine per timestep — `cosine_distance(..., axis=-1)` over the flattened encoder output — not a per-spatial-token cosine. Our global scoring in §9 is therefore faithful to CDP on this point, not a departure from it
diff   the predictor's output is an unbounded `nn.Linear`, while every real latent is `tanh`-bounded by the encoder. Predicted latents leave `(-1,1)` and carry ~60% of the real scale, and are fed straight back into `spatial_proj`, which only ever saw bounded input

### 9. Self-prediction loss — `objectives.py:jepa_self_prediction_loss`
in     online: predictor rolled autoregressively K steps · target: EMA target encoder on the real future frame
out    scalar
knobs  `jepa_jumps=5` · `jepa_projection_dim=64` · EMA tau 0.99 -> 0.999 linear over world steps · `jepa_anticollapse="ema"`
src    `mila-iqia/spr` `spr_loss` (`src/models.py:287-293`: `F.normalize(p=2,eps=1e-3)` then summed MSE); ramp FORM from I-JEPA `src/train.py:228`
inherit  `jepa_jumps` cartpole (ladder 5/8/11 run there) · `projection_dim` default · EMA endpoints default
tested `sigreg` vs `ema` row 18 (sigreg worse; `ema` retained)
diff   scores ONE global vector per frame: `[B,K,4,64]` is flattened to `[B,K,256]`, projected to 64, normalized once. SPR's per-position branch (`local_spr_loss`) is shipped but off by default (`scripts/run.py:106`); I-JEPA (`train.py:297,311`: LayerNorm + Smooth-L1) and V-JEPA 2 (`app/vjepa/train.py:447`: per-token L1) score per token and have no global variant. Note §8: Dreamer-CDP, the source of the predictor, IS global
diff   **no t0 term.** SPR with `jumps=K` scores K+1 positions: `models.py:449` appends the current frame's own latent before the jump loop and `algos.py:296-298` splits it off as `t0_spr_loss` (weight 1.0). We score K transitioned positions only. SPR's t0 term is augmentation-invariance, and the augmentation is not ported, so it would degenerate — but it removes half of SPR's anti-collapse pressure
diff   `JepaProjector` is SPR's OPTIONAL `mlp` branch, at a different width. SPR's MLP branches use hidden = 2x out (`models.py:210`: 512/256; `:239`: 2s/s); ours uses hidden = out. SPR's shipped defaults are `--classifier q_l1` (a `QL1Head` off the Q head) with `--final-classifier linear`, i.e. a single `nn.Linear` prediction head
diff   EMA ramp endpoints are local (I-JEPA publishes `[0.996, 1.0]`). SPR equivalence is NOT source-verified: its `update_state_dict` comes from `rlpyt.models.utils` (`models.py:5`), which is neither vendored nor installed
diff   SPR's visual augmentation is not ported
diff   SPR runs self-prediction as an auxiliary beside Q-learning and reward (`algos.py:131-136`); here it is the primary encoder signal
diff   `sigreg` alternative exists (LeJEPA `SlicingUnivariateTest`+`EppsPulley`, digest-pinned and loaded in isolation so executed == verified) and is not the default. Its surrounding use — SIGReg on action-predicted projected tokens — is a local integration, not LeJEPA's multi-view training algorithm

### 10. Loss composition — `training.py:_jepa_world_loss`
out    `jepa_weight*jepa + reward*reward_n + continuation*continuation_n`
knobs  all weights 1.0 · AdamW lr 1e-4, wd 1e-2 · clip 1.0 · warmup 1,000 (linear, constant after — no decay) · 20,000 updates · batch 8 · optional `encoder_learning_rate` splits the encoder into its own param group
inherit  every optimizer knob and the whole budget cartpole
diff   `WorldLossNormalizer` registers EmaRms for flow/reward/continuation/cdp/reconstruction — there is no `"jepa"` term, so the JEPA loss is unnormalized while the other two are
diff   parameters marked `_no_weight_decay` by their source are given a `weight_decay=0.0` group (see §7); this affects the M arm only

### 11. Task heads — `model.py:forward_task_heads`
in     post-transition agent tokens of the IMAGINED rollout (`return_rollout_agents`)
out    reward logits + centers, continue logits
knobs  `reward_horizon=8` · `reward_bins=255` · symlog `[-10,10]` · `continuation_horizon=8` · `jepa_terminal_weight=8`
src    reward head is MMBench2 `RewardHeadMTP`, unmodified. **Continuation is LOCAL** — the pinned `src/model.py` has no continuation head at all. `bins=255` from DreamerV3 `configs.yaml:98` (reward head; `:101` is its value head, same value)
inherit  horizons and bins default/source · symlog range default · `terminal_weight` cartpole
diff   the two heads pool differently: `RewardHeadMTP` uses a learned attention pool (`pool_agent="attn"`), `ContinuationHeadMTP` a plain mean
diff   heads receive `jepa_jumps` positions per batch, not `sequence_length`, so with `jepa_jumps=5` and `reward/continuation_horizon=8` the MTP lead slots taper 5:4:3:2:1 and **leads 5-7 get no loss gradient at all**. The effective trained horizon is 5, not 8; deployment reads lead 0 only. The untrained rows still move under AdamW weight decay. The base/CDP arms feed all `T=16` positions and do train all eight leads

### 12. BC policy — `common.py:BCPolicy`
in     agent tokens (world frozen)
out    17-way categorical
knobs  3,000 updates · batch 16 · lr 1e-4 · AdamW wd 1e-2 · clip 1.0 · warmup 250 · cross-entropy of positions `[:-1]` against led-to actions `[1:]`
src    MMBench2 `PolicyHeadMTP` — attention pooling, MLP shape, small output init, head-only gradient
inherit  code CartPole-era but env-agnostic · every budget cartpole
diff   upstream emits `L x act_dim_max` tanh-squashed continuous action means; ours is a SINGLE-distance 17-way categorical. The MTP structure the source head exists for is dropped
diff   the Dreamer 4 reproduction optimizes `{"dyn","task","pi","rew"}` jointly under a combined shortcut+policy+reward loss (`train_bc_rew_heads.py:791-799`); we freeze the whole world and train the head alone

### 13. Imagination — `imagination_actor_critic.py`
in     replay contexts, frozen world/heads/BC prior
out    actor (`BCPolicy`) + `ValueHead`
knobs  500 updates · batch 64 · `context=8` · `horizon=32` · gamma 0.997 · lambda 0.95 · alpha 0.5 · beta 0.3 · value support 255 bins over `[-10,10]`
src    Dreamer 4 eqs 10-11: actor init from BC, PMPO + reverse `KL(actor||BC)`, TD-lambda symexp-twohot value, one rollout per context
inherit  gamma/lambda/alpha/beta source · updates, batch, context, horizon all cartpole (the 500 was selected on CartPole dev seeds)
diff   `imagine_trajectory` re-slices `[:, -context:]` each step and `_sample_next_jepa` neither accepts nor returns a `MambaTemporalState`, so every dynamics pass re-scans the 8-state window from a fresh zero SSM state. The precise defect is the loss of the EXPLICIT recurrent state, not of all long-range information: each generated latent is a function of the window that produced it, so older history survives lossily by recursive summarization. `sample_next_packed` is still called with `use_cache=True`; the JEPA branch ignores that argument
diff   **the rollout leaves the trained support after 8 of 32 steps.** Training rolls `K=5` steps from an 11-state all-real context (max 4/15 = 27% synthetic, real window start always in view); deployment pins the window at 8, so from step 7 it is 100% synthetic and stays so for 24 more steps
diff   the sliding window itself is NOT a departure from the cited source: `edwhu/dreamer4-jax` `dreamer/imagination.py:409-415` concatenates and re-slices `[:, -context_length:]` identically. The departures are the absent recurrent cache and the context length
diff   context 8 vs Dreamer 4's `C=192` for Minecraft (`appendix.txt:15`) but `C=96` for SOAR and Epic Kitchens (`:37,:45`); the cited JAX code defaults to `context_length=16` (`train_policy.py:147`). No task tokens, no `tau_ctx` corruption (the paper specifies signal level 0.1)
diff   value support 255 bins over `[-10,10]` vs the JAX code's `num_reward_bins=num_value_bins=101` (`train_policy.py:136,139`)
diff   TD-lambda carries the predicted continuation factor per Dreamer 4 eq 10; the inspected JAX runner omits continuation from its recursion

### 14. Executed evaluation — `craftax_achievement.py`
knobs  frozen policy in live Craftax · sampled at temperature 1 · official Crafter geometric-mean score (matches `danijar/crafter` `analysis/common.py:47-55`) · paired seed bootstrap · `context=8`
diff   seeds are exploratory, not a sealed tier. The policy RNG seed is a deterministic function of the environment seed, so the bootstrap resamples fixed joint environment/policy-seed OUTCOMES: it excludes training-seed variance and alternative policy-sampling schedules, and does not isolate environment variance either
diff   evaluation reads the agent token at the end of a sliding 8-frame window. Causal BC training on a 16-frame batch does train context lengths 1-15, including 8, but weights them uniformly; execution uses length 8 for every post-warm-up step

### 15. Oracle — `craftax_oracle.py` (instrument, not architecture)
knobs  per-target probes on frozen latents vs constant / timestep / raw-pixel references · episode-level splits · self-audit on perfect/constant/misaligned/timestep-shifted inputs
diff   nonlinear ceilings are asymmetric by design: a CNN over raw frames for the pixel reference, an MLP for the low-dimensional latent
diff   `preserved` means "as recoverable as from raw pixels" — an absolute standard, not a necessary condition for control. Not a gate
diff   `achievement_group` has no constant/timestep/pixel reference
diff   episode-bootstrap CIs cover the latent LINEAR R² only; the nonlinear latent/pixel estimates and every achievement metric (AUROC/AP/Brier) are reported without intervals

### 16. Provenance — `source.py`, `checkpoint.py`
knobs  digest-pinned sources recorded into every checkpoint; strict atomic saves with full RNG
diff   `source_report(cfg)` is config-conditional: MMBench2 and Craftax always, Mamba-2 only for the M arm, LeJEPA only under `sigreg`. `source_report()` with no argument keeps the legacy MMBench2/Mamba-2/CartPole triple. Load re-verifies exactly the sources a checkpoint RECORDED, so older checkpoints stay loadable and a T-arm world no longer requires an installed Mamba
diff   still omits SPR, I-JEPA, V-JEPA 2, Dreamer-CDP, Dreamer 4 and DreamerV3 — those are read references, not imported code
diff   `implementation_sha256()` hashes a fixed allowlist that excludes `craftax_env.py`, `craftax_data.py`, `craftax_achievement.py`, `craftax_oracle.py`, `oracle_metrics.py`, `craftax_resolution.py`, `executed_control.py` and all of `expert/`; those can change without invalidating a world checkpoint
diff   RNG capture is narrower than "every checkpoint": the world writer stores torch CPU + CUDA + the supplied NumPy generator ONLY when an optimizer is passed; the tokenizer writer stores no RNG at all; the BC, actor and value heads store format + config + paired world digest, with no RNG and no source report
diff   provenance blocks carry `sources_schema`. A schema-2 block must cover `source_names_for(config)` at minimum; a block with no schema must be exactly the historical MMBench2/Mamba-2/CartPole triple. Without that split, "verify what was recorded" would accept an empty block
diff   `POLICY_FORMAT` and `imagination_actor_critic.FORMAT`/`EVALUATION_FORMAT` still carry `cartpole`; the world, tokenizer, run, replay and value formats do not. They are serialized identifiers compared on load, so renaming them would invalidate existing checkpoints

---

## Inherited-value summary

Nothing in components 4-13 was SELECTED on Craftax — no Craftax result has yet
replaced a live value. Encoder geometry, packing, dynamics width and the
predictor are `D4LiteConfig` defaults never chosen for any task; the sampler,
optimizer, budgets, `jepa_jumps`, `terminal_weight`, Mamba state expansion and
the whole actor/value schedule were selected on CartPole. Craftax-native choices
exist only at the boundaries: the environment, the replay, the executed
evaluation and the oracle.

Selected-on-Craftax is not the same as untested-on-Craftax. The `tested` lines
above name the knobs that now have Craftax rows — `d_bottleneck`, `n_latents`,
`terminal_fraction`, predictor context, encoder LR, `sigreg` vs `ema`. Each
returned "no effect" or "rejected", which is why the live value is unchanged.

## Transition convention

`(obs_t, action_t) -> (obs_{t+1}, reward_{t+1}, continue_{t+1})`. In the
block-causal sequence the action stored at state position `t+1` is `action_t`,
the action that led to that state. Head distance 0 predicts the transition that
led to the current position.

Position 0 gets `-1` only when the window begins at a true episode start;
a window sampled from mid-episode gets the real preceding action
(`data.py:98-101`, `common.py:117-120`). Either way `outcome_valid[:, 0]` is
`False`, because the outcome of that transition lies outside the window.

## Not reproductions

Dreamer-CDP, DRAMA and NE-Dreamer are sources we borrow components from, not
architectures we reproduce. Scoped to the live Craftax JEPA world objective and
the live JEPA imagination path: no RSSM latent, no online joint world+control
learning, no reconstruction or RSSM-KL term, and no recurrent cache in the
rollout.

That scope is load-bearing. The live actor does use a KL — reverse
`KL(actor||BC)` in §13 — so "KL terms absent" is false of the system as a whole.
Transformer KV-cache and Mamba recurrent-cache code exist in `rollout.py` and
`temporal.py`, and the CDP reconstruction-anchor and generative rollout paths
exist in the package; all are simply unreachable from `craftax_jepa_config()`.
