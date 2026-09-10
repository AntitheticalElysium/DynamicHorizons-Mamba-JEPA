# Gates

The separate TC-LeWM–Mamba family uses [component-scoped M0–M3 gates](lewm/STATUS.md) in `lewm_diagnostics.py`: source/data/device, objective, construction, recurrence, normalization and actual-batch resource checks. CUDA execution was validated outside the sandbox. A failure stops its component and records no architecture verdict. These gates do not authorize G1 continuation, M4, H16, actor training or rendering; the legacy gates below are unchanged.

What must hold before any number from this repo is a result. Gates run on the
deployment device across the Stage-A lattice — `{flow, direct} × {attention,
mamba}` — via `python -m d4mj`, and every one is seeded: a gate that draws fresh
weights each run has a pass that is a coin flip, and this suite reported both 22
and 24 passing on consecutive runs of one commit before that was fixed.

Gates come before results. An arm that fails one is not a result, it is a bug
wearing a number.

## The six

| Gate | Claim | What it would catch |
|---|---|---|
| `alignment` | The temporal contract, on episodes whose every value identifies its own index | Any shift in the led-to convention; a window crossing an episode boundary; a start-of-episode action leaking into a mid-episode window; a relevant row drawn without the whole event transition; a batch that stratifies during pretraining |
| `scan_step_parity` | One batched scan equals the same frames stepped one at a time carrying memory — in **outputs and in state** | Training reads the scan and imagination reads the steps, so a divergence is a model correct in every loss and wrong in every rollout. Runs past `dynamics_context` so the windowed branch actually executes |
| `reset_parity` | An episode boundary erases the previous episode | A driver threading state across a boundary. Asserted by *running* two episodes and requiring the second to match the same episode run alone, including the step counter that dates RoPE and the decode window |
| `firewall` | Agent/task state reaches the world only through the chosen action | A mask that blocks one direction only. Asserted both ways, with the agent inactive, and across `q` once tasks are enabled |
| `branch_nonmutation` | Evaluating a candidate does not mutate the state it was evaluated from | The flow arm's rungs are read-only; a candidate that wrote memory would make the commit path depend on how many rungs ran |
| `recurrent_carry` | The rollout depends on history | A model ignoring its own memory passes every one-step loss and produces a constant trajectory. This is also the only place the flow arm's history dependence is tested, since `World.predict` reads the current block's own corrupted latent |

## Two things the suite learned the hard way

**Confounded comparisons pass.** `reset_parity` and `recurrent_carry` once compared
`initial` against `advance`. Those commit at different points in the noise stream,
so for the flow arm the corruption draw alone moved the output and both gates
passed with memory entirely inert. They now vary memory against one *identical*
committed tensor.

**A gate that only ever passes proves nothing.** Each history gate is
mutation-tested against the mutation the *other* one survives: `recurrent_carry`
must fail when memory is made inert, `reset_parity` must fail when a reset leaks
the previous episode's state. Inert memory correctly passes `reset_parity`, which
is why the two are not duplicates.

## Tolerances, and why they differ

Output drift is checked at `1e-3` relative. Measured: 8e-5 for Mamba, 4e-7 for
attention, neither growing with sequence length.

State drift is checked separately at `5e-3`. An SSM state is a raw internal
quantity the output projection contracts, so Mamba's state drifts 1.0e-3 while
its output drifts 8e-5. That is kernel numerics rather than misalignment, and the
evidence is that it is **flat in length** — 1.04e-3 at 8 blocks, 8.99e-4 at 200.
A misalignment is order one and accumulates, so the gap between 1e-3 and 1 is
what the gate actually tests.

## What the gates do not cover

Gates establish executability and contract-conformance, not experimental validity.
They say nothing about whether a trained model is any good, and passing them is
not evidence for any claim in `DECISIONS.md`. The preregistered evaluation (S52)
is a separate instrument, run once on sealed seeds after all selection.

`pytest d4mj/tests` is the companion: gates assert cross-arm contracts on the
deployment device, tests pin semantics on CPU.

## Stage-B representation gates

The six above remain unchanged. Each new representation also passes two gates
before its cached latents may train a world model.

`representation_export` binds `C*` to the representation family, checkpoint and
declared copy; emits finite dense `[B,T,64,16]` latents with every training
corruption off; and matches recurrent scan, cached targets, and deployment.
EMA-JEPA-R must export its stopped EMA target. LeVJEPA-R must export its
evaluation Polyak copy without projector output; changing an excluded token
cannot change its training readout, and any later compact path must match padded
outputs, gradients, and state.

`representation_retention` freezes the encoder and applies the same DEV probes
to every export: state/inventory/health, action-conditioned successor, death,
and achievement-event information. Thresholds are fixed from the MAE anchor
before either JEPA arm is inspected. Noncollapse or a good clip readout is not a
substitute: LeVJEPA directly supervises only that readout, while the world model
consumes dense latents.

Source-fidelity checks also record EMA masks/predictor/momentum, or LeVJEPA
same-window views, two-sided gradients, per-view SIGReg, true keep positions,
and the actual global sample count seen by SIGReg. Optimizer accumulation does
not enlarge that count unless embeddings are regularized jointly. Phase 1A uses
the full corpus and no task, reward, death, or achievement labels.

## Task-conditioning extension

When achievement conditioning is enabled, `firewall` additionally varies `q`
at fixed observation, action, and memory. World latents, non-agent memory streams,
and next-latent predictions must remain identical; agent memory, policy, task
reward, and value may change; continuation may not. The eligible task set,
sparse labels, prompt, and task-switch handling of agent memory are derived from
TRAIN only and frozen before DEV or FINAL execution. A switch never resets world
streams. Per-task actors compare against their own frozen task-conditioned BC
priors and random; aggregate control is reported separately, not substituted for
that matched prior.

## Trained outcome gate

The six structural gates run before training. A separate gate runs after Phase 2
because its claim depends on learned weights and the real simulator. On held-out
seeds, the BC prior supplies states and Craftax executes all 17 actions from each
identical state. Phase 3 is refused unless:

- at least three states have action-dependent rewards and three have
  action-dependent death;
- generated-successor reward choice regret beats the best state-blind
  action-marginal predictor on those forks; and
- terminal BCE beats the corresponding action marginal and terminal AUC exceeds
  0.5.

The raw forks are saved per arm. After actor training, exact policy-weighted
one-step death on those same forks may not exceed the frozen BC prior. A static
action prior is mutation-tested and cannot pass; the pre-fix Direct-A checkpoint
also fails (terminal BCE 3.121 versus its 0.567 action marginal).

The same fork now also reads every simulator-produced successor through the real
observation path. Those observed-successor reward/death metrics do not change the
gate; they localise a failure. If they pass while generated successors fail, the
transition lost outcome information. If both fail, the logged-data head itself is
unidentified. This distinction must precede any new loss or data intervention.

For Direct, a matched rerun also localises observed and generated readouts on the
same saved states after S76. Separately, the S35 diagnostic holds `(state, action)`
fixed while varying only simulator RNG and compares Direct-to-mean,
Direct-to-nearest-mode, and Flow precision/coverage. It is diagnostic evidence,
not an automatic gate that enables `K > 1`.
