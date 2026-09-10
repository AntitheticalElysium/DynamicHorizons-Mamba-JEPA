# M03 capability suite: completed results and the localization that remains

Run: `artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2`, completed 2026-09-09.
`complete.json` reports `suite_complete: true`; the baseline decision is
`m03_capability: insufficient_coverage` and `m4_authorized: false`. Extracted numbers
and their source hashes are in [evidence/m03_20260910/summary.json](evidence/summary.json).
Read with the [paired research record](../20260906_lewm_paired/README.md) and the
[gate contract](../../../d4mj/m03/README.md).

M4 remains blocked. Nothing below is a control result, a policy result, or an
architecture attribution: Direct consumes 64 frames and LeWM four, so all cross-family
rows are descriptive comparisons of trained systems.

## Label

**TC is a positive mechanistic result and a negative recipe result for this Craftax
interface.** Temporal centering did what the objective is specified to do, measurably
and repeatably across five panels. On this corpus that same reallocation is paid for
out of the absolute state that control needs. Calling it simply a negative Craftax
result discards the mechanism; calling it a promising dynamics result ignores its cost.
Both halves are one trade, and the variance decomposition is what ties them together.

There is one training seed. The paired raw/TC contrast is the strongest causal evidence
available here, and it is not a replication.

## The dissociation

TC loses the state pathway and wins the action-effect pathway, on every panel.

Action-effect geometry, on the three panels where all four arms exclude zero roots and
therefore share an identical `equivalence_chance` baseline:

| panel | arm | NSE ↓ | retrieval | chance | retrieval − chance | Hungarian | rank ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| exact961 | raw | 0.202 | 0.477 | 0.472 | +0.005 | 0.559 | 4.55 |
| | tc | 0.037 | 0.686 | 0.472 | +0.214 | 0.763 | 2.39 |
| | direct-M | 0.151 | 0.843 | 0.472 | +0.371 | 0.911 | 2.30 |
| policy104 | raw | 0.197 | 0.463 | 0.478 | −0.015 | 0.554 | 4.53 |
| | tc | 0.049 | 0.700 | 0.478 | +0.222 | 0.766 | 2.50 |
| | direct-M | 0.181 | 0.854 | 0.478 | +0.375 | 0.905 | 2.35 |
| hazard5402 | raw | 0.210 | 0.510 | 0.492 | +0.019 | 0.580 | 4.40 |
| | tc | 0.032 | 0.735 | 0.492 | +0.243 | 0.787 | 2.31 |
| | direct-M | 0.147 | 0.886 | 0.492 | +0.394 | 0.930 | 2.11 |

Raw's action-effect retrieval is **at chance**. Raw does not have a measurably
meaningful action-effect geometry, so TC clearing it is a low bar. On the one
geometry metric that is representation-independent — retrieval against pixel-derived
equivalence classes — the order is **Direct > TC ≫ Raw ≈ chance**. `effect_nse`, the
metric on which TC beats Direct, is normalized within each arm's own coordinates and is
marked `within_arm_coordinates_descriptive_only`; it is not a cross-arm ranking.

On `primary` and `legacy751` the Raw/TC arms exclude 40 and 45 zero-effect roots while
Direct excludes none, so their chance baselines differ (0.474 vs 0.639 on primary) and
retrieval is not comparable across families there.

## Why: relative reallocation, not conservation

Variance decomposition of the observed-successor embedding — a property of the encoder,
not the world model:

| arm | root share | action share | interaction |
|---|---:|---:|---:|
| raw | 0.94–0.97 | 0.003–0.018 | 0.02–0.04 |
| tc | 0.52–0.67 | 0.047–0.158 | 0.29–0.38 |
| direct | 0.81–0.89 | 0.013–0.046 | 0.10–0.16 |

These are normalized shares in differently scaled latent spaces, and they do **not**
show energy leaving the root axis. In absolute terms TC's root energy is larger than
Raw's on every panel:

| panel | TC root ÷ Raw root | TC (action+interaction) ÷ Raw | TC total ÷ Raw total |
|---|---:|---:|---:|
| primary | 3.30× | 59.9× | 4.82× |
| exact961 | 3.42× | 53.4× | 6.19× |
| policy104 | 3.08× | 57.0× | 5.58× |
| hazard5402 | 3.69× | 53.2× | 6.49× |
| legacy751 | 3.64× | 66.9× | 6.35× |

The mechanism is **differential amplification**: action+interaction grew ~53–67× while
root grew ~3.1–3.7×, a consistent ~16–18× ratio, inside a latent whose total energy is
4.8–6.5× Raw's (std 2.10 vs 0.99). The decomposition establishes relative reallocation
and concentration. The *cost* is established independently — by the semantic probes,
effective rank 5.68 vs 39.23, and the action-only comparisons below.

## The cost, and the control that limits both arms

TC loses to Raw on absolute-state and rich-successor semantics on every panel, every
representation, both probe families: `successor_binary` −0.05 to −0.14,
`root_binary` −0.03 to −0.16, with large negative continuous R² differences. Static
retention: Raw CLS continuous R² 0.150 → 0.062 after projection; TC CLS −0.014 → −0.120.

TC's `[z,h]` does beat Raw on the coarse `outcomes` task in 9 of 10 panel×family cells
(+0.014 to +0.165; only legacy751/mlp negative at −0.076). That win is confined to that
one task of five, is concentrated in reward / achievement / inventory / tile — "something
happened" — and its consistent weak spot is death and damage, negative on primary
(−0.114 / −0.086, joint MLP) and legacy751 (−0.277 / −0.247).

The decisive control is `generated_minus_action`. Against an independently fitted
action-only decoder, `joint − action_only` is at or below zero in **19 of 20** cells for
both arms (Raw −0.004…−0.112, TC −0.004…−0.085; sole exception primary/linear/TC at
+0.010). TC is less far below the floor than Raw. Neither arm demonstrates consequence
information beyond what the candidate action alone predicts. The gate added this control
precisely because a Mamba state can re-encode the candidate action.

Context interventions are flat for both arms: c4 versus c1/c16/c64 and versus
within-root shuffled pair histories all move the score by ≤0.03. With four training
frames, `h` is local completed-pair information, not emergent long-horizon memory.

## The family-level positive

On all four historical panels, **both** LeWM arms lose essentially nothing through the
transition while Direct loses a large fraction of a better representation:

| panel | raw gen−obs | tc gen−obs | direct-A gen−obs |
|---|---:|---:|---:|
| exact961 | −0.005 | −0.003 | −0.239 |
| policy104 | −0.000 | −0.002 | −0.206 |
| hazard5402 | −0.007 | −0.000 | −0.127 |
| legacy751 | −0.011 | +0.002 | −0.218 |

(MLP family; the linear family is the same pattern.) Direct starts from a much stronger
representation — observed coarse AUC 0.85–0.92 versus 0.73–0.81 — and its transition
destroys much of it. This is a result for the jointly-trained LeWM transition family,
shared by both arms, independent of the centering argument. It is not control evidence.

## What is unresolved

The encoder diagnosis is now specific: Raw has a real CLS→z projector bottleneck, and
TC's loss begins **before** the projector, at CLS, with projection worsening it. What
is not known is whether health, inventory, prerequisites and spatial state survive in
the 81 ViT patch tokens that [`projected_and_cls`](../../../d4mj/lewm.py) discards at extraction.
This matters because the pinned TC-LeWM paper's downstream policy consumes patch tokens
alongside CLS, while our exported world state is projected CLS only; its robot-policy
result therefore does not validate a CLS-only bottleneck.

The next diagnostic is the frozen feature ladder — patch tokens → mean-pooled patches →
CLS → projected z, against Direct's 64×16 and matched 192-D compressions of it — scored
with the existing decomposition, effective rank and semantic probes on the same
all-action successors. It is a localization test, not another search for whether TC
changed the geometry; that part is established. Its protocol and status are tracked in
the experiment ledger under `artifacts/`.

Coverage must be repaired before any capability verdict: only 13/25 static binary and
18/25 successor binary targets are supported, and the baseline decision remains
`insufficient_coverage` for that reason.
