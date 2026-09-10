# Feature ladder: where does LeWM lose absolute state?

Status: **complete**, 2026-09-10. Verdict: pooling/export bottleneck, not capacity
and not a reshaped encoder. Result in [`evidence/ladder.json`](evidence/ladder.json).

## Question

[M03](../20260909_m03_capability/README.md) established *that* temporal centering
reshaped the representation and *that* absolute state was lost with it. It cannot say
**where** in the encoder that happens. We already know Raw has a CLS→z projector
bottleneck (continuous R² 0.150 → 0.062) and that TC is already weak at CLS
(−0.014 → −0.120). What is unknown is whether health, inventory, prerequisites and
spatial state survive in the 81 ViT patch tokens that
[`projected_and_cls`](../../../d4mj/lewm.py) discards at extraction.

This matters because the pinned TC-LeWM paper's downstream policy consumes patch tokens
alongside CLS, while our exported world state is projected CLS only. Its robot-policy
result does not validate a CLS-only bottleneck.

This is a localization test. It is **not** another search for whether TC changed the
geometry; that is settled.

## Decisive outcomes

| Reading | Conclusion |
|---|---|
| Patch tokens good, CLS bad | Pooling/export bottleneck. TC's mechanism may be usable with a spatial or persistent stream |
| CLS good, z bad | Projector bottleneck |
| Patch tokens already bad | Centering reshaped the whole encoder; no readout change recovers the state |
| Compressed Direct collapses similarly | Dimensional/spatial capacity explains much of the gap |
| Compressed Direct stays strong | The LeWM objective or its pooling is the more likely cause |

## Protocol

Four LeWM rungs and two Direct rungs, all frozen, all scored identically:

1. `patch_pca192` — the 81×192 ViT grid, reduced by TRAIN-fitted PCA to 192
2. `patch_mean` — mean-pooled patch tokens
3. `cls` — the CLS token
4. `projected_z` — the exported latent
5. `full_1024` / `pca192` — Direct's 64×16 latent and the same PCA reduction

The patch grid and Direct's 1024-D latent get the **identical** reduction, so "the patch
grid survives compression" and "Direct collapses under compression" are one symmetric
comparison, and every rung enters a probe of equal width. Each rung reports effective
rank, the root/action/interaction decomposition, and the sealed M03 probe suite
(linear + fixed MLP; static binary/continuous on roots, rich successor binary/continuous
on all seventeen forks), fitted on TRAIN only.

Nothing is trained and no label is re-derived. Roots, all-action successor pixels and
labels come from the sealed M03 sidecar; Direct rungs are read from that run's published
feature cache. LeWM rungs are captured by a forward hook on the frozen ViT **during the
same `projected_and_cls` call the gate uses** — no edit to `lewm.py`, which would
otherwise invalidate the raw/TC feature cache through the runtime hash in
`_feature_dependencies`.

## Parity

The `z` and `cls` rungs must reproduce the sealed encoding within a declared **1e-5
relative** bound, checked before any probe is fitted, failing closed. Parity is relative
rather than absolute because the arms differ in latent scale (TC `|z|` max 9.43 vs Raw
4.25); an absolute bound would penalise TC for being larger. Measured: 3.4e-7 raw,
7.0e-7 tc — ordinary FP32 for a 12-layer forward re-tiled at a different batch size.

## Data identity

Source run `artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2`, sidecar SHA256
`ead287f0…` as recorded in `evidence/ladder.json`. Primary support-v2 panel: 256 TRAIN /
128 DEV roots × 17 actions. Historical panels would need their per-seed replay shards
and are a separate extension.

## Outcome

Parity passed (raw 1.1e-6, tc 4.4e-6 relative). Every rung that overlaps the sealed M03
report reproduces it to three decimals — `raw` CLS 0.800/0.150, `raw` z 0.788/0.062,
`tc` CLS 0.749/−0.014, `tc` z 0.752/−0.120, Direct 0.848/0.107 — so the hook path is
the gate's own encoding.

MLP probes, DEV, macro AUC / mean R²:

| arm | rung | rank | root share | static AUC | static R² | succ AUC | succ R² |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | patch_pca192 | 24.7 | 0.942 | **0.832** | **0.233** | **0.805** | **0.320** |
| raw | patch_mean | 7.6 | 0.943 | 0.808 | 0.118 | 0.731 | 0.202 |
| raw | cls | 15.4 | 0.981 | 0.800 | 0.150 | 0.709 | 0.271 |
| raw | projected_z | 31.8 | 0.973 | 0.788 | 0.062 | 0.702 | 0.141 |
| tc | patch_pca192 | 11.7 | 0.911 | **0.796** | −0.095 | **0.746** | 0.104 |
| tc | patch_mean | 3.7 | 0.912 | 0.788 | **0.178** | 0.713 | **0.201** |
| tc | cls | 9.4 | 0.712 | 0.749 | −0.014 | 0.672 | 0.078 |
| tc | projected_z | 6.1 | 0.668 | 0.752 | −0.120 | 0.655 | −0.043 |
| direct | full_1024 | 47.4 | 0.886 | 0.848 | 0.107 | 0.832 | 0.178 |
| direct | pca192 | 30.0 | 0.901 | 0.806 | 0.024 | **0.823** | 0.169 |

**1. Patch tokens are the best rung for both arms; the export is a real bottleneck.**
Retention decreases monotonically patch → CLS → z on almost every cell. Raw recovers
+0.103 successor AUC (0.702 → 0.805) and more than doubles successor R² (0.141 → 0.320)
from features the encoder already computes and throws away. TC recovers +0.091
successor AUC (0.655 → 0.746). Nothing was retrained to get this.

**2. TC's action-concentration is created at the CLS token, not in the patch grid.**
Action + interaction share: TC patch 0.089, CLS 0.288, z 0.332 — against Raw patch
0.058, CLS 0.019, z 0.027, and Direct 0.114. In the spatial features TC resembles Raw
and Direct; the reallocation M03 measured appears when CLS pools the grid. Centering
shapes *what CLS pools*, not the whole encoder.

**3. Capacity does not explain the gap.** Direct loses almost nothing under the same
PCA-192 reduction (successor AUC 0.832 → 0.823). At matched 192-D width and matched
reduction: Direct 0.823 > raw-patch 0.805 > tc-patch 0.746 ≫ raw-z 0.702 > tc-z 0.655.
This is the "compressed Direct remains strong" branch: the LeWM objective and its
CLS-only pooling are the cause, not dimensionality or spatial extent.

### Qualifications

- TC's *continuous* retention stays poor at every rung — best is `patch_mean` at 0.178,
  and `patch_pca192` is negative. Patch tokens largely repair TC's binary/AUC retention,
  not its scalar regression. Mean-pooling recovers TC's continuous state best while
  scoring worst on rank, consistent with pooling low-passing the action-driven variation
  and leaving scene-constant magnitudes.
- TC's effective rank is roughly half Raw's at *every* rung, patch included (11.7 vs
  24.7). The action concentration is CLS-specific; the rank compression is not.
- Linear R² is negative for every arm and rung. These scalars are not linearly decodable
  from any of these features; only the fixed MLP recovers them.
- One training seed, one panel, and the M03 coverage gap is unchanged. This is
  retention, not control.

## What this cannot establish

Retention is not control. A rung that decodes state well is not evidence that the world
model can use it, that a policy can act on it, or that M4 should be authorized. The
panel's known coverage gap is unchanged: 13/25 static binary and 18/25 successor binary
targets are supported, so sparse targets remain `insufficient_coverage`, not failures.
One training seed; no replication.

## Run

```bash
TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu .venv/bin/python \
  artifacts/experiments/20260910_feature_ladder/ladder.py --device cuda
```

Output is immutable: `evidence/ladder.json` is never overwritten. `--limit N` writes
`ladder.smoke.json` and stamps `structural_smoke_not_a_result`.
