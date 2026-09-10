# Feature ladder: where does LeWM lose absolute state?

Status: ready to run. Structural smoke passed on CUDA; no result recorded.

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
