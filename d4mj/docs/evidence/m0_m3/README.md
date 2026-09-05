# M0–M3 evidence, 2026-09-05

These are technical verification artifacts, not trained research results. Read [implementation status](../../TC_LEWM_M0_M3_STATUS.md) for scope and limitations.

- `summary.json`: final suite counts, both full-architecture GPU preflights and GPU resume result.
- `raw/`, `tc/`: final resolved verification recipes, source closure, data audit and all component gates. These preserve actual B128/F4/J1024, native image geometry and six-layer Mamba settings. `purpose=verification` explicitly identifies the bounded real-data fixture.
- `recurrence_calibration.json`: two seeds, small and full-width mixers, nonzero incoming state, FP32/BF16, T2/17/65/257, source/reference/step and gradient error measurements.
- `world_calibration.json`: full six-layer state comparisons at T17/65/257 and an additional BF16 mixer gradient gate.
- `calibrate_recurrence.py`, `calibrate_world.py`: numerical measurement scripts used for those records. Run from the repo root with `PYTHONPATH=.` and the measured `.venv`; create `/tmp/lewm-audit` first for output. They call the production recurrence, not a separate prototype.
- `gpu_resume.json`, `verify_gpu_resume.py`: full-architecture, four-update verification against two-plus-two resume; model weights, BN buffers and metrics were exact. The script expects a fresh passing preflight in `/tmp/lewm-audit/full_gpu_preflight_final` and the fixture described below. Checkpoints remain local temporary verification files; this evidence directory does not contain binary training snapshots.
- `pytest_cpu.txt`: final `187 passed, 5 skipped` regression log. CUDA tests skipped in the sandbox are supplemented by actual GPU preflights outside it.

The fixture uses all 24 complete episodes in `artifacts/craftax_support_v2/shard-000000.pt`, whose SHA256 must match the parent manifest. Its minimal store manifest references that shard and retains parent manifest, collector, expert and split identities; `tc/dataset.json` records the exact contract. It is not a selected research corpus. Only its 18 TRAIN episodes enter optimization. The 3 DEV and 3 FINAL episode records are audited for disjointness, not used to calibrate BN or train.

Source manifests record pinned commits/licenses, installed operator bytes, dependency versions, math backend settings and hashes of the active runtime files. A future code or source edit invalidates them for launch; preserve these records as historical evidence and rerun preflight. SSM numerical gates use absolute-only tolerances. None of these gates authorizes G1 continuation, M4, H16, an actor or a renderer.
