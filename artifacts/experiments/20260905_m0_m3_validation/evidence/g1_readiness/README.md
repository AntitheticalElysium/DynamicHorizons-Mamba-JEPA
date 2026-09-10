# G1 readiness, 2026-09-06

This is evaluator and orchestration validation, not the update-2,000 research result.

- `pytest_cpu.txt`: 205 passed, 7 skipped across the full suite. Includes G1 split/alignment, deterministic TRAIN-only probes, tied AUC and clustered uncertainty, initialization, actual paired histories, component stops, report tampering, exact-parent rejection and checkpoint continuation lineage.
- `corpus_integrity.json`: all420 source shards verified against the full support-v2 manifest (3,179,062 transitions). The subsequent research preflight verifies and audits that whole corpus independently.
- `raw/`, `tc/`: full B128/F4/J1024 BF16 architecture preflights using the intact first-shard technical fixture, with explicit verification-only four-update recipes and a two-update screen.
- `verification_screen.json`, `gpu_summary.json`: G1 mechanics passed on the resulting GPU checkpoints. The permissive two-update verification result is not a research G1 result and cannot authorize a checkpoint with the research recipe ID.
- `verify_gpu.py`: exact readiness driver used in the session. It requires the first-shard technical fixture created by the earlier integration verifier. Its output directory must be fresh; the recorded artifacts remain under `artifacts/lewm_gates_20260906/gpu_verification`.

The research launch uses the checked-in 10,000-update recipes, stops at update2,000 for the full `joint_screen.json` settings, and permits further updates only through `require_joint_screen`. M4/control remain blocked regardless of the G1 decision.
