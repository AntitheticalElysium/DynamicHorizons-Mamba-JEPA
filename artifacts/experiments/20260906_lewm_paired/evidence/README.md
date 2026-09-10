# Paired joint research evidence

Runtime `ae896f5d4ce49d1713636794d6cece3673ab4dc7`; shared integration `0d49d7e`. See the [interpretation and stop decision](../README.md).

## Committed records

- `summary.json`: concise completed-budget, proxy and numerical results. It does not authorize M4.
- `pair.json`, `screen_recipe.json`, `raw_resolved_recipe.json`, `tc_resolved_recipe.json`: pre-training model/evaluation seals.
- `raw_gates.json`, `tc_gates.json` and corresponding source manifests: actual full-corpus research G0 reports under the original execution setting.
- `g1_screen.json`: original update2,000 report, with its parent checkpoint and diagnostic-row hashes. It authorized only the original10,000-update budget.
- `raw_checkpoints.json`, `tc_checkpoints.json`: all21 snapshots per arm, including initialization, screening and completion. The final audit verified every listed file's bytes.
- `completed_components.json`: final lineage/data audit and independent encoder probes. Both original full-stack recurrence checks fail, so this report deliberately says `stop_component` and omits dependent world predictions. This is not an architecture failure label.
- `learned_recurrence.json`: same learned weights and inputs at lengths2/17/65/257, unchanged Triton source versus the FP32 reference. Both first-layer FP32/BF16 source/gradient checks pass; full-stack state measurements localize the stop.
- `ieee_diagnostic.json`: both full recurrence audits pass unchanged tolerances when the sole execution change is explicit IEEE FP32 dot precision. Original checkpoint identity is verified before the first kernel call; weights/buffers stay unchanged. This diagnostic is not a training resume or a replacement G1 report.
- `ieee_raw_gates.json`, `ieee_tc_gates.json`, their source manifests and `ieee_preflight_summary.json`: fresh full-corpus G0 reports with IEEE explicitly recorded. All eight components pass for both arms, including actual B128/F4/J1024 BF16 resource checks with three optimizer updates. These updates are disposable resource measurements using the actual architecture, not new research training or a substitute model.
- `completion.json`: authoritative completion/stop record, distinct from the original launcher's historical process status.

The three Python files here are reproducible analysis drivers over shared runtime functions. They add no alternate world model or trainer. `evaluate_completed.py` runs independent encoder probes after a recurrence stop and preserves the failing status. `diagnose_recurrence.py` records the failure without changing thresholds. `diagnose_ieee.py` performs one explicit execution intervention in a fresh process.

## Local artifacts

The complete pair is `artifacts/lewm_gates_20260906/paired/`. Immutable weights, per-update histories, selected windows, projected/CLS features and probe score tensors remain local there. The committed reports contain their paths and SHA256 identities; these large files are not included in git.

- Final audit: `paired/completed_components/`.
- Original G1: `paired/G1/`.
- Bounded recurrence diagnostics: `paired/learned_recurrence.json`, `paired/ieee_diagnostic.json`.
- Original paired launch: `launch.json`, `research.log`. Its status/PID is historical after an environment refresh and explicit TC checkpoint resume. The final audit checks all187 original heartbeats against complete histories, including the replay overlap at8,600/8,700.
- The first audit attempt (`paired/completed_audit/`) contains the tuple-versus-JSON-list driver error. The corrected attempt (`paired/completed_audit_v2/`) verifies matching windows and records the first actual recurrence stop. Both preserve the exact driver and process output. Neither replaces a research report.

For a reproduction with the original checkpoints, start a fresh process with `TRITON_F32_DEFAULT` **unset**, `PYTHONPATH=.`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and use the pinned `.venv/bin/python`. Each driver takes `--pair artifacts/lewm_gates_20260906/paired --out <fresh-path>`. The completed audit uses an output directory; the two bounded diagnostics use a JSON file. A completed audit exit code1 is expected while the original recurrence contract remains failed. The IEEE driver explicitly records its intervention after validating the original execution identity.

The preceding regression evidence remains [205 CPU tests passed](../../20260905_m0_m3_validation/evidence/g1_readiness/pytest_cpu.txt), [targeted CUDA tests](../../20260905_m0_m3_validation/evidence/integration/pytest_gpu.txt), and [exact legacy integration parity](../../../../d4mj/spec/lewm/INTEGRATION.md). This final work changes documentation and analysis drivers, not the training runtime. No extra research budget, actor training, H16 bridge, renderer fit, FINAL evaluation or external score comparison was run.

Fresh IEEE preflights are reproduced with `TRITON_F32_DEFAULT=ieee OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m d4mj preflight --recipe d4mj/recipes/lewm_mamba_<raw-or-tc>.json --dataset artifacts/craftax_support_v2 --out <fresh-directory>`. The recorded pair used `record_joint_preflight` sequentially after loading the shared corpus once. Research recipes were unchanged; the source manifest explicitly records the new execution setting. Peak allocated/reserved memory is952,169,984 /1,193,279,488 bytes for both arms.
