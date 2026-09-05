# Post-integration evidence

These records validate the shared runtime on 2026-09-06. They are component and regression evidence, not learned-control or external benchmark results.

- `legacy_parity.json`: exact comparisons to pre-LeWM git caller functions for all four legacy GPU arms. `verify_legacy.py` reproduces the check from the repository root with `PYTHONPATH=.`.
- `joint_parity.json`: original joint trainer/sampler versus integrated code, four CPU fixture updates. Original source hashes and comparison scope are explicit. The original untracked modules are no longer runtime files; this comparison record is historical, not a standalone rerun script.
- `raw/`, `tc/`: fresh full-architecture resolved recipes, gate reports, source closures, dataset lineage and baseline scope records.
- `gpu_summary.json`: both full-size preflights and exact GPU pause/resume with immutable checkpoint identity. `verify_gpu.py --work <fresh-directory>` recreates the technical shard manifest from the existing support corpus, performs both preflights and the resume check. It preserves the 10,000-update schedule while stopping verification after four updates.
- `pytest_cpu.txt`, `pytest_gpu.txt`: final regression logs. The CUDA subset also contains four intentionally CPU-configured legacy Mamba tests that remain skipped; new CUDA adapter tests, functional recurrence tests and git-reference checks exercise the GPU separately.

Final results: **199 passed, 7 skipped** in the full CPU suite; **21 passed, 4 skipped** in the targeted CUDA invocation. The skipped CUDA-invocation cases use the unchanged CPU fixture; they are not reported as CUDA passes. Both full-size raw/TC GPU gate reports contain eight passing components.

Commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest d4mj/tests -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  d4mj/tests/test_integration.py d4mj/tests/test_mamba_recurrence.py \
  d4mj/tests/test_time_mixer.py -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  d4mj/docs/evidence/integration/verify_legacy.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  d4mj/docs/evidence/integration/verify_gpu.py --work /tmp/d4mj-integration-fresh
```

GPU commands require host GPU access outside the sandbox. Source verification uses the current imported runtime and pinned third-party trees. Dataset and checkpoint paths in reports identify temporary technical artifacts; `verify_gpu.py` reconstructs the fixture from its recorded parent corpus. It does not infer unknown collector training access or authorize a research/control result.
