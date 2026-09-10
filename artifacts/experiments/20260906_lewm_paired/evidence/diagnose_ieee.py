"""Explicit evaluation-only precision intervention; no checkpoint reauthorization.

Load and verify both original checkpoints under their recorded environment before
the first Triton kernel invocation. Then use IEEE FP32 dot products in a scoped
diagnostic, retaining all existing tolerances and the original source evidence.
"""
import argparse
import json
from pathlib import Path

import torch
import triton

from d4mj.data import _sha256, atomic_manifest
from d4mj.gates import contract_digest
from d4mj.lewm_diagnostics import recurrence_audit
from d4mj.sources import lewm_source_manifest, tensor_state_digest
from d4mj.world_api import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("use a fresh diagnostic report path")
    before = lewm_source_manifest()
    bundles = {v: load_bundle(args.pair/v/"joint/step-010000.pt")[0] for v in ("raw", "tc")}
    report = {"schema": "d4mj_ieee_precision_diagnostic_v1", "recorded_sources": before,
              "driver_sha256": _sha256(Path(__file__)), "arms": {},
              "scope": "evaluation-only precision intervention, original research gate remains closed",
              "m4_authorized": False, "architecture_verdict": "not_evaluated"}
    with triton.knobs.language.scope():
        triton.knobs.language.fp32_default = "ieee"
        report["diagnostic_sources"] = lewm_source_manifest()
        for variant, bundle in bundles.items():
            identity = tensor_state_digest(bundle.world.state_dict())
            bundle.world.requires_grad_(True)
            try:
                result = {"status": "pass", "detail": recurrence_audit(bundle)}
            except Exception as error:
                result = {"status": "fail", "reason": str(error)}
            result["weights_and_buffers_unchanged"] = identity == tensor_state_digest(bundle.world.state_dict())
            report["arms"][variant] = result
            print(json.dumps({"variant": variant, "status": result["status"], "reason": result.get("reason")}), flush=True)
    report["environment_restored"] = lewm_source_manifest() == before
    report["report_id"] = contract_digest(report)
    atomic_manifest(args.out, report)


if __name__ == "__main__":
    main()
