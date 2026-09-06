"""Bounded, read-only localization of the completed-checkpoint recurrence stop.

Measure the unchanged source scan, streaming API and FP32 reference on identical
learned weights. Record failures without changing a tolerance or enabling M4.
"""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from d4mj.data import _sha256, atomic_manifest
from d4mj.gates import contract_digest
from d4mj.lewm_diagnostics import normalization_audit, recurrence_audit, mixer_numerical_audit
from d4mj.sources import lewm_source_manifest, tensor_state_digest
from d4mj.world_api import load_bundle


def measured(check):
    try:
        return {"status": "pass", "detail": check()}
    except Exception as error:
        return {"status": "fail", "reason": str(error)}


def differences(bundle, left, right):
    rows = []
    for i, (a, b) in enumerate(zip(bundle.state_tensors(left), bundle.state_tensors(right))):
        quantity = "latent" if i == 0 else "history" if i == 1 else "conv" if i % 2 == 0 else "ssm"
        d = (a-b).float().abs()
        rows.append({"quantity": quantity, "layer": (i-2)//2 if i >= 2 else None,
                     "max_abs": float(d.max()), "reference_max_abs": float(b.abs().max()),
                     "rms_error": float(d.square().mean().sqrt()),
                     "above_1e4_absolute": int((d > 1e-4).sum()), "elements": d.numel()})
    return rows


@torch.no_grad()
def sequence(bundle, z, actions, backend):
    w = bundle.world
    full = w.teacher(z, actions, backend=backend)
    state = w.start(z[:, :1])
    predictions = []
    for t in range(actions.shape[1]):
        predicted, history, memory = w.scan_pairs(state.latent, actions[:, t:t+1], state.memory, backend=backend)
        predictions.append(predicted)
        state = replace(state, latent=z[:, t+1:t+2].clone(), history=history, memory=memory, step=state.step+1)
    split = min(7, actions.shape[1]-1)
    prefix = w.teacher(z[:, :split+1], actions[:, :split], backend=backend).state
    chunk = w.teacher(z[:, split:], actions[:, split:], state=prefix, backend=backend)
    return {"scan_step": differences(bundle, full.state, state),
            "scan_chunk": differences(bundle, full.state, chunk.state),
            "prediction_scan_step_max_abs": float((full.predicted-torch.cat(predictions,1)).abs().max())}, full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("use a fresh diagnostic report path")
    report = {"schema": "d4mj_learned_recurrence_diagnostic_v1", "sources": lewm_source_manifest(),
              "driver_sha256": _sha256(Path(__file__)), "arms": {},
              "m4_authorized": False, "architecture_verdict": "not_evaluated",
              "scope": "localize numerical failure, no gate or recipe revision"}
    for variant in ("raw", "tc"):
        path = args.pair / variant / "joint/step-010000.pt"
        bundle, _ = load_bundle(path)
        before = tensor_state_digest(bundle.world.state_dict())
        bundle.world.requires_grad_(True)
        arm = report["arms"][variant] = {"checkpoint_sha256": _sha256(path), "sequence_rows": []}
        arm["normalization_gate"] = measured(lambda: normalization_audit(bundle))
        arm["recurrence_gate"] = measured(lambda: recurrence_audit(bundle))
        print(json.dumps({"variant": variant, "recurrence_gate": arm["recurrence_gate"]}), flush=True)
        arm["first_mixer"] = {p: measured(lambda p=p: mixer_numerical_audit(bundle.world.layers[0].mixer, p))
                              for p in ("fp32", "bf16")}
        bundle.world.requires_grad_(False)
        for length in (2, 17, 65, 257):
            rng = torch.Generator(device=bundle.device).manual_seed(500)
            z = torch.randn(2, length+1, 1, 192, device=bundle.device, generator=rng)
            actions = torch.randint(bundle.n_actions, (2, length), device=bundle.device, generator=rng)
            row = {"length": length, "seed": 500, "input": "standard_normal_synthetic"}
            row["triton"], full = sequence(bundle, z, actions, "triton")
            row["reference"], reference = sequence(bundle, z, actions, "reference")
            row["triton_reference"] = differences(bundle, full.state, reference.state)
            row["prediction_triton_reference_max_abs"] = float((full.predicted-reference.predicted).abs().max())
            arm["sequence_rows"].append(row)
            print(json.dumps({"variant": variant, "length": length,
                              "triton_ssm": max(r["max_abs"] for r in row["triton"]["scan_step"] if r["quantity"] == "ssm"),
                              "reference_ssm": max(r["max_abs"] for r in row["reference"]["scan_step"] if r["quantity"] == "ssm")}), flush=True)
        arm["world_buffers_and_weights_unchanged"] = before == tensor_state_digest(bundle.world.state_dict())
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    report["report_id"] = contract_digest(report)
    atomic_manifest(args.out, report)


if __name__ == "__main__":
    main()
