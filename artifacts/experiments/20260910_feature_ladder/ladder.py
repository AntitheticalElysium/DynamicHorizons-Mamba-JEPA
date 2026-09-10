"""Frozen feature ladder: locate where LeWM loses absolute state.

M03 established *that* temporal centering reshaped the geometry.  This locates
*where* the state information goes, by scoring the same frozen encoder at four
depths -- ViT patch tokens, mean-pooled patches, CLS, projected z -- against
Direct's 64x16 latent and a matched 192-D compression of it.

Nothing here is trained and nothing is re-derived: labels, roots and successor
pixels come from the sealed M03 sidecar, and the Direct rungs are read from that
run's published feature cache.  The LeWM rungs are captured with a forward hook
on the frozen ViT during the very same ``projected_and_cls`` call the gate uses,
so the ``z`` and ``cls`` rungs must reproduce the sealed encoding to within
``PARITY_TOLERANCE``.  That check runs, and fails closed, before any probe is
fitted.  It is not exact equality: this script batches differently from the gate,
which changes float32 GEMM tiling by ~1e-7 without changing the function.

Patch tokens and Direct's 1024-D latent are reduced by the identical operation
-- TRAIN-fitted PCA to 192 -- so "the patch grid survives compression" and
"Direct collapses under compression" are read off one symmetric comparison, and
every rung enters a probe of the same width.

Scope: the primary support-v2 panel only.  Historical panels would need their
per-seed replay shards and are a separate extension.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "d4mj_feature_ladder"
LEWM_RUNGS = ("patch_pca192", "patch_mean", "cls", "projected_z")
DIRECT_RUNGS = ("full_1024", "pca192")
# The gate encoded at encode_batch=16 with four-frame calls and stacked all
# seventeen successors at once.  Re-encoding at another batch size is the same
# function under different GEMM tiling, so parity is numerical, not exact.  The
# bound is relative because the two arms differ in latent scale (TC's |z| max is
# 9.43 against Raw's 4.25), which would otherwise penalise TC for being larger.
# Measured here: 3.4e-7 raw, 7.0e-7 tc, against a 12-layer FP32 forward.  The
# repo's sealed profile allows 1e-5 for a reference FP32 output.
PARITY_RELATIVE_TOLERANCE = 1e-5


def _encode_rungs(bundle, frames, batch):
    """Patch tokens, pooled patches, CLS and z from one frozen forward pass."""
    captured = {}
    handle = bundle.encoder.backbone.register_forward_hook(
        lambda module, args, output: captured.__setitem__("h", output.last_hidden_state))
    rows = {"patch": [], "patch_mean": [], "cls": [], "projected_z": []}
    try:
        with torch.inference_mode():
            for start in range(0, len(frames), batch):
                chunk = frames[start:start + batch].to(bundle.device)
                z, cls = bundle.encoder.projected_and_cls(chunk[:, None])
                tokens = captured["h"]
                if tokens.shape[1] < 2:
                    raise ValueError("ladder: encoder returned no patch tokens")
                rows["patch"].append(tokens[:, 1:].flatten(1).cpu())
                rows["patch_mean"].append(tokens[:, 1:].mean(1).cpu())
                rows["cls"].append(cls[:, 0].cpu())
                rows["projected_z"].append(z[:, 0, 0].cpu())
    finally:
        handle.remove()
    return {name: torch.cat(values) for name, values in rows.items()}


def _pca(fit: torch.Tensor, width: int):
    """TRAIN-fitted linear reduction, applied unchanged to every later split."""
    fit = fit.float()
    mean = fit.mean(0, keepdim=True)
    _, _, basis = torch.svd_lowrank(fit - mean, q=min(width, *fit.shape), niter=4)
    return lambda x: (x.float() - mean) @ basis


def _describe(successors: torch.Tensor) -> dict:
    """Effective rank plus the root/action/interaction split, on DEV successors."""
    from d4mj.lewm_diagnostics import covariance_summary
    from d4mj.m03.diagnostics import decompose

    parts = decompose(successors.double())
    energy = {k: float(v.expand_as(successors).square().sum())
              for k, v in parts.items() if k != "grand"}
    total = sum(energy.values())
    spectrum = covariance_summary(successors)
    return {"dimension": successors.shape[-1],
            "effective_rank": spectrum["effective_rank"],
            "coordinate_variance": spectrum["coordinate_variance"],
            "energy": energy,
            "shares": {k: v / total if total > 1e-12 else None for k, v in energy.items()}}


def _score(train_x, dev_x, train, dev, settings, device):
    """The sealed M03 probe recipe, unchanged, on one rung's features."""
    from d4mj.m03.gate import (STATIC_BINARY, STATIC_CONTINUOUS, _binary_metrics,
                               _expanded_roots, _fit_probe_many, _regression_metrics)

    roots = dev["episode"]
    fork_roots = _expanded_roots(roots, dev["next_binary"][..., 0])
    root_train, root_dev = train_x["root"].to(device), dev_x["root"].to(device)
    next_train = train_x["successor"].flatten(0, 1).to(device)
    next_dev = dev_x["successor"].flatten(0, 1).to(device)
    report = {}
    for hidden, family in ((False, "linear"), (True, "mlp")):
        fit = lambda x, y, e, binary: _fit_probe_many(x, y.to(device), {"dev": e}, settings,
                                                      hidden=hidden, binary=binary)["dev"]
        report[family] = {
            "static_binary": _binary_metrics(
                fit(root_train, train["root_binary"].float(), root_dev, True),
                dev["root_binary"].bool(), roots, STATIC_BINARY, settings),
            "static_continuous": _regression_metrics(
                fit(root_train, train["root_continuous"].float(), root_dev, False),
                dev["root_continuous"].float(), roots, STATIC_CONTINUOUS, settings),
            "successor_binary": _binary_metrics(
                fit(next_train, train["next_binary"].flatten(0, 1).float(), next_dev, True),
                dev["next_binary"].flatten(0, 1).bool(), fork_roots, STATIC_BINARY, settings),
            "successor_continuous": _regression_metrics(
                fit(next_train, train["next_continuous"].flatten(0, 1).float(), next_dev, False),
                dev["next_continuous"].flatten(0, 1).float(), fork_roots, STATIC_CONTINUOUS, settings),
        }
    return report


def run(source: Path, device: str, batch: int, limit: int = 0) -> dict:
    from d4mj.data import _sha256
    from d4mj.m03.cache import resolve_payload
    from d4mj.m03.gate import M03Settings, load_m03_bundle

    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    settings = M03Settings()
    manifest = json.loads((source / "sidecar/manifest.json").read_text())
    sidecar_path = source / "sidecar/sidecar.probe_only.pt"
    if _sha256(sidecar_path) != manifest["sidecar"]["sha256"]:
        raise ValueError("ladder: sidecar bytes differ from their sealed manifest")
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    splits = sidecar["splits"]
    if limit:
        # Structural smoke only: too few roots for coverage or an interval.
        splits = {s: {k: v[:limit] if isinstance(v, torch.Tensor) else v for k, v in rows.items()}
                  for s, rows in splits.items()}
    contract = json.loads((source / "run.json").read_text())["contract"]

    report = {"schema": SCHEMA, "panel": "primary_support_v2", "device": device,
              "source_run": {"path": str(source.resolve()), "contract_sha256": _sha(contract)},
              "sidecar_sha256": manifest["sidecar"]["sha256"],
              "probe_recipe": "sealed M03 settings; TRAIN-fit standardization; linear + fixed MLP",
              "reduction": "TRAIN-fitted PCA-192, identical for the LeWM patch grid and Direct 1024-D",
              "parity": {}, "arms": {}, "m4_authorized": False}

    for arm in ("raw", "tc"):
        bundle, payload, _ = load_m03_bundle(Path(contract[f"{arm}_checkpoint"]["path"]),
                                             device=device, dataset_sha256=sidecar["dataset_sha256"])
        del payload
        raw_rungs, cached = {}, {}
        for split, values in splits.items():
            roots = _encode_rungs(bundle, values["context"][:, -1], batch)
            forks = _encode_rungs(bundle, values["successors"].flatten(0, 1), batch)
            count = len(values["context"])
            raw_rungs[split] = {"root": roots,
                                "successor": {k: v.reshape(count, 17, -1) for k, v in forks.items()}}
            values_cache = resolve_payload(torch.load(
                source / f"features/{arm}.{split}.pt", map_location="cpu", weights_only=False))["features"]
            cached[split] = {k: v[:limit] if limit else v for k, v in values_cache.items()}
        # The hook must reproduce the sealed encoding, or no rung below is comparable.
        parity = {}
        for name, rung, reference in (("root_projected", ("root", "projected_z"), "projected"),
                                      ("successor_projected", ("successor", "projected_z"), "observed_successor"),
                                      ("root_cls", ("root", "cls"), "cls")):
            gap = max(float((raw_rungs[s][rung[0]][rung[1]] - cached[s][reference]).abs().max()) for s in splits)
            scale = max(float(cached[s][reference].abs().max()) for s in splits)
            parity[f"{name}_max_abs"] = gap
            parity[f"{name}_relative"] = gap / scale if scale > 0 else float("inf")
        parity["relative_tolerance"] = PARITY_RELATIVE_TOLERANCE
        parity["max_relative"] = max(v for k, v in parity.items() if k.endswith("_relative"))
        parity["status"] = "pass" if parity["max_relative"] <= PARITY_RELATIVE_TOLERANCE else "fail"
        report["parity"][arm] = parity
        if parity["status"] != "pass":
            raise ValueError(f"ladder: {arm} hook disagrees with the sealed encoding: {parity}")
        project = _pca(raw_rungs["train"]["successor"]["patch"].flatten(0, 1), 192)
        arm_report = {}
        for rung in LEWM_RUNGS:
            key = "patch" if rung == "patch_pca192" else rung
            features = {}
            for split in splits:
                root = raw_rungs[split]["root"][key]
                successor = raw_rungs[split]["successor"][key]
                if rung == "patch_pca192":
                    root = project(root)
                    successor = project(successor.flatten(0, 1)).reshape(len(root), 17, -1)
                features[split] = {"root": root, "successor": successor}
            arm_report[rung] = {**_describe(features["dev"]["successor"]),
                                "probes": _score(features["train"], features["dev"],
                                                 splits["train"], splits["dev"], settings, device)}
        report["arms"][arm] = arm_report
        del bundle, raw_rungs
        if device == "cuda":
            torch.cuda.empty_cache()

    for arm in ("direct_attention", "direct_mamba"):
        cached = {s: {k: v[:limit] if limit else v for k, v in resolve_payload(torch.load(
                      source / f"features/{arm}.{s}.pt", map_location="cpu", weights_only=False))["features"].items()}
                  for s in splits}
        project = _pca(cached["train"]["observed_successor"].flatten(0, 1), 192)
        arm_report = {}
        for rung in DIRECT_RUNGS:
            features = {}
            for split in splits:
                root, successor = cached[split]["projected"], cached[split]["observed_successor"]
                if rung == "pca192":
                    root = project(root)
                    successor = project(successor.flatten(0, 1)).reshape(len(root), 17, -1)
                features[split] = {"root": root, "successor": successor}
            arm_report[rung] = {**_describe(features["dev"]["successor"]),
                                "probes": _score(features["train"], features["dev"],
                                                 splits["train"], splits["dev"], settings, device)}
        report["arms"][arm] = arm_report
    return report


def _sha(value) -> str:
    from hashlib import sha256
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2",
                        help="sealed M03 run supplying the sidecar and Direct feature cache")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0,
                        help="structural smoke over the first N roots per split; never a result")
    args = parser.parse_args(argv)

    import sys
    sys.path.insert(0, str(ROOT))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / ("ladder.smoke.json" if args.limit else "ladder.json")
    if destination.exists():
        raise FileExistsError(f"ladder: refusing to replace {destination}")
    report = run(args.source, args.device, args.batch, args.limit)
    if args.limit:
        report["mode"] = "structural_smoke_not_a_result"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    worst = max(arm["max_relative"] for arm in report["parity"].values())
    print(json.dumps({"status": "complete", "report": str(destination),
                      "sealed_encoding_max_relative": worst,
                      "tolerance": PARITY_RELATIVE_TOLERANCE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
