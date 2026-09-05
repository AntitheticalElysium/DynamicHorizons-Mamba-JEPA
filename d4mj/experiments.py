"""Recipe preflight and joint-training orchestration for the shared D4MJ package."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .data import atomic_manifest, _sha256
from .data import load_joint_corpus
from .cache import cache_latents_to_store
from .config import load_recipe, recipe_dict, recipe_digest, config_from_dict
from .lewm_config import LeWMConfig
from .gates import preflight, require_joint_gates, ComponentGateError
from .train import train_joint
from .world_api import load_bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--backend", choices=("reference", "triton"))
    p.add_argument("--verification", action="store_true", help="non-research contract, does not certify GPU fit")
    p = sub.add_parser("joint")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stop-at", type=int)
    p.add_argument("--resume", type=Path)
    p = sub.add_parser("export")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true", help="allow an explicitly partial joint checkpoint")
    for stage in ("bridge", "actor", "render-fit", "play"):
        sub.add_parser(stage, help="not enabled by M0-M3")
    args = parser.parse_args(argv)
    if args.command in ("bridge", "actor", "render-fit", "play"):
        parser.error(f"phase_gate: {args.command} requires M4 or later; M0-M3 never authorizes it")
    destination = None
    try:
        if args.command == "preflight":
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("run_output: preflight requires a fresh directory")
            config = load_recipe(args.recipe)
            if not isinstance(config, LeWMConfig):
                if args.backend or args.verification or args.dataset:
                    raise ValueError("legacy preflight takes its mixer from the recipe and uses synthetic gate inputs")
                config = replace(config, device=args.device or config.device)
                args.out.mkdir(parents=True, exist_ok=True)
                destination = args.out
                atomic_manifest(args.out / "resolved_recipe.json", recipe_dict(config))
                report = preflight(config)
                atomic_manifest(args.out / "gates.json", report)
                failed = [name for name, result in report["components"].items() if result["status"] != "pass"]
                if failed:
                    raise ComponentGateError(failed[0], report["components"][failed[0]]["reason"])
                print(json.dumps({"stage": "Stage-A preflight", "status": "pass", "recipe_id": recipe_digest(config)}))
                return 0
            if args.dataset is None:
                raise ValueError("joint preflight requires --dataset")
            config = replace(config,
                             runtime=replace(config.runtime, device=args.device or config.runtime.device,
                                             purpose="verification" if args.verification else config.runtime.purpose),
                             dynamics=replace(config.dynamics, backend=args.backend or config.dynamics.backend))
            episodes, contract = load_joint_corpus(args.dataset, config)
            args.out.mkdir(parents=True, exist_ok=True)
            destination = args.out
            atomic_manifest(args.out / "resolved_recipe.json", recipe_dict(config))
            atomic_manifest(args.out / "dataset.json", {"path": str(args.dataset.resolve()), "contract": contract})
            atomic_manifest(args.out / "dataset_audit.json", contract["audit"])
            atomic_manifest(args.out / "baseline_manifest.json", {
                "scope": "M0-M3 mechanics; no control or external benchmark claim",
                "legacy": {"family": "MAE64x16 Flow|Direct x Attention|Mamba", "status": "retained_in_shared_runtime"},
                "new_comparison": "raw versus temporally centered SIGReg; paired initial weights and sampler seeds",
                "external_dreamerv3": {"status": "unresolved", "claim_authorized": False,
                    "required": ["same Craftax variant", "metric", "training access", "compute", "evaluation protocol"]},
                "collector_training_access": contract["collector_training_access"],
            })
            report = preflight(config, episodes, contract)
            atomic_manifest(args.out / "gates.json", report)
            if "sources" in report:
                atomic_manifest(args.out / "source_manifest.json", report["sources"])
            require_joint_gates(report, config, contract)
            print(json.dumps({"stage": "M0-M3 preflight", "status": "pass", "recipe_id": recipe_digest(config),
                              "m4_authorized": False}))
            return 0
        destination = args.run
        config = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
        if not isinstance(config, LeWMConfig):
            raise ComponentGateError("recipe_family", "joint/export require a LeWM recipe; legacy phase trainers remain in d4mj.train")
        recorded = json.loads((args.run / "dataset.json").read_text())
        episodes, contract = load_joint_corpus(recorded["path"], config)
        if contract != recorded["contract"]:
            raise ComponentGateError("dataset_identity", "dataset changed after M0")
        gates = json.loads((args.run / "gates.json").read_text())
        require_joint_gates(gates, config, contract)
        if args.command == "joint":
            train_joint(episodes, config, args.run / "joint", dataset_contract=contract,
                        gate_report=gates, stop_at=args.stop_at, resume=args.resume)
        else:
            bundle, checkpoint = load_bundle(args.checkpoint)
            if checkpoint["recipe_id"] != recipe_digest(config) or checkpoint["dataset"] != contract:
                raise ComponentGateError("export_parent", "checkpoint does not belong to this run")
            if not checkpoint["capabilities"]["joint_complete"] and not args.diagnostic:
                raise ComponentGateError("joint_completion", "use --diagnostic for a partial export")
            cache_latents_to_store(bundle.encoder, episodes, config, args.out,
                                   source_contract=contract, parent_checkpoint=_sha256(args.checkpoint))
            manifest = json.loads((args.out / "manifest.json").read_text())
            manifest["cache"]["joint_step"] = checkpoint["step"]
            manifest["cache"]["parent_checkpoint_path"] = str(args.checkpoint.resolve())
            manifest["cache"]["diagnostic_only"] = not checkpoint["capabilities"]["joint_complete"]
            atomic_manifest(args.out / "manifest.json", manifest)
        return 0
    except Exception as error:
        failure = {"status": "stopped", "component": getattr(error, "component", "m0_m3_execution"),
                   "reason": str(error), "architecture_verdict": "not_evaluated", "m4_authorized": False}
        if destination is not None and destination.exists():
            atomic_manifest(destination / "failure.json", failure)
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
