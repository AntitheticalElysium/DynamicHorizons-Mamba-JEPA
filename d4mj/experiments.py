"""Recipe preflight and joint-training orchestration for the shared D4MJ package."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .data import atomic_manifest, _sha256
from .data import load_joint_corpus
from .cache import cache_latents_to_store
from .config import load_recipe, recipe_dict, recipe_digest, config_from_dict
from .lewm_config import LeWMConfig, ScreenConfig
from .gates import preflight, require_joint_gates, ComponentGateError
from .train import train_joint, initialize_joint
from .world_api import load_bundle


def record_joint_preflight(config, episodes, contract, dataset, out):
    """One artifact writer and gate runner for standalone and paired launches."""
    atomic_manifest(out / "resolved_recipe.json", recipe_dict(config))
    atomic_manifest(out / "dataset.json", {"path": str(dataset.resolve()), "contract": contract})
    atomic_manifest(out / "dataset_audit.json", contract["audit"])
    atomic_manifest(out / "baseline_manifest.json", {
        "scope": "M0-M3 mechanics; no control or external benchmark claim",
        "legacy": {"family": "MAE64x16 Flow|Direct x Attention|Mamba", "status": "retained_in_shared_runtime"},
        "new_comparison": "raw versus temporally centered SIGReg; paired initial weights and sampler seeds",
        "external_dreamerv3": {"status": "unresolved", "claim_authorized": False,
            "required": ["same Craftax variant", "metric", "training access", "compute", "evaluation protocol"]},
        "collector_training_access": contract["collector_training_access"],
    })
    report = preflight(config, episodes, contract)
    atomic_manifest(out / "gates.json", report)
    if "sources" in report:
        atomic_manifest(out / "source_manifest.json", report["sources"])
    require_joint_gates(report, config, contract)
    return report


def run_joint_pair(configs, settings, dataset, output, *, screen_only=False):
    """G0 -> saved initialization -> paired 2k -> G1 -> accepted finite budget."""
    import gc
    import os
    import time
    import torch
    from .gates import contract_digest
    from .lewm_diagnostics import screen_joint_pair
    from .data import screen_windows

    if not isinstance(settings,ScreenConfig) or not all(isinstance(c,LeWMConfig) for c in configs.values()):
        raise ValueError("paired-run requires two model recipes and a screen recipe")
    if set(configs) != {"raw","tc"} or any(c.variant != v for v,c in configs.items()):
        raise ComponentGateError("pair_recipe", "paired recipes must name raw and tc")
    identity = [{k:v for k,v in recipe_dict(c).items() if k != "variant"} for c in configs.values()]
    if identity[0] != identity[1]:
        raise ComponentGateError("pair_recipe", "recipes may differ only in centering")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def status(stage, **detail):
        row = {"stage":stage,"pid":os.getpid(),"updated_unix":time.time(),"m4_authorized":False,**detail}
        atomic_manifest(output/"status.json",row)
        print(json.dumps(row),flush=True)
    atomic_manifest(output/"screen_recipe.json",recipe_dict(settings))
    for variant,c in configs.items():
        atomic_manifest(output/f"{variant}_recipe.json",recipe_dict(c))
    status("dataset_validation")
    episodes, contract = load_joint_corpus(dataset,configs["raw"])
    atomic_manifest(output/"dataset.json",{"path":str(Path(dataset).resolve()),"contract":contract})
    # Fail coverage before spending the joint budget; no model features are inspected.
    for split in ("train","dev"):
        screen_windows(episodes,configs["raw"],settings,split)
    runs = {v:output/v for v in configs}
    reports, initial = {}, {}
    for variant,c in configs.items():
        status("preflight",variant=variant)
        runs[variant].mkdir()
        reports[variant] = record_joint_preflight(c,episodes,contract,Path(dataset),runs[variant])
        _,initial[variant] = initialize_joint(episodes,c,runs[variant]/"joint",dataset_contract=contract,gate_report=reports[variant])
        gc.collect(); torch.cuda.empty_cache()
    if initial["raw"] != initial["tc"]:
        raise ComponentGateError("initial_identity", "paired initial weights differ")
    atomic_manifest(output/"pair.json",{"schema":"d4mj_joint_pair_v1","dataset_id":contract_digest(contract),
                    "initial_identity":initial,"screen_settings_id":recipe_digest(settings),
                    "recipes":{v:recipe_digest(c) for v,c in configs.items()},"m4_authorized":False})
    for variant,c in configs.items():
        status("joint_to_screen",variant=variant,target_update=c.joint.screen_step)
        bundle,_ = train_joint(episodes,c,runs[variant]/"joint",dataset_contract=contract,gate_report=reports[variant],
                               stop_at=c.joint.screen_step,resume=runs[variant]/"joint/step-000000.pt")
        del bundle; gc.collect(); torch.cuda.empty_cache()
    status("G1")
    report = screen_joint_pair(runs,episodes,contract,settings,output/"G1")
    status("G1_complete",decision=report["decision"],component=report.get("blocked_component"))
    if report["decision"] != "continue_joint_budget":
        return 1
    if screen_only:
        return 0
    for variant,c in configs.items():
        if c.joint.steps == c.joint.screen_step:
            continue
        status("joint_to_budget",variant=variant,target_update=c.joint.steps)
        bundle,_ = train_joint(episodes,c,runs[variant]/"joint",dataset_contract=contract,gate_report=reports[variant],
                               stop_at=c.joint.steps,resume=runs[variant]/"joint"/f"step-{c.joint.screen_step:06d}.pt",
                               screen_report=report)
        del bundle; gc.collect(); torch.cuda.empty_cache()
    status("joint_budget_complete",next_component="M4",next_component_status="blocked")
    return 0


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
    p.add_argument("--screen-report", type=Path)
    p = sub.add_parser("export")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true", help="allow an explicitly partial joint checkpoint")
    p = sub.add_parser("paired-run", help="paired joint run with a sealed G1 stop")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    recipes = Path(__file__).with_name("recipes")
    p.add_argument("--raw-recipe", type=Path, default=recipes/"lewm_mamba_raw.json")
    p.add_argument("--tc-recipe", type=Path, default=recipes/"lewm_mamba_tc.json")
    p.add_argument("--screen-recipe", type=Path, default=recipes/"joint_screen.json")
    p.add_argument("--screen-only", action="store_true", help="pause after G1 even if it passes")
    for stage in ("bridge", "actor", "render-fit", "play"):
        sub.add_parser(stage, help="not enabled by M0-M3")
    args = parser.parse_args(argv)
    if args.command in ("bridge", "actor", "render-fit", "play"):
        parser.error(f"phase_gate: {args.command} requires M4 or later; M0-M3 never authorizes it")
    destination = None
    try:
        if args.command == "paired-run":
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("run_output: paired run requires a fresh directory")
            configs = {v:load_recipe(getattr(args, v+"_recipe")) for v in ("raw","tc")}
            settings = load_recipe(args.screen_recipe)
            destination = args.out
            return run_joint_pair(configs,settings,args.dataset,args.out,screen_only=args.screen_only)
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
            record_joint_preflight(config, episodes, contract, args.dataset, args.out)
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
                        gate_report=gates, stop_at=args.stop_at, resume=args.resume,
                        screen_report=None if args.screen_report is None else json.loads(args.screen_report.read_text()))
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
