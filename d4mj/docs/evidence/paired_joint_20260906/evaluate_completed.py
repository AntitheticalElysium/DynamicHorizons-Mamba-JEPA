"""Read-only completed-budget audit using the sealed runtime's existing probes.

Run from the repository root with PYTHONPATH=. and the pinned CUDA environment.
This artifact driver changes no recipe, threshold, runtime source or checkpoint.
It neither replaces the update-2000 G1 report nor authorizes M4.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from d4mj.checkpoint import read_lewm_bundle
from d4mj.config import load_recipe, recipe_digest
from d4mj.data import _sha256, atomic_manifest, load_joint_corpus, screen_windows
from d4mj.diagnostics import paired_auc_interval
from d4mj.gates import ComponentGateError, contract_digest, require_joint_screen
from d4mj.lewm_diagnostics import (
    covariance_summary, normalization_audit, recurrence_audit, screen_features,
    screen_prediction_report, screen_retention,
)
from d4mj.sources import lewm_source_manifest, tensor_state_digest
from d4mj.train import learning_rate
from d4mj.world_api import load_bundle


def require(condition, component, reason):
    if not condition:
        raise ComponentGateError(component, reason)


@torch.no_grad()
def encoder_features(bundle, windows, settings):
    """Independent projection boundary: never invoke failed recurrent dynamics."""
    bundle.encoder.eval()
    before = tensor_state_digest(bundle.encoder.state_dict())
    rows = {"projected": [], "cls": []}
    for start in range(0, len(windows["frames"]), settings.encode_batch):
        frames = windows["frames"][start:start+settings.encode_batch].to(bundle.device)
        z, cls = bundle.encoder.projected_and_cls(frames)
        rows["projected"].append(z[:, :, 0].cpu())
        rows["cls"].append(cls.cpu())
    require(before == tensor_state_digest(bundle.encoder.state_dict()), "encoder_features", "encoder buffers changed")
    return {key: torch.cat(values) for key, values in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "audit_output", "use a fresh audit directory")
    args.out.mkdir(parents=True)
    report = {
        "schema": "d4mj_completed_joint_audit_v1",
        "scope": "completed M0-M3 budget; repeated fixed G1 diagnostics, not a G2 pass",
        "sources": lewm_source_manifest(), "driver_sha256": _sha256(Path(__file__)),
        "arms": {}, "components": {}, "m4_authorized": False,
        "architecture_verdict": "not_evaluated", "decision": "stop_component",
    }
    component = "completed_pair_identity"
    try:
        settings = load_recipe(args.pair / "screen_recipe.json")
        g1 = json.loads((args.pair / "G1/screen.json").read_text())
        report["g1_report_id"] = g1["report_id"]
        report["g1_report_sha256"] = _sha256(args.pair / "G1/screen.json")
        report["settings_id"] = recipe_digest(settings)
        payloads, histories, configs = {}, {}, {}
        for variant in ("raw", "tc"):
            run = args.pair / variant
            config = configs[variant] = load_recipe(run / "resolved_recipe.json")
            checkpoint = run / "joint" / f"step-{config.joint.steps:06d}.pt"
            payload = payloads[variant] = read_lewm_bundle(checkpoint)
            require(payload["step"] == config.joint.steps, component, "joint budget incomplete")
            require(payload["config"] == json.loads((run / "resolved_recipe.json").read_text()),
                    component, "checkpoint differs from resolved recipe")
            require_joint_screen(g1, config, payload["dataset"], checkpoint)
            index_path = run / "joint/checkpoints.json"
            index = json.loads(index_path.read_text())
            expected = {f"step-{step:06d}.pt" for step in range(0, config.joint.steps+1, config.joint.checkpoint_every)}
            expected.add(f"step-{config.joint.screen_step:06d}.pt")
            require(set(index["snapshots"]) == expected, component, "snapshot set differs from cadence")
            for filename, digest in index["snapshots"].items():
                require(_sha256(run / "joint" / filename) == digest, component, "snapshot bytes changed")
            require((run / "joint/latest.pt").resolve() == checkpoint.resolve(), component, "latest pointer is stale")
            require(index["snapshots"][f"step-{config.joint.screen_step:06d}.pt"] == g1["arms"][variant]["checkpoint_sha256"],
                    component, "G1 checkpoint identity lost")
            initial = read_lewm_bundle(run / "joint/step-000000.pt")
            for name, state in payload["modules"].items():
                require(tensor_state_digest(initial["modules"][name]) == payload["initial_identity"][name],
                        component, "saved initialization differs")
                require(all(bool(torch.isfinite(x).all()) for x in state.values()), component, "nonfinite model state")
                for key, trained in payload["requires_grad"][name].items():
                    if not trained:
                        require(torch.equal(state[key], initial["modules"][name][key]), component, "frozen parameter changed")
            del initial
            require(all(int(state["step"]) == config.joint.steps for state in payload["optimizer"]["state"].values()),
                    component, "optimizer completed updates differ")
            history_path = run / "joint/metrics.jsonl"
            history = histories[variant] = [json.loads(line) for line in history_path.read_text().splitlines()]
            require([r["update"] for r in history] == list(range(1, config.joint.steps+1)), component, "missing or duplicate updates")
            require(all(all(math.isfinite(x) for x in r.values() if isinstance(x, (int, float))) for r in history),
                    component, "nonfinite training history")
            require(all(r["learning_rate"] == learning_rate(config, r["update"]-1) for r in history), component, "schedule changed")
            report["arms"][variant] = {
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
                "recipe_id": payload["recipe_id"], "checkpoint_index": index,
                "metrics_sha256": _sha256(history_path), "updates": len(history),
                "capabilities": payload["capabilities"], "initial_identity": payload["initial_identity"],
            }
        raw, tc = payloads["raw"], payloads["tc"]
        require({k:v for k,v in raw["config"].items() if k != "variant"} ==
                {k:v for k,v in tc["config"].items() if k != "variant"}, component, "paired recipes differ")
        require(raw["dataset"] == tc["dataset"] and raw["initial_identity"] == tc["initial_identity"], component, "paired identities differ")
        for stream in ("projection_rng", "cpu_rng"):
            require(torch.equal(raw[stream], tc[stream]), component, f"final {stream} differs")
        require(torch.equal(raw["sampler"]["generator"], tc["sampler"]["generator"]), component, "final sampler streams differ")
        require(all(a["windows"] == b["windows"] and a["learning_rate"] == b["learning_rate"]
                    for a,b in zip(histories["raw"], histories["tc"], strict=True)), component, "actual paired windows differ")
        first = [histories[v][0] for v in ("raw", "tc")]
        require(abs(first[0]["prediction"]-first[1]["prediction"]) <= 1e-6 and
                abs(first[0]["regularization"]-first[1]["regularization"]) > 1e-6, component, "objective contrast absent")
        heartbeats = []
        for line in (args.pair.parent / "research.log").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if {"variant", "update", "loss", "prediction"} <= row.keys():
                actual = histories[row["variant"]][row["update"]-1]
                require(all(row[k] == actual[k] for k in ("loss", "prediction")), component, "original heartbeat differs from retained history")
                heartbeats.append({k: row[k] for k in ("variant", "update", "loss", "prediction")})
        atomic_manifest(args.out / "original_heartbeats.json", heartbeats)
        report["components"][component] = {"status": "pass", "original_heartbeats_verified": len(heartbeats)}
        component = "dataset_and_windows"
        recorded = json.loads((args.pair / "dataset.json").read_text())
        episodes, contract = load_joint_corpus(recorded["path"], configs["raw"])
        require(contract == raw["dataset"] == recorded["contract"], component, "dataset identity changed")
        report["dataset_id"] = contract_digest(contract)
        by_id = {episode.episode_id: episode for episode in episodes}
        sampled_episodes, count = set(), 0
        frames = configs["raw"].joint.frames
        for row in histories["raw"]:
            require(len(row["windows"]) == configs["raw"].joint.batch,
                    component, "statistical batch changed")
            for episode_id, start in row["windows"]:
                episode = by_id[episode_id]
                require(episode.split == "train" and episode.uniform_eligible and
                        0 <= start <= len(episode)+1-frames,
                        component, "training window violates split or alignment")
                sampled_episodes.add(episode_id)
                count += 1
        report["training_exposure"] = {
            "per_arm_sampled_windows": count, "train_episodes_sampled": len(sampled_episodes),
            "per_arm_repeated_transition_exposures": count*(frames-1),
            "per_arm_repeated_frame_exposures": count*frames,
            "dev_or_final_training_windows": 0,
            "collector_training_access": contract["collector_training_access"],
        }
        windows = {split: screen_windows(episodes, configs["raw"], settings, split) for split in ("train", "dev")}
        window_rows = {split:{"episode_ids": list(w["episode_ids"]), "starts":w["starts"].tolist()} for split,w in windows.items()}
        require(window_rows == json.loads((args.pair / "G1/windows.json").read_text()), component, "evaluation windows changed")
        atomic_manifest(args.out / "windows.json", window_rows)
        report["components"][component] = {"status": "pass", "final_features_encoded": False}
        del payloads, raw, tc, histories, episodes
        gc.collect()
        for variant in ("raw", "tc"):
            entry = report["arms"][variant]
            print(json.dumps({"stage": "completed_checkpoint_audit", "variant": variant}), flush=True)
            bundle, _ = load_bundle(entry["checkpoint"])
            bundle.world.requires_grad_(True)
            for name, check in (("normalization", normalization_audit), ("recurrence", recurrence_audit)):
                component = f"{variant}_{name}"
                try:
                    report["components"][component] = {"status": "pass", "detail": check(bundle)}
                except Exception as error:
                    report["components"][component] = {"status": "fail", "reason": str(error)}
                    if name == "normalization":
                        raise ComponentGateError(component, str(error)) from error
            bundle.world.requires_grad_(False)
            component = f"{variant}_features"
            recurrence_passed = report["components"][f"{variant}_recurrence"]["status"] == "pass"
            extract = screen_features if recurrence_passed else encoder_features
            features = {split: extract(bundle, w, settings) for split, w in windows.items()}
            z = features["dev"]["projected"]
            entry["spectra"] = {"raw": covariance_summary(z), "residual": covariance_summary(z-z.mean(1,keepdim=True)),
                                "persistent": covariance_summary(z.mean(1))}
            entry["temporal_power"] = torch.fft.rfft(z.double(), dim=1).abs().square().mean((0,2)).tolist()
            variance = covariance_summary(features["train"]["projected"])["coordinate_variance"]
            require(variance >= settings.variance_floor, component, "numerical latent collapse")
            if recurrence_passed:
                entry["prediction"], errors = screen_prediction_report(features["dev"], windows["dev"]["clusters"], settings, variance=variance)
            else:
                entry["prediction"], errors = {"status": "blocked_by_recurrence"}, {}
            entry["retention"], probes = screen_retention(features["train"], features["dev"], windows["train"], windows["dev"],
                                                        settings, bundle.device, bundle.n_actions)
            torch.save({"features": features, "errors": errors, "probes": probes}, args.out / f"{variant}_rows.pt")
            component = f"{variant}_projection_retention"
            report["components"][component] = {
                "status": "fail" if entry["retention"]["projection_stop"] else "pass",
                "scope": "short-future proxies only",
            }
            print(json.dumps({"variant": variant, "prediction": entry["prediction"],
                              "rank": entry["spectra"]["raw"]["effective_rank"]}), flush=True)
            del bundle, features, probes, errors
            gc.collect()
            torch.cuda.empty_cache()
        # Descriptive matched-arm comparison; no new promotion or stopping threshold.
        saved = {v: torch.load(args.out / f"{v}_rows.pt", weights_only=False)["probes"] for v in ("raw", "tc")}
        selected = torch.tensor([row["supported"] for row in report["arms"]["raw"]["retention"]["coverage"]])
        report["paired_proxy_comparison"] = {
            "direction": "TC projected minus raw projected",
            "scope": "descriptive short-future proxies; no critical semantic or control claim",
            "probes": {},
        }
        for family in ("linear", "mlp"):
            row = saved["raw"]
            require(all(torch.equal(row[k], saved["tc"][k]) for k in ("truth", "valid", "clusters")),
                    "paired_proxy_comparison", "paired labels or episode clusters differ")
            comparison = paired_auc_interval(
                saved["tc"]["scores"][family]["projected"][:,selected],
                row["scores"][family]["projected"][:,selected],
                row["truth"][:,selected], row["valid"][:,selected], row["clusters"],
                draws=settings.bootstrap_draws, seed=settings.seed+400,
            ) if bool(selected.any()) else {"status": "insufficient_coverage"}
            report["paired_proxy_comparison"]["probes"][family] = comparison
        failed = [name for name, row in report["components"].items() if row["status"] == "fail"]
        if failed:
            report["blocked_components"] = failed
            report["blocked_component"] = failed[0]
        else:
            report["decision"] = "joint_budget_audited_m4_blocked"
        report["next_requirements"] = ["critical semantic retention with adequate labels", "M4 implementation authorization",
                                       "matched observed control and action-consequence gates before H16"]
    except Exception as error:
        component = getattr(error, "component", component)
        report["components"][component] = {"status": "fail", "reason": str(error)}
        report["blocked_component"] = component
    report["artifacts"] = {p.name: _sha256(p) for p in args.out.iterdir() if p.is_file()}
    report["artifact_root"] = str(args.out.resolve())
    report["report_id"] = contract_digest(report)
    atomic_manifest(args.out / "report.json", report)
    print(json.dumps({"decision": report["decision"], "component": report.get("blocked_component")}), flush=True)
    return int(report["decision"] == "stop_component")


if __name__ == "__main__":
    raise SystemExit(main())
