from dataclasses import asdict
from pathlib import Path
import os
import tempfile

import torch

from .config import Config
from .sources import source_digests, verify_sources

FORMAT = "d4mj_checkpoint_v1"


def save(path: Path, config: Config, **objects) -> None:
    """Atomic, and carrying enough to prove what produced it and to resume it: the
    config, the digests of every pinned source a decision rests on, and any
    plain-dict state -- the running-RMS normalisers and, via
    `train.generator_state`, the sampler and model generators that actually drive
    training. The global stream is stored too, but nothing here draws from it."""
    payload = {
        "format": FORMAT,
        "config": asdict(config),
        "sources": source_digests(config),
        "rng": torch.get_rng_state(),
        "modules": {
            name: value.state_dict() if hasattr(value, "state_dict") else value
            for name, value in objects.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.rename(path)


def load(path: Path, config: Config, **objects) -> dict:
    """Restores modules and plain dicts in place."""
    payload = torch.load(path, weights_only=False)
    if payload["format"] != FORMAT:
        raise ValueError(f"expected {FORMAT}, found {payload['format']}")
    stored, requested = payload["config"], asdict(config)
    # A field added after a checkpoint was written is absent from its stored config, and
    # a whole-dict comparison then rejects every older checkpoint. Fields listed here
    # take their default when missing -- and only when the caller is asking for that
    # default, so a checkpoint that never trained with alignment cannot be loaded as
    # though it had.
    for field, default in (("align_weight", 0.0),):
        if field not in stored:
            if requested.get(field, default) != default:
                raise ValueError(
                    f"checkpoint predates `{field}` and cannot be loaded with "
                    f"{field}={requested[field]}"
                )
            stored = stored | {field: default}
    if stored != requested:
        raise ValueError("checkpoint config differs from the one requested")
    verify_sources(payload["sources"], config)
    torch.set_rng_state(payload["rng"])
    for name, target in objects.items():
        stored = payload["modules"][name]
        if hasattr(target, "load_state_dict"):
            target.load_state_dict(stored)
        else:
            target.clear()
            target.update(stored)
    return payload


LEWM_FORMAT = "d4mj_lewm_bundle_v2"


def save_lewm_bundle(path, bundle, *, step: int, dataset_contract: dict, initial_identity: dict,
                     optimizer, sampler, projection_rng, gate_report: dict) -> dict:
    """Resumable joint encoder AND predictor; no new phase is implied by a save."""
    from .config import recipe_dict, recipe_digest
    from .sources import lewm_source_manifest

    if not 0 <= step <= bundle.config.joint.steps:
        raise ValueError("checkpoint_schedule: step outside the declared joint schedule")
    modules = {"encoder": bundle.encoder, "world": bundle.world}
    payload = {
        "format": LEWM_FORMAT, "phase": "joint", "step": step,
        "config": recipe_dict(bundle.config), "recipe_id": recipe_digest(bundle.config),
        "sources": lewm_source_manifest(), "dataset": dataset_contract,
        "initial_identity": initial_identity, "gates": gate_report,
        "modules": {k: v.state_dict() for k, v in modules.items()},
        "modes": {k: {n: m.training for n, m in v.named_modules()} for k, v in modules.items()},
        "requires_grad": {k: {n: p.requires_grad for n, p in v.named_parameters()} for k, v in modules.items()},
        "encoder_frozen": bundle.encoder._frozen,
        "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
        "projection_rng": projection_rng.get_state(), "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Scheduler is a pure function of completed optimizer updates + sealed config.
        "scheduler": {"type": "linear_warmup_cosine_v1", "completed_updates": step},
        "capabilities": {"joint_complete": step == bundle.config.joint.steps,
                         "trained_recursive_depth": 0, "validated_recursive_depth": 0,
                         "readout_trained": False, "m4_authorized": False},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Immutable snapshots: publishing must fail if this step already exists.
    # A separate rolling pointer may move; it must never replace these bytes.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(payload, tmp)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return payload


def publish_lewm_latest(path) -> None:
    """Index a preserved snapshot by hash and atomically move the convenience link."""
    import json
    from .data import atomic_manifest, _sha256

    path = Path(path)
    index_path = path.parent / "checkpoints.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "schema": "d4mj_lewm_checkpoints_v1", "snapshots": {}}
    digest = _sha256(path)
    if path.name in index["snapshots"] and index["snapshots"][path.name] != digest:
        raise ValueError("checkpoint_identity: immutable snapshot changed")
    index["snapshots"][path.name] = digest
    index["latest"] = path.name
    atomic_manifest(index_path, index)
    temporary = path.parent / "latest.pt.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(path.name)
    temporary.replace(path.parent / "latest.pt")


def read_lewm_bundle(path) -> dict:
    from .config import config_from_dict, recipe_digest
    from .sources import verify_lewm_sources

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != LEWM_FORMAT or payload.get("phase") != "joint":
        raise ValueError("checkpoint_family: expected a v2 LeWM joint bundle")
    config = config_from_dict(payload["config"])
    if recipe_digest(config) != payload["recipe_id"]:
        raise ValueError("checkpoint_recipe: payload recipe digest mismatch")
    if (type(payload["step"]) is not int or not 0 <= payload["step"] <= config.joint.steps
            or payload["scheduler"] != {"type": "linear_warmup_cosine_v1", "completed_updates": payload["step"]}):
        raise ValueError("checkpoint_schedule: inconsistent completed updates")
    capabilities = {"joint_complete": payload["step"] == config.joint.steps,
                    "trained_recursive_depth": 0, "validated_recursive_depth": 0,
                    "readout_trained": False, "m4_authorized": False}
    if payload.get("capabilities") != capabilities:
        raise ValueError("checkpoint_phase: unsupported capabilities in an M0-M3 bundle")
    verify_lewm_sources(payload["sources"])
    return payload


def restore_lewm_bundle(path, bundle, *, dataset_contract, optimizer, sampler, projection_rng) -> dict:
    from .config import recipe_dict

    payload = read_lewm_bundle(path)
    if payload["config"] != recipe_dict(bundle.config):
        raise ValueError("checkpoint_recipe: resume cannot change family, normalization, backend or schedule")
    if payload["dataset"] != dataset_contract:
        raise ValueError("checkpoint_dataset: data bytes, split or collector lineage changed")
    if payload["encoder_frozen"]:
        raise ValueError("checkpoint_phase: an exported frozen encoder is not a joint-training resume")
    if payload["cuda_rng"] and (
        not torch.cuda.is_available() or len(payload["cuda_rng"]) != torch.cuda.device_count()
    ):
        raise ValueError("checkpoint_rng: CUDA devices changed during resume")
    for name, module in (("encoder", bundle.encoder), ("world", bundle.world)):
        module.load_state_dict(payload["modules"][name], strict=True)
        for key, child in module.named_modules():
            child.training = payload["modes"][name][key]
        for key, parameter in module.named_parameters():
            parameter.requires_grad_(payload["requires_grad"][name][key])
    bundle.encoder._frozen = payload["encoder_frozen"]
    optimizer.load_state_dict(payload["optimizer"])
    sampler.load_state_dict(payload["sampler"])
    projection_rng.set_state(payload["projection_rng"].cpu())
    torch.set_rng_state(payload["cpu_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload
