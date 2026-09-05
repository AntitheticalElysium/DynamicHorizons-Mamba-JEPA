import hashlib
from pathlib import Path

from .config import Config

ROOT = Path(__file__).resolve().parent.parent / "third_party"

PINNED = {
    "dreamer4_paper": "papers/2509.24527v1.pdf",
    "mamba2_module": "sources/state-spaces__mamba/mamba_ssm/modules/mamba2.py",
    "lejepa_minimal": "sources/rbalestr-lab__lejepa/MINIMAL.md",
    "vjepa2_ac_train": "vjepa2/app/vjepa_droid/train.py",
    "dreamerv3_agent": "sources/danijar__dreamerv3/dreamerv3/agent.py",
    "mop_jepa_paper": "papers/2607.05238v1.pdf",
}
"""Sources whose bytes a decision in spec/DECISIONS.md rests on.

Deliberately not every file we read: a manifest that lists everything consulted
stops being checked. Reproductions that only corroborate are cited in the spec
but not pinned here.
"""


def source_digests(config: Config) -> dict[str, str]:
    names = list(PINNED)
    if config.time_mixer != "mamba":
        names.remove("mamba2_module")
    return {name: _digest(ROOT / PINNED[name]) for name in names}


def verify_sources(recorded: dict[str, str], config: Config) -> None:
    expected = source_digests(config)
    missing = set(expected) - set(recorded)
    if missing:
        raise ValueError(f"checkpoint omits required sources: {sorted(missing)}")
    drifted = [name for name, value in expected.items() if recorded[name] != value]
    if drifted:
        raise ValueError(f"pinned sources changed since this checkpoint: {drifted}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lewm_source_manifest() -> dict:
    """Actual imported files plus pinned analogues; separate from the legacy closure.

    Verify all imported Mamba Triton helpers against the pinned checkout. A version
    string alone cannot detect an edited installed operator.
    """
    import importlib.metadata
    import importlib.util
    import inspect
    import subprocess
    import os
    import torch
    from transformers import ViTConfig, ViTModel

    lewm = ROOT / "sources/lucas-maes__le-wm"
    stable = ROOT / "sources/galilai-group__stable-pretraining"
    mamba = ROOT / "sources/state-spaces__mamba/mamba_ssm"
    installed = Path(importlib.util.find_spec("mamba_ssm").origin).parent
    references = {
        "sources_lock": ROOT / "SOURCES.lock",
        "papers_lock": ROOT / "PAPERS.lock",
        "dependencies_lock": ROOT.parent / "requirements-lewm-rtx3060.lock.txt",
        "tc_paper": ROOT / "papers/2607.26924v3-tclewm.pdf",
        "vit_helper": stable / "stable_pretraining/backbone/utils.py",
        "pixel_statistics": stable / "stable_pretraining/data/dataset_stats.py",
    }
    # Recoverable repository identities, not just hashes of unattached local files.
    pins = {}
    lock_lines = [line.split() for line in (ROOT / "SOURCES.lock").read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
    for directory in (lewm, stable, mamba.parent):
        entry = next((row for row in lock_lines if row[0] == directory.name), None)
        if entry is None or len(entry) != 3:
            raise ValueError(f"source_pin: missing recoverable pin for {directory.name}")
        head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if head != entry[2]:
            raise ValueError(f"source_pin: checkout differs from SOURCES.lock: {directory.name}")
        pins[directory.name] = {"url": entry[1], "commit": entry[2]}
        references[f"{directory.name}/LICENSE"] = directory / "LICENSE"
    for name in ("train.py", "module.py", "jepa.py", "utils.py", "config/train/lewm.yaml",
                 "config/train/model/lewm.yaml"):
        references[f"lewm/{name}"] = lewm / name
    paths = [Path("modules/mamba2.py"), *[p.relative_to(mamba) for p in sorted((mamba / "ops/triton").rglob("*.py"))]]
    for relative in paths:
        wanted, actual = mamba / relative, installed / relative
        if not actual.exists() or _digest(wanted) != _digest(actual):
            raise ValueError(f"source_mamba: installed operator differs from pin: {relative}")
        references[f"mamba/{relative}"] = wanted
    import transformers.modeling_utils
    import transformers.integrations.sdpa_attention
    runtime = {"transformers_vit": Path(inspect.getfile(ViTModel)),
               "transformers_modeling_utils": Path(inspect.getfile(transformers.modeling_utils)),
               "transformers_sdpa": Path(inspect.getfile(transformers.integrations.sdpa_attention)),
               "transformers_config": Path(inspect.getfile(ViTConfig))}
    # Runtime closure includes the functions actually controlling loss, state and resume.
    for name in ("lewm_config.py", "lewm.py", "mamba_recurrence.py", "world_api.py", "state.py",
                 "config.py", "cache.py", "train.py", "checkpoint.py", "data.py", "sources.py",
                 "gates.py", "lewm_diagnostics.py", "experiments.py", "__main__.py",
                 "execution.py", "imagination.py", "diagnostics.py"):
        runtime[f"d4mj/{name}"] = Path(__file__).parent / name
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "mamba-ssm", "triton", "einops", "numpy", "safetensors")}
    return {"schema": "d4mj_lewm_sources_v1",
            "pins": pins,
            "references": {name: _digest(path) for name, path in references.items()},
            "runtime": {name: _digest(path) for name, path in runtime.items()},
            "versions": versions,
            "execution": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark,
                          "cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT", "unset")}}


def verify_lewm_sources(recorded: dict) -> None:
    current = lewm_source_manifest()
    if recorded != current:
        changed = [section for section in current if recorded.get(section) != current[section]]
        raise ValueError(f"source_identity: LeWM source/dependency drift in {changed}")


def tensor_state_digest(state: dict) -> str:
    """Stable identity including dtype/shape and buffers; works for BF16 and scalars."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(repr((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
