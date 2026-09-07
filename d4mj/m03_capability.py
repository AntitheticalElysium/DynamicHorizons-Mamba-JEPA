"""Sealed, evaluation-only capability gate for completed M0--M3 LeWM bundles.

The gate deliberately reuses the old Craftax exact-replay and all-action-fork
methodology without importing any legacy head or actor result into LeWM.  It
answers only questions the completed joint bundles can answer:

* do frozen CLS/projected exports retain critical visible state;
* do their observed and one-step generated successors preserve all-action
  semantic consequences; and
* is a deterministic successor compatible with one of the simulator's sampled
  modes, and does a valid four-frame prefix matter?

It does *not* train heads, a bridge, a policy, an actor, or a longer recursive
world.  ``m4_authorized`` is therefore always false in its output.

The existing ``artifacts/eda/replay.py`` is the authority for deterministic
support-v2 reconstruction.  Its legacy fork collectors encode a MAE feature and
select TRAIN roots, so this module constructs a fresh, probe-only DEV sidecar
instead of reusing those cached latents.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor, nn

from .config import Config, config_from_dict, recipe_digest
from .data import _sha256, atomic_manifest
from .diagnostics import binary_auc
from .lewm_config import LeWMConfig
from .sources import lewm_source_manifest
from .world_api import ModelBundle


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "d4mj_m03_capability_gate_v1"
SIDECAR_SCHEMA = "d4mj_m03_probe_sidecar_v1"
RUN_SCHEMA = "d4mj_m03_run_v1"
FEATURE_SCHEMA = "d4mj_m03_feature_cache_v1"
STAGE_SCHEMA = "d4mj_m03_stage_cache_v1"


@dataclass(frozen=True)
class M03Settings:
    """Fixed evaluation choices, independent of the two training recipes."""

    schema: str = SCHEMA
    seed: int = 20260907
    train_roots: int = 256
    dev_roots: int = 128
    terminal_tail_fraction: float = 0.5
    legacy_context: int = 16
    lewm_context: int = 4
    replay_checks: int = 4
    mode_samples: int = 4
    encode_batch: int = 16
    probe_hidden: int = 128
    probe_steps: int = 200
    probe_batch: int = 256
    probe_learning_rate: float = 1e-3
    probe_weight_decay: float = 1e-4
    bootstrap_draws: int = 1000
    minimum_positive: int = 10
    minimum_negative: int = 10
    ridge: float = 1.0

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("m03_settings: unsupported schema")
        if self.train_roots < 2 or self.dev_roots < 2:
            raise ValueError("m03_settings: both splits need at least two roots")
        if (self.legacy_context, self.lewm_context) != (16, 4):
            raise ValueError("m03_settings: invalid context lengths")
        if not 0.0 <= self.terminal_tail_fraction <= 1.0:
            raise ValueError("m03_settings: tail fraction must lie in [0, 1]")
        for name in ("replay_checks", "mode_samples", "encode_batch", "probe_hidden", "probe_steps",
                     "probe_batch", "bootstrap_draws", "minimum_positive", "minimum_negative"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"m03_settings: {name} must be a positive integer")
        for name in ("probe_learning_rate", "probe_weight_decay", "ridge"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"m03_settings: {name} must be finite and nonnegative")
        if self.probe_learning_rate <= 0 or self.ridge <= 0:
            raise ValueError("m03_settings: learning rate and ridge must be positive")


STATIC_CONTINUOUS = (
    "health", "food", "drink", "energy",
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)
STATIC_BINARY = (
    "front_lava", "front_water", "front_tree", "front_resource", "front_ripe_plant",
    "near_table", "near_furnace", "adjacent_mob", "sleeping",
    "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)
OUTCOME_BINARY = ("death", "damage", "reward_positive", "achievement_event", "inventory_changed", "tile_changed")
# A run with only easy, common labels is evidence of neither retention nor
# generated-state semantics.  These are intentionally the full visible/control
# suite rather than a post-hoc subset selected after looking at an arm.
CRITICAL_STATIC_CONTINUOUS = STATIC_CONTINUOUS
CRITICAL_STATIC_BINARY = STATIC_BINARY
CRITICAL_OUTCOME_BINARY = OUTCOME_BINARY
CRITICAL_SUCCESSOR_CONTINUOUS = STATIC_CONTINUOUS
CRITICAL_SUCCESSOR_BINARY = STATIC_BINARY


def _atomic_torch_save(path: Path, value: Any) -> None:
    """Publish a sidecar once; never overwrite a completed gate payload."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"m03_output: refusing to replace {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(path: Path, value: dict[str, Any]) -> None:
    """Publish an immutable JSON record exactly once."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"m03_output: refusing to replace {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_identity(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _run_contract(*, raw_checkpoint: Path, tc_checkpoint: Path, dataset: Path, settings: M03Settings,
                  device: str, include_direct: bool, structural_smoke: bool) -> dict[str, Any]:
    """Identity of every immutable input needed to resume safely."""

    contract: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "evaluator_source": _input_identity(Path(__file__)),
        "replay_source": _input_identity(ROOT / "artifacts/eda/replay.py"),
        "settings": asdict(settings),
        "dataset": _input_identity(dataset),
        "raw_checkpoint": _input_identity(raw_checkpoint),
        "tc_checkpoint": _input_identity(tc_checkpoint),
        "execution": {"device": device, "structural_smoke": structural_smoke,
                      "direct_anchors_included": include_direct},
    }
    if include_direct:
        report = ROOT / "artifacts/eda/capacity6k/n64d16_s1/training_report.json"
        contract["direct_anchors"] = {
            "encoder": _input_identity(report.parent / "encoder_006000.pt"),
            "attention": _input_identity(ROOT / "artifacts/eda/v2_direct_attention/world_020000.pt"),
            "mamba": _input_identity(ROOT / "artifacts/eda/v2_direct_mamba/world_020000.pt"),
        }
    return contract


def _open_run(output: Path, contract: dict[str, Any]) -> str:
    """Create, or safely reopen, one immutable-input M03 run directory."""

    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)
    run_path = output / "run.json"
    digest = _sha(contract)
    if run_path.exists():
        stored = json.loads(run_path.read_text())
        if stored.get("schema") != RUN_SCHEMA or stored.get("contract_sha256") != digest or stored.get("contract") != contract:
            raise ValueError("m03_resume: output directory belongs to different inputs or settings")
    elif any(output.iterdir()):
        raise FileExistsError("m03_resume: nonempty output lacks a run contract")
    else:
        _atomic_json_save(run_path, {"schema": RUN_SCHEMA, "contract": contract, "contract_sha256": digest})
        _atomic_json_save(output / "settings.json", contract["settings"])
    return digest


def _write_status(output: Path, stage: str, **details: Any) -> None:
    """Mutable progress pointer; immutable sidecars/caches remain the authority."""

    atomic_manifest(Path(output) / "status.json", {"schema": RUN_SCHEMA, "stage": stage,
                                                     "updated_unix": time.time(), **details})


def _array(value, *, dtype=None) -> np.ndarray:
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_replay():
    """Import the audited support-v2 replay lazily, keeping JAX out of normal imports."""

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from artifacts.eda import replay

    return replay


def _state_scalars(state) -> np.ndarray:
    inv = state.inventory
    return np.asarray(
        [state.player_health, state.player_food, state.player_drink, state.player_energy,
         inv.wood, inv.stone, inv.coal, inv.iron, inv.diamond, inv.sapling,
         inv.wood_pickaxe, inv.stone_pickaxe, inv.iron_pickaxe,
         inv.wood_sword, inv.stone_sword, inv.iron_sword],
        dtype=np.float32,
    )


def _state_binary_labels(state) -> np.ndarray:
    """Visible-state and source-derived action-prerequisite labels.

    The predicates mirror Craftax-Classic ``game_logic.py``.  They deliberately
    describe the *pre-state* ability for an action to have its intended effect;
    the all-action outcome labels are separate and are never reverse-engineered
    from the evaluated successor.
    """

    from craftax.craftax_classic.constants import BlockType, CLOSE_BLOCKS, DIRECTIONS, SOLID_BLOCKS

    board = _array(state.map)
    mobs = _array(state.mob_map).astype(bool)
    position = _array(state.player_position, dtype=np.int64)
    direction = int(_array(state.player_direction))
    shape = np.asarray(board.shape, dtype=np.int64)

    def inside(point: np.ndarray) -> bool:
        return bool(np.all(point >= 0) and np.all(point < shape))

    def tile(point: np.ndarray) -> int:
        return int(board[tuple(point)]) if inside(point) else int(BlockType.OUT_OF_BOUNDS.value)

    def mob(point: np.ndarray) -> bool:
        return bool(mobs[tuple(point)]) if inside(point) else False

    front = position + _array(DIRECTIONS[direction], dtype=np.int64)
    front_tile = tile(front)
    front_in_bounds = inside(front)
    front_open = front_in_bounds and not mob(front)
    solid = {int(value) for value in _array(SOLID_BLOCKS)}
    wall_front = front_tile in solid
    near = lambda block: any(tile(position + _array(offset, dtype=np.int64)) == int(block)
                             for offset in _array(CLOSE_BLOCKS))
    inv = state.inventory

    # Movement is an action precondition, not a claim that it has a useful outcome.
    def valid_move(action: int) -> bool:
        point = position + _array(DIRECTIONS[action], dtype=np.int64)
        # Craftax permits entry into lava; it is a valid (and potentially fatal)
        # movement consequence, not a disabled action.
        return inside(point) and not mob(point) and tile(point) not in solid

    do_enabled = front_in_bounds and (
        mob(front)
        or front_tile in {
            int(BlockType.TREE.value), int(BlockType.WATER.value), int(BlockType.RIPE_PLANT.value),
            # Grass has a stochastic sapling-collection consequence.
            int(BlockType.GRASS.value),
        }
        or (front_tile in {int(BlockType.STONE.value), int(BlockType.COAL.value)} and bool(inv.wood_pickaxe))
        or (front_tile == int(BlockType.IRON.value) and bool(inv.stone_pickaxe))
        or (front_tile == int(BlockType.DIAMOND.value) and bool(inv.iron_pickaxe))
    )
    place_ok = front_open
    table = place_ok and not wall_front and int(inv.wood) >= 2
    furnace = place_ok and not wall_front and int(inv.stone) > 0
    stone = place_ok and int(inv.stone) > 0 and (front_tile == int(BlockType.WATER.value) or not wall_front)
    plant = place_ok and front_tile == int(BlockType.GRASS.value) and int(inv.sapling) > 0
    at_table, at_furnace = near(BlockType.CRAFTING_TABLE.value), near(BlockType.FURNACE.value)
    wood_pick = at_table and int(inv.wood) >= 1
    stone_pick = at_table and int(inv.wood) >= 1 and int(inv.stone) >= 1
    iron_pick = at_table and at_furnace and int(inv.wood) >= 1 and int(inv.stone) >= 1 and int(inv.iron) >= 1 and int(inv.coal) >= 1
    wood_sword = at_table and int(inv.wood) >= 1
    stone_sword = at_table and int(inv.wood) >= 1 and int(inv.stone) >= 1
    iron_sword = at_table and at_furnace and int(inv.wood) >= 1 and int(inv.stone) >= 1 and int(inv.iron) >= 1 and int(inv.coal) >= 1

    awake = not bool(state.is_sleeping)
    return np.asarray(
        [front_tile == int(BlockType.LAVA.value), front_tile == int(BlockType.WATER.value),
         front_tile == int(BlockType.TREE.value), front_tile in {int(BlockType.STONE.value), int(BlockType.COAL.value), int(BlockType.IRON.value), int(BlockType.DIAMOND.value)},
         front_tile == int(BlockType.RIPE_PLANT.value), at_table, at_furnace, mob(front), bool(state.is_sleeping),
         awake and valid_move(1), awake and valid_move(2), awake and valid_move(3), awake and valid_move(4), awake and do_enabled,
         awake and int(state.player_energy) < 9, awake and stone, awake and table, awake and furnace, awake and plant,
         awake and wood_pick, awake and stone_pick, awake and iron_pick, awake and wood_sword, awake and stone_sword, awake and iron_sword],
        dtype=np.bool_,
    )


def _action_outcomes(root, successor, reward: float, *, root_binary: np.ndarray, successor_binary: np.ndarray, replay) -> np.ndarray:
    root_values, next_values = _state_scalars(root), _state_scalars(successor)
    tile_changed = bool(np.any(_array(root.map) != _array(successor.map)))
    return np.asarray(
        [replay.is_dead(successor), next_values[0] < root_values[0], reward > 0,
         bool(_array(successor.achievements).sum() > _array(root.achievements).sum()),
         bool(np.any(next_values[4:] != root_values[4:])), tile_changed],
        dtype=np.bool_,
    )


def _candidate_rows(replay, split: str, count: int, settings: M03Settings) -> list[dict[str, Any]]:
    """Select distinct episodes with a declared terminal-tail/broad mixture."""

    manifest = replay.manifest()
    terminal, ordinary = [], []
    for shard_index, record in enumerate(manifest["shards"]):
        payload = torch.load(replay.STORE / record["file"], weights_only=False, mmap=True)
        for slot, fields in enumerate(payload["episodes"]):
            if fields["split"] != split:
                continue
            steps = len(fields["actions_taken"])
            if steps <= settings.legacy_context:
                continue
            row = {"shard": shard_index, "slot": slot, "steps": steps, "episode_id": fields["episode_id"]}
            if bool(fields["terminated"][-1]):
                terminal.append(row)
            else:
                ordinary.append(row)
        del payload
    wanted_terminal = round(count * settings.terminal_tail_fraction)
    if len(terminal) < wanted_terminal or len(ordinary) < count - wanted_terminal:
        raise RuntimeError(f"m03_coverage: {split} lacks required terminal/broad episode roots")
    rng = np.random.default_rng(settings.seed + (1 if split == "train" else 2))
    selected = []
    for kind, pool, take in (("terminal_tail", terminal, wanted_terminal), ("ordinary", ordinary, count-wanted_terminal)):
        choices = rng.choice(len(pool), size=take, replace=False)
        for pick in choices.tolist():
            row = dict(pool[pick])
            row["stratum"] = kind
            row["t"] = row["steps"] - 1 if kind == "terminal_tail" else int(rng.integers(settings.legacy_context - 1, row["steps"] - 1))
            selected.append(row)
    rng.shuffle(selected)
    return selected


def _stack_rows(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any]:
    if not rows:
        raise ValueError("m03_sidecar: no rows to stack")
    tensor = lambda key: torch.from_numpy(np.stack([row[key] for row in rows]))
    return {
        "context": tensor("context"), "past_actions": tensor("past_actions").long(), "successors": tensor("successors"),
        "root_continuous": tensor("root_continuous").float(), "root_binary": tensor("root_binary").bool(),
        "next_continuous": tensor("next_continuous").float(), "next_binary": tensor("next_binary").bool(),
        "outcomes": tensor("outcomes").bool(), "modes": tensor("modes").bool(),
        "episode": torch.tensor([row["episode"] for row in rows], dtype=torch.long),
        "time": torch.tensor([row["time"] for row in rows], dtype=torch.long),
        "stratum": [row["stratum"] for row in rows], "episode_id": [row["episode_id"] for row in rows],
        "rows": [{key: row[key] for key in ("shard", "slot", "time", "episode_id", "stratum")} for row in rows],
        "names": names,
    }


def build_sidecar(dataset: Path, output: Path, settings: M03Settings) -> dict[str, Any]:
    """Create the fresh probe-only exact-replay sidecar and its integrity report."""

    replay = _load_replay()
    import jax

    dataset = Path(dataset).resolve()
    expected_dataset = (replay.STORE / "manifest.json").resolve()
    if dataset != expected_dataset:
        raise ValueError(f"m03_dataset: exact replay is bound to {expected_dataset}, not {dataset}")
    final_output = Path(output)
    if final_output.exists():
        raise FileExistsError(f"m03_output: sidecar path already exists: {final_output}")
    # A crash cannot leave a directory that looks like a finished sidecar.  The
    # temporary directory is intentionally retained for forensic inspection; a
    # later invocation recomputes rather than trusting partial bytes.
    output = final_output.parent / f".{final_output.name}.building-{uuid4().hex}"
    output.mkdir(parents=True)
    _, _, _, step_fn, frame_fn = replay.env_and_render()
    selected = {"train": _candidate_rows(replay, "train", settings.train_roots, settings),
                "dev": _candidate_rows(replay, "dev", settings.dev_roots, settings)}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    pixel_error = 0
    started = time.time()
    for split, selections in selected.items():
        for number, selected_row in enumerate(selections):
            state = replay.advance_to(selected_row["shard"], selected_row["slot"], selected_row["t"])
            fields = replay.episode_fields(selected_row["shard"], selected_row["slot"])
            frames = _array(fields["observations"])
            context = frames[selected_row["t"] - settings.legacy_context + 1:selected_row["t"] + 1]
            actions = _array(fields["actions_taken"])[selected_row["t"] - settings.legacy_context + 1:selected_row["t"]]
            rendered = _array(frame_fn(state))
            error = int(np.abs(rendered.astype(np.int16) - context[-1].astype(np.int16)).max())
            pixel_error = max(pixel_error, error)
            if error:
                raise RuntimeError(f"m03_replay: root pixel mismatch at {selected_row['episode_id']}:{selected_row['t']}: {error}")
            root_continuous, root_binary = _state_scalars(state), _state_binary_labels(state)
            primary_key = jax.random.PRNGKey(settings.seed)
            # Stable, replay-address-derived common randomness.  It is deliberately
            # independent of the loop order and shared by every one of the 17 forks.
            for field in (selected_row["shard"], selected_row["slot"], selected_row["t"]):
                primary_key = jax.random.fold_in(primary_key, int(field))
            successors, next_continuous, next_binary, outcomes, modes = [], [], [], [], []
            for action in range(17):
                _, nxt, reward, _, _ = step_fn(primary_key, state, action)
                successor_binary = _state_binary_labels(nxt)
                successors.append(_array(frame_fn(nxt), dtype=np.uint8))
                next_continuous.append(_state_scalars(nxt))
                next_binary.append(successor_binary)
                outcomes.append(_action_outcomes(state, nxt, float(reward), root_binary=root_binary,
                                                successor_binary=successor_binary, replay=replay))
                sampled = []
                for mode in range(settings.mode_samples):
                    key = jax.random.fold_in(primary_key, 1 + mode)
                    _, stochastic_next, stochastic_reward, _, _ = step_fn(key, state, action)
                    sampled.append(_action_outcomes(state, stochastic_next, float(stochastic_reward),
                                                    root_binary=root_binary,
                                                    successor_binary=_state_binary_labels(stochastic_next), replay=replay))
                modes.append(np.stack(sampled))
            rows_by_split[split].append({
                "context": context, "past_actions": actions, "successors": np.stack(successors),
                "root_continuous": root_continuous, "root_binary": root_binary,
                "next_continuous": np.stack(next_continuous), "next_binary": np.stack(next_binary),
                "outcomes": np.stack(outcomes), "modes": np.stack(modes),
                "episode": number, "time": selected_row["t"], "episode_id": selected_row["episode_id"],
                "shard": selected_row["shard"], "slot": selected_row["slot"], "stratum": selected_row["stratum"],
            })
    replay_checks = []
    for split, selections in selected.items():
        for row in selections[:settings.replay_checks]:
            replay_checks.append(replay.verify(row["shard"], row["slot"], checks=settings.replay_checks))
    payload = {
        "schema": SIDECAR_SCHEMA, "probe_only": True, "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": _sha256(Path(dataset)), "settings": asdict(settings),
        "continuous_names": STATIC_CONTINUOUS, "binary_names": STATIC_BINARY, "outcome_names": OUTCOME_BINARY,
        "splits": {split: _stack_rows(rows, STATIC_BINARY) for split, rows in rows_by_split.items()},
    }
    sidecar = output / "sidecar.probe_only.pt"
    _atomic_torch_save(sidecar, payload)
    coverage = {}
    for split, values in payload["splits"].items():
        outcomes = values["outcomes"]
        coverage[split] = {name: {"positive": int(outcomes[..., index].sum()),
                                  "negative": int((~outcomes[..., index]).sum())}
                           for index, name in enumerate(OUTCOME_BINARY)}
    manifest = {
        "schema": SIDECAR_SCHEMA, "probe_only": True, "sidecar": {"path": str(sidecar.resolve()), "sha256": _sha256(sidecar)},
        "dataset": payload["dataset"], "dataset_sha256": payload["dataset_sha256"], "settings": asdict(settings),
        "replay": {"root_pixel_max_abs": pixel_error, "status": "pass" if pixel_error == 0 else "fail",
                   "checked_roots": settings.train_roots + settings.dev_roots,
                   "trajectory_check_max_abs": max(replay_checks, default=0),
                   "trajectory_checked_episodes": len(replay_checks)},
        "coverage": coverage, "seconds": time.time() - started,
    }
    atomic_manifest(output / "manifest.json", manifest)
    os.replace(output, final_output)
    return payload


def _load_or_build_sidecar(dataset: Path, output: Path, settings: M03Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse only a fully published sidecar whose bytes and contract still match."""

    output, dataset = Path(output), Path(dataset).resolve()
    sidecar_path, manifest_path = output / "sidecar.probe_only.pt", output / "manifest.json"
    if not output.exists():
        payload = build_sidecar(dataset, output, settings)
        return payload, json.loads((output / "manifest.json").read_text())
    if not sidecar_path.exists() or not manifest_path.exists():
        raise RuntimeError("m03_resume: sidecar directory is incomplete; refusing to trust partial replay bytes")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": SIDECAR_SCHEMA, "probe_only": True, "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset), "settings": asdict(settings),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"m03_resume: sidecar {name} differs from this run contract")
    if manifest.get("replay", {}).get("status") != "pass" or manifest["replay"].get("root_pixel_max_abs") != 0:
        raise ValueError("m03_resume: cached sidecar did not pass exact root replay")
    if _sha256(sidecar_path) != manifest.get("sidecar", {}).get("sha256"):
        raise ValueError("m03_resume: cached sidecar hash mismatch")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != SIDECAR_SCHEMA or payload.get("probe_only") is not True
            or payload.get("dataset_sha256") != expected["dataset_sha256"]
            or payload.get("settings") != expected["settings"]):
        raise ValueError("m03_resume: cached sidecar payload contract mismatch")
    return payload, manifest


def _current_source_with_ieee_delta(recorded: dict) -> tuple[dict, dict]:
    """Allow precisely the declared execution delta; reject every other drift."""

    current = lewm_source_manifest()
    adjusted = json.loads(json.dumps(recorded))
    before = adjusted.get("execution", {}).get("triton_f32_default")
    after = current.get("execution", {}).get("triton_f32_default")
    if before != "unset" or after != "ieee":
        raise ValueError(f"m03_precision: expected recorded unset -> evaluation ieee, found {before!r} -> {after!r}")
    adjusted["execution"]["triton_f32_default"] = after
    if adjusted != current:
        changed = [key for key in current if adjusted.get(key) != current[key]]
        raise ValueError(f"m03_source_identity: unapproved LeWM source/environment drift in {changed}")
    return current, {"triton_f32_default": {"recorded": before, "evaluation": after}}


def load_m03_bundle(path: Path, *, device: str, dataset_sha256: str) -> tuple[ModelBundle, dict, dict]:
    """Load a frozen joint bundle under the explicitly recorded IEEE evaluation delta."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "d4mj_lewm_bundle_v2" or payload.get("phase") != "joint":
        raise ValueError("m03_checkpoint: expected a LeWM joint bundle")
    config = config_from_dict(payload["config"])
    if not isinstance(config, LeWMConfig) or payload.get("recipe_id") != recipe_digest(config):
        raise ValueError("m03_checkpoint: recipe identity mismatch")
    expected = {"joint_complete": True, "trained_recursive_depth": 0, "validated_recursive_depth": 0,
                "readout_trained": False, "m4_authorized": False}
    if payload.get("capabilities") != expected:
        raise ValueError("m03_checkpoint: requires a completed M0-M3-only checkpoint")
    if payload.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("m03_checkpoint: checkpoint and exact-replay manifest bytes differ")
    current_source, source_delta = _current_source_with_ieee_delta(payload["sources"])
    config = replace(config, runtime=replace(config.runtime, device=device),
                     dynamics=replace(config.dynamics, backend="triton" if device == "cuda" else "reference"))
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"], strict=True)
    bundle.world.load_state_dict(payload["modules"]["world"], strict=True)
    bundle.encoder.freeze()
    bundle.world.requires_grad_(False)
    return bundle.eval(), payload, {"current": current_source, "allowed_delta": source_delta}


def _load_legacy_cpu(path: Path, config: Config, **objects) -> dict:
    """Read historical CUDA snapshots on either CPU or CUDA without weakening identity checks."""

    from .checkpoint import FORMAT
    from .sources import verify_sources

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("m03_anchor: unexpected legacy checkpoint format")
    stored, requested = dict(payload["config"]), asdict(config)
    # The historical encoder predates this optional field; preserve the loader's
    # compatibility rule while retaining an exact requested configuration check.
    if "align_weight" not in stored and requested.get("align_weight", 0.0) == 0.0:
        stored["align_weight"] = 0.0
    if stored != requested:
        raise ValueError("m03_anchor: checkpoint config differs from declared anchor config")
    verify_sources(payload["sources"], config)
    for name, target in objects.items():
        target.load_state_dict(payload["modules"][name])
    return payload


def _legacy_anchor(name: str, *, device: str) -> tuple[ModelBundle, dict]:
    """Load the common MAE export plus a declared historical Direct world anchor."""

    from .representation import Decoder, Encoder
    from .transition import World

    if name not in ("direct_attention", "direct_mamba"):
        raise ValueError("m03_anchor: unknown legacy anchor")
    mixer = "attention" if name.endswith("attention") else "mamba"
    # Model placement is an evaluation concern.  The archived config identity
    # remains CUDA and is checked before its CPU-mapped tensor payload is loaded.
    base = replace(Config(), n_latents=64, d_bottleneck=16, device=device)
    stored_base = replace(base, device="cuda")
    report_path = ROOT / "artifacts/eda/capacity6k/n64d16_s1/training_report.json"
    encoder_path = report_path.parent / "encoder_006000.pt"
    report = json.loads(report_path.read_text())
    encoder = Encoder(base).to(device)
    _load_legacy_cpu(encoder_path, replace(stored_base, batch=report["batch"], seed=report["seed"]),
                     part0=encoder, part1=Decoder(base))
    config = replace(base, transition="direct", time_mixer=mixer)
    world_path = ROOT / f"artifacts/eda/v2_direct_{mixer}/world_020000.pt"
    world = World(config).to(device)
    _load_legacy_cpu(world_path, replace(config, device="cuda"), part0=world)
    bundle = ModelBundle.from_models(config, encoder.eval(), world.eval()).eval()
    identity = {"name": name, "encoder": {"path": str(encoder_path.resolve()), "sha256": _sha256(encoder_path)},
                "world": {"path": str(world_path.resolve()), "sha256": _sha256(world_path)},
                "archived_config": asdict(replace(config, device="cuda")),
                "evaluation_device": device}
    return bundle, identity


def _to_device(values: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in values.items()}


def _load_or_encode_features(output: Path, *, arm: str, split: str, identity: dict[str, Any],
                             sidecar_sha256: str, encode) -> tuple[dict[str, Tensor], str]:
    """Cache one arm/split encoding atomically and validate it before reuse."""

    directory = Path(output) / "features"
    directory.mkdir(exist_ok=True)
    path = directory / f"{arm}.{split}.pt"
    manifest_path = directory / f"{arm}.{split}.manifest.json"
    metadata = {"arm": arm, "split": split, "identity": identity, "sidecar_sha256": sidecar_sha256}
    metadata_sha256 = _sha(metadata)

    if path.exists():
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("schema") != FEATURE_SCHEMA or manifest.get("metadata_sha256") != metadata_sha256
                    or manifest.get("sha256") != _sha256(path)):
                raise ValueError(f"m03_resume: cached {arm}/{split} features have a different identity or hash")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != FEATURE_SCHEMA or _sha(payload.get("metadata")) != metadata_sha256:
            raise ValueError(f"m03_resume: cached {arm}/{split} feature payload contract mismatch")
        features = payload.get("features")
        if not isinstance(features, dict) or not features or not all(isinstance(value, Tensor) and value.device.type == "cpu"
                                                                      for value in features.values()):
            raise ValueError(f"m03_resume: cached {arm}/{split} features are invalid")
        # A process may have stopped after atomic tensor publication and before
        # publishing its small manifest.  The validated tensor is safe to adopt.
        if not manifest_path.exists():
            _atomic_json_save(manifest_path, {"schema": FEATURE_SCHEMA, "sha256": _sha256(path),
                                              "metadata_sha256": metadata_sha256})
        return features, _sha256(manifest_path)

    if manifest_path.exists():
        raise RuntimeError(f"m03_resume: cached {arm}/{split} manifest lacks its tensor payload")
    features = encode()
    if not features or not all(isinstance(value, Tensor) and value.device.type == "cpu" for value in features.values()):
        raise ValueError(f"m03_cache: {arm}/{split} encoder returned non-CPU tensors")
    _atomic_torch_save(path, {"schema": FEATURE_SCHEMA, "metadata": metadata, "features": features})
    _atomic_json_save(manifest_path, {"schema": FEATURE_SCHEMA, "sha256": _sha256(path),
                                      "metadata_sha256": metadata_sha256})
    return features, _sha256(manifest_path)


def _load_or_compute_stage(output: Path, name: str, metadata: dict[str, Any], compute) -> dict[str, Any]:
    """Persist expensive, JSON-only probe stages so resumption never refits them."""

    directory = Path(output) / "stages"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.json"
    metadata_sha256 = _sha(metadata)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("schema") != STAGE_SCHEMA or record.get("metadata_sha256") != metadata_sha256:
            raise ValueError(f"m03_resume: cached {name} stage belongs to different feature bytes")
        if not isinstance(record.get("result"), dict):
            raise ValueError(f"m03_resume: cached {name} stage has no result")
        return record["result"]
    result = compute()
    _atomic_json_save(path, {"schema": STAGE_SCHEMA, "metadata_sha256": metadata_sha256,
                             "metadata": metadata, "result": result})
    return result


@torch.inference_mode()
def _encode_lewm(bundle: ModelBundle, values: dict[str, Any], settings: M03Settings) -> dict[str, Tensor]:
    context, actions, successors = values["context"], values["past_actions"], values["successors"]
    projected, cls, observed_successor, generated, reset_generated = [], [], [], [], []
    device = bundle.device
    all_actions = torch.arange(bundle.n_actions, device=device, dtype=torch.long)
    for start in range(0, len(context), settings.encode_batch):
        end = min(len(context), start + settings.encode_batch)
        frame = context[start:end, -settings.lewm_context:].to(device)
        past = actions[start:end, -settings.lewm_context + 1:].to(device)
        z, c = bundle.encoder.projected_and_cls(frame)
        state = bundle.prefill(z, past)
        branches = bundle.repeat_state(state, bundle.n_actions)
        action = all_actions.repeat(end - start)[:, None]
        predicted, _ = bundle.advance(branches, action)
        generated.append(predicted.latent[:, 0].reshape(end-start, bundle.n_actions, -1).cpu())
        reset = bundle.start(z[:, -1:])
        reset = bundle.repeat_state(reset, bundle.n_actions)
        reset, _ = bundle.advance(reset, action)
        reset_generated.append(reset.latent[:, 0].reshape(end-start, bundle.n_actions, -1).cpu())
        flat_successor = successors[start:end].reshape(-1, 1, *successors.shape[-3:]).to(device)
        obs_z, _ = bundle.encoder.projected_and_cls(flat_successor)
        observed_successor.append(obs_z[:, 0, 0].reshape(end-start, bundle.n_actions, -1).cpu())
        projected.append(z[:, -1, 0].cpu())
        cls.append(c[:, -1].cpu())
    return {"projected": torch.cat(projected), "cls": torch.cat(cls),
            "observed_successor": torch.cat(observed_successor), "generated_successor": torch.cat(generated),
            "reset_generated_successor": torch.cat(reset_generated)}


@torch.inference_mode()
def _encode_legacy(bundle: ModelBundle, values: dict[str, Any], settings: M03Settings) -> dict[str, Tensor]:
    context, actions, successors = values["context"], values["past_actions"], values["successors"]
    root, observed_successor, generated = [], [], []
    device = bundle.device
    all_actions = torch.arange(bundle.n_actions, device=device, dtype=torch.long)
    for start in range(0, len(context), settings.encode_batch):
        end = min(len(context), start + settings.encode_batch)
        frame, past = context[start:end].to(device), actions[start:end].to(device)
        # Unlike joint LeWM, the legacy MAE encoder has an observation-memory
        # cache.  Reconstruct the real state through its public observation path
        # so each all-action branch can legitimately consume successor pixels.
        state = None
        for offset in range(settings.legacy_context):
            incoming = None if offset == 0 else past[:, offset-1:offset]
            state, _ = bundle.observe(state, incoming, frame[:, offset:offset+1])
        branches = bundle.repeat_state(state, bundle.n_actions)
        action = all_actions.repeat(end-start)[:, None]
        prediction, _ = bundle.advance(branches, action)
        generated.append(bundle.world_state(prediction).latent[:, 0].flatten(1).reshape(end-start, bundle.n_actions, -1).cpu())
        observed, _ = bundle.observe(branches, action, successors[start:end].reshape(-1, 1, *successors.shape[-3:]).to(device))
        observed_successor.append(bundle.world_state(observed).latent[:, 0].flatten(1).reshape(end-start, bundle.n_actions, -1).cpu())
        root.append(bundle.world_state(state).latent[:, 0].flatten(1).cpu())
    return {"projected": torch.cat(root), "observed_successor": torch.cat(observed_successor),
            "generated_successor": torch.cat(generated)}


def _standardize(train: Tensor, dev: Tensor) -> tuple[Tensor, Tensor]:
    mean, scale = train.mean(0, keepdim=True), train.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (train - mean) / scale, (dev - mean) / scale


def _fit_probe_many(train_x: Tensor, train_y: Tensor, evaluation_x: dict[str, Tensor], settings: M03Settings, *,
                    hidden: bool, binary: bool) -> dict[str, Tensor]:
    """Fit one sealed probe once and apply it to one or more held-out inputs.

    This matters for observed-vs-generated attribution: both distributions are
    evaluated by literally the same trained decoder and train-derived affine map,
    never by separately re-fit probes that merely share hyperparameters.
    """

    device = train_x.device
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / scale
    seed = settings.seed + (19 if hidden else 7) + (100 if binary else 0)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == "cuda" else []):
        torch.manual_seed(seed)
        model = (nn.Sequential(nn.Linear(train_x.shape[1], settings.probe_hidden), nn.GELU(),
                               nn.Linear(settings.probe_hidden, train_y.shape[1])) if hidden
                 else nn.Linear(train_x.shape[1], train_y.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.probe_learning_rate,
                                  weight_decay=settings.probe_weight_decay)
    if binary:
        positive = train_y.sum(0)
        negative = len(train_y) - positive
        weight = (negative / positive.clamp_min(1)).clamp(max=100.0)
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(settings.probe_steps):
        index = torch.randint(len(train_x), (min(settings.probe_batch, len(train_x)),), device=device, generator=rng)
        prediction = model(train_x[index])
        loss = (nn.functional.binary_cross_entropy_with_logits(prediction, train_y[index], pos_weight=weight)
                if binary else (prediction-train_y[index]).square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        return {name: model.eval()((value.to(device) - mean) / scale).cpu()
                for name, value in evaluation_x.items()}


def _fit_probe(train_x: Tensor, train_y: Tensor, dev_x: Tensor, settings: M03Settings, *, hidden: bool,
               binary: bool) -> Tensor:
    """Single-evaluation wrapper for fixed static probes."""

    return _fit_probe_many(train_x, train_y, {"dev": dev_x}, settings, hidden=hidden, binary=binary)["dev"]


def _ridge_predict(train_x: Tensor, train_y: Tensor, dev_x: Tensor, ridge: float) -> Tensor:
    """Linear raw-pixel reference via the dual ridge form (safe when p >> n)."""

    train_x, dev_x = _standardize(train_x.float(), dev_x.float())
    y = train_y.float()
    kernel = train_x @ train_x.T
    solution = torch.linalg.solve(kernel + ridge * torch.eye(len(train_x)), y)
    return (dev_x @ train_x.T @ solution).cpu()


def _root_bootstrap(values: Tensor, roots: Tensor, fn, *, draws: int, seed: int) -> tuple[float | None, list[float] | None]:
    keys = roots.unique(sorted=True)
    all_indices = torch.arange(len(roots))
    point = fn(all_indices)
    if point is None:
        return None, None
    groups = [torch.where(roots == key)[0] for key in keys]
    rng = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(draws):
        sampled = torch.randint(len(groups), (len(groups),), generator=rng)
        index = torch.cat([groups[number] for number in sampled])
        value = fn(index)
        if value is not None:
            samples.append(value)
    if len(samples) < .95 * draws:
        return point, None
    bounds = torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return point, bounds


def _expanded_roots(roots: Tensor, target: Tensor) -> Tensor:
    """Repeat root IDs over non-root target dimensions, never over sibling roots."""

    if target.shape[0] != len(roots):
        raise ValueError("m03_metrics: target/root leading dimensions differ")
    return roots.reshape(len(roots), *([1] * (target.ndim - 1))).expand_as(target).reshape(-1)


def _binary_metrics(logits: Tensor, truth: Tensor, roots: Tensor, names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    names = tuple(names)
    report, macro = {}, []
    for index, name in enumerate(names):
        target_shape = truth[..., index]
        score, target = logits[..., index].reshape(-1), target_shape.reshape(-1).bool()
        root = _expanded_roots(roots, target_shape)
        positive, negative = int(target.sum()), int((~target).sum())
        if positive < settings.minimum_positive or negative < settings.minimum_negative:
            report[name] = {"status": "insufficient_coverage", "positive": positive, "negative": negative}
            continue
        def auc(rows):
            return binary_auc(score[rows], target[rows])
        point, interval = _root_bootstrap(score, root, auc, draws=settings.bootstrap_draws, seed=settings.seed+index)
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage", "positive": positive,
                        "negative": negative, "auc": point, "interval": interval}
        if point is not None:
            macro.append(point)
    return {"targets": report, "macro_auc": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names)}


def _regression_metrics(prediction: Tensor, truth: Tensor, roots: Tensor, names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    names = tuple(names)
    report, values = {}, []
    for index, name in enumerate(names):
        target, predicted = truth[:, index], prediction[:, index]
        total_variance = float(target.var(unbiased=False))
        if total_variance <= 1e-12:
            report[name] = {"status": "insufficient_variation", "variance": total_variance}
            continue
        denominator = ((target-target.mean()).square().sum()).clamp_min(1e-12)
        r2 = float(1 - (predicted-target).square().sum()/denominator)
        def score(rows):
            selected = target[rows]
            denom = ((selected-selected.mean()).square().sum()).clamp_min(1e-12)
            return float(1 - (predicted[rows]-selected).square().sum()/denom)
        _, interval = _root_bootstrap(predicted, roots, score, draws=settings.bootstrap_draws, seed=settings.seed+400+index)
        report[name] = {"r2": r2, "interval": interval, "mae": float((predicted-target).abs().mean()),
                        "variance": total_variance,
                        "status": "measured" if interval is not None else "insufficient_coverage"}
        if interval is not None:
            values.append(r2)
    return {"targets": report, "mean_r2": float(np.mean(values)) if values else None,
            "supported_targets": len(values), "total_targets": len(names)}


def _paired_binary_difference(left_logits: Tensor, right_logits: Tensor, truth: Tensor, roots: Tensor,
                              names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    """Paired root-bootstrap AUC(left) - AUC(right), with common fork rows."""

    names = tuple(names)
    report, macro = {}, []
    for index, name in enumerate(names):
        target_shape = truth[..., index]
        left, right, target = (left_logits[..., index].reshape(-1), right_logits[..., index].reshape(-1),
                               target_shape.reshape(-1).bool())
        root = _expanded_roots(roots, target_shape)
        positive, negative = int(target.sum()), int((~target).sum())
        if positive < settings.minimum_positive or negative < settings.minimum_negative:
            report[name] = {"status": "insufficient_coverage", "positive": positive, "negative": negative}
            continue

        def difference(rows: Tensor):
            left_auc, right_auc = binary_auc(left[rows], target[rows]), binary_auc(right[rows], target[rows])
            return None if left_auc is None or right_auc is None else left_auc - right_auc

        point, interval = _root_bootstrap(left, root, difference, draws=settings.bootstrap_draws,
                                          seed=settings.seed + 2000 + index)
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage",
                        "positive": positive, "negative": negative, "auc_difference": point, "interval": interval}
        if interval is not None and point is not None:
            macro.append(point)
    return {"targets": report, "mean_auc_difference": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names), "direction": "left_minus_right"}


def _paired_regression_difference(left: Tensor, right: Tensor, truth: Tensor, roots: Tensor,
                                  names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    """Paired root-bootstrap R²(left) - R²(right) for successor scalar semantics."""

    names = tuple(names)
    report, macro = {}, []
    for index, name in enumerate(names):
        target, left_pred, right_pred = truth[:, index], left[:, index], right[:, index]
        variance = float(target.var(unbiased=False))
        if variance <= 1e-12:
            report[name] = {"status": "insufficient_variation", "variance": variance}
            continue

        def r2(prediction: Tensor, rows: Tensor) -> float | None:
            selected = target[rows]
            denominator = ((selected - selected.mean()).square().sum()).clamp_min(1e-12)
            if float(denominator) <= 1e-12:
                return None
            return float(1 - (prediction[rows] - selected).square().sum() / denominator)

        def difference(rows: Tensor):
            left_r2, right_r2 = r2(left_pred, rows), r2(right_pred, rows)
            return None if left_r2 is None or right_r2 is None else left_r2 - right_r2

        point, interval = _root_bootstrap(left_pred, roots, difference, draws=settings.bootstrap_draws,
                                          seed=settings.seed + 2100 + index)
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage",
                        "variance": variance, "r2_difference": point, "interval": interval}
        if interval is not None and point is not None:
            macro.append(point)
    return {"targets": report, "mean_r2_difference": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names), "direction": "left_minus_right"}


def _static_report(train: dict[str, Tensor], dev: dict[str, Tensor], features: dict[str, dict[str, Tensor]],
                   settings: M03Settings, *, device: str) -> dict[str, Any]:
    """Fit identical static probes for every representation without DEV selection."""

    output: dict[str, Any] = {"continuous_names": STATIC_CONTINUOUS, "binary_names": STATIC_BINARY, "sources": {}}
    train_binary, dev_binary = train["root_binary"].float().to(device), dev["root_binary"].bool()
    train_cont, dev_cont = train["root_continuous"].float().to(device), dev["root_continuous"].float()
    roots = dev["episode"]
    for name, pair in features.items():
        train_x, dev_x = pair["train"].float().to(device), pair["dev"].float().to(device)
        source = {"linear": {}, "mlp": {}}
        for hidden, family in ((False, "linear"), (True, "mlp")):
            binary = _fit_probe(train_x, train_binary, dev_x, settings, hidden=hidden, binary=True)
            continuous = _fit_probe(train_x, train_cont, dev_x, settings, hidden=hidden, binary=False)
            source[family] = {"binary": _binary_metrics(binary, dev_binary, roots, STATIC_BINARY, settings),
                              "continuous": _regression_metrics(continuous, dev_cont, roots, STATIC_CONTINUOUS, settings)}
        output["sources"][name] = source
    # Trivial and raw-pixel references use a fixed ridge, never an MLP selected on DEV.
    t = train["time"].float()[:, None]
    time_train = torch.cat((torch.ones_like(t), t, t.square()), 1).to(device)
    d = dev["time"].float()[:, None]
    time_dev = torch.cat((torch.ones_like(d), d, d.square()), 1).to(device)
    pixel_train, pixel_dev = train["context"][:, -1].flatten(1).to(device), dev["context"][:, -1].flatten(1).to(device)
    output["controls"] = {}
    for name, x_train, x_dev in (("timestep", time_train, time_dev), ("pixels_linear_ridge", pixel_train, pixel_dev)):
        binary = _ridge_predict(x_train, train_binary, x_dev, settings.ridge)
        continuous = _ridge_predict(x_train, train_cont, x_dev, settings.ridge)
        output["controls"][name] = {"binary": _binary_metrics(binary, dev_binary, roots, STATIC_BINARY, settings),
                                     "continuous": _regression_metrics(continuous, dev_cont, roots, STATIC_CONTINUOUS, settings)}
    return output


def _one_hot_actions(count: int, *, device: str) -> Tensor:
    return torch.eye(17, device=device).repeat(count, 1)


def _choice_summary(logits: Tensor, truth: Tensor, roots: Tensor, index: int, *, minimize: bool,
                    settings: M03Settings) -> dict[str, Any]:
    """Does a generated all-action ranking select the safer/better actual fork?"""

    scores, labels = logits[..., index], truth[..., index].bool()
    eligible = labels.any(1) & (~labels).any(1)
    # ``death`` is good when false; ``reward_positive`` is good when true.
    desired = ~labels if minimize else labels

    def selected_score(rows: Tensor) -> float | None:
        valid = eligible[rows]
        if not bool(valid.any()):
            return None
        selected = scores[rows].argmin(1) if minimize else scores[rows].argmax(1)
        return float(desired[rows, selected][valid].float().mean())

    def chance_score(rows: Tensor) -> float | None:
        valid = eligible[rows]
        if not bool(valid.any()):
            return None
        return float(desired[rows][valid].float().mean())

    point, interval = _root_bootstrap(scores, roots, selected_score, draws=settings.bootstrap_draws,
                                      seed=settings.seed + 1300 + index + int(minimize))
    chance, _ = _root_bootstrap(scores, roots, chance_score, draws=settings.bootstrap_draws,
                                seed=settings.seed + 1400 + index + int(minimize))
    return {"opportunity_roots": int(eligible.sum()), "selected_actual_rate": point,
            "selected_actual_interval": interval, "uniform_action_rate": chance,
            "lift_over_uniform": None if point is None or chance is None else point - chance,
            "status": "measured" if interval is not None else "insufficient_coverage"}


def _equivalence_summary(logits: Tensor, truth: Tensor, roots: Tensor, settings: M03Settings) -> dict[str, Any]:
    """Check that predicted action effects separate actually distinct outcomes."""

    first, second = torch.triu_indices(17, 17, offset=1)
    equal = (truth[:, first] == truth[:, second]).all(-1)
    distance = (logits.sigmoid()[:, first] - logits.sigmoid()[:, second]).abs().mean(-1)

    def contrast(rows: Tensor) -> float | None:
        selected_equal, selected_distance = equal[rows], distance[rows]
        if not bool(selected_equal.any()) or not bool((~selected_equal).any()):
            return None
        return float(selected_distance[~selected_equal].mean() - selected_distance[selected_equal].mean())

    point, interval = _root_bootstrap(distance, roots, contrast, draws=settings.bootstrap_draws,
                                      seed=settings.seed + 1500)
    return {"equal_outcome_pairs": int(equal.sum()), "different_outcome_pairs": int((~equal).sum()),
            "non_equivalent_minus_equivalent_probability_distance": point, "interval": interval,
            "status": "measured" if interval is not None else "insufficient_coverage"}


def _mode_summary(logits: Tensor, truth: Tensor, modes: Tensor, roots: Tensor, settings: M03Settings) -> dict[str, Any]:
    """A deterministic discriminative score may be nearer to one valid outcome mode.

    These are squared score distances, deliberately not Brier scores: class-weighted
    BCE logits are useful for discrimination/ranking but not calibrated probabilities.
    """

    probability = logits.sigmoid()
    primary_distance = (probability - truth.float()).square().mean(-1)
    # [root, action, mode, label] -> pick the simulator outcome mode nearest
    # to the deterministic prediction.  This is diagnostic, never likelihood.
    closest_distance = (probability[:, :, None] - modes.float()).square().mean(-1).min(-1).values

    def mean(value: Tensor):
        def score(rows: Tensor) -> float:
            return float(value[rows].mean())
        return _root_bootstrap(value, roots, score, draws=settings.bootstrap_draws, seed=settings.seed + 1600)

    primary, primary_interval = mean(primary_distance)
    closest, closest_interval = mean(closest_distance)
    varying = (modes.max(2).values != modes.min(2).values).any(-1)
    return {"sampled_modes_per_state_action": int(modes.shape[2]),
            "primary_outcome_mean_squared_score_distance": primary, "primary_outcome_interval": primary_interval,
            "nearest_sampled_mode_mean_squared_score_distance": closest, "nearest_sampled_mode_interval": closest_interval,
            "varying_state_action_pairs": int(varying.sum()), "total_state_action_pairs": int(varying.numel()),
            "status": "advisory_only_deterministic_model"}


def _context_summary(generated: Tensor, reset: Tensor, truth: Tensor) -> dict[str, Any]:
    """Measure whether a four-frame prefix materially changes all-action predictions."""

    delta = (generated.sigmoid() - reset.sigmoid()).abs()
    death = OUTCOME_BINARY.index("death")
    reward = OUTCOME_BINARY.index("reward_positive")
    return {"mean_probability_abs_delta": float(delta.mean()),
            "death_ranking_changed_roots": int((generated[..., death].argmin(1) != reset[..., death].argmin(1)).sum()),
            "reward_ranking_changed_roots": int((generated[..., reward].argmax(1) != reset[..., reward].argmax(1)).sum()),
            "roots": int(len(truth)), "status": "measured"}


def _paired_semantic_differences(predictions: dict[str, dict[str, dict[str, dict[str, Tensor]]]],
                                 dev: dict[str, Tensor], settings: M03Settings) -> dict[str, Any]:
    """Predeclared paired comparisons; no arm is selected after inspecting DEV."""

    roots = dev["episode"]
    outcome_truth = dev["outcomes"]
    successor_binary_truth = dev["next_binary"].flatten(0, 1)
    successor_continuous_truth = dev["next_continuous"].flatten(0, 1)
    fork_roots = _expanded_roots(roots, outcome_truth[..., 0])
    requested = [
        ("raw_generated_minus_raw_observed", "raw", "generated", "raw", "observed"),
        ("tc_generated_minus_tc_observed", "tc", "generated", "tc", "observed"),
        ("raw_generated_minus_direct_attention_generated", "raw", "generated", "direct_attention", "generated"),
        ("raw_generated_minus_direct_mamba_generated", "raw", "generated", "direct_mamba", "generated"),
        ("tc_generated_minus_direct_attention_generated", "tc", "generated", "direct_attention", "generated"),
        ("tc_generated_minus_direct_mamba_generated", "tc", "generated", "direct_mamba", "generated"),
        ("tc_generated_minus_raw_generated", "tc", "generated", "raw", "generated"),
    ]
    output: dict[str, Any] = {"unit": "replayed_root_episode", "comparisons": {}}
    for name, left_arm, left_state, right_arm, right_state in requested:
        if left_arm not in predictions or right_arm not in predictions:
            output["comparisons"][name] = {"status": "unavailable_missing_anchor"}
            continue
        families = {}
        for family in ("linear", "mlp"):
            left, right = predictions[left_arm][family], predictions[right_arm][family]
            families[family] = {
                "coarse_outcomes": _paired_binary_difference(
                    left["coarse_outcomes"][left_state], right["coarse_outcomes"][right_state],
                    outcome_truth, roots, OUTCOME_BINARY, settings),
                "successor_binary_semantics": _paired_binary_difference(
                    left["successor_binary"][left_state], right["successor_binary"][right_state],
                    successor_binary_truth, fork_roots, STATIC_BINARY, settings),
                "successor_continuous_semantics": _paired_regression_difference(
                    left["successor_continuous"][left_state], right["successor_continuous"][right_state],
                    successor_continuous_truth, fork_roots, STATIC_CONTINUOUS, settings),
            }
        output["comparisons"][name] = {"status": "measured", "families": families}
    return output


def _outcome_report(train: dict[str, Tensor], dev: dict[str, Tensor], features: dict[str, dict[str, Tensor]],
                    settings: M03Settings, *, device: str) -> dict[str, Any]:
    """All-action consequence and rich successor-semantic transfer, fitted on TRAIN only."""

    output: dict[str, Any] = {"outcome_names": OUTCOME_BINARY, "successor_continuous_names": STATIC_CONTINUOUS,
                              "successor_binary_names": STATIC_BINARY, "arms": {}}
    train_truth, dev_truth = train["outcomes"].reshape(-1, len(OUTCOME_BINARY)).float().to(device), dev["outcomes"]
    root = dev["episode"]
    fork_roots = _expanded_roots(root, dev_truth[..., 0])
    train_next_binary = train["next_binary"].flatten(0, 1).float().to(device)
    dev_next_binary = dev["next_binary"].flatten(0, 1).bool()
    train_next_continuous = train["next_continuous"].flatten(0, 1).float().to(device)
    dev_next_continuous = dev["next_continuous"].flatten(0, 1).float()
    action_train, action_dev = _one_hot_actions(len(train["outcomes"]), device=device), _one_hot_actions(len(dev["outcomes"]), device=device)
    rng = torch.Generator(device=device).manual_seed(settings.seed + 900)
    shuffled_train = action_train[torch.randperm(len(action_train), generator=rng, device=device)]
    shuffled_dev = action_dev[torch.randperm(len(action_dev), generator=rng, device=device)]
    predictions: dict[str, dict[str, dict[str, dict[str, Tensor]]]] = {}
    for arm, values in features.items():
        train_root = values["train_projected"].float().to(device).repeat_interleave(17, 0)
        dev_root = values["dev_projected"].float().to(device).repeat_interleave(17, 0)
        train_observed = values["train_observed_successor"].flatten(0, 1).float().to(device)
        dev_observed = values["dev_observed_successor"].flatten(0, 1).float().to(device)
        dev_generated = values["dev_generated_successor"].flatten(0, 1).float().to(device)
        dev_reset = values.get("dev_reset_generated_successor")
        if dev_reset is not None:
            dev_reset = dev_reset.flatten(0, 1).float().to(device)
        rows = {}
        predictions[arm] = {}
        for hidden, family in ((False, "linear"), (True, "mlp")):
            transfer_input = {"observed": torch.cat((dev_observed, action_dev), 1),
                              "generated": torch.cat((dev_generated, action_dev), 1)}
            if dev_reset is not None:
                transfer_input["reset"] = torch.cat((dev_reset, action_dev), 1)
            # The observed successor decoder is fitted exactly once.  Generated,
            # observed, and reset-context latent states share those parameters and
            # train-derived normalization statistics.
            transfer = _fit_probe_many(torch.cat((train_observed, action_train), 1), train_truth,
                                       transfer_input, settings, hidden=hidden, binary=True)
            observed_model, generated_model = transfer["observed"], transfer["generated"]
            successor_binary_transfer = _fit_probe_many(
                torch.cat((train_observed, action_train), 1), train_next_binary, transfer_input,
                settings, hidden=hidden, binary=True)
            successor_continuous_transfer = _fit_probe_many(
                torch.cat((train_observed, action_train), 1), train_next_continuous, transfer_input,
                settings, hidden=hidden, binary=False)
            root_model = _fit_probe(torch.cat((train_root, action_train), 1), train_truth,
                                    torch.cat((dev_root, action_dev), 1), settings, hidden=hidden, binary=True)
            action_model = _fit_probe(action_train, train_truth, action_dev, settings, hidden=hidden, binary=True)
            shuffled_model = _fit_probe(torch.cat((train_root, shuffled_train), 1), train_truth,
                                        torch.cat((dev_root, shuffled_dev), 1), settings, hidden=hidden, binary=True)
            family_report = {
                "observed_successor": _binary_metrics(observed_model, dev_truth, root, OUTCOME_BINARY, settings),
                "generated_successor": _binary_metrics(generated_model, dev_truth, root, OUTCOME_BINARY, settings),
                "root_action": _binary_metrics(root_model, dev_truth, root, OUTCOME_BINARY, settings),
                "action_only": _binary_metrics(action_model, dev_truth, root, OUTCOME_BINARY, settings),
                "shuffled_action": _binary_metrics(shuffled_model, dev_truth, root, OUTCOME_BINARY, settings),
                "fatal_safe_ranking": _choice_summary(generated_model.reshape(-1, 17, len(OUTCOME_BINARY)), dev_truth,
                                                       root, OUTCOME_BINARY.index("death"), minimize=True, settings=settings),
                "reward_ranking": _choice_summary(generated_model.reshape(-1, 17, len(OUTCOME_BINARY)), dev_truth,
                                                  root, OUTCOME_BINARY.index("reward_positive"), minimize=False, settings=settings),
                "action_effect_equivalence": _equivalence_summary(generated_model.reshape(-1, 17, len(OUTCOME_BINARY)),
                                                                    dev_truth, root, settings),
                "stochastic_mode_compatibility": _mode_summary(generated_model.reshape(-1, 17, len(OUTCOME_BINARY)),
                                                               dev_truth, dev["modes"], root, settings),
                "successor_semantics": {
                    "binary": {
                        "observed_successor": _binary_metrics(successor_binary_transfer["observed"], dev_next_binary,
                                                               fork_roots, STATIC_BINARY, settings),
                        "generated_successor": _binary_metrics(successor_binary_transfer["generated"], dev_next_binary,
                                                                fork_roots, STATIC_BINARY, settings),
                    },
                    "continuous": {
                        "observed_successor": _regression_metrics(successor_continuous_transfer["observed"], dev_next_continuous,
                                                                    fork_roots, STATIC_CONTINUOUS, settings),
                        "generated_successor": _regression_metrics(successor_continuous_transfer["generated"], dev_next_continuous,
                                                                     fork_roots, STATIC_CONTINUOUS, settings),
                    },
                },
            }
            if dev_reset is not None:
                reset_model = transfer["reset"]
                family_report["reset_context_generated_successor"] = _binary_metrics(reset_model, dev_truth, root, OUTCOME_BINARY, settings)
                family_report["context_sensitivity"] = _context_summary(
                    generated_model.reshape(-1, 17, len(OUTCOME_BINARY)),
                    reset_model.reshape(-1, 17, len(OUTCOME_BINARY)), dev_truth)
                family_report["successor_semantics"]["binary"]["reset_context_generated_successor"] = _binary_metrics(
                    successor_binary_transfer["reset"], dev_next_binary, fork_roots, STATIC_BINARY, settings)
                family_report["successor_semantics"]["continuous"]["reset_context_generated_successor"] = _regression_metrics(
                    successor_continuous_transfer["reset"], dev_next_continuous, fork_roots, STATIC_CONTINUOUS, settings)
            predictions[arm][family] = {
                "coarse_outcomes": {"observed": observed_model, "generated": generated_model},
                "successor_binary": {"observed": successor_binary_transfer["observed"],
                                     "generated": successor_binary_transfer["generated"]},
                "successor_continuous": {"observed": successor_continuous_transfer["observed"],
                                         "generated": successor_continuous_transfer["generated"]},
            }
            rows[family] = family_report
        output["arms"][arm] = rows
    output["paired_differences"] = _paired_semantic_differences(predictions, dev, settings)
    return output


def _advisory_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    """Expose the non-gating context/mode checks without duplicating score data."""

    return {"mode_interpretation": "nearest sampled mode is descriptive, not a likelihood or promotion criterion",
            "context_interpretation": "reset-context differences test prefix use, not policy quality",
            "locations": "per-arm/per-probe-family entries under all_action_one_step"}


def _unsupported_targets(section: dict[str, Any], names: Iterable[str]) -> list[str]:
    return [name for name in names if section.get("targets", {}).get(name, {}).get("status") != "measured"]


def _critical_coverage(static: dict[str, Any], outcomes: dict[str, Any]) -> dict[str, Any]:
    """Coverage is a gate condition, not a footnote beneath a macro average."""

    missing: list[str] = []
    for source in ("raw_projected", "tc_projected"):
        for family in ("linear", "mlp"):
            branch = static["sources"].get(source, {}).get(family, {})
            for name in _unsupported_targets(branch.get("continuous", {}), CRITICAL_STATIC_CONTINUOUS):
                missing.append(f"static:{source}:{family}:continuous:{name}")
            for name in _unsupported_targets(branch.get("binary", {}), CRITICAL_STATIC_BINARY):
                missing.append(f"static:{source}:{family}:binary:{name}")
    for arm in ("raw", "tc"):
        for family in ("linear", "mlp"):
            branch = outcomes["arms"].get(arm, {}).get(family, {})
            for observed_or_generated in ("observed_successor", "generated_successor"):
                for name in _unsupported_targets(branch.get(observed_or_generated, {}), CRITICAL_OUTCOME_BINARY):
                    missing.append(f"outcome:{arm}:{family}:{observed_or_generated}:{name}")
            semantics = branch.get("successor_semantics", {})
            for observed_or_generated in ("observed_successor", "generated_successor"):
                for name in _unsupported_targets(semantics.get("continuous", {}).get(observed_or_generated, {}),
                                                 CRITICAL_SUCCESSOR_CONTINUOUS):
                    missing.append(f"successor_continuous:{arm}:{family}:{observed_or_generated}:{name}")
                for name in _unsupported_targets(semantics.get("binary", {}).get(observed_or_generated, {}),
                                                 CRITICAL_SUCCESSOR_BINARY):
                    missing.append(f"successor_binary:{arm}:{family}:{observed_or_generated}:{name}")
    return {"status": "adequate" if not missing else "insufficient_coverage", "missing": missing,
            "required": {"static_continuous": CRITICAL_STATIC_CONTINUOUS, "static_binary": CRITICAL_STATIC_BINARY,
                         "outcome_binary": CRITICAL_OUTCOME_BINARY,
                         "successor_continuous": CRITICAL_SUCCESSOR_CONTINUOUS,
                         "successor_binary": CRITICAL_SUCCESSOR_BINARY}}


def _decision(static: dict[str, Any], outcomes: dict[str, Any], sidecar: dict[str, Any], *,
              structural_smoke: bool) -> dict[str, Any]:
    """Conservative component verdict; sparse targets are never promoted to pass."""

    coverage = _critical_coverage(static, outcomes)
    enough = coverage["status"] == "adequate"
    status = "structural_smoke_not_a_result" if structural_smoke else ("measured" if enough else "insufficient_coverage")
    return {
        "m03_capability": status,
        "critical_coverage": coverage,
        "raw_status": "structural_smoke_not_a_result" if structural_smoke else ("measured" if enough else "insufficient_coverage"),
        "tc_status": "structural_smoke_not_a_result" if structural_smoke else ("measured" if enough else "insufficient_coverage"),
        "tc_promotion": "not_evaluated_by_scalar_gate",
        "m4_data_contract": "unreviewed",
        "m4_authorized": False,
        "scope": "M0-M3 frozen representation and one-step semantic diagnostics only",
    }


def run_gate(*, raw_checkpoint: Path, tc_checkpoint: Path, dataset: Path, output: Path,
             settings: M03Settings, device: str, include_direct: bool = True,
             structural_smoke: bool = False) -> dict[str, Any]:
    """Run or safely resume one complete M03 report for immutable declared inputs."""

    if device != "cuda" and not structural_smoke:
        raise RuntimeError("m03_runtime: full Raw/TC/Direct gate requires CUDA with IEEE Triton execution; CPU is smoke-only")
    if device != "cuda" and include_direct:
        raise RuntimeError("m03_runtime: Direct-M historical anchor requires CUDA; use --skip-direct only for a structural CPU smoke")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("m03_runtime: requested CUDA but no CUDA device is available")
    output = Path(output)
    contract = _run_contract(raw_checkpoint=raw_checkpoint, tc_checkpoint=tc_checkpoint, dataset=dataset,
                             settings=settings, device=device, include_direct=include_direct,
                             structural_smoke=structural_smoke)
    contract_sha256 = _open_run(output, contract)
    report_path = output / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("schema") != SCHEMA or report.get("run_contract_sha256") != contract_sha256:
            raise ValueError("m03_resume: final report belongs to different inputs or settings")
        return report

    _write_status(output, "opening", contract_sha256=contract_sha256)
    sidecar, sidecar_manifest = _load_or_build_sidecar(dataset, output / "sidecar", settings)
    sidecar_sha256 = sidecar_manifest["sidecar"]["sha256"]
    _write_status(output, "sidecar_ready", sidecar_sha256=sidecar_sha256)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    # Encode/cache one model at a time.  This makes each arm/split a safe resume
    # boundary and avoids holding Raw, TC, and both historical worlds in GPU RAM.
    feature_rows: dict[str, dict[str, Tensor]] = {}
    feature_manifests: dict[str, str] = {}
    parents: dict[str, Any] = {}
    for variant, path in (("raw", raw_checkpoint), ("tc", tc_checkpoint)):
        bundle, payload, source = load_m03_bundle(path, device=device, dataset_sha256=sidecar["dataset_sha256"])
        parents[variant] = {"path": str(Path(path).resolve()), "sha256": _sha256(path), "step": payload["step"],
                            "recipe_id": payload["recipe_id"], "source_evaluation": source}
        for split in ("train", "dev"):
            features, manifest_sha256 = _load_or_encode_features(
                output, arm=variant, split=split, identity=parents[variant], sidecar_sha256=sidecar_sha256,
                encode=lambda b=bundle, s=split: _encode_lewm(b, sidecar["splits"][s], settings))
            feature_rows.setdefault(variant, {})[split] = features
            feature_manifests[f"{variant}:{split}"] = manifest_sha256
            _write_status(output, "encoding", arm=variant, split=split, sidecar_sha256=sidecar_sha256)
        del bundle
        if device == "cuda":
            torch.cuda.empty_cache()

    if include_direct:
        for name in ("direct_attention", "direct_mamba"):
            bundle, parents[name] = _legacy_anchor(name, device=device)
            for split in ("train", "dev"):
                features, manifest_sha256 = _load_or_encode_features(
                    output, arm=name, split=split, identity=parents[name], sidecar_sha256=sidecar_sha256,
                    encode=lambda b=bundle, s=split: _encode_legacy(b, sidecar["splits"][s], settings))
                feature_rows.setdefault(name, {})[split] = features
                feature_manifests[f"{name}:{split}"] = manifest_sha256
                _write_status(output, "encoding", arm=name, split=split, sidecar_sha256=sidecar_sha256)
            del bundle
            torch.cuda.empty_cache()

    static_features: dict[str, dict[str, Tensor]] = {}
    for variant in ("raw", "tc"):
        static_features[f"{variant}_cls"] = {split: feature_rows[variant][split]["cls"] for split in ("train", "dev")}
        static_features[f"{variant}_projected"] = {split: feature_rows[variant][split]["projected"] for split in ("train", "dev")}
    for name in ("direct_attention", "direct_mamba") if include_direct else ():
        static_features[f"{name}_projected"] = {split: feature_rows[name][split]["projected"] for split in ("train", "dev")}
    static_metadata = {"sidecar_sha256": sidecar_sha256, "feature_manifests": feature_manifests,
                       "sources": tuple(static_features)}
    static = _load_or_compute_stage(
        output, "static_retention", static_metadata,
        lambda: _static_report(sidecar["splits"]["train"], sidecar["splits"]["dev"], static_features, settings, device=device))
    _write_status(output, "static_retention_ready", sidecar_sha256=sidecar_sha256)

    outcome_features = {}
    for variant in ("raw", "tc"):
        outcome_features[variant] = {
            "train_projected": feature_rows[variant]["train"]["projected"],
            "dev_projected": feature_rows[variant]["dev"]["projected"],
            "train_observed_successor": feature_rows[variant]["train"]["observed_successor"],
            "dev_observed_successor": feature_rows[variant]["dev"]["observed_successor"],
            "dev_generated_successor": feature_rows[variant]["dev"]["generated_successor"],
            "dev_reset_generated_successor": feature_rows[variant]["dev"]["reset_generated_successor"],
        }
    for name in ("direct_attention", "direct_mamba") if include_direct else ():
        outcome_features[name] = {
            "train_projected": feature_rows[name]["train"]["projected"],
            "dev_projected": feature_rows[name]["dev"]["projected"],
            "train_observed_successor": feature_rows[name]["train"]["observed_successor"],
            "dev_observed_successor": feature_rows[name]["dev"]["observed_successor"],
            "dev_generated_successor": feature_rows[name]["dev"]["generated_successor"],
        }
    outcome_metadata = {"sidecar_sha256": sidecar_sha256, "feature_manifests": feature_manifests,
                        "arms": tuple(outcome_features)}
    outcomes = _load_or_compute_stage(
        output, "all_action_one_step", outcome_metadata,
        lambda: _outcome_report(sidecar["splits"]["train"], sidecar["splits"]["dev"], outcome_features, settings, device=device))
    _write_status(output, "all_action_one_step_ready", sidecar_sha256=sidecar_sha256)
    advisory = _advisory_summary(outcomes)
    report = {
        "schema": SCHEMA, "run_contract_sha256": contract_sha256, "settings": asdict(settings), "parents": parents,
        "execution": {"device": device, "mode": "structural_smoke" if structural_smoke else "sealed_full_gate",
                      "direct_anchors_included": include_direct},
        "sidecar": sidecar_manifest,
        "static_retention": static, "all_action_one_step": outcomes, "advisory": advisory,
        "decision": _decision(static, outcomes, sidecar, structural_smoke=structural_smoke),
    }
    _atomic_json_save(report_path, report)
    _write_status(output, "complete", sidecar_sha256=sidecar_sha256, report_sha256=_sha256(report_path))
    return report


def _settings_for_smoke(settings: M03Settings) -> M03Settings:
    return replace(settings, train_roots=8, dev_roots=4, mode_samples=2, encode_batch=2,
                   probe_hidden=8, probe_steps=2, probe_batch=8, bootstrap_draws=16,
                   minimum_positive=1, minimum_negative=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt")
    parser.add_argument("--tc-checkpoint", type=Path, default=ROOT / "artifacts/lewm_gates_20260906/paired/tc/joint/step-010000.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/craftax_support_v2/manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-direct", action="store_true", help="debug-only; output cannot be a full historical-anchor gate")
    parser.add_argument("--smoke", action="store_true", help="tiny structural fixture, never a research result")
    args = parser.parse_args(argv)
    settings = _settings_for_smoke(M03Settings()) if args.smoke else M03Settings()
    try:
        report = run_gate(raw_checkpoint=args.raw_checkpoint, tc_checkpoint=args.tc_checkpoint,
                          dataset=args.dataset, output=args.out, settings=settings, device=args.device,
                          include_direct=not args.skip_direct, structural_smoke=args.smoke)
        print(json.dumps({"status": "complete", "report": str((args.out / "report.json").resolve()),
                          "m03_capability": report["decision"]["m03_capability"], "m4_authorized": False}))
        return 0
    except Exception as error:
        if args.out.exists():
            atomic_manifest(args.out / "failure.json", {"schema": SCHEMA, "status": "stopped", "reason": str(error),
                                                         "m4_authorized": False})
        print(json.dumps({"status": "stopped", "reason": str(error), "m4_authorized": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
