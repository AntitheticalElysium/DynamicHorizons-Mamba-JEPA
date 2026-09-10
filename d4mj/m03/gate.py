"""Sealed, evaluation-only capability gate for completed M0--M3 LeWM bundles.

The gate deliberately reuses the old Craftax exact-replay and all-action-fork
methodology without importing any legacy head or actor result into LeWM.  It
answers only questions the completed joint bundles can answer:

* do frozen CLS/projected exports retain critical visible state;
* do their observed and one-step generated successors preserve all-action
  semantic consequences; and
* is a deterministic successor compatible with one of the simulator's sampled
  modes, and does a valid four-frame prefix matter?

Direct is evaluated under its native 64-frame production prefix; LeWM receives
the final four frames from that same physical prefix.  The primary all-action
fork uses the stored episode step key and verifies that its logged action
reproduces the stored successor before any model is encoded.

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
from functools import lru_cache
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

from .cache import active, memoized, PROBE_SETTINGS, REPLAY_SETTINGS, content, resolve_payload
from ..config import Config, config_from_dict, recipe_digest
from ..data import _sha256, atomic_manifest
from ..diagnostics import binary_auc
from ..lewm_config import LeWMConfig
from ..sources import lewm_source_manifest
from ..world_api import ModelBundle


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "d4mj_m03_capability_gate"
SIDECAR_SCHEMA = "d4mj_m03_probe_sidecar"
RUN_SCHEMA = "d4mj_m03_run"
FEATURE_SCHEMA = "d4mj_m03_feature_cache"
STAGE_SCHEMA = "d4mj_m03_stage_cache"
DIRECT_INPUT_PROTOCOL = "native_prefix64_with_first_incoming_action_v1"
# The primary fork is the exact key that produced the recorded next observation.
# It makes the logged action an executable factual-replay assertion while still
# giving all 17 counterfactual actions the same environmental randomness.
PRIMARY_FORK_PROTOCOL = "recorded_step_key_v1"
# Additional stochastic samples are deliberately separate from the factual fork.
# They answer only the advisory mode question and can never be mistaken for the
# stored trajectory successor.
MODE_FORK_PROTOCOL = "derived_common_rng_v1"
# The fixed 384-root preflight observed 383 exact renders and one single-pixel,
# one-count renderer quantization difference.  State replay remains exact; this
# bounded image tolerance makes that renderer round-off visible rather than
# silently pretending it is zero.
REPLAY_PIXEL_TOLERANCE = 1
# This compares M03's adapter route with the independently preserved Direct
# production-evaluator route.  It is a floating-point comparison rather than
# simulator replay, hence a numerical (not pixel) tolerance.
DIRECT_NATIVE_PARITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class M03Settings:
    """Fixed evaluation choices, independent of the two training recipes."""

    schema: str = SCHEMA
    seed: int = 20260907
    train_roots: int = 256
    dev_roots: int = 128
    terminal_tail_fraction: float = 0.5
    # Direct's MAE encoder reaches 31 frames and its V2 dynamics cache reaches
    # 48.  The archived production evaluator uses 64-frame prefixes, so M03
    # retains 64 and routes only the final four to LeWM.
    direct_context: int = 64
    lewm_context: int = 4
    replay_checks: int = 4
    mode_samples: int = 4
    encode_batch: int = 16
    # Direct's native 64-frame successor encoding is materially larger than a
    # four-frame LeWM pass.  These are memory-only execution choices, sealed in
    # the contract; they do not alter either model's inputs or outputs.
    direct_encode_batch: int = 1
    direct_successor_batch: int = 4
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
        if (self.direct_context, self.lewm_context) != (64, 4):
            raise ValueError("m03_settings: invalid context lengths")
        if not 0.0 <= self.terminal_tail_fraction <= 1.0:
            raise ValueError("m03_settings: tail fraction must lie in [0, 1]")
        for name in ("replay_checks", "mode_samples", "encode_batch", "direct_encode_batch",
                     "direct_successor_batch", "probe_hidden", "probe_steps", "probe_batch",
                     "bootstrap_draws", "minimum_positive", "minimum_negative"):
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
                  device: str, include_direct: bool, structural_smoke: bool, include_history: bool = True) -> dict[str, Any]:
    """Identity of every immutable input needed to resume safely."""

    import importlib.util
    craftax_root = Path(importlib.util.find_spec("craftax").origin).parent

    contract: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "evaluator_source": _input_identity(Path(__file__)),
        "cache_source": _input_identity(Path(__file__).with_name("cache.py")),
        "cache_compatibility": active().compatibility_identity if active() is not None else None,
        "replay_source": _input_identity(ROOT / "artifacts/eda/replay.py"),
        "diagnostics_source": _input_identity(ROOT / "d4mj/m03/diagnostics.py"),
        "historical_source": _input_identity(ROOT / "d4mj/m03/history.py"),
        "environment_source": _input_identity(ROOT / "d4mj/env.py"),
        "evaluation_sources": [_input_identity(ROOT / p) for p in (
            "d4mj/m03/diagnostics.py", "d4mj/diagnostics.py", "d4mj/data.py", "d4mj/config.py",
            "d4mj/representation.py", "d4mj/transition.py", "d4mj/world_api.py", "artifacts/eda/legacy.py")],
        "historical_panels_included": include_history,
        "simulator_sources": [_input_identity(p) for p in sorted(craftax_root.rglob("*.py"))],
        "settings": asdict(settings),
        "dataset": _input_identity(dataset),
        "raw_checkpoint": _input_identity(raw_checkpoint),
        "tc_checkpoint": _input_identity(tc_checkpoint),
        "execution": {"device": device, "structural_smoke": structural_smoke,
                      "direct_anchors_included": include_direct},
    }
    if include_history:
        from .history import source_paths
        contract["historical_inputs"] = [_input_identity(p) for p in source_paths(ROOT)]
    if include_direct:
        report = ROOT / "artifacts/eda/capacity6k/n64d16_s1/training_report.json"
        contract["direct_anchors"] = {
            "encoder": _input_identity(report.parent / "encoder_006000.pt"),
            "encoder_report": _input_identity(report),
            "attention": _input_identity(ROOT / "artifacts/eda/v2_direct_attention/world_020000.pt"),
            "mamba": _input_identity(ROOT / "artifacts/eda/v2_direct_mamba/world_020000.pt"),
        }
    contract["replay_dependencies"] = _replay_dependencies(contract, settings)
    return contract


def _replay_dependencies(contract, settings):
    from .history import replay_seed, verify_reference
    from .diagnostics import state_features
    from .cache import implementation
    functions = (_state_scalars, _state_binary_labels, _action_outcomes, state_features,
                 replay_seed, verify_reference)
    return {'dataset':content(contract['dataset']), 'historical_inputs':content(contract.get('historical_inputs', [])),
            'replay_source':content(contract['replay_source']), 'environment':content(contract['environment_source']),
            'simulator':content(contract['simulator_sources']),
            'settings':{k:getattr(settings,k) for k in REPLAY_SETTINGS},
            'code':[implementation(fn) for fn in functions]}


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
    terminal, eligible = [], []
    for shard_index, record in enumerate(manifest["shards"]):
        payload = torch.load(replay.STORE / record["file"], weights_only=False, mmap=True)
        for slot, fields in enumerate(payload["episodes"]):
            if fields["split"] != split:
                continue
            steps = len(fields["actions_taken"])
            if steps < settings.direct_context:
                continue
            row = {"shard": shard_index, "slot": slot, "steps": steps, "episode_id": fields["episode_id"]}
            eligible.append(row)
            if bool(fields["terminated"][-1]):
                terminal.append(row)
        del payload
    wanted_terminal = round(count * settings.terminal_tail_fraction)
    if len(terminal) < wanted_terminal:
        raise RuntimeError(f"m03_coverage: {split} lacks required terminal-tail episode roots")
    rng = np.random.default_rng(settings.seed + (1 if split == "train" else 2))
    selected = []
    choices = rng.choice(len(terminal), size=wanted_terminal, replace=False)
    selected_keys = set()
    for pick in choices.tolist():
        row = dict(terminal[pick])
        row["stratum"], row["t"] = "terminal_tail", row["steps"] - 1
        selected.append(row)
        selected_keys.add((row["shard"], row["slot"]))
    # "Broad" means a non-terminal *time*, not a timeout-only episode.  It may
    # come from a terminal trajectory as long as that episode did not supply a
    # terminal-tail root; this maintains distinct bootstrap units.
    broad = [
        row for row in eligible
        if (row["shard"], row["slot"]) not in selected_keys
        and _broad_time_range(row["steps"], settings) is not None
    ]
    wanted_broad = count - wanted_terminal
    if len(broad) < wanted_broad:
        raise RuntimeError(f"m03_coverage: {split} lacks required distinct broad episode roots")
    choices = rng.choice(len(broad), size=wanted_broad, replace=False)
    for pick in choices.tolist():
        row = dict(broad[pick])
        row["stratum"] = "broad"
        low, high = _broad_time_range(row["steps"], settings)
        row["t"] = int(rng.integers(low, high))
        selected.append(row)
    rng.shuffle(selected)
    return selected


def _broad_time_range(steps: int, settings: M03Settings) -> tuple[int, int] | None:
    """Half-open root-time range with both a native prefix and successor.

    A 64-action episode can supply the terminal action at t=63, but it cannot
    supply a non-terminal root after the 64-frame Direct prefix.  Keeping that
    distinction explicit avoids passing an empty interval to ``rng.integers``.
    """

    low, high = settings.direct_context - 1, steps - 1
    return None if low >= high else (low, high)


def _stack_rows(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any]:
    if not rows:
        raise ValueError("m03_sidecar: no rows to stack")
    tensor = lambda key: torch.from_numpy(np.stack([row[key] for row in rows]))
    return {
        "context": tensor("context"), "past_actions": tensor("past_actions").long(), "successors": tensor("successors"),
        **{k: tensor(k).float() for k in ("oracle_visible", "oracle_timing", "oracle_full_state")},
        "root_continuous": tensor("root_continuous").float(), "root_binary": tensor("root_binary").bool(),
        "next_continuous": tensor("next_continuous").float(), "next_binary": tensor("next_binary").bool(),
        "outcomes": tensor("outcomes").bool(), "modes": tensor("modes").bool(),
        "factual_action": torch.tensor([row["factual_action"] for row in rows], dtype=torch.long),
        "factual_max_abs": torch.tensor([row["factual_max_abs"] for row in rows], dtype=torch.long),
        "direct_first_action": torch.tensor([row["direct_first_action"] for row in rows], dtype=torch.long),
        "episode": torch.tensor([row["episode"] for row in rows], dtype=torch.long),
        "time": torch.tensor([row["time"] for row in rows], dtype=torch.long),
        "stratum": [row["stratum"] for row in rows], "episode_id": [row["episode_id"] for row in rows],
        "rows": [{key: row[key] for key in ("shard", "slot", "time", "episode_id", "stratum")}
                 | {"factual_action": row["factual_action"], "factual_max_abs": row["factual_max_abs"],
                    "direct_first_action": row["direct_first_action"]}
                 for row in rows],
        "names": names,
    }


def _recorded_step_key(replay, row: dict[str, Any]):
    """The exact environment key for the logged outgoing action at a root."""

    _, step_keys = replay._slot_keys(int(row["shard"]), int(row["slot"]))
    return step_keys[int(row["t"])]


def _derived_mode_key(jax, row: dict[str, Any], mode: int, settings: M03Settings):
    """A deterministic fresh key for advisory stochastic-mode samples only."""

    key = jax.random.PRNGKey(settings.seed)
    for field in (row["shard"], row["slot"], row["t"], mode):
        key = jax.random.fold_in(key, int(field))
    return key


def _direct_first_incoming_action(actions: np.ndarray, context_start: int) -> int:
    """Incoming action for the first frame of a cropped Direct prefix.

    ``past_actions`` carries the 63 outgoing actions between 64 retained frames.
    Direct also consumes the action that produced the first retained frame; it is
    BOS only when that frame is the true episode start.
    """

    return 17 if context_start == 0 else int(actions[context_start - 1])


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
    pixel_error, pixel_nonzero = 0, 0
    factual_error, factual_nonzero = 0, 0
    started = time.time()
    for split, selections in selected.items():
        for number, selected_row in enumerate(selections):
            state = replay.advance_to(selected_row["shard"], selected_row["slot"], selected_row["t"])
            fields = replay.episode_fields(selected_row["shard"], selected_row["slot"])
            frames = _array(fields["observations"])
            all_actions = _array(fields["actions_taken"])
            context_start = selected_row["t"] - settings.direct_context + 1
            context = frames[context_start:selected_row["t"] + 1]
            actions = all_actions[context_start:selected_row["t"]]
            direct_first_action = _direct_first_incoming_action(all_actions, context_start)
            rendered = _array(frame_fn(state))
            delta = np.abs(rendered.astype(np.int16) - context[-1].astype(np.int16))
            error = int(delta.max())
            pixel_error = max(pixel_error, error)
            pixel_nonzero += int((delta > 0).sum())
            if error > REPLAY_PIXEL_TOLERANCE:
                raise RuntimeError(f"m03_replay: root pixel mismatch at {selected_row['episode_id']}:{selected_row['t']}: {error}")
            root_continuous, root_binary = _state_scalars(state), _state_binary_labels(state)
            primary_key = _recorded_step_key(replay, selected_row)
            factual_action = int(all_actions[selected_row["t"]])
            _, factual_next, _, _, _ = step_fn(primary_key, state, factual_action)
            factual_frame = _array(frame_fn(factual_next), dtype=np.uint8)
            expected_frame = frames[selected_row["t"] + 1]
            factual_delta = np.abs(factual_frame.astype(np.int16) - expected_frame.astype(np.int16))
            factual_max_abs = int(factual_delta.max())
            factual_error = max(factual_error, factual_max_abs)
            factual_nonzero += int((factual_delta > 0).sum())
            if factual_max_abs > REPLAY_PIXEL_TOLERANCE:
                raise RuntimeError(
                    "m03_factual_replay: logged action did not reproduce its stored successor at "
                    f"{selected_row['episode_id']}:{selected_row['t']}: {factual_max_abs}"
                )
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
                    key = _derived_mode_key(jax, selected_row, mode, settings)
                    _, stochastic_next, stochastic_reward, _, _ = step_fn(key, state, action)
                    sampled.append(_action_outcomes(state, stochastic_next, float(stochastic_reward),
                                                    root_binary=root_binary,
                                                    successor_binary=_state_binary_labels(stochastic_next), replay=replay))
                modes.append(np.stack(sampled))
            from .diagnostics import state_features
            rows_by_split[split].append({
                **state_features(state),
                "context": context, "past_actions": actions, "successors": np.stack(successors),
                "root_continuous": root_continuous, "root_binary": root_binary,
                "next_continuous": np.stack(next_continuous), "next_binary": np.stack(next_binary),
                "outcomes": np.stack(outcomes), "modes": np.stack(modes),
                "factual_action": factual_action, "factual_max_abs": factual_max_abs,
                "direct_first_action": direct_first_action,
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
        "direct_input_protocol": DIRECT_INPUT_PROTOCOL,
        "primary_fork_protocol": PRIMARY_FORK_PROTOCOL, "mode_fork_protocol": MODE_FORK_PROTOCOL,
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
        "direct_input_protocol": DIRECT_INPUT_PROTOCOL,
        "replay": {"root_pixel_max_abs": pixel_error, "root_pixel_nonzero_elements": pixel_nonzero,
                   "root_pixel_tolerance": REPLAY_PIXEL_TOLERANCE,
                   "status": "pass" if pixel_error <= REPLAY_PIXEL_TOLERANCE else "fail",
                   "checked_roots": settings.train_roots + settings.dev_roots,
                   "trajectory_check_max_abs": max(replay_checks, default=0),
                   "trajectory_checked_episodes": len(replay_checks)},
        "fork": {"primary_protocol": PRIMARY_FORK_PROTOCOL, "mode_protocol": MODE_FORK_PROTOCOL,
                 "factual_pixel_max_abs": factual_error, "factual_pixel_nonzero_elements": factual_nonzero,
                 "factual_pixel_tolerance": REPLAY_PIXEL_TOLERANCE,
                 "factual_roots_checked": settings.train_roots + settings.dev_roots,
                 "status": "pass" if factual_error <= REPLAY_PIXEL_TOLERANCE else "fail"},
        "coverage": coverage, "seconds": time.time() - started,
    }
    atomic_manifest(output / "manifest.json", manifest)
    os.replace(output, final_output)
    return payload


def _load_or_build_sidecar(dataset: Path, output: Path, settings: M03Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse only a fully published sidecar whose bytes and contract still match."""

    output, dataset = Path(output), Path(dataset).resolve()
    sidecar_path, manifest_path = output / "sidecar.probe_only.pt", output / "manifest.json"
    cache = active()
    replay_dependencies = None
    if cache is not None:
        parent = json.loads((output.parent/'run.json').read_text())['contract']
        replay_dependencies = dict(parent['replay_dependencies'])
        replay_dependencies.pop('historical_inputs',None)
        key, _ = cache.key('primary_replay',replay_dependencies)
        row = cache.db.execute('SELECT external FROM nodes WHERE key=?',(key,)).fetchone()
        source = Path(row[0]).parent if row and row[0] else cache.imports.get('sidecar')
        if not output.exists() and source:
            if row: cache.read(key)  # Integrity before hard-linking the raw object.
            output.mkdir(parents=True)
            for name in ('sidecar.probe_only.pt','manifest.json'):
                os.link(Path(source)/name, output/name)
    if not output.exists():
        build_sidecar(dataset, output, settings)
    if not sidecar_path.exists() or not manifest_path.exists():
        raise RuntimeError("m03_resume: sidecar directory is incomplete; refusing to trust partial replay bytes")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": SIDECAR_SCHEMA, "probe_only": True, "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset), "settings": asdict(settings),
        "direct_input_protocol": DIRECT_INPUT_PROTOCOL,
    }
    for name, value in expected.items():
        equal = ({k:manifest.get(name, {}).get(k) for k in REPLAY_SETTINGS} == {k:value[k] for k in REPLAY_SETTINGS}
                 if name == 'settings' else manifest.get(name) == value)
        if not equal:
            raise ValueError(f"m03_resume: sidecar {name} differs from this run contract")
    if (manifest.get("replay", {}).get("status") != "pass"
            or manifest["replay"].get("root_pixel_tolerance") != REPLAY_PIXEL_TOLERANCE
            or manifest["replay"].get("root_pixel_max_abs", float("inf")) > REPLAY_PIXEL_TOLERANCE):
        raise ValueError("m03_resume: cached sidecar did not pass exact root replay")
    if (manifest.get("fork", {}).get("status") != "pass"
            or manifest["fork"].get("primary_protocol") != PRIMARY_FORK_PROTOCOL
            or manifest["fork"].get("mode_protocol") != MODE_FORK_PROTOCOL
            or manifest["fork"].get("factual_pixel_tolerance") != REPLAY_PIXEL_TOLERANCE
            or manifest["fork"].get("factual_pixel_max_abs", float("inf")) > REPLAY_PIXEL_TOLERANCE):
        raise ValueError("m03_resume: cached sidecar did not pass factual all-action replay")
    if _sha256(sidecar_path) != manifest.get("sidecar", {}).get("sha256"):
        raise ValueError("m03_resume: cached sidecar hash mismatch")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != SIDECAR_SCHEMA or payload.get("probe_only") is not True
            or payload.get("dataset_sha256") != expected["dataset_sha256"]
            or {k:payload.get("settings", {}).get(k) for k in REPLAY_SETTINGS} != {k:expected["settings"][k] for k in REPLAY_SETTINGS}
            or payload.get("direct_input_protocol") != DIRECT_INPUT_PROTOCOL
            or payload.get("primary_fork_protocol") != PRIMARY_FORK_PROTOCOL
            or payload.get("mode_fork_protocol") != MODE_FORK_PROTOCOL):
        raise ValueError("m03_resume: cached sidecar payload contract mismatch")
    if cache is not None:
        cache.get('primary_replay',replay_dependencies,lambda:payload,
                  external=(sidecar_path,_sha256(sidecar_path),None))
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

    from ..checkpoint import FORMAT
    from ..sources import verify_sources

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

    from ..representation import Decoder, Encoder
    from ..transition import World

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
        payload = resolve_payload(torch.load(path, map_location="cpu", weights_only=False))
        if payload.get("schema") != FEATURE_SCHEMA or _sha(payload.get("metadata")) != metadata_sha256:
            raise ValueError(f"m03_resume: cached {arm}/{split} feature payload contract mismatch")
        features = payload.get("features")
        if not isinstance(features, dict) or not features or not all(isinstance(value, Tensor) and value.device.type == "cpu"
                                                                      for value in features.values()):
            raise ValueError(f"m03_resume: cached {arm}/{split} features are invalid")
        # A process may have stopped after atomic tensor publication and before
        # publishing its small manifest.  The validated tensor is safe to adopt.
        if not manifest_path.exists():
            manifest = {"schema": FEATURE_SCHEMA, "sha256": _sha256(path), "metadata_sha256": metadata_sha256}
            if 'cache_ref' in payload: manifest['dependency_key'] = payload['cache_ref']['key']
            _atomic_json_save(manifest_path, manifest)
        return features, manifest.get("dependency_key", _sha256(manifest_path))

    if manifest_path.exists():
        raise RuntimeError(f"m03_resume: cached {arm}/{split} manifest lacks its tensor payload")
    cache = active()
    dependency_key = None
    if cache is not None:
        dependencies = _feature_dependencies(arm, split, identity, sidecar_sha256, cache)
        external = cache.imports.get('features', {}).get((arm, split))
        features, dependency_key = cache.get('features:'+arm+':'+split, dependencies, encode, external=external)
    else:
        features = encode()
    if not features or not all(isinstance(value, Tensor) and value.device.type == "cpu" for value in features.values()):
        raise ValueError(f"m03_cache: {arm}/{split} encoder returned non-CPU tensors")
    payload = {"schema": FEATURE_SCHEMA, "metadata": metadata}
    if cache is None:
        payload['features'] = features
    else:
        payload['cache_ref'] = {'database':str(cache.path),'key':dependency_key}
    _atomic_torch_save(path, payload)
    manifest = {"schema": FEATURE_SCHEMA, "sha256": _sha256(path), "metadata_sha256": metadata_sha256}
    if dependency_key: manifest['dependency_key'] = dependency_key
    _atomic_json_save(manifest_path, manifest)
    return features, dependency_key or _sha256(manifest_path)


def _feature_dependencies(arm, split, identity, sidecar, cache):
    from . import history, diagnostics
    settings = cache.settings
    if arm == 'replay':
        functions = (history.replay_seed, history.verify_reference, _state_scalars,
                     _state_binary_labels, _action_outcomes, diagnostics.state_features)
        fields = REPLAY_SETTINGS
    elif split.startswith(('primary_', 'historical_')):
        functions = (diagnostics.encode_memory, diagnostics.shuffle_completed_pairs)
        fields = ('seed','encode_batch')
    elif arm in ('raw','tc'):
        functions = (_encode_lewm,)
        fields = ('lewm_context','encode_batch')
    else:
        functions = (_encode_legacy, _legacy_native_parity_preflight)
        fields = ('direct_context','direct_encode_batch','direct_successor_batch')
    # API/runtime changes invalidate their encodings; an unrelated probe edit does not.
    runtime = ('world_api.py','state.py','data.py','config.py') + (
        ('lewm.py','lewm_config.py','mamba_recurrence.py') if arm in ('raw','tc') else
        ('representation.py','transition.py','time_mixer.py'))
    return {'identity': identity, 'data': sidecar, 'functions':[cache.code(f) for f in functions],
            'execution':dict({k:getattr(settings,k) for k in fields},device=cache.device if arm != 'replay' else 'cpu'),
            'runtime':{p:_sha256(ROOT/'d4mj'/p) for p in runtime if (ROOT/'d4mj'/p).exists()}}


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
    cache = active()
    scope = 'historical' if Path(output).name == 'historical' else 'primary'
    imported = cache.imports.get('stages', {}).get((scope,name)) if cache else None
    if imported:
        origin, checksum, selector = imported
        if _sha256(Path(origin)) != checksum: raise ValueError('m03_import: completed stage bytes changed')
        result = json.loads(Path(origin).read_text())['result']
        if selector: result = result[selector]
        cache.counts['import:stage'] += 1
    else:
        result = compute()
    _atomic_json_save(path, {"schema": STAGE_SCHEMA, "metadata_sha256": metadata_sha256,
                             "metadata": metadata, "result": result, "imported_from": imported})
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
    length = context.shape[1]
    if not 1 <= length <= settings.direct_context or actions.shape[1] != length - 1:
        raise ValueError("m03_direct_context: sidecar does not carry the sealed native Direct prefix")
    first_incoming = values.get("direct_first_action")
    if not isinstance(first_incoming, Tensor) or first_incoming.shape != (len(context),):
        raise ValueError("m03_direct_context: sidecar lacks the first incoming action for each native prefix")
    if length < settings.direct_context:
        times = values.get("time")
        if times is None or not bool((times == length-1).all()) or not bool((first_incoming == 17).all()):
            raise ValueError("m03_direct_context: short history must start at the true episode BOS")
    required = max(bundle.config.receptive_field, bundle.config.dynamics_context)
    if settings.direct_context < required:
        raise ValueError(f"m03_direct_context: {settings.direct_context} truncates native Direct context {required}")
    root, observed_successor, generated = [], [], []
    device = bundle.device
    all_actions = torch.arange(bundle.n_actions, device=device, dtype=torch.long)
    for start in range(0, len(context), settings.direct_encode_batch):
        end = min(len(context), start + settings.direct_encode_batch)
        frame, past = context[start:end].to(device), actions[start:end].to(device)
        first = first_incoming[start:end, None].to(device)
        # This is intentionally the archived Direct transfer contract: encode the
        # complete native prefix in one causal pass, construct the world state from
        # every encoded block, then encode each real successor appended to that same
        # prefix.  A 16-frame sequential reconstruction would truncate both the
        # MAE encoder's 31-frame receptive field and the world's 48-step cache.
        context_latent = bundle.encode(frame)
        state = bundle.prefill(context_latent, past, first_action=first)
        branches = bundle.repeat_state(state, bundle.n_actions)
        action = all_actions.repeat(end-start)[:, None]
        prediction, _ = bundle.advance(branches, action)
        generated.append(bundle.world_state(prediction).latent[:, 0].flatten(1).reshape(end-start, bundle.n_actions, -1).cpu())
        successor_latents = []
        for action_start in range(0, bundle.n_actions, settings.direct_successor_batch):
            action_end = min(bundle.n_actions, action_start + settings.direct_successor_batch)
            branch = successors[start:end, action_start:action_end].to(device)
            width = action_end - action_start
            prefix = frame[:, None].expand(end-start, width, *frame.shape[1:])
            full = torch.cat((prefix, branch[:, :, None]), dim=2).reshape(
                (end-start) * width, frame.shape[1] + 1, *frame.shape[2:]
            )
            successor_latents.append(bundle.encode(full)[:, -1].flatten(1).reshape(end-start, width, -1).cpu())
        observed_successor.append(torch.cat(successor_latents, dim=1))
        root.append(context_latent[:, -1].flatten(1).cpu())
    return {"projected": torch.cat(root), "observed_successor": torch.cat(observed_successor),
            "generated_successor": torch.cat(generated)}


@torch.inference_mode()
def _legacy_native_parity_preflight(bundle: ModelBundle, values: dict[str, Any],
                                    settings: M03Settings) -> dict[str, Any]:
    """Prove one M03 Direct row agrees with the archived production evaluator.

    ``_encode_legacy`` intentionally uses the adapter API, which is what the
    gate needs for normal execution.  This short preflight follows the older
    evaluator's explicit encoder/``commit_inputs``/world path instead.  It
    protects the easy-to-miss first-incoming-action and full-prefix contract
    without importing any old labels, heads, or reported measurements.
    """

    from ..transition import commit_inputs

    if bundle.config.transition != "direct":
        raise ValueError("m03_direct_parity: requires a Direct anchor")
    one = {
        "context": values["context"][:1],
        "past_actions": values["past_actions"][:1],
        "successors": values["successors"][:1],
        "direct_first_action": values["direct_first_action"][:1],
    }
    if "time" in values:
        one["time"] = values["time"][:1]
    candidate = _encode_legacy(bundle, one, settings)
    frame = one["context"].to(bundle.device)
    past = one["past_actions"].to(bundle.device)
    first = one["direct_first_action"][:, None].to(bundle.device)
    successors = one["successors"].to(bundle.device)

    # This is the historical Direct evaluator's production calculation, stated
    # here without the adapter: full causal encoder prefix, one committed world
    # pass with the incoming first action, and all seventeen next-action heads.
    history = bundle.encode(frame)
    led_to_action = torch.cat((first, past), dim=1)
    committed, conditioning = commit_inputs(history, None, bundle.config)
    features, _, _ = bundle.world(None, led_to_action, committed, conditioning)
    actions = torch.arange(bundle.n_actions, dtype=torch.long, device=bundle.device)[None]
    generated = bundle.world.predict(
        features[:, -1:].expand(1, bundle.n_actions, *features.shape[2:]), actions
    ).flatten(2).cpu()
    observed_chunks = []
    for start in range(0, bundle.n_actions, settings.direct_successor_batch):
        end = min(bundle.n_actions, start + settings.direct_successor_batch)
        full = torch.cat((frame.expand(end - start, *frame.shape[1:]), successors[0, start:end, None]), dim=1)
        observed_chunks.append(bundle.encode(full)[:, -1].flatten(1).cpu())
    reference = {
        "projected": history[:, -1].flatten(1).cpu(),
        "observed_successor": torch.cat(observed_chunks)[None],
        "generated_successor": generated,
    }
    max_abs = {
        name: float((candidate[name] - reference[name]).abs().max().item())
        for name in reference
    }
    maximum = max(max_abs.values())
    return {
        "status": "pass" if maximum <= DIRECT_NATIVE_PARITY_TOLERANCE else "fail",
        "protocol": DIRECT_INPUT_PROTOCOL,
        "reference": "archived_direct_full_prefix_encoder_commit_world_v1",
        "row": 0,
        "tolerance": DIRECT_NATIVE_PARITY_TOLERANCE,
        "max_abs": max_abs,
        "maximum_max_abs": maximum,
    }


def _standardize(train: Tensor, dev: Tensor) -> tuple[Tensor, Tensor]:
    mean, scale = train.mean(0, keepdim=True), train.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (train - mean) / scale, (dev - mean) / scale


@memoized(PROBE_SETTINGS)
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


@memoized()
def _ridge_predict(train_x: Tensor, train_y: Tensor, dev_x: Tensor, ridge: float) -> Tensor:
    """Linear raw-pixel reference via the dual ridge form (safe when p >> n)."""

    train_x, dev_x = _standardize(train_x.float(), dev_x.float())
    y = train_y.float()
    kernel = train_x @ train_x.T
    solution = torch.linalg.solve(
        kernel + ridge * torch.eye(len(train_x), device=train_x.device, dtype=train_x.dtype), y
    )
    return (dev_x @ train_x.T @ solution).cpu()


@lru_cache(maxsize=64)
def _bootstrap_multiplicities(groups: int, draws: int, seed: int) -> np.ndarray:
    """Exactly the legacy per-draw Torch RNG stream, cached as seed counts."""
    rng = torch.Generator().manual_seed(seed)
    selected = torch.stack([torch.randint(groups, (groups,), generator=rng) for _ in range(draws)])
    offsets = torch.arange(draws)[:, None] * groups
    result = torch.bincount((selected + offsets).flatten(), minlength=draws*groups).reshape(draws,groups).numpy().astype(np.int32)
    result.setflags(write=False)
    return result


def _bootstrap_bounds(samples, draws):
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if len(samples) < .95 * draws:
        return None
    # The original materializes Python floats with Torch's default dtype before
    # quantiles. Retain that rounding instead of introducing a float64 interval.
    return torch.tensor(samples.tolist()).quantile(torch.tensor([.025,.975])).tolist()


def _bootstrap_mean_draws(values, mask, roots, *, draws, seed):
    keys,inverse = roots.unique(sorted=True,return_inverse=True)
    selected = mask.cpu().numpy().astype(bool); group = inverse.numpy()
    columns = values.detach().cpu().double().numpy()
    counts = np.bincount(group[selected],minlength=len(keys))
    sums = np.stack([np.bincount(group[selected],weights=v[selected],minlength=len(keys)) for v in columns.T],1)
    weights = _bootstrap_multiplicities(len(keys),draws,seed)
    denominator = weights @ counts
    result = np.full((draws,columns.shape[1]),np.nan)
    np.divide(weights @ sums,denominator[:,None],out=result,where=denominator[:,None]>0)
    # The old callbacks take float() of each float32 Torch mean.
    return result.astype(np.float32).astype(np.float64)


def _weighted_auc_draws(score, target, inverse, multiplicities):
    """Sort once; tied-score positive/negative counts implement Mann-Whitney."""
    score = score.detach().cpu().double().numpy()
    if not np.isfinite(score).all(): raise ValueError('probe scores are nonfinite')
    order = np.argsort(score, kind='stable')
    labels = target.detach().cpu().bool().numpy()[order]
    group = inverse[order]
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1])+1]
    samples = np.full(len(multiplicities), np.nan)
    # Bound temporaries to a few MiB on the large all-action panels.
    for first in range(0,len(multiplicities),32):
        weights = multiplicities[first:first+32,group]
        positive = np.add.reduceat(weights*labels[None],starts,axis=1,dtype=np.int64)
        negative = np.add.reduceat(weights*(~labels)[None],starts,axis=1,dtype=np.int64)
        p, n = positive.sum(1), negative.sum(1)
        numerator = (positive*(negative.cumsum(1)-.5*negative)).sum(1)
        valid = (p > 0) & (n > 0)
        np.divide(numerator,p*n,out=samples[first:first+len(weights)],where=valid)
    return samples


def _auc_bootstrap(left, target, roots, *, draws, seed, right=None):
    point = binary_auc(left,target)
    if point is None: return None,None
    if right is not None:
        other = binary_auc(right,target)
        if other is None: return None,None
        point -= other
    keys, inverse = roots.unique(sorted=True,return_inverse=True)
    if len(keys) < 2: return point,None
    counts = _bootstrap_multiplicities(len(keys),draws,seed)
    samples = _weighted_auc_draws(left,target,inverse.numpy(),counts)
    if right is not None:
        samples -= _weighted_auc_draws(right,target,inverse.numpy(),counts)
    return point,_bootstrap_bounds(samples,draws)


def _regression_bootstrap(target, left, roots, *, draws, seed, right=None):
    """Per-seed sufficient statistics; retain legacy fallback at degeneracy."""
    def original(rows):
        selected = target[rows]
        denominator = (selected-selected.mean()).square().sum().clamp_min(1e-12)
        if right is not None and float(denominator) <= 1e-12: return None
        a = float(1-(left[rows]-selected).square().sum()/denominator)
        return a if right is None else a-float(1-(right[rows]-selected).square().sum()/denominator)
    point = original(torch.arange(len(roots)))
    keys,inverse = roots.unique(sorted=True,return_inverse=True)
    if point is None or len(keys) < 2: return point,None
    group = inverse.numpy()
    # Center first to avoid cancellation for large offsets / small variance.
    y = target.detach().cpu().double().numpy(); centered = y-y[0]
    if np.max(np.abs(y))*torch.finfo(target.dtype).eps > np.std(y)*1e-3:
        return _root_bootstrap(left,roots,original,draws=draws,seed=seed)
    columns = [np.ones(len(y)),centered,centered*centered,
               (left-target).square().double().numpy()]
    if right is not None: columns.append((right-target).square().double().numpy())
    sums = np.stack([np.bincount(group,weights=v,minlength=len(keys)) for v in columns],1)
    counts = _bootstrap_multiplicities(len(keys),draws,seed)
    moments = counts @ sums
    n, sy, sy2, error = moments[:,:4].T
    variance_sum = sy2-sy*sy/n
    denominator = np.maximum(variance_sum,1e-12)
    samples = 1-error/denominator
    if right is not None:
        samples -= 1-moments[:,4]/denominator
    # Cancellation and constant resamples need the exact old float32 rule.
    threshold = np.maximum(1e-12,32*np.finfo(np.float64).eps*(np.abs(sy2)+sy*sy/n))
    uncertain = np.flatnonzero(variance_sum <= threshold)
    if len(uncertain):
        groups = [torch.where(inverse==i)[0] for i in range(len(keys))]
        rng = torch.Generator().manual_seed(seed)
        uncertain = set(uncertain.tolist())
        for index in range(draws):
            selected = torch.randint(len(keys),(len(keys),),generator=rng)
            if index in uncertain:
                value = original(torch.cat([groups[i] for i in selected]))
                samples[index] = np.nan if value is None else value
    return point,_bootstrap_bounds(samples,draws)


def _root_bootstrap(values: Tensor, roots: Tensor, fn, *, draws: int, seed: int) -> tuple[float | None, list[float] | None]:
    keys = roots.unique(sorted=True)
    all_indices = torch.arange(len(roots))
    point = fn(all_indices)
    if point is None:
        return None, None
    if len(keys) < 2:
        return point, None
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


@memoized()
def _binary_metrics(logits: Tensor, truth: Tensor, roots: Tensor, names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    names = tuple(names)
    report, macro = {}, []
    for index, name in enumerate(names):
        target_shape = truth[..., index]
        score, target = logits[..., index].reshape(-1), target_shape.reshape(-1).bool()
        root = _expanded_roots(roots, target_shape)
        positive, negative = int(target.sum()), int((~target).sum())
        positive_clusters, negative_clusters = len(root[target].unique()), len(root[~target].unique())
        if (positive < settings.minimum_positive or negative < settings.minimum_negative
                or positive_clusters < settings.minimum_positive or negative_clusters < settings.minimum_negative):
            report[name] = {"status": "insufficient_coverage", "positive": positive, "negative": negative,
                            "positive_seed_clusters": positive_clusters, "negative_seed_clusters": negative_clusters}
            continue
        point, interval = _auc_bootstrap(score,target,root,draws=settings.bootstrap_draws,seed=settings.seed+index)
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage", "positive": positive,
                        "negative": negative, "positive_seed_clusters": positive_clusters, "negative_seed_clusters": negative_clusters,
                        "auc": point, "interval": interval}
        if point is not None:
            macro.append(point)
    return {"targets": report, "macro_auc": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names)}


@memoized()
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
        _, interval = _regression_bootstrap(target,predicted,roots,draws=settings.bootstrap_draws,seed=settings.seed+400+index)
        report[name] = {"r2": r2, "interval": interval, "mae": float((predicted-target).abs().mean()),
                        "variance": total_variance,
                        "status": "measured" if interval is not None else "insufficient_coverage"}
        if interval is not None:
            values.append(r2)
    return {"targets": report, "mean_r2": float(np.mean(values)) if values else None,
            "supported_targets": len(values), "total_targets": len(names)}


@memoized()
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
        positive_clusters, negative_clusters = len(root[target].unique()), len(root[~target].unique())
        if (positive < settings.minimum_positive or negative < settings.minimum_negative
                or positive_clusters < settings.minimum_positive or negative_clusters < settings.minimum_negative):
            report[name] = {"status": "insufficient_coverage", "positive": positive, "negative": negative,
                            "positive_seed_clusters": positive_clusters, "negative_seed_clusters": negative_clusters}
            continue

        point, interval = _auc_bootstrap(left,target,root,right=right,draws=settings.bootstrap_draws,seed=settings.seed+2000+index)
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage",
                        "positive": positive, "negative": negative, "auc_difference": point, "interval": interval}
        if interval is not None and point is not None:
            macro.append(point)
    return {"targets": report, "mean_auc_difference": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names), "direction": "left_minus_right"}


@memoized()
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

        point, interval = _regression_bootstrap(target,left_pred,roots,right=right_pred,draws=settings.bootstrap_draws,seed=settings.seed+2100+index)
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage",
                        "variance": variance, "r2_difference": point, "interval": interval}
        if interval is not None and point is not None:
            macro.append(point)
    return {"targets": report, "mean_r2_difference": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names), "direction": "left_minus_right"}


def _fit_label_coverage(truth: Tensor, roots: Tensor, names, settings: M03Settings) -> dict:
    result = {}
    for i, name in enumerate(names):
        target = truth[..., i].bool()
        expanded = _expanded_roots(roots, target)
        flat = target.flatten()
        pos, neg = len(expanded[flat].unique()), len(expanded[~flat].unique())
        result[name] = {"positive_seed_clusters": pos, "negative_seed_clusters": neg,
                        "status": "adequate" if pos >= settings.minimum_positive and neg >= settings.minimum_negative
                        else "insufficient_coverage"}
    return result


def _static_report(train: dict[str, Tensor], dev: dict[str, Tensor], features: dict[str, dict[str, Tensor]],
                   settings: M03Settings, *, device: str) -> dict[str, Any]:
    """Fit identical static probes for every representation without DEV selection."""

    output: dict[str, Any] = {"continuous_names": STATIC_CONTINUOUS, "binary_names": STATIC_BINARY, "sources": {},
        "fit_coverage": _fit_label_coverage(train["root_binary"], train["episode"], STATIC_BINARY, settings)}
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


@memoized()
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


@memoized()
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


@memoized()
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


@memoized()
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
                              "successor_binary_names": STATIC_BINARY, "arms": {},
                              "fit_coverage": _fit_label_coverage(train["outcomes"], train["episode"], OUTCOME_BINARY, settings),
                              "successor_fit_coverage": _fit_label_coverage(train["next_binary"], train["episode"], STATIC_BINARY, settings)}
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
    for scope, coverage in (("static_fit", static.get("fit_coverage", {})),
                            ("outcome_fit", outcomes.get("fit_coverage", {})),
                            ("successor_fit", outcomes.get("successor_fit_coverage", {}))):
        for name, entry in coverage.items():
            if entry["status"] != "adequate":
                missing.append(f"{scope}:{name}")
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
             structural_smoke: bool = False, include_history: bool = True) -> dict[str, Any]:
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
                             structural_smoke=structural_smoke, include_history=include_history)
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
    direct_native_preflights: dict[str, dict[str, Any]] = {}
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
        del bundle, payload
        if device == "cuda":
            torch.cuda.empty_cache()

    if include_direct:
        for name in ("direct_attention", "direct_mamba"):
            bundle, parents[name] = _legacy_anchor(name, device=device)
            # Persist the parity proof independently.  If a process stops while
            # later Direct features are running, a resume checks this immutable
            # record rather than trusting an unrecorded in-memory preflight.
            direct_native_preflights[name] = _load_or_compute_stage(
                output,
                f"native_direct_parity_{name}",
                {
                    "sidecar_sha256": sidecar_sha256,
                    "anchor": parents[name],
                    "protocol": DIRECT_INPUT_PROTOCOL,
                    "tolerance": DIRECT_NATIVE_PARITY_TOLERANCE,
                },
                lambda b=bundle: _legacy_native_parity_preflight(
                    b, sidecar["splits"]["dev"], settings
                ),
            )
            if direct_native_preflights[name].get("status") != "pass":
                raise RuntimeError(
                    f"m03_direct_parity: {name} disagrees with the native production route: "
                    f"{direct_native_preflights[name]}"
                )
            _write_status(output, "native_direct_parity_ready", arm=name,
                          sidecar_sha256=sidecar_sha256)
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
    from .diagnostics import eda_report
    eda = _load_or_compute_stage(output, "corrected_eda", outcome_metadata,
        lambda: eda_report(sidecar["splits"]["train"], sidecar["splits"]["dev"], outcome_features, settings, device=device))
    from .diagnostics import observability_report
    observability = _load_or_compute_stage(output, "observability", outcome_metadata,
        lambda: observability_report(sidecar["splits"]["train"], sidecar["splits"]["dev"], settings, device=device))
    # Release primary payloads before the larger historical panels.
    dataset_sha256 = sidecar["dataset_sha256"]
    del feature_rows, outcome_features, static_features
    if device == "cuda":
        torch.cuda.empty_cache()
    historical = {"status": "not_requested", "m4_authorized": False}
    if include_history:
        from .history import run_history
        historical = run_history(output, settings, root=ROOT,
            checkpoints={"raw": raw_checkpoint, "tc": tc_checkpoint}, dataset_sha256=dataset_sha256,
            device=device, include_direct=include_direct, smoke=structural_smoke)
    advisory = _advisory_summary(outcomes)
    report = {
        "schema": SCHEMA, "run_contract_sha256": contract_sha256, "settings": asdict(settings), "parents": parents,
        "execution": {"device": device, "mode": "structural_smoke" if structural_smoke else "sealed_full_gate",
                      "direct_anchors_included": include_direct},
        "sidecar": sidecar_manifest,
        "native_direct_parity": direct_native_preflights,
        "static_retention": static, "all_action_one_step": outcomes, "corrected_eda": eda,
        "historical_regression": historical, "observability": observability, "advisory": advisory,
        "decision": _decision(static, outcomes, sidecar, structural_smoke=structural_smoke),
    }
    report["decision"]["suite_complete"] = include_history and include_direct and not structural_smoke
    _atomic_json_save(report_path, report)
    _write_status(output, "complete", sidecar_sha256=sidecar_sha256, report_sha256=_sha256(report_path))
    return report


def _settings_for_smoke(settings: M03Settings) -> M03Settings:
    return replace(settings, train_roots=8, dev_roots=4, mode_samples=2, encode_batch=2,
                   probe_hidden=8, probe_steps=2, probe_batch=8, bootstrap_draws=16,
                   minimum_positive=1, minimum_negative=1)


def resume_archived_baseline(output, *, validate_only=False):
    """Resume the exact retired evaluator; archive bytes must match run.json.

    Module names and __file__ retain their original values for contract identity.
    Only reads of retired gate source paths resolve through the checked archive;
    every other source, setting, input and cache check runs unchanged.
    """
    import sys
    import types
    import zipfile
    output = Path(output).resolve()
    stored = json.loads((output / "run.json").read_text())
    contract = stored['contract']
    if stored['contract_sha256'] != _sha(contract):
        raise ValueError('m03_archive: invalid baseline contract')
    expected = {}
    def collect(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value: expected[value['path']] = value['sha256']
            for v in value.values(): collect(v)
        elif isinstance(value, list):
            for v in value: collect(v)
    collect(contract)
    names = ('capability', 'diagnostics', 'observability', 'history')
    with zipfile.ZipFile(output / 'source.zip') as archive:
        sources = {str(ROOT / f'd4mj/m03_{name}.py'): archive.read(f'd4mj/m03_{name}.py') for name in names}
    for path, content in sources.items():
        if sha256(content).hexdigest() != expected.get(path):
            raise ValueError(f'm03_archive: evaluator bytes differ from original contract: {path}')
    saved, modules = {}, {}
    try:
        for name in names:
            key = 'd4mj.m03_' + name
            saved[key] = sys.modules.get(key)
            module = types.ModuleType(key)
            module.__file__ = str(ROOT / f'd4mj/m03_{name}.py')
            module.__package__ = 'd4mj'
            sys.modules[key] = modules[name] = module
        for module in modules.values():
            exec(compile(sources[module.__file__], module.__file__, 'exec'), module.__dict__)
        original = modules['capability']
        identity = original._input_identity
        def archived_identity(path):
            path = str(Path(path).resolve())
            return {'path': path, 'sha256': sha256(sources[path]).hexdigest()} if path in sources else identity(Path(path))
        original._input_identity = archived_identity
        settings = original.M03Settings(**contract['settings'])
        kwargs = dict(raw_checkpoint=Path(contract['raw_checkpoint']['path']), tc_checkpoint=Path(contract['tc_checkpoint']['path']),
                      dataset=Path(contract['dataset']['path']), settings=settings, device=contract['execution']['device'],
                      include_direct=contract['execution']['direct_anchors_included'],
                      structural_smoke=contract['execution']['structural_smoke'], include_history=contract['historical_panels_included'])
        actual = original._run_contract(**kwargs)
        if actual != contract:
            raise ValueError('m03_archive: live inputs differ from archived run; resume refused')
        if validate_only:
            return {'status': 'pass', 'contract_sha256': stored['contract_sha256']}
        return original.run_gate(output=output, **kwargs)
    finally:
        for key, module in saved.items():
            if module is None: sys.modules.pop(key, None)
            else: sys.modules[key] = module


def main(argv=None) -> int:
    started = (time.perf_counter(),time.process_time(),time.time())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-checkpoint", type=Path)
    parser.add_argument("--tc-checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/craftax_support_v2/manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-direct", action="store_true", help="debug-only; output cannot be a full historical-anchor gate")
    parser.add_argument("--skip-history", action="store_true", help="partial diagnostic only; omits historical regression panels")
    parser.add_argument("--smoke", action="store_true", help="tiny structural fixture, never a research result")
    parser.add_argument("--baseline", type=Path, help="add the Mamba supplement to an existing sealed gate")
    parser.add_argument("--resume-baseline", action="store_true", help="finish --baseline using its hash-checked source.zip first")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/lewm_gates_20260906/cache.sqlite3")
    parser.add_argument("--reuse-from", type=Path, help="adopt verified compatible artifacts from a retired gate")
    parser.add_argument("--memory-first", action="store_true", help="score Mamba using --reuse-from replay before remaining baseline work")
    parser.add_argument("--cache-compatibility", type=Path, help="hash-checked numerical-equivalence proof for prior statistics")
    args = parser.parse_args(argv)
    if args.memory_first and (args.reuse_from is None or args.baseline is not None):
        parser.error('--memory-first requires --reuse-from and cannot combine with --baseline')
    if args.resume_baseline and args.baseline is None:
        parser.error('--resume-baseline requires --baseline')
    settings = _settings_for_smoke(M03Settings()) if args.smoke else M03Settings()
    from .cache import ArtifactCache, prepare_imports, use_cache
    from .history import run_memory
    cache = None
    try:
        parent = json.loads((args.baseline/'run.json').read_text())['contract'] if args.baseline else None
        checkpoints = {a: getattr(args,a+'_checkpoint') or
                       (Path(parent[a+'_checkpoint']['path']) if parent else
                        ROOT / f'artifacts/lewm_gates_20260906/paired/{a}/joint/step-010000.pt')
                       for a in ('raw','tc')}
        imports = prepare_imports(args.reuse_from, root=ROOT, settings=settings, dataset=args.dataset,
                                  raw_checkpoint=checkpoints['raw'],tc_checkpoint=checkpoints['tc'],compatibility=args.cache_compatibility) if args.reuse_from else {}
        cache = ArtifactCache(args.cache,imports=imports,settings=settings,device=args.device,
                              compatibility=args.cache_compatibility,timing_path=args.out/"timing.json")
        cache.started,cache.cpu_started,cache.started_unix = started
        with use_cache(cache):
            if args.baseline is not None:
                if args.resume_baseline:
                    resume_archived_baseline(args.baseline)
                report = run_memory(args.baseline,args.out,settings,device=args.device,smoke=args.smoke,
                                    allow_incomplete=True,checkpoints=checkpoints)
                report_path = args.out/'report.json'
            else:
                kwargs = dict(raw_checkpoint=checkpoints['raw'],tc_checkpoint=checkpoints['tc'],
                              dataset=args.dataset,settings=settings,device=args.device,
                              include_direct=not args.skip_direct,structural_smoke=args.smoke,include_history=not args.skip_history)
                _open_run(args.out,_run_contract(**kwargs))
                if args.memory_first:
                    _write_status(args.out,'memory_first',baseline_scoring='queued')
                    report = run_memory(args.reuse_from,args.out/'memory',settings,device=args.device,
                                        smoke=args.smoke,allow_incomplete=True,checkpoints=checkpoints)
                baseline_report = run_gate(output=args.out,**kwargs)
                if not args.memory_first:
                    report = run_memory(args.out,args.out/'memory',settings,device=args.device,
                                        smoke=args.smoke,checkpoints=checkpoints)
                report_path = args.out/'complete.json'
                combined = {'baseline':_input_identity(args.out/'report.json'),
                    'memory':_input_identity(args.out/'memory/report.json'),
                    'decision':dict(report['decision'],suite_complete=bool(not args.smoke and
                        baseline_report['decision']['suite_complete'] and len(report['panels']) == 5))}
                if report_path.exists():
                    if json.loads(report_path.read_text()) != combined:
                        raise ValueError('m03_resume: combined report differs from its inputs')
                else: _atomic_json_save(report_path,combined)
            print(json.dumps({"status":"complete","report":str(report_path.resolve()),
                              "m03_capability":report['decision'].get('status','measured'),"m4_authorized":False}))
        return 0
    except Exception as error:
        if args.out.exists():
            atomic_manifest(args.out/'failure.json', {'schema':SCHEMA,'status':'stopped','reason':str(error),'m4_authorized':False})
            _write_status(args.out,'stopped',reason=str(error))
        import traceback
        traceback.print_exc()
        print(json.dumps({'status':'stopped','reason':str(error),'m4_authorized':False}))
        return 1
    finally:
        if cache is not None:
            if args.out.exists(): atomic_manifest(args.out/'cache_usage.json',dict(cache.counts))
            cache.close()


if __name__ == "__main__":
    from d4mj.m03.gate import main as canonical_main
    raise SystemExit(canonical_main())
