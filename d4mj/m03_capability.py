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
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"m03_output: sidecar path already exists: {output}")
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
    return payload


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


def _binary_metrics(logits: Tensor, truth: Tensor, roots: Tensor, names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    names = tuple(names)
    report, macro = {}, []
    for index, name in enumerate(names):
        target_shape = truth[..., index]
        score, target = logits[..., index].reshape(-1), target_shape.reshape(-1).bool()
        root = roots.reshape(len(roots), *([1] * (target_shape.ndim - 1))).expand_as(target_shape).reshape(-1)
        positive, negative = int(target.sum()), int((~target).sum())
        if positive < settings.minimum_positive or negative < settings.minimum_negative:
            report[name] = {"status": "insufficient_coverage", "positive": positive, "negative": negative}
            continue
        def auc(rows):
            return binary_auc(score[rows], target[rows])
        point, interval = _root_bootstrap(score, root, auc, draws=settings.bootstrap_draws, seed=settings.seed+index)
        probability = score.sigmoid()
        report[name] = {"status": "measured" if interval is not None else "insufficient_coverage", "positive": positive,
                        "negative": negative, "auc": point, "interval": interval,
                        "brier": float((probability-target.float()).square().mean())}
        if point is not None:
            macro.append(point)
    return {"targets": report, "macro_auc": float(np.mean(macro)) if macro else None,
            "supported_targets": len(macro), "total_targets": len(names)}


def _regression_metrics(prediction: Tensor, truth: Tensor, roots: Tensor, names: Iterable[str], settings: M03Settings) -> dict[str, Any]:
    names = tuple(names)
    report, values = {}, []
    for index, name in enumerate(names):
        target, predicted = truth[:, index], prediction[:, index]
        denominator = ((target-target.mean()).square().sum()).clamp_min(1e-12)
        r2 = float(1 - (predicted-target).square().sum()/denominator)
        def score(rows):
            selected = target[rows]
            denom = ((selected-selected.mean()).square().sum()).clamp_min(1e-12)
            return float(1 - (predicted[rows]-selected).square().sum()/denom)
        _, interval = _root_bootstrap(predicted, roots, score, draws=settings.bootstrap_draws, seed=settings.seed+400+index)
        report[name] = {"r2": r2, "interval": interval, "mae": float((predicted-target).abs().mean()),
                        "status": "measured" if interval is not None else "insufficient_coverage"}
        values.append(r2)
    return {"targets": report, "mean_r2": float(np.mean(values)) if values else None}


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
    """A deterministic prediction may match one valid simulator outcome mode."""

    probability = logits.sigmoid()
    primary_brier = (probability - truth.float()).square().mean(-1)
    # [root, action, mode, label] -> pick the simulator outcome mode nearest
    # to the deterministic prediction.  This is diagnostic, never likelihood.
    closest_brier = (probability[:, :, None] - modes.float()).square().mean(-1).min(-1).values

    def mean(value: Tensor):
        def score(rows: Tensor) -> float:
            return float(value[rows].mean())
        return _root_bootstrap(value, roots, score, draws=settings.bootstrap_draws, seed=settings.seed + 1600)

    primary, primary_interval = mean(primary_brier)
    closest, closest_interval = mean(closest_brier)
    varying = (modes.max(2).values != modes.min(2).values).any(-1)
    return {"sampled_modes_per_state_action": int(modes.shape[2]),
            "primary_outcome_brier": primary, "primary_outcome_interval": primary_interval,
            "nearest_sampled_mode_brier": closest, "nearest_sampled_mode_interval": closest_interval,
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


def _outcome_report(train: dict[str, Tensor], dev: dict[str, Tensor], features: dict[str, dict[str, Tensor]],
                    settings: M03Settings, *, device: str) -> dict[str, Any]:
    """All-action outcome probes and generated-state transfer, fitted on TRAIN only."""

    output: dict[str, Any] = {"outcome_names": OUTCOME_BINARY, "arms": {}}
    train_truth, dev_truth = train["outcomes"].reshape(-1, len(OUTCOME_BINARY)).float().to(device), dev["outcomes"]
    root = dev["episode"]
    action_train, action_dev = _one_hot_actions(len(train["outcomes"]), device=device), _one_hot_actions(len(dev["outcomes"]), device=device)
    rng = torch.Generator(device=device).manual_seed(settings.seed + 900)
    shuffled_train = action_train[torch.randperm(len(action_train), generator=rng, device=device)]
    shuffled_dev = action_dev[torch.randperm(len(action_dev), generator=rng, device=device)]
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
            }
            if dev_reset is not None:
                reset_model = transfer["reset"]
                family_report["reset_context_generated_successor"] = _binary_metrics(reset_model, dev_truth, root, OUTCOME_BINARY, settings)
                family_report["context_sensitivity"] = _context_summary(
                    generated_model.reshape(-1, 17, len(OUTCOME_BINARY)),
                    reset_model.reshape(-1, 17, len(OUTCOME_BINARY)), dev_truth)
            rows[family] = family_report
        output["arms"][arm] = rows
    return output


def _advisory_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    """Expose the non-gating context/mode checks without duplicating score data."""

    return {"mode_interpretation": "nearest sampled mode is descriptive, not a likelihood or promotion criterion",
            "context_interpretation": "reset-context differences test prefix use, not policy quality",
            "locations": "per-arm/per-probe-family entries under all_action_one_step"}


def _decision(static: dict[str, Any], outcomes: dict[str, Any], sidecar: dict[str, Any], *,
              structural_smoke: bool) -> dict[str, Any]:
    """Conservative component verdict; sparse targets are never promoted to pass."""

    def measured(source: str, family: str, part: str) -> bool:
        return source in static["sources"] and static["sources"][source][family][part]["supported_targets"] > 0
    raw_static = all(measured("raw_projected", family, "binary") for family in ("linear", "mlp"))
    tc_static = all(measured("tc_projected", family, "binary") for family in ("linear", "mlp"))
    raw_generated = "raw" in outcomes["arms"]
    tc_generated = "tc" in outcomes["arms"]
    enough = raw_static and tc_static and raw_generated and tc_generated
    status = "structural_smoke_not_a_result" if structural_smoke else ("measured" if enough else "insufficient_coverage")
    return {
        "m03_capability": status,
        "raw_status": "structural_smoke_not_a_result" if structural_smoke else ("measured" if raw_static and raw_generated else "insufficient_coverage"),
        "tc_status": "structural_smoke_not_a_result" if structural_smoke else ("measured" if tc_static and tc_generated else "insufficient_coverage"),
        "tc_promotion": "not_evaluated_by_scalar_gate",
        "m4_data_contract": "unreviewed",
        "m4_authorized": False,
        "scope": "M0-M3 frozen representation and one-step semantic diagnostics only",
    }


def run_gate(*, raw_checkpoint: Path, tc_checkpoint: Path, dataset: Path, output: Path,
             settings: M03Settings, device: str, include_direct: bool = True,
             structural_smoke: bool = False) -> dict[str, Any]:
    """Run one complete fresh M03 report.  ``output`` must be an empty directory."""

    if device != "cuda" and not structural_smoke:
        raise RuntimeError("m03_runtime: full Raw/TC/Direct gate requires CUDA with IEEE Triton execution; CPU is smoke-only")
    if device != "cuda" and include_direct:
        raise RuntimeError("m03_runtime: Direct-M historical anchor requires CUDA; use --skip-direct only for a structural CPU smoke")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("m03_runtime: requested CUDA but no CUDA device is available")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("m03_output: gate requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_manifest(output / "settings.json", asdict(settings))
    sidecar = build_sidecar(dataset, output / "sidecar", settings)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    bundles, parents = {}, {}
    for variant, path in (("raw", raw_checkpoint), ("tc", tc_checkpoint)):
        bundle, payload, source = load_m03_bundle(path, device=device, dataset_sha256=sidecar["dataset_sha256"])
        bundles[variant] = bundle
        parents[variant] = {"path": str(Path(path).resolve()), "sha256": _sha256(path), "step": payload["step"],
                            "recipe_id": payload["recipe_id"], "source_evaluation": source}
    anchors = {}
    if include_direct:
        for name in ("direct_attention", "direct_mamba"):
            anchors[name], parents[name] = _legacy_anchor(name, device=device)
    feature_rows: dict[str, dict[str, Tensor]] = {}
    for variant, bundle in bundles.items():
        for split in ("train", "dev"):
            feature_rows.setdefault(variant, {})[split] = _encode_lewm(bundle, sidecar["splits"][split], settings)
    for name, bundle in anchors.items():
        for split in ("train", "dev"):
            feature_rows.setdefault(name, {})[split] = _encode_legacy(bundle, sidecar["splits"][split], settings)
    static_features: dict[str, dict[str, Tensor]] = {}
    for variant in ("raw", "tc"):
        static_features[f"{variant}_cls"] = {split: feature_rows[variant][split]["cls"] for split in ("train", "dev")}
        static_features[f"{variant}_projected"] = {split: feature_rows[variant][split]["projected"] for split in ("train", "dev")}
    for name in anchors:
        static_features[f"{name}_projected"] = {split: feature_rows[name][split]["projected"] for split in ("train", "dev")}
    static = _static_report(sidecar["splits"]["train"], sidecar["splits"]["dev"], static_features, settings, device=device)
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
    for name in anchors:
        outcome_features[name] = {
            "train_projected": feature_rows[name]["train"]["projected"],
            "dev_projected": feature_rows[name]["dev"]["projected"],
            "train_observed_successor": feature_rows[name]["train"]["observed_successor"],
            "dev_observed_successor": feature_rows[name]["dev"]["observed_successor"],
            "dev_generated_successor": feature_rows[name]["dev"]["generated_successor"],
        }
    outcomes = _outcome_report(sidecar["splits"]["train"], sidecar["splits"]["dev"], outcome_features, settings, device=device)
    advisory = _advisory_summary(outcomes)
    report = {
        "schema": SCHEMA, "settings": asdict(settings), "parents": parents,
        "execution": {"device": device, "mode": "structural_smoke" if structural_smoke else "sealed_full_gate",
                      "direct_anchors_included": include_direct},
        "sidecar": json.loads((output / "sidecar" / "manifest.json").read_text()),
        "static_retention": static, "all_action_one_step": outcomes, "advisory": advisory,
        "decision": _decision(static, outcomes, sidecar, structural_smoke=structural_smoke),
    }
    atomic_manifest(output / "report.json", report)
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
