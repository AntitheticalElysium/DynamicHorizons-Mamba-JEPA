"""Encoder-specific latent functions over one shared, resumable episode-store writer.

MAE preserves bounded temporal memory and its historical cache digest. LeWM keeps
its retained projector and frozen BN buffers. Storage and recovery are shared.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch

from .config import Config, canonical_json
from .lewm_config import LeWMConfig
from .data import (FORMAT, STORE_FORMAT, Episode, EpisodeCorpus, atomic_manifest,
                   load_episodes, save_episode_shard, patchify, audit_episodes, _sha256)
from .representation import Encoder, pack
from .sources import source_digests, tensor_state_digest

def _cache_digest(encoder: Encoder, config: Config) -> str:
    """Identity of the latent cache: the whole latent function, not just its weights.

    Every field the encoder's forward pass depends on is included. Weights alone are
    not identity -- two encoders with identical parameters but different resolution
    and patch layout produced the same digest, and the cache from one would load
    against the other. The time mixer is excluded because the tokenizer is shared
    and always attention.
    """
    import hashlib

    shape = (
        config.patch,
        config.resolution,
        config.channels,
        config.n_patches,
        config.window,
        config.n_latents,
        config.d_bottleneck,
        config.packing,
        config.d_model_encoder,
        config.depth_encoder,
        config.n_heads_encoder,
        config.time_every,
        config.receptive_field,
        FORMAT,
    )
    weights = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        weights.update(name.encode())
        weights.update(tensor.detach().cpu().numpy().tobytes())
    visual = source_digests(replace(config, time_mixer="attention"))
    return hashlib.sha256(repr((shape, visual, weights.hexdigest())).encode()).hexdigest()[:16]


def _joint_encoder_digest(encoder) -> str:
    """Only the exported latent function: no predictor or actor weight identity."""
    import importlib.metadata
    import inspect
    from transformers import ViTConfig, ViTModel

    identity = {"family": "lewm_mamba", "schema": "d4mj_lewm_encoder_v1",
                "settings": asdict(encoder.settings), "mode": "eval", "storage_dtype": "float32",
                "weights_and_buffers": tensor_state_digest(encoder.state_dict()),
                "implementation": _sha256(Path(__file__).with_name("lewm.py")),
                "vit": _sha256(Path(inspect.getfile(ViTModel))),
                "vit_config": _sha256(Path(inspect.getfile(ViTConfig))),
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers")}
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


@torch.no_grad()
def _cache_episode(
    encoder: Encoder, episode: Episode, config: Config, digest: str
) -> Episode:
    if episode.observations is None:
        raise ValueError("raw observations are required to build a latent cache")
    frames = patchify(episode.observations[None], config.patch).to(config.device)
    latents, memory = [], None
    for start in range(0, frames.shape[1], config.sequence_long):
        chunk = frames[:, start : start + config.sequence_long]
        z, memory, _ = encoder(chunk, memory, offset=start)
        latents.append(z)
    packed = pack(torch.cat(latents, dim=1), config)[0].cpu()
    return replace(episode, latents=packed, latent_digest=digest)


def encoder_digest(encoder, config=None) -> str:
    if isinstance(config, Config):
        if not isinstance(encoder, Encoder):
            raise ValueError("cache_family: expected the MAE encoder")
        return _cache_digest(encoder, config)
    from .lewm import LeWMEncoder
    if not isinstance(encoder, LeWMEncoder):
        raise ValueError("cache_family: a legacy encoder needs its Config")
    return _joint_encoder_digest(encoder)


@torch.no_grad()
def _joint_cache_episode(encoder, episode, config, digest):
    device = next(encoder.parameters()).device
    encoded = []
    for start in range(0, len(episode)+1, config.runtime.cache_chunk):
        frames = episode.observations[start:start+config.runtime.cache_chunk][None].to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            z = encoder(frames).float()
            for t in sorted({0, frames.shape[1]-1}):
                single = encoder(frames[:, t:t+1]).float()
                torch.testing.assert_close(single, z[:, t:t+1], atol=1e-5, rtol=1e-4)
        if not bool(torch.isfinite(z).all()):
            raise ValueError("encoder_export: nonfinite latent")
        encoded.append(z[0].cpu())
    return replace(episode, observations=None, latents=torch.cat(encoded), latent_digest=digest)


def _prepare_encoder(encoder, episodes, config):
    if isinstance(config, LeWMConfig):
        audit_episodes(episodes, config)
        encoder.freeze()
    else:
        encoder.eval()
    return encoder_digest(encoder, config)


@torch.no_grad()
def cache_latents(encoder, episodes, config):
    digest = _prepare_encoder(encoder, episodes, config)
    encode = _joint_cache_episode if isinstance(config, LeWMConfig) else _cache_episode
    cached = [encode(encoder, episode, config, digest) for episode in episodes]
    return EpisodeCorpus(cached, source=episodes.source) if isinstance(episodes, EpisodeCorpus) else cached


def _write_latent_store(episodes, output, contract, encode_episode, shard_episodes, verify_encoder):
    """One recovery path: verify registered shards, never overwrite an orphan shard."""
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if {key: manifest.get(key) for key in contract} != contract:
            raise ValueError("latent-cache contract changed")
        # Validate already published bytes before continuing an interrupted export.
        digest = contract.get("cache_digest", contract.get("cache", {}).get("latent_digest"))
        load_episodes(output, digest=digest, allow_incomplete=True, verify=True)
        if manifest.get("complete"):
            return manifest
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = contract | {"complete": False, "episodes": 0, "transitions": 0,
                               "terminal_episodes": 0, "shards": []}
        atomic_manifest(manifest_path, manifest)
    start = int(manifest["episodes"])
    if start > len(episodes):
        raise ValueError("latent cache contains more episodes than its source")
    for first in range(start, len(episodes), shard_episodes):
        last = min(first+shard_episodes, len(episodes))
        cached = [replace(encode_episode(e), observations=None) for e in episodes[first:last]]
        path = output / f"shard-{len(manifest['shards']):06d}.pt"
        if path.exists():
            raise FileExistsError(f"unregistered latent-cache shard exists: {path}")
        record = save_episode_shard(path, cached)
        manifest["shards"].append(record)
        for key in ("episodes", "transitions", "terminal_episodes"):
            manifest[key] += record[key]
        atomic_manifest(manifest_path, manifest)
    verify_encoder()
    manifest["complete"] = True
    atomic_manifest(manifest_path, manifest)
    return manifest


@torch.no_grad()
def cache_latents_to_store(encoder, episodes, config, out: Path, *, source_contract: dict,
                            shard_episodes: int | None = None, parent_checkpoint: str | None = None):
    """Shared export entry point; each family keeps its own latent/identity contract.

    MAE retains resumable exports and its existing manifest schema. Joint exports
    require a fresh directory and a checkpoint parent; their float32 cache cannot
    be mistaken for a MAE cache. Returns a verified EpisodeCorpus in both cases.
    """
    out = Path(out)
    joint = isinstance(config, LeWMConfig)
    shard_episodes = (1 if joint else 32) if shard_episodes is None else shard_episodes
    if shard_episodes < 1:
        raise ValueError("shard_episodes must be positive")
    if joint:
        if not parent_checkpoint:
            raise ValueError("cache_parent: joint export requires its checkpoint hash")
        if out.exists() and any(out.iterdir()):
            raise ValueError("cache_output: refusing to overwrite a nonempty export directory")
    digest = _prepare_encoder(encoder, episodes, config)
    before = tensor_state_digest(encoder.state_dict())
    if joint:
        contract = {"format": STORE_FORMAT, "planned_episodes": len(episodes), "shard_episodes": shard_episodes,
                    "cache": {"schema": "d4mj_lewm_cache_v1", "family": config.family,
                              "latent_digest": digest, "shape": [1, config.encoder.latent_dim],
                              "dtype": "float32", "encoder_mode": "eval", "parent_checkpoint": parent_checkpoint,
                              "dataset_contract": source_contract}}
        encode = _joint_cache_episode
    else:
        contract = {"format": STORE_FORMAT, "kind": "d4mj_latent_cache_v1", "cache_digest": digest,
                    "source": source_contract, "planned_episodes": len(episodes), "shard_episodes": shard_episodes}
        encode = _cache_episode
    def verify_encoder():
        if before != tensor_state_digest(encoder.state_dict()):
            raise ValueError("encoder_normalization: export mutated encoder parameters/buffers")
    _write_latent_store(episodes, out, contract, lambda e: encode(encoder,e,config,digest),
                         shard_episodes, verify_encoder)
    return load_latent_cache(out, encoder, config)


def load_latent_cache(path: str | Path, encoder, config=None):
    path = Path(path)
    if isinstance(config, Config):
        return load_episodes(path, digest=encoder_digest(encoder,config), verify=True)
    manifest = json.loads((path / "manifest.json").read_text())
    cache = manifest.get("cache", {})
    if cache.get("schema") != "d4mj_lewm_cache_v1" or cache.get("family") != "lewm_mamba":
        raise ValueError("cache_family: this is not a LeWM export")
    digest = encoder_digest(encoder,config)
    if cache.get("latent_digest") != digest or cache.get("dtype") != "float32" or cache.get("shape") != [1,encoder.settings.latent_dim]:
        raise ValueError("cache_identity: encoder geometry, buffers, preprocessing or dtype changed")
    episodes = load_episodes(path,digest=digest,verify=True)
    for e in episodes:
        if e.latents.dtype != torch.float32 or e.latents.shape[1:] != (1,encoder.settings.latent_dim):
            raise ValueError("cache_shape: payload contradicts exported geometry/dtype")
    return episodes
