"""Does the model need same-state death/survival contrasts, or just better tails?

Three arms, identical apart from what the terminal supervision contains:

  control         ordinary production training only -- already trained as
                  `production_1b/world.pt` with the same encoder, corpus, budget,
                  initialisation and streams
  factual         plus a terminal term over the 3,198 actionable roots, always
                  supervising the single logged fatal transition
  counterfactual  plus a terminal term over the same roots on the same root schedule,
                  supervising one of the 17 successors per presentation

Five things this gets right that a first draft did not, each of which would have made the
comparison mean something other than it claims:

  action history   the terminal pass feeds the real incoming actions, not BOS at every
                   block. Feeding BOS is the exact defect that made the original fork
                   trainer's ordinary term an unconditional next-latent loss.
  separate RNG     the terminal pass draws from its own generator, so `transition_loss`
                   consumes the model stream exactly as the control does and the ordinary
                   batches stay aligned step for step.
  production path  the terminal term is blended into `dynamics` before `_balance`, which
                   is how `train_agent` mixes `terminal_dynamics_mass`. Hand-rolling a
                   weighted sum bypasses the running-RMS normalisation and leaves the
                   production checkpoint an unmatched control.
  schedule         a deterministic shuffled pass over all 3,198 x 17 = 54,366
                   (root, action) pairs. Cycling `(step + i) % 17` over randomly drawn
                   roots cycles globally, so no particular root is guaranteed to see all
                   17 actions. One full pass is 13,592 steps at batch 4; 20,000 gives
                   every pair once and about half a second pass.
  matched roots    the factual arm walks the identical root order and substitutes that
                   root's logged fatal action, so the arms differ only in which successor
                   is supervised -- never in how many.

Do not read a counterfactual null before step 13,592, which is the first complete pass.

The outcome is oracle-selected: all-17 replay is what identified which tails were
actionable, so the factual arm tests whether informative tails suffice, and is not a
sampling rule anyone could deploy.


Exposure, and the two arms that separate coverage from balance
---------------------------------------------------------------

The paired analysis found the factual arm gaining across its first pass and stopping,
while the counterfactual arm was flat to 13,592 and then gained on its partial second
pass. The exposure arithmetic is the obvious candidate. At four targets per step, 20,000
steps is 80,000 presentations. The counterfactual arm spreads those over 54,366 distinct
(root, action) pairs -- 1.47 sightings each -- while the factual arm spends all 80,000 on
3,198 distinct transitions, about 25 each.

`--terminal-actions 17` reads seventeen action-conditioned targets off one root's
features. The history and the backbone pass are what cost anything and `predict` is a
pool, a one-block mixer and a readout, so at four roots per step this measures 0.392
s/step against the one-action arm's 0.390 -- seventeen times the targets for nothing.
Four roots, not one: the root order is the same column of the same shuffled pass either
way, so both arms present 25 roots per root over 20,000 steps and differ only in how
many of each root's successors are supervised. Taking a fresh root permutation for the
17-action form would have cut root presentations fourfold while raising throughput, and
no result could then be attributed to either.

  arm 2  factual        4 roots x 1 action    25 presentations/root   1 lethal target
  arm 3  counterfactual 4 roots x 1 action    25 presentations/root   1.47 passes
  arm 4  full-action    4 roots x 17 actions  25 presentations/root   25 passes
  arm 5  balanced       arm 4, reweighted     25 presentations/root   25 passes

Arm 4 is not a clean repetition test. Against arm 3 it changes both how often a target is
seen and whether all seventeen are averaged inside one update, so it establishes that
full-action exposure-efficient supervision helps, not that repetition alone caused the
delayed gain. Extending arm 3 to two complete passes at 27,184 steps remains the cleaner
repetition test, and this trainer can now do it.

Arm 4 against arm 5 is the clean one. Uniform all-action supervision is 77.9% deaths --
13.25 lethal successors per root, measured, not assumed -- so `--balance-outcomes` gives
each root's lethal and surviving halves half its loss each:

    L_root = 0.5 * mean(L over lethal actions) + 0.5 * mean(L over safe actions)

Identical roots, successors, predictions, target count, compute and terminal mass; only
the weighting differs. It is transparently class-weighted successor MSE, not a new loss
and not a death classifier. The weights are uncapped and reported instead: they span
0.0312 to 0.5000 against 0.0588 uniform, because 239 roots have a single lethal action
and 497 have a single safe one, and a cap is a knob that invites tuning the arm until it
wins. Because that population is bimodal -- 87.8% of roots carry 13 or more lethal
successors, 10.6% carry two or fewer -- read the escape-rich and trap-heavy strata
alongside the aggregate.


Extendable, not merely resumable
--------------------------------

`sample_batch` reads the short/long curriculum off the *declared total*, so passing the
requested stopping point made a 40,000-step run a different experiment from step zero
rather than a continuation of a 20,000-step one. `--horizon` fixes the curriculum to its
own 20,000-step schedule regardless of where the run stops: an extension past the
horizon simply stays in the terminal long-only regime, which is where the schedule had
already put it. The stopping point is therefore not part of the resume contract, and
27,184 steps -- exactly two complete passes at four targets per step -- is a meaningful
first extension in a way that a round 40,000 is not.

The seed-0 arms already on disk cannot be extended by any of this: they were trained
before the resume path existed and hold weights only, with no optimizer moments, no
running-RMS balance and no generator states. Extension starts with the new-seed matrix.

`--seed` moves initialisation, the sampler, model noise, the terminal stream and the
root schedule together. It deliberately does not touch the corpus split, which is baked
into the latent cache this reads: the 197 evaluation roots were chosen under the default
split, and reshuffling it would move them into training and invalidate every paired
comparison against the existing arms.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.data import load_episodes, sample_batch
from d4mj.train import (_balance, _checkpoint, _generators, _share_initialisation, _to,
                        _update, optimizer)
from d4mj.state import WorldState
from d4mj.transition import World, advance, commit_inputs, transition_loss

DEVICE = "cuda"
CACHE = HERE / "latent_cache_64"
N_ACTIONS = 17


EXPECTED_ROOTS = 3198
EXPECTED_BROAD = 3651


def load_roots():
    """Every completeness check that a partial cache would otherwise slip past.

    A first version loaded whatever shards happened to exist, so an interrupted encode
    would have silently run the whole experiment on a fraction of the roots.
    """
    manifest_path = HERE / "actionable_latents" / "manifest.json"
    assert manifest_path.exists(), "actionable_latents/manifest.json missing: encode first"
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("z_history_source"), (
        "z_history has not been aligned to the production cache; run "
        "align_actionable_latents.py")

    shards = sorted(glob.glob(str(HERE / "actionable_latents" / "shard-*.pt")))
    assert len(shards) == manifest["shards"], (
        f"{len(shards)} shards on disk, manifest declares {manifest['shards']}")
    rows = []
    for path in shards:
        rows += torch.load(path, weights_only=False)
    assert len(rows) == EXPECTED_ROOTS == manifest["roots"], (
        f"{len(rows)} roots loaded, expected {EXPECTED_ROOTS}, "
        f"manifest declares {manifest['roots']}")

    table_path = HERE / "actionable_actions.pt"
    assert table_path.exists(), "actionable_actions.pt missing: run build_actionable_actions.py"
    table = torch.load(table_path, weights_only=False)
    keys = [(int(r["shard"]), int(r["slot"])) for r in rows]
    assert len(set(keys)) == len(keys), "duplicate roots in the latent shards"
    missing = [k for k in keys if k not in table]
    assert not missing, f"{len(missing)} roots without an action history"

    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])
    assert torch.isfinite(history).all() and torch.isfinite(branch).all(), "non-finite latents"
    terminated = torch.stack([r["terminated"] for r in rows]).bool()
    lethal = torch.tensor([r["lethal_action"] for r in rows])
    # actionability is defined as at least one surviving alternative, and the logged
    # action is fatal by construction, so neither half of the 50/50 split is ever empty
    assert terminated[torch.arange(len(rows)), lethal].all(), "a logged action survived"
    assert (terminated.sum(1) < N_ACTIONS).all(), "a root offers no safe action"
    return (history, branch, torch.stack([table[k] for k in keys]), lethal, terminated)


def repeat_memory(memory, roots: int, actions: int):
    """Repeat each root's memory across the actions it carries.

    The world's memory is laid out (roots x tokens, heads, time, dim) -- token-major, not
    batch-major -- so `repeat_interleave(dim=0)` on it interleaves token positions and
    silently produces a memory that belongs to no root. It has to be unflattened to
    (roots, tokens, ...) first.
    """
    out = []
    for pair in memory:
        widened = []
        for tensor in pair:
            rest = tensor.shape[1:]
            widened.append(tensor.view(roots, -1, *rest)
                           .repeat_interleave(actions, dim=0).reshape(-1, *rest))
        out.append(tuple(widened))
    return tuple(out)


def v2_roots():
    """The v2 corpus: all 17 first successors and a second NOOP successor wherever the
    first left the branch alive. `second_valid` travels with it and gates the loss, so a
    zeroed post-terminal slot can never be trained on."""
    import glob as _glob

    rows = []
    for path in sorted(_glob.glob(str(HERE / "broad_latents_v2" / "shard-*.pt"))):
        rows += torch.load(path, weights_only=False)
    assert rows, "no v2 latents; run encode_broad_forks.py"
    history = torch.stack([r["z_history"] for r in rows]).float()
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    second = torch.stack([r["z_second"] for r in rows]).float()
    valid = torch.stack([r["second_valid"] for r in rows]).bool()
    terminated = torch.stack([r["terminated"] for r in rows]).bool()
    led = torch.stack([r["led_to_action"] for r in rows]).long()
    assert (second[~valid].abs().sum() == 0), "an invalid second target is not zero"
    assert torch.isfinite(history).all() and torch.isfinite(branch).all()
    return history, branch, led, terminated.float().argmax(1), terminated, second, valid


def broad_roots():
    """The wider all-action fit population -- nearly the inverse of the terminal tails.

    The terminal corpus is 87.8% trap-heavy and holds no all-safe root at all, because
    every one of its roots is a recorded terminal tail; 77.9% of its (root, action) pairs
    are fatal. This population is 81.7% all-safe and 10.0% fatal overall. It is the fit
    split of the same forkset the evaluator scores against, so the whole-seed split that
    holds out evaluation roots is respected unchanged.
    """
    from train_phase1b_fork import fork_actions, load_forkset, seed_split

    rows = [r for r in load_forkset(HERE / "forkset_s1_n64")
            if seed_split(int(r["seed"])) == "fit"]
    assert len(rows) == EXPECTED_BROAD, f"{len(rows)} fit roots, expected {EXPECTED_BROAD}"
    history = torch.stack([r["z_history"] for r in rows]).float()
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    terminated = torch.stack([r["terminated"] for r in rows]).bool()
    assert torch.isfinite(history).all() and torch.isfinite(branch).all(), "non-finite latents"
    # `lethal` is only read by the factual arm, which is terminal-only by construction:
    # a root with no fatal action has no logged fatal transition to supervise
    return history, branch, fork_actions(rows), terminated.float().argmax(1), terminated


def regimes(terminated: torch.Tensor):
    """all-safe, escape-rich and trap-heavy, on the cut the evaluation uses."""
    n = terminated.sum(1).numpy()
    return {"all-safe": np.where(n == 0)[0], "escape-rich": np.where((n >= 1) & (n <= 2))[0],
            "trap-heavy": np.where(n >= 14)[0], "middle": np.where((n >= 3) & (n <= 13))[0]}


def regime_schedule(terminated: torch.Tensor, seed: int, presentations: int):
    """Equal long-run mass across the three regimes, each cycling its own shuffled order.

    The middle band is excluded rather than folded into a neighbour: it holds 11 of 3,651
    roots, too few to carry a third of the mass and too ambiguous to assign.

    Equal mass is not equal repetition, and the difference is the whole point of naming
    it. The three groups are 2,983 / 273 / 384 roots, so an equal third of the
    presentations repeats an escape-rich root about eleven times more often than an
    all-safe one. That is the intended intervention, and it is printed at startup so
    "balanced" cannot again hide what exposure actually means.
    """
    generator = np.random.default_rng(seed)
    groups = regimes(terminated)
    names = ("all-safe", "escape-rich", "trap-heavy")
    cycles = [groups[k][generator.permutation(len(groups[k]))] for k in names]
    position = [0, 0, 0]
    order = np.empty(presentations, dtype=np.int64)
    for i in range(presentations):
        k = i % 3
        if position[k] == len(cycles[k]):
            cycles[k] = cycles[k][generator.permutation(len(cycles[k]))]
            position[k] = 0
        order[i] = cycles[k][position[k]]
        position[k] += 1
    share = np.array([(order[i::3].size) for i in range(3)])
    assert share.max() - share.min() <= 1, "regimes did not receive equal mass"
    assert not np.isin(order, groups["middle"]).any(), "a middle root entered the schedule"
    return order, np.tile(np.arange(N_ACTIONS), (presentations, 1))


def schedule(n_roots: int, seed: int, actions: int):
    """A deterministic shuffled pass over every (root, action) pair.

    Both forms walk the *same root order* -- the root column of the one shuffled pass
    over all 54,366 pairs -- so the arms are step-for-step root-identical and differ
    only in how many of that root's successors are supervised when it comes up.
    `actions == 1` takes the action the pair schedule paired it with; `actions == 17`
    reads all seventeen off the one encoder forward. Deriving the 17-action form from
    its own root permutation instead would have moved root presentations as well as
    target throughput, and no result could then be attributed to either.

    Widths between 1 and 17 are refused: 17 is prime, so every other chunk size leaves
    a ragged final group, which either repeats pairs inside a pass or makes the batch
    shape depend on the step.
    """
    pairs = np.stack(np.meshgrid(np.arange(n_roots), np.arange(N_ACTIONS),
                                 indexing="ij"), axis=-1).reshape(-1, 2)
    pairs = pairs[np.random.default_rng(seed).permutation(len(pairs))]
    assert len(pairs) == n_roots * N_ACTIONS
    assert len({tuple(x) for x in pairs}) == len(pairs), "schedule repeats a pair"
    order = pairs[:, 0]
    if actions == 1:
        choice = pairs[:, 1:2]
    else:
        assert actions == N_ACTIONS, "1 or 17 actions per root only; 17 is prime"
        choice = np.tile(np.arange(N_ACTIONS), (len(order), 1))
    assert (np.bincount(order, minlength=n_roots) == N_ACTIONS).all(), (
        "some root is not presented seventeen times per traversal")
    assert (np.bincount(choice.reshape(-1), minlength=N_ACTIONS)
            == choice.size // N_ACTIONS).all(), "actions are not uniform over the pass"
    return order, choice


def data_identity(source: str) -> str:
    """What the arms are trained on, folded into the resume contract. `Config` cannot
    see it: the roots, their aligned histories and their action tables all live on
    disk, and a resume across a re-encoded cache would splice two datasets.

    It must follow the source in use. Hashing only the terminal-tail files, as this did,
    left a broad-data run able to resume across a changed forkset without failing --
    exactly the splice the digest exists to prevent.
    """
    digest = hashlib.sha256()
    paths = [CACHE / "manifest.json"]
    paths += ([HERE / "forkset_s1_n64" / "manifest.json", HERE / "fork_actions.pt"]
              if source == "broad" else
              [HERE / "actionable_latents" / "manifest.json", HERE / "actionable_actions.pt"])
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("factual", "counterfactual"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=20000,
                        help="the curriculum's own length, fixed independently of where "
                             "the run stops, so an extension continues one schedule "
                             "instead of splicing two")
    parser.add_argument("--seed", type=int, default=None,
                        help="initialisation, sampler, model noise, terminal stream and "
                             "root schedule; never the corpus split, which the cache fixes")
    parser.add_argument("--terminal-roots", type=int, default=4,
                        help="roots per step, one encoder forward each")
    parser.add_argument("--terminal-actions", type=int, default=1, choices=(1, N_ACTIONS),
                        help="action-conditioned targets read off each root's features")
    parser.add_argument("--terminal-mass", type=float, default=0.2)
    parser.add_argument("--roots", choices=("terminal", "broad", "v2"), default="terminal",
                        help="terminal tails, the 3,651-root fit population, or the v2 "
                             "corpus with second-step targets")
    parser.add_argument("--time-mixer", choices=("attention", "mamba"), default="attention",
                        help="the Stage-A T-versus-M arm; `_share_initialisation` gives "
                             "both the same starting weights wherever they share a "
                             "parameter, which manual_seed alone does not")
    parser.add_argument("--second-weight", type=float, default=0.5,
                        help="weight on the second generated state, matching "
                             "`_direct_loss`, which averages its two rollout terms")
    parser.add_argument("--regime-balance", action="store_true",
                        help="equal long-run mass across all-safe, escape-rich and "
                             "trap-heavy roots, rather than uniform over roots")
    parser.add_argument("--balance-outcomes", action="store_true",
                        help="weight each root's seventeen successors so the lethal and "
                             "surviving halves carry half the root's loss each")
    parser.add_argument("--milestones", type=int, nargs="+",
                        default=(5000, 10000, 13592, 20000))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true",
                        help="marks the contract, so a smoke checkpoint can never be "
                             "mistaken for a real run's resume point")
    parser.add_argument("--overwrite", action="store_true")
    # Dreamer 4 reports C = 3*T_short and T_long = 4*T_short, which `Config` asserts, so
    # the short length determines the other two. A 16-step rollout needs a longer row: at
    # sequence 16 it would leave no observed prefix at all, and the assert would reject it.
    parser.add_argument("--sequence", type=int, default=None)
    parser.add_argument("--direct-rollout", type=int, default=None)
    parser.add_argument("--align-weight", type=float, default=0.0,
                        help="stop-gradient pull of the generated readout onto the "
                             "observed one at the matched rollout positions")
    args = parser.parse_args()
    out = args.out = args.out or HERE / f"terminal_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)

    # the factual arm supervises one target -- its logged fatal transition -- so
    # seventeen copies of it would be the same gradient with a longer runtime
    assert args.arm == "counterfactual" or args.terminal_actions == 1, (
        "the factual arm has one successor per root; --terminal-actions 17 would "
        "average seventeen copies of it")
    assert args.roots != "v2" or args.terminal_actions == N_ACTIONS, (
        "the v2 corpus is an all-action corpus")
    assert args.roots == "terminal" or args.arm == "counterfactual", (
        "the factual arm supervises a root's logged fatal transition, which a root with "
        "no fatal action does not have")
    assert args.roots == "terminal" or args.terminal_actions == N_ACTIONS, (
        "the broader population is an all-action corpus")
    assert not args.regime_balance or args.roots == "broad", (
        "regime balancing needs the wider population; the terminal tails hold no "
        "all-safe root at all")
    assert not args.balance_outcomes or args.terminal_actions == N_ACTIONS, (
        "balancing reweights the seventeen successors against each other and is "
        "undefined when only one of them is supervised per presentation")

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    if args.sequence is not None:
        base = replace(base, sequence=args.sequence,
                       dynamics_context=3 * args.sequence,
                       sequence_long=4 * args.sequence)
    if args.direct_rollout is not None:
        base = replace(base, direct_rollout=args.direct_rollout)
    config = replace(base, transition="direct", time_mixer=args.time_mixer,
                     align_weight=args.align_weight)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)
    second = valid = None
    if args.roots == "v2":
        history, branch, led_history, lethal, terminated, second, valid = v2_roots()
    elif args.roots == "broad":
        history, branch, led_history, lethal, terminated = broad_roots()
    else:
        history, branch, led_history, lethal, terminated = load_roots()
    if args.regime_balance:
        order, choice = regime_schedule(terminated, config.seed + 7,
                                        args.steps * args.terminal_roots)
    else:
        order, choice = schedule(len(history), config.seed + 7, args.terminal_actions)
    targets = args.terminal_roots * args.terminal_actions
    n_pairs = len(history) * N_ACTIONS
    passes = args.steps * targets / n_pairs
    per_root = args.steps * args.terminal_roots / len(history)

    # 0.5/n on each side, so a root with one lethal action puts half its loss on that
    # one target. Reported rather than capped: a cap is a knob, and a knob invites
    # tuning the arm until it wins.
    dead = terminated.sum(1)
    weights = torch.where(terminated, 0.5 / dead[:, None].float(),
                          0.5 / (N_ACTIONS - dead)[:, None].float())
    print(f"{args.arm}: {len(episodes)} cached episodes, {len(history)} roots, seed "
          f"{config.seed}, {targets} targets/step, {per_root:.1f} presentations per root, "
          f"{passes:.2f} passes over {n_pairs:,} pairs in {args.steps:,} steps", flush=True)
    print(f"  lethal successors per root {dead.float().mean():.2f}/17, "
          f"{terminated.float().mean():.3f} of all pairs", flush=True)
    groups = regimes(terminated)
    # count what the run actually consumes, not what the schedule array holds: the
    # uniform schedule is 62,067 long and a 20,000-step run walks 80,000 presentations
    # through it, so the array's own bincount understates exposure by the wrap factor
    consumed = order[np.arange(args.steps * args.terminal_roots) % len(order)]
    seen = np.bincount(consumed, minlength=len(history))
    for name in ("all-safe", "escape-rich", "middle", "trap-heavy"):
        index = groups[name]
        mass = seen[index].sum() / max(seen.sum(), 1)
        each = seen[index].mean() if len(index) else 0.0
        print(f"  {name:<12}{len(index):>6} roots{mass:>8.1%} of presentations"
              f"{each:>8.1f} each", flush=True)
    if args.balance_outcomes:
        print(f"  balanced weights span {weights.min():.4f} to {weights.max():.4f} "
              f"against {1/N_ACTIONS:.4f} uniform "
              f"({weights.max()*N_ACTIONS:.1f}x at the extreme)", flush=True)

    # identical to train_dynamics: same seed, same construction, same streams
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(DEVICE)
    opt = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, rng = _generators(config, 1)
    terminal_rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    spatial, d = config.n_spatial, config.d_spatial

    # A resume needs optimizer moments, the running-RMS balance and every stream, not
    # weights. The contract carries everything `Config` cannot see and that changing
    # would make the continuation a different experiment -- the sampler, the loss mass,
    # the data on disk and the curriculum's own horizon. It deliberately excludes the
    # stopping point: that is exactly what an extension is allowed to change.
    resume_path = out / "resume.pt"
    streams = {"sampler": sampler, "model": rng, "terminal": terminal_rng}
    contract = ":".join(["terminal", args.arm, f"seed{config.seed}",
                         f"roots{args.terminal_roots}", f"actions{args.terminal_actions}",
                         f"mass{args.terminal_mass:g}", args.roots,
                         f"second{args.second_weight:g}", args.time_mixer,
                         "regime" if args.regime_balance else "flat",
                         "balanced" if args.balance_outcomes else "uniform",
                         f"horizon{args.horizon}",
                         data_identity(args.roots),
                         "smoke" if args.smoke else "run"])
    begin = _checkpoint(resume_path if resume_path.exists() else None, config,
                        [world, opt], balance, streams, contract=contract)
    if begin:
        print(f"resumed from step {begin} of {args.steps}", flush=True)
    elif (out / "world.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out} already holds a finished run and there is no resume "
                         f"point to continue from; pass --overwrite to discard it")
    if begin >= args.steps:
        raise SystemExit(f"the checkpoint is already at step {begin}")

    started, curve = time.time(), []
    for step in range(begin, args.steps):
        batch = _to(sample_batch(episodes, sampler, config, step, args.horizon), DEVICE)
        dynamics = transition_loss(world, batch, rng, config, step=step)

        window = (step * args.terminal_roots
                  + np.arange(args.terminal_roots)) % len(order)
        roots = torch.from_numpy(order[window])
        action = (lethal[roots][:, None] if args.arm == "factual"
                  else torch.from_numpy(choice[window]))
        z = history[roots].to(DEVICE)
        n, t = z.shape[0], z.shape[1]
        committed, conditioning = commit_inputs(z.view(n, t, spatial, d),
                                                terminal_rng, config)
        # the memory is kept: the second generated state has to continue this history
        # through `advance`, and a state built with memory=None trains a history-free
        # transition that is not the production recurrence at all
        features, _, memory = world(None, led_history[roots].to(DEVICE), committed,
                                    conditioning)

        # the history and the backbone pass are what cost anything, so every action a
        # root carries reads off the same features; `predict` is a pool, a one-block
        # mixer and a readout, and is the only part that has to run per action
        a = action.shape[1]
        flat = action.reshape(-1, 1)
        target = branch[roots.repeat_interleave(a), flat[:, 0]].to(DEVICE)
        error = (world.predict(features[:, -1:].repeat_interleave(a, dim=0),
                               flat.to(DEVICE)).flatten(2)[:, 0] - target).pow(2)
        if args.balance_outcomes:
            # half the root's loss on its lethal successors, half on its survivors,
            # then the mean across roots -- the same targets, reweighted
            share = weights[roots].reshape(-1).to(DEVICE)
            terminal = (share * error.mean(-1)).sum() / len(roots)
        else:
            terminal = error.mean()   # the unbalanced path is left exactly as it was

        if second is not None:
            # Both generated states through `advance`, exactly as `_direct_loss` rolls
            # them: the carried memory, the step offset, and the first accepted latent
            # feeding the second. Scored only where the branch survived its first action.
            keep = valid[roots].reshape(-1).to(DEVICE)
            if bool(keep.any()):
                carried = repeat_memory(memory, n, a)
                state = WorldState(z[:, -1:].view(n, 1, spatial, d).repeat_interleave(a, 0),
                                   carried, t, features[:, -1:].repeat_interleave(a, dim=0))
                first, _ = advance(world, state, flat.to(DEVICE), terminal_rng, config)
                noop = torch.zeros_like(flat).to(DEVICE)
                secondly, _ = advance(world, first, noop, terminal_rng, config)
                target2 = second[roots.repeat_interleave(a), flat[:, 0]].to(DEVICE)
                error2 = (secondly.latent.flatten(2)[:, 0] - target2).pow(2).mean(-1)
                terminal = terminal + args.second_weight * (error2 * keep).sum() / keep.sum()

        blended = (1.0 - args.terminal_mass) * dynamics + args.terminal_mass * terminal
        _update(opt, _balance({"dynamics": blended}, balance, config), [world], config, step)
        curve.append({"step": step + 1, "dynamics": float(dynamics.detach()),
                      "terminal": float(terminal.detach())})
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            recent = curve[-500:]
            print(f"  {step+1}/{args.steps} "
                  f"dyn={sum(c['dynamics'] for c in recent)/len(recent):.5f} "
                  f"term={sum(c['terminal'] for c in recent)/len(recent):.5f} "
                  f"[{time.time()-started:.0f}s]", flush=True)
        if (step + 1) in tuple(args.milestones):
            save(out / f"world_{step+1:06d}.pt", config, part0=world)
        if (step + 1) % args.resume_every == 0 or step + 1 == args.steps:
            _checkpoint(resume_path, config, [world, opt], balance, streams,
                        step + 1, contract)

    torch.save({"world": world.state_dict()}, out / "world.pt")
    (out / "training_report.json").write_text(json.dumps(
        {"arm": args.arm, "steps": args.steps, "horizon": args.horizon,
         "seed": config.seed, "roots": len(history), "pairs": n_pairs,
         "passes": passes, "presentations_per_root": per_root,
         "terminal_roots": args.terminal_roots,
         "terminal_actions": args.terminal_actions, "targets_per_step": targets,
         "terminal_mass": args.terminal_mass, "balance_outcomes": args.balance_outcomes,
         "root_source": args.roots, "regime_balance": args.regime_balance,
         "time_mixer": args.time_mixer, "second_weight": args.second_weight,
         "align_weight": args.align_weight, "direct_rollout": config.direct_rollout,
         "sequence": config.sequence, "sequence_long": config.sequence_long,
         "dynamics_context": config.dynamics_context,
         "regime_presentations": {k: int(seen[v].sum()) for k, v in groups.items()},
         "regime_repeats_per_root": {k: (float(seen[v].mean()) if len(v) else 0.0)
                                     for k, v in groups.items()},
         "lethal_fraction": float(terminated.float().mean()),
         "weight_min": float(weights.min()), "weight_max": float(weights.max()),
         "weight_uniform": 1 / N_ACTIONS, "contract": contract,
         "resumed_from": begin, "seconds": time.time() - started,
         "curve_tail": curve[-500:]}, indent=2))
    print(f"done in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
