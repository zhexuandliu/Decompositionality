"""Deterministic online training stream and fixed evaluation manifests.

Design
------
* The training stream is *stateless*: logical optimizer batch ``step`` is a pure
  function of ``(seed, step, train strings, effective batch size, held-out set)``.
  Model initialization, validation, microbatching, OOM retries, profiling and
  resumption therefore cannot change which raw examples are seen.
* Separate RNG streams (distinct SeedSequence keys) are used for grids, operation
  selection, and PAD placement.
* Input grids depend only on ``(seed, step)``, so both experiments see the identical
  input-grid sequence; only the operation schedule differs.
* Operation strings follow a deterministic balanced schedule over the *global*
  sample index: each consecutive block of ``k = len(train_strings)`` samples is a
  random permutation of the k strings. Counts after any prefix differ by at most one.
* Validation/test base grids come from independent streams, are unique, mutually
  disjoint, and excluded from training (any colliding training grid is
  deterministically resampled).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import operations as O

DATA_VERSION = "decomp-gridops-v1"

STREAM_GRID = 101
STREAM_GRID_RESAMPLE = 102
STREAM_OPS = 201
STREAM_PAD = 301
STREAM_VAL = 401
STREAM_TEST = 501
STREAM_OVERFIT = 601  # debugging-only diagnostic; never used by the benchmark stream


def make_rng(seed: int, stream: int, *keys: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(stream), *map(int, keys)])))


def grid_bytes(g: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(g, dtype=np.uint8)).tobytes()


def grid_sha256(g: np.ndarray) -> str:
    return hashlib.sha256(grid_bytes(g)).hexdigest()


def random_grids(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, O.NUM_DIGITS, size=(n, O.GRID_SIZE, O.GRID_SIZE), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Held-out base grids
# ---------------------------------------------------------------------------

def _unique_grids(rng: np.random.Generator, n: int, exclude: set) -> np.ndarray:
    out: List[np.ndarray] = []
    seen = set(exclude)
    while len(out) < n:
        for g in random_grids(rng, n):
            b = grid_bytes(g)
            if b not in seen:
                seen.add(b)
                out.append(g)
                if len(out) == n:
                    break
    return np.stack(out)


def build_heldout_grids(seed: int, n_val: int = 300, n_test: int = 1000) -> Dict[str, np.ndarray]:
    val = _unique_grids(make_rng(seed, STREAM_VAL), n_val, set())
    test = _unique_grids(make_rng(seed, STREAM_TEST), n_test, {grid_bytes(g) for g in val})
    return {"val_grids": val, "test_grids": test}


def aggregate_hash(grids: np.ndarray) -> str:
    h = hashlib.sha256()
    for g in grids:
        h.update(grid_bytes(g))
    return h.hexdigest()


@dataclass
class Manifests:
    seed: int
    val_grids: np.ndarray
    test_grids: np.ndarray
    info: dict

    @property
    def heldout_bytes(self) -> set:
        return {grid_bytes(g) for g in np.concatenate([self.val_grids, self.test_grids])}

    @property
    def manifest_hash(self) -> str:
        return self.info["combined_hash"]


def _manifest_info(seed: int, val: np.ndarray, test: np.ndarray) -> dict:
    return {
        "data_version": DATA_VERSION,
        "seed": seed,
        "n_val_grids": int(len(val)),
        "n_test_grids": int(len(test)),
        "streams": {"val": STREAM_VAL, "test": STREAM_TEST, "grid": STREAM_GRID,
                    "grid_resample": STREAM_GRID_RESAMPLE, "ops": STREAM_OPS, "pad": STREAM_PAD},
        "rng": "numpy PCG64 seeded by SeedSequence([seed, stream, *keys])",
        "ordering": "grid_id = row index in val_grids/test_grids; eval examples are grid-major, "
                    "string-minor in the listed string order",
        "val_hash": aggregate_hash(val),
        "test_hash": aggregate_hash(test),
        "combined_hash": hashlib.sha256((aggregate_hash(val) + aggregate_hash(test)).encode()).hexdigest(),
        "val_grid_sha256": [grid_sha256(g) for g in val],
        "test_grid_sha256": [grid_sha256(g) for g in test],
        "test_strings": O.TEST_STRINGS,
        "train_sets": O.TRAIN_SETS,
    }


def validate_manifests(m: Manifests) -> None:
    vb = [grid_bytes(g) for g in m.val_grids]
    tb = [grid_bytes(g) for g in m.test_grids]
    assert len(set(vb)) == len(vb), "duplicate validation grids"
    assert len(set(tb)) == len(tb), "duplicate test grids"
    assert not (set(vb) & set(tb)), "validation/test overlap"
    assert m.val_grids.dtype == np.uint8 and m.val_grids.max() < 10
    assert m.info["val_grid_sha256"] == [hashlib.sha256(b).hexdigest() for b in vb]
    assert m.info["test_grid_sha256"] == [hashlib.sha256(b).hexdigest() for b in tb]


def load_or_build_manifests(directory: Optional[os.PathLike], seed: int, n_val: int = 300,
                            n_test: int = 1000) -> Manifests:
    """Build (deterministically) and persist manifests, or load and verify saved ones."""
    built = build_heldout_grids(seed, n_val, n_test)
    m = Manifests(seed, built["val_grids"], built["test_grids"],
                  _manifest_info(seed, built["val_grids"], built["test_grids"]))
    validate_manifests(m)
    if directory is None:
        return m
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    npz, js = d / "heldout_grids.npz", d / "manifest.json"
    if npz.exists() and js.exists():
        saved = np.load(npz)
        info = json.loads(js.read_text())
        if (not np.array_equal(saved["val_grids"], m.val_grids)
                or not np.array_equal(saved["test_grids"], m.test_grids)
                or info.get("combined_hash") != m.manifest_hash
                or info.get("data_version") != DATA_VERSION):
            raise RuntimeError(f"Saved manifests in {d} do not match the deterministic regeneration "
                               "(different seed/version/size?). Use a fresh output directory.")
        return m
    from .utils import atomic_write_bytes, atomic_write_text
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, val_grids=m.val_grids, test_grids=m.test_grids)
    atomic_write_bytes(npz, buf.getvalue())
    atomic_write_text(js, json.dumps(m.info, indent=1))
    return m


# ---------------------------------------------------------------------------
# Evaluation sets
# ---------------------------------------------------------------------------

def make_eval_set(base_grids: np.ndarray, strings: Sequence[str], split: str) -> Dict[str, np.ndarray]:
    """Grid-major x string-minor cross product with canonical right padding."""
    n, k = len(base_grids), len(strings)
    grid_id = np.repeat(np.arange(n), k)
    sidx = np.tile(np.arange(k), n)
    strs = np.array(strings, dtype=object)[sidx]
    x = base_grids[grid_id]
    y = O.apply_strings_batch(x, strs)
    ops = np.array([O.slots_for_string(s) for s in strings], dtype=np.int64)[sidx]
    return {"x": x.astype(np.uint8), "y": y.astype(np.uint8), "ops": ops, "grid_id": grid_id,
            "string": np.array([str(s) for s in strs]), "split": split}


def subset_eval_set(es: Dict[str, np.ndarray], n_grids: int) -> Dict[str, np.ndarray]:
    keep = es["grid_id"] < n_grids
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in es.items()}


# ---------------------------------------------------------------------------
# Training stream
# ---------------------------------------------------------------------------

class TrainStream:
    """Stateless generator: ``batch(step)`` always returns the same logical batch."""

    def __init__(self, seed: int, train_strings: Sequence[str], effective_batch_size: int,
                 heldout_bytes: Optional[Iterable[bytes]] = None):
        self.seed = int(seed)
        self.strings = list(train_strings)
        assert all(len(s) == 2 for s in self.strings), "training strings must have length exactly two"
        assert len(set(self.strings)) == len(self.strings)
        self.k = len(self.strings)
        self.B = int(effective_batch_size)
        self.heldout = set(heldout_bytes or [])

    def _string_indices(self, start: int, n: int) -> np.ndarray:
        idx = np.arange(start, start + n)
        blocks = idx // self.k
        out = np.empty(n, dtype=np.int64)
        for b in np.unique(blocks):
            perm = make_rng(self.seed, STREAM_OPS, int(b)).permutation(self.k)
            sel = blocks == b
            out[sel] = perm[idx[sel] % self.k]
        return out

    def _grids(self, step: int) -> np.ndarray:
        g = random_grids(make_rng(self.seed, STREAM_GRID, step), self.B)
        if self.heldout:
            for j in range(self.B):
                attempt = 0
                while grid_bytes(g[j]) in self.heldout:
                    g[j] = random_grids(make_rng(self.seed, STREAM_GRID_RESAMPLE, step, j, attempt), 1)[0]
                    attempt += 1
        return g

    def batch(self, step: int) -> Dict[str, np.ndarray]:
        step = int(step)
        x = self._grids(step)
        sidx = self._string_indices(step * self.B, self.B)
        strs = np.array(self.strings)[sidx]
        placement = make_rng(self.seed, STREAM_PAD, step).integers(0, 3, size=self.B)
        ops = np.array([O.pair_slots(s, int(p)) for s, p in zip(strs, placement)], dtype=np.int64)
        y = O.apply_strings_batch(x, strs)
        return {"x": x, "y": y.astype(np.uint8), "ops": ops, "string": strs, "placement": placement,
                "sample_index": np.arange(step * self.B, (step + 1) * self.B)}


def overfit_examples(seed: int, train_strings: Sequence[str], n: int, heldout_bytes: set) -> Dict[str, np.ndarray]:
    """Small fixed set of length-two examples for the overfit DIAGNOSTIC only."""
    rng = make_rng(seed, STREAM_OVERFIT)
    grids = _unique_grids(rng, n, set(heldout_bytes))
    strs = np.array([train_strings[i % len(train_strings)] for i in range(n)])
    placement = rng.integers(0, 3, size=n)
    ops = np.array([O.pair_slots(s, int(p)) for s, p in zip(strs, placement)], dtype=np.int64)
    return {"x": grids, "y": O.apply_strings_batch(grids, strs).astype(np.uint8), "ops": ops,
            "string": strs, "grid_id": np.arange(n), "split": "overfit"}
