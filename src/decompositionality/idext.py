"""Identity-format operation-COUNT extrapolation (notebooks 12/13): Transformer AR and Looped AR only.

Five visible operation slots per instruction (unused slots = learnable <identity>, no PAD), exactly as in the
successful identity_02345 experiment (notebook 10); only the training SUPPORT over k (number of A/B/C tokens) differs:

  idext_train02 : k in {0, 2}      -> 128 + 128 records per update            (640,000 each over 5000 updates)
  idext_train023: k in {0, 2, 3}   -> 85 each + 1 extra rotating 0,2,3,...    (426,667 / 426,667 / 426,666)

Records are indexed WITHIN k: the j-th record of length k is a pure function of (seed, k, j) (word uniform over
{A,B,C}^k, ordered slot subset uniform over C(5,k), fresh grid). Hence the common k = 0/2 records of the two
experiments share one ordered per-k stream, while their exposure counts and batch positions differ. train023 is NOT
train02 plus triples: it replaces some pair/EMPTY exposure with triples at an equal total budget.
The final test manifest is the notebook-10 identity test manifest (376,000 records), shared by all four runs.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import idvar as V
from .data import STREAM_GRID_RESAMPLE, grid_bytes, make_rng, random_grids
from .words import to_label, words_of_length

IDEXT_EXPERIMENTS = {"idext_train02": (0, 2), "idext_train023": (0, 2, 3)}
IDEXT_DIRS = {"idext_train02": "train02", "idext_train023": "train023"}
IDEXT_RESULTS_SUBDIR = "identity_ar_count_extrapolation"
IDEXT_MODELS = ["transformer_ar", "looped_transformer_ar"]
IDEXT_DATA_VERSION = "decomp-idext-v1"

STREAM_EXT_ORDER = 901
STREAM_EXT_RECORD = 902
STREAM_EXT_GRID = 903
STREAM_EXT_VALSLOT = 911


def is_idext(experiment: str) -> bool:
    return experiment in IDEXT_EXPERIMENTS


def train_ks(experiment: str) -> tuple:
    return IDEXT_EXPERIMENTS[experiment]


def training_words(experiment: str) -> List[str]:
    return [w for k in train_ks(experiment) for w in words_of_length(k)]


def training_labels(experiment: str) -> List[str]:
    return [to_label(w) for w in training_words(experiment)]


def support_counts(experiment: str) -> dict:
    pc = V.pattern_counts()
    ks = train_ks(experiment)
    return {"logical_words": sum(3 ** k for k in ks), "physical_patterns": sum(pc[k] for k in ks)}


def per_update_counts(ks: Sequence[int], B: int, step: int) -> Dict[int, int]:
    K = len(ks)
    base, extra = B // K, B - (B // K) * K
    c = {k: base for k in ks}
    for e in range(extra):
        c[ks[(step * extra + e) % K]] += 1
    return c


def records_before(ks: Sequence[int], B: int, step: int, k: int) -> int:
    """Number of length-k records in updates 0..step-1 (closed form; rotating extras start at ks[0] on update 0)."""
    K = len(ks)
    base, extra = B // K, B - (B // K) * K
    i = list(ks).index(k)
    n = step * extra
    return step * base + (max(0, (n - i + K - 1) // K) if extra else 0)


def record(seed: int, k: int, j: int, heldout=None):
    """The j-th record of length k (pure function): word, ordered slot subset, grid."""
    rng = make_rng(seed, STREAM_EXT_RECORD, k, j)
    word = "".join("ABC"[c] for c in rng.integers(0, 3, k))
    slots = tuple(sorted(np.argsort(rng.random(5))[:k].tolist()))
    g = random_grids(make_rng(seed, STREAM_EXT_GRID, k, j), 1)[0]
    attempt = 0
    while heldout and grid_bytes(g) in heldout:
        g = random_grids(make_rng(seed, STREAM_GRID_RESAMPLE, 9000 + k, j, attempt), 1)[0]
        attempt += 1
    return word, slots, g


class ExtStream:
    """Stateless: batch(step) depends only on (seed, step, B, train ks, held-out grids)."""

    def __init__(self, seed: int, experiment: str, effective_batch_size: int, heldout_bytes=None):
        self.seed, self.experiment, self.B = int(seed), experiment, int(effective_batch_size)
        self.ks = train_ks(experiment)
        self.heldout = set(heldout_bytes or [])

    def batch(self, step: int) -> Dict[str, np.ndarray]:
        step = int(step)
        cnt = per_update_counts(self.ks, self.B, step)
        recs = []
        for k in self.ks:
            j0 = records_before(self.ks, self.B, step, k)
            for j in range(j0, j0 + cnt[k]):
                recs.append((k, j) + record(self.seed, k, j, self.heldout))
        recs = [recs[i] for i in make_rng(self.seed, STREAM_EXT_ORDER, step).permutation(len(recs))]  # mixed batch
        ks = np.array([r[0] for r in recs])
        words = [r[2] for r in recs]
        x = np.stack([r[4] for r in recs]).astype(np.uint8)
        return {"x": x, "y": V.apply_words(x, words).astype(np.uint8),
                "slots5": np.stack([V.slots5_for(w, r[3]) for w, r in zip(words, recs)]),
                "k": ks, "group": ks, "length": ks, "string": np.array([to_label(w) for w in words]),
                "k_index": np.array([r[1] for r in recs]),
                "sample_id": np.array([f"k{r[0]}:{r[1]}" for r in recs]), "condition": "identity"}


def make_val_set(val_grids: np.ndarray, seed: int, experiment: str) -> Dict[str, np.ndarray]:
    """Each validation grid once per allowed k; balanced words (all words covered); per-k placement RNG, so the
    common k = 0/2 records are IDENTICAL across experiments."""
    n = len(val_grids)
    rows = []
    for k in train_ks(experiment):
        words = V._balanced_words(seed, k, n)
        keys = make_rng(seed, STREAM_EXT_VALSLOT, k).random((n, 5))
        for g in range(n):
            rows.append((g, k, words[g], tuple(sorted(np.argsort(keys[g])[:k].tolist()))))
    gid = np.array([r[0] for r in rows])
    ks = np.array([r[1] for r in rows])
    words = [r[2] for r in rows]
    x = val_grids[gid].astype(np.uint8)
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8),
            "slots5": np.stack([V.slots5_for(r[2], r[3]) for r in rows]), "k": ks, "group": ks, "length": ks,
            "grid_id": gid, "string": np.array([to_label(w) for w in words]), "condition": "identity", "split": "val"}


def check_supports() -> dict:
    out = {e: support_counts(e) for e in IDEXT_EXPERIMENTS}
    assert out["idext_train02"] == {"logical_words": 10, "physical_patterns": 91}
    assert out["idext_train023"] == {"logical_words": 37, "physical_patterns": 361}
    for e in IDEXT_EXPERIMENTS:
        assert 1 not in train_ks(e)
    return out


def full_run_counts(experiment: str, B: int = 256, steps: int = 5000) -> Dict[int, int]:
    ks = train_ks(experiment)
    return {k: records_before(ks, B, steps, k) for k in ks}
