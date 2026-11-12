"""Identity-filler vs variable-length instruction experiments (notebooks 10/11), all nine models.

Both conditions consume the SAME indexed logical records. A record is (word over {A,B,C} with k in {0,2,3,4,5},
ordered slot subset of size k among 5 slots, fresh grid, target = word applied left to right). The record is stored
in a five-slot form ``slots5`` (A/B/C in the chosen slots, <identity> elsewhere) and SERIALIZED per condition:

* ``identity``        -> all 5 tokens, fillers are the visible, learnable <identity> op (no PAD anywhere);
* ``variable_length`` -> only the k A/B/C tokens in order (no PAD, no <identity>, no reserved slots).

Sampling per logical update of B=256: 51 records for each k plus one extra record for k = K[step % 5], so every
k gets exactly 256 records per 5 updates (256,000 over 5000 updates). Within k the word is uniform over {A,B,C}^k
and the slot subset uniform over the C(5,k) ordered subsets. No k = 1 instruction is ever generated for training or
checkpoint-selection validation (the singleton exclusion concerns the NUMBER of A/B/C tokens: e.g. ACC is allowed).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import operations as O
from .data import STREAM_GRID, STREAM_GRID_RESAMPLE, grid_bytes, make_rng, random_grids
from .words import to_label, words_of_length

IDVAR_EXPERIMENTS = {"idvar_identity": "identity", "idvar_variable": "variable_length"}
IDVAR_DIRS = {"idvar_identity": "identity", "idvar_variable": "variable_length"}
IDVAR_RESULTS_SUBDIR = "identity_varlen_02345_all_models"
IDVAR_DATA_VERSION = "decomp-idvar02345-v1"
N_SLOTS = 5
K_VALUES = (0, 2, 3, 4, 5)
ID_OP = 4  # slot value of the visible <identity> operation (token ID tokenization.IDENTITY_ID)

STREAM_IDV_ORDER = 801
STREAM_IDV_WORD = 802
STREAM_IDV_SLOT = 803
STREAM_IDV_VALSLOT = 811
STREAM_IDV_TESTSLOT = 821
STREAM_IDV_VALWORD = 831


def is_idvar(experiment: str) -> bool:
    return experiment in IDVAR_EXPERIMENTS


def condition_of(experiment: str) -> str:
    from .idext import is_idext
    from .idslot import is_idslot
    from .longk import is_longk
    from .auxtask import is_auxtask
    if is_longk(experiment) or is_auxtask(experiment):
        return "identity"
    from .rowsep import condition_format, is_rowsep
    if is_rowsep(experiment):
        return condition_format(experiment)
    if is_idext(experiment) or is_idslot(experiment):   # identity format (five or S visible slots)
        return "identity"
    return IDVAR_EXPERIMENTS[experiment]


def training_words() -> List[str]:
    return [w for k in K_VALUES for w in words_of_length(k)]


def training_labels() -> List[str]:
    return [to_label(w) for w in training_words()]


def pattern_counts() -> Dict[int, int]:
    """Five-slot patterns over {I,A,B,C} with k non-identity tokens: C(5,k) * 3^k."""
    from math import comb
    return {k: comb(5, k) * 3 ** k for k in range(6)}


def check_supports() -> dict:
    words = training_words()
    pc = pattern_counts()
    assert len(words) == 361 and not any(len(w) == 1 for w in words)
    assert sum(pc[k] for k in K_VALUES) == 1009 and pc[1] == 15 and sum(pc.values()) == 4 ** 5
    return {"logical_training_words": 361, "training_patterns": 1009, "singleton_patterns": 15, "all_patterns": 1024}


def slots5_for(word: str, positions: Sequence[int]) -> np.ndarray:
    s = np.full(N_SLOTS, ID_OP, dtype=np.int64)
    for p, ch in zip(sorted(positions), word):
        s[p] = O.ALIAS_TO_OP[ch]
    return s


def serialize_ops(slots5: np.ndarray, condition: str) -> np.ndarray:
    """Model input operation tokens for a batch of records with EQUAL k (callers group by k).
    identity: [n, 5] incl. <identity> fillers. variable_length: [n, k] with only the A/B/C tokens, in order."""
    slots5 = np.asarray(slots5)
    if condition == "identity":
        return slots5
    assert condition == "variable_length", condition
    keep = slots5 != ID_OP
    ks = keep.sum(1)
    if len(ks) and not (ks == ks[0]).all():
        raise ValueError("variable-length serialization requires equal-k groups (no padding is ever added)")
    k = int(ks[0]) if len(ks) else 0
    return slots5[keep].reshape(len(slots5), k)


def word_of_slots(slots5: np.ndarray) -> str:
    return "".join(O.OP_TO_ALIAS[int(v)] for v in slots5 if v != ID_OP)


def apply_words(x: np.ndarray, words: Sequence[str]) -> np.ndarray:
    out = np.array(x, copy=True)
    words = np.asarray(list(words), dtype=object)
    for w in set(words.tolist()):
        if w:
            idx = np.nonzero(words == w)[0]
            out[idx] = O.apply_string(x[idx], w)
    return out


def _random_subsets(rng: np.random.Generator, ks: np.ndarray) -> List[tuple]:
    """Uniform ordered k-subsets of the 5 slots (sorted positions)."""
    keys = rng.random((len(ks), N_SLOTS))
    return [tuple(sorted(np.argsort(keys[i])[: int(ks[i])].tolist())) for i in range(len(ks))]


class IdVarStream:
    """Stateless: ``batch(step)`` depends only on (seed, step, B, held-out grids); identical for both conditions
    except for the ``condition`` tag that selects serialization."""

    def __init__(self, seed: int, effective_batch_size: int, condition: str, heldout_bytes=None):
        assert condition in ("identity", "variable_length")
        self.seed, self.B, self.condition = int(seed), int(effective_batch_size), condition
        self.heldout = set(heldout_bytes or [])

    def ks_for_step(self, step: int) -> np.ndarray:
        base = self.B // len(K_VALUES)
        extra = self.B - base * len(K_VALUES)
        ks = []
        for i, k in enumerate(K_VALUES):
            ks += [k] * base
        for e in range(extra):   # rotating extra record(s)
            ks.append(K_VALUES[(step * extra + e) % len(K_VALUES)])
        ks = np.array(ks, dtype=np.int64)
        return ks[make_rng(self.seed, STREAM_IDV_ORDER, step).permutation(self.B)]

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
        ks = self.ks_for_step(step)
        letters = make_rng(self.seed, STREAM_IDV_WORD, step).integers(0, 3, (self.B, N_SLOTS))
        words = ["".join("ABC"[c] for c in letters[i, : ks[i]]) for i in range(self.B)]
        subsets = _random_subsets(make_rng(self.seed, STREAM_IDV_SLOT, step), ks)
        x = self._grids(step)
        y = apply_words(x, words).astype(np.uint8)
        slots5 = np.stack([slots5_for(w, s) for w, s in zip(words, subsets)])
        return {"x": x, "y": y, "slots5": slots5, "k": ks, "group": ks, "length": ks,
                "string": np.array([to_label(w) for w in words]),
                "sample_index": np.arange(step * self.B, (step + 1) * self.B), "condition": self.condition}


def _balanced_words(seed: int, k: int, n: int) -> List[str]:
    ws = words_of_length(k)
    out, rep = [], 0
    while len(out) < n:
        perm = make_rng(seed, STREAM_IDV_VALWORD, k, rep).permutation(len(ws))
        out += [ws[i] for i in perm]
        rep += 1
    return out[:n]


def make_val_set(val_grids: np.ndarray, seed: int, condition: str) -> Dict[str, np.ndarray]:
    """One record per (grid, k in 0/2/3/4/5); balanced words covering all 361; deterministic shared slot subsets."""
    n = len(val_grids)
    words = {k: _balanced_words(seed, k, n) for k in K_VALUES}
    gid, ws, ks = [], [], []
    for g in range(n):
        for k in K_VALUES:
            gid.append(g); ws.append(words[k][g]); ks.append(k)
    ks = np.array(ks)
    subsets = _random_subsets(make_rng(seed, STREAM_IDV_VALSLOT), ks)
    gid = np.array(gid)
    x = val_grids[gid].astype(np.uint8)
    return {"x": x, "y": apply_words(x, ws).astype(np.uint8),
            "slots5": np.stack([slots5_for(w, s) for w, s in zip(ws, subsets)]), "k": ks, "group": ks, "length": ks,
            "grid_id": gid, "string": np.array([to_label(w) for w in ws]), "condition": condition, "split": "val"}


def make_test_set(test_grids: np.ndarray, seed: int, condition: str) -> Dict[str, np.ndarray]:
    """Per grid: all 361 trained words (one deterministic slot subset per (word, grid), shared across models and
    conditions) + singletons (identity: A/B/C x 5 placements; variable_length: A/B/C once)."""
    seen = training_words()
    rows = []   # (grid, word, slots5, kind, position)
    for g in range(len(test_grids)):
        ks = np.array([len(w) for w in seen])
        subsets = _random_subsets(make_rng(seed, STREAM_IDV_TESTSLOT, g), ks)
        for w, s in zip(seen, subsets):
            rows.append((g, w, slots5_for(w, s), "seen", 0))
        for op in "ABC":
            if condition == "identity":
                for p in range(N_SLOTS):
                    rows.append((g, op, slots5_for(op, (p,)), "single", p + 1))
            else:
                rows.append((g, op, slots5_for(op, (0,)), "single", 0))
    gid = np.array([r[0] for r in rows])
    words = [r[1] for r in rows]
    x = test_grids[gid].reshape(len(rows), 6, 6).astype(np.uint8)
    ks = np.array([len(w) for w in words])
    return {"x": x, "y": apply_words(x, words).astype(np.uint8), "slots5": np.stack([r[2] for r in rows]),
            "k": ks, "group": ks, "length": ks, "grid_id": gid, "string": np.array([to_label(w) for w in words]),
            "kind": np.array([r[3] for r in rows]), "position": np.array([r[4] for r in rows]),
            "condition": condition, "split": "test"}


def subset_by_grids(es: Dict[str, np.ndarray], n_grids: int) -> Dict[str, np.ndarray]:
    keep = es["grid_id"] < n_grids
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in es.items()}


def expected_test_records(condition: str, n_grids: int) -> int:
    return n_grids * (361 + (15 if condition == "identity" else 3))
