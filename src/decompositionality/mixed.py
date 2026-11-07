"""Mixed-length experiments: effective operation counts 0, 2, 3, 4, 5 (length 1 NEVER trained).

Supports (training == checkpoint-selection validation support):

=========  ==============================  ==============================
length     mixed_02345_distinct_pairs      mixed_02345_all_pairs
=========  ==============================  ==============================
0          EMPTY (identity target)         EMPTY
1          -- never --                     -- never --
2          AB AC BA BC CA CB               AA AB AC BA BB BC CA CB CC
3, 4, 5    all 27 / 81 / 243 words         all 27 / 81 / 243 words (identical)
total      358 words                       361 words
=========  ==============================  ==============================

ONLY the length-two support differs between the two experiments.

Sampling (stateless, keyed by the global sample index i = step * B + j):
* lengths: each consecutive block of 5 samples is a seeded permutation of (0, 2, 3, 4, 5), so every
  length receives exactly 20% of samples (counts differ by at most one in any prefix) and sample i is
  the b-th sample of its length, b = i // 5;
* words: the b-th sample of length L takes position b % k_L of a seeded permutation (block b // k_L)
  of that length's k_L allowed words -> per-word counts differ by at most one. The seed key does not
  depend on the experiment except at L = 2;
* five operation slots: a uniformly chosen order-preserving placement of the k operations among the
  5 slots (C(5, k) choices), all other slots PAD; length 0 -> five PADs, length 5 -> no PAD;
* grids: the same online grid stream as the legacy experiments, excluding validation/test grids.
Evaluation uses canonical right padding.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import operations as O
from .data import STREAM_GRID, STREAM_GRID_RESAMPLE, grid_bytes, make_rng, random_grids
from .words import EMPTY_LABEL, from_label, to_label, words_of_length

MIXED_EXPERIMENTS = {"mixed_02345_distinct_pairs": "distinct_pairs", "mixed_02345_all_pairs": "all_pairs"}
MIXED_RESULTS_SUBDIR = "mixed_length_02345"
LENGTHS = (0, 2, 3, 4, 5)
TEST_LENGTHS = (0, 1, 2, 3, 4, 5)
N_SLOTS = 5
MIXED_DATA_VERSION = "decomp-mixed02345-v1"

# Compact-layout positional-encoding ablation (notebooks 07/08). Same 361-word all-pairs support and the same
# length/word/grid schedule as mixed_02345_all_pairs, but operations are ALWAYS left-aligned (no random gaps).
COMPACT_EXPERIMENTS = {"compact_02345_rope": "rope", "compact_02345_sinusoidal": "sinusoidal",
                       "compact_02345_cnn": None}      # None = shared CNN controls (no attention PE switch)
COMPACT_DIRS = {"compact_02345_rope": "rope", "compact_02345_sinusoidal": "sinusoidal", "compact_02345_cnn": "shared_cnn"}
COMPACT_RESULTS_SUBDIR = "compact_02345_pe"
COMPACT_DATA_VERSION = "decomp-compact02345-v1"
LAYOUT_VERSION = "compact_left-v1"
ATTENTION_MODELS = ["transformer_ar", "looped_transformer_ar", "transformer_encoder", "looped_transformer_encoder",
                    "vit", "looped_vit", "trm"]
CNN_MODELS = ["rescnn_film", "unet_film"]

STREAM_LEN = 701
STREAM_WORD = 711
STREAM_PAD5 = 721
STREAM_VAL_WORDS = 731
_PAIRSET_KEY = {"distinct_pairs": 1, "all_pairs": 2}
COMBOS = {k: list(itertools.combinations(range(N_SLOTS), k)) for k in range(N_SLOTS + 1)}


def is_mixed(experiment: str) -> bool:
    """True for every 5-slot mixed-length experiment (random placement 04/05 or compact 07/08)."""
    return experiment in MIXED_EXPERIMENTS or experiment in COMPACT_EXPERIMENTS


def is_compact(experiment: str) -> bool:
    return experiment in COMPACT_EXPERIMENTS


def placement(experiment: str) -> str:
    return "compact_left" if is_compact(experiment) else "random_order_preserving"


def pairset_name(experiment: str) -> str:
    return "all_pairs" if is_compact(experiment) else MIXED_EXPERIMENTS[experiment]


def support(experiment: str) -> Dict[int, List[str]]:
    pairs = O.TRAIN_SETS[pairset_name(experiment)]
    return {0: [""], 2: list(pairs), 3: words_of_length(3), 4: words_of_length(4), 5: words_of_length(5)}


def training_words(experiment: str) -> List[str]:
    return [w for L in LENGTHS for w in support(experiment)[L]]


def training_labels(experiment: str) -> List[str]:
    return [to_label(w) for w in training_words(experiment)]


def test_words() -> List[str]:
    """All 364 words of length 0..5 (plain strings; "" = EMPTY), ordered by length then lexicographically."""
    return [w for L in TEST_LENGTHS for w in words_of_length(L)]


def test_labels() -> List[str]:
    return [to_label(w) for w in test_words()]


def pair_set(experiment: str) -> List[str]:
    return list(O.TRAIN_SETS[pairset_name(experiment)])


def mixed_group(label: str, experiment: str) -> str:
    w = from_label(label)
    L = len(w)
    if L == 0:
        return "EMPTY"
    if L == 1:
        return "singles"
    if L == 2:
        return "pairs_seen" if w in set(pair_set(experiment)) else "pairs_unseen"
    return f"len{L}_seen"


MIXED_GROUP_ORDER = ["EMPTY", "singles", "pairs_seen", "pairs_unseen", "len3_seen", "len4_seen", "len5_seen"]
MIXED_GROUP_LABELS = {"EMPTY": "EMPTY (identity)", "singles": "Singles A/B/C (never trained)",
                      "pairs_seen": "Pairs: seen", "pairs_unseen": "Pairs: unseen (AA/BB/CC)",
                      "len3_seen": "Length 3 (seen strings, new grids)", "len4_seen": "Length 4 (seen strings, new grids)",
                      "len5_seen": "Length 5 (seen strings, new grids)"}


def slots_for(word: str, positions: Sequence[int], n_slots: int = N_SLOTS) -> List[int]:
    assert len(positions) == len(word) and list(positions) == sorted(positions)
    s = [O.PAD_OP] * n_slots
    for p, ch in zip(positions, word):
        s[p] = O.ALIAS_TO_OP[ch]
    return s


def canonical_slots(word: str, n_slots: int = N_SLOTS) -> List[int]:
    if len(word) > n_slots:
        raise ValueError(f"word {word!r} longer than {n_slots} slots")
    return slots_for(word, list(range(len(word))), n_slots)


def apply_word_batch(x: np.ndarray, words: Sequence[str]) -> np.ndarray:
    """Oracle targets (training-code oracle); the empty word returns the input unchanged."""
    out = np.array(x, copy=True)
    words = np.asarray(list(words), dtype=object)
    for w in set(words.tolist()):
        if w == "":
            continue
        idx = np.nonzero(words == w)[0]
        out[idx] = O.apply_string(x[idx], w)
    return out


class MixedTrainStream:
    """Stateless: ``batch(step)`` is a pure function of (seed, step, experiment, B, held-out grids)."""

    def __init__(self, seed: int, experiment: str, effective_batch_size: int, heldout_bytes=None):
        assert is_mixed(experiment)
        self.seed, self.experiment, self.B = int(seed), experiment, int(effective_batch_size)
        self.compact = is_compact(experiment)
        self.sup = support(experiment)
        self.heldout = set(heldout_bytes or [])

    def lengths_and_words(self, start: int, n: int):
        idx = np.arange(start, start + n)
        blk, pos = idx // len(LENGTHS), idx % len(LENGTHS)
        lens = np.empty(n, dtype=np.int64)
        words = np.empty(n, dtype=object)
        perm_cache = {}
        for b in np.unique(blk):
            perm_cache[int(b)] = make_rng(self.seed, STREAM_LEN, int(b)).permutation(len(LENGTHS))
        for t in range(n):
            L = LENGTHS[perm_cache[int(blk[t])][pos[t]]]
            lens[t] = L
            ws = self.sup[L]
            j = int(blk[t])                       # this is the j-th sample of length L
            ck = (L, j // len(ws))
            if ck not in perm_cache:
                key = [STREAM_WORD, L] + ([_PAIRSET_KEY[pairset_name(self.experiment)]] if L == 2 else [])
                perm_cache[ck] = make_rng(self.seed, *key, j // len(ws)).permutation(len(ws))
            words[t] = ws[perm_cache[ck][j % len(ws)]]
        return lens, words

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
        lens, words = self.lengths_and_words(step * self.B, self.B)
        u = make_rng(self.seed, STREAM_PAD5, step).random(self.B)
        ops = np.empty((self.B, N_SLOTS), dtype=np.int64)
        placement = np.empty(self.B, dtype=np.int64)
        for t in range(self.B):
            if self.compact:      # compact left alignment: the j-th operation always sits in slot j
                placement[t] = 0
                ops[t] = canonical_slots(words[t])
                continue
            combos = COMBOS[int(lens[t])]
            k = min(int(u[t] * len(combos)), len(combos) - 1)
            placement[t] = k
            ops[t] = slots_for(words[t], combos[k])
        y = apply_word_batch(x, words)
        return {"x": x, "y": y.astype(np.uint8), "ops": ops, "length": lens,
                "string": np.array([to_label(w) for w in words]), "placement": placement,
                "sample_index": np.arange(step * self.B, (step + 1) * self.B)}


# ---------------------------------------------------------------------------
# Validation and test sets
# ---------------------------------------------------------------------------

def _balanced_sequence(seed: int, key: list, words: List[str], n: int) -> List[str]:
    out, rep = [], 0
    while len(out) < n:
        perm = make_rng(seed, STREAM_VAL_WORDS, *key, rep).permutation(len(words))
        out += [words[i] for i in perm]
        rep += 1
    return out[:n]


def make_val_set(val_grids: np.ndarray, experiment: str, seed: int) -> Dict[str, np.ndarray]:
    """Exactly one record per (validation grid, length in 0/2/3/4/5): 5 * n_grids records.
    Words follow a fixed balanced schedule covering every allowed word when n_grids >= k_L; the
    schedule (and therefore the words) is identical across experiments except at length 2."""
    n = len(val_grids)
    sup = support(experiment)
    assign = {}
    for L in LENGTHS:
        key = [L] + ([_PAIRSET_KEY[pairset_name(experiment)]] if L == 2 else [])
        assign[L] = _balanced_sequence(seed, key, sup[L], n)
    gid, words, lens = [], [], []
    for g in range(n):
        for L in LENGTHS:
            gid.append(g); words.append(assign[L][g]); lens.append(L)
    gid = np.array(gid)
    x = val_grids[gid].astype(np.uint8)
    return {"x": x, "y": apply_word_batch(x, words).astype(np.uint8),
            "ops": np.array([canonical_slots(w) for w in words], dtype=np.int64), "grid_id": gid,
            "string": np.array([to_label(w) for w in words]), "length": np.array(lens), "split": "val"}


def make_test_set(test_grids: np.ndarray, words: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Grid-major x word-minor cross product over all 364 words of length 0..5 (canonical right padding)."""
    words = list(words if words is not None else test_words())
    n, k = len(test_grids), len(words)
    gid = np.repeat(np.arange(n), k)
    widx = np.tile(np.arange(k), n)
    x = test_grids[gid].astype(np.uint8)
    y = np.empty_like(x)
    for j, w in enumerate(words):
        sel = widx == j
        y[sel] = x[sel] if w == "" else O.apply_string(x[sel], w)
    ops_table = np.array([canonical_slots(w) for w in words], dtype=np.int8)
    labels = np.array([to_label(w) for w in words])
    return {"x": x, "y": y, "ops": ops_table[widx].astype(np.int64), "grid_id": gid, "string": labels[widx],
            "length": np.array([len(w) for w in words])[widx], "split": "test"}


def subset_by_grids(es: Dict[str, np.ndarray], n_grids: int) -> Dict[str, np.ndarray]:
    keep = es["grid_id"] < n_grids
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in es.items()}


def check_supports() -> dict:
    a, b = (training_words(e) for e in MIXED_EXPERIMENTS)
    assert len(a) == 358 and len(b) == 361 and len(test_words()) == 364
    assert not any(len(w) == 1 for w in a + b)
    assert set(a) ^ set(b) == {"AA", "BB", "CC"}
    assert {w for w in a if len(w) != 2} == {w for w in b if len(w) != 2}
    return {"mixed_02345_distinct_pairs": len(a), "mixed_02345_all_pairs": len(b), "test_words": len(test_words())}


# ---------------------------------------------------------------------------
# Metrics (integer counts) for mixed evaluation records
# ---------------------------------------------------------------------------

def metrics_tables(strings: np.ndarray, pred: np.ndarray, valid: np.ndarray, target: np.ndarray, experiment: str):
    """Per-word, per-group and summary tables with integer successes/denominators.
    Overall metrics are reported both micro (record-weighted) and length-macro (each length 0..5 equal)."""
    import pandas as pd
    P = np.asarray(pred).reshape(len(pred), 36).astype(np.int16)
    Y = np.asarray(target).reshape(len(target), 36).astype(np.int16)
    cc = (P == Y).sum(1)
    ex = np.asarray(valid, bool) & (cc == 36)
    df = pd.DataFrame({"string": np.asarray(strings).astype(str), "exact": ex, "cell_correct": cc,
                       "invalid": ~np.asarray(valid, bool)})
    pw = df.groupby("string", sort=False).agg(exact_correct=("exact", "sum"), n=("exact", "size"),
                                              cell_correct=("cell_correct", "sum"), invalid=("invalid", "sum")).reset_index()
    order = {s: i for i, s in enumerate(test_labels())}
    pw = pw.sort_values("string", key=lambda c: c.map(order)).reset_index(drop=True)
    pw["length"] = pw["string"].map(lambda s: len(from_label(s)))
    pw["group"] = pw["string"].map(lambda s: mixed_group(s, experiment))
    tl = set(training_labels(experiment))
    pw["seen_in_training"] = pw["string"].isin(tl)
    pw["cells_total"] = pw["n"] * 36
    rows = []
    for g in MIXED_GROUP_ORDER:
        sub = pw[pw.group == g]
        rows.append({"group": g, "label": MIXED_GROUP_LABELS[g], "n_words": len(sub),
                     "exact_correct": int(sub.exact_correct.sum()), "n": int(sub.n.sum()),
                     "cell_correct": int(sub.cell_correct.sum()), "cells_total": int(sub.cells_total.sum()),
                     "invalid": int(sub.invalid.sum())})
    for s in ("A", "B", "C"):
        sub = pw[pw.string == s]
        rows.append({"group": f"single_{s}", "label": f"Single {s}", "n_words": len(sub),
                     "exact_correct": int(sub.exact_correct.sum()), "n": int(sub.n.sum()),
                     "cell_correct": int(sub.cell_correct.sum()), "cells_total": int(sub.cells_total.sum()),
                     "invalid": int(sub.invalid.sum())})
    gr = pd.DataFrame(rows)
    for t in (pw, gr):
        n = t["n"].where(t["n"] > 0)
        t["exact_acc"] = t["exact_correct"] / n
        t["cell_acc"] = t["cell_correct"] / t["cells_total"].where(t["cells_total"] > 0)
        t["invalid_rate"] = t["invalid"] / n
    singles = pw[pw.length == 1]
    by_len = pw.groupby("length").apply(lambda d: d.exact_correct.sum() / d.n.sum())
    summ = {"singles_macro_exact": float(singles.exact_acc.mean()) if len(singles) else np.nan,
            "singles_macro_cell": float(singles.cell_acc.mean()) if len(singles) else np.nan,
            "overall_micro_exact (record-weighted)": float(pw.exact_correct.sum() / pw.n.sum()),
            "overall_length_macro_exact (lengths 0..5 equal weight)": float(by_len.mean()),
            **{f"length_{int(L)}_exact": float(v) for L, v in by_len.items()}}
    return pw, gr, summ
