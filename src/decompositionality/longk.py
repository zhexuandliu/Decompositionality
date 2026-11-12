"""Notebook 21: Transformer AR vs causal encoder, identity-filler format, evaluated at the LAST checkpoint, with the
focus on held-out singletons and longer (k > max trained k) instructions.

  condition     S   train k          held-out tests (all within S, so no new physical positions)
  S5_k0234      5   0,2,3,4          k=1 (every slot), k=5
  S6_k02345     6   0,2,3,4,5        k=1, k=6
  S7_k02345     7   0,2,3,4,5        k=1, k=6, k=7
  S7_k023456    7   0,2,3,4,5,6      k=1, k=7

All four conditions share positional capacity 7 (AR sequence <= 84 tokens) and therefore ONE fresh initialization per
architecture (same config, same seed). The logical stream (k, word, grid, target) is idext.record(seed, k, j) with the
rotating per-update allocation, so conditions with the same training k (S6_k02345 / S7_k02345) see the identical
logical stream; placements are a uniform ordered k-subset of the S slots (the notebook-16 placement stream).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import idext as E
from . import idslot as IS
from . import idvar as V
from .data import make_rng
from .words import to_label, words_of_length

SUBDIR = "longk_ar_vs_causal_encoder_v1"
DATA_VERSION = "decomp-longk-v1"
CAPACITY = 7
MODELS = ["transformer_ar", "causal_encoder"]
CONDITIONS: Dict[str, tuple] = {
    "S5_k0234": (5, (0, 2, 3, 4)),
    "S6_k02345": (6, (0, 2, 3, 4, 5)),
    "S7_k02345": (7, (0, 2, 3, 4, 5)),
    "S7_k023456": (7, (0, 2, 3, 4, 5, 6)),
}
STREAM_LONG_TEST = 1131


def is_longk(e: str) -> bool:
    return e in CONDITIONS


def n_slots(e):
    return CONDITIONS[e][0]


def train_ks(e):
    return CONDITIONS[e][1]


def heldout_long_ks(e) -> List[int]:
    S, ks = CONDITIONS[e]
    return list(range(max(ks) + 1, S + 1))


def training_labels(e) -> List[str]:
    return [to_label(w) for k in train_ks(e) for w in words_of_length(k)]


class LongKStream:
    """Stateless: batch(step) depends only on (seed, step, B, condition, held-out grids)."""

    def __init__(self, seed: int, e: str, B: int, heldout_bytes=None):
        self.seed, self.e, self.B = int(seed), e, int(B)
        self.S, self.ks = CONDITIONS[e]
        self.heldout = set(heldout_bytes or [])

    def batch(self, step: int) -> Dict[str, np.ndarray]:
        step = int(step)
        cnt = E.per_update_counts(self.ks, self.B, step)
        recs = []
        for k in self.ks:
            j0 = E.records_before(self.ks, self.B, step, k)
            for j in range(j0, j0 + cnt[k]):
                word, _, g = E.record(self.seed, k, j, self.heldout)
                pos = IS.subset(make_rng(self.seed, IS.STREAM_SLOT_TRAIN, self.S, k, j), self.S, k)
                recs.append((k, j, word, IS.slots_for(word, pos, self.S), g))
        recs = [recs[i] for i in make_rng(self.seed, E.STREAM_EXT_ORDER, step).permutation(len(recs))]
        ks = np.array([r[0] for r in recs])
        words = [r[2] for r in recs]
        x = np.stack([r[4] for r in recs]).astype(np.uint8)
        return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), "slots": np.stack([r[3] for r in recs]),
                "k": ks, "group": ks, "length": ks, "string": np.array([to_label(w) for w in words]),
                "k_index": np.array([r[1] for r in recs]), "condition": "identity"}


def make_val_set(val_grids, seed, e):
    S, ks = CONDITIONS[e]
    n = len(val_grids)
    rows = []
    for k in ks:
        words = V._balanced_words(seed, k, n)
        rng = make_rng(seed, IS.STREAM_SLOT_VAL, S, k)
        for g in range(n):
            rows.append((g, k, words[g], IS.slots_for(words[g], IS.subset(rng, S, k), S)))
    gid = np.array([r[0] for r in rows])
    ks_ = np.array([r[1] for r in rows])
    words = [r[2] for r in rows]
    x = val_grids[gid].astype(np.uint8)
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), "slots": np.stack([r[3] for r in rows]),
            "k": ks_, "group": ks_, "length": ks_, "grid_id": gid, "string": np.array([to_label(w) for w in words]),
            "condition": "identity", "split": "val"}


def long_grids(k: int, n_grids: int, k7_grids: int) -> int:
    return min(n_grids, k7_grids) if k >= 7 else n_grids


def make_test_chunk(test_grids, seed, e, g0, g1, n_grids, k7_grids):
    """Test records for grids g0..g1-1: every trained word (one deterministic layout per word/grid), every singleton
    at EVERY slot, and every word of each held-out longer k (one layout each; k=S has a single layout). Grids beyond
    the k=7 budget get no k=7 records."""
    S, ks = CONDITIONS[e]
    seen = [w for k in ks for w in words_of_length(k)]
    rows = []
    for g in range(g0, g1):
        rng = make_rng(seed, IS.STREAM_SLOT_TEST, S, g)
        for w in seen:
            rows.append((g, w, IS.slots_for(w, IS.subset(rng, S, len(w)), S), "seen", 0))
        for op in "ABC":
            for p in range(S):
                rows.append((g, op, IS.slots_for(op, (p,), S), "single", p + 1))
        for k in heldout_long_ks(e):
            if g >= long_grids(k, n_grids, k7_grids):
                continue
            lr = make_rng(seed, STREAM_LONG_TEST, S, k, g)
            for w in words_of_length(k):
                rows.append((g, w, IS.slots_for(w, IS.subset(lr, S, k), S), f"long{k}", 0))
    gid = np.array([r[0] for r in rows])
    words = [r[1] for r in rows]
    x = test_grids[gid].reshape(len(rows), 6, 6).astype(np.uint8)
    k_ = np.array([len(w) for w in words])
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), "slots": np.stack([r[2] for r in rows]),
            "k": k_, "group": k_, "length": k_, "grid_id": gid, "string": np.array([to_label(w) for w in words]),
            "kind": np.array([r[3] for r in rows]), "position": np.array([r[4] for r in rows]), "condition": "identity"}


def expected_records(e, n_grids, k7_grids) -> int:
    S, ks = CONDITIONS[e]
    n = n_grids * (sum(3 ** k for k in ks) + 3 * S)
    for k in heldout_long_ks(e):
        n += long_grids(k, n_grids, k7_grids) * 3 ** k
    return n


def model_config(model_id: str, overrides=None) -> dict:
    import dataclasses
    from .models.registry import default_config
    c = dataclasses.asdict(default_config(model_id))
    c.update(overrides or {})
    c.update(n_slots=CAPACITY, op_format="variable", identity_token=True)
    return c
