"""Slot-count intervention in the identity format (one notebook, eight runs): Transformer AR and Looped AR.

  condition           S   train k          held-out primary tests
  slots2_train02      2   0, 2             1
  slots3_train023     3   0, 2, 3          1
  slots4_train0234    4   0, 2, 3, 4       1
  slots6_train02345   6   0, 2, 3, 4, 5    1 and 6

Every instruction has exactly S VISIBLE operation tokens: k A/B/C tokens in a uniform ordered k-subset of the S
slots, the remaining S-k slots filled with the learnable <identity> (never PAD). The AR sequence has 77+S tokens.

Logical records reuse the per-k indexed stream of the five-slot count-extrapolation experiments
(``idext.record(seed, k, j)``: word and grid), and the same per-update allocation and batch order
(``idext.per_update_counts`` / ``STREAM_EXT_ORDER``). For supports 02/023/0234 the (word, grid, target) stream and
training order are therefore IDENTICAL to the historical S=5 runs (notebooks 12-14); only the slot subset is drawn
within S (separate RNG stream). The historical 02345 run (notebook 10) used a different sampler, so S=6 is a
new stream relative to it.
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence

import numpy as np

from . import idext as E
from . import idvar as V
from .data import make_rng
from .words import to_label, words_of_length

IDSLOT_EXPERIMENTS = {"slots2_train02": (2, (0, 2)), "slots3_train023": (3, (0, 2, 3)),
                      "slots4_train0234": (4, (0, 2, 3, 4)), "slots6_train02345": (6, (0, 2, 3, 4, 5))}
IDSLOT_RESULTS_SUBDIR = "identity_ar_slot_count"
IDSLOT_MODELS = ["transformer_ar", "looped_transformer_ar"]
IDSLOT_DATA_VERSION = "decomp-idslot-v1"
POSITION_CAPACITY_SLOTS = 6     # positional table sized for the largest S (83 AR tokens); actual inputs are S tokens
HISTORICAL_S5 = {"slots2_train02": ("idext", "idext_train02"), "slots3_train023": ("idext", "idext_train023"),
                 "slots4_train0234": ("idext", "idext_train0234"), "slots6_train02345": ("idvar", "idvar_identity")}

STREAM_SLOT_TRAIN = 1001
STREAM_SLOT_VAL = 1011
STREAM_SLOT_TEST = 1021


def is_idslot(experiment: str) -> bool:
    return experiment in IDSLOT_EXPERIMENTS


def n_slots(experiment: str) -> int:
    return IDSLOT_EXPERIMENTS[experiment][0]


def train_ks(experiment: str) -> tuple:
    return IDSLOT_EXPERIMENTS[experiment][1]


def training_words(experiment: str) -> List[str]:
    return [w for k in train_ks(experiment) for w in words_of_length(k)]


def training_labels(experiment: str) -> List[str]:
    return [to_label(w) for w in training_words(experiment)]


def slots_for(word: str, positions: Sequence[int], S: int) -> np.ndarray:
    s = np.full(S, V.ID_OP, dtype=np.int64)
    for p, ch in zip(sorted(positions), word):
        s[p] = "ABC".index(ch)
    return s


def subset(rng: np.random.Generator, S: int, k: int) -> tuple:
    return tuple(sorted(np.argsort(rng.random(S))[:k].tolist()))


def support_counts(experiment: str) -> dict:
    S, ks = IDSLOT_EXPERIMENTS[experiment]
    return {"S": S, "logical_words": sum(3 ** k for k in ks),
            "physical_patterns": sum(comb(S, k) * 3 ** k for k in ks)}


def full_run_counts(experiment: str, B: int = 256, steps: int = 5000) -> Dict[int, int]:
    ks = train_ks(experiment)
    return {k: E.records_before(ks, B, steps, k) for k in ks}


def check_supports() -> dict:
    out = {e: support_counts(e) for e in IDSLOT_EXPERIMENTS}
    expect = {"slots2_train02": (10, 10), "slots3_train023": (37, 55), "slots4_train0234": (118, 244),
              "slots6_train02345": (361, 3349)}
    for e, (w, p) in expect.items():
        assert (out[e]["logical_words"], out[e]["physical_patterns"]) == (w, p), (e, out[e])
    assert 3349 + comb(6, 1) * 3 + 3 ** 6 == 4 ** 6
    no_i = {e: full_run_counts(e).get(n_slots(e), 0) for e in IDSLOT_EXPERIMENTS}
    assert no_i == {"slots2_train02": 640000, "slots3_train023": 426666, "slots4_train0234": 320000,
                    "slots6_train02345": 0}, no_i
    for e in IDSLOT_EXPERIMENTS:
        assert 1 not in train_ks(e) and sum(full_run_counts(e).values()) == 1_280_000
    return {"supports": out, "no_identity_training_records": no_i}


class SlotStream:
    """Stateless; batch(step) depends only on (seed, step, B, S, train ks, held-out grids). Word/grid/order come from
    the shared per-k indexed stream; the slot subset is drawn uniformly within S from a separate stream."""

    def __init__(self, seed: int, experiment: str, effective_batch_size: int, heldout_bytes=None):
        self.seed, self.experiment, self.B = int(seed), experiment, int(effective_batch_size)
        self.S, self.ks = IDSLOT_EXPERIMENTS[experiment]
        self.heldout = set(heldout_bytes or [])

    def batch(self, step: int) -> Dict[str, np.ndarray]:
        step = int(step)
        cnt = E.per_update_counts(self.ks, self.B, step)
        recs = []
        for k in self.ks:
            j0 = E.records_before(self.ks, self.B, step, k)
            for j in range(j0, j0 + cnt[k]):
                word, _, g = E.record(self.seed, k, j, self.heldout)
                pos = subset(make_rng(self.seed, STREAM_SLOT_TRAIN, self.S, k, j), self.S, k)
                recs.append((k, j, word, pos, g))
        recs = [recs[i] for i in make_rng(self.seed, E.STREAM_EXT_ORDER, step).permutation(len(recs))]
        ks = np.array([r[0] for r in recs])
        words = [r[2] for r in recs]
        x = np.stack([r[4] for r in recs]).astype(np.uint8)
        return {"x": x, "y": V.apply_words(x, words).astype(np.uint8),
                "slots": np.stack([slots_for(r[2], r[3], self.S) for r in recs]), "k": ks, "group": ks, "length": ks,
                "string": np.array([to_label(w) for w in words]), "k_index": np.array([r[1] for r in recs]),
                "sample_id": np.array([f"k{r[0]}:{r[1]}" for r in recs]), "condition": "identity"}


def make_val_set(val_grids: np.ndarray, seed: int, experiment: str) -> Dict[str, np.ndarray]:
    """Each validation grid once per training k; balanced words (all covered; same word schedule as the S=5
    experiments); deterministic placements within S."""
    S = n_slots(experiment)
    n = len(val_grids)
    rows = []
    for k in train_ks(experiment):
        words = V._balanced_words(seed, k, n)
        rng = make_rng(seed, STREAM_SLOT_VAL, S, k)
        for g in range(n):
            rows.append((g, k, words[g], subset(rng, S, k)))
    gid = np.array([r[0] for r in rows])
    ks = np.array([r[1] for r in rows])
    words = [r[2] for r in rows]
    x = val_grids[gid].astype(np.uint8)
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8),
            "slots": np.stack([slots_for(r[2], r[3], S) for r in rows]), "k": ks, "group": ks, "length": ks,
            "grid_id": gid, "string": np.array([to_label(w) for w in words]), "condition": "identity", "split": "val"}


def make_test_set(test_grids: np.ndarray, seed: int, experiment: str) -> Dict[str, np.ndarray]:
    """Per grid: every training word (one deterministic placement within S), every singleton at every one of the S
    positions, and (S=6 only) all 729 six-operation words (one layout each: no identity)."""
    S, ks = IDSLOT_EXPERIMENTS[experiment]
    seen = training_words(experiment)
    rows = []
    for g in range(len(test_grids)):
        rng = make_rng(seed, STREAM_SLOT_TEST, S, g)
        for w in seen:
            rows.append((g, w, slots_for(w, subset(rng, S, len(w)), S), "seen", 0))
        for op in "ABC":
            for p in range(S):
                rows.append((g, op, slots_for(op, (p,), S), "single", p + 1))
        if S == 6:
            for w in words_of_length(6):
                rows.append((g, w, slots_for(w, tuple(range(6)), 6), "six", 0))
    gid = np.array([r[0] for r in rows])
    words = [r[1] for r in rows]
    x = test_grids[gid].reshape(len(rows), 6, 6).astype(np.uint8)
    k = np.array([len(w) for w in words])
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), "slots": np.stack([r[2] for r in rows]),
            "k": k, "group": k, "length": k, "grid_id": gid, "string": np.array([to_label(w) for w in words]),
            "kind": np.array([r[3] for r in rows]), "position": np.array([r[4] for r in rows]),
            "condition": "identity", "split": "test"}


def expected_test_records(experiment: str, n_grids: int) -> int:
    S, ks = IDSLOT_EXPERIMENTS[experiment]
    return n_grids * (sum(3 ** k for k in ks) + 3 * S + (729 if S == 6 else 0))


def grown_initial_state(model_id: str, seed: int):
    """Common fresh initialization for all four conditions: the S=5 identity model's seed-``seed`` initialization
    (identical to the historical five-slot runs' initial tensors), with the learned position table grown from 82 to
    83 rows (capacity for S=6). Added rows use a SEPARATE deterministic generator; no other draw order changes."""
    import dataclasses
    import torch
    from .models.registry import build_model, default_config
    base = dataclasses.asdict(default_config(model_id))
    base.update(n_slots=5, identity_token=True)
    src = build_model(model_id, base, seed=seed).state_dict()
    big = dict(base, n_slots=POSITION_CAPACITY_SLOTS, op_format="variable")
    dst = build_model(model_id, big, seed=seed).state_dict()
    gen = torch.Generator().manual_seed(int(seed) * 1000 + 83)
    out = {}
    for k, v in dst.items():
        s = src[k]
        if s.shape == v.shape:
            out[k] = s.clone()
        else:
            # only per-position tables grow (AR: pos_emb; encoders: slot_emb); everything else is copied
            assert k in ("pos_emb.weight", "slot_emb.weight") and v.shape[0] > s.shape[0] and v.shape[1:] == s.shape[1:], (k, s.shape, v.shape)
            extra = torch.randn(v.shape[0] - s.shape[0], *v.shape[1:], generator=gen) * 0.02  # gpt_init std
            out[k] = torch.cat([s.clone(), extra], dim=0)
    return out


def record_metadata(es: Dict[str, np.ndarray], experiment: str) -> Dict[str, np.ndarray]:
    """Per-record S, number of visible identity tokens (S-k), and membership of the record's induced
    36-cell permutation in the training-support transformation set."""
    from .diagnosis import group as G
    from .words import from_label
    cat = G.Catalog()
    S = n_slots(experiment)
    train_ids = {cat.word_id(w) for w in training_words(experiment)}
    labels = es["string"].astype(str)
    pid = {s: cat.word_id(from_label(s)) for s in np.unique(labels)}
    perm = np.array([pid[s] for s in labels], dtype=np.int16)
    n_id = (np.asarray(es["slots"]) == V.ID_OP).sum(1)
    assert (n_id == S - np.asarray(es["k"])).all()
    return {"S": np.full(len(labels), S, dtype=np.int8), "n_identity": n_id.astype(np.int8), "perm_id": perm,
            "transform_in_training": np.isin(perm, sorted(train_ids))}


def transformation_coverage():
    """Training-support transformation sets (computed, then checked against the documented values) and, for S=6,
    the membership of all 729 six-operation words."""
    import pandas as pd
    from .diagnosis import group as G
    cat = G.Catalog()
    rows = []
    sets = {}
    for e in IDSLOT_EXPERIMENTS:
        S = n_slots(e)
        T = {cat.word_id(w) for w in training_words(e)}
        sets[e] = T
        for k in range(S + 1):
            ws = words_of_length(k)
            pres = sum(cat.word_id(w) in T for w in ws)
            rows.append({"condition": e, "S": S, "k": k, "status": "train" if k in train_ks(e) else "HELD OUT",
                         "words": len(ws), "words_with_training_transform": pres,
                         "words_with_NEW_transform": len(ws) - pres, "n_training_transforms": len(T)})
    df = pd.DataFrame(rows)
    assert {e: len(t) for e, t in sets.items()} == {"slots2_train02": 9, "slots3_train023": 30,
                                                    "slots4_train0234": 62, "slots6_train02345": 111}
    six = df[(df.condition == "slots6_train02345") & (df.k == 6)].iloc[0]
    assert (six.words_with_training_transform, six.words_with_NEW_transform) == (591, 138), six
    single = df[df.k == 1].set_index("condition").words_with_training_transform.to_dict()
    assert single == {"slots2_train02": 0, "slots3_train023": 3, "slots4_train0234": 3, "slots6_train02345": 3}
    return df
