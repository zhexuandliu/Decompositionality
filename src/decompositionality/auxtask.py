"""Notebook 26: does an UNRELATED auxiliary task that covers long instructions / late positions transfer positional
behaviour to the main grid-operation task?

Main task (always): S=7 identity-filler format, A/B/C (spatial operations), trained k in {0,2,3,4,5}; tests:
singletons at every slot, held-out k=6 and k=7.
Auxiliary task: value operations that never move a cell and so never coincide with any A/B/C composition:
  D: d -> (d+1) mod 10     (order 10)      E: d -> 9-d     (order 2; E D E = D^-1)
Same 7 slots, same <identity> fillers, uniform ordered placement. Every instruction starts with a task marker
(<main> / <aux>) in an extra leading operation position, so all conditions use 8 operation tokens and share ONE model
configuration and initialization per architecture.

  condition       aux task k            share per update (main / aux)
  aux_none        -                     256 / 0     (baseline = the notebook-23b S7_k02345 setting + marker)
  aux_k02345      0,2,3,4,5             192 / 64    (extra task, no extra length)
  aux_k0234567    0,2,3,4,5,6,7         192 / 64    (extra task WITH long instructions and zero-identity inputs)
  aux_k01234567   0,1,2,...,7           192 / 64    (as above, plus single D / E instructions)

Main records are idext.record(seed, k, j) at the same per-k indices as notebooks 21-23 (fewer per update when the
aux task is present). Checkpoint selection uses the MAIN task only.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List

import numpy as np

from . import idext as E
from . import idslot as IS
from . import idvar as V
from . import tokenization as T
from .data import STREAM_GRID_RESAMPLE, grid_bytes, make_rng, random_grids
from .words import to_label, words_of_length

SUBDIR = "aux_task_transfer_v1"
DATA_VERSION = "decomp-auxtask-v1"
S = 7
MAIN_KS = (0, 2, 3, 4, 5)
CAPACITY = S + 1                      # task marker + 7 slots
MODELS = ["transformer_ar", "causal_encoder"]
CONDITIONS: Dict[str, tuple] = {      # name -> (aux ks, aux per update)
    "aux_none": ((), 0),
    "aux_k02345": ((0, 2, 3, 4, 5), 64),
    "aux_k0234567": ((0, 2, 3, 4, 5, 6, 7), 64),
    "aux_k01234567": ((0, 1, 2, 3, 4, 5, 6, 7), 64),
}
STREAM_AUX_RECORD, STREAM_AUX_GRID, STREAM_AUX_SLOT, STREAM_AUX_TEST = 1201, 1202, 1203, 1211
AUX_LETTERS = "DE"
AUX_SLOT = {"D": T.D_OP, "E": T.E_OP}


def is_auxtask(e: str) -> bool:
    return e in CONDITIONS


def aux_ks(e):
    return CONDITIONS[e][0]


def training_labels(e) -> List[str]:
    return [to_label(w) for k in MAIN_KS for w in words_of_length(k)]


def apply_aux(x: np.ndarray, word: str) -> np.ndarray:
    y = np.asarray(x).astype(np.int64)
    for ch in word:
        y = (y + 1) % 10 if ch == "D" else 9 - y
    return y.astype(np.uint8)


def _main_slots(word, k, rng) -> np.ndarray:
    s = IS.slots_for(word, IS.subset(rng, S, k), S)
    return np.concatenate([[T.TASK_MAIN_OP], s])


def _aux_slots(word, k, rng) -> np.ndarray:
    s = np.full(S, V.ID_OP, dtype=np.int64)
    for p, ch in zip(IS.subset(rng, S, k), word):
        s[p] = AUX_SLOT[ch]
    return np.concatenate([[T.TASK_AUX_OP], s])


def aux_record(seed, k, j, heldout):
    rng = make_rng(seed, STREAM_AUX_RECORD, k, j)
    word = "".join(AUX_LETTERS[c] for c in rng.integers(0, 2, k))
    g = random_grids(make_rng(seed, STREAM_AUX_GRID, k, j), 1)[0]
    attempt = 0
    while heldout and grid_bytes(g) in heldout:
        g = random_grids(make_rng(seed, STREAM_GRID_RESAMPLE, 12000 + k, j, attempt), 1)[0]
        attempt += 1
    return word, g


class AuxStream:
    def __init__(self, seed: int, e: str, B: int, heldout_bytes=None):
        self.seed, self.e, self.B = int(seed), e, int(B)
        self.aks, self.n_aux = CONDITIONS[e]
        self.n_main = self.B - self.n_aux
        self.heldout = set(heldout_bytes or [])

    def batch(self, step: int) -> Dict[str, np.ndarray]:
        step = int(step)
        recs = []
        cnt = E.per_update_counts(MAIN_KS, self.n_main, step)
        for k in MAIN_KS:
            j0 = E.records_before(MAIN_KS, self.n_main, step, k)
            for j in range(j0, j0 + cnt[k]):
                word, _, g = E.record(self.seed, k, j, self.heldout)
                sl = _main_slots(word, k, make_rng(self.seed, IS.STREAM_SLOT_TRAIN, S, k, j))
                recs.append(("main", k, word, sl, g, V.apply_words(g[None], [word])[0]))
        if self.n_aux:
            cnt = E.per_update_counts(self.aks, self.n_aux, step)
            for k in self.aks:
                j0 = E.records_before(self.aks, self.n_aux, step, k)
                for j in range(j0, j0 + cnt[k]):
                    word, g = aux_record(self.seed, k, j, self.heldout)
                    sl = _aux_slots(word, k, make_rng(self.seed, STREAM_AUX_SLOT, k, j))
                    recs.append(("aux", k, word, sl, g, apply_aux(g, word)))
        recs = [recs[i] for i in make_rng(self.seed, E.STREAM_EXT_ORDER, step, 26).permutation(len(recs))]
        return {"x": np.stack([r[4] for r in recs]).astype(np.uint8), "y": np.stack([r[5] for r in recs]).astype(np.uint8),
                "slots": np.stack([r[3] for r in recs]), "k": np.array([r[1] for r in recs]),
                "group": np.zeros(len(recs), dtype=np.int64), "task": np.array([r[0] for r in recs]),
                "string": np.array([r[0] + ":" + to_label(r[2]) for r in recs]), "condition": "identity"}


def make_val_set(val_grids, seed):
    """Main task only (checkpoint selection): 300 records per main k; length key -> k-macro selection."""
    n = len(val_grids)
    rows = []
    for k in MAIN_KS:
        words = V._balanced_words(seed, k, n)
        rng = make_rng(seed, IS.STREAM_SLOT_VAL, S, k)
        for g in range(n):
            rows.append((g, k, words[g], _main_slots(words[g], k, rng)))
    gid = np.array([r[0] for r in rows])
    ks = np.array([r[1] for r in rows])
    words = [r[2] for r in rows]
    x = val_grids[gid].astype(np.uint8)
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), "slots": np.stack([r[3] for r in rows]), "k": ks,
            "group": np.zeros(len(rows), dtype=np.int64), "length": ks, "grid_id": gid,
            "string": np.array([to_label(w) for w in words]), "condition": "identity", "split": "val"}


def make_test(test_grids, seed, n_single_grids=1000, n_seen_per_k=5000, n_long_per_k=20000, n_aux_per_k=1000):
    """Main: the notebook-23 sampled S7_k02345 test set (identical records), prefixed with <main>.
    Aux: n_aux_per_k records for every aux k in 0..7 (grids from the test grids)."""
    from .longk import make_test_sampled
    t = make_test_sampled(test_grids, seed, "S7_k02345", n_single_grids, n_seen_per_k, n_long_per_k)
    n = len(t["x"])
    slots = np.concatenate([np.full((n, 1), T.TASK_MAIN_OP), t["slots"]], axis=1)
    rows = []
    for k in range(S + 1):
        words = ["".join(w) for w in product(AUX_LETTERS, repeat=k)]
        rng = make_rng(seed, STREAM_AUX_TEST, k)
        order = rng.permutation(len(words))
        gids = rng.integers(0, len(test_grids), n_aux_per_k)
        for r in range(n_aux_per_k):
            w = words[order[r % len(words)]]
            rows.append((int(gids[r]), w, _aux_slots(w, k, rng)))
    ax = test_grids[np.array([r[0] for r in rows])].reshape(len(rows), 6, 6).astype(np.uint8)
    ay = np.stack([apply_aux(x, r[1]) for x, r in zip(ax, rows)])
    ak = np.array([len(r[1]) for r in rows])
    return {"x": np.concatenate([t["x"], ax]), "y": np.concatenate([t["y"], ay]),
            "slots": np.concatenate([slots, np.stack([r[2] for r in rows])]),
            "k": np.concatenate([t["k"], ak]), "group": np.zeros(n + len(rows), dtype=np.int64),
            "grid_id": np.concatenate([t["grid_id"], [r[0] for r in rows]]),
            "string": np.concatenate([t["string"].astype("U12"), np.array(["aux:" + (r[1] or "EMPTY") for r in rows])]),
            "kind": np.concatenate([t["kind"].astype("U8"), np.array([f"aux{k}" for k in ak])]),
            "position": np.concatenate([t["position"], np.zeros(len(rows), dtype=int)]), "condition": "identity"}


def model_config(model_id: str, overrides=None) -> dict:
    import dataclasses
    from .models.registry import default_config
    c = dataclasses.asdict(default_config(model_id))
    c.update(overrides or {})
    c.update(n_slots=CAPACITY, op_format="variable", identity_token=True, aux_vocab=True)
    return c
