"""Behavioural diagnostics of saved AR predictions (notebook 17): the exact 288-element spatial group with a canonical
mirror/rotation/translation parameterization, honest exact matching of predicted grids to legal transformations,
residual (extra-C) classification, nearest-transform agreement and held-out source-position copying maps.

Conventions (verified in ``verify_conventions``):
  * X is a 6x6 grid flattened row-major; a transformation g is stored as its OUTPUT-TO-SOURCE permutation p_g with
    g(x) = x[p_g].
  * "h then g" (g o h) has permutation p_h[p_g]; ``Group.mult[a, b]`` is the id of "a then b" = p_a[p_b].
    A logical word is executed left to right, so p_AB = p_A[p_B].
  * Canonical form g = T_(dr,dc) o B^r o C^m: mirror first, rotation second, translation last, with
    T_(dr,dc)(X)[r,c] = X[(r-dr) mod 6, (c-dc) mod 6]  (A = T_(0,1)).
  * The GEOMETRIC mirror class is m (= number of C tokens mod 2 for any word). It is NOT the parity of the
    36-cell permutation: C is an even cell permutation and B an odd one.
  * Delta_out = P_hat o P^-1 (prediction = Delta_out applied to the correct output): p = p_true_inv[p_pred].
    Delta_in  = P^-1 o P_hat (prediction = P applied after Delta_in on the input):   p = p_pred[p_true_inv].
    extra output-C: P_hat = C o P   (p_pred = p_true[p_C]),  i.e. Delta_out == C;
    extra input-C:  P_hat = P o C   (p_pred = p_C[p_true]),  i.e. Delta_in  == C.

Nothing here trains or runs a model; all functions operate on saved inputs / predictions.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

N = 6
NC = 36
OUTCOMES = ["correct", "wrong: invalid output", "wrong: valid, no legal match",
            "wrong: valid, unique legal transform", "wrong: valid, multiple legal transforms"]
RESIDUALS = ["extra output-C only", "extra input-C only", "both extra-C hypotheses",
             "other orientation-reversing", "orientation-preserving nonidentity", "ambiguous residual"]


def _src(fn) -> np.ndarray:
    return np.array([fn(r, c) for r in range(N) for c in range(N)], dtype=np.int64)


P_A = _src(lambda r, c: r * N + (c - 1) % N)
P_B = _src(lambda r, c: (N - 1 - c) * N + r)
P_C = _src(lambda r, c: r * N + (N - 1 - c))
P_I = np.arange(NC, dtype=np.int64)
GENERATORS = {"A": P_A, "B": P_B, "C": P_C, "I": P_I}


def translation(dr: int, dc: int) -> np.ndarray:
    return _src(lambda r, c: ((r - dr) % N) * N + (c - dc) % N)


def then(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Permutation of 'first, then second' (second o first)."""
    return np.asarray(first)[np.asarray(second)]


def word_perm(word: str) -> np.ndarray:
    p = P_I.copy()
    for ch in word:
        p = then(p, GENERATORS[ch])
    return p


class Group:
    """All 288 transformations, id = ((m*4 + r)*6 + dr)*6 + dc for g = T_(dr,dc) o B^r o C^m."""

    def __init__(self):
        perms, params = [], []
        for m in range(2):
            for r in range(4):
                for dr in range(N):
                    for dc in range(N):
                        p = P_I.copy()
                        for _ in range(m):
                            p = then(p, P_C)
                        for _ in range(r):
                            p = then(p, P_B)
                        p = then(p, translation(dr, dc))
                        perms.append(p)
                        params.append((m, r, dr, dc))
        self.perms = np.stack(perms)
        self.params = np.array(params, dtype=np.int64)
        self.mirror = self.params[:, 0].copy()
        self.index = {p.tobytes(): i for i, p in enumerate(self.perms)}
        assert len(self.index) == 288, "canonical representations are not distinct"
        self.size = 288
        self.mult = np.array([[self.index[then(a, b).tobytes()] for b in self.perms] for a in self.perms])
        self.inv = np.array([self.index[np.argsort(p).tobytes()] for p in self.perms])
        self.I, self.A, self.B, self.C = (self.index[g.tobytes()] for g in (P_I, P_A, P_B, P_C))

    def id_of(self, p) -> int:
        return self.index.get(np.asarray(p, dtype=np.int64).tobytes(), -1)

    def word_id(self, word: str) -> int:
        return self.index[word_perm(word).tobytes()]

    def label(self, gid: int) -> str:
        if gid < 0:
            return "none"
        m, r, dr, dc = self.params[gid]
        parts = [f"T({dr},{dc})" if (dr or dc) else "", f"B^{r}" if r else "", "C" if m else ""]
        s = " o ".join(x for x in parts if x)
        return s or "I"


_GROUP = None


def group() -> Group:
    global _GROUP
    if _GROUP is None:
        _GROUP = Group()
    return _GROUP


def verify_conventions(G: Group = None) -> dict:
    """Independent array checks of every convention used by the diagnostics. Raises on any violation."""
    G = G or group()
    rng = np.random.default_rng(0)
    X = rng.integers(0, 10, (N, N))
    idx = np.arange(NC).reshape(N, N)   # indexed grid: oracle verification only, never a model input
    A = lambda Z: np.roll(Z, 1, axis=1)                          # noqa: E731  Z[r,(c-1) mod 6]
    B = lambda Z: np.array([[Z[N - 1 - c, r] for c in range(N)] for r in range(N)])  # noqa: E731
    C = lambda Z: Z[:, ::-1]                                     # noqa: E731
    for name, f, p in (("A", A, P_A), ("B", B, P_B), ("C", C, P_C)):
        assert np.array_equal(f(idx).reshape(-1), p), name
        assert np.array_equal(f(X).reshape(-1), X.reshape(-1)[p]), name
    assert np.array_equal(B(A(idx)).reshape(-1), word_perm("AB")) and np.array_equal(word_perm("AB"), P_A[P_B])
    assert not np.array_equal(P_A[P_B], P_B[P_A])
    for w, order in (("A", 6), ("B", 4), ("C", 2)):
        assert np.array_equal(word_perm(w * order), P_I)
        assert all(not np.array_equal(word_perm(w * j), P_I) for j in range(1, order))
    assert not np.array_equal(word_perm("AAA"), P_I)
    assert np.array_equal(word_perm("CAC"), np.argsort(P_A)) and np.array_equal(word_perm("CBC"), np.argsort(P_B))
    for (dr, dc) in ((0, 1), (2, 3), (5, 0)):
        T = translation(dr, dc)
        assert np.array_equal(np.roll(idx, (dr, dc), axis=(0, 1)).reshape(-1), T)
    assert np.array_equal(translation(0, 1), P_A)
    # canonical decomposition, mirror class = #C mod 2 for every word up to length 6, cell parity differs
    from itertools import product
    for k in range(7):
        for w in product("ABC", repeat=k):
            gid = G.word_id("".join(w))
            assert G.mirror[gid] == w.count("C") % 2
    parity = lambda p: int(round(np.linalg.det(np.eye(NC)[p]))) # noqa: E731
    cell_parity = {"A": parity(P_A), "B": parity(P_B), "C": parity(P_C)}
    assert cell_parity["C"] == 1 and cell_parity["B"] == -1
    # multiplication / inverse and residual formulas
    for _ in range(50):
        a, b = rng.integers(0, 288, 2)
        assert np.array_equal(G.perms[G.mult[a, b]], G.perms[a][G.perms[b]])
        assert np.array_equal(G.perms[G.inv[a]], np.argsort(G.perms[a]))
        t, q = int(a), int(b)
        dout, din = residuals(G, t, q)
        tinv = np.argsort(G.perms[t])
        assert np.array_equal(G.perms[dout], tinv[G.perms[q]]) and np.array_equal(G.perms[din], G.perms[q][tinv])
        # prediction = Delta_out applied to the correct output; = P after Delta_in on the input
        y = X.reshape(-1)[G.perms[t]]
        assert np.array_equal(y[G.perms[dout]], X.reshape(-1)[G.perms[q]])
        assert np.array_equal(X.reshape(-1)[G.perms[din]][G.perms[t]], X.reshape(-1)[G.perms[q]])
    t = G.word_id("AB")
    assert residuals(G, t, G.mult[t, G.C])[0] == G.C and residuals(G, t, G.mult[G.C, t])[1] == G.C
    assert G.mult[G.C, G.C] == G.I
    return {"group_size": G.size, "canonical_unique": True, "AB_is_pA[pB]": True, "orders": "A^6=B^4=C^2=I, A^3!=I",
            "CAC=A^-1, CBC=B^-1": True, "mirror_class=#C mod 2 (words k<=6)": True,
            "cell_permutation_parity": cell_parity, "residual_formulas": True}


def residuals(G: Group, t: int, q: int) -> Tuple[int, int]:
    """(Delta_out, Delta_in) ids for true transform t and predicted transform q."""
    tinv = G.inv[t]
    return int(G.mult[tinv, q]), int(G.mult[q, tinv])


def residual_class(G: Group, t: int, q: int) -> str:
    dout, din = residuals(G, t, q)
    oc, ic = dout == G.C, din == G.C
    if oc and ic:
        return "both extra-C hypotheses"
    if oc:
        return "extra output-C only"
    if ic:
        return "extra input-C only"
    return "other orientation-reversing" if G.mirror[dout] else "orientation-preserving nonidentity"


class Matcher:
    """Per input grid: dict full 36-byte transformed output -> ALL matching group ids. Cached by grid hash (LRU)."""

    def __init__(self, G: Group = None, max_grids: int = 4000):
        self.G = G or group()
        self.cache: "OrderedDict[str, dict]" = OrderedDict()
        self.max_grids = max_grids
        self.builds = 0

    def table(self, grid_hash: str, x36: np.ndarray) -> dict:
        d = self.cache.get(grid_hash)
        if d is not None:
            self.cache.move_to_end(grid_hash)
            return d
        out = np.asarray(x36, dtype=np.int8).reshape(NC)[self.G.perms]          # [288, 36]
        d = {}
        for i, row in enumerate(out):
            d.setdefault(row.tobytes(), []).append(i)
        d = {k: tuple(v) for k, v in d.items()}
        self.cache[grid_hash] = d
        self.builds += 1
        if len(self.cache) > self.max_grids:
            self.cache.popitem(last=False)
        return d

    def match(self, pred: np.ndarray, inputs: np.ndarray, grid_hash: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Object array of tuples (all exact legal matches; () for invalid or no match)."""
        pred = np.ascontiguousarray(np.asarray(pred, dtype=np.int8).reshape(len(pred), NC))
        inputs = np.asarray(inputs).reshape(len(inputs), NC)
        out = np.empty(len(pred), dtype=object)
        empty = ()
        for i in range(len(pred)):
            if not valid[i]:
                out[i] = empty
                continue
            out[i] = self.table(grid_hash[i], inputs[i]).get(pred[i].tobytes(), empty)
        return out


def brute_force_match(G: Group, pred: np.ndarray, x: np.ndarray) -> tuple:
    x = np.asarray(x).reshape(NC)
    return tuple(i for i in range(G.size) if np.array_equal(x[G.perms[i]], np.asarray(pred).reshape(NC)))


def classify(G: Group, matches: np.ndarray, true_ids: np.ndarray, valid: np.ndarray, correct: np.ndarray) -> dict:
    """Per-record outcome / attribute / hypothesis / residual columns (lists of equal length)."""
    n = len(matches)
    outcome = np.empty(n, dtype=object)
    pred_mirror = np.empty(n, dtype=object)
    residual = np.empty(n, dtype=object)
    unique_id = np.full(n, -1, dtype=np.int16)
    dout_id = np.full(n, -1, dtype=np.int16)
    n_match = np.zeros(n, dtype=np.int16)
    h_id = np.zeros(n, bool)
    h_true = np.zeros(n, bool)
    h_oc = np.zeros(n, bool)
    h_ic = np.zeros(n, bool)
    oc_of = G.mult[:, G.C]     # t then C   -> extra output-C prediction
    ic_of = G.mult[G.C, :]     # C then t   -> extra input-C prediction
    for i in range(n):
        ms = matches[i]
        t = int(true_ids[i])
        n_match[i] = len(ms)
        if correct[i]:
            outcome[i] = "correct"
        elif not valid[i]:
            outcome[i] = "wrong: invalid output"
        elif not ms:
            outcome[i] = "wrong: valid, no legal match"
        elif len(ms) == 1:
            outcome[i] = "wrong: valid, unique legal transform"
        else:
            outcome[i] = "wrong: valid, multiple legal transforms"
        if not valid[i]:
            pred_mirror[i] = "invalid"
        elif not ms:
            pred_mirror[i] = "no legal match"
        else:
            mm = {int(G.mirror[q]) for q in ms}
            pred_mirror[i] = str(mm.pop()) if len(mm) == 1 else "ambiguous"
        if ms:
            s = set(ms)
            h_id[i] = G.I in s
            h_true[i] = t in s
            h_oc[i] = int(oc_of[t]) in s
            h_ic[i] = int(ic_of[t]) in s
            if len(ms) == 1:
                unique_id[i] = ms[0]
        if ms and not correct[i]:
            cats = {residual_class(G, t, q) for q in ms}
            residual[i] = cats.pop() if len(cats) == 1 else "ambiguous residual"
            if len(ms) == 1:
                dout_id[i] = residuals(G, t, ms[0])[0]
        else:
            residual[i] = ""
    true_m = G.mirror[np.asarray(true_ids, dtype=np.int64)]
    pm_known = np.isin(pred_mirror, ["0", "1"])
    mirror_wrong = np.where(pm_known, pred_mirror != true_m.astype(str), False)
    return {"outcome": outcome, "n_match": n_match, "unique_id": unique_id, "pred_mirror": pred_mirror,
            "true_mirror": true_m, "mirror_identifiable": pm_known, "mirror_wrong": mirror_wrong,
            "match_identity": h_id, "match_true": h_true, "match_extra_out_C": h_oc, "match_extra_in_C": h_ic,
            "residual": residual, "delta_out_id": dout_id}


def nearest(G: Group, pred: np.ndarray, x: np.ndarray) -> dict:
    """Cell agreement of a prediction with every legal transform of its input (exact legality is separate)."""
    ag = (np.asarray(x).reshape(NC)[G.perms] == np.asarray(pred).reshape(NC)).sum(1)
    order = np.argsort(-ag, kind="stable")
    best = int(ag[order[0]])
    ties = tuple(int(i) for i in np.flatnonzero(ag == best))
    second = int(ag[order[len(ties)]]) if len(ties) < G.size else best
    return {"best_cells": best, "n_ties": len(ties), "tied_ids": ties, "margin": best - second,
            "tied_mirror": "".join(sorted({str(int(G.mirror[i])) for i in ties}))}


def source_map(pred_fit, x_fit, valid_fit, pred_ev, x_ev, valid_ev, G: Group = None) -> dict:
    """Fixed source-position copying map: on the fit half, count for each output cell how often the model's
    prediction equals each of the 36 input cells; argmax with ties retained (point map tie rule: lowest source
    index). Fidelity is measured on the held-out half (exact grid and cells; invalid predictions count as
    mismatches). Descriptive only."""
    G = G or group()
    P = np.asarray(pred_fit).reshape(-1, NC)
    X = np.asarray(x_fit).reshape(-1, NC)
    vf = np.asarray(valid_fit, bool)
    counts = np.zeros((NC, NC), dtype=np.int64)
    for o in range(NC):
        counts[o] = ((P[:, o][:, None] == X) & vf[:, None]).sum(0)
    best = counts.max(1)
    ties = (counts == best[:, None]).sum(1)
    mp = counts.argmax(1)                       # lowest source index among ties
    Pe = np.asarray(pred_ev).reshape(-1, NC)
    Xe = np.asarray(x_ev).reshape(-1, NC)
    ve = np.asarray(valid_ev, bool)
    mapped = Xe[:, mp]
    cell = (mapped == Pe) & ve[:, None]
    bij = len(set(mp.tolist())) == NC
    gid = G.id_of(mp) if bij else -1
    return {"map": mp, "fit_records": len(P), "eval_records": len(Pe),
            "fit_support_mean": float(best.mean() / max(1, len(P))), "cells_with_tied_sources": int((ties > 1).sum()),
            "bijective": bij, "distinct_sources": len(set(mp.tolist())), "legal_transform": G.label(gid) if gid >= 0 else "not legal",
            "legal_id": gid, "heldout_exact_fidelity": float(cell.all(1).mean()) if len(Pe) else np.nan,
            "heldout_cell_fidelity": float(cell.mean()) if len(Pe) else np.nan}
