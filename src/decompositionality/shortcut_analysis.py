"""Notebook 37 (read-only): WHAT do the models compute on held-out inputs when max trained k is short in S=7?

For each condition (S7_k02 / k023 / k0234, and S7_k02345 as the success reference) and model, using the saved
sampled LAST-checkpoint predictions (seed 42):

  1. singletons (one op, six <identity>): outcome = correct / identity copy (input returned unchanged) /
     another legal transform / several legal transforms (symmetric grid) / no legal transform / invalid;
  2. which transform a wrong singleton actually applies, written as its SHORTEST A/B/C word, and whether that
     transform is the overall transform of some TRAINED instruction;
  3. per (op, slot): the modal predicted transform and how consistently it is used across grids
     (systematic rule vs noise);
  4. the first held-out longer k (+1 step): the same outcome classes, plus "= the word with one op dropped".
"""
from __future__ import annotations

import hashlib
from collections import Counter, deque

import numpy as np

from . import longk as LK
from . import longk_exp as LX
from . import mirror_diag as M
from .words import words_of_length

SUBDIR = "S7_support_sweep_20k_v1"
CONDITIONS = ["S7_k02", "S7_k023", "S7_k0234", "S7_k02345"]
MODELS = ["transformer_ar", "ar_slotpos_digithead_target", "query_ar", "causal_encoder"]
LABELS = {"transformer_ar": "original AR", "ar_slotpos_digithead_target": "AR34", "query_ar": "query AR",
          "causal_encoder": "causal encoder", "ar_trace_slotstep": "Aligned AR + grids (slot step)",
          "ar_trace_sepstep": "Aligned AR + grids (separate step)",
          "ar_scratch_slotstep": "Aligned AR + scratch (no step loss)"}


def shortest_words(G) -> dict:
    """Group id -> a shortest A/B/C word (BFS, ties broken alphabetically)."""
    out = {G.I: "I"}
    q = deque([(G.I, "")])
    gen = {"A": G.A, "B": G.B, "C": G.C}
    while q:
        g, w = q.popleft()
        for ch in "ABC":
            h = int(G.mult[g, gen[ch]])                 # g then ch
            if h not in out:
                out[h] = w + ch
                q.append((h, w + ch))
    assert len(out) == G.size
    return out


def _outcome(ex, valid, ms, G):
    if ex:
        return "correct"
    if not valid:
        return "invalid output"
    if not ms:
        return "no legal transform"
    if G.I in ms and len(ms) == 1:
        return "identity copy"
    if G.I in ms:
        return "identity copy (ambiguous)"
    return "other legal transform" if len(ms) == 1 else "several legal transforms"


def analyse(results_root, mode="full", seed=42, conditions=None, models=None, show=True, log=print,
            review_name="Shortcut_Analysis_S7_short_k_Review.ipynb", out_name=None):
    import pandas as pd
    from .longk_analysis import _records
    from .mirror_report import Report, write_review
    G = M.group()
    SW = shortest_words(G)
    conditions = list(conditions or CONDITIONS)
    models = list(models or MODELS)
    R = Report(show)
    out = LX.family_root(results_root, mode, SUBDIR) / "report" / (out_name or f"seed_{seed}_shortcut_analysis")
    R.md(f"# What do the models compute on held-out inputs? (seed {seed}, LAST checkpoint)\n"
         "Read-only: rebuilt sampled test sets are checked record-by-record against the saved predictions. "
         "A transform is written as its shortest A/B/C word (A = right shift, B = clockwise rotation, "
         "C = left-right mirror; executed left to right). 'seen' = the transform is the overall transform of some "
         "trained instruction.")
    T = {k: [] for k in ("status", "outcome", "outcome_slot", "wrong_transform", "per_op_slot", "long", "drop_index")}
    for e in conditions:
        S, ks = LK.CONDITIONS[e]
        seen = {G.word_id(w) for k in ks for w in words_of_length(k)}
        for m in models:
            r, msg = _records(results_root, mode, seed, SUBDIR, e, m)
            T["status"].append({"condition": e, "model": LABELS[m], "status": msg})
            if r is None:
                continue
            X = r["x"].reshape(-1, 36)
            gh = np.array([hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest()[:16] for x in X])
            matcher = M.Matcher(G)
            kind, words = r["kind"].astype(str), r["string"].astype(str)
            base = {"condition": e, "model": LABELS[m]}
            # ---- singletons ------------------------------------------------------------------------------------
            sq = np.flatnonzero(kind == "single")
            ms = matcher.match(r["pred"][sq], X[sq], gh[sq], r["valid"][sq])
            oc = np.array([_outcome(r["exact"][i], r["valid"][i], mm, G) for i, mm in zip(sq, ms)])
            ops, pos = words[sq], r["position"][sq]
            for op in "ABC":
                q = ops == op
                c = Counter(oc[q])
                T["outcome"].append({**base, "op": op, "records": int(q.sum()),
                                     **{k: round(c.get(k, 0) / q.sum(), 3) for k in
                                        ("correct", "identity copy", "identity copy (ambiguous)", "other legal transform",
                                         "several legal transforms", "no legal transform", "invalid output")}})
                for p in range(1, S + 1):
                    qq = q & (pos == p)
                    c = Counter(oc[qq])
                    uid = [mm[0] for mm, o in zip(ms[qq], oc[qq]) if len(mm) == 1]
                    mode_id, mode_n = Counter(uid).most_common(1)[0] if uid else (-1, 0)
                    T["outcome_slot"].append({**base, "op": op, "slot": p, "correct": round(c.get("correct", 0) / qq.sum(), 3),
                                              "identity copy": round((c.get("identity copy", 0)) / qq.sum(), 3),
                                              "other legal": round(c.get("other legal transform", 0) / qq.sum(), 3),
                                              "no legal / invalid": round((c.get("no legal transform", 0)
                                                                           + c.get("invalid output", 0)) / qq.sum(), 3)})
                    T["per_op_slot"].append({**base, "op": op, "slot": p,
                                             "modal transform (shortest word)": SW.get(int(mode_id), "-"),
                                             "share of ALL records with the modal transform": round(mode_n / qq.sum(), 3),
                                             "modal transform seen in training": (int(mode_id) in seen) if mode_id >= 0 else None})
                wrong = [mm[0] for mm, o in zip(ms[q], oc[q]) if o in ("other legal transform",)]
                for gid, n in Counter(wrong).most_common(5):
                    T["wrong_transform"].append({**base, "op": op, "applied transform": SW[int(gid)],
                                                 "length of shortest word": len(SW[int(gid)].replace("I", "")),
                                                 "seen in training": int(gid) in seen,
                                                 "share of this op's singletons": round(n / q.sum(), 3)})
            # ---- first held-out longer k -----------------------------------------------------------------------
            k1 = max(ks) + 1
            lq = np.flatnonzero(r["k"] == k1)
            if len(lq):
                ms = matcher.match(r["pred"][lq], X[lq], gh[lq], r["valid"][lq])
                oc = np.array([_outcome(r["exact"][i], r["valid"][i], mm, G) for i, mm in zip(lq, ms)])
                w = words[lq]
                drop = np.zeros(len(lq), bool)
                trunc = np.zeros(len(lq), bool)
                seen_t = np.zeros(len(lq), bool)
                hit = np.zeros((len(lq), k1), bool)
                for j, (mm, ww) in enumerate(zip(ms, w)):
                    if not mm or oc[j] == "correct":
                        continue
                    s = set(mm)
                    for i in range(len(ww)):
                        hit[j, i] = G.word_id(ww[:i] + ww[i + 1:]) in s
                    drop[j] = hit[j].any()
                    trunc[j] = G.word_id(ww[:-1]) in s
                    seen_t[j] = bool(s & seen)
                uniq = hit.sum(1) == 1
                for i in range(k1):
                    role = "first" if i == 0 else ("last" if i == k1 - 1 else "middle")
                    T["drop_index"].append({**base, "k": k1, "dropped position": i + 1, "role": role,
                                            "unique": int((hit[:, i] & uniq).sum()), "any": int(hit[:, i].sum()),
                                            "share of unique drops": round((hit[:, i] & uniq).sum() / max(1, uniq.sum()), 3)})
                c = Counter(oc)
                n = len(lq)
                T["long"].append({**base, "k": k1, "records": n, "correct": round(c.get("correct", 0) / n, 3),
                                  "identity copy": round(c.get("identity copy", 0) / n, 3),
                                  "= one op dropped": round(drop.mean(), 3), "  of which = last op dropped": round(trunc.mean(), 3),
                                  "other legal transform": round((c.get("other legal transform", 0)
                                                                  + c.get("several legal transforms", 0)) / n, 3),
                                  "no legal / invalid": round((c.get("no legal transform", 0) + c.get("invalid output", 0)) / n, 3)})
    R.table("Availability", pd.DataFrame(T["status"]), 30)
    R.md("**Which operation is dropped?** For held-out longer instructions whose prediction equals the word with exactly "
         "ONE operation removed, the position (1 = first, k = last) of that operation. 'unique' counts only records where "
         "exactly one position fits; 'any' counts every fitting position.")
    R.table("5. First held-out longer k: which operation is dropped", pd.DataFrame(T["drop_index"]), 60)
    R.table("1. Singletons: outcome shares by operation", pd.DataFrame(T["outcome"]), 60)
    R.table("2. Wrong singletons that apply ONE other legal transform: the five most common, as shortest words",
            pd.DataFrame(T["wrong_transform"]), 200)
    R.table("3a. Singletons by operation and slot: outcome shares", pd.DataFrame(T["outcome_slot"]), 400)
    R.table("3b. Singletons by operation and slot: the modal predicted transform and its share of ALL records "
            "(1.0 = the same transform on every grid)", pd.DataFrame(T["per_op_slot"]), 400)
    R.table("4. First held-out longer k: outcome shares ('one op dropped' and 'other legal' overlap)",
            pd.DataFrame(T["long"]), 30)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, v in T.items():
        pd.DataFrame(v).to_csv(out / "tables" / f"{k}.csv", index=False)
    rp = write_review(R.items, out, "Shortcut analysis",
                      "# Shortcut analysis, S=7 short supports — review snapshot\n\nBaked outputs; nothing to run.",
                      review_name)
    log(f"REVIEW FILE: {rp}")
    return rp
