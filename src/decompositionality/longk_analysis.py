"""Notebook 24: read-only analysis of the S=7 sweep (notebooks 23a/23b) longer-instruction results.

For S7_k02345 (held-out k=6 and k=7) and S7_k023456 (held-out k=7), both models, LAST checkpoint:
  1. k=6 accuracy by WHICH slot holds the single <identity> (7 layouts);
  2. k=6 / k=7 accuracy by number of C tokens and by whether the word's overall transform occurs in training;
     per-word accuracy distribution (full per-word CSV exported);
  3. failure types (exact matching to the 288 legal transforms: identity copy / other legal / non-legal / invalid);
  4. "dropped one operation": does a wrong prediction equal the word with exactly one operation removed (as if
     that slot were <identity>)? Reported by the dropped operation's order index and, for k=6, by I slot.
The sampled test set is rebuilt deterministically (longk.make_test_sampled with the saved plan) and checked
record-by-record against the saved predictions (word and grid id). Nothing is retrained or re-evaluated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import idvar as V
from . import longk as LK
from . import longk_exp as LX
from . import mirror_diag as M
from .words import words_of_length

TARGETS = {"S7_k02345": [6, 7], "S7_k023456": [7]}


def _records(results_root, mode, seed, subdir, e, m):
    from .data import load_or_build_manifests
    cfg = LX.make_cfg(e, results_root, mode, [m], seed, subdir)
    af = LX.eval_dir(cfg, m) / "ARTIFACT.json"
    if not af.exists():
        return None, "no ARTIFACT"
    a = json.loads(af.read_text())
    if not a.get("complete") or a.get("evaluation") != "sampled":
        return None, "incomplete or not a sampled evaluation"
    man = load_or_build_manifests(cfg.manifest_dir, seed)
    es = LK.make_test_sampled(man.test_grids[: cfg.n_test_grids], seed, e, **a["plan"])
    d = LX.load_eval(cfg, m)
    if len(d.get("exact", [])) != len(es["x"]) or not (d["string"].astype(str) == es["string"].astype(str)).all() \
            or not (d["grid_id"].astype(int) == es["grid_id"].astype(int)).all():
        return None, "saved predictions do not align with the rebuilt test set"
    return {**es, "pred": d["pred"].reshape(-1, 36), "exact": d["exact"].astype(bool),
            "valid": d["valid"].astype(bool), "plan": a["plan"]}, "ok"


def analyse(results_root: str, mode: str = "full", seed: int = 42, subdir: str = "S7_support_sweep_20k_v1",
            show: bool = True, log=print) -> Path:
    import pandas as pd
    import matplotlib.pyplot as plt
    from .mirror_report import Report, write_review
    G = M.group()
    R = Report(show)
    root = LX.family_root(results_root, mode, subdir)
    out = root / "report" / f"seed_{seed}_longer_k_analysis"
    R.md("# Longer-instruction analysis of the S=7 sweep (read-only; LAST checkpoint)\n"
         "Sampled test sets are rebuilt deterministically and checked record-by-record against the saved predictions. "
         "Held-out longer k: S7_k02345 → k=6 (one `<identity>` left) and k=7 (no `<identity>`); S7_k023456 → k=7.")
    T = {k: [] for k in ("status", "by_I_slot", "by_nC", "by_transform", "word_summary", "failure_types",
                         "dropped_op", "dropped_op_by_I_slot")}
    perword = []
    for e, ks in TARGETS.items():
        S, train_k = LK.CONDITIONS[e]
        tt = {G.word_id(w) for k in train_k for w in words_of_length(k)}
        for m in LK.MODELS:
            r, msg = _records(results_root, mode, seed, subdir, e, m)
            T["status"].append({"condition": e, "model": m, "status": msg})
            if r is None:
                continue
            for k in ks:
                sel = np.flatnonzero(r["k"] == k)
                words = r["string"][sel].astype(str)
                ex = r["exact"][sel]
                X = r["x"][sel].reshape(-1, 36)
                P = r["pred"][sel]
                valid = r["valid"][sel]
                slots = r["slots"][sel]
                nC = np.char.count(words, "C")
                tid = np.array([G.word_id(w) for w in words])
                inT = np.isin(tid, sorted(tt))
                base = {"condition": e, "model": m, "k": k}
                if k < S:     # one <identity>: which slot
                    islot = (slots == V.ID_OP).argmax(1) + 1
                    for p in range(1, S + 1):
                        q = islot == p
                        T["by_I_slot"].append({**base, "I slot": p, "records": int(q.sum()), "correct": int(ex[q].sum()),
                                               "exact_acc": float(ex[q].mean()) if q.any() else np.nan})
                for c in sorted(set(nC.tolist())):
                    q = nC == c
                    T["by_nC"].append({**base, "n_C": c, "parity": "odd" if c % 2 else "even", "records": int(q.sum()),
                                       "words": len(set(words[q])), "exact_acc": float(ex[q].mean())})
                for flag in (True, False):
                    q = inT == flag
                    T["by_transform"].append({**base, "overall transform occurs in training": flag,
                                              "words": len(set(words[q])), "records": int(q.sum()),
                                              "exact_acc": float(ex[q].mean()) if q.any() else np.nan})
                pw = pd.DataFrame({"word": words, "exact": ex}).groupby("word").exact.agg(["sum", "size", "mean"]).reset_index()
                pw.columns = ["word", "correct", "records", "exact_acc"]
                pw.insert(0, "k", k); pw.insert(0, "model", m); pw.insert(0, "condition", e)
                perword.append(pw)
                T["word_summary"].append({**base, "words": len(pw), "records per word (median)": float(pw.records.median()),
                                          "words always right": int((pw.exact_acc == 1).sum()),
                                          "words always wrong": int((pw.exact_acc == 0).sum()),
                                          "per-word acc median": float(pw.exact_acc.median()),
                                          "per-word acc IQR": f"{pw.exact_acc.quantile(.25):.2f}-{pw.exact_acc.quantile(.75):.2f}"})
                # failure types
                gh = np.array([hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest()[:16] for x in X])
                ms = M.Matcher(G).match(P, X, gh, valid)
                cl = M.classify(G, ms, tid, valid, ex)
                fail = ~ex
                oc = pd.Series(cl["outcome"][fail]).value_counts()
                T["failure_types"].append({**base, "records": len(ex), "correct": int(ex.sum()), "failed": int(fail.sum()),
                                           "identity copy": int((fail & cl["match_identity"]).sum()),
                                           "wrong legal transform": int(oc.get("wrong: valid, unique legal transform", 0)
                                                                        + oc.get("wrong: valid, multiple legal transforms", 0)),
                                           "  of which wrong mirror class": int((fail & cl["mirror_wrong"]).sum()),
                                           "non-legal grid": int(oc.get("wrong: valid, no legal match", 0)),
                                           "invalid output": int(oc.get("wrong: invalid output", 0))})
                # dropped-one-op: prediction == word with the i-th operation removed (that slot treated as identity)
                hit = np.zeros((len(ex), k), bool)
                for i in range(k):
                    for w in np.unique(words[fail]):
                        q = fail & (words == w) & valid
                        if not q.any():
                            continue
                        p = G.perms[G.word_id(w[:i] + w[i + 1:])]
                        hit[q, i] = (X[q][:, p] == P[q]).all(1)
                anyhit = hit.any(1)
                nf = int(fail.sum())
                T["dropped_op"].append({**base, "failed": nf,
                                        **{f"= drop op #{i + 1}": int(hit[:, i].sum()) for i in range(k)},
                                        "= drop SOME op (union)": int(anyhit.sum()),
                                        "union / failed": round(anyhit.sum() / nf, 3) if nf else np.nan,
                                        "matches >1 drop (ambiguous)": int((hit.sum(1) > 1).sum())})
                if k < S:
                    for pslot in range(1, S + 1):
                        q = fail & (islot == pslot)
                        T["dropped_op_by_I_slot"].append({**base, "I slot": pslot, "failed": int(q.sum()),
                                                          "= drop some op": int((anyhit & q).sum())})
    for k_ in T:
        T[k_] = pd.DataFrame(T[k_])
    T["per_word"] = pd.concat(perword, ignore_index=True) if perword else pd.DataFrame()
    R.table("Sources", T["status"])
    R.table("1. k=6 (S7_k02345): exact accuracy by which slot holds the single <identity>", T["by_I_slot"], 30)
    R.table("2a. Accuracy by number of C tokens", T["by_nC"], 60)
    R.table("2b. Accuracy by whether the word's overall transform occurs in training (descriptive: different word sets)",
            T["by_transform"], 20)
    R.table("2c. Per-word accuracy summary (full per-word table in per_word.csv)", T["word_summary"], 20)
    if len(T["per_word"]):
        for (e, m, k), g in T["per_word"].groupby(["condition", "model", "k"]):
            g = g.sort_values("exact_acc")
            R.table(f"2d. {e} / {m} / k={k}: 8 worst and 8 best words",
                    pd.concat([g.head(8), g.tail(8)])[["word", "correct", "records", "exact_acc"]], 20)
    R.table("3. Failure types (exact matching to the 288 legal transforms)", T["failure_types"], 20)
    R.table("4. Does a wrong prediction equal the word with ONE operation removed? (by the removed op's order index; "
            "overlaps kept, union counted once)", T["dropped_op"], 20)
    R.table("4b. k=6: dropped-one-op matches by <identity> slot", T["dropped_op_by_I_slot"], 30)
    b = T["by_I_slot"]
    if len(b):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for mi, (m, col) in enumerate(zip(LK.MODELS, ("#2a78d6", "#8e6bd6"))):
            q = b[b.model == m].sort_values("I slot")
            ax.bar(q["I slot"] + (mi - 0.5) * 0.4, q.exact_acc, 0.4, color=col, label=m)
        ax.set(xlabel="slot holding the single <identity>", ylabel="k=6 exact accuracy", ylim=(0, 1.05),
               title="S7_k02345: held-out k=6 by identity position (last checkpoint)")
        ax.legend(fontsize=8)
        R.fig(fig, "k=6 accuracy by identity slot", out / "figures" / "k6_by_I_slot.png")
    pwd = T["per_word"]
    if len(pwd):
        groups = list(pwd.groupby(["condition", "model", "k"]))
        fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3), squeeze=False)
        for ax, ((e, m, k), g) in zip(axes[0], groups):
            ax.hist(g.exact_acc, bins=np.linspace(0, 1, 21), color="#2a78d6")
            ax.set_title(f"{e}\n{m} k={k}", fontsize=7)
            ax.set_xlabel("per-word exact acc", fontsize=7)
        fig.tight_layout()
        R.fig(fig, "Per-word accuracy distributions", out / "figures" / "per_word_hist.png")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k_, t in T.items():
        t.to_csv(out / "tables" / f"{k_}.csv", index=False)
    rp = write_review(R.items, out, "S=7 longer-instruction analysis",
                      "# S=7 longer-instruction analysis — review snapshot\n\nBaked outputs; nothing to run.",
                      "S7_LongerK_Analysis_Review.ipynb")
    log(f"REVIEW FILE: {rp}")
    return rp
