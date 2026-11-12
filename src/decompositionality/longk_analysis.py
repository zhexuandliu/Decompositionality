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


# ---------------------------------------------------------------------------
# Notebook 25: follow-up analyses
# ---------------------------------------------------------------------------

def _reduce_cancel(w: str) -> str:
    """Cancel adjacent identity blocks (CC, BBBB, AAAAAA) repeatedly; a purely syntactic reduction."""
    prev = None
    while prev != w:
        prev = w
        for blk in ("CC", "BBBB", "AAAAAA"):
            w = w.replace(blk, "")
    return w


def _max_run(w: str, ch: str) -> int:
    best = cur = 0
    for c in w:
        cur = cur + 1 if c == ch else 0
        best = max(best, cur)
    return best


def followup(results_root: str, mode: str = "full", seed: int = 42, subdir: str = "S7_support_sweep_20k_v1",
             show: bool = True, log=print) -> Path:
    import pandas as pd
    import matplotlib.pyplot as plt
    from .diagnosis.group import Catalog
    from .mirror_report import Report, write_review
    G = M.group()
    cat = Catalog()
    R = Report(show)
    root = LX.family_root(results_root, mode, subdir)
    out = root / "report" / f"seed_{seed}_longer_k_followup"
    R.md("# Follow-up analyses of the S=7 longer-instruction results (read-only; LAST checkpoint)\n"
         "1. Causal encoder, S7_k02345, k=7: which word structures succeed?\n"
         "2. Transformer AR, S7_k02345, k=6: which operation is 'dropped'?\n"
         "3. Causal encoder, S7_k02345, k=6: the early-`<identity>` failures.\n"
         "All groupings are descriptive (groups differ in other word properties too).")
    T = {}
    enc, msg_e = _records(results_root, mode, seed, subdir, "S7_k02345", "causal_encoder")
    ar, msg_a = _records(results_root, mode, seed, subdir, "S7_k02345", "transformer_ar")
    R.table("Sources", pd.DataFrame([{"run": "S7_k02345/causal_encoder", "status": msg_e},
                                     {"run": "S7_k02345/transformer_ar", "status": msg_a}]))
    # ---- 1. per-word structure, encoder k=7 -------------------------------------------------------------------
    if enc is not None:
        sel = enc["k"] == 7
        pw = pd.DataFrame({"word": enc["string"][sel].astype(str), "exact": enc["exact"][sel]}).groupby("word").exact.agg(
            ["sum", "size", "mean"]).reset_index()
        pw.columns = ["word", "correct", "records", "acc"]
        w = pw.word
        pw["first"] = w.str[0]
        pw["last"] = w.str[-1]
        pw["first2"] = w.str[:2]
        pw["last2"] = w.str[-2:]
        for ch in "ABC":
            pw[f"n_{ch}"] = w.str.count(ch)
        pw["max C run"] = [_max_run(x, "C") for x in w]
        pw["has CC"] = w.str.contains("CC")
        red = [_reduce_cancel(x) for x in w]
        pw["length after cancelling CC/BBBB/AAAAAA"] = [len(x) for x in red]
        pw["shortest word for the transform"] = [len(cat.words[cat.word_id(x)]) for x in w]
        pw["status"] = np.where(pw.acc == 1, "always right", np.where(pw.acc == 0, "always wrong", "mixed"))
        T["enc_k7_per_word"] = pw
        feats = ["first", "last", "first2", "last2", "n_A", "n_B", "n_C", "max C run", "has CC",
                 "length after cancelling CC/BBBB/AAAAAA", "shortest word for the transform"]
        rows = []
        for f in feats:
            g = pw.groupby(f).agg(words=("word", "size"), mean_word_acc=("acc", "mean"),
                                  always_right=("status", lambda s: int((s == "always right").sum())),
                                  always_wrong=("status", lambda s: int((s == "always wrong").sum()))).reset_index()
            g.insert(0, "feature", f)
            g = g.rename(columns={f: "value"})
            g["value"] = g["value"].astype(str)
            rows.append(g)
            # spread of the feature: range of group means (weighted by size >= 20 words)
        bf = pd.concat(rows, ignore_index=True)
        T["enc_k7_by_feature"] = bf
        spread = bf[bf.words >= 20].groupby("feature").mean_word_acc.agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
        R.table("1a. Causal encoder k=7: how much each word feature separates success (max − min of group mean word "
                "accuracy, groups with ≥ 20 words)", spread.rename("spread").reset_index(), 20)
        for f in list(spread.index[:4]) + ["length after cancelling CC/BBBB/AAAAAA", "shortest word for the transform"]:
            R.table(f"1b. by {f}", bf[bf.feature == f].drop(columns="feature"), 40)
        R.table("1c. Status counts (each word tested on ~9 grids)", pw.status.value_counts().rename("words").reset_index(), 5)
        # figure: accuracy vs length after cancellation
        g = pw.groupby("length after cancelling CC/BBBB/AAAAAA").acc.mean()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(g.index.astype(int), g.values, color="#8e6bd6")
        ax.set(xlabel="word length after cancelling CC / BBBB / AAAAAA", ylabel="mean word accuracy", ylim=(0, 1.05),
               title="Causal encoder, S7_k02345, held-out k=7")
        R.fig(fig, "Encoder k=7 accuracy vs reduced length", out / "figures" / "enc_k7_reduced_length.png")
    # ---- 2. AR k=6: which op is dropped ------------------------------------------------------------------------
    if ar is not None:
        sel = np.flatnonzero((ar["k"] == 6) & ~ar["exact"] & ar["valid"])
        words = ar["string"][sel].astype(str)
        X = ar["x"][sel].reshape(-1, 36)
        P = ar["pred"][sel]
        slots = ar["slots"][sel]
        hit = np.zeros((len(sel), 6), bool)
        for i in range(6):
            for wd in np.unique(words):
                q = words == wd
                p = G.perms[G.word_id(wd[:i] + wd[i + 1:])]
                hit[q, i] = (X[q][:, p] == P[q]).all(1)
        unique = hit.sum(1) == 1
        dropped = np.where(unique, hit.argmax(1), -1)
        T["ar_k6_drop_unique"] = pd.DataFrame({"dropped op #": np.arange(1, 7),
                                               "failures where this is the ONLY matching drop": [int((dropped == i).sum()) for i in range(6)],
                                               "failures matching this drop (incl. ambiguous)": [int(hit[:, i].sum()) for i in range(6)]})
        R.table("2a. Transformer AR k=6 failures: which operation's removal reproduces the prediction "
                f"({len(sel)} valid failures; 'only' = unambiguous)", T["ar_k6_drop_unique"], 10)
        islot = (slots == V.ID_OP).argmax(1) + 1
        op_slot = np.array([np.flatnonzero(s != V.ID_OP)[4] + 1 for s in slots])     # slot of the 5th operation
        letter = np.array([wd[4] for wd in words])
        d5 = hit[:, 4]
        rows = []
        for name, key in (("letter of op #5", letter), ("slot of op #5", op_slot), ("<identity> slot", islot)):
            for v in sorted(set(key.tolist())):
                q = key == v
                rows.append({"grouping": name, "value": str(v), "failures": int(q.sum()), "= drop op #5": int(d5[q].sum()),
                             "share": round(float(d5[q].mean()), 3)})
        T["ar_k6_drop5_by"] = pd.DataFrame(rows)
        R.table("2b. 'Drop op #5' share among AR k=6 failures, by the 5th operation's letter / slot and by the identity slot",
                T["ar_k6_drop5_by"], 40)
        # also: with the 5th op removed, what remains is a 5-op word: is it the 'first 4 + last 1'?
        T["ar_k6_note"] = pd.DataFrame([{"note": "drop op #5 = execute ops 1,2,3,4,6 (the last op is kept)"}])
        R.table("2c. Reading", T["ar_k6_note"], 2)
    # ---- 3. encoder k=6 early-identity failures ----------------------------------------------------------------
    if enc is not None:
        sel = np.flatnonzero(enc["k"] == 6)
        words = enc["string"][sel].astype(str)
        ex = enc["exact"][sel]
        slots = enc["slots"][sel]
        islot = (slots == V.ID_OP).argmax(1) + 1
        nC = np.char.count(words, "C")
        first = np.array([wd[0] for wd in words])
        rows = []
        for grp, q0 in (("I in slot 1-3", islot <= 3), ("I in slot 4-7", islot >= 4)):
            for name, key in (("n_C", nC), ("first op", first)):
                for v in sorted(set(key.tolist())):
                    q = q0 & (key == v)
                    if q.sum() < 30:
                        continue
                    rows.append({"layout": grp, "grouping": name, "value": str(v), "records": int(q.sum()),
                                 "exact_acc": round(float(ex[q].mean()), 4)})
        for v in range(1, 8):
            q = islot == v
            rows.append({"layout": f"I in slot {v}", "grouping": "all", "value": "-", "records": int(q.sum()),
                         "exact_acc": round(float(ex[q].mean()), 4)})
        T["enc_k6_early_I"] = pd.DataFrame(rows)
        R.table("3. Causal encoder k=6: early vs late identity slot, by number of C and by first operation",
                T["enc_k6_early_I"], 60)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k_, t in T.items():
        t.to_csv(out / "tables" / f"{k_}.csv", index=False)
    rp = write_review(R.items, out, "S=7 longer-instruction follow-up",
                      "# S=7 longer-instruction follow-up — review snapshot\n\nBaked outputs; nothing to run.",
                      "S7_LongerK_Followup_Review.ipynb")
    log(f"REVIEW FILE: {rp}")
    return rp
