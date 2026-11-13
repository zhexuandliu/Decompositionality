"""Notebook 31: ablations of the original Transformer AR on S7_k02345 (what makes it worse than the causal encoder?).

  transformer_ar          original (notebook 23b)
  ar_slotpos              no 1-D position table pos(t); op tokens get an op-slot embedding instead
  ar_digiteos             11-way output head (digits 0-9 + EOS); loss on 36 digits + EOS; 37 decoding steps
  ar_slotpos_digiteos     both
  (10-way, no-EOS variants ar_digithead / ar_slotpos_digithead are also registered but not run)
  query_ar / causal_encoder   references (notebooks 29b / 23b)

Every ablation shares all other initial tensors with transformer_ar under the same seed (the slot embedding starts
from the original pos(t) rows of the op slots, the digit head from the original head's digit rows). Same data stream,
20,000 updates, the same sampled test records, LAST checkpoint.
"""
from __future__ import annotations

import numpy as np

from . import longk as LK
from . import longk_exp as LX

CONDITION = "S7_k02345"
SUBDIR = "S7_support_sweep_20k_v1"
ABLATIONS = ["ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos"]
COMPARE = ["transformer_ar", "ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos", "nexttoken_ar", "query_ar",
           "causal_encoder"]
LABELS = {"transformer_ar": "original AR", "ar_slotpos": "AR − pos(t)", "ar_digithead": "AR + 10-way head",
          "ar_slotpos_digithead": "AR − pos(t) + 10-way head", "ar_digiteos": "AR + 11-way head (digits+EOS)",
          "ar_slotpos_digiteos": "AR − pos(t) + 11-way head", "nexttoken_ar": "query AR, next-token feedback", "query_ar": "query AR", "causal_encoder": "causal encoder"}


def report(results_root, mode="full", seed=42, models=None, show=True, log=print,
           review_name="AR_Ablation_S7_k02345_Review.ipynb"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from . import mirror_diag as M
    from .longk_analysis import _records
    from .mirror_report import Report, write_review
    from .plotstyle import MODEL_STYLE as STYLE
    models = list(models or COMPARE)
    root = LX.family_root(results_root, mode, SUBDIR)
    out = root / "report" / f"seed_{seed}_ar_ablation"
    R = Report(show)
    S, ks = LK.CONDITIONS[CONDITION]
    R.md(f"# Transformer AR ablations on {CONDITION} (seed {seed}, LAST checkpoint, 20,000 updates)\n"
         "Trained k = 0,2,3,4,5 in 7 slots; held out: singletons (k=1) at every slot, k=6, k=7. "
         "References: query AR (29b) and causal encoder (23b).")
    G = M.group()
    acc, single, drops, status = [], [], [], []
    for m in models:
        d, why = _records(results_root, mode, seed, SUBDIR, CONDITION, m)
        status.append({"model": m, "label": LABELS.get(m, m), "status": why})
        if d is None:
            continue
        k, ex, valid = d["k"].astype(int), d["exact"], d["valid"]
        for kk in sorted(set(k.tolist())):
            q = k == kk
            acc.append({"model": LABELS.get(m, m), "k": kk, "exact_acc": ex[q].mean(), "valid_rate": valid[q].mean(),
                        "role": "trained" if kk in ks else ("singleton" if kk == 1 else f"+{kk - max(ks)} step")})
        kind, words = d["kind"].astype(str), d["string"].astype(str)
        sq = kind == "single"
        for op in "ABC":
            for p in range(1, S + 1):
                q = sq & (words == op) & (d["position"] == p)
                single.append({"model": LABELS.get(m, m), "op": op, "slot": p, "exact_acc": ex[q].mean()})
        # k = 6 failures that equal the word with exactly one operation dropped
        q6 = k == 6
        X, P = d["x"].reshape(-1, 36)[q6], d["pred"][q6]
        w6, f6, v6 = words[q6], ~ex[q6], valid[q6]
        hit = np.zeros((q6.sum(), 6), bool)
        for i in range(6):
            for w in np.unique(w6[f6]):
                q = f6 & (w6 == w) & v6
                if q.any():
                    hit[q, i] = (X[q][:, G.perms[G.word_id(w[:i] + w[i + 1:])]] == P[q]).all(1)
        nf = int(f6.sum())
        drops.append({"model": LABELS.get(m, m), "k=6 records": int(q6.sum()), "failed": nf,
                      "invalid outputs": int((~v6).sum()),
                      **{f"drop #{i + 1}": int(hit[:, i].sum()) for i in range(6)},
                      "drop some op / failed": round(hit.any(1).sum() / nf, 3) if nf else np.nan})
    R.table("Availability", pd.DataFrame(status), 20)
    if not acc:
        return None
    A = pd.DataFrame(acc)
    wide = A.pivot_table(index=["k", "role"], columns="model", values="exact_acc", sort=False)
    wide = wide[[LABELS.get(m, m) for m in models if LABELS.get(m, m) in wide.columns]].reset_index().round(4)
    R.table("1. Exact accuracy by test k (k=1 = singletons over all slots)", wide, 20)
    Sg = pd.DataFrame(single)
    by_op = Sg.groupby(["op", "model"], sort=False).exact_acc.mean().unstack("model")
    by_op = by_op[[c for c in wide.columns if c in by_op.columns]].reset_index().round(4)
    R.table("2a. Singleton accuracy by operation", by_op, 10)
    by_slot = Sg.pivot_table(index=["op", "slot"], columns="model", values="exact_acc", sort=False)
    by_slot = by_slot[[c for c in wide.columns if c in by_slot.columns]].reset_index().round(3)
    R.table("2b. Singleton accuracy by operation and slot", by_slot, 30)
    R.table("3. k=6 failures that equal the word with ONE operation dropped (counts; 'drop #i' = i-th operation)",
            pd.DataFrame(drops), 10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kk in ks:
        ax.axvspan(kk - 0.5, kk + 0.5, color="#eeeeee", zorder=0)
    for m in models:
        q = A[A.model == LABELS.get(m, m)].sort_values("k")
        if len(q):
            st = STYLE.get(m, {})
            ax.plot(q.k, q.exact_acc, marker="o", color=st.get("color"), ls=st.get("ls", "-"), lw=2, label=LABELS.get(m, m))
    ax.set(xlabel="test k (k=1: singletons; grey = trained)", ylabel="exact accuracy", ylim=(-0.03, 1.05),
           xticks=range(0, S + 1), title=f"{CONDITION}, seed {seed}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    R.fig(fig, "Accuracy vs test k", out / "figures" / "acc_vs_k.png")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    A.to_csv(out / "tables" / "acc_by_k.csv", index=False)
    Sg.to_csv(out / "tables" / "singletons.csv", index=False)
    rp = write_review(R.items, out, "Transformer AR ablations",
                      "# Transformer AR ablations on S7_k02345 — review snapshot\n\nBaked outputs; nothing to run.",
                      review_name)
    log(f"REVIEW FILE: {rp}")
    return rp
