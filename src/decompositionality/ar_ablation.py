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
COMPARE = ["transformer_ar", "ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos", "ar_slotpos_digiteos_target",
           "ar_digithead", "ar_slotpos_digithead", "ar_digithead_target", "ar_slotpos_digithead_target", "nexttoken_ar", "query_ar", "causal_encoder"]
LABELS = {"transformer_ar": "original AR", "ar_slotpos": "AR − pos(t)", "ar_digithead": "AR + 10-way head (36 digits)",
          "ar_slotpos_digithead": "AR − pos(t) + 10-way head (36 digits)", "ar_digiteos": "AR + 11-way head (digits+EOS)",
          "ar_slotpos_digiteos": "AR − pos(t) + 11-way head", "nexttoken_ar": "query AR, next-token feedback",
          "ar_slotpos_digiteos_target": "AR − pos(t) + 11-way head + target coords",
          "ar_slotpos_digithead_target": "AR − pos(t) + 10-way head (36 digits) + target coords",
          "ar_digithead_target": "AR + 10-way head (36 digits) + target coords", "query_ar": "query AR", "causal_encoder": "causal encoder"}


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


# ---------------------------------------------------------------------------------------------------------------------
# Notebook 41: figure of the three-change recipe (read-only)
# ---------------------------------------------------------------------------------------------------------------------
RECIPE_MODELS = ["transformer_ar", "ar_slotpos_digithead", "ar_digithead_target", "ar_slotpos_digithead_target"]
RECIPE_LABELS = {"transformer_ar": "Original AR",
                 "ar_slotpos_digithead": "No pos(t) + 10-way head (35b)",
                 "ar_digithead_target": "pos(t) + 10-way head + target coords (35c)",
                 "ar_slotpos_digithead_target": "No pos(t) + 10-way head + target coords (AR34)"}
# Okabe-Ito (colour-blind safe); the original AR is the neutral grey baseline
RECIPE_COLORS = {"transformer_ar": "#7a7a7a", "ar_slotpos_digithead": "#E69F00", "ar_digithead_target": "#56B4E9",
                 "ar_slotpos_digithead_target": "#009E73"}
RECIPE_MARKERS = {"transformer_ar": "o", "ar_slotpos_digithead": "s", "ar_digithead_target": "^",
                  "ar_slotpos_digithead_target": "D"}


def plot_recipe(results_root, mode="full", seed=42, condition=CONDITION, models=None, show=True, log=print,
                review_name="AR_Recipe_Figures_Review.ipynb"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from .mirror_report import Report, write_review
    models = list(models or RECIPE_MODELS)
    S, ks = LK.CONDITIONS[condition]
    out = LX.family_root(results_root, mode, SUBDIR) / "report" / f"seed_{seed}_ar_recipe_figures"
    R = Report(show)
    R.md(f"# Transformer AR: which changes make it generalize? ({condition}, seed {seed}, LAST checkpoint)\n"
         "Trained k = " + ", ".join(map(str, ks)) + f" in {S} slots. Held out: singletons (k=1) at every slot, k=6, k=7.")
    rows, srows, status = [], [], []
    for m in models:
        cfg = LX.make_cfg(condition, results_root, mode, [m], seed, SUBDIR)
        d = LX.load_eval(cfg, m)
        if not d:
            status.append({"model": RECIPE_LABELS.get(m, m), "status": "not available"})
            continue
        status.append({"model": RECIPE_LABELS.get(m, m), "status": "ok"})
        k, ex = d["k"].astype(int), d["exact"].astype(bool)
        words = d["string"].astype(str)
        for kk in sorted(set(k.tolist())):
            rows.append({"model": m, "k": kk, "exact_acc": ex[k == kk].mean(), "records": int((k == kk).sum())})
        for op in "ABC":
            q = (k == 1) & (words == op)
            srows.append({"model": m, "op": op, "exact_acc": ex[q].mean()})
    R.table("Availability", pd.DataFrame(status), 10)
    if not rows:
        return None
    A, Sg = pd.DataFrame(rows), pd.DataFrame(srows)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    # ---- figure 1: accuracy vs test k --------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for kk in ks:
        ax.axvspan(kk - 0.5, kk + 0.5, color="#f0f0f0", zorder=0, lw=0)
    ax.axvline(max(ks) + 0.5, color="#b0b0b0", ls="--", lw=1)
    for m in models:
        q = A[A.model == m].sort_values("k")
        if not len(q):
            continue
        held = ~q.k.isin(ks)
        ax.plot(q.k, q.exact_acc, color=RECIPE_COLORS[m], lw=2, zorder=3, label=RECIPE_LABELS[m])
        ax.scatter(q.k[~held], q.exact_acc[~held], color=RECIPE_COLORS[m], marker=RECIPE_MARKERS[m], s=40, zorder=4,
                   edgecolors="white", linewidths=1.5)
        ax.scatter(q.k[held], q.exact_acc[held], facecolors="white", edgecolors=RECIPE_COLORS[m],
                   marker=RECIPE_MARKERS[m], s=55, lw=2, zorder=4)
    ax.set(xlabel="test instruction length k  (k = 1: singletons)", ylabel="exact-match accuracy", ylim=(-0.03, 1.05),
           xlim=(-0.5, S + 0.5), xticks=range(0, S + 1))
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.set_title(f"{condition}: accuracy by test length\n(grey bands = trained k; open markers = held out)", fontsize=10)
    fig.tight_layout()
    R.fig(fig, "1. Accuracy by test length", out / "figures" / "acc_vs_k.png")
    # ---- figure 2: held-out metrics, grouped bars --------------------------------------------------------------------
    metrics = [("singleton A", lambda m: Sg[(Sg.model == m) & (Sg.op == "A")].exact_acc),
               ("singleton B", lambda m: Sg[(Sg.model == m) & (Sg.op == "B")].exact_acc),
               ("singleton C", lambda m: Sg[(Sg.model == m) & (Sg.op == "C")].exact_acc)]
    metrics += [(f"k = {kk}", (lambda kk: lambda m: A[(A.model == m) & (A.k == kk)].exact_acc)(kk))
                for kk in range(max(ks) + 1, S + 1)]
    present = [m for m in models if (A.model == m).any()]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.8 / len(present)
    for i, m in enumerate(present):
        vals = [float(f(m).iloc[0]) if len(f(m)) else np.nan for _, f in metrics]
        xs = np.arange(len(metrics)) + (i - (len(present) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=RECIPE_COLORS[m], label=RECIPE_LABELS[m], edgecolor="white", lw=1)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax.set_xticks(range(len(metrics)), [n for n, _ in metrics])
    ax.set(ylabel="exact-match accuracy", ylim=(0, 1.12))
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title(f"{condition}: held-out accuracy", fontsize=10)
    fig.tight_layout()
    R.fig(fig, "2. Held-out accuracy: singletons by operation, longer instructions", out / "figures" / "heldout_bars.png")
    tab = A.pivot_table(index="model", columns="k", values="exact_acc").reindex(present)
    tab.index = [RECIPE_LABELS[m] for m in tab.index]
    for op in "ABC":
        tab[f"singleton {op}"] = [float(Sg[(Sg.model == m) & (Sg.op == op)].exact_acc.iloc[0]) for m in present]
    R.table("3. Exact accuracy (columns 0–7 = test k)", tab.round(4).reset_index().rename(columns={"index": "model"}), 10)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    A.to_csv(out / "tables" / "acc_by_k.csv", index=False)
    Sg.to_csv(out / "tables" / "singletons_by_op.csv", index=False)
    rp = write_review(R.items, out, "AR recipe figures", "# AR recipe figures — review snapshot\n\nBaked outputs.",
                      review_name)
    log(f"REVIEW FILE: {rp}")
    log(f"FIGURES: {out / 'figures'}")
    return rp


# ---------------------------------------------------------------------------------------------------------------------
# Notebook 45: Aligned AR on S7_k02345 over several seeds (read-only plot)
# ---------------------------------------------------------------------------------------------------------------------
SEED_LABELS = {"ar_slotpos_digithead_target": "Aligned AR", "transformer_ar": "Standard AR"}
SEED_COLORS = {"ar_slotpos_digithead_target": "#009E73", "transformer_ar": "#7a7a7a"}


def plot_seeds(results_root, mode="full", seeds=(42, 43, 44), condition=CONDITION, models=None, show=True, log=print,
               review_name="AlignedAR_S7_k02345_3seeds_Review.ipynb"):
    """Accuracy by test k for every seed (thin lines) and the mean (thick line), plus a per-seed table of held-out
    accuracy (singletons A/B/C, longer k). Models without saved results are listed as unavailable and skipped."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from .mirror_report import Report, write_review
    models = list(models or ["ar_slotpos_digithead_target", "transformer_ar"])
    seeds = list(seeds)
    S, ks = LK.CONDITIONS[condition]
    out = LX.family_root(results_root, mode, SUBDIR) / "report" / f"aligned_ar_{condition}_seeds"
    R = Report(show)
    R.md(f"# Aligned AR on {condition}: {len(seeds)} seeds {seeds} (LAST checkpoint, 20,000 updates)\n"
         "Seeds change the initialization and the training stream; the test records are identical for every seed.")
    rows, status = [], []
    for m in models:
        for sd in seeds:
            cfg = LX.make_cfg(condition, results_root, mode, [m], sd, SUBDIR)
            d = LX.load_eval(cfg, m)
            status.append({"model": SEED_LABELS.get(m, m), "seed": sd, "status": "ok" if d else "not available"})
            if not d:
                continue
            k, ex, w = d["k"].astype(int), d["exact"].astype(bool), d["string"].astype(str)
            for kk in sorted(set(k.tolist())):
                rows.append({"model": m, "seed": sd, "metric": f"k={kk}", "k": kk, "exact_acc": ex[k == kk].mean()})
            for op in "ABC":
                q = (k == 1) & (w == op)
                rows.append({"model": m, "seed": sd, "metric": f"single {op}", "k": np.nan, "exact_acc": ex[q].mean()})
    R.table("Availability", pd.DataFrame(status), 20)
    if not rows:
        return None
    D = pd.DataFrame(rows)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for kk in ks:
        ax.axvspan(kk - 0.5, kk + 0.5, color="#f0f0f0", lw=0, zorder=0)
    for m in models:
        q = D[(D.model == m) & D.k.notna()]
        if not len(q):
            continue
        c = SEED_COLORS.get(m, "#333333")
        for sd in seeds:
            qs = q[q.seed == sd].sort_values("k")
            ax.plot(qs.k, qs.exact_acc, color=c, lw=1, alpha=0.45, zorder=2)
        mean = q.groupby("k").exact_acc.agg(["mean", "min", "max", "count"]).reset_index()
        ax.fill_between(mean.k, mean["min"], mean["max"], color=c, alpha=0.15, lw=0, zorder=1)
        ax.plot(mean.k, mean["mean"], color=c, lw=2.5, marker="o", zorder=3,
                label=f"{SEED_LABELS.get(m, m)} (mean of {int(mean['count'].max())} seeds; thin = each seed)")
    ax.set(xlabel="number of operations k  (k = 1: single operation)", ylabel="exact-match accuracy",
           ylim=(-0.03, 1.05), xticks=range(0, S + 1))
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_title(f"{S} slots, trained on k = " + ", ".join(map(str, ks)) + " (grey); band = min–max over seeds",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    fig.tight_layout()
    R.fig(fig, "Accuracy by test length over seeds", out / "figures" / "acc_vs_k_seeds.png")
    held = [f"single {op}" for op in "ABC"] + [f"k={kk}" for kk in range(max(ks) + 1, S + 1)]
    T = D[D.metric.isin(held)].pivot_table(index=["model", "seed"], columns="metric", values="exact_acc")[held]
    T = T.reset_index()
    T["model"] = T.model.map(lambda m: SEED_LABELS.get(m, m))
    R.table("Held-out accuracy per seed", T.round(4), 20)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "tables" / "per_seed.csv", index=False)
    rp = write_review(R.items, out, "Aligned AR seeds", "# Aligned AR over seeds — review snapshot\n\nBaked outputs.",
                      review_name)
    log(f"REVIEW FILE: {rp}")
    return rp
