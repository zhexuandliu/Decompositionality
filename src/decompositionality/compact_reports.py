"""Compact-layout PE ablation (notebooks 07/08): combined tables and the paired RoPE-vs-sinusoidal comparison.

The seven attention models exist once per PE condition; the two CNN controls are ONE shared run each and are
listed once, marked "n/a for PE contrast". Paired comparisons align identical (grid_id, word) test examples
and verify identical inputs/targets. One seed: no error bars, no seed-robust ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import mixed as M
from .names import DISPLAY_NAMES
from .plotstyle import _plt, save_fig

PE_DIRS = {"rope": "rope", "sinusoidal": "sinusoidal", "learned": "learned"}


def combined_table(root: Path):
    """One row per UNIQUE run: 7 attention models x available PE conditions + 2 shared CNN controls (once)."""
    import pandas as pd
    root = Path(root)
    rows = []
    for exp, sub in M.COMPACT_DIRS.items():
        d = root / sub
        for m in (M.CNN_MODELS if exp == "compact_02345_cnn" else M.ATTENTION_MODELS):
            f = d / "eval" / m / "test_groups.csv"
            if not f.exists():
                rows.append({"run": f"{sub}/{m}", "model": DISPLAY_NAMES[m], "condition": sub, "status": "missing"})
                continue
            g = pd.read_csv(f, keep_default_na=False, na_values=[""]).set_index("group")
            ev = json.loads((d / "eval" / m / "eval_summary.json").read_text())
            r = {"run": f"{sub}/{m}", "model": DISPLAY_NAMES[m],
                 "condition": sub if sub != "shared_cnn" else "shared CNN control (n/a for PE contrast)",
                 "status": "complete", "selected_step": ev["selected_step"]}
            for k, lab in (("single_A", "A"), ("single_B", "B"), ("single_C", "C"), ("EMPTY", "EMPTY"),
                           ("pairs_seen", "len2"), ("len3_seen", "len3"), ("len4_seen", "len4"), ("len5_seen", "len5")):
                r[lab] = f"{int(g.loc[k, 'exact_correct'])}/{int(g.loc[k, 'n'])}"
            r["singles_macro_exact"] = ev["test"]["singles_macro_exact"]
            rows.append(r)
    df = pd.DataFrame(rows)
    out = root / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_unique_runs_summary.csv", index=False)
    return df


def paired_pe_comparison(root: Path, log=print, a_pe: str = "rope", b_pe: str = "sinusoidal"):
    """``a_pe`` minus ``b_pe`` (default: rope minus sinusoidal) on IDENTICAL test examples, per model and
    word/group, with agreement counts. Returns None (and says why) if either condition is incomplete."""
    import pandas as pd
    root = Path(root)
    out = root / "comparisons"
    rows = []
    for m in M.ATTENTION_MODELS:
        fr, fs = (root / PE_DIRS[pe] / "eval" / m / "test_predictions.npz" for pe in (a_pe, b_pe))
        if not (fr.exists() and fs.exists()):
            continue
        with np.load(fr) as a, np.load(fs) as b:
            ka = pd.DataFrame({"g": a["grid_id"], "s": a["string"].astype(str), "ia": np.arange(len(a["grid_id"]))})
            kb = pd.DataFrame({"g": b["grid_id"], "s": b["string"].astype(str), "ib": np.arange(len(b["grid_id"]))})
            j = ka.merge(kb, on=["g", "s"], how="inner", validate="one_to_one")
            if len(j) != len(ka) or len(j) != len(kb):
                raise RuntimeError(f"{m}: {a_pe} and {b_pe} example sets differ")
            ia, ib = j.ia.values, j.ib.values
            if not (np.array_equal(a["input"][ia], b["input"][ib]) and np.array_equal(a["target"][ia], b["target"][ib])):
                raise RuntimeError(f"{m}: paired examples have different inputs/targets")
            ea, eb = a["exact"][ia], b["exact"][ib]
            ca, cb = a["cell_correct"][ia] / 36, b["cell_correct"][ib] / 36
            same = (a["pred"][ia] == b["pred"][ib]).reshape(len(ia), -1).all(1)
        j["ea"], j["eb"], j["ca"], j["cb"], j["same"] = ea, eb, ca, cb, same
        j["group"] = j.s.map(lambda w: M.mixed_group(w, "compact_02345_rope"))
        for key, sub in list(j.groupby("s")) + [(f"[group] {g}", d) for g, d in j.groupby("group")]:
            rows.append({"model": m, "word_or_group": key, "n": len(sub),
                         f"{a_pe}_exact": int(sub.ea.sum()), f"{b_pe}_exact": int(sub.eb.sum()),
                         f"{a_pe}_exact_acc": sub.ea.mean(), f"{b_pe}_exact_acc": sub.eb.mean(),
                         f"delta_exact ({a_pe} - {b_pe})": sub.ea.mean() - sub.eb.mean(),
                         f"delta_cell ({a_pe} - {b_pe})": sub.ca.mean() - sub.cb.mean(),
                         "both_correct": int((sub.ea & sub.eb).sum()), f"only_{a_pe}_correct": int((sub.ea & ~sub.eb).sum()),
                         f"only_{b_pe}_correct": int((~sub.ea & sub.eb).sum()), "both_wrong": int((~sub.ea & ~sub.eb).sum()),
                         "identical_prediction_rate": sub.same.mean()})
    if not rows:
        log(f"Paired {a_pe}-vs-{b_pe} comparison unavailable: both conditions must have completed evaluations "
            "(run the other notebook; this notebook does not require it).")
        return None
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"paired_{a_pe}_minus_{b_pe}"
    df.to_csv(out / f"{tag}.csv", index=False)
    plt = _plt()
    models = [m for m in M.ATTENTION_MODELS if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.5))
    w = 0.38
    keys = ["A", "B", "C"]
    for ax in (axes[0],):
        for i, m in enumerate(models):
            s = df[df.model == m].set_index("word_or_group")
            for k, key in enumerate(keys):
                x = i * 4 + k
                ax.bar(x - w / 2, s.loc[key, f"{a_pe}_exact_acc"], width=w, color="#2a78d6", label=a_pe if i == k == 0 else None)
                ax.bar(x + w / 2, s.loc[key, f"{b_pe}_exact_acc"], width=w, color="#eb6834",
                       label=b_pe if i == k == 0 else None)
                ax.text(x, -0.07, key, ha="center", fontsize=7)
        ax.set_xticks([i * 4 + 1 for i in range(len(models))])
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], fontsize=7, rotation=20)
        ax.tick_params(axis="x", pad=14)
        ax.set(ylim=(0, 1.05), ylabel="exact accuracy (1000 test grids each)",
               title=f"Singletons A/B/C (never trained): {a_pe} vs {b_pe}, identical examples (seed 42, best ckpt)")
        ax.legend(fontsize=8)
    ax = axes[1]
    groups = [g for g in ("EMPTY", "singles", "pairs_seen", "len3_seen", "len4_seen", "len5_seen")]
    mat = np.full((len(models), len(groups)), np.nan)
    for i, m in enumerate(models):
        s = df[df.model == m].set_index("word_or_group")
        for jx, g in enumerate(groups):
            if f"[group] {g}" in s.index:
                mat[i, jx] = s.loc[f"[group] {g}", f"delta_exact ({a_pe} - {b_pe})"]
    ax.grid(False)
    ax.imshow(np.nan_to_num(mat), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for jx in range(mat.shape[1]):
            ax.text(jx, i, "n/a" if np.isnan(mat[i, jx]) else f"{mat[i, jx]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(["EMPTY (len 0)", "singles (len 1, HELD OUT)", "len 2", "len 3",
                                                          "len 4", "len 5"], fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([DISPLAY_NAMES[m] for m in models], fontsize=7)
    ax.set_title(f"Exact-accuracy difference {a_pe} - {b_pe} by effective length (same examples)")
    fig.tight_layout()
    save_fig(fig, out, tag, df)
    log(f"Paired comparison written to {out}")
    return df
