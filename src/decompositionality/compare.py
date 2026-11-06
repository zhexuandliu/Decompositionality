"""Cross-experiment comparison: all_pairs (experiment 2) minus distinct_pairs (experiment 1).

Joins both experiments' per-example test records on (model, string, grid_id), checks that
the joined examples are identical (same manifest hash, same inputs and targets), and
reports exact-grid accuracy changes per string and per group.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import operations as O
from .evaluation import load_predictions
from .models.registry import DISPLAY_NAMES, MODEL_IDS
from .reports import plot_string_heatmap, save_fig, _plt, bar_style

EXP1, EXP2 = "distinct_pairs", "all_pairs"

COMPARE_GROUPS = {
    "singles": lambda s: len(s) == 1,
    "pairs_distinct (seen in both)": lambda s: len(s) == 2 and s[0] != s[1],
    "pairs_repeated AA/BB/CC (unseen in exp1, seen in exp2)": lambda s: len(s) == 2 and s[0] == s[1],
    "triples_all_distinct": lambda s: len(s) == 3 and len(set(s)) == 3,
    "triples_two_types": lambda s: len(s) == 3 and len(set(s)) == 2,
    "triples_one_type": lambda s: len(s) == 3 and len(set(s)) == 1,
}


def missing_experiments(results_root: Path):
    return [e for e in (EXP1, EXP2) if not (Path(results_root) / e / "eval").exists()]


def join_experiments(results_root: Path):
    import pandas as pd
    root = Path(results_root)
    h1 = json.loads((root / EXP1 / "manifests" / "manifest.json").read_text())["combined_hash"]
    h2 = json.loads((root / EXP2 / "manifests" / "manifest.json").read_text())["combined_hash"]
    if h1 != h2:
        raise RuntimeError("the two experiments used different evaluation manifests; cannot join")
    frames = []
    for m in MODEL_IDS:
        f1, f2 = (root / e / "eval" / m / "test_predictions.npz" for e in (EXP1, EXP2))
        if not (f1.exists() and f2.exists()):
            continue
        a, b = load_predictions(f1), load_predictions(f2)
        da = pd.DataFrame({"grid_id": a["grid_id"], "string": a["string"].astype(str), "exact_1": a["exact"],
                           "cell_1": a["cell_correct"] / 36, "_k1": np.arange(len(a["exact"]))})
        db = pd.DataFrame({"grid_id": b["grid_id"], "string": b["string"].astype(str), "exact_2": b["exact"],
                           "cell_2": b["cell_correct"] / 36, "_k2": np.arange(len(b["exact"]))})
        j = da.merge(db, on=["grid_id", "string"], how="inner", validate="one_to_one")
        if len(j) != len(da) or len(j) != len(db):
            raise RuntimeError(f"{m}: example sets differ between experiments")
        if not (np.array_equal(a["input"][j["_k1"]], b["input"][j["_k2"]])
                and np.array_equal(a["target"][j["_k1"]], b["target"][j["_k2"]])):
            raise RuntimeError(f"{m}: joined examples have different inputs/targets")
        j.insert(0, "model", m)
        frames.append(j.drop(columns=["_k1", "_k2"]))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compare(results_root: Path, out_dir: Optional[Path] = None, log=print):
    import pandas as pd
    root = Path(results_root)
    miss = missing_experiments(root)
    if miss:
        log(f"Cross-experiment comparison skipped: missing result folder(s) {[str(root / m) for m in miss]}. "
            "Run the other notebook (with the same results root) to enable it.")
        return None
    joined = join_experiments(root)
    if joined is None:
        log("Cross-experiment comparison skipped: no model has completed evaluations in BOTH experiments.")
        return None
    out = Path(out_dir) if out_dir else root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    ps = joined.groupby(["model", "string"]).agg(exact_acc_exp1=("exact_1", "mean"), exact_acc_exp2=("exact_2", "mean"),
                                                 cell_acc_exp1=("cell_1", "mean"), cell_acc_exp2=("cell_2", "mean"),
                                                 n=("exact_1", "size")).reset_index()
    ps["delta_exact (exp2-exp1)"] = ps["exact_acc_exp2"] - ps["exact_acc_exp1"]
    ps["delta_cell (exp2-exp1)"] = ps["cell_acc_exp2"] - ps["cell_acc_exp1"]
    ps.to_csv(out / "per_string_delta.csv", index=False)
    rows = []
    for m in ps["model"].unique():
        sub = ps[ps["model"] == m]
        for g, f in COMPARE_GROUPS.items():
            s = sub[sub["string"].map(f)]
            rows.append({"model": m, "group": g, "n_strings": len(s),
                         "exact_acc_exp1": s["exact_acc_exp1"].mean(), "exact_acc_exp2": s["exact_acc_exp2"].mean(),
                         "delta_exact (exp2-exp1)": s["delta_exact (exp2-exp1)"].mean()})
    gr = pd.DataFrame(rows)
    gr.to_csv(out / "group_delta.csv", index=False)

    plt = _plt()
    models = [m for m in MODEL_IDS if m in set(gr["model"])]
    G = list(COMPARE_GROUPS)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = gr[gr["model"] == m].set_index("group")
        xs = np.arange(len(G)) + (i - (len(models) - 1) / 2) * w
        ax.bar(xs, [sub.loc[g, "delta_exact (exp2-exp1)"] for g in G], width=w * 0.92, label=DISPLAY_NAMES[m], **bar_style(m))
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.set_xticks(range(len(G)))
    ax.set_xticklabels([g.replace(" (", "\n(") for g in G], fontsize=8)
    ax.set_ylabel("Change in exact-grid accuracy\n(all_pairs minus distinct_pairs)")
    ax.set_title("Effect of adding AA/BB/CC to training (same budget, same test examples; single seed)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, out, "compare_group_delta", gr)
    hm = ps.rename(columns={"delta_exact (exp2-exp1)": "delta"})
    plot_string_heatmap(hm, out, [], value="delta", name="compare_per_string_delta_heatmap", cmap="RdBu",
                        vmin=-1, vmax=1, title="Per-string change in exact-grid accuracy: all_pairs minus distinct_pairs")
    log(f"Comparison written to {out}")
    return {"per_string": ps, "groups": gr, "out_dir": out}
