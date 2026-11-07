"""Read-only comparison of four training supports in the five-slot identity format (Transformer AR / Looped AR):

  train02    = notebook 12  (k in {0,2})
  train023   = notebook 13  (k in {0,2,3})
  train0234  = notebook 14  (k in {0,2,3,4})
  train02345 = notebook 10  (identity_02345, k in {0,2,3,4,5})

No training. Predictions are located through each family's artifact index. Compatibility is CHECKED, not assumed:
same test manifest hash, identical test records (word, grid, placement, grid hash) and targets, same model config
hash. Caveat reported in the output: notebook 10 used a different sampling scheme (20% per k from a different
indexed stream, all nine models trained there), so the 02345 run is a non-identical training stream; all four
supports share the model configuration and the test records. One seed: descriptive only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import idext as E
from . import idvar as V
from .names import DISPLAY_NAMES
from .plotstyle import _plt, save_fig
from .words import words_of_length

SUPPORTS = {  # name -> (family, experiment id, train ks)
    "train02": ("idext", "idext_train02", (0, 2)),
    "train023": ("idext", "idext_train023", (0, 2, 3)),
    "train0234": ("idext", "idext_train0234", (0, 2, 3, 4)),
    "train02345": ("idvar", "idvar_identity", (0, 2, 3, 4, 5)),
}
MODELS = ["transformer_ar", "looped_transformer_ar"]
COLORS = {"train02": "#2a78d6", "train023": "#eb6834", "train0234": "#1baf7a", "train02345": "#6250d6"}


def _training_words(name):
    return [w for k in SUPPORTS[name][2] for w in words_of_length(k)]


def locate(results_root: Path, seed: int = 42) -> Dict[tuple, dict]:
    """(support, model) -> artifact record (with paths) for every COMPLETE run found; never guesses paths."""
    from .idext_reports import build_index
    from .idvar_pipeline import build_artifact_index
    root = Path(results_root)
    ext_root, var_root = root / E.IDEXT_RESULTS_SUBDIR, root / V.IDVAR_RESULTS_SUBDIR
    found = {}
    ext_idx = build_index(ext_root) if ext_root.exists() else {}
    var_idx = build_artifact_index(var_root) if var_root.exists() else {}
    for name, (fam, exp, _) in SUPPORTS.items():
        for m in MODELS:
            if fam == "idext":
                a = ext_idx.get(f"{exp}/seed_{seed}/{m}")
                run_cfg = ext_root / E.IDEXT_DIRS[exp] / f"seed_{seed}" / "runs" / m / "run_config.json"
                man = ext_root / "manifests" / "manifest.json"
            else:
                a = var_idx.get(f"identity/{m}")
                run_cfg = var_root / "identity" / "runs" / m / "run_config.json"
                man = var_root / "manifests" / "manifest.json"
            if a and a.get("complete"):
                found[(name, m)] = {**a, "run_config": str(run_cfg), "manifest_json": str(man)}
    return found


def load(rec: dict) -> Dict[str, np.ndarray]:
    with np.load(rec["predictions"]) as z:
        return {k: z[k] for k in z.files}


def compatibility(found: dict, data: dict):
    import pandas as pd
    rows = []
    ref_key = next(iter(data)) if data else None
    ref = data[ref_key] if ref_key else None
    if ref is not None:
        o_ref = np.lexsort((ref["position"], ref["grid_id"], ref["string"]))
    for key, a in data.items():
        rec = found[key]
        man = json.loads(Path(rec["manifest_json"]).read_text()).get("combined_hash") if Path(rec["manifest_json"]).exists() else None
        rc = json.loads(Path(rec["run_config"]).read_text()) if Path(rec["run_config"]).exists() else {}
        o = np.lexsort((a["position"], a["grid_id"], a["string"]))
        same_records = (len(a["string"]) == len(ref["string"]) and all(
            np.array_equal(a[f][o], ref[f][o_ref]) for f in ("string", "grid_id", "position", "grid_hash", "target")))
        rows.append({"support": key[0], "model": key[1], "n_records": len(a["string"]), "manifest_hash": (man or "?")[:12],
                     "model_hash": rc.get("model_hash"), "selected_step": rec.get("selected_step"),
                     "test_records_identical_to_reference": bool(same_records),
                     "training_stream": "per-k indexed (idext)" if SUPPORTS[key[0]][0] == "idext"
                     else "notebook-10 stream (20% per k; NOT the per-k indexed stream)"})
    df = pd.DataFrame(rows)
    ok = bool(len(df) and df.test_records_identical_to_reference.all() and df.manifest_hash.nunique() == 1
              and all(df[df.model == m].model_hash.nunique() == 1 for m in df.model.unique()))
    return df, ok


def coverage():
    import pandas as pd
    from .diagnosis import group as G
    cat = G.Catalog()
    rows = []
    for name in SUPPORTS:
        S = {cat.word_id(w) for w in _training_words(name)}
        for k in range(6):
            for w in words_of_length(k):
                rows.append({"support": name, "k": k, "string": w or "EMPTY", "perm_id": cat.word_id(w),
                             "transform_in_training": cat.word_id(w) in S, "n_training_transforms": len(S)})
    return pd.DataFrame(rows)


def tables(data: dict, cov):
    import pandas as pd
    by_k, singles, split = [], [], []
    for (name, m), a in data.items():
        tks = SUPPORTS[name][2]
        n = len(a["pred"])
        P, Y = a["pred"].reshape(n, 36), a["target"].reshape(n, 36)
        df = pd.DataFrame({"k": a["k"], "string": a["string"].astype(str), "position": a["position"], "kind": a["kind"],
                           "exact": a["exact"].astype(bool), "cell": (P == Y).mean(1)})
        c = cov[cov.support == name].set_index("string")
        df["present"] = c.loc[df.string, "transform_in_training"].values
        g = df.groupby("k").agg(exact_correct=("exact", "sum"), n=("exact", "size"), exact_acc=("exact", "mean"),
                                cell_acc=("cell", "mean")).reset_index()
        g["status"] = np.where(g.k.isin(tks), "seen", "HELD OUT")
        g.insert(0, "model", m); g.insert(0, "support", name)
        by_k.append(g)
        s = df[df.kind == "single"].groupby(["string", "position"]).agg(exact_correct=("exact", "sum"), n=("exact", "size"),
                                                                       cell_acc=("cell", "mean")).reset_index()
        s.insert(0, "model", m); s.insert(0, "support", name)
        singles.append(s)
        for k in range(6):
            for pres in (True, False):
                x = df[(df.k == k) & (df.present == pres)]
                split.append({"support": name, "model": m, "k": k, "status": "seen" if k in tks else "HELD OUT",
                              "transform_in_training": pres, "exact_correct": int(x.exact.sum()), "n": len(x),
                              "exact_acc": x.exact.mean() if len(x) else np.nan})
    return pd.concat(by_k, ignore_index=True), pd.concat(singles, ignore_index=True), pd.DataFrame(split)


def transitions(data: dict):
    """Consecutive supports, same model, identical test records: correctness transitions per k."""
    import pandas as pd
    rows = []
    names = list(SUPPORTS)
    for (a_name, b_name), m in itertools.product(zip(names, names[1:]), MODELS):
        if (a_name, m) not in data or (b_name, m) not in data:
            rows.append({"pair": f"{a_name} -> {b_name}", "model": m, "k": "all", "status": "not yet available"})
            continue
        a, b = data[(a_name, m)], data[(b_name, m)]
        ka = pd.DataFrame({"s": a["string"], "g": a["grid_id"], "p": a["position"], "h": a["grid_hash"], "ea": a["exact"], "k": a["k"]})
        kb = pd.DataFrame({"s": b["string"], "g": b["grid_id"], "p": b["position"], "h": b["grid_hash"], "eb": b["exact"]})
        j = ka.merge(kb, on=["s", "g", "p", "h"], validate="one_to_one")
        for k, x in j.groupby("k"):
            rows.append({"pair": f"{a_name} -> {b_name}", "model": m, "k": int(k), "n": len(x),
                         "before_exact": int(x.ea.sum()), "after_exact": int(x.eb.sum()),
                         "both_correct": int((x.ea & x.eb).sum()), "gained": int((~x.ea & x.eb).sum()),
                         "lost": int((x.ea & ~x.eb).sum()), "both_wrong": int((~x.ea & ~x.eb).sum())})
    return pd.DataFrame(rows)


def figures(by_k, singles, split, out: Path):
    plt = _plt()
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), squeeze=False)
    for ax, m in zip(axes[0], MODELS):
        for name in SUPPORTS:
            s = by_k[(by_k.model == m) & (by_k.support == name)].sort_values("k")
            if not len(s):
                continue
            col = COLORS[name]
            ax.plot(s.k, s.exact_acc, color=col, lw=1.6, label=f"{name} (train k={list(SUPPORTS[name][2])})")
            seen = s.status == "seen"
            ax.scatter(s.k[seen], s.exact_acc[seen], color=col, s=50, zorder=3)
            ax.scatter(s.k[~seen], s.exact_acc[~seen], facecolors="white", edgecolors=col, s=50, lw=1.6, zorder=3)
        ax.set(xticks=range(6), ylim=(-0.02, 1.05), xlabel="k (A/B/C tokens; always 5 visible op tokens)",
               ylabel="exact-grid accuracy", title=f"{DISPLAY_NAMES[m]}: filled = trained k, open = held out (seed 42)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths += save_fig(fig, out, "support_comparison_exact_by_k", by_k)
    runs = [(n, m) for m in MODELS for n in SUPPORTS if len(singles[(singles.support == n) & (singles.model == m)])]
    if runs:
        fig, axes = plt.subplots(len(MODELS), len(SUPPORTS), figsize=(4.4 * len(SUPPORTS), 3.2 * len(MODELS)), squeeze=False)
        for i, m in enumerate(MODELS):
            for j, n in enumerate(SUPPORTS):
                ax = axes[i, j]
                r = singles[(singles.support == n) & (singles.model == m)]
                if not len(r):
                    ax.axis("off"); ax.text(0.5, 0.5, "not yet available", ha="center", transform=ax.transAxes); continue
                mat = np.array([[r[(r.string == op) & (r.position == p)].exact_correct.iloc[0] for p in range(1, 6)] for op in "ABC"])
                ax.grid(False)
                ax.imshow(mat / 1000, cmap="Blues", vmin=0, vmax=1)
                for a_ in range(3):
                    for b_ in range(5):
                        ax.text(b_, a_, str(mat[a_, b_]), ha="center", va="center", fontsize=7,
                                color="white" if mat[a_, b_] > 600 else "#0b0b0b")
                ax.set_xticks(range(5)); ax.set_xticklabels([f"s{p}" for p in range(1, 6)], fontsize=7)
                ax.set_yticks(range(3)); ax.set_yticklabels(list("ABC"))
                ax.set_title(f"{DISPLAY_NAMES[m]} / {n}", fontsize=8)
        fig.suptitle("Singletons (held out everywhere): exact successes of 1000 per operation x placement")
        fig.tight_layout()
        paths += save_fig(fig, out, "support_comparison_singletons", singles)
    k5 = split[(split.k == 5) & (split.n > 0)]
    if len(k5):
        fig, ax = plt.subplots(figsize=(12, 4))
        labels, vals = [], []
        for _, r in k5.iterrows():
            labels.append(f"{r.support}\n{DISPLAY_NAMES[r.model][:14]}\n{'transform seen' if r.transform_in_training else 'NEW transform'}")
            vals.append(r.exact_acc)
        ax.bar(range(len(vals)), vals, color=["#2a78d6" if t else "#eb6834" for t in k5.transform_in_training])
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6)
        ax.set(ylabel="exact accuracy at k=5", title="k=5 split by whether the word's transformation occurs in training")
        fig.tight_layout()
        paths += save_fig(fig, out, "support_comparison_k5_transform_split", k5)
    return paths


def run(results_root: str, seed: int = 42, log=print) -> dict:
    root = Path(results_root)
    out = root / "support_comparison" / f"seed_{seed}"
    found = locate(root, seed)
    missing = [f"{n}/{m}" for n in SUPPORTS for m in MODELS if (n, m) not in found]
    if missing:
        log("not yet available: " + ", ".join(missing))
    data = {k: load(v) for k, v in found.items()}
    if not data:
        log("No runs found; nothing to compare.")
        return {"found": found, "missing": missing}
    compat, ok = compatibility(found, data)
    log("compatibility: " + ("OK (same manifest, identical test records and targets, same model config per model)" if ok
                             else "PROBLEM - see compatibility table; comparisons may be invalid"))
    cov = coverage()
    by_k, singles, split = tables(data, cov)
    tr = transitions(data)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("compatibility", compat), ("by_k", by_k), ("singletons", singles), ("transform_split", split),
                     ("transitions", tr)):
        df.to_csv(out / f"{name}.csv", index=False)
    cov.groupby(["support", "k"]).agg(seen_transform_words=("transform_in_training", "sum"),
                                      words=("string", "size")).reset_index().to_csv(out / "coverage_summary.csv", index=False)
    figs = figures(by_k, singles, split, out / "figures")
    log(f"written to {out}")
    return {"compatibility": compat, "compatible": ok, "by_k": by_k, "singletons": singles, "transform_split": split,
            "transitions": tr, "missing": missing, "figures": figs, "out": out}
