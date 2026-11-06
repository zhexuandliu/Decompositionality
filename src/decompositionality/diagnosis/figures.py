"""Figures for the generalization-failure diagnostics (both experiments; torch-free).
Every figure is saved as PNG + PDF together with a CSV of the plotted numbers."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List

import numpy as np

from .. import operations as O
from ..names import DISPLAY_NAMES, MODEL_IDS
from ..plotstyle import MODEL_STYLE, _plt, draw_grid, save_fig
from . import group as G

EXPS = ["distinct_pairs", "all_pairs"]
EXP_LABEL = {"distinct_pairs": "Exp. 1 distinct_pairs", "all_pairs": "Exp. 2 all_pairs"}
OOD_GROUPS = ["singles", "pairs_unseen", "triples_all_distinct", "triples_two_types", "triples_one_type"]
GROUP_SHORT = {"singles": "singles", "pairs_seen": "seen pairs", "pairs_unseen": "unseen pairs",
               "triples_all_distinct": "triples 3-distinct", "triples_two_types": "triples 2-types",
               "triples_one_type": "triples 1-type"}


def _models(df) -> List[str]:
    present = set(df["model"]) if len(df) else set()
    return [m for m in MODEL_IDS if m in present] + sorted(present - set(MODEL_IDS))


def _name(m):
    return DISPLAY_NAMES.get(m, m)


def _string_heatmap(ax, mat, models, exp, title, cmap="Blues", vmin=0, vmax=1, text=None, fmt="{:.2f}"):
    ax.grid(False)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    train = set(O.TRAIN_SETS.get(exp, []))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            t = text[i][j] if text is not None else (fmt.format(v) if not np.isnan(v) else "")
            if np.isnan(v) and text is None:
                t = "n/a"
            frac = 0 if np.isnan(v) else (v - vmin) / (vmax - vmin)
            ax.text(j, i, t, ha="center", va="center", fontsize=5.5,
                    color="white" if (cmap in ("Blues", "Purples", "Greens") and frac > 0.6) else "#0b0b0b")
    ax.set_xticks(range(len(O.TEST_STRINGS)))
    ax.set_xticklabels([s + ("*" if s in train else "") for s in O.TEST_STRINGS], fontsize=7, family="monospace")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_name(m) for m in models], fontsize=8)
    start = 0
    for g in O.GROUP_ORDER:
        n = O.EXPECTED_GROUP_COUNTS[g]
        if start:
            ax.axvline(start - 0.5, color="#0b0b0b", lw=1.4)
        ax.text(start + n / 2 - 0.5, -0.75, O.GROUP_LABELS[g], ha="center", va="bottom", fontsize=8, weight="bold")
        start += n
    ax.set_title(title + "   (* = string seen in this experiment's training set)", pad=24, fontsize=10)
    return im


def _per_string_matrix(ps, exp, col):
    sub = ps[ps.experiment == exp]
    models = _models(sub)
    mat = np.full((len(models), 39), np.nan)
    for i, m in enumerate(models):
        s = sub[sub.model == m].set_index("string")
        for j, w in enumerate(O.TEST_STRINGS):
            if w in s.index and col in s:
                mat[i, j] = s.loc[w, col]
    return models, mat


def heatmap_pair(R, col, name, title, out, cmap="Blues", vmin=0, vmax=1):
    import pandas as pd
    plt = _plt()
    ps = R["per_string"]
    exps = [e for e in EXPS if e in set(ps.experiment)]
    if not exps:
        return None
    heights = [1.2 + 0.42 * len(_models(ps[ps.experiment == e])) for e in exps]
    fig, axes = plt.subplots(len(exps), 1, figsize=(24, sum(heights) + 1), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    data = []
    for ax, e in zip(axes[:, 0], exps):
        models, mat = _per_string_matrix(ps, e, col)
        im = _string_heatmap(ax, mat, models, e, f"{EXP_LABEL[e]}: {title}", cmap, vmin, vmax)
        df = pd.DataFrame(mat, columns=O.TEST_STRINGS)
        df.insert(0, "model", models); df.insert(0, "experiment", e)
        data.append(df)
    fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.012, pad=0.01)
    return save_fig(fig, out, name, pd.concat(data))


def fig_categories(R, out):
    import pandas as pd
    plt = _plt()
    ps = R["per_string"]
    cats = [("exact_acc", "correct", "#2a78d6"), ("wrong_unique_legal_rate", "wrong: unique legal transform", "#eb6834"),
            ("wrong_ambiguous_legal_rate", "wrong: ambiguous legal (>1 transform)", "#eda100"),
            ("no_legal_match_rate", "wrong: no legal transform matches", "#8c8c8c")]
    groups = ["singles", "pairs_seen", "pairs_unseen", "triples_all_distinct", "triples_two_types", "triples_one_type"]
    exps = [e for e in EXPS if e in set(ps.experiment)]
    fig, axes = plt.subplots(len(exps), len(groups), figsize=(22, 3.6 * len(exps)), squeeze=False, sharey="row")
    rows = []
    for r, e in enumerate(exps):
        sub = ps[ps.experiment == e]
        models = _models(sub)
        for c, g in enumerate(groups):
            ax = axes[r, c]
            sg = sub[sub.group == g]
            left = np.zeros(len(models))
            if not len(sg):
                ax.text(0.5, 0.5, "N/A (no strings in\nthis category)", ha="center", va="center", transform=ax.transAxes)
            for col, lab, colr in cats:
                vals = np.array([np.average(sg[sg.model == m][col], weights=sg[sg.model == m]["n"]) if len(sg[sg.model == m]) else 0
                                 for m in models])
                ax.barh(range(len(models)), vals, left=left, color=colr, label=lab, edgecolor="white", linewidth=0.6)
                left += vals
                for m, v in zip(models, vals):
                    rows.append({"experiment": e, "group": g, "model": m, "category": lab, "fraction": v})
            ax.set_xlim(0, 1)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([_name(m) for m in models], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{EXP_LABEL[e]}\n{GROUP_SHORT[g]}", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.99, wspace=0.12, hspace=0.45, top=0.86)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.25), ncol=4, fontsize=8, frameon=False)
    fig.suptitle("What do predictions match? Exact match against all 288 legal grid transformations "
                 "(post-hoc; categories exclusive per example)", y=0.995)
    return save_fig(fig, out, "03_explanation_categories", pd.DataFrame(rows))


def fig_dominant(R, out):
    import pandas as pd
    plt = _plt()
    ps = R["per_string"]
    exps = [e for e in EXPS if e in set(ps.experiment)]
    heights = [1.2 + 0.42 * len(_models(ps[ps.experiment == e])) for e in exps]
    fig, axes = plt.subplots(len(exps), 1, figsize=(24, sum(heights) + 1), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    data = []
    for ax, e in zip(axes[:, 0], exps):
        sub = ps[ps.experiment == e]
        models = _models(sub)
        mat = np.full((len(models), 39), np.nan)
        txt = [["" for _ in range(39)] for _ in models]
        for i, m in enumerate(models):
            s = sub[sub.model == m].set_index("string")
            for j, w in enumerate(O.TEST_STRINGS):
                if w in s.index:
                    mat[i, j] = s.loc[w, "dominant_wrong_unique_share"]
                    wd = s.loc[w, "dominant_wrong_unique_word"]
                    txt[i][j] = (wd if len(wd) <= 5 else wd[:4] + "…") if isinstance(wd, str) and wd else "–"
        _string_heatmap(ax, mat, models, e, f"{EXP_LABEL[e]}: share of examples whose ONLY exact legal match is the "
                                            "dominant wrong transformation (text = its canonical word)", "Oranges", 0, 1, txt)
        d = sub[["experiment", "model", "string", "dominant_wrong_unique_word", "dominant_wrong_unique_share",
                 "dominant_is_training_transform"]]
        data.append(d)
    return save_fig(fig, out, "04_dominant_wrong_transformation", pd.concat(data))


def fig_hypotheses(R, out):
    import pandas as pd
    plt = _plt()
    ht = R["hypothesis_types"]
    if not len(ht):
        return None
    ht = ht[ht.group != "pairs_seen"]
    exps = [e for e in EXPS if e in set(ht.experiment)]
    types = [t for t in G.HYPOTHESIS_TYPES if t in set(ht.hypothesis_type)]
    fig, axes = plt.subplots(1, len(exps), figsize=(9 * len(exps), 6), squeeze=False)
    data = []
    for ax, e in zip(axes[0], exps):
        sub = ht[ht.experiment == e]
        models = _models(sub)
        mat = np.full((len(types), len(models)), np.nan)
        for i, t in enumerate(types):
            for j, m in enumerate(models):
                s = sub[(sub.hypothesis_type == t) & (sub.model == m)]
                if len(s):
                    mat[i, j] = np.average(s["compatible_rate_any"], weights=s["n"])
        ax.grid(False)
        ax.imshow(mat, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        for i in range(len(types)):
            for j in range(len(models)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > 0.6 else "#0b0b0b")
        ax.set_xticks(range(len(models))); ax.set_xticklabels([_name(m) for m in models], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(types))); ax.set_yticklabels(types, fontsize=8)
        ax.set_title(f"{EXP_LABEL[e]}: fraction of UNSEEN-string examples exactly compatible\nwith any non-truth hypothesis "
                     "of each type (types overlap; rows need not sum to 1)", fontsize=9)
        df = pd.DataFrame(mat, index=types, columns=models).reset_index().rename(columns={"index": "hypothesis_type"})
        df.insert(0, "experiment", e)
        data.append(df)
    fig.tight_layout()
    return save_fig(fig, out, "05_hypothesis_compatibility", pd.concat(data))


def fig_nearest_vs_controls(R, out):
    import pandas as pd
    plt = _plt()
    exps = [e for e in EXPS if any(k.startswith(e + "/") for k in R["per_run"])]
    if not exps:
        return None
    fig, axes = plt.subplots(1, len(exps), figsize=(9 * len(exps), 4.8), squeeze=False, sharey=True)
    data = []
    xs = np.arange(37)
    for ax, e in zip(axes[0], exps):
        for k, res in R["per_run"].items():
            if not k.startswith(e + "/"):
                continue
            m = k.split("/", 1)[1]
            pe = res["per_example"]
            b = pe[(pe.group != "pairs_seen") & ~pe.exact]["best_agreement"].values
            if not len(b):
                continue
            h = np.bincount(b, minlength=37) / len(b)
            st = MODEL_STYLE.get(m, {"color": "#52514e", "ls": "-"})
            ax.plot(xs, h, color=st["color"], ls=st["ls"], lw=1.4, label=_name(m))
            data += [{"experiment": e, "series": m, "best_agreement_cells": x, "fraction": v} for x, v in zip(xs, h)]
        if "controls" in R:
            hist = R["controls"]["histogram"]
            for (cname, sub), ls in zip(hist.groupby("control", sort=False), [":", "-.", "--", (0, (1, 3)), (0, (5, 1))]):
                ax.plot(sub["best_agreement_cells"], sub["fraction"], color="#0b0b0b", ls=ls, lw=1.0, label=f"control: {cname}")
        ax.set(xlabel="Best cell agreement with any of the 288 legal transforms of the input (of 36)",
               ylabel="Fraction of examples", title=f"{EXP_LABEL[e]}: wrong predictions on unseen strings vs calibration controls\n"
                                                   "(max over 288 candidates has a selection advantage)")
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    return save_fig(fig, out, "06_nearest_group_agreement_vs_controls", pd.DataFrame(data))


def fig_z(R, out):
    import pandas as pd
    plt = _plt()
    z = R["z_table"]
    if not len(z):
        return None
    exps = [e for e in EXPS if e in set(z.experiment)]
    cols = [("singles", "odd"), ("pairs_seen", "even"), ("pairs_unseen", "even"), ("triples_all_distinct", "odd"),
            ("triples_two_types", "odd"), ("triples_one_type", "odd")]
    fig, axes = plt.subplots(len(exps), 1, figsize=(15, 4.2 * len(exps)), squeeze=False)
    data = []
    for ax, e in zip(axes[:, 0], exps):
        sub = z[z.experiment == e]
        models = _models(sub)
        mat = np.full((len(models), 2 * len(cols)), np.nan)
        labels = []
        for c, (g, par) in enumerate(cols):
            labels += [f"{GROUP_SHORT[g]}\n({par} length)\nTRUE target", f"{GROUP_SHORT[g]}\n({par} length)\nALT generator"]
            for i, m in enumerate(models):
                s = sub[(sub.model == m) & (sub.group == g)]
                if len(s):
                    mat[i, 2 * c] = np.average(s["exact_acc"], weights=s["n"])
                    mat[i, 2 * c + 1] = np.average(s["alt_generator_exact_rate"], weights=s["n"])
        ax.grid(False)
        ax.imshow(np.nan_to_num(mat, nan=-1), cmap="Greens", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, "N/A" if np.isnan(mat[i, j]) else f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if (not np.isnan(mat[i, j]) and mat[i, j] > 0.6) else "#0b0b0b")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=0)
        for b in range(2, len(labels), 2):
            ax.axvline(b - 0.5, color="#0b0b0b", lw=1.2)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([_name(m) for m in models], fontsize=8)
        ax.set_title(f"{EXP_LABEL[e]}: original target vs alternate generators A'=ZA, B'=ZB, C'=ZC "
                     "(odd length -> Z(target); even length -> target).\nExact agreement of model predictions; N/A = empty category", fontsize=9)
        df = pd.DataFrame(mat, columns=[l.replace("\n", " ") for l in labels])
        df.insert(0, "model", models); df.insert(0, "experiment", e)
        data.append(df)
    fig.tight_layout()
    return save_fig(fig, out, "07_alternate_generator_vs_target", pd.concat(data))


def fig_fidelity(R, out):
    import pandas as pd
    plt = _plt()
    ps = R["per_string"]
    exps = [e for e in EXPS if e in set(ps.experiment)]
    specs = [("stable_rule_diag_exact_fidelity", "DIAG split: exact fidelity of ONE fixed legal transform fitted on FIT grids"),
             ("perm_diag_cell_fidelity", "DIAG split: cell fidelity of a fitted bijective source permutation (any of 36!)"),
             ("argmax_map_diag_cell_fidelity", "DIAG split: cell fidelity of a fitted many-to-one source-copy map")]
    panels = [(e, c, t) for e in exps for c, t in specs]
    h = 1.2 + 0.42 * max(len(_models(ps[ps.experiment == e])) for e in exps)
    fig, axes = plt.subplots(len(panels), 1, figsize=(24, h * len(panels) + 1), squeeze=False)
    data = []
    for ax, (e, c, t) in zip(axes[:, 0], panels):
        models, mat = _per_string_matrix(ps, e, c)
        _string_heatmap(ax, mat, models, e, f"{EXP_LABEL[e]}: {t}", "Greens", 0, 1)
        df = pd.DataFrame(mat, columns=O.TEST_STRINGS)
        df.insert(0, "measure", c); df.insert(0, "model", models); df.insert(0, "experiment", e)
        data.append(df)
    fig.tight_layout()
    return save_fig(fig, out, "08_stable_rule_and_source_map_fidelity_DIAG", pd.concat(data))


def fig_composition(R, out):
    import pandas as pd
    plt = _plt()
    cp = R["composition"]
    if not len(cp):
        return None
    cp = cp[cp.map_kind == "bijective_permutation"]
    exps = [e for e in EXPS if e in set(cp.experiment)]
    strs = [s for s in O.TEST_STRINGS if len(s) >= 2]
    fig, axes = plt.subplots(2 * len(exps), 1, figsize=(20, 3.2 * 2 * len(exps)), squeeze=False)
    k = 0
    for e in exps:
        sub = cp[cp.experiment == e]
        models = _models(sub)
        for col, lab in (("agreement_with_model_exact", "composed single-op maps vs MODEL predictions (exact)"),
                         ("agreement_with_target_exact", "composed single-op maps vs TRUE targets (exact)")):
            mat = np.full((len(models), len(strs)), np.nan)
            for i, m in enumerate(models):
                s = sub[sub.model == m].set_index("string")
                for j, w in enumerate(strs):
                    if w in s.index:
                        mat[i, j] = s.loc[w, col]
            ax = axes[k, 0]; k += 1
            ax.grid(False)
            ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                                color="white" if mat[i, j] > 0.6 else "#0b0b0b")
            ax.set_xticks(range(len(strs))); ax.set_xticklabels(strs, fontsize=7, family="monospace")
            ax.set_yticks(range(len(models))); ax.set_yticklabels([_name(m) for m in models], fontsize=7)
            ax.set_title(f"{EXP_LABEL[e]} (DIAG split): {lab}. Weak when single-op maps are unstable (see table).", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, out, "09_composition_consistency_DIAG", cp)


def fig_source_maps(R, out, cfg):
    import pandas as pd
    plt = _plt()
    paths = []
    for e in EXPS:
        sel = [(m, s) for m in cfg["example_models"] for s in cfg["example_strings"][:4]
               if f"{e}/{m}" in R["per_run"] and s in R["per_run"][f"{e}/{m}"]["maps"]]
        if not sel:
            continue
        ncol = 4
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
        rows = []
        for ax, (m, s) in zip(axes.flat, sel):
            mp = R["per_run"][f"{e}/{m}"]["maps"][s]
            ax.grid(False)
            ax.imshow(mp["M"], cmap="Blues", vmin=0, vmax=1)
            tp = G.word_perm(s)
            ax.scatter(tp, np.arange(36), s=8, facecolors="none", edgecolors="#e34948", lw=0.7, label="true source")
            ax.scatter(mp["bijective_map"], np.arange(36), s=3, color="#eda100", label="fitted permutation")
            ax.set_title(f"{_name(m)}: '{s}'", fontsize=8)
            ax.set_xlabel("source input cell k", fontsize=7); ax.set_ylabel("output cell j", fontsize=7)
            rows.append({"experiment": e, "model": m, "string": s,
                         "fitted_permutation": " ".join(map(str, mp["bijective_map"])),
                         "argmax_map": " ".join(map(str, mp["argmax_map"]))})
        for ax in list(axes.flat)[len(sel):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=6, loc="lower right")
        fig.suptitle(f"{EXP_LABEL[e]}: copy matrix M[j,k] = P(prediction cell j == input cell k) on FIT grids "
                     "(predeclared examples; all maps saved in maps/*.npz)", fontsize=10)
        fig.tight_layout()
        paths += save_fig(fig, out, f"10_source_maps_{e}", pd.DataFrame(rows))
    return paths


def fig_paired(R, out):
    import pandas as pd
    plt = _plt()
    pc = R["paired"]
    if not len(pc):
        return None
    models = _models(pc)
    specs = [("delta_exact (all_pairs - distinct_pairs)", "true exact accuracy"),
             ("delta_any_legal (all_pairs - distinct_pairs)", "any exact legal-transform match"),
             ("delta_alt_generator_exact (all_pairs - distinct_pairs)", "alternate-generator (Z) exact agreement"),
             ("identical_prediction_rate", "IDENTICAL prediction in both experiments (not a delta)")]
    fig, axes = plt.subplots(len(specs), 1, figsize=(24, (1.2 + 0.42 * len(models)) * len(specs) + 1), squeeze=False)
    for ax, (c, lab) in zip(axes[:, 0], specs):
        mat = np.full((len(models), 39), np.nan)
        for i, m in enumerate(models):
            s = pc[pc.model == m].set_index("string")
            for j, w in enumerate(O.TEST_STRINGS):
                if w in s.index:
                    mat[i, j] = s.loc[w, c]
        delta = c.startswith("delta")
        _string_heatmap(ax, mat, models, "distinct_pairs",
                        f"Paired by model/string/grid: {lab}" + (" (exp2 - exp1); AA/BB/CC are UNSEEN in exp1 but SEEN in exp2" if delta else ""),
                        "RdBu" if delta else "Blues", -1 if delta else 0, 1)
    fig.tight_layout()
    return save_fig(fig, out, "11_paired_experiment_comparison", pc)


def fig_examples(R, out, cfg):
    """One figure per (experiment, model) for predeclared grid IDs and strings."""
    import pandas as pd
    plt = _plt()
    cat = R["catalog"]
    paths = []
    for e in EXPS:
        for r in [r for r in R["runs"] if r.experiment == e and r.model in cfg["example_models"] and r.key in R["per_run"]]:
            pe = R["per_run"][r.key]["per_example"]
            items = []
            for gid in cfg["example_grid_ids"]:
                for s in cfg["example_strings"]:
                    idx = np.nonzero((r.string == s) & (r.grid_id == gid))[0]
                    if len(idx):
                        items.append((int(idx[0]), pe.iloc[int(idx[0])]))
            if not items:
                continue
            fig, axes = plt.subplots(len(items), 5, figsize=(9.5, 2.0 * len(items) + 1.0), squeeze=False)
            rows = []
            for row, (i, d) in zip(axes, items):
                x, y, p, s = r.X[i], r.Y[i], r.P[i], r.string[i]
                ag = (x[cat.perms] == p[None]).sum(1)
                best = int(np.argmax(ag))
                near = x[cat.perms[best]]
                draw_grid(row[0], x, f"input (grid {r.grid_id[i]})")
                draw_grid(row[1], y, f"true target '{s}'")
                draw_grid(row[2], p, f"model prediction\n({36 - int((p == y).sum())} cells wrong)", (p != y).reshape(6, 6))
                draw_grid(row[3], near, f"nearest legal transform\n{cat.label(best)} ({int(ag[best])}/36 agree)",
                          (near != p).reshape(6, 6))
                if len(s) % 2:
                    zt = G.apply_perm(G.Z_SRC, y)
                    draw_grid(row[4], zt, f"Z(target)\n({int((zt == p).sum())}/36 agree)", (zt != p).reshape(6, 6))
                else:
                    row[4].axis("off")
                    row[4].text(0.5, 0.5, "even length:\nalt-generator\nprediction = target", ha="center", va="center",
                                fontsize=7, transform=row[4].transAxes)
                rows.append({"experiment": e, "model": r.model, "grid_id": int(r.grid_id[i]), "string": s,
                             "exact": bool(d["exact"]), "nearest_word": cat.label(best), "nearest_agreement": int(ag[best])})
            fig.suptitle(f"{EXP_LABEL[e]} - {_name(r.model)}: predeclared examples. Red boxes: column 3 = cells differing "
                         "from the target;\ncolumns 4-5 = cells differing from the model prediction. Nearest legal "
                         "transforms are post-hoc explanations, NOT corrected answers.", fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (2.0 * len(items) + 1.0)))
            paths += save_fig(fig, out, f"12_examples_{e}__{r.model}", pd.DataFrame(rows))
    return paths


def make_all(R, out: Path, log=print) -> dict:
    out = Path(out)
    cfg = R["config"]
    jobs = {
        "01_true_exact": lambda: heatmap_pair(R, "exact_acc", "01_true_exact_accuracy_per_string",
                                              "TRUE exact-grid accuracy (recomputed from raw records)", out),
        "02_any_legal": lambda: heatmap_pair(R, "any_exact_group_match_rate", "02_any_exact_legal_transform_match",
                                             "fraction of predictions exactly equal to SOME legal transform of the input", out,
                                             "Purples"),
        "03_categories": lambda: fig_categories(R, out),
        "04_dominant": lambda: fig_dominant(R, out),
        "05_hypotheses": lambda: fig_hypotheses(R, out),
        "06_controls": lambda: fig_nearest_vs_controls(R, out),
        "07_z": lambda: fig_z(R, out),
        "08_fidelity": lambda: fig_fidelity(R, out),
        "09_composition": lambda: fig_composition(R, out),
        "10_maps": lambda: fig_source_maps(R, out, cfg),
        "11_paired": lambda: fig_paired(R, out),
        "12_examples": lambda: fig_examples(R, out, cfg),
    }
    done = {}
    for k, f in jobs.items():
        try:
            done[k] = f()
        except Exception:  # report loudly but keep the remaining figures
            log(f"FIGURE FAILED: {k}\n{traceback.format_exc()}")
            done[k] = "FAILED"
    return done
