"""Figures for the generalization-failure diagnostics (legacy or mixed family; torch-free).
Every figure is saved as PNG + PDF together with a CSV of the plotted numbers."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List

import numpy as np

from ..names import DISPLAY_NAMES, MODEL_IDS
from ..plotstyle import MODEL_STYLE, _plt, draw_grid, save_fig
from ..words import from_label
from . import group as G


def _models(df) -> List[str]:
    present = set(df["model"]) if len(df) else set()
    return [m for m in MODEL_IDS if m in present] + sorted(present - set(MODEL_IDS))


def _name(m):
    return DISPLAY_NAMES.get(m, m)


def _exps(R, df=None):
    spec = R["spec"]
    have = set(df.experiment) if df is not None and len(df) else {r.experiment for r in R["runs"]}
    return [e for e in spec.experiments if e in have]


def _elabel(R, e):
    return R["spec"].exp_labels.get(e, e)


def panel_heatmap(ax, mat, models, words, panel_groups, seen, title, cmap="Blues", vmin=0, vmax=1, text=None):
    ax.grid(False)
    im = ax.imshow(np.nan_to_num(mat, nan=0.0 if cmap == "RdBu" else vmin), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            t = text[i][j] if text is not None else ("n/a" if np.isnan(v) else f"{v:.2f}")
            frac = 0 if np.isnan(v) else (v - vmin) / (vmax - vmin)
            dark = (frac > 0.6) if cmap in ("Blues", "Purples", "Greens", "Oranges") else abs(0 if np.isnan(v) else v) > 0.6
            ax.text(j, i, t, ha="center", va="center", fontsize=5.5, color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels([w + ("*" if w in seen else "") for w in words], fontsize=7, family="monospace", rotation=90)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_name(m) for m in models], fontsize=8)
    start = 0
    for g, n in panel_groups:
        if start:
            ax.axvline(start - 0.5, color="#0b0b0b", lw=1.4)
        ax.text(start + n / 2 - 0.5, -0.75, g, ha="center", va="bottom", fontsize=8 if n > 3 else 6.5, weight="bold")
        start += n
    ax.set_title(title + "   (* = word in this experiment's training support)", pad=24, fontsize=10)
    return im


def _matrix(ps, exp, col, words):
    sub = ps[ps.experiment == exp]
    models = _models(sub)
    mat = np.full((len(models), len(words)), np.nan)
    for i, m in enumerate(models):
        s = sub[sub.model == m].set_index("string")
        for j, w in enumerate(words):
            if w in s.index and col in s:
                mat[i, j] = s.loc[w, col]
    return models, mat


def heatmap_pair(R, col, name, title, out, cmap="Blues", vmin=0, vmax=1, df=None):
    import pandas as pd
    plt = _plt()
    spec = R["spec"]
    ps = R["per_string"] if df is None else df
    exps = _exps(R, ps)
    if not exps:
        return None
    words = spec.common_panel
    heights = [1.4 + 0.42 * len(_models(ps[ps.experiment == e])) for e in exps]
    fig, axes = plt.subplots(len(exps), 1, figsize=(max(14, 0.6 * len(words)), sum(heights) + 1.2), squeeze=False,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.95})
    data = []
    for ax, e in zip(axes[:, 0], exps):
        models, mat = _matrix(ps, e, col, words)
        im = panel_heatmap(ax, mat, models, words, spec.panel_groups, set(spec.train_labels(e)),
                           f"{_elabel(R, e)}: {title}", cmap, vmin, vmax)
        d = pd.DataFrame(mat, columns=words)
        d.insert(0, "model", models); d.insert(0, "experiment", e)
        data.append(d)
    fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.012, pad=0.01)
    return save_fig(fig, out, name, pd.concat(data))


def fig_categories(R, out):
    import pandas as pd
    plt = _plt()
    spec = R["spec"]
    ps = R["per_string"]
    cats = [("exact_acc", "correct", "#2a78d6"), ("wrong_unique_legal_rate", "wrong: unique legal transform", "#eb6834"),
            ("wrong_ambiguous_legal_rate", "wrong: ambiguous legal (>1 transform)", "#eda100"),
            ("no_legal_match_rate", "wrong: no legal transform matches", "#8c8c8c")]
    groups = spec.group_order
    exps = _exps(R, ps)
    fig, axes = plt.subplots(len(exps), len(groups), figsize=(3.4 * len(groups) + 1, 3.6 * len(exps)), squeeze=False,
                             sharey="row")
    rows = []
    for r, e in enumerate(exps):
        sub = ps[ps.experiment == e]
        models = _models(sub)
        for c, g in enumerate(groups):
            ax = axes[r, c]
            sg = sub[sub.group == g]
            left = np.zeros(len(models))
            if not len(sg):
                ax.text(0.5, 0.5, "N/A (no words in\nthis category)", ha="center", va="center", transform=ax.transAxes)
            for col, lab, colr in cats:
                vals = np.array([np.average(sg[sg.model == m][col], weights=sg[sg.model == m]["n"])
                                 if len(sg[sg.model == m]) else 0 for m in models])
                ax.barh(range(len(models)), vals, left=left, color=colr, label=lab, edgecolor="white", linewidth=0.6)
                left += vals
                for m, v in zip(models, vals):
                    rows.append({"experiment": e, "group": g, "model": m, "category": lab, "fraction": v,
                                 "records": int(sg[sg.model == m]["n"].sum())})
            ax.set_xlim(0, 1)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([_name(m) for m in models], fontsize=7)
            ax.invert_yaxis()
            n_rec = int(sg[sg.model == models[0]]["n"].sum()) if len(sg) and models else 0
            ax.set_title(f"{_elabel(R, e)}\n{spec.group_labels.get(g, g)} (n={n_rec:,}/model)", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.99, wspace=0.15, hspace=0.55, top=0.84)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.3), ncol=4, fontsize=8, frameon=False)
    fig.suptitle("What do predictions match? Exact match against all 288 legal grid transformations "
                 "(post-hoc; categories exclusive per example)", y=0.995)
    return save_fig(fig, out, "03_explanation_categories", pd.DataFrame(rows))


def fig_dominant(R, out):
    import pandas as pd
    plt = _plt()
    spec = R["spec"]
    ps = R["per_string"]
    exps = _exps(R, ps)
    words = spec.common_panel
    heights = [1.4 + 0.42 * len(_models(ps[ps.experiment == e])) for e in exps]
    fig, axes = plt.subplots(len(exps), 1, figsize=(max(14, 0.6 * len(words)), sum(heights) + 1.2), squeeze=False,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.95})
    data = []
    for ax, e in zip(axes[:, 0], exps):
        sub = ps[ps.experiment == e]
        models = _models(sub)
        mat = np.full((len(models), len(words)), np.nan)
        txt = [["" for _ in words] for _ in models]
        for i, m in enumerate(models):
            s = sub[sub.model == m].set_index("string")
            for j, w in enumerate(words):
                if w in s.index:
                    mat[i, j] = s.loc[w, "dominant_wrong_unique_share"]
                    wd = s.loc[w, "dominant_wrong_unique_word"]
                    txt[i][j] = (wd if len(wd) <= 5 else wd[:4] + "…") if isinstance(wd, str) and wd else "–"
        panel_heatmap(ax, mat, models, words, spec.panel_groups, set(spec.train_labels(e)),
                      f"{_elabel(R, e)}: share of examples whose ONLY exact legal match is the dominant wrong "
                      "transformation (text = its canonical word)", "Oranges", 0, 1, txt)
        data.append(sub[["experiment", "model", "string", "dominant_wrong_unique_word", "dominant_wrong_unique_share",
                         "dominant_is_training_transform", "dominant_is_original_pair"]])
    return save_fig(fig, out, "04_dominant_wrong_transformation", pd.concat(data))


def fig_hypotheses(R, out):
    """Unconditional coverage of each rule on the main declared population, annotated with the conditional
    rate and the eligible record count (explicit denominators)."""
    plt = _plt()
    hs = R.get("hypothesis_summary")
    if hs is None or not len(hs):
        return None
    pop = list(R["spec"].populations)[0]
    hs = hs[hs.population == pop]
    exps = _exps(R, hs)
    types = [t for t in G.HYPOTHESIS_TYPES if t in set(hs.hypothesis_type)]
    fig, axes = plt.subplots(1, len(exps), figsize=(10 * len(exps), 7), squeeze=False)
    for ax, e in zip(axes[0], exps):
        sub = hs[hs.experiment == e]
        models = _models(sub)
        mat = np.full((len(types), len(models)), np.nan)
        txt = [["" for _ in models] for _ in types]
        for i, t in enumerate(types):
            for j, m in enumerate(models):
                r = sub[(sub.hypothesis_type == t) & (sub.model == m)]
                if len(r):
                    r = r.iloc[0]
                    mat[i, j] = r["unconditional_coverage (matched/population)"]
                    cond = r["conditional_rate (matched/eligible)"]
                    txt[i][j] = (f"{mat[i, j]:.2f}\ncond {cond:.2f}\nelig {int(r.eligible_records):,}"
                                 if r.eligible_records else "not\neligible")
        ax.grid(False)
        ax.imshow(np.nan_to_num(mat), cmap="Purples", vmin=0, vmax=1, aspect="auto")
        for i in range(len(types)):
            for j in range(len(models)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=5.5,
                        color="white" if (not np.isnan(mat[i, j]) and mat[i, j] > 0.6) else "#0b0b0b")
        ax.set_xticks(range(len(models))); ax.set_xticklabels([_name(m) for m in models], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(types))); ax.set_yticklabels(types, fontsize=8)
        popn = int(sub["population_records"].iloc[0]) if len(sub) else 0
        ax.set_title(f"{_elabel(R, e)} - population: {pop} ({popn:,} records/model)\ncolour = UNCONDITIONAL coverage "
                     "(unique matched / population); cond = matched / eligible. Rules overlap; truth-equivalent "
                     "candidates excluded.", fontsize=8.5)
    fig.tight_layout()
    return save_fig(fig, out, "05_hypothesis_compatibility", hs)


def fig_nearest_vs_controls(R, out):
    import pandas as pd
    plt = _plt()
    spec = R["spec"]
    pop_groups = list(spec.populations.values())[0]
    exps = _exps(R)
    fig, axes = plt.subplots(1, len(exps), figsize=(9 * len(exps), 4.8), squeeze=False, sharey=True)
    data = []
    xs = np.arange(37)
    for ax, e in zip(axes[0], exps):
        for k, res in R["per_run"].items():
            if not k.startswith(e + "/"):
                continue
            m = k.split("/", 1)[1]
            pe = res["per_example"]
            b = pe[pe.group.isin(pop_groups) & ~pe.exact]["best_agreement"].values
            if not len(b):
                continue
            h = np.bincount(b.astype(int), minlength=37) / len(b)
            st = MODEL_STYLE.get(m, {"color": "#52514e", "ls": "-"})
            ax.plot(xs, h, color=st["color"], ls=st["ls"], lw=1.4, label=f"{_name(m)} (n={len(b):,})")
            data += [{"experiment": e, "series": m, "best_agreement_cells": x, "fraction": v} for x, v in zip(xs, h)]
        if "controls" in R:
            hist = R["controls"]["histogram"]
            for (cname, sub), ls in zip(hist.groupby("control", sort=False), [":", "-.", "--", (0, (1, 3)), (0, (5, 1))]):
                ax.plot(sub["best_agreement_cells"], sub["fraction"], color="#0b0b0b", ls=ls, lw=1.0, label=f"control: {cname}")
        ax.set(xlabel="Best cell agreement with any of the 288 legal transforms of the input (of 36)",
               ylabel="Fraction of examples",
               title=f"{_elabel(R, e)}: WRONG predictions ({list(spec.populations)[0]}) vs calibration controls\n"
                     "(the max over 288 candidates has a selection advantage; its random baseline is NOT 0.1)")
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    return save_fig(fig, out, "06_nearest_group_agreement_vs_controls", pd.DataFrame(data))


def fig_z(R, out):
    import pandas as pd
    plt = _plt()
    spec = R["spec"]
    z = R["z_table"]
    if not len(z):
        return None
    exps = _exps(R, z)
    groups = [g for g in spec.group_order if g != "EMPTY"]
    fig, axes = plt.subplots(len(exps), 1, figsize=(max(15, 2.2 * len(groups)), 4.2 * len(exps)), squeeze=False)
    data = []
    for ax, e in zip(axes[:, 0], exps):
        sub = z[z.experiment == e]
        models = _models(sub)
        mat = np.full((len(models), 2 * len(groups)), np.nan)
        labels = []
        for c, g in enumerate(groups):
            sg = sub[sub.group == g]
            par = ("odd" if int(sg["length"].iloc[0]) % 2 else "even") if len(sg) else "n/a"
            labels += [f"{g}\n({par})\nTRUE target", f"{g}\n({par})\nALT generator"]
            for i, m in enumerate(models):
                s = sg[sg.model == m]
                if len(s):
                    mat[i, 2 * c] = np.average(s["exact_acc"], weights=s["n"])
                    mat[i, 2 * c + 1] = np.average(s["alt_generator_exact_rate"], weights=s["n"])
        ax.grid(False)
        ax.imshow(np.nan_to_num(mat, nan=-1), cmap="Greens", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, "N/A" if np.isnan(mat[i, j]) else f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if (not np.isnan(mat[i, j]) and mat[i, j] > 0.6) else "#0b0b0b")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6.5)
        for b in range(2, len(labels), 2):
            ax.axvline(b - 0.5, color="#0b0b0b", lw=1.2)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([_name(m) for m in models], fontsize=8)
        ax.set_title(f"{_elabel(R, e)}: original target vs half-period alternate generators A'=ZA, B'=ZB, C'=ZC "
                     "(secondary check; odd length -> Z(target)). N/A = empty category", fontsize=9)
        d = pd.DataFrame(mat, columns=[l.replace("\n", " ") for l in labels])
        d.insert(0, "model", models); d.insert(0, "experiment", e)
        data.append(d)
    fig.tight_layout()
    return save_fig(fig, out, "07_alternate_generator_vs_target", pd.concat(data))


def fig_fidelity(R, out):
    import pandas as pd
    plt = _plt()
    spec = R["spec"]
    ps = R["per_string"]
    exps = _exps(R, ps)
    words = spec.common_panel
    specs = [("stable_rule_diag_exact_fidelity", "DIAG: exact fidelity of ONE fixed legal transform fitted on FIT grids"),
             ("perm_diag_cell_fidelity", "DIAG: cell fidelity of a fitted bijective source permutation"),
             ("argmax_map_diag_cell_fidelity", "DIAG: cell fidelity of a fitted many-to-one source-copy map")]
    panels = [(e, c, t) for e in exps for c, t in specs]
    h = 1.4 + 0.42 * max(len(_models(ps[ps.experiment == e])) for e in exps)
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(14, 0.6 * len(words)), h * len(panels) + 1), squeeze=False)
    data = []
    for ax, (e, c, t) in zip(axes[:, 0], panels):
        models, mat = _matrix(ps, e, c, words)
        panel_heatmap(ax, mat, models, words, spec.panel_groups, set(spec.train_labels(e)), f"{_elabel(R, e)}: {t}",
                      "Greens", 0, 1)
        d = pd.DataFrame(mat, columns=words)
        d.insert(0, "measure", c); d.insert(0, "model", models); d.insert(0, "experiment", e)
        data.append(d)
    fig.tight_layout()
    return save_fig(fig, out, "08_stable_rule_and_source_map_fidelity_DIAG", pd.concat(data))


def fig_legal_vs_map(R, out):
    """Exact legal-transform matching vs general source-map fidelity for out-of-support words."""
    plt = _plt()
    spec = R["spec"]
    ps = R["per_string"]
    pop_groups = list(spec.populations.values())[0]
    sub = ps[ps.group.isin(pop_groups)]
    if not len(sub):
        return None
    exps = _exps(R, sub)
    fig, axes = plt.subplots(1, len(exps), figsize=(7 * len(exps), 5.5), squeeze=False)
    for ax, e in zip(axes[0], exps):
        s = sub[sub.experiment == e]
        for m in _models(s):
            sm = s[s.model == m]
            st = MODEL_STYLE.get(m, {"color": "#52514e"})
            ax.scatter(sm["any_exact_group_match_rate"], sm["argmax_map_diag_cell_fidelity"], color=st["color"], s=28,
                       label=_name(m), alpha=0.85, edgecolor="white")
            for _, r in sm.iterrows():
                ax.annotate(r["string"], (r["any_exact_group_match_rate"], r["argmax_map_diag_cell_fidelity"]), fontsize=5,
                            xytext=(2, 2), textcoords="offset points")
        ax.set(xlim=(-0.03, 1.03), ylim=(0, 1.03), xlabel="Exact legal-transform match rate (all records)",
               ylabel="DIAG cell fidelity of fitted source-copy map",
               title=f"{_elabel(R, e)}: {list(spec.populations)[0]} words\n(low exact legal match can coexist with high "
                     "source-map fidelity)")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    return save_fig(fig, out, "13_exact_legal_match_vs_source_map_fidelity", sub[[
        "experiment", "model", "string", "group", "any_exact_group_match_rate", "argmax_map_diag_cell_fidelity",
        "perm_diag_cell_fidelity", "stable_rule_diag_exact_fidelity"]])


def fig_composition(R, out):
    plt = _plt()
    spec = R["spec"]
    cp = R["composition"]
    if not len(cp):
        return None
    cp = cp[cp.map_kind == "bijective_permutation"]
    exps = _exps(R, cp)
    words = [w for w in spec.common_panel if len(from_label(w)) >= 2]
    fig, axes = plt.subplots(2 * len(exps), 1, figsize=(20, 3.2 * 2 * len(exps)), squeeze=False)
    k = 0
    for e in exps:
        sub = cp[cp.experiment == e]
        models = _models(sub)
        for col, lab in (("agreement_with_model_exact", "composed single-op maps vs MODEL predictions (exact)"),
                         ("agreement_with_target_exact", "composed single-op maps vs TRUE targets (exact)")):
            mat = np.full((len(models), len(words)), np.nan)
            for i, m in enumerate(models):
                s = sub[sub.model == m].set_index("string")
                for j, w in enumerate(words):
                    if w in s.index:
                        mat[i, j] = s.loc[w, col]
            ax = axes[k, 0]; k += 1
            ax.grid(False)
            ax.imshow(np.nan_to_num(mat), cmap="Greens", vmin=0, vmax=1, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                                color="white" if mat[i, j] > 0.6 else "#0b0b0b")
            ax.set_xticks(range(len(words))); ax.set_xticklabels(words, fontsize=7, family="monospace", rotation=90)
            ax.set_yticks(range(len(models))); ax.set_yticklabels([_name(m) for m in models], fontsize=7)
            ax.set_title(f"{_elabel(R, e)} (DIAG split): {lab}. WEAK when single-op maps are unstable (see table).",
                         fontsize=9)
    fig.tight_layout()
    return save_fig(fig, out, "09_composition_consistency_DIAG", cp)


def fig_source_maps(R, out, cfg):
    import pandas as pd
    plt = _plt()
    paths = []
    for e in _exps(R):
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
            tp = G.word_perm(from_label(s))
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
        fig.suptitle(f"{_elabel(R, e)}: copy matrix M[j,k] = P(prediction cell j == input cell k) on FIT grids "
                     "(predeclared examples; all maps saved in maps/*.npz)", fontsize=10)
        fig.tight_layout()
        paths += save_fig(fig, out, f"10_source_maps_{e}", pd.DataFrame(rows))
    return paths


def fig_paired(R, out):
    plt = _plt()
    spec = R["spec"]
    pc = R["paired"]
    if not len(pc) or not spec.paired:
        return None
    ea, eb = spec.paired
    models = _models(pc)
    words = spec.common_panel
    specs = [(f"delta_exact ({eb} - {ea})", "true exact accuracy"),
             (f"delta_any_legal ({eb} - {ea})", "any exact legal-transform match"),
             ("identical_prediction_rate", "IDENTICAL prediction in both experiments (not a delta)")]
    fig, axes = plt.subplots(len(specs), 1, figsize=(max(14, 0.6 * len(words)), (1.4 + 0.42 * len(models)) * len(specs) + 1),
                             squeeze=False)
    for ax, (c, lab) in zip(axes[:, 0], specs):
        mat = np.full((len(models), len(words)), np.nan)
        for i, m in enumerate(models):
            s = pc[pc.model == m].set_index("string")
            for j, w in enumerate(words):
                if w in s.index:
                    mat[i, j] = s.loc[w, c]
        delta = c.startswith("delta")
        panel_heatmap(ax, mat, models, words, spec.panel_groups, set(spec.train_labels(ea)),
                      f"Paired by model/word/grid: {lab}" + (f" ({eb} - {ea}); * marks {ea} training support" if delta else ""),
                      "RdBu" if delta else "Blues", -1 if delta else 0, 1)
    fig.tight_layout()
    return save_fig(fig, out, "11_paired_experiment_comparison", pc)


def fig_examples(R, out, cfg):
    import pandas as pd
    plt = _plt()
    cat = R["catalog"]
    paths = []
    for r in [r for r in R["runs"] if r.model in cfg["example_models"] and r.key in R["per_run"]]:
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
            if len(from_label(s)) % 2:
                zt = G.apply_perm(G.Z_SRC, y)
                draw_grid(row[4], zt, f"Z(target)\n({int((zt == p).sum())}/36 agree)", (zt != p).reshape(6, 6))
            else:
                row[4].axis("off")
                row[4].text(0.5, 0.5, "even length:\nalt-generator\nprediction = target", ha="center", va="center",
                            fontsize=7, transform=row[4].transAxes)
            rows.append({"experiment": r.experiment, "model": r.model, "grid_id": int(r.grid_id[i]), "string": s,
                         "exact": bool(d["exact"]), "nearest_word": cat.label(best), "nearest_agreement": int(ag[best])})
        fig.suptitle(f"{_elabel(R, r.experiment)} - {_name(r.model)}: predeclared examples. Red boxes: column 3 = cells "
                     "differing from the target;\ncolumns 4-5 = cells differing from the model prediction. Nearest legal "
                     "transforms are post-hoc explanations, NOT corrected answers.", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (2.0 * len(items) + 1.0)))
        paths += save_fig(fig, out, f"12_examples_{r.experiment}__{r.model}", pd.DataFrame(rows))
    return paths


def make_all(R, out: Path, log=print) -> dict:
    out = Path(out)
    cfg = R["config"]
    jobs = {
        "01_true_exact": lambda: heatmap_pair(R, "exact_acc", "01_true_exact_accuracy_per_string",
                                              "TRUE exact-grid accuracy (recomputed from raw records)", out),
        "02_any_legal": lambda: heatmap_pair(R, "any_exact_group_match_rate", "02_any_exact_legal_transform_match",
                                             "fraction of predictions exactly equal to SOME legal transform of the input",
                                             out, "Purples"),
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
        "13_legal_vs_map": lambda: fig_legal_vs_map(R, out),
    }
    done = {}
    for k, f in jobs.items():
        try:
            done[k] = f()
        except Exception:  # report loudly but keep the remaining figures
            log(f"FIGURE FAILED: {k}\n{traceback.format_exc()}")
            done[k] = "FAILED"
    return done
