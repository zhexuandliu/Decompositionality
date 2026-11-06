"""Figures, tables and numerical summaries for one experiment directory.

Every figure is saved as PNG + PDF together with the CSV of the plotted numbers, so no
result ever has to be read off a plot. Only one training seed is used: no error bars or
significance statements are produced.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import operations as O
from .evaluation import fmt, load_predictions
from .models.registry import DISPLAY_NAMES, MODEL_IDS, TIED_PAIRS, model_summary

# Six hue families = slots 1-6 of the validated reference categorical palette, in its
# documented (adjacent-pair validated) order. Each tied baseline uses its
# untied counterpart's hue, lighter and hatched (composite encoding instead of a 9th hue).
_FAMILY = {"ar": "#2a78d6", "enc": "#eb6834", "vit": "#1baf7a", "trm": "#eda100", "res": "#e87ba4", "unet": "#008300"}
MODEL_STYLE = {
    "transformer_ar": dict(color=_FAMILY["ar"], hatch=None, ls="-"),
    "looped_transformer_ar": dict(color=_FAMILY["ar"], hatch="///", ls="--"),
    "transformer_encoder": dict(color=_FAMILY["enc"], hatch=None, ls="-"),
    "looped_transformer_encoder": dict(color=_FAMILY["enc"], hatch="///", ls="--"),
    "vit": dict(color=_FAMILY["vit"], hatch=None, ls="-"),
    "looped_vit": dict(color=_FAMILY["vit"], hatch="///", ls="--"),
    "trm": dict(color=_FAMILY["trm"], hatch=None, ls="-"),
    "rescnn_film": dict(color=_FAMILY["res"], hatch=None, ls="-"),
    "unet_film": dict(color=_FAMILY["unet"], hatch=None, ls="-"),
}
# Fixed categorical colors for digit values 0..9 (digits are also printed in every cell).
DIGIT_COLORS = ["#f2f2f2", "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#8c8c8c",
                "#e87ba4", "#eb6834", "#6ec6e8", "#7a1f2b"]
INVALID_COLOR = "#000000"
GROUPS_FOR_BARS = ["singles", "pairs_seen", "pairs_unseen", "triples_all_distinct", "triples_two_types",
                   "triples_one_type"]
SHARED_EXAMPLE_GRID_IDS = [0]
SHARED_EXAMPLE_STRINGS = ["A", "B", "C", "AB", "AA", "ABC", "ABA", "CCC"]  # predetermined, not selected by outcome


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.6, "font.size": 10, "savefig.dpi": 150,
                         "axes.titleweight": "bold"})
    return plt


def _light(hex_color: str, amount: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    rgb = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def bar_style(model_id: str) -> dict:
    s = MODEL_STYLE[model_id]
    face = _light(s["color"]) if s["hatch"] else s["color"]
    return dict(facecolor=face, edgecolor=s["color"], hatch=s["hatch"], linewidth=0.8)


def save_fig(fig, out: Path, name: str, data=None) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    if data is not None:
        data.to_csv(out / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def available_models(exp_dir: Path, kind: str = "eval") -> List[str]:
    base = exp_dir / ("eval" if kind == "eval" else "runs")
    marker = "EVAL_COMPLETE.json" if kind == "eval" else "val_log.csv"
    return [m for m in MODEL_IDS if (base / m / marker).exists()]


def load_tables(exp_dir: Path):
    import pandas as pd
    models = available_models(exp_dir)
    ps = [pd.read_csv(exp_dir / "eval" / m / "test_per_string.csv", keep_default_na=False, na_values=[""]) for m in models]
    gr = [pd.read_csv(exp_dir / "eval" / m / "test_groups.csv") for m in models]
    if not ps:
        return models, None, None
    return models, pd.concat(ps, ignore_index=True), pd.concat(gr, ignore_index=True)


# ---------------------------------------------------------------------------
# (a) training curves
# ---------------------------------------------------------------------------

def plot_training_curves(exp_dir: Path, out: Path):
    import pandas as pd
    plt = _plt()
    models = available_models(exp_dir, "runs")
    if not models:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rows = []
    for m in models:
        st = MODEL_STYLE[m]
        tl = exp_dir / "runs" / m / "train_log.csv"
        vl = pd.read_csv(exp_dir / "runs" / m / "val_log.csv")
        if tl.exists():
            t = pd.read_csv(tl)
            lbl = t["loss_label"].iloc[0] if "loss_label" in t else ""
            axes[0].plot(t["step"], t["loss"], color=st["color"], ls=st["ls"], lw=1.6,
                         label=f"{DISPLAY_NAMES[m]}: {lbl}")
            for _, r in t.iterrows():
                rows.append({"model": m, "panel": "training_objective", "objective": lbl, "step": r["step"], "value": r["loss"]})
        axes[1].plot(vl["step"], vl["val_cell_ce"], color=st["color"], ls=st["ls"], lw=1.6, marker="o", ms=3,
                     label=f"{DISPLAY_NAMES[m]} ({vl['val_cell_ce_label'].iloc[0]})")
        axes[2].plot(vl["step"], vl["subset_exact_acc"], color=st["color"], ls=st["ls"], lw=1.6, marker="o", ms=3,
                     label=DISPLAY_NAMES[m])
        for _, r in vl.iterrows():
            rows.append({"model": m, "panel": "val_cell_ce", "objective": r["val_cell_ce_label"], "step": r["step"], "value": r["val_cell_ce"]})
            rows.append({"model": m, "panel": "subset_exact_acc", "objective": "free-running/argmax exact-grid accuracy",
                         "step": r["step"], "value": r["subset_exact_acc"]})
    axes[0].set(title="Training objective (objectives differ by model family)", xlabel="Optimizer step",
                ylabel="Training loss (see legend)", yscale="log")
    axes[1].set(title="Validation final-answer cell CE (seen pairs, full val set)", xlabel="Optimizer step",
                ylabel="Cell cross-entropy", yscale="log")
    axes[2].set(title="Fixed 32-grid validation SUBSET: exact-grid accuracy\n(AR = free-running decoding)",
                xlabel="Optimizer step", ylabel="Exact-grid accuracy", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return save_fig(fig, out, "a_training_curves", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# (b) grouped bars
# ---------------------------------------------------------------------------

def plot_group_bars(groups, out: Path, metric: str = "exact_acc", name: str = "b_group_exact_accuracy"):
    import pandas as pd
    plt = _plt()
    models = [m for m in MODEL_IDS if m in set(groups["model"])]
    G = GROUPS_FOR_BARS
    fig, ax = plt.subplots(figsize=(16, 5.5))
    width = 0.8 / max(1, len(models))
    rows = []
    for i, m in enumerate(models):
        sub = groups[groups["model"] == m].set_index("group")
        vals = [sub.loc[g, metric] if g in sub.index else float("nan") for g in G]
        xs = np.arange(len(G)) + (i - (len(models) - 1) / 2) * width
        ax.bar(xs, np.nan_to_num(vals, nan=0.0), width=width * 0.92, label=DISPLAY_NAMES[m], **bar_style(m))
        for x, v, g in zip(xs, vals, G):
            if isinstance(v, float) and math.isnan(v):
                ax.text(x, 0.02, "N/A", rotation=90, ha="center", va="bottom", fontsize=7, color="#52514e")
            rows.append({"model": m, "group": g, metric: v})
    ax.set_xticks(np.arange(len(G)))
    ax.set_xticklabels([O.GROUP_LABELS[g] for g in G])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel({"exact_acc": "Exact-grid accuracy (macro over strings)",
                   "cell_acc": "Cell accuracy (macro over strings)"}[metric])
    ax.set_title(f"Test {'exact-grid' if metric == 'exact_acc' else 'cell'} accuracy by string group "
                 "(1000 held-out grids per string; single seed, no error bars)")
    ax.legend(ncol=3, fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    return save_fig(fig, out, name, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# (c) per-string heatmap
# ---------------------------------------------------------------------------

def plot_string_heatmap(per_string, out: Path, train_strings: Sequence[str], value: str = "exact_acc",
                        name: str = "c_per_string_exact_heatmap", cmap: str = "Blues", vmin=0.0, vmax=1.0,
                        title: Optional[str] = None):
    plt = _plt()
    models = [m for m in MODEL_IDS if m in set(per_string["model"])]
    strings = O.TEST_STRINGS
    mat = np.full((len(models), len(strings)), np.nan)
    for i, m in enumerate(models):
        sub = per_string[per_string["model"] == m].set_index("string")
        for j, s in enumerate(strings):
            if s in sub.index:
                mat[i, j] = sub.loc[s, value]
    fig, ax = plt.subplots(figsize=(26, 1.0 + 0.55 * len(models)))
    ax.grid(False)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                dark = (v - vmin) / (vmax - vmin) > 0.6 if cmap == "Blues" else abs(v) > 0.6 * max(abs(vmin), abs(vmax))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(len(strings)))
    ax.set_xticklabels([s + ("*" if s in train_strings else "") for s in strings], fontsize=8, family="monospace")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in models], fontsize=9)
    bounds = np.cumsum([O.EXPECTED_GROUP_COUNTS[g] for g in O.GROUP_ORDER])[:-1]
    for b in bounds:
        ax.axvline(b - 0.5, color="#0b0b0b", lw=1.5)
    start = 0
    for g in O.GROUP_ORDER:
        n = O.EXPECTED_GROUP_COUNTS[g]
        ax.text(start + n / 2 - 0.5, -0.65, O.GROUP_LABELS[g], ha="center", va="bottom", fontsize=9, weight="bold")
        start += n
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    ax.set_title(title or "Per-string test exact-grid accuracy (* = seen training string; strings applied left to right)",
                 pad=36)
    import pandas as pd
    df = pd.DataFrame(mat, columns=strings)
    df.insert(0, "model", models)
    return save_fig(fig, out, name, df)


# ---------------------------------------------------------------------------
# (d) examples
# ---------------------------------------------------------------------------

def draw_grid(ax, grid, title=None, errors=None):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    g = np.asarray(grid).reshape(6, 6).astype(int)
    shown = np.where(g < 0, 10, g)
    ax.imshow(shown, cmap=ListedColormap(DIGIT_COLORS + [INVALID_COLOR]), vmin=-0.5, vmax=10.5)
    for r in range(6):
        for c in range(6):
            v = g[r, c]
            txt = "?" if v < 0 else str(v)
            light_bg = v in (0, 4, 5, 6, 8) and v >= 0
            ax.text(c, r, txt, ha="center", va="center", fontsize=7, color="#0b0b0b" if light_bg else "white")
            if errors is not None and errors[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#ff0000", lw=2.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, fontweight="normal")


def plot_shared_examples(exp_dir: Path, out: Path):
    import pandas as pd
    plt = _plt()
    models = available_models(exp_dir)
    if not models:
        return None
    preds = {m: load_predictions(exp_dir / "eval" / m / "test_predictions.npz") for m in models}
    ref = preds[models[0]]
    sel = [int(np.nonzero((ref["grid_id"] == g) & (ref["string"] == s))[0][0])
           for g in SHARED_EXAMPLE_GRID_IDS for s in SHARED_EXAMPLE_STRINGS
           if np.any((ref["grid_id"] == g) & (ref["string"] == s))]
    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(sel), ncol, figsize=(1.55 * ncol, 1.75 * len(sel)), squeeze=False)
    rows = []
    for i, k in enumerate(sel):
        tgt = ref["target"][k]
        draw_grid(axes[i, 0], ref["input"][k], f"input (grid {ref['grid_id'][k]})")
        draw_grid(axes[i, 1], tgt, f"target: {ref['string'][k]}")
        for j, m in enumerate(models):
            p = preds[m]
            assert p["grid_id"][k] == ref["grid_id"][k] and p["string"][k] == ref["string"][k]
            err = p["pred"][k] != tgt
            draw_grid(axes[i, 2 + j], p["pred"][k], f"{DISPLAY_NAMES[m]}\n{'CORRECT' if p['exact'][k] else f'{int(err.sum())} wrong'}", err)
            rows.append({"model": m, "grid_id": int(ref["grid_id"][k]), "string": str(ref["string"][k]),
                         "exact": bool(p["exact"][k]), "cell_correct": int(p["cell_correct"][k])})
    fig.suptitle("Predetermined shared test examples (grid 0; strings fixed in advance). Red boxes = wrong cells; "
                 "black '?' = invalid AR token", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, out, "d_shared_examples", pd.DataFrame(rows))


def plot_failure_examples(exp_dir: Path, out: Path, per_model: int = 3):
    """Additional FAILURE examples: the first failing test examples of each model in manifest
    order (grid-major), restricted to strings of length 1 and 3 -- a deterministic rule,
    clearly labeled, not chosen to favor any model."""
    import pandas as pd
    plt = _plt()
    models = available_models(exp_dir)
    if not models:
        return None
    fig, axes = plt.subplots(len(models), 3 * per_model, figsize=(1.6 * 3 * per_model, 1.85 * len(models)), squeeze=False)
    rows = []
    for i, m in enumerate(models):
        p = load_predictions(exp_dir / "eval" / m / "test_predictions.npz")
        lens = np.char.str_len(p["string"].astype(str))
        fails = np.nonzero(~p["exact"] & (lens != 2))[0][:per_model]
        for j in range(per_model):
            axs = axes[i, 3 * j:3 * j + 3]
            if j >= len(fails):
                for a in axs:
                    a.axis("off")
                axs[0].text(0.5, 0.5, "no further\nfailure", ha="center", va="center", fontsize=8, transform=axs[0].transAxes)
                continue
            k = fails[j]
            draw_grid(axs[0], p["input"][k], f"{DISPLAY_NAMES[m]}\ninput g{p['grid_id'][k]}")
            draw_grid(axs[1], p["target"][k], f"target {p['string'][k]}")
            draw_grid(axs[2], p["pred"][k], f"prediction ({36 - int(p['cell_correct'][k])} wrong)", p["pred"][k] != p["target"][k])
            rows.append({"model": m, "grid_id": int(p["grid_id"][k]), "string": str(p["string"][k]),
                         "cell_correct": int(p["cell_correct"][k]), "valid": bool(p["valid"][k])})
    fig.suptitle("Additional FAILURE examples: first failures per model in manifest order (length-1/3 strings only)",
                 fontsize=10)
    fig.tight_layout()
    return save_fig(fig, out, "d_failure_examples", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# (e) resource table, (f) numerical summary
# ---------------------------------------------------------------------------

def resource_table(exp_dir: Path, out: Path):
    import pandas as pd
    rows = []
    for m in MODEL_IDS:
        s = model_summary(m)
        row = {
            "model": m, "display_name": DISPLAY_NAMES[m], "kind": s["kind"],
            "unique_learned_params": s["unique_params"],
            "shared_core_params": s["shared_core_params"],
            "untied_equivalent_params": s["untied_equivalent_params"],
            "param_note": ("weight-tied: ~1.8M unique (NOT a 10M model); untied counterpart " + TIED_PAIRS[m])
            if m in TIED_PAIRS else "",
            "block_applications_per_forward": s["block_applications_per_forward"],
            "core_calls_per_forward": s["core_calls_per_forward"],
            "grad_block_applications_per_forward": s.get("grad_block_applications_per_forward",
                                                         s["block_applications_per_forward"]),
            "input_tokens": s["input_tokens"], "decoded_output_steps": s["decoded_output_steps"],
            "training_objective": s["loss_label"],
        }
        tsf = exp_dir / "runs" / m / "train_summary.json"
        if tsf.exists():
            t = json.loads(tsf.read_text())
            row.update({k: t.get(k) for k in ("optimizer_steps", "examples_seen", "train_seconds",
                                              "peak_gpu_memory_gb_this_session", "hardware", "dtype",
                                              "microbatch_size", "accumulation_steps", "oom_retries",
                                              "best_step", "repo_commit")})
        esf = exp_dir / "eval" / m / "eval_summary.json"
        if esf.exists():
            e = json.loads(esf.read_text())
            row["eval_seconds"] = e["val_eval_seconds"] + e["test_eval_seconds"]
            row["peak_gpu_memory_gb_eval"] = e.get("peak_gpu_memory_gb_eval")
        rows.append(row)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e_resource_table.csv", index=False)
    (out / "e_resource_table.md").write_text(_to_markdown(df))
    return df


def _to_markdown(df) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("N/A" if math.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


FIT_THRESHOLD = 0.90


def numerical_summary(exp_dir: Path, out: Path, train_strings: Sequence[str]):
    import pandas as pd
    models, ps, gr = load_tables(exp_dir)
    if gr is None:
        return None
    rows = []
    for m in models:
        g = gr[gr["model"] == m].set_index("group")
        ev = json.loads((exp_dir / "eval" / m / "eval_summary.json").read_text())
        row = {"model": m, "display_name": DISPLAY_NAMES[m],
               "val_seen_exact (full val)": ev["val"]["exact_acc"],
               "test_pairs_seen_exact": g.loc["pairs_seen", "exact_acc"],
               "test_pairs_unseen_exact": g.loc["pairs_unseen", "exact_acc"],
               "test_singles_exact": g.loc["singles", "exact_acc"],
               "test_triples_all_distinct_exact": g.loc["triples_all_distinct", "exact_acc"],
               "test_triples_two_types_exact": g.loc["triples_two_types", "exact_acc"],
               "test_triples_one_type_exact": g.loc["triples_one_type", "exact_acc"],
               "test_triples_all_exact": g.loc["triples_all", "exact_acc"],
               "test_pairs_seen_cell_acc": g.loc["pairs_seen", "cell_acc"],
               "selected_step": ev["selected_step"]}
        fit = row["test_pairs_seen_exact"]
        row["seen_pair_fit"] = "adequate" if fit >= FIT_THRESHOLD else "INADEQUATE"
        rows.append(row)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "f_numerical_summary.csv", index=False)
    lines = ["# Seen-pair fit vs. single/triple generalization", "",
             f"Experiment directory: `{exp_dir}`. Training strings: {', '.join(train_strings)}. One seed (42); "
             "no error bars or significance claims. Exact-grid accuracy, macro-averaged over strings; AR models "
             "use free-running greedy decoding. 'N/A' = empty group.", "",
             f"A model whose test seen-pair exact accuracy is below {FIT_THRESHOLD:.2f} is flagged INADEQUATE: its "
             "poor single/triple accuracy should NOT be interpreted as a compositionality finding.", "",
             "| model | val seen | test seen pairs | unseen pairs | singles | triples 3-distinct | triples 2-types | "
             "triples 1-type | seen-pair fit |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['display_name']} | {fmt(r['val_seen_exact (full val)'])} | {fmt(r['test_pairs_seen_exact'])} | "
                     f"{fmt(r['test_pairs_unseen_exact'])} | {fmt(r['test_singles_exact'])} | "
                     f"{fmt(r['test_triples_all_distinct_exact'])} | {fmt(r['test_triples_two_types_exact'])} | "
                     f"{fmt(r['test_triples_one_type_exact'])} | {r['seen_pair_fit']} |")
    text = "\n".join(lines) + "\n"
    (out / "f_numerical_summary.md").write_text(text)
    return df, text


def build_reports(exp_dir: Path, train_strings: Sequence[str], log=print) -> dict:
    exp_dir = Path(exp_dir)
    fig_dir, tab_dir = exp_dir / "figures", exp_dir / "tables"
    res = {"training_curves": plot_training_curves(exp_dir, fig_dir)}
    models, ps, gr = load_tables(exp_dir)
    res["resource_table"] = resource_table(exp_dir, tab_dir)
    if ps is not None:
        ps.to_csv(tab_dir / "test_per_string_all_models.csv", index=False)
        gr.to_csv(tab_dir / "test_groups_all_models.csv", index=False)
        res["group_exact"] = plot_group_bars(gr, fig_dir, "exact_acc", "b_group_exact_accuracy")
        res["group_cell"] = plot_group_bars(gr, fig_dir, "cell_acc", "b_group_cell_accuracy")
        res["heatmap"] = plot_string_heatmap(ps, fig_dir, train_strings)
        res["shared_examples"] = plot_shared_examples(exp_dir, fig_dir)
        res["failure_examples"] = plot_failure_examples(exp_dir, fig_dir)
        res["summary"] = numerical_summary(exp_dir, tab_dir, train_strings)
        log(res["summary"][1])
    else:
        log("No completed evaluations yet: only training curves/resource table were produced.")
    log(f"Figures: {fig_dir}\nTables: {tab_dir}")
    return res
