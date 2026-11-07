"""Figures/tables/summary/archive for ONE mixed-length experiment directory (training + evaluation).
Every figure is saved as PNG + PDF with a CSV of the plotted numbers. One seed: no error bars."""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from . import mixed as M
from .names import DISPLAY_NAMES, MODEL_IDS, TIED_PAIRS
from .plotstyle import MODEL_STYLE, _plt, bar_style, save_fig

FIT_THRESHOLD = 0.90


def _models_with(exp_dir: Path, rel: str):
    return [m for m in MODEL_IDS if (exp_dir / rel.format(m=m)).exists()]


def load_eval_tables(exp_dir: Path):
    import pandas as pd
    ms = _models_with(exp_dir, "eval/{m}/EVAL_COMPLETE.json")
    if not ms:
        return ms, None, None
    pw = pd.concat([pd.read_csv(exp_dir / "eval" / m / "test_per_word.csv", keep_default_na=False, na_values=[""])
                    for m in ms], ignore_index=True)
    gr = pd.concat([pd.read_csv(exp_dir / "eval" / m / "test_groups.csv", keep_default_na=False, na_values=[""])
                    for m in ms], ignore_index=True)
    return ms, pw, gr


def plot_length_curves(exp_dir: Path, out: Path):
    import pandas as pd
    plt = _plt()
    ms = _models_with(exp_dir, "runs/{m}/val_log.csv")
    if not ms:
        return None
    fig, axes = plt.subplots(2, 5, figsize=(24, 8), squeeze=False, sharex=True)
    rows = []
    for m in ms:
        v = pd.read_csv(exp_dir / "runs" / m / "val_log.csv")
        st = MODEL_STYLE[m]
        for j, L in enumerate(M.LENGTHS):
            for i, (col, lab) in enumerate(((f"val_cell_ce_len{L}", "validation cell CE"),
                                            (f"subset_exact_len{L}", "validation-SUBSET exact accuracy"))):
                if col in v:
                    axes[i, j].plot(v["step"], v[col], color=st["color"], ls=st["ls"], lw=1.4, marker="o", ms=2.5,
                                    label=DISPLAY_NAMES[m])
                    rows += [{"model": m, "length": L, "measure": lab, "step": s, "value": x} for s, x in zip(v["step"], v[col])]
    n_sub = None
    for m in ms:
        v = pd.read_csv(exp_dir / "runs" / m / "val_log.csv")
        n_sub = int(v["n_subset_examples"].iloc[0]) // len(M.LENGTHS)
    for j, L in enumerate(M.LENGTHS):
        axes[0, j].set(title=f"length {L}{' (EMPTY)' if L == 0 else ''}: val cell CE", yscale="log")
        axes[1, j].set(title=f"length {L}: SUBSET exact acc (n={n_sub}/length)", xlabel="Optimizer step", ylim=(-0.02, 1.02))
    axes[0, 0].set_ylabel("cell CE (AR: teacher-forced digits)")
    axes[1, 0].set_ylabel("exact accuracy (AR free-running)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Per-length validation curves (lengths 0/2/3/4/5 only; length 1 is never validated)")
    fig.tight_layout()
    return save_fig(fig, out, "a2_validation_by_length", pd.DataFrame(rows))


def plot_singles(gr, out: Path, exp_label: str):
    import pandas as pd
    plt = _plt()
    models = [m for m in MODEL_IDS if m in set(gr.model)]
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5), gridspec_kw={"width_ratios": [3, 3, 1.4]})
    rows = []
    for ax, metric, lab in ((axes[0], "exact_acc", "exact-grid accuracy"), (axes[1], "cell_acc", "cell accuracy")):
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            g = gr[gr.model == m].set_index("group")
            keys = ["single_A", "single_B", "single_C"]
            vals = [g.loc[k, metric] for k in keys] + [np.mean([g.loc[k, metric] for k in keys])]
            xs = np.arange(4) + (i - (len(models) - 1) / 2) * w
            ax.bar(xs, vals, width=w * 0.92, label=DISPLAY_NAMES[m], **bar_style(m))
            if metric == "exact_acc":
                for x, k in zip(xs[:3], keys):
                    ax.text(x, g.loc[k, metric] + 0.01, f"{int(g.loc[k, 'exact_correct'])}/{int(g.loc[k, 'n'])}",
                            rotation=90, ha="center", va="bottom", fontsize=6)
            rows += [{"model": m, "metric": metric, "word": k, "value": v} for k, v in zip(keys + ["macro"], vals)]
        ax.set_xticks(range(4)); ax.set_xticklabels(["A", "B", "C", "macro A/B/C"])
        ax.set(ylim=(0, 1.12), ylabel=lab, title=f"{exp_label}: PRIMARY single-operation {lab} (never trained)")
    axes[0].legend(fontsize=7, ncol=3)
    ax = axes[2]
    for i, m in enumerate(models):
        g = gr[gr.model == m].set_index("group")
        ax.barh(i, g.loc["EMPTY", "exact_acc"], **bar_style(m))
        ax.text(min(g.loc["EMPTY", "exact_acc"], 0.6) + 0.02, i, f"{int(g.loc['EMPTY', 'exact_correct'])}/{int(g.loc['EMPTY', 'n'])}",
                va="center", fontsize=7)
        rows.append({"model": m, "metric": "exact_acc", "word": "EMPTY", "value": g.loc["EMPTY", "exact_acc"]})
    ax.set_yticks(range(len(models))); ax.set_yticklabels([DISPLAY_NAMES[m] for m in models], fontsize=7)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.05), title="EMPTY (identity / copy)\nexact accuracy (separate)")
    fig.tight_layout()
    return save_fig(fig, out, "b1_singles_primary_and_empty", pd.DataFrame(rows))


def plot_by_length(gr, out: Path, exp_label: str):
    import pandas as pd
    plt = _plt()
    models = [m for m in MODEL_IDS if m in set(gr.model)]
    present = set(gr[gr["n"] > 0]["group"])
    groups = [g for g in M.MIXED_GROUP_ORDER if g in present]   # drop categories that are empty by design
    fig, axes = plt.subplots(2, 1, figsize=(20, 9))
    rows = []
    for ax, metric in zip(axes, ("exact_acc", "cell_acc")):
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            g = gr[gr.model == m].set_index("group")
            vals = [g.loc[k, metric] if k in g.index else np.nan for k in groups]
            xs = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * w
            ax.bar(xs, np.nan_to_num(vals), width=w * 0.92, label=DISPLAY_NAMES[m], **bar_style(m))
            for x, v in zip(xs, vals):
                if isinstance(v, float) and math.isnan(v):
                    ax.text(x, 0.02, "N/A", rotation=90, fontsize=7, ha="center")
            rows += [{"model": m, "metric": metric, "group": k, "value": v} for k, v in zip(groups, vals)]
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([M.MIXED_GROUP_LABELS[k].replace("Singles A/B/C (never trained)", "Length 1 = singles A/B/C\n"
                                                             "(HELD OUT: never trained)") for k in groups], fontsize=8)
        ax.set(ylim=(0, 1.08), ylabel=metric.replace("_", " "),
               title=f"{exp_label}: test {metric.replace('_', ' ')} by effective length / seen status "
                     "(lengths 3-5 are SEEN strings on NEW grids)")
    axes[0].legend(fontsize=7, ncol=5)
    fig.tight_layout()
    return save_fig(fig, out, "b2_accuracy_by_length_and_seen_status", pd.DataFrame(rows))


def plot_unseen_pairs(pw, out: Path, experiment: str, exp_label: str):
    import pandas as pd
    plt = _plt()
    sub = pw[pw.group == "pairs_unseen"]
    if not len(sub):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{exp_label}: AA/BB/CC are TRAINED here -> unseen-pair category is N/A (not 0)",
                ha="center", va="center")
        return save_fig(fig, out, "b3_unseen_pairs_AA_BB_CC", pd.DataFrame([{"experiment": experiment, "unseen_pairs": "N/A"}]))
    models = [m for m in MODEL_IDS if m in set(sub.model)]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        s = sub[sub.model == m].set_index("string")
        vals = [s.loc[k, "exact_acc"] for k in ("AA", "BB", "CC")]
        xs = np.arange(3) + (i - (len(models) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, label=DISPLAY_NAMES[m], **bar_style(m))
        for x, k in zip(xs, ("AA", "BB", "CC")):
            ax.text(x, s.loc[k, "exact_acc"] + 0.01, f"{int(s.loc[k, 'exact_correct'])}/{int(s.loc[k, 'n'])}",
                    rotation=90, fontsize=6, ha="center", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels(["AA (unseen)", "BB (unseen)", "CC (unseen; = identity)"])
    ax.set(ylim=(0, 1.15), ylabel="exact-grid accuracy", title=f"{exp_label}: unseen length-two pairs")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return save_fig(fig, out, "b3_unseen_pairs_AA_BB_CC", sub)


def resource_table(exp_dir: Path, out: Path, sequence_pe: str = None, models=None):
    import dataclasses
    import pandas as pd
    from .models.common import count_parameters
    from .models.registry import build_model, default_config
    rows = []
    for m in (models or MODEL_IDS):
        cfg = dataclasses.asdict(default_config(m))
        cfg["n_slots"] = M.N_SLOTS
        if sequence_pe and m not in M.CNN_MODELS:
            cfg["sequence_pe"] = sequence_pe
        model = build_model(m, cfg)
        n = count_parameters(model)
        info = model.cost_info()
        row = {"model": m, "display_name": DISPLAY_NAMES[m], "n_slots": M.N_SLOTS,
               "sequence_pe": cfg.get("sequence_pe", "n/a (CNN)" if m in M.CNN_MODELS else "learned"),
               "unique_learned_params": n,
               "shared_core_params": model.shared_core_parameters(), "untied_equivalent_params": model.untied_equivalent_parameters(),
               "param_note": "weight-tied (~1.8M unique; NOT a 10M model)" if m in TIED_PAIRS else "",
               **{k: info[k] for k in ("block_applications_per_forward", "core_calls_per_forward", "input_tokens",
                                       "decoded_output_steps")}, "training_objective": model.loss_label}
        del model
        tsf = exp_dir / "runs" / m / "train_summary.json"
        if tsf.exists():
            t = json.loads(tsf.read_text())
            row.update({k: t.get(k) for k in ("optimizer_steps", "examples_seen", "train_seconds",
                                              "peak_gpu_memory_gb_this_session", "hardware", "dtype", "microbatch_size",
                                              "accumulation_steps", "oom_retries", "best_step", "repo_commit")})
        esf = exp_dir / "eval" / m / "eval_summary.json"
        if esf.exists():
            e = json.loads(esf.read_text())
            row["test_eval_seconds"] = e.get("test_eval_seconds")
            row["peak_gpu_memory_gb_eval"] = e.get("peak_gpu_memory_gb_eval")
        rows.append(row)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e_resource_table.csv", index=False)
    return df


def summary_table(exp_dir: Path, out: Path, experiment: str):
    import pandas as pd
    ms, pw, gr = load_eval_tables(exp_dir)
    if gr is None:
        return None, "No completed evaluations yet."
    rows = []
    for m in ms:
        g = gr[gr.model == m].set_index("group")
        ev = json.loads((exp_dir / "eval" / m / "eval_summary.json").read_text())

        def cnt(k):
            return f"{int(g.loc[k, 'exact_correct'])}/{int(g.loc[k, 'n'])}" if k in g.index and g.loc[k, "n"] else "N/A"

        fit = min(g.loc[k, "exact_acc"] for k in ("pairs_seen", "len3_seen", "len4_seen", "len5_seen") if g.loc[k, "n"])
        rows.append({"model": DISPLAY_NAMES[m], "selected_step": ev["selected_step"],
                     "A": cnt("single_A"), "B": cnt("single_B"), "C": cnt("single_C"),
                     "singles_macro_exact": ev["test"]["singles_macro_exact"], "EMPTY": cnt("EMPTY"),
                     "seen_pairs": cnt("pairs_seen"), "unseen_pairs": cnt("pairs_unseen"),
                     "len3": cnt("len3_seen"), "len4": cnt("len4_seen"), "len5": cnt("len5_seen"),
                     "overall_micro": ev["test"]["overall_micro_exact (record-weighted)"],
                     "overall_length_macro": ev["test"]["overall_length_macro_exact (lengths 0..5 equal weight)"],
                     "seen_support_fit": "adequate" if fit >= FIT_THRESHOLD else "INADEQUATE"})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "f_summary.csv", index=False)
    from .diagnosis.report import _md
    text = "\n".join([
        f"# {experiment}: singles (never trained) vs seen support", "",
        "Integer exact successes / records on the held-out test grids. Lengths 3-5 are SEEN strings on NEW grids "
        "(not unseen-combination generalization). Overall micro = record-weighted over 364 words (dominated by length 5); "
        "length-macro = lengths 0..5 equally weighted. One seed; no error bars.", "",
        f"seen_support_fit = INADEQUATE if any seen category (pairs, len3, len4, len5) is below {FIT_THRESHOLD}: "
        "single-operation failure of such a model is NOT a clean decomposition failure.", "", _md(df)])
    (out / "f_summary.md").write_text(text)
    return df, text


def build_mixed_reports(exp_dir: Path, experiment: str, log=print) -> dict:
    from .reports import plot_training_curves, plot_string_heatmap
    exp_dir = Path(exp_dir)
    fig, tab = exp_dir / "figures", exp_dir / "tables"
    label = {"mixed_02345_distinct_pairs": "Mixed 0/2-5, distinct pairs",
             "mixed_02345_all_pairs": "Mixed 0/2-5, all pairs",
             "compact_02345_rope": "Compact 0/2-5, RoPE (seed 42, best ckpt)",
             "compact_02345_sinusoidal": "Compact 0/2-5, sinusoidal (seed 42, best ckpt)",
             "compact_02345_cnn": "Compact 0/2-5, shared CNN controls (no PE switch)"}.get(experiment, experiment)
    pe = M.COMPACT_EXPERIMENTS.get(experiment) if M.is_compact(experiment) else None
    models = (M.CNN_MODELS if experiment == "compact_02345_cnn" else M.ATTENTION_MODELS) if M.is_compact(experiment) else None
    res = {"training_curves": plot_training_curves(exp_dir, fig), "length_curves": plot_length_curves(exp_dir, fig),
           "resource_table": resource_table(exp_dir, tab, pe, models)}
    ms, pw, gr = load_eval_tables(exp_dir)
    if gr is not None:
        pw.to_csv(tab / "test_per_word_all_models.csv", index=False)
        gr.to_csv(tab / "test_groups_all_models.csv", index=False)
        res["singles"] = plot_singles(gr, fig, label)
        res["by_length"] = plot_by_length(gr, fig, label)
        if not M.is_compact(experiment):      # every pair is trained in the compact experiments
            res["unseen_pairs"] = plot_unseen_pairs(pw, fig, experiment, label)
        df, text = summary_table(exp_dir, tab, experiment)
        res["summary"] = (df, text)
        log(text)
    else:
        log("No completed evaluations yet: only training curves/resource table were produced.")
    return res


def make_compact_archive(exp_dir: Path, log=print) -> Path:
    """Everything except checkpoints, evaluation chunks and analysis caches."""
    exp_dir = Path(exp_dir)
    dst = exp_dir / f"results_{exp_dir.name}.zip"
    skip_dirs = {"chunks", "cache", "selfcheck"}
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(exp_dir.rglob("*")):
            rel = p.relative_to(exp_dir)
            if (p.is_file() and p.suffix not in (".pt", ".pkl") and p != dst and not p.name.endswith(".zip")
                    and ".tmp" not in p.name and not (set(rel.parts) & skip_dirs)):
                zf.write(p, Path(exp_dir.name) / rel)
    log(f"Compact archive: {dst} ({dst.stat().st_size / 2**20:.1f} MiB)")
    return dst
