"""Audits, singleton diagnostics, figures and the cross-format comparison for notebooks 10/11.

All numbers are read from saved artifacts (located through artifact_index.json). One seed: descriptive only.
Singleton diagnostics reuse the verified 288-transformation analysis (diagnosis.analysis.analyze_run) on each
(condition, slot) singleton population; identity placements are analysed separately and never pooled into
independent observations.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import idvar as V
from . import operations as O
from .names import DISPLAY_NAMES, MODEL_IDS
from .plotstyle import _plt, bar_style, save_fig

CONDS = ("identity", "variable_length")


# ---------------------------------------------------------------------------
# Pre-training audits (no training data consumed)
# ---------------------------------------------------------------------------

def model_config(model_id: str, condition: str) -> dict:
    from .models.registry import default_config
    c = dataclasses.asdict(default_config(model_id))
    c["n_slots"] = 5
    c["op_format"] = "variable" if condition == "variable_length" else "fixed"
    if "identity_token" in c:
        c["identity_token"] = True
    if "conditioner" in c:
        c["conditioner"] = "set_mlp"
    return c


def init_audit(models: List[str], seed: int = 42):
    """Same trainable tensor names/shapes/values and counts across the two formats, per architecture."""
    import pandas as pd
    import torch
    from .models.registry import build_model
    rows = []
    for m in models:
        sd = {}
        for cond in CONDS:
            mod = build_model(m, model_config(m, cond), seed=seed)
            sd[cond] = {k: v.detach().clone() for k, v in mod.named_parameters()}
            info = mod.cost_info()
        same = sd["identity"].keys() == sd["variable_length"].keys() and all(
            torch.equal(sd["identity"][k], sd["variable_length"][k]) for k in sd["identity"])
        h = hashlib.sha256(b"".join(v.numpy().tobytes() for v in sd["identity"].values())).hexdigest()[:16]
        rows.append({"model": m, "display_name": DISPLAY_NAMES[m],
                     "unique_params": sum(v.numel() for v in sd["identity"].values()),
                     "identical_initial_tensors_across_formats": bool(same), "init_sha256": h,
                     "block_applications_per_forward": info["block_applications_per_forward"]})
    return pd.DataFrame(rows)


def allocation_audit(seed: int = 42, B: int = 256, steps: int = 5000):
    import pandas as pd
    s = V.IdVarStream(seed, B, "identity")
    ks = np.concatenate([s.ks_for_step(t) for t in range(5)])
    counts = {int(k): int((ks == k).sum()) for k in range(6)}
    rows = [{"k": k, "records_in_first_5_updates": counts.get(k, 0),
             "records_full_run": counts.get(k, 0) * steps // 5,
             "five_slot_patterns": V.pattern_counts()[k], "status": "HELD OUT" if k == 1 else "trained"}
            for k in range(6)]
    df = pd.DataFrame(rows)
    assert df.records_full_run.sum() == B * steps and counts.get(1, 0) == 0
    return df


def shape_audit(models: List[str]):
    """Actual operation-token lengths, sequence lengths and (AR) answer spans for k = 0, 1, 2, 5 per family/format."""
    import pandas as pd
    import torch
    from . import tokenization as T
    from .models.registry import build_model
    rows = []
    for m in models:
        for cond in CONDS:
            mod = build_model(m, model_config(m, cond), seed=0).eval()
            for k in (0, 1, 2, 5):
                word = "ABCAB"[:k]
                slots5 = V.slots5_for(word, tuple(range(k)))[None]
                ops = torch.as_tensor(V.serialize_ops(slots5, cond))
                b = {"ops": ops, "x": torch.zeros(1, 6, 6, dtype=torch.long), "y": torch.zeros(1, 6, 6, dtype=torch.long)}
                seen = {}
                target = getattr(mod, "stack", None) or (mod.core[0] if hasattr(mod, "core") else None)
                hook = target.register_forward_pre_hook(lambda mod_, args: seen.setdefault("L", args[0].shape[1])) \
                    if target is not None else None
                with torch.no_grad():
                    mod.val_cell_ce(b)
                if hook:
                    hook.remove()
                r = {"model": m, "condition": cond, "k": k, "op_tokens_input": int(ops.shape[1]),
                     "op_token_ids": " ".join(map(str, T.op_slots_to_tokens(ops)[0].tolist())),
                     "backbone_sequence_length": seen.get("L", "n/a (CNN: 6x6 spatial; conditioner sums "
                                                                f"{int(ops.shape[1])} tokens)")}
                if m in ("transformer_ar", "looped_transformer_ar"):
                    lay = T.ARLayout(int(ops.shape[1]))
                    r.update({"teacher_forced_tokens": lay.length, "output_index": lay.idx_output,
                              "answer_span": f"{lay.out_start}..{lay.out_start + 35}", "eos_index": lay.idx_eos,
                              "positions": f"0..{lay.length - 1} (contiguous)"})
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loading through the artifact index
# ---------------------------------------------------------------------------

def load_predictions(root: Path, condition: str, model: str) -> Optional[Dict[str, np.ndarray]]:
    from .idvar_pipeline import build_artifact_index
    idx = build_artifact_index(root)
    a = idx.get(f"{condition}/{model}")
    if not a or not a.get("complete"):
        return None
    with np.load(a["predictions"]) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Singleton diagnostics
# ---------------------------------------------------------------------------

def singles_spec():
    from .diagnosis.spec import DiagSpec
    exps = ["variable_length"] + [f"identity@slot{p}" for p in range(1, 6)]
    return DiagSpec(
        name="idvar_singles", experiments=exps, words=["A", "B", "C"], common_panel=["A", "B", "C"],
        panel_groups=[("Singletons (held out)", 3)], group_order=["singles"],
        group_labels={"singles": "Singletons A/B/C (held out)"}, train_labels=lambda e: V.training_labels(),
        pair_words=lambda e: list(O.TRAIN_SETS["all_pairs"]), group_fn=lambda e, s: "singles",
        populations={"singletons": ["singles"]}, controls_words=["A", "B", "C"], n_slots=5)


def singleton_diagnostics(root: Path, models: List[str], split_seed: int = 20251106, log=print):
    """Per (condition, slot, model): 288-transform matching, explicit hypotheses with union counts, FIT/DIAG fixed
    transform and source maps; plus identity placement consistency. Missing runs are reported, not zero-filled."""
    import pandas as pd
    from .diagnosis import group as G
    from .diagnosis.analysis import analyze_run, diagnostic_split, hypothesis_population_summary
    from .diagnosis.artifacts import Run
    spec = singles_spec()
    cat = G.Catalog()
    out = {k: [] for k in ("per_string", "hypotheses", "hypothesis_summary", "stable_rules", "source_maps",
                           "consistency", "status")}
    for cond in CONDS:
        for m in models:
            a = load_predictions(root, cond, m)
            if a is None:
                out["status"].append({"condition": cond, "model": m, "status": "not yet available"})
                continue
            out["status"].append({"condition": cond, "model": m, "status": "analysed"})
            single = a["kind"] == "single"
            split = diagnostic_split(np.unique(a["grid_id"]), split_seed)
            positions = [0] if cond == "variable_length" else [1, 2, 3, 4, 5]
            for p in positions:
                sel = single & (a["position"] == p)
                pseudo = "variable_length" if cond == "variable_length" else f"identity@slot{p}"
                r = Run(pseudo, m, "original", {}, a["input"][sel].reshape(-1, 36).astype(np.int16),
                        a["target"][sel].reshape(-1, 36).astype(np.int16), a["pred"][sel].reshape(-1, 36).astype(np.int16),
                        a["grid_id"][sel].astype(np.int64), a["string"][sel].astype(str), {}, {"file_sha256": {}})
                res = analyze_run(r, cat, spec, split)
                for key in ("per_string", "hypotheses", "stable_rules", "source_maps"):
                    df = res[key].copy()
                    if len(df):
                        df.insert(0, "position", p if p else "n/a")
                        df.insert(0, "condition", cond)
                        out[key].append(df)
                hs = hypothesis_population_summary(res["hypothesis_types"], res["per_string"], spec)
                hs.insert(0, "position", p if p else "n/a")
                hs.insert(0, "condition", cond)
                out["hypothesis_summary"].append(hs)
            if cond == "identity":
                for op in "ABC":
                    P = []
                    for p in range(1, 6):
                        s = single & (a["string"] == op) & (a["position"] == p)
                        o = np.argsort(a["grid_id"][s])
                        P.append(a["pred"][s][o].reshape(-1, 36))
                    P = np.stack(P)                                    # [5, grids, 36]
                    same_all = (P == P[0:1]).all(axis=(0, 2))
                    pair = [float((P[i] == P[j]).all(1).mean()) for i in range(5) for j in range(i + 1, 5)]
                    out["consistency"].append({"model": m, "op": op, "n_grids": P.shape[1],
                                               "grids_identical_across_all_5_placements": int(same_all.sum()),
                                               "rate_identical_all_5": float(same_all.mean()),
                                               "mean_pairwise_identical_rate": float(np.mean(pair))})
    res = {k: (pd.concat(v, ignore_index=True) if v and isinstance(v[0], pd.DataFrame) else pd.DataFrame(v))
           for k, v in out.items()}
    d = Path(root) / "comparisons" / "singleton_diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        v.to_csv(d / f"{k}.csv", index=False)
    log(f"Singleton diagnostics written to {d}")
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _read(root: Path, cond: str, name: str):
    import pandas as pd
    fs = sorted((Path(root) / cond / "eval").glob(f"*/{name}.csv"))
    return pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in fs], ignore_index=True) if fs else None


def plot_by_k(root: Path, out: Path):
    plt = _plt()
    paths = []
    for cond in CONDS:
        t = _read(root, cond, "test_by_k")
        if t is None:
            continue
        models = [m for m in MODEL_IDS if m in set(t.model)]
        fig, ax = plt.subplots(figsize=(16, 5))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            s = t[t.model == m].set_index("k")
            xs = np.arange(6) + (i - (len(models) - 1) / 2) * w
            ax.bar(xs, [s.loc[k, "exact_acc"] if k in s.index else np.nan for k in range(6)], width=w * 0.92,
                   label=DISPLAY_NAMES[m], **bar_style(m))
        ax.axvspan(0.5, 1.5, color="#e34948", alpha=0.08)
        ax.set_xticks(range(6)); ax.set_xticklabels(["k=0 EMPTY", "k=1 HELD OUT\n(singletons)", "k=2", "k=3", "k=4", "k=5"])
        ax.set(ylim=(0, 1.05), ylabel="exact-grid accuracy",
               title=f"{cond}: exact accuracy by number of A/B/C tokens (seed 42, best ckpt; k>=2 = seen words on new grids)")
        ax.legend(fontsize=7, ncol=5)
        fig.tight_layout()
        paths += save_fig(fig, out, f"by_k_{cond}", t)
    return paths


def plot_singletons(root: Path, out: Path):
    import pandas as pd
    plt = _plt()
    paths = []
    idt, var = _read(root, "identity", "test_singletons"), _read(root, "variable_length", "test_singletons")
    # (a) op x position heatmaps for identity
    if idt is not None:
        models = [m for m in MODEL_IDS if m in set(idt.model)]
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), squeeze=False)
        for ax, m in zip(axes.flat, models):
            s = idt[idt.model == m]
            mat = np.array([[s[(s.string == op) & (s.position == p)]["exact_acc"].iloc[0] for p in range(1, 6)] for op in "ABC"])
            cnt = np.array([[int(s[(s.string == op) & (s.position == p)]["exact_correct"].iloc[0]) for p in range(1, 6)]
                            for op in "ABC"])
            ax.grid(False)
            ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(5):
                    ax.text(j, i, f"{cnt[i, j]}", ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > 0.6 else "#0b0b0b")
            ax.set_xticks(range(5)); ax.set_xticklabels([f"slot {p}" + (" (canonical)" if p == 1 else "") for p in range(1, 6)],
                                                        fontsize=7)
            ax.set_yticks(range(3)); ax.set_yticklabels(list("ABC"))
            ax.set_title(DISPLAY_NAMES[m], fontsize=9)
        for ax in list(axes.flat)[len(models):]:
            ax.axis("off")
        fig.suptitle("identity: singleton exact successes per operation x placement (of 1000 grids each; other 4 slots = <identity>)")
        fig.tight_layout()
        paths += save_fig(fig, out, "singletons_identity_op_by_position", idt)
    # (b) variable three-row panel
    if var is not None:
        models = [m for m in MODEL_IDS if m in set(var.model)]
        mat = np.array([[var[(var.model == m) & (var.string == op)]["exact_acc"].iloc[0] for m in models] for op in "ABC"])
        cnt = np.array([[int(var[(var.model == m) & (var.string == op)]["exact_correct"].iloc[0]) for m in models] for op in "ABC"])
        fig, ax = plt.subplots(figsize=(14, 3.2))
        ax.grid(False)
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(len(models)):
                ax.text(j, i, f"{cnt[i, j]}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] > 0.6 else "#0b0b0b")
        ax.set_xticks(range(len(models))); ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], fontsize=7, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(list("ABC"))
        ax.set_title("variable_length: singleton exact successes (bare one-token instruction; of 1000 grids each)")
        fig.tight_layout()
        paths += save_fig(fig, out, "singletons_variable_length", var)
    # (c) A/B/C exact and cell across both formats
    rows = []
    for cond, t in (("identity", idt), ("variable_length", var)):
        if t is None:
            continue
        for (m, op), s in t.groupby(["model", "string"]):
            rows.append({"condition": cond, "model": m, "op": op, "exact_mean": s.exact_acc.mean(),
                         "exact_min": s.exact_acc.min(), "exact_max": s.exact_acc.max(), "cell_mean": s.cell_acc.mean()})
    if rows:
        df = pd.DataFrame(rows)
        models = [m for m in MODEL_IDS if m in set(df.model)]
        fig, axes = plt.subplots(2, 1, figsize=(18, 8))
        for ax, metric, lab in ((axes[0], "exact_mean", "exact accuracy"), (axes[1], "cell_mean", "cell accuracy")):
            for i, m in enumerate(models):
                for j, op in enumerate("ABC"):
                    for c, (cond, colr) in enumerate((("identity", "#2a78d6"), ("variable_length", "#eb6834"))):
                        r = df[(df.model == m) & (df.op == op) & (df.condition == cond)]
                        if not len(r):
                            continue
                        x = i * 4 + j + (c - 0.5) * 0.4
                        ax.bar(x, r[metric].iloc[0], width=0.38, color=colr,
                               label=(f"{cond}" + (" (mean over 5 placements; bar = min..max)" if cond == "identity" else ""))
                               if i == j == 0 else None)
                        if cond == "identity" and metric == "exact_mean":
                            ax.vlines(x, r.exact_min.iloc[0], r.exact_max.iloc[0], color="#0b0b0b", lw=1)
                    ax.text(i * 4 + j, -0.08, op, ha="center", fontsize=7)
            ax.set_xticks([i * 4 + 1 for i in range(len(models))])
            ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], fontsize=7)
            ax.tick_params(axis="x", pad=14)
            ax.set(ylim=(0, 1.05), ylabel=lab)
            ax.legend(fontsize=7)
        axes[0].set_title("Singletons A/B/C by model and format (seed 42, best ckpt; min..max is over placements, NOT seeds)")
        fig.tight_layout()
        paths += save_fig(fig, out, "singletons_by_format", df)
    return paths


def plot_dominant_and_coverage(diag: dict, out: Path):
    import pandas as pd
    plt = _plt()
    ps = diag.get("per_string")
    if ps is None or not len(ps):
        return []
    cols = [("variable_length", "n/a", op) for op in "ABC"] + [("identity", p, op) for op in "ABC" for p in range(1, 6)]
    models = [m for m in MODEL_IDS if m in set(ps.model)]
    mat = np.full((len(models), len(cols)), np.nan)
    txt = [["" for _ in cols] for _ in models]
    for i, m in enumerate(models):
        for j, (c, p, op) in enumerate(cols):
            r = ps[(ps.model == m) & (ps.condition == c) & (ps.position.astype(str) == str(p)) & (ps.string == op)]
            if len(r):
                mat[i, j] = r.dominant_wrong_unique_share.iloc[0]
                w = r.dominant_wrong_unique_word.iloc[0]
                txt[i][j] = (w if isinstance(w, str) and w and len(w) <= 4 else (w[:3] + "…" if isinstance(w, str) and w else "-"))
                if r.exact_acc.iloc[0] >= 0.5:
                    txt[i][j] = "ok"
    fig, ax = plt.subplots(figsize=(22, 1.2 + 0.45 * len(models)))
    ax.grid(False)
    ax.imshow(np.nan_to_num(mat), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, "n/a" if np.isnan(mat[i, j]) else txt[i][j], ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"var {op}" if c == "variable_length" else f"id {op}@{p}" for c, p, op in cols], rotation=90, fontsize=7)
    ax.axvline(2.5, color="#0b0b0b", lw=1.5)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([DISPLAY_NAMES[m] for m in models], fontsize=8)
    ax.set_title("Dominant wrong legal transformation of singleton predictions (text = canonical word, 'ok' = >=50% correct; "
                 "colour = share of records whose ONLY exact legal match is that transform)")
    fig.tight_layout()
    paths = save_fig(fig, out, "singletons_dominant_error", ps)
    hs = diag.get("hypothesis_summary")
    if hs is not None and len(hs):
        agg = hs.groupby(["condition", "model", "hypothesis_type"]).agg(
            population=("population_records", "sum"), eligible=("eligible_records", "sum"),
            matched=("matched_unique_records", "sum"), any_legal=("population_any_legal_match_records", "sum")).reset_index()
        agg["unconditional"] = agg.matched / agg.population.where(agg.population > 0)
        agg.to_csv(out / "singletons_hypothesis_coverage.csv", index=False)
        keep = ["identity", "first_and_last", "duplicate_single", "X_then_C", "C_then_X", "original_pair"]
        fig, axes = plt.subplots(1, 2, figsize=(18, 5), squeeze=False)
        for ax, cond in zip(axes[0], CONDS):
            s = agg[(agg.condition == cond) & agg.hypothesis_type.isin(keep)]
            ms = [m for m in MODEL_IDS if m in set(s.model)]
            M = np.full((len(keep), len(ms)), np.nan)
            for i, h in enumerate(keep):
                for j, m in enumerate(ms):
                    r = s[(s.hypothesis_type == h) & (s.model == m)]
                    if len(r):
                        M[i, j] = r.unconditional.iloc[0]
            ax.grid(False)
            ax.imshow(np.nan_to_num(M), cmap="Purples", vmin=0, vmax=1, aspect="auto")
            for i in range(len(keep)):
                for j in range(len(ms)):
                    ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(ms))); ax.set_xticklabels([DISPLAY_NAMES[m] for m in ms], rotation=40, ha="right", fontsize=7)
            ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=8)
            ax.set_title(f"{cond}: unconditional coverage of singleton records (matched/population; union within rule; "
                         "CC = identity)", fontsize=8)
        fig.tight_layout()
        paths += save_fig(fig, out, "singletons_hypothesis_coverage", agg)
    return paths


# ---------------------------------------------------------------------------
# Cross-format comparison
# ---------------------------------------------------------------------------

def compare_formats(root: Path, models: List[str], log=print):
    """Pairs by model, word and grid hash. Singletons: the ONE variable prediction per (op, grid) vs each identity
    placement (transition tables per placement, plus the identity mean); seen words: per k."""
    import pandas as pd
    root = Path(root)
    out = root / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    rows, seen_rows, status = [], [], []
    for m in models:
        a, b = load_predictions(root, "identity", m), load_predictions(root, "variable_length", m)
        if a is None or b is None:
            status.append({"model": m, "status": "not yet available: " + ", ".join(
                c for c, x in (("identity", a), ("variable_length", b)) if x is None) + " missing"})
            continue
        status.append({"model": m, "status": "compared"})
        vs = b["kind"] == "single"
        vkey = pd.DataFrame({"op": b["string"][vs], "grid": b["grid_id"][vs], "h": b["grid_hash"][vs],
                             "v_exact": b["exact"][vs], "vi": np.nonzero(vs)[0]})
        for p in range(1, 6):
            s = (a["kind"] == "single") & (a["position"] == p)
            ikey = pd.DataFrame({"op": a["string"][s], "grid": a["grid_id"][s], "h": a["grid_hash"][s],
                                 "i_exact": a["exact"][s], "ii": np.nonzero(s)[0]})
            j = vkey.merge(ikey, on=["op", "grid", "h"], validate="one_to_one")
            same = (a["pred"][j.ii.values] == b["pred"][j.vi.values]).reshape(len(j), -1).all(1)
            j["same"] = same
            for op, g in j.groupby("op"):
                rows.append({"model": m, "op": op, "identity_slot": p, "n_grids": len(g),
                             "variable_exact": int(g.v_exact.sum()), "identity_exact": int(g.i_exact.sum()),
                             "both_correct": int((g.v_exact & g.i_exact).sum()),
                             "only_variable_correct": int((g.v_exact & ~g.i_exact).sum()),
                             "only_identity_correct": int((~g.v_exact & g.i_exact).sum()),
                             "both_wrong": int((~g.v_exact & ~g.i_exact).sum()),
                             "identical_prediction_rate": float(g.same.mean())})
        ka = pd.DataFrame({"w": a["string"], "grid": a["grid_id"], "h": a["grid_hash"], "ea": a["exact"], "k": a["k"],
                           "kind": a["kind"]})
        kb = pd.DataFrame({"w": b["string"], "grid": b["grid_id"], "h": b["grid_hash"], "eb": b["exact"], "kind": b["kind"]})
        j = ka[ka.kind == "seen"].merge(kb[kb.kind == "seen"], on=["w", "grid", "h"], validate="one_to_one")
        for k, g in j.groupby("k"):
            seen_rows.append({"model": m, "k": int(k), "n": len(g), "identity_exact": int(g.ea.sum()),
                              "variable_exact": int(g.eb.sum()), "both_correct": int((g.ea & g.eb).sum()),
                              "only_identity_correct": int((g.ea & ~g.eb).sum()),
                              "only_variable_correct": int((~g.ea & g.eb).sum()), "both_wrong": int((~g.ea & ~g.eb).sum())})
    single = pd.DataFrame(rows)
    seen = pd.DataFrame(seen_rows)
    st = pd.DataFrame(status)
    single.to_csv(out / "singletons_identity_vs_variable.csv", index=False)
    seen.to_csv(out / "seen_words_identity_vs_variable.csv", index=False)
    st.to_csv(out / "comparison_status.csv", index=False)
    summary = None
    if len(single):
        summary = single.groupby(["model", "op"]).agg(
            variable_exact=("variable_exact", "first"), n_grids=("n_grids", "first"),
            identity_exact_mean_over_slots=("identity_exact", "mean"), identity_exact_min=("identity_exact", "min"),
            identity_exact_max=("identity_exact", "max"), identity_slot1_exact=("identity_exact", "first")).reset_index()
        summary.to_csv(out / "singletons_format_summary.csv", index=False)
    log("Format comparison: " + "; ".join(f"{r.model}: {r.status}" for r in st.itertuples()))
    return {"singletons": single, "seen": seen, "status": st, "summary": summary}


def status_table(root: Path, models: List[str]):
    """Per-model completion status for both formats (a notebook is complete only if every model is)."""
    import pandas as pd
    from .idvar_pipeline import build_artifact_index
    idx = build_artifact_index(root)
    rows = []
    for cond in CONDS:
        for m in models:
            d = Path(root) / cond
            rows.append({"condition": cond, "model": m,
                         "trained": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(),
                         "evaluated": bool(idx.get(f"{cond}/{m}", {}).get("complete")),
                         "records": idx.get(f"{cond}/{m}", {}).get("n_records")})
    return pd.DataFrame(rows)
