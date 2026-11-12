"""Notebook 20: causal encoder (parallel output queries) in four identity-format settings, paired with the existing
AR runs (same training streams, validation and test manifests as notebooks 10 / 14 / 16).

  setting      experiment id        slots  train k        AR reference (read-only)
  S5_k02345    idvar_identity       5      0,2,3,4,5      identity_varlen_02345_all_models/identity
  S5_k0234     idext_train0234      5      0,2,3,4        identity_ar_count_extrapolation/train0234/seed_42
  S4_k0234     slots4_train0234     4      0,2,3,4        identity_ar_slot_count/slots4_train0234/seed_42
  S6_k02345    slots6_train02345    6      0,2,3,4,5      identity_ar_slot_count/slots6_train02345/seed_42

Model: [op slots] + [36 input cells] + [36 output queries], causal attention over the WHOLE sequence, all 36
answers read at the queries in one parallel pass (no autoregressive feedback). ``causal_encoder`` (6 untied blocks)
and ``looped_causal_encoder`` (one shared block x 6). One common initialization per architecture across the four
settings (the slot table is grown with a separate generator for S6). Results go to a NEW root; the reference roots
are only read.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SUBDIR = "causal_encoder_identity_v1"
MODELS = ["causal_encoder", "looped_causal_encoder"]
SETTINGS = {
    "S5_k02345": ("idvar_identity", 5, (0, 2, 3, 4, 5), "identity_varlen_02345_all_models/identity"),
    "S5_k0234": ("idext_train0234", 5, (0, 2, 3, 4), "identity_ar_count_extrapolation/train0234/seed_42"),
    "S4_k0234": ("slots4_train0234", 4, (0, 2, 3, 4), "identity_ar_slot_count/slots4_train0234/seed_42"),
    "S6_k02345": ("slots6_train02345", 6, (0, 2, 3, 4, 5), "identity_ar_slot_count/slots6_train02345/seed_42"),
}
REFERENCE_MODELS = ["transformer_ar", "looped_transformer_ar", "transformer_encoder", "looped_transformer_encoder"]


def family_root(results_root: str, mode: str) -> Path:
    r = Path(results_root) / SUBDIR
    return r if mode == "full" else r / "smoke"


def make_cfg(setting: str, results_root: str, mode: str, models: List[str], seed: int = 42, **kw):
    from .config import make_experiment_config
    exp = SETTINGS[setting][0]
    root = str(Path(results_root) / SUBDIR)
    cfg = make_experiment_config(exp, mode, root, models, seed=seed, **kw)
    cfg.resume = True
    return cfg


def prepare(setting: str, cfg, repo_dir=None, log=print):
    exp = SETTINGS[setting][0]
    if exp == "idvar_identity":
        from .idvar_pipeline import prepare_idvar
        return prepare_idvar(cfg, repo_dir, log=log)
    if exp.startswith("idext_"):
        from .idext_reports import prepare_idext
        return prepare_idext(cfg, repo_dir, log=log)
    from .idslot_reports import prepare_condition
    return prepare_condition(cfg, repo_dir, log=log)


def _artifact(eval_dir: Path) -> Optional[dict]:
    f = Path(eval_dir) / "ARTIFACT.json"
    if not f.exists():
        return None
    try:
        a = json.loads(f.read_text())
    except Exception:
        return None
    if not a.get("complete"):
        return None
    p = Path(a.get("predictions", ""))
    if not p.exists():
        p = Path(eval_dir) / "test_predictions.npz"
    return {**a, "predictions": str(p)} if p.exists() else None


def status_table(results_root: str, mode: str, seed: int = 42):
    import pandas as pd
    rows = []
    for s in SETTINGS:
        for m in MODELS:
            cfg = make_cfg(s, results_root, mode, [m], seed)
            d = cfg.experiment_dir
            a = _artifact(d / "eval" / m)
            rows.append({"setting": s, "experiment": SETTINGS[s][0], "model": m,
                         "trained": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(), "evaluated": a is not None,
                         "selected_step": a.get("selected_step") if a else None, "records": a.get("n_records") if a else None})
    df = pd.DataFrame(rows)
    df["complete"] = df.trained & df.evaluated
    return df


def run_queue(results_root: str, mode: str, seed: int, settings: List[str], models: List[str], repo_dir=None,
              chunk_grids: int = 100, device="auto", dtype="auto", log=print):
    from .idvar_pipeline import evaluate_idvar_model
    from .training import train_model
    from .utils import free_memory
    for s in settings:
        cfg = make_cfg(s, results_root, mode, models, seed, device=device, dtype=dtype)
        ctx = prepare(s, cfg, repo_dir, log)
        for m in models:
            log(f"\n=== {s} ({SETTINGS[s][0]}) / {m} ===")
            try:
                train_model(m, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, log=log, repo_dir=repo_dir)
                evaluate_idvar_model(m, ctx, log=log, chunk_grids=chunk_grids)
            except Exception:
                traceback.print_exc()
                log(f"[{s}/{m}] FAILED -> rerun to resume")
            finally:
                free_memory()
        del ctx
        free_memory()
    return status_table(results_root, mode, seed)


def init_audit(seed: int = 42):
    import pandas as pd
    import torch
    from .config import make_experiment_config
    from .idslot import grown_initial_state
    from .models.common import count_parameters
    from .models.registry import build_model
    from .training import model_config_for
    rows = []
    for m in MODELS:
        sds = {}
        for s, (exp, *_r) in SETTINGS.items():
            mod = build_model(m, model_config_for(make_experiment_config(exp, models=[m]), m), seed=seed)
            if exp.startswith("slots"):
                mod.load_state_dict(grown_initial_state(m, seed))
            sds[s] = (mod.state_dict(), count_parameters(mod))
        ref = sds["S6_k02345"][0]
        same = []
        for s, (sd, _) in sds.items():
            same.append(all(torch.equal(sd[k], ref[k][: sd[k].shape[0]]) if sd[k].shape != ref[k].shape else torch.equal(sd[k], ref[k])
                            for k in sd))
        rows.append({"model": m, "params_S5": sds["S5_k02345"][1], "params_S6": sds["S6_k02345"][1],
                     "common_init_across_4_settings (slot table grown for S4/S6)": bool(all(same))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report (reads saved artifacts only)
# ---------------------------------------------------------------------------

def _frame(a):
    import pandas as pd
    n = len(a["pred"])
    P, Y = a["pred"].reshape(n, 36), a["target"].reshape(n, 36)
    return pd.DataFrame({"word": a["string"].astype(str), "k": a["k"].astype(int), "kind": a["kind"].astype(str),
                         "position": a["position"].astype(int), "grid_hash": a["grid_hash"],
                         "exact": a["exact"].astype(bool), "cell": (P == Y).mean(1)})


def report(results_root: str, mode: str, seed: int = 42, show: bool = True, log=print) -> Path:
    import pandas as pd
    import matplotlib.pyplot as plt
    from . import mirror_diag as M
    from .mirror_report import Report, write_review
    from .words import from_label
    G = M.group()
    R = Report(show)
    root = family_root(results_root, mode)
    out = root / "report" / f"seed_{seed}"
    st = status_table(results_root, mode, seed)
    R.md(f"# Causal encoder in four identity settings (seed {seed})\n{int(st.complete.sum())}/8 runs complete.\n\n"
         "Model: [op slots] + [36 input cells] + [36 output queries]; causal attention over the whole sequence; all "
         "36 answers predicted in parallel at the queries (no autoregressive feedback). AR and bidirectional-encoder "
         "references are the existing runs with the SAME training stream, validation and test manifests.")
    R.table("Run status", st, 20)
    rows_k, rows_s, rows_geo, rows_pair, rows_slot = [], [], [], [], []
    for s, (exp, S, ks, ref_rel) in SETTINGS.items():
        cfg = make_cfg(s, results_root, mode, MODELS, seed)
        runs = {m: _artifact(cfg.experiment_dir / "eval" / m) for m in MODELS}
        refs = {m: _artifact(Path(results_root) / ref_rel / "eval" / m) for m in REFERENCE_MODELS}
        frames = {}
        for m, a in {**runs, **refs}.items():
            if a is None:
                continue
            frames[m] = _frame(np.load(a["predictions"]))
        for m in MODELS + REFERENCE_MODELS:
            src = "NEW causal encoder" if m in MODELS else "existing reference"
            if m not in frames:
                if m in MODELS or m.startswith("transformer_ar") or m.startswith("looped_transformer_ar"):
                    rows_s.append({"setting": s, "model": m, "source": src, "status": "not available"})
                continue
            df = frames[m]
            sg = df[df.kind == "single"]
            r = {"setting": s, "model": m, "source": src, "status": "available",
                 **{op: f"{int(sg[sg.word == op].exact.sum())}/{int((sg.word == op).sum())}" for op in "ABC"},
                 "singleton exact": f"{int(sg.exact.sum())}/{len(sg)}", "singleton cell acc": round(sg.cell.mean(), 4),
                 "seen-support exact": round(df[df.k.isin(ks)].exact.mean(), 4)}
            if (df.kind == "six").any():
                r["k=6 exact (of 729000)"] = int(df[df.kind == "six"].exact.sum())
            rows_s.append(r)
            g = df.groupby("k").agg(exact_correct=("exact", "sum"), n=("exact", "size"), exact_acc=("exact", "mean"),
                                    cell_acc=("cell", "mean")).reset_index()
            g["status"] = np.where(g.k.isin(ks), "trained k", "HELD OUT")
            g.insert(0, "model", m); g.insert(0, "setting", s)
            rows_k.append(g)
            ps = sg.groupby(["word", "position"]).exact.sum().unstack(fill_value=0)
            for op, row in ps.iterrows():
                rows_slot.append({"setting": s, "model": m, "op": op, **{f"slot{int(p)}": int(v) for p, v in row.items()}})
            if m in MODELS:
                a = np.load(runs[m]["predictions"])
                si = np.flatnonzero(a["kind"] == "single")
                P = a["pred"].reshape(-1, 36)[si]
                X = a["input"].reshape(-1, 36)[si]
                labels = a["string"][si].astype(str)
                tids = np.array([G.word_id(from_label(w)) for w in labels])
                ms = M.Matcher(G).match(P, X, a["grid_hash"][si], a["valid"][si])
                c = M.classify(G, ms, tids, a["valid"][si], a["exact"][si])
                gd = pd.DataFrame({"op": labels, "exact": a["exact"][si], "outcome": c["outcome"],
                                   "identity": c["match_identity"], "n_match": c["n_match"]})
                for op, gg in gd.groupby("op"):
                    rows_geo.append({"setting": s, "model": m, "op": op, "n": len(gg), "correct": int(gg.exact.sum()),
                                     "identity copy": int((~gg.exact & gg.identity).sum()),
                                     "other legal transform": int((~gg.exact & (gg.n_match > 0) & ~gg.identity).sum()),
                                     "non-legal grid": int((gg.outcome == "wrong: valid, no legal match").sum())})
        # paired singleton comparison (within-grid mean over slots; same grids, same layouts per setting)
        for m in MODELS:
            if m not in frames:
                continue
            new = frames[m][frames[m].kind == "single"].groupby(["word", "grid_hash"]).exact.mean()
            for ref in REFERENCE_MODELS:
                if ref not in frames:
                    continue
                old = frames[ref][frames[ref].kind == "single"].groupby(["word", "grid_hash"]).exact.mean()
                for op in "ABC":
                    j = pd.concat([old.xs(op, level=0).rename("ref"), new.xs(op, level=0).rename("new")], axis=1, join="inner")
                    rows_pair.append({"setting": s, "causal model": m, "reference": ref, "op": op, "grids": len(j),
                                      "reference mean": round(j.ref.mean(), 4), "causal mean": round(j.new.mean(), 4),
                                      "causal better": int((j.new > j.ref).sum()), "tied": int((j.new == j.ref).sum()),
                                      "causal worse": int((j.new < j.ref).sum())})
    T = {"singletons": pd.DataFrame(rows_s), "by_k": pd.concat(rows_k, ignore_index=True) if rows_k else pd.DataFrame(),
         "singleton_by_slot": pd.DataFrame(rows_slot), "singleton_outcomes": pd.DataFrame(rows_geo),
         "paired_vs_reference": pd.DataFrame(rows_pair)}
    R.table("Singletons (held out) — new causal encoders vs existing AR / bidirectional-encoder runs", T["singletons"], 60)
    R.table("Singleton exact successes by operation x slot (each cell of 1000)", T["singleton_by_slot"], 100)
    R.table("Causal-encoder singleton outcomes (exact matching to the 288 legal transforms)", T["singleton_outcomes"], 60)
    R.table("Paired by (operation, grid): within-grid mean over slots, causal encoder vs reference", T["paired_vs_reference"], 200)
    R.table("Exact accuracy by k", T["by_k"], 300)
    s1 = T["singletons"]
    if len(s1) and "singleton exact" in s1:
        q = s1[s1.status == "available"].copy()
        q["acc"] = [int(v.split("/")[0]) / int(v.split("/")[1]) for v in q["singleton exact"]]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        mods = [m for m in MODELS + REFERENCE_MODELS if m in set(q.model)]
        w = 0.8 / max(1, len(mods))
        for i, m in enumerate(mods):
            v = [q[(q.setting == s) & (q.model == m)].acc.iloc[0] if len(q[(q.setting == s) & (q.model == m)]) else np.nan
                 for s in SETTINGS]
            ax.bar(np.arange(len(SETTINGS)) + i * w, v, w, label=m)
        ax.set_xticks(np.arange(len(SETTINGS)) + 0.4 - w / 2); ax.set_xticklabels(list(SETTINGS))
        ax.set(ylim=(0, 1.05), ylabel="singleton exact accuracy (all slots)", title="Held-out singletons")
        ax.legend(fontsize=7)
        R.fig(fig, "Singleton accuracy by setting and model", out / "figures" / "singletons.png")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, t in T.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    rp = write_review(R.items, out, "Causal encoder in four identity settings",
                      "# Causal encoder — review snapshot\n\nBaked outputs; nothing to run.", "Causal_Encoder_Review.ipynb")
    log(f"REVIEW FILE: {rp}")
    return rp
