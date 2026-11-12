"""Notebook 26a/26b driver: train, evaluate the LAST checkpoint on the main (sampled S7_k02345 test set) and the
auxiliary task, and report."""
from __future__ import annotations

import io
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import auxtask as AX
from . import tokenization as T
from . import idvar as V


def family_root(results_root, mode):
    r = Path(results_root) / AX.SUBDIR
    return r if mode == "full" else r / "smoke"


def make_cfg(e, results_root, mode, models, seed=42, **kw):
    from .config import make_experiment_config
    cfg = make_experiment_config(e, mode, str(Path(results_root) / AX.SUBDIR), models, seed=seed, **kw)
    cfg.resume = True
    return cfg


def prepare(cfg, repo_dir=None, log=print):
    from .data import load_or_build_manifests
    from .pipeline import Context
    from .utils import atomic_write_json
    m = load_or_build_manifests(cfg.manifest_dir, cfg.train.seed)
    val = AX.make_val_set(m.val_grids[: cfg.n_val_grids], cfg.train.seed)
    subset = V.subset_by_grids(val, cfg.val_subset_grids)
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json",
                      {**cfg.to_dict(), "main_ks": list(AX.MAIN_KS), "aux_ks": list(AX.aux_ks(cfg.experiment)),
                       "aux_per_update": AX.CONDITIONS[cfg.experiment][1], "manifest_hash": m.manifest_hash})
    log(f"[{cfg.experiment}] aux k={list(AX.aux_ks(cfg.experiment))} ({AX.CONDITIONS[cfg.experiment][1]}/update) | "
        f"val {len(val['x'])} (main only) | manifests {m.manifest_hash[:12]}")
    return Context(cfg, m, val, subset, None, repo_dir)


def eval_dir(cfg, model_id):
    return cfg.experiment_dir / "eval_last" / model_id


def _npz(**a):
    buf = io.BytesIO()
    np.savez_compressed(buf, **a)
    return buf.getvalue()


def evaluate_last(model_id, ctx, plan, chunk_records=25000, log=print):
    import torch
    from .evaluation import run_model_on_set
    from .models.registry import build_model
    from .training import run_config, run_dir, training_complete
    from .utils import atomic_write_bytes, atomic_write_json, free_memory, resolve_amp_dtype, resolve_device
    cfg, e = ctx.cfg, ctx.cfg.experiment
    rc = run_config(cfg, model_id, ctx.manifests.manifest_hash)
    ed = eval_dir(cfg, model_id)
    af = ed / "ARTIFACT.json"
    if af.exists():
        a = json.loads(af.read_text())
        if a.get("complete") and a.get("config_hash") == rc["config_hash"] and a.get("plan") == plan:
            log(f"[{e}/{model_id}] evaluation already complete -> skipping")
            return a
    if not training_complete(cfg, model_id, rc):
        raise RuntimeError("training not complete")
    st = torch.load(run_dir(cfg, model_id) / "state.pt", map_location="cpu", weights_only=False)
    assert st["config_hash"] == rc["config_hash"] and st["global_step"] == cfg.train.optimizer_steps
    tc = cfg.train
    device = resolve_device(tc.device)
    amp = resolve_amp_dtype(device, tc.dtype)
    model = build_model(model_id, rc["model_config"]).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    es = AX.make_test(ctx.manifests.test_grids[: cfg.n_test_grids], tc.seed, **plan)
    n = len(es["x"])
    cd = ed / "chunks"
    cd.mkdir(parents=True, exist_ok=True)
    tag = json.dumps(plan, sort_keys=True)
    t0 = time.time()
    starts = list(range(0, n, chunk_records))
    for ci, r0 in enumerate(starts):
        f = cd / f"chunk_{ci:03d}.npz"
        if f.exists():
            with np.load(f) as z:
                if str(z["config_hash"]) == rc["config_hash"] and str(z["plan"]) == tag:
                    continue
        idx = np.arange(r0, min(n, r0 + chunk_records))
        sub = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in es.items()}
        r = run_model_on_set(model, sub, device, amp, tc.eval_batch_size, want_ce=False)
        Y = sub["y"].reshape(-1, 36)
        P = r["pred"].reshape(-1, 36)
        atomic_write_bytes(f, _npz(exact=r["valid"] & (P == Y).all(1), cell=(P == Y).sum(1).astype(np.int8),
                                   valid=r["valid"], k=sub["k"].astype(np.int8), kind=sub["kind"].astype("U8"),
                                   position=sub["position"].astype(np.int8), slots=sub["slots"].astype(np.int8),
                                   string=sub["string"].astype("U12"), grid_id=sub["grid_id"].astype(np.int16),
                                   pred=P.astype(np.int8), x=sub["x"].reshape(-1, 36).astype(np.int8),
                                   config_hash=np.array(rc["config_hash"]), plan=np.array(tag)))
        log(f"  [{e}/{model_id}] chunk {ci + 1}/{len(starts)} ({len(idx):,}) | {(time.time() - t0) / 60:.1f} min")
    del model
    free_memory()
    a = {"experiment": e, "model": model_id, "checkpoint": "last", "step": st["global_step"], "best_step": st.get("best_step"),
         "n_records": n, "complete": True, "plan": plan, "config_hash": rc["config_hash"], "eval_seconds": time.time() - t0}
    atomic_write_json(af, a)
    log(f"[{e}/{model_id}] LAST checkpoint (step {st['global_step']}) evaluated on {n:,} records")
    return a


def load_eval(cfg, model_id):
    fs = sorted((eval_dir(cfg, model_id) / "chunks").glob("chunk_*.npz"))
    parts = [dict(np.load(f)) for f in fs]
    if not parts:
        return {}
    keys = [k for k in parts[0] if np.asarray(parts[0][k]).ndim > 0]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def status_table(results_root, mode, seed=42, conditions=None):
    import pandas as pd
    rows = []
    for e in (conditions or list(AX.CONDITIONS)):
        for m in AX.MODELS:
            cfg = make_cfg(e, results_root, mode, [m], seed)
            d = cfg.experiment_dir
            af = eval_dir(cfg, m) / "ARTIFACT.json"
            a = json.loads(af.read_text()) if af.exists() else {}
            rows.append({"condition": e, "model": m, "trained": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(),
                         "evaluated_last": bool(a.get("complete")), "records": a.get("n_records")})
    df = pd.DataFrame(rows)
    df["complete"] = df.trained & df.evaluated_last
    return df


def run_queue(results_root, mode, seed, conditions, models, plan, steps=None, repo_dir=None, device="auto", log=print):
    from .training import train_model
    from .utils import free_memory
    for e in conditions:
        over = {"optimizer_steps": steps} if steps else {}
        cfg = make_cfg(e, results_root, mode, models, seed, device=device, **over)
        ctx = prepare(cfg, repo_dir, log)
        p = dict(plan)
        if mode != "full":
            p = {"n_single_grids": 2, "n_seen_per_k": 20, "n_long_per_k": 40, "n_aux_per_k": 10}
        for m in models:
            log(f"\n=== {e} / {m} ===")
            try:
                train_model(m, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, log=log, repo_dir=repo_dir)
                evaluate_last(m, ctx, p, log=log)
            except Exception:
                traceback.print_exc()
                log(f"[{e}/{m}] FAILED -> rerun to resume")
            finally:
                free_memory()
    return status_table(results_root, mode, seed, conditions)


def report(results_root, mode, seed=42, conditions=None, out_name=None, review_name="AuxTask_Review.ipynb",
           show=True, log=print):
    import pandas as pd
    import matplotlib.pyplot as plt
    from . import mirror_diag as M
    from .mirror_report import Report, write_review
    G = M.group()
    conds = conditions or list(AX.CONDITIONS)
    root = family_root(results_root, mode)
    out = root / "report" / (out_name or f"seed_{seed}")
    R = Report(show)
    st = status_table(results_root, mode, seed, conds)
    R.md(f"# Auxiliary-task transfer (seed {seed}; LAST checkpoint)\n{int(st.complete.sum())}/{len(st)} runs complete.\n\n"
         "Main task: S=7, A/B/C, trained k ∈ {0,2,3,4,5}; held out: singletons (every slot), k=6, k=7. Auxiliary task: "
         "value operations D (d+1 mod 10) and E (9−d), never moving a cell, with the training k listed per condition.")
    R.table("Run status", st, 10)
    main, aux, islot, drop = [], [], [], []
    for e in conds:
        for m in AX.MODELS:
            cfg = make_cfg(e, results_root, mode, [m], seed)
            af = eval_dir(cfg, m) / "ARTIFACT.json"
            if not af.exists():
                main.append({"condition": e, "model": m, "status": "not yet available"})
                continue
            d = load_eval(cfg, m)
            kind = d["kind"].astype(str)
            ex = d["exact"].astype(bool)
            k = d["k"].astype(int)
            word = np.array([s.split(":", 1)[1] if ":" in s else s for s in d["string"].astype(str)])
            mm = ~np.char.startswith(kind, "aux")
            s1 = kind == "single"
            row = {"condition": e, "model": m, "aux k": str(list(AX.aux_ks(e))), "status": "available",
                   **{f"single {op}": f"{int(ex[s1 & (word == op)].sum())}/{int((s1 & (word == op)).sum())}" for op in "ABC"},
                   "singletons": f"{int(ex[s1].sum())}/{int(s1.sum())}",
                   "main trained-k exact": round(float(ex[mm & np.isin(k, AX.MAIN_KS)].mean()), 4)}
            for kk in (6, 7):
                q = mm & (k == kk)
                row[f"main k={kk}"] = f"{int(ex[q].sum())}/{int(q.sum())} ({ex[q].mean():.3f})"
            main.append(row)
            for kk in range(AX.S + 1):
                q = kind == f"aux{kk}"
                aux.append({"condition": e, "model": m, "aux k": kk,
                            "status": "trained" if kk in AX.aux_ks(e) else "not trained",
                            "exact_acc": round(float(ex[q].mean()), 4) if q.any() else np.nan, "records": int(q.sum())})
            # k=6: identity slot and AR 'drop op #5'
            q6 = np.flatnonzero(mm & (k == 6))
            sl = d["slots"][q6][:, 1:]
            isl = (sl == V.ID_OP).argmax(1) + 1
            for p in range(1, AX.S + 1):
                qq = isl == p
                islot.append({"condition": e, "model": m, "I slot": p, "k=6 exact_acc": round(float(ex[q6][qq].mean()), 4)})
            fail = q6[~ex[q6] & d["valid"][q6]]
            X, P = d["x"][fail], d["pred"][fail]
            hit5 = np.zeros(len(fail), bool)
            for w in np.unique(word[fail]):
                qq = word[fail] == w
                p5 = G.perms[G.word_id(w[:4] + w[5:])]
                hit5[qq] = (X[qq][:, p5] == P[qq]).all(1)
            drop.append({"condition": e, "model": m, "k=6 failures": len(fail), "= drop op #5": int(hit5.sum()),
                         "share": round(float(hit5.mean()), 3) if len(fail) else np.nan})
    T_ = {"main": pd.DataFrame(main), "aux": pd.DataFrame(aux), "k6_by_I_slot": pd.DataFrame(islot), "drop5": pd.DataFrame(drop)}
    R.table("Main task (held out: singletons, k=6, k=7)", T_["main"], 20)
    R.table("Auxiliary task accuracy by k", T_["aux"].pivot_table(index=["condition", "model"], columns="aux k",
                                                                   values="exact_acc").reset_index() if len(T_["aux"]) else T_["aux"], 20)
    R.table("Main k=6 accuracy by <identity> slot", T_["k6_by_I_slot"].pivot_table(index=["condition", "model"], columns="I slot",
                                                                                   values="k=6 exact_acc").reset_index()
            if len(T_["k6_by_I_slot"]) else T_["k6_by_I_slot"], 20)
    R.table("Main k=6 failures equal to 'drop the 5th operation'", T_["drop5"], 20)
    mt = T_["main"]
    if len(mt) and "main k=6" in mt:
        q = mt[mt.status == "available"].copy()
        f = lambda s: float(s.split("(")[1].rstrip(")"))  # noqa: E731
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        for ax, col, title in zip(axes, ("singletons", "main k=6", "main k=7"), ("singletons", "k=6", "k=7")):
            vals = q[col].map(lambda s: int(s.split("/")[0]) / int(s.split("/")[1].split()[0])) if col == "singletons" else q[col].map(f)
            labels = [f"{c}\n{m.replace('transformer_ar', 'AR').replace('causal_encoder', 'enc')}" for c, m in zip(q.condition, q.model)]
            ax.bar(range(len(q)), vals, color=["#2a78d6" if "ar" in m else "#8e6bd6" for m in q.model])
            ax.set_xticks(range(len(q))); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
            ax.set(ylim=(0, 1.05), title=f"main {title}")
        fig.tight_layout()
        R.fig(fig, "Main-task held-out accuracy by condition", out / "figures" / "main_heldout.png")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k_, t in T_.items():
        t.to_csv(out / "tables" / f"{k_}.csv", index=False)
    rp = write_review(R.items, out, "Auxiliary-task transfer", "# Auxiliary-task transfer — review snapshot\n\nBaked outputs.",
                      review_name)
    log(f"REVIEW FILE: {rp}")
    return rp
