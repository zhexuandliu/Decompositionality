"""Notebook 21 driver: train, evaluate the LAST checkpoint (chunked; test sets built per grid chunk), and report."""
from __future__ import annotations

import hashlib
import io
import json
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np

from . import longk as LK


def family_root(results_root: str, mode: str, subdir: str = LK.SUBDIR) -> Path:
    r = Path(results_root) / subdir
    return r if mode == "full" else r / "smoke"


def make_cfg(e, results_root, mode, models, seed=42, subdir: str = LK.SUBDIR, **kw):
    """``kw`` are TrainConfig overrides (e.g. optimizer_steps=10000); a different budget must use its own subdir."""
    from .config import make_experiment_config
    cfg = make_experiment_config(e, mode, str(Path(results_root) / subdir), models, seed=seed, **kw)
    cfg.resume = True
    return cfg


def prepare(cfg, repo_dir=None, log=print):
    from .data import load_or_build_manifests
    from .idvar import subset_by_grids
    from .pipeline import Context
    from .utils import atomic_write_json
    m = load_or_build_manifests(cfg.manifest_dir, cfg.train.seed)
    val = LK.make_val_set(m.val_grids[: cfg.n_val_grids], cfg.train.seed, cfg.experiment)
    assert set(np.unique(val["k"])) == set(LK.train_ks(cfg.experiment)) and not (val["k"] == 1).any()
    subset = subset_by_grids(val, cfg.val_subset_grids)
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json",
                      {**cfg.to_dict(), "S": LK.n_slots(cfg.experiment), "train_ks": list(LK.train_ks(cfg.experiment)),
                       "heldout_long_ks": LK.heldout_long_ks(cfg.experiment), "manifest_hash": m.manifest_hash})
    log(f"[{cfg.experiment}] S={LK.n_slots(cfg.experiment)} train k={list(LK.train_ks(cfg.experiment))} | val {len(val['x'])} "
        f"| manifests {m.manifest_hash[:12]}")
    return Context(cfg, m, val, subset, None, repo_dir)


def _npz(**a) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **a)
    return buf.getvalue()


def eval_dir(cfg, model_id) -> Path:
    return cfg.experiment_dir / "eval_last" / model_id


def evaluate_last(model_id, ctx, n_grids=1000, k7_grids=300, chunk_grids=50, log=print) -> dict:
    """Evaluate the FINAL-update model (state.pt) on singletons at every slot, all trained words, and every word of
    each held-out longer k. Resumable per grid chunk."""
    import torch
    from .evaluation import run_model_on_set
    from .models.registry import build_model
    from .training import run_config, run_dir, training_complete
    from .utils import atomic_write_bytes, atomic_write_json, free_memory, resolve_amp_dtype, resolve_device
    cfg, e = ctx.cfg, ctx.cfg.experiment
    rc = run_config(cfg, model_id, ctx.manifests.manifest_hash)
    ed = eval_dir(cfg, model_id)
    art_f = ed / "ARTIFACT.json"
    if art_f.exists():
        a = json.loads(art_f.read_text())
        if a.get("complete") and a.get("config_hash") == rc["config_hash"] and a.get("k7_grids") == k7_grids \
                and a.get("n_grids") == n_grids:
            log(f"[{e}/{model_id}] last-checkpoint evaluation already complete -> skipping")
            return a
    if not training_complete(cfg, model_id, rc):
        raise RuntimeError("training not complete")
    st = torch.load(run_dir(cfg, model_id) / "state.pt", map_location="cpu", weights_only=False)
    assert st["config_hash"] == rc["config_hash"] and st["global_step"] == cfg.train.optimizer_steps, "not the final state"
    tc = cfg.train
    device = resolve_device(tc.device)
    amp = resolve_amp_dtype(device, tc.dtype)
    model = build_model(model_id, rc["model_config"]).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    grids = ctx.manifests.test_grids
    cd = ed / "chunks"
    cd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    starts = list(range(0, n_grids, chunk_grids))
    total = 0
    for ci, g0 in enumerate(starts):
        f = cd / f"chunk_{ci:03d}.npz"
        g1 = min(n_grids, g0 + chunk_grids)
        es = LK.make_test_chunk(grids, tc.seed, e, g0, g1, n_grids, k7_grids)
        if f.exists():
            with np.load(f) as z:
                if str(z["config_hash"]) == rc["config_hash"] and len(z["exact"]) == len(es["x"]):
                    total += len(es["x"])
                    continue
        r = run_model_on_set(model, es, device, amp, tc.eval_batch_size, want_ce=False)
        Y = es["y"].reshape(-1, 36)
        P = r["pred"].reshape(-1, 36)
        exact = r["valid"] & (P == Y).all(1)
        keep = es["kind"] == "single"
        atomic_write_bytes(f, _npz(
            exact=exact, cell=(P == Y).sum(1).astype(np.int8), valid=r["valid"], k=es["k"].astype(np.int8),
            kind=es["kind"].astype("U6"), position=es["position"].astype(np.int8), grid_id=es["grid_id"].astype(np.int16),
            string=es["string"].astype("U8"), pred=P.astype(np.int8), config_hash=np.array(rc["config_hash"]),
            single_input=es["x"][keep].reshape(-1, 36).astype(np.int8)))
        total += len(es["x"])
        log(f"  [{e}/{model_id}] chunk {ci + 1}/{len(starts)} ({len(es['x']):,} records) | {(time.time() - t0) / 60:.1f} min")
    del model
    free_memory()
    exp = LK.expected_records(e, n_grids, k7_grids)
    a = {"experiment": e, "model": model_id, "checkpoint": "last", "step": st["global_step"],
         "best_step": st.get("best_step"), "n_records": total, "expected_records": exp, "complete": total == exp,
         "n_grids": n_grids, "k7_grids": k7_grids, "chunks": len(starts), "config_hash": rc["config_hash"],
         "eval_seconds": time.time() - t0}
    atomic_write_json(art_f, a)
    log(f"[{e}/{model_id}] LAST checkpoint (step {st['global_step']}; best was {st.get('best_step')}) evaluated: "
        f"{total:,} records")
    return a


def load_eval(cfg, model_id) -> dict:
    ed = eval_dir(cfg, model_id)
    fs = sorted((ed / "chunks").glob("chunk_*.npz"))
    parts = [dict(np.load(f)) for f in fs]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "config_hash"} if parts else {}
    return out


def status_table(results_root, mode, seed=42, conditions=None, subdir: str = LK.SUBDIR):
    import pandas as pd
    rows = []
    for e in (conditions or list(LK.CONDITIONS)):
        for m in LK.MODELS:
            cfg = make_cfg(e, results_root, mode, [m], seed, subdir)
            d = cfg.experiment_dir
            a = eval_dir(cfg, m) / "ARTIFACT.json"
            a = json.loads(a.read_text()) if a.exists() else {}
            rows.append({"condition": e, "model": m, "trained": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(),
                         "evaluated_last": bool(a.get("complete")), "records": a.get("n_records"),
                         "best_step (reference)": a.get("best_step")})
    df = pd.DataFrame(rows)
    df["complete"] = df.trained & df.evaluated_last
    return df


def run_queue(results_root, mode, seed, conditions, models, repo_dir=None, n_grids=1000, k7_grids=300,
              chunk_grids=50, device="auto", dtype="auto", log=print, subdir: str = LK.SUBDIR, steps=None):
    from .training import train_model
    from .utils import free_memory
    for e in conditions:
        over = {"optimizer_steps": steps} if steps else {}
        cfg = make_cfg(e, results_root, mode, models, seed, subdir, device=device, dtype=dtype, **over)
        ctx = prepare(cfg, repo_dir, log)
        ng = min(n_grids, cfg.n_test_grids)
        for m in models:
            log(f"\n=== {e} / {m} ===")
            try:
                train_model(m, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, log=log, repo_dir=repo_dir)
                evaluate_last(m, ctx, ng, min(k7_grids, ng), min(chunk_grids, ng), log)
            except Exception:
                traceback.print_exc()
                log(f"[{e}/{m}] FAILED -> rerun to resume")
            finally:
                free_memory()
    return status_table(results_root, mode, seed, conditions, subdir)


def report(results_root, mode, seed=42, show=True, log=print, conditions=None, subdir: str = LK.SUBDIR,
           compare_subdir=None, title_note: str = "", review_name: str = "LongK_AR_vs_CausalEnc_Review.ipynb") -> Path:
    import pandas as pd
    import matplotlib.pyplot as plt
    from . import mirror_diag as M
    from .mirror_report import Report, write_review
    from .words import from_label
    G = M.group()
    R = Report(show)
    conds = conditions or list(LK.CONDITIONS)
    root = family_root(results_root, mode, subdir)
    out = root / "report" / f"seed_{seed}"
    st = status_table(results_root, mode, seed, conds, subdir)
    R.md(f"# Transformer AR vs causal encoder — LAST checkpoint, singletons and longer instructions (seed {seed})\n"
         f"{int(st.complete.sum())}/{len(st)} runs complete. Every number below is for the model after the final "
         f"update (not the validation-selected checkpoint). {title_note}")
    R.table("Run status", st, 10)
    rk, rs, rslot, rgeo, rpair = [], [], [], [], []
    data = {}
    for e in conds:
        S, ks = LK.CONDITIONS[e]
        for m in LK.MODELS:
            cfg = make_cfg(e, results_root, mode, [m], seed, subdir)
            a = eval_dir(cfg, m) / "ARTIFACT.json"
            if not a.exists() or not json.loads(a.read_text()).get("complete"):
                rs.append({"condition": e, "model": m, "status": "not yet available"})
                continue
            d = load_eval(cfg, m)
            data[(e, m)] = d
            df = pd.DataFrame({"k": d["k"].astype(int), "kind": d["kind"].astype(str), "word": d["string"].astype(str),
                               "position": d["position"].astype(int), "grid": d["grid_id"].astype(int),
                               "exact": d["exact"].astype(bool), "cell": d["cell"] / 36})
            g = df.groupby("k").agg(exact_correct=("exact", "sum"), n=("exact", "size"), exact_acc=("exact", "mean"),
                                    cell_acc=("cell", "mean")).reset_index()
            g["status"] = ["trained" if k in ks else ("HELD OUT singleton" if k == 1 else "HELD OUT longer") for k in g.k]
            g.insert(0, "model", m); g.insert(0, "condition", e)
            rk.append(g)
            sg = df[df.kind == "single"]
            row = {"condition": e, "model": m, "status": "available",
                   **{op: f"{int(sg[sg.word == op].exact.sum())}/{int((sg.word == op).sum())}" for op in "ABC"},
                   "singletons": f"{int(sg.exact.sum())}/{len(sg)}", "singleton cell acc": round(sg.cell.mean(), 4),
                   "trained-k exact": round(df[df.k.isin(ks)].exact.mean(), 4)}
            for k in LK.heldout_long_ks(e):
                q = df[df.k == k]
                row[f"k={k} exact"] = f"{int(q.exact.sum())}/{len(q)} ({q.exact.mean():.3f})"
            rs.append(row)
            ps = sg.groupby(["word", "position"]).exact.sum().unstack(fill_value=0)
            for op, r_ in ps.iterrows():
                rslot.append({"condition": e, "model": m, "op": op, **{f"slot{int(p)}": int(v) for p, v in r_.items()}})
            # singleton failure types
            si = np.flatnonzero(d["kind"] == "single")
            P = d["pred"][si]
            X = d["single_input"]
            labels = d["string"][si].astype(str)
            tids = np.array([G.word_id(from_label(w)) for w in labels])
            gh = np.array([hashlib.sha256(r.tobytes()).hexdigest()[:16] for r in X.astype(np.uint8)])
            ms = M.Matcher(G).match(P, X, gh, d["valid"][si])
            c = M.classify(G, ms, tids, d["valid"][si], d["exact"][si])
            gd = pd.DataFrame({"op": labels, "exact": d["exact"][si], "outcome": c["outcome"], "identity": c["match_identity"],
                               "n_match": c["n_match"]})
            for op, gg in gd.groupby("op"):
                rgeo.append({"condition": e, "model": m, "op": op, "n": len(gg), "correct": int(gg.exact.sum()),
                             "identity copy": int((~gg.exact & gg.identity).sum()),
                             "other legal transform": int((~gg.exact & (gg.n_match > 0) & ~gg.identity).sum()),
                             "non-legal grid": int((gg.outcome == "wrong: valid, no legal match").sum())})
        # paired AR vs causal encoder on identical records (same chunk construction -> same order)
        if (e, "transformer_ar") in data and (e, "causal_encoder") in data:
            a_, b_ = data[(e, "transformer_ar")], data[(e, "causal_encoder")]
            assert (a_["string"] == b_["string"]).all() and (a_["grid_id"] == b_["grid_id"]).all()
            for k in sorted(set(a_["k"].tolist())):
                msk = a_["k"] == k
                ea, eb = a_["exact"][msk].astype(bool), b_["exact"][msk].astype(bool)
                rpair.append({"condition": e, "k": int(k), "n": int(msk.sum()), "AR correct": int(ea.sum()),
                              "causal enc correct": int(eb.sum()), "both": int((ea & eb).sum()),
                              "only AR": int((ea & ~eb).sum()), "only causal enc": int((~ea & eb).sum()),
                              "neither": int((~ea & ~eb).sum())})
    T = {"summary": pd.DataFrame(rs), "by_k": pd.concat(rk, ignore_index=True) if rk else pd.DataFrame(),
         "singleton_by_slot": pd.DataFrame(rslot), "singleton_outcomes": pd.DataFrame(rgeo),
         "paired_ar_vs_causal": pd.DataFrame(rpair)}
    R.table("Summary: singletons and longer held-out instructions (LAST checkpoint)", T["summary"], 20)
    R.table("Exact / cell accuracy by k", T["by_k"], 100)
    R.table("Singleton exact successes by operation x slot (each cell of the number of grids)", T["singleton_by_slot"], 50)
    R.table("Singleton outcomes (exact matching to the 288 legal transforms)", T["singleton_outcomes"], 30)
    R.table("AR vs causal encoder on identical test records", T["paired_ar_vs_causal"], 60)
    bk = T["by_k"]
    if len(bk):
        fig, axes = plt.subplots(1, len(conds), figsize=(4.5 * len(conds), 3.8), squeeze=False)
        for ax, e in zip(axes[0], conds):
            for m, col in zip(LK.MODELS, ("#2a78d6", "#8e6bd6")):
                q = bk[(bk.condition == e) & (bk.model == m)].sort_values("k")
                if not len(q):
                    continue
                ax.plot(q.k, q.exact_acc, marker="o", color=col, label=m)
                ho = q.status != "trained"
                ax.scatter(q.k[ho], q.exact_acc[ho], s=90, facecolors="none", edgecolors="red", zorder=3)
            ax.set(title=f"{e} (S={LK.n_slots(e)})", xlabel="k", ylim=(-0.02, 1.05), xticks=range(LK.n_slots(e) + 1))
            ax.legend(fontsize=7)
        axes[0, 0].set_ylabel("exact accuracy (last checkpoint)")
        fig.suptitle("Red circles = held out (k=1 and k > max trained k)")
        fig.tight_layout()
        R.fig(fig, "Exact accuracy by k", out / "figures" / "by_k.png")
    if compare_subdir:
        # same conditions, same init and logical stream, shorter budget (read-only)
        rows = []
        for e in conds:
            for m in LK.MODELS:
                for label, sd in (("previous budget", compare_subdir), ("this run", subdir)):
                    cfg = make_cfg(e, results_root, mode, [m], seed, sd)
                    af = eval_dir(cfg, m) / "ARTIFACT.json"
                    if not af.exists() or not json.loads(af.read_text()).get("complete"):
                        rows.append({"condition": e, "model": m, "run": label, "status": "not available"})
                        continue
                    a = json.loads(af.read_text())
                    d = load_eval(cfg, m)
                    k_, ex = d["k"].astype(int), d["exact"].astype(bool)
                    s1 = d["kind"] == "single"
                    r = {"condition": e, "model": m, "run": label, "status": "available", "final step": a.get("step"),
                         "best step (ref)": a.get("best_step"),
                         **{f"single {op}": int(ex[s1 & (d["string"] == op)].sum()) for op in "ABC"},
                         "singletons": f"{int(ex[s1].sum())}/{int(s1.sum())}",
                         "trained-k exact": round(float(ex[np.isin(k_, LK.train_ks(e))].mean()), 4)}
                    for kk in LK.heldout_long_ks(e):
                        r[f"k={kk} exact"] = round(float(ex[k_ == kk].mean()), 4)
                    rows.append(r)
        T["budget_comparison"] = pd.DataFrame(rows)
        R.table("Same runs at the previous budget vs this budget (last checkpoint each)", T["budget_comparison"], 40)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, t in T.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    rp = write_review(R.items, out, "AR vs causal encoder (last checkpoint)",
                      "# AR vs causal encoder — review snapshot\n\nBaked outputs; nothing to run.", review_name)
    log(f"REVIEW FILE: {rp}")
    return rp
