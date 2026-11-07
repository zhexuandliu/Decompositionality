"""Placement probe (INFERENCE ONLY) for the mixed-length experiments.

Question: does a model's behaviour on a single operation depend on WHERE that operation sits among the
five slots? E.g. canonical test input [X, PAD, PAD, PAD, PAD] vs [PAD, PAD, X, PAD, PAD].

* Loads each run's already-SELECTED checkpoint (runs/<model>/best.pt); never trains, never re-selects.
* model.eval(), no gradients, frozen parameters, no optimizer; AR models decode free-running.
* Probe set A: singles A/B/C at each of the 5 slot positions (others PAD) on the test base grids.
* Probe set B (control): the experiment's TRAINED length-two pairs in all C(5,2)=10 order-preserving
  placements (all of which occur in training) on a subset of grids.
* Each prediction is compared with the truth and with ALL 288 legal transformations; explicit candidate
  rules are reported (e.g. "X then C", "X twice", identity, each original training pair).
Outputs go to <mixed_root>/placement_probe/<experiment>/<model>/ and never touch training/eval artifacts.
"""
from __future__ import annotations

import io
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import mixed as M
from . import operations as O
from .diagnosis import group as G

PROBE_VERSION = "placement-probe-v1"


def make_probe_set(test_grids: np.ndarray, experiment: str, n_single_grids: int, n_pair_grids: int) -> Dict[str, np.ndarray]:
    rows = []  # (grid_id, word, positions, kind)
    for g in range(n_single_grids):
        for w in "ABC":
            for p in range(M.N_SLOTS):
                rows.append((g, w, (p,), "single"))
    for g in range(n_pair_grids):
        for w in M.pair_set(experiment):
            for pos in itertools.combinations(range(M.N_SLOTS), 2):
                rows.append((g, w, pos, "pair_control"))
    gid = np.array([r[0] for r in rows])
    x = test_grids[gid].reshape(len(rows), 6, 6).astype(np.uint8)
    words = [r[1] for r in rows]
    y = M.apply_word_batch(x, words).astype(np.uint8)
    ops = np.array([M.slots_for(r[1], r[2]) for r in rows], dtype=np.int64)
    pos_label = np.array(["slots " + ",".join(str(p + 1) for p in r[2]) for r in rows])
    canonical = np.array([tuple(r[2]) == tuple(range(len(r[1]))) for r in rows])
    return {"x": x, "y": y, "ops": ops, "grid_id": gid, "string": np.array(words), "positions": pos_label,
            "kind": np.array([r[3] for r in rows]), "canonical": canonical}


def load_selected_model(exp_dir: Path, model_id: str, device):
    import torch
    from .models.registry import build_model
    rdir = exp_dir / "runs" / model_id
    rc = json.loads((rdir / "run_config.json").read_text())
    ck = torch.load(rdir / "best.pt", map_location="cpu", weights_only=False)
    if ck.get("config_hash") != rc.get("config_hash"):
        raise RuntimeError(f"{exp_dir.name}/{model_id}: best.pt does not match run_config.json")
    if ck.get("n_slots", 3) != M.N_SLOTS:
        raise RuntimeError(f"{exp_dir.name}/{model_id}: checkpoint has {ck.get('n_slots', 3)} slots, expected {M.N_SLOTS}")
    model = build_model(model_id, rc["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"selected_step": ck.get("step"), "config_hash": ck.get("config_hash")}


def analyse(es: Dict[str, np.ndarray], pred: np.ndarray, valid: np.ndarray, experiment: str, cat: G.Catalog):
    """Per (word, placement): exact counts and exact compatibility with explicit candidate transformations."""
    import pandas as pd
    P = pred.reshape(len(pred), 36).astype(np.int16)
    X = es["x"].reshape(len(pred), 36).astype(np.int16)
    Y = es["y"].reshape(len(pred), 36).astype(np.int16)
    exact = valid & (P == Y).all(1)
    agree = (X[:, cat.perms] == P[:, None, :]).sum(-1)          # [N, 288]
    full = agree == 36
    n_exact = full.sum(1)
    uniq = np.where(n_exact == 1, full.argmax(1), -1)
    pairs = M.pair_set(experiment)
    rows = []
    keys = pd.DataFrame({"kind": es["kind"], "string": es["string"], "positions": es["positions"],
                         "canonical": es["canonical"]})
    for (kind, w, pos, canon), idx in keys.groupby(["kind", "string", "positions", "canonical"], sort=False).groups.items():
        idx = np.asarray(idx)
        k = len(idx)
        cands = {"truth": w, "X then C": w + "C", "X twice": w + w, "identity": "",
                 **{f"pair {q}": q for q in pairs}}
        r = {"kind": kind, "string": w, "positions": pos, "canonical_placement": bool(canon), "n": k,
             "exact_correct": int(exact[idx].sum()), "exact_acc": exact[idx].mean(),
             "cell_acc": (P[idx] == Y[idx]).mean(), "invalid_rate": (~valid[idx]).mean(),
             "any_legal_match_rate": (n_exact[idx] > 0).mean()}
        for name, cw in cands.items():
            if kind == "pair_control" and name in ("X then C", "X twice"):
                continue
            r[f"match: {name}"] = full[idx, cat.word_id(cw)].mean()
        u = uniq[idx]
        wrong_u = u[(u >= 0) & ~exact[idx]]
        if len(wrong_u):
            vals, cnts = np.unique(wrong_u, return_counts=True)
            j = int(np.argmax(cnts))
            r["dominant_wrong_transform"] = cat.label(int(vals[j]))
            r["dominant_wrong_share"] = cnts[j] / k
        else:
            r["dominant_wrong_transform"] = ""
            r["dominant_wrong_share"] = 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def run_probe(mixed_root: str, experiments: List[str], models: List[str], out_root: Optional[str] = None,
              n_single_grids: int = 1000, n_pair_grids: int = 200, device: str = "auto", dtype: str = "auto",
              eval_batch_size: Optional[int] = None, log=print):
    import pandas as pd
    import torch
    from .data import load_or_build_manifests
    from .evaluation import run_model_on_set
    from .utils import atomic_write_bytes, atomic_write_json, free_memory, resolve_amp_dtype, resolve_device

    root = Path(mixed_root)
    out_root = Path(out_root) if out_root else root / "placement_probe"
    dev = resolve_device(device)
    amp = resolve_amp_dtype(dev, dtype)
    cat = G.Catalog()
    tables = []
    for exp in experiments:
        exp_dir = root / exp
        if not (exp_dir / "manifests" / "heldout_grids.npz").exists():
            log(f"[{exp}] no manifests found at {exp_dir} -> experiment skipped (reported as missing)")
            continue
        with np.load(exp_dir / "manifests" / "heldout_grids.npz") as z:
            test_grids = z["test_grids"]
        es = make_probe_set(test_grids, exp, min(n_single_grids, len(test_grids)), min(n_pair_grids, len(test_grids)))
        for m in models:
            odir = out_root / exp / m
            if not (exp_dir / "runs" / m / "best.pt").exists():
                log(f"[{exp}/{m}] no selected checkpoint -> MISSING (not zero)")
                continue
            f = odir / "probe_predictions.npz"
            meta_f = odir / "probe_meta.json"
            model, ckinfo = None, None
            if f.exists() and meta_f.exists():
                meta = json.loads(meta_f.read_text())
                with np.load(f) as z:
                    ok = meta.get("probe_version") == PROBE_VERSION and int(z["pred"].shape[0]) == len(es["x"])
                    if ok:
                        pred, valid = z["pred"], z["valid"]
                        log(f"[{exp}/{m}] cached probe predictions loaded")
            else:
                ok = False
            if not ok:
                t = time.time()
                model, ckinfo = load_selected_model(exp_dir, m, dev)
                r = run_model_on_set(model, es, dev, amp, eval_batch_size, want_ce=False, want_pred=True)
                pred, valid = r["pred"].reshape(-1, 6, 6), r["valid"]
                buf = io.BytesIO()
                np.savez_compressed(buf, pred=pred, valid=valid, input=es["x"], target=es["y"], ops=es["ops"].astype(np.int8),
                                    grid_id=es["grid_id"], string=es["string"], positions=es["positions"], kind=es["kind"])
                atomic_write_bytes(f, buf.getvalue())
                atomic_write_json(meta_f, {"probe_version": PROBE_VERSION, "experiment": exp, "model": m, **ckinfo,
                                           "checkpoint": str(exp_dir / "runs" / m / "best.pt"), "n_records": len(pred),
                                           "seconds": time.time() - t, "inference_only": True, "optimizer_created": False})
                log(f"[{exp}/{m}] probed {len(pred):,} records (selected step {ckinfo['selected_step']}) in {time.time() - t:.0f}s")
                del model
                free_memory()
            df = analyse(es, pred, valid, exp, cat)
            df.insert(0, "model", m)
            df.insert(0, "experiment", exp)
            tables.append(df)
    res = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out_root.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_root / "placement_probe_results.csv", index=False)
    return res, out_root


def plot_probe(res, out_dir: Path):
    """Heatmaps (rows = models; columns = single word x slot position) per experiment."""
    from .names import DISPLAY_NAMES, MODEL_IDS
    from .plotstyle import _plt, save_fig
    plt = _plt()
    paths = []
    singles = res[res.kind == "single"]
    if not len(singles):
        return paths
    exps = list(dict.fromkeys(singles.experiment))
    measures = [("exact_acc", "TRUE exact accuracy", "Blues"),
                ("match: X then C", "prediction exactly equals 'X then C'", "Oranges"),
                ("match: X twice", "prediction exactly equals 'X twice'", "Purples"),
                ("match: identity", "prediction exactly equals the INPUT (identity)", "Greens")]
    cols = [(w, f"slots {p + 1}") for w in "ABC" for p in range(5)]
    for e in exps:
        sub = singles[singles.experiment == e]
        models = [m for m in MODEL_IDS if m in set(sub.model)]
        fig, axes = plt.subplots(len(measures), 1, figsize=(15, (1.2 + 0.42 * len(models)) * len(measures) + 1), squeeze=False)
        for ax, (col, lab, cmap) in zip(axes[:, 0], measures):
            mat = np.full((len(models), len(cols)), np.nan)
            for i, m in enumerate(models):
                s = sub[sub.model == m].set_index(["string", "positions"])
                for j, key in enumerate(cols):
                    if key in s.index:
                        mat[i, j] = s.loc[key, col]
            ax.grid(False)
            ax.imshow(np.nan_to_num(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, "n/a" if np.isnan(mat[i, j]) else f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if (not np.isnan(mat[i, j]) and mat[i, j] > 0.6) else "#0b0b0b")
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"{w}\nslot {p[-1]}" + (" (canonical)" if p == "slots 1" else "") for w, p in cols], fontsize=7)
            for b in (5, 10):
                ax.axvline(b - 0.5, color="#0b0b0b", lw=1.4)
            ax.set_yticks(range(len(models))); ax.set_yticklabels([DISPLAY_NAMES[m] for m in models], fontsize=8)
            ax.set_title(f"{e}: {lab} - single operation placed in each of the 5 slots (other slots PAD)", fontsize=9)
        fig.tight_layout()
        paths += save_fig(fig, out_dir, f"placement_probe_singles_{e}", sub)
    ctrl = res[res.kind == "pair_control"]
    if len(ctrl):
        fig, axes = plt.subplots(1, len(exps), figsize=(8 * len(exps), 4.5), squeeze=False)
        for ax, e in zip(axes[0], exps):
            c = ctrl[ctrl.experiment == e]
            models = [m for m in MODEL_IDS if m in set(c.model)]
            g = c.groupby(["model", "canonical_placement"]).apply(lambda d: d.exact_correct.sum() / d.n.sum()).unstack()
            xs = np.arange(len(models))
            for k, (canon, lab) in enumerate(((True, "canonical slots 1,2"), (False, "other 9 placements"))):
                vals = [g.loc[m, canon] if m in g.index and canon in g.columns else np.nan for m in models]
                ax.bar(xs + (k - 0.5) * 0.38, vals, width=0.36, label=lab, color=["#2a78d6", "#eb6834"][k])
            ax.set_xticks(xs); ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=40, ha="right", fontsize=7)
            ax.set(ylim=(0, 1.05), ylabel="exact accuracy", title=f"{e}: CONTROL - trained pairs in all 10 placements")
            ax.legend(fontsize=8)
        fig.tight_layout()
        paths += save_fig(fig, out_dir, "placement_probe_pair_control", ctrl)
    return paths
