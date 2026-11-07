"""Pipeline for the mixed-length (0/2/3/4/5) experiments: manifests -> train (shared, resumable
trainer) -> chunked, restartable evaluation on all 364 words -> integer metric tables.

Results live in a NEW subtree, e.g.
``/content/drive/MyDrive/decompositionality_results/mixed_length_02345/<experiment>/``; legacy results are
never touched.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import mixed as M
from .config import ExperimentConfig
from .data import load_or_build_manifests
from .evaluation import run_model_on_set
from .models.registry import build_model
from .pipeline import Context
from .training import run_config, run_dir, training_complete
from .utils import (atomic_write_bytes, atomic_write_json, dtype_name, environment_info, free_memory,
                    hardware_name, resolve_amp_dtype, resolve_device)


def prepare_mixed(cfg: ExperimentConfig, repo_dir: Optional[str] = None, legacy_manifest: Optional[str] = None,
                  log=print) -> Context:
    """The FULL fixed base-grid manifests (300 val / 1000 test) are regenerated deterministically from the
    seed; they are byte-identical to the legacy manifests when the legacy hash matches (checked if a
    legacy manifest.json path is given)."""
    assert cfg.is_mixed
    mdir = cfg.manifest_dir
    m = load_or_build_manifests(mdir, cfg.train.seed)
    val = M.make_val_set(m.val_grids[: cfg.n_val_grids], cfg.experiment, cfg.train.seed)
    subset = M.subset_by_grids(val, cfg.val_subset_grids)
    test = M.make_test_set(m.test_grids[: cfg.n_test_grids])
    legacy_match = "unavailable (no legacy manifest found)"
    if legacy_manifest and Path(legacy_manifest).exists():
        h = json.loads(Path(legacy_manifest).read_text()).get("combined_hash")
        legacy_match = "identical to legacy manifests" if h == m.manifest_hash else "DIFFERENT from legacy manifests"
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    info = {**cfg.to_dict(), "mixed_data_version": M.MIXED_DATA_VERSION, "manifest_hash": m.manifest_hash,
            "placement": M.placement(cfg.experiment), "sequence_pe": cfg.sequence_pe,
            "legacy_manifest_comparison": legacy_match, "n_val_records": len(val["x"]),
            "n_subset_records": len(subset["x"]), "n_test_records": len(test["x"]),
            "n_training_words": len(cfg.train_strings)}
    atomic_write_json(cfg.experiment_dir / "experiment_config.json", info)
    log(f"Manifests {mdir} (hash {m.manifest_hash[:12]}; {legacy_match}) | validation {len(val['x'])} records "
        f"({cfg.n_val_grids} grids x 5 lengths) | fixed validation subset {len(subset['x'])} | test {len(test['x']):,} "
        f"({cfg.n_test_grids} grids x 364 words)")
    ctx = Context(cfg, m, val, subset, test, repo_dir)
    ctx.legacy_manifest_comparison = legacy_match
    return ctx


def eval_dir(cfg: ExperimentConfig, model_id: str) -> Path:
    return cfg.experiment_dir / "eval" / model_id


def evaluation_complete(cfg, model_id, rc) -> bool:
    f = eval_dir(cfg, model_id) / "EVAL_COMPLETE.json"
    if not f.exists():
        return False
    info = json.loads(f.read_text())
    need = ["test_predictions.npz", "test_per_word.csv", "test_groups.csv", "eval_summary.json"]
    return info.get("config_hash") == rc["config_hash"] and all((f.parent / n).exists() for n in need)


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def evaluate_mixed_model(model_id: str, ctx: Context, log=print, chunk_grids: int = 100, force: bool = False) -> dict:
    """Evaluate the SELECTED checkpoint (best.pt) on all 364 words x test grids, in restartable chunks of
    ``chunk_grids`` base grids. AR models decode greedily free-running from the prefix; others argmax."""
    cfg = ctx.cfg
    rc = run_config(cfg, model_id, ctx.manifests.manifest_hash)
    edir = eval_dir(cfg, model_id)
    if not force and evaluation_complete(cfg, model_id, rc):
        log(f"[{model_id}] evaluation already complete -> skipping")
        return json.loads((edir / "eval_summary.json").read_text())
    if not training_complete(cfg, model_id, rc):
        raise RuntimeError(f"[{model_id}] training not complete")
    ck = torch.load(run_dir(cfg, model_id) / "best.pt", map_location="cpu", weights_only=False)
    if ck["config_hash"] != rc["config_hash"] or ck.get("n_slots", 3) != cfg.n_slots:
        raise RuntimeError("best.pt does not match this experiment's configuration / slot count")
    tc = cfg.train
    device = resolve_device(tc.device)
    amp = resolve_amp_dtype(device, tc.dtype)
    model = build_model(model_id, rc["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    es = ctx.test_set
    n_grids = int(es["grid_id"].max()) + 1
    cdir = edir / "chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts = []
    starts = list(range(0, n_grids, chunk_grids))
    for ci, g0 in enumerate(starts):
        f = cdir / f"chunk_{ci:03d}.npz"
        sel = (es["grid_id"] >= g0) & (es["grid_id"] < g0 + chunk_grids)
        if f.exists():
            with np.load(f) as z:
                if str(z["config_hash"]) == rc["config_hash"] and int(z["n"]) == int(sel.sum()):
                    parts.append({k: z[k] for k in z.files})
                    continue
        sub = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in es.items()}
        r = run_model_on_set(model, sub, device, amp, tc.eval_batch_size)
        part = {"pred": r["pred"], "valid": r["valid"], "cell_ce": r["cell_ce"],
                "eos_ok": r.get("eos_ok", np.ones(len(r["valid"]), bool)), "n": np.array(int(sel.sum())),
                "config_hash": np.array(rc["config_hash"])}
        atomic_write_bytes(f, _npz_bytes(**part))
        parts.append(part)
        done = ci + 1
        el = time.time() - t0
        log(f"  [{model_id}] test chunk {done}/{len(starts)} ({int(sel.sum()):,} records) | elapsed {el / 60:.1f} min")
    pred = np.concatenate([p["pred"] for p in parts])
    valid = np.concatenate([p["valid"] for p in parts])
    cell_ce = np.concatenate([p["cell_ce"] for p in parts])
    eos_ok = np.concatenate([p["eos_ok"] for p in parts])
    t_test = time.time() - t0
    # full validation set with the selected checkpoint (for the record; selection already happened)
    rv = run_model_on_set(model, ctx.val_set, device, amp, tc.eval_batch_size)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    del model
    free_memory()
    Y = es["y"].reshape(-1, 36)
    cc = (pred.reshape(-1, 36).astype(np.int16) == Y.astype(np.int16)).sum(1)
    exact = valid & (cc == 36)
    arrays = {"pred": pred.reshape(-1, 6, 6).astype(np.int8), "target": es["y"].reshape(-1, 6, 6).astype(np.int8),
              "input": es["x"].reshape(-1, 6, 6).astype(np.int8), "grid_id": es["grid_id"].astype(np.int32),
              "string": es["string"].astype("U5"), "length": es["length"].astype(np.int8),
              "ops": es["ops"].astype(np.int8), "valid": valid, "cell_correct": cc.astype(np.int16), "exact": exact,
              "cell_ce": cell_ce.astype(np.float32)}
    if model_id in ("transformer_ar", "looped_transformer_ar"):
        arrays["eos_ok"] = eos_ok
    atomic_write_bytes(edir / "test_predictions.npz", _npz_bytes(**arrays))
    pw, gr, summ = M.metrics_tables(es["string"], pred, valid, es["y"], cfg.experiment)
    for t in (pw, gr):
        t.insert(0, "model", model_id)
        t.insert(0, "experiment", cfg.experiment)
    pw.to_csv(edir / "test_per_word.csv", index=False)
    gr.to_csv(edir / "test_groups.csv", index=False)
    vp = (rv["pred"].reshape(-1, 36).astype(np.int16) == ctx.val_set["y"].reshape(-1, 36)).sum(1)
    vex = rv["valid"] & (vp == 36)
    val_by_len = {int(L): {"exact": float(vex[ctx.val_set["length"] == L].mean()),
                           "cell_ce": float(rv["cell_ce"][ctx.val_set["length"] == L].mean())}
                  for L in np.unique(ctx.val_set["length"])}
    dec = "greedy free-running (+1 EOS diagnostic step)" if model_id in ("transformer_ar", "looped_transformer_ar") \
        else ("parallel argmax of final refinement" if model_id == "trm" else "parallel argmax")
    summary = {"model_id": model_id, "experiment": cfg.experiment, "seed": tc.seed, "n_slots": cfg.n_slots,
               "selected_step": ck["step"], "selected_val_cell_ce": ck["val_cell_ce"],
               "selection_rule": "lowest length-macro validation final-answer cell CE over lengths 0/2/3/4/5 "
                                 "(ties -> earlier); singles/unseen pairs never used",
               "test": {"n_records": int(len(exact)), **summ}, "val_by_length": val_by_len,
               "test_eval_seconds": t_test, "peak_gpu_memory_gb_eval": peak, "hardware": hardware_name(device),
               "dtype": dtype_name(amp), "decoding": dec, "config_hash": rc["config_hash"],
               "manifest_hash": ctx.manifests.manifest_hash, "repo_commit": environment_info(ctx.repo_dir)["repo_commit"]}
    atomic_write_json(edir / "eval_summary.json", summary)
    atomic_write_json(edir / "EVAL_COMPLETE.json", {"config_hash": rc["config_hash"], "selected_step": ck["step"]})
    g = gr.set_index("group")
    log(f"[{model_id}] selected step {ck['step']} | singles A/B/C exact "
        + "/".join(f"{int(g.loc[f'single_{s}', 'exact_correct'])}" for s in "ABC")
        + f" of {int(g.loc['single_A', 'n'])} each | EMPTY {g.loc['EMPTY', 'exact_acc']:.4f} | "
          f"seen pairs {g.loc['pairs_seen', 'exact_acc']:.4f} | eval {t_test / 60:.1f} min")
    return summary
