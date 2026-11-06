"""High-level pipeline shared by both notebooks and the CLI:
manifests -> train (resumable) -> evaluate selected checkpoints -> reports/plots -> archive."""
from __future__ import annotations

import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from . import operations as O
from .config import ExperimentConfig
from .data import Manifests, load_or_build_manifests, make_eval_set, subset_eval_set
from .evaluation import run_model_on_set, save_eval_outputs
from .models.registry import build_model
from .training import run_config, run_dir, train_model, training_complete
from .utils import (atomic_write_json, dtype_name, environment_info, free_memory, hardware_name,
                    resolve_amp_dtype, resolve_device)


@dataclass
class Context:
    cfg: ExperimentConfig
    manifests: Manifests
    val_set: Dict[str, np.ndarray]
    subset_set: Dict[str, np.ndarray]
    test_set: Dict[str, np.ndarray]
    repo_dir: Optional[str] = None


def prepare(cfg: ExperimentConfig, repo_dir: Optional[str] = None, log=print) -> Context:
    """Build/load the FULL fixed manifests (300 val / 1000 test base grids, shared by all
    models and both experiments). Smoke mode evaluates on a prefix of them; the training
    stream always excludes the full held-out sets."""
    mdir = cfg.experiment_dir / "manifests"
    m = load_or_build_manifests(mdir, cfg.train.seed)
    assert cfg.n_val_grids <= len(m.val_grids) and cfg.n_test_grids <= len(m.test_grids)
    val = make_eval_set(m.val_grids[: cfg.n_val_grids], cfg.train_strings, "val")
    subset = subset_eval_set(val, cfg.val_subset_grids)
    test = make_eval_set(m.test_grids[: cfg.n_test_grids], O.TEST_STRINGS, "test")
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json", cfg.to_dict())
    log(f"Manifests: {mdir} (hash {m.manifest_hash[:12]}) | val {len(val['x'])} examples "
        f"({cfg.n_val_grids} grids x {len(cfg.train_strings)} seen strings) | fixed val subset "
        f"{len(subset['x'])} | test {len(test['x'])} ({cfg.n_test_grids} grids x 39 strings)")
    return Context(cfg, m, val, subset, test, repo_dir)


def eval_dir(cfg: ExperimentConfig, model_id: str) -> Path:
    return cfg.experiment_dir / "eval" / model_id


def evaluation_complete(cfg: ExperimentConfig, model_id: str, rc: dict) -> bool:
    f = eval_dir(cfg, model_id) / "EVAL_COMPLETE.json"
    if not f.exists():
        return False
    info = json.loads(f.read_text())
    needed = ["test_predictions.npz", "test_per_string.csv", "test_groups.csv", "val_per_string.csv",
              "eval_summary.json"]
    return info.get("config_hash") == rc["config_hash"] and all((f.parent / n).exists() for n in needed)


def evaluate_model(model_id: str, ctx: Context, log=print, force: bool = False) -> dict:
    cfg = ctx.cfg
    rc = run_config(cfg, model_id, ctx.manifests.manifest_hash)
    edir = eval_dir(cfg, model_id)
    if not force and evaluation_complete(cfg, model_id, rc):
        log(f"[{model_id}] evaluation already complete -> skipping")
        return json.loads((edir / "eval_summary.json").read_text())
    if not training_complete(cfg, model_id, rc):
        raise RuntimeError(f"[{model_id}] training not complete; train before evaluating")
    rdir = run_dir(cfg, model_id)
    ck = torch.load(rdir / "best.pt", map_location="cpu", weights_only=False)
    if ck["config_hash"] != rc["config_hash"]:
        raise RuntimeError("best.pt config hash mismatch")
    tc = cfg.train
    device = resolve_device(tc.device)
    amp_dtype = resolve_amp_dtype(device, tc.dtype)
    model = build_model(model_id, rc["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen parameters; no test-time gradients
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    meta = {"experiment": cfg.experiment, "model_id": model_id, "seed": tc.seed}
    t = time.time()
    rv = run_model_on_set(model, ctx.val_set, device, amp_dtype, tc.eval_batch_size)
    val_sum = save_eval_outputs(edir, ctx.val_set, rv, meta, cfg.train_strings, "val")
    t_val = time.time() - t
    t = time.time()
    rt = run_model_on_set(model, ctx.test_set, device, amp_dtype, tc.eval_batch_size)
    test_sum = save_eval_outputs(edir, ctx.test_set, rt, meta, cfg.train_strings, "test")
    t_test = time.time() - t
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    summary = {
        "model_id": model_id, "experiment": cfg.experiment, "seed": tc.seed,
        "selected_step": ck["step"], "selected_val_cell_ce": ck["val_cell_ce"],
        "selection_rule": "lowest seen-pair validation cell CE (ties -> earlier step)",
        "val": val_sum, "test": test_sum, "val_eval_seconds": t_val, "test_eval_seconds": t_test,
        "peak_gpu_memory_gb_eval": peak, "hardware": hardware_name(device), "dtype": dtype_name(amp_dtype),
        "decoding": model.cost_info()["decoding"],
        "cell_ce_label": ("teacher-forced answer-cell CE (diagnostic; not free-running)" if model.kind == "ar"
                          else "final-refinement cell CE" if model.kind == "trm" else "cell CE"),
        "config_hash": rc["config_hash"], "repo_commit": environment_info(ctx.repo_dir)["repo_commit"],
    }
    atomic_write_json(edir / "eval_summary.json", summary)
    atomic_write_json(edir / "EVAL_COMPLETE.json", {"config_hash": rc["config_hash"], "selected_step": ck["step"]})
    log(f"[{model_id}] selected step {ck['step']} | val exact {val_sum['exact_acc']:.3f} | "
        f"test exact {test_sum['exact_acc']:.3f} cell {test_sum['cell_acc']:.3f} | "
        f"eval {t_val + t_test:.0f}s")
    del model
    free_memory()
    return summary


def train_and_evaluate(ctx: Context, models: Optional[List[str]] = None, log=print, evaluate: bool = True) -> dict:
    """Train (or resume/skip) then evaluate each selected model sequentially, releasing
    GPU memory between models. A failure in one model is reported and re-raised at the end
    rather than silently skipped."""
    out, failures = {}, {}
    for mid in models or ctx.cfg.models:
        log(f"\n=== {mid} ===")
        try:
            tr = train_model(mid, ctx.cfg, ctx.manifests, ctx.val_set, ctx.subset_set, log=log, repo_dir=ctx.repo_dir)
            ev = evaluate_model(mid, ctx, log=log) if evaluate else None
            out[mid] = {"train": tr, "eval": ev}
        except Exception as exc:  # report, keep going, then fail loudly
            import traceback
            traceback.print_exc()
            failures[mid] = repr(exc)
        finally:
            free_memory()
    if failures:
        raise RuntimeError(f"models failed: {failures}")
    return out


def make_archive(cfg: ExperimentConfig, log=print) -> Path:
    """Compact results archive: everything except checkpoints (*.pt)."""
    root = cfg.experiment_dir
    dst = root / f"results_{cfg.experiment}{'_smoke' if cfg.mode == 'smoke' else ''}.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix != ".pt" and p != dst and not p.name.endswith(".zip") and ".tmp" not in p.name:
                zf.write(p, p.relative_to(root.parent))
    log(f"Results archive: {dst} ({dst.stat().st_size / 2**20:.1f} MiB)")
    return dst
