"""Pipeline for the identity / variable_length experiments (notebooks 10/11): shared manifests, chunked restartable
evaluation of the SELECTED checkpoint, integer metric tables, record audits and an artifact index.

Results live under ``<RESULTS_ROOT>/identity_varlen_02345_all_models/`` with ``manifests/``, ``identity/<model>``,
``variable_length/<model>`` and ``comparisons/``. Every completed evaluation writes ``eval/<model>/ARTIFACT.json``;
``build_artifact_index`` scans these (never guessing paths) into ``artifact_index.json`` for consumers.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from . import idvar as V
from .config import ExperimentConfig
from .data import load_or_build_manifests
from .evaluation import run_model_on_set
from .models.registry import build_model
from .pipeline import Context
from .training import run_config, run_dir, training_complete
from .utils import (atomic_write_bytes, atomic_write_json, dtype_name, environment_info, free_memory, hardware_name,
                    resolve_amp_dtype, resolve_device)
from .words import from_label

AR_MODELS = ("transformer_ar", "looped_transformer_ar")


def prepare_idvar(cfg: ExperimentConfig, repo_dir: Optional[str] = None, legacy_manifest: Optional[str] = None,
                  log=print) -> Context:
    assert cfg.is_idvar
    cond = V.condition_of(cfg.experiment)
    m = load_or_build_manifests(cfg.manifest_dir, cfg.train.seed)
    val = V.make_val_set(m.val_grids[: cfg.n_val_grids], cfg.train.seed, cond)
    subset = V.subset_by_grids(val, cfg.val_subset_grids)
    test = V.make_test_set(m.test_grids[: cfg.n_test_grids], cfg.train.seed, cond)
    assert len(test["x"]) == V.expected_test_records(cond, cfg.n_test_grids)
    assert not (val["k"] == 1).any() and set(np.unique(val["k"])) == set(V.K_VALUES)
    legacy = "unavailable (no legacy manifest found)"
    if legacy_manifest and Path(legacy_manifest).exists():
        h = json.loads(Path(legacy_manifest).read_text()).get("combined_hash")
        legacy = "identical to legacy manifests" if h == m.manifest_hash else "DIFFERENT from legacy manifests"
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json",
                      {**cfg.to_dict(), "condition": cond, "idvar_data_version": V.IDVAR_DATA_VERSION,
                       "manifest_hash": m.manifest_hash, "legacy_manifest_comparison": legacy,
                       "n_val_records": len(val["x"]), "n_subset_records": len(subset["x"]),
                       "n_test_records": len(test["x"])})
    log(f"[{cond}] manifests {cfg.manifest_dir} (hash {m.manifest_hash[:12]}; {legacy}) | validation {len(val['x'])} "
        f"| subset {len(subset['x'])} | test {len(test['x']):,}")
    return Context(cfg, m, val, subset, test, repo_dir)


def grid_hashes(x: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(np.asarray(x).reshape(len(x), 36).astype(np.uint8))
    return np.array([hashlib.sha256(r.tobytes()).hexdigest()[:16] for r in flat])


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def eval_dir(cfg, model_id) -> Path:
    return cfg.experiment_dir / "eval" / model_id


def evaluation_complete(cfg, model_id, rc) -> bool:
    f = eval_dir(cfg, model_id) / "ARTIFACT.json"
    if not f.exists():
        return False
    a = json.loads(f.read_text())
    return (a.get("complete") and a.get("config_hash") == rc["config_hash"]
            and (eval_dir(cfg, model_id) / "test_predictions.npz").exists())


def metrics_tables(arr: Dict[str, np.ndarray], condition: str):
    """Integer exact successes / denominators by word, k, singleton operation (x position for identity)."""
    import pandas as pd
    P = arr["pred"].reshape(len(arr["pred"]), 36).astype(np.int16)
    Y = arr["target"].reshape(len(arr["target"]), 36).astype(np.int16)
    valid = arr["valid"].astype(bool)
    cc = (P == Y).sum(1)
    exact = valid & (cc == 36)
    df = pd.DataFrame({"string": arr["string"].astype(str), "k": arr["k"], "kind": arr["kind"].astype(str),
                       "position": arr["position"], "exact": exact, "cell": cc / 36, "invalid": ~valid})

    def agg(g):
        return pd.Series({"exact_correct": int(g.exact.sum()), "n": len(g), "exact_acc": g.exact.mean(),
                          "cell_acc": g.cell.mean(), "invalid_rate": g.invalid.mean()})
    by_word = df[df.kind == "seen"].groupby("string", sort=False).apply(agg).reset_index()
    by_k = df.groupby("k").apply(agg).reset_index()
    by_k["status"] = np.where(by_k.k == 1, "HELD OUT (singletons)", "seen (trained lengths, new grids)")
    singles = df[df.kind == "single"]
    if condition == "identity":
        by_single = singles.groupby(["string", "position"]).apply(agg).reset_index()
    else:
        by_single = singles.groupby("string").apply(agg).reset_index()
    per_op = singles.groupby("string").apply(agg).reset_index()
    if condition == "identity":
        rng = by_single.groupby("string")["exact_acc"].agg(["min", "max"]).reset_index()
        per_op = per_op.merge(rng.rename(columns={"min": "min_over_positions", "max": "max_over_positions"}), on="string")
    summ = {"singleton_macro_exact": float(by_single["exact_acc"].mean()),
            "empty_exact": float(df[df.string == "EMPTY"].exact.mean())}
    return by_word, by_k, by_single, per_op, summ


def evaluate_idvar_model(model_id: str, ctx: Context, log=print, chunk_grids: int = 100, force: bool = False) -> dict:
    cfg = ctx.cfg
    cond = V.condition_of(cfg.experiment)
    rc = run_config(cfg, model_id, ctx.manifests.manifest_hash)
    edir = eval_dir(cfg, model_id)
    if not force and evaluation_complete(cfg, model_id, rc):
        log(f"[{cond}/{model_id}] evaluation already complete -> skipping")
        return json.loads((edir / "eval_summary.json").read_text())
    if not training_complete(cfg, model_id, rc):
        raise RuntimeError(f"[{cond}/{model_id}] training not complete")
    ck = torch.load(run_dir(cfg, model_id) / "best.pt", map_location="cpu", weights_only=False)
    if ck["config_hash"] != rc["config_hash"]:
        raise RuntimeError("best.pt does not match this run's configuration")
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
    parts, starts = [], list(range(0, n_grids, chunk_grids))
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
        part = {"pred": r["pred"], "valid": r["valid"], "cell_ce": r["cell_ce"], "n": np.array(int(sel.sum())),
                "config_hash": np.array(rc["config_hash"])}
        if "raw_tokens" in r:
            part["raw_tokens"] = r["raw_tokens"]
        if "eos_ok" in r:
            part["eos_ok"] = r["eos_ok"]
        atomic_write_bytes(f, _npz_bytes(**part))
        parts.append(part)
        log(f"  [{cond}/{model_id}] chunk {ci + 1}/{len(starts)} ({int(sel.sum()):,} records) | "
            f"{(time.time() - t0) / 60:.1f} min")
    t_test = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    del model
    free_memory()
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    pred, valid = cat("pred"), cat("valid")
    Y = es["y"].reshape(-1, 36)
    cc = (pred.reshape(-1, 36).astype(np.int16) == Y.astype(np.int16)).sum(1)
    op_len = np.full(len(pred), 5) if cond == "identity" else es["k"]
    arrays = {"pred": pred.reshape(-1, 6, 6).astype(np.int8), "target": es["y"].reshape(-1, 6, 6).astype(np.int8),
              "input": es["x"].reshape(-1, 6, 6).astype(np.int8), "grid_id": es["grid_id"].astype(np.int32),
              "grid_hash": grid_hashes(es["x"]), "string": es["string"].astype("U5"), "k": es["k"].astype(np.int8),
              "kind": es["kind"].astype("U6"), "position": es["position"].astype(np.int8),
              "slots5": es["slots5"].astype(np.int8), "op_input_length": op_len.astype(np.int8),
              "valid": valid, "cell_correct": cc.astype(np.int16), "exact": valid & (cc == 36),
              "cell_ce": cat("cell_ce").astype(np.float32)}
    if "raw_tokens" in parts[0]:
        arrays["raw_tokens"] = cat("raw_tokens").astype(np.int16)
        arrays["eos_ok"] = cat("eos_ok")
    pfile = edir / "test_predictions.npz"
    atomic_write_bytes(pfile, _npz_bytes(**arrays))
    by_word, by_k, by_single, per_op, summ = metrics_tables(arrays, cond)
    for name, t in (("test_by_word", by_word), ("test_by_k", by_k), ("test_singletons", by_single),
                    ("test_singletons_per_op", per_op)):
        t.insert(0, "model", model_id)
        t.insert(0, "condition", cond)
        t.to_csv(edir / f"{name}.csv", index=False)
    summary = {"model_id": model_id, "condition": cond, "seed": tc.seed, "selected_step": ck["step"],
               "selected_val_cell_ce": ck["val_cell_ce"],
               "selection_rule": "lowest k-macro validation final-answer digit CE over k in 0/2/3/4/5 (ties -> earlier)",
               "n_records": int(len(pred)), "expected_records": V.expected_test_records(cond, n_grids), **summ,
               "test_eval_seconds": t_test, "peak_gpu_memory_gb_eval": peak, "hardware": hardware_name(device),
               "dtype": dtype_name(amp), "config_hash": rc["config_hash"], "manifest_hash": ctx.manifests.manifest_hash,
               "repo_commit": environment_info(ctx.repo_dir)["repo_commit"]}
    atomic_write_json(edir / "eval_summary.json", summary)
    atomic_write_json(edir / "ARTIFACT.json", {
        "condition": cond, "model": model_id, "complete": len(pred) == summary["expected_records"],
        "predictions": str(pfile), "predictions_sha256": sha256_file(pfile), "n_records": int(len(pred)),
        "expected_records": summary["expected_records"], "chunks": len(starts), "selected_step": ck["step"],
        "config_hash": rc["config_hash"], "manifest_hash": ctx.manifests.manifest_hash})
    po = per_op.set_index("string")
    log(f"[{cond}/{model_id}] step {ck['step']} | singletons exact " +
        " ".join(f"{s}={int(po.loc[s, 'exact_correct'])}/{int(po.loc[s, 'n'])}" for s in "ABC") +
        f" | EMPTY {summ['empty_exact']:.4f} | eval {t_test / 60:.1f} min")
    return summary


def build_artifact_index(root: Path) -> dict:
    """Scan <root>/<condition>/eval/<model>/ARTIFACT.json (never guess paths) into <root>/artifact_index.json."""
    root = Path(root)
    idx = {}
    for cond in ("identity", "variable_length"):
        for f in sorted((root / cond / "eval").glob("*/ARTIFACT.json")):
            a = json.loads(f.read_text())
            idx[f"{cond}/{a['model']}"] = a
    atomic_write_json(root / "artifact_index.json", idx)
    return idx


def audit_predictions(npz_path: Path, condition: str, manifest_test_grids: Optional[np.ndarray] = None) -> dict:
    """Counts, duplicates, target recomputation with the INDEPENDENT oracle, input/manifest alignment, token validity,
    instruction-length consistency, stored-metric consistency."""
    from .diagnosis.group import oracle_apply_string
    with np.load(npz_path) as z:
        a = {k: z[k] for k in z.files}
    n = len(a["pred"])
    n_grids = int(a["grid_id"].max()) + 1
    keys = np.char.add(np.char.add(a["string"].astype("U5"), "@"), np.char.add(a["position"].astype(str),
                                                                              np.char.add(":", a["grid_id"].astype(str))))
    tgt = np.empty_like(a["target"])
    for s in np.unique(a["string"]):
        m = a["string"] == s
        tgt[m] = oracle_apply_string(a["input"][m], from_label(s))
    out = {"condition": condition, "n_records": n, "expected_records": V.expected_test_records(condition, n_grids),
           "duplicate_keys": int(n - len(np.unique(keys))), "target_oracle_mismatches": int((tgt != a["target"]).any((1, 2)).sum()),
           "prediction_values_out_of_range": int(((a["pred"] < -1) | (a["pred"] > 9)).sum()),
           "invalid_records": int((~a["valid"]).sum()),
           "k1_records": int((a["k"] == 1).sum()), "singleton_records": int((a["kind"] == "single").sum())}
    words_k = np.array([len(from_label(s)) for s in a["string"]])
    out["k_field_mismatches"] = int((words_k != a["k"]).sum())
    expect_len = np.full(n, 5) if condition == "identity" else a["k"]
    out["op_input_length_mismatches"] = int((a["op_input_length"] != expect_len).sum())
    if condition == "variable_length":
        out["identity_or_pad_tokens_in_variable_inputs"] = 0  # serialization strips them; verified in tests
    cc = (a["pred"] == a["target"]).reshape(n, -1).sum(1)
    out["stored_exact_mismatch"] = int(((a["valid"] & (cc == 36)) != a["exact"]).sum())
    if manifest_test_grids is not None:
        man = manifest_test_grids.reshape(len(manifest_test_grids), 36)[a["grid_id"]]
        out["input_manifest_mismatches"] = int((man != a["input"].reshape(n, 36)).any(1).sum())
    # perfect predictor through the SAME scoring path
    perfect = dict(a, pred=a["target"], valid=np.ones(n, bool))
    _, by_k, by_single, _, _ = metrics_tables(perfect, condition)
    out["perfect_predictor_all_100pct"] = bool((by_k.exact_acc == 1).all() and (by_single.exact_acc == 1).all())
    out["ok"] = (out["n_records"] == out["expected_records"] and out["duplicate_keys"] == 0
                 and out["target_oracle_mismatches"] == 0 and out["prediction_values_out_of_range"] == 0
                 and out["k_field_mismatches"] == 0 and out["op_input_length_mismatches"] == 0
                 and out["stored_exact_mismatch"] == 0 and out.get("input_manifest_mismatches", 0) == 0
                 and out["perfect_predictor_all_100pct"])
    return out
