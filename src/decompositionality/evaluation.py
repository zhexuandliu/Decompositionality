"""Inference over fixed evaluation sets, metrics, aggregation and result files.

Evaluation uses frozen parameters (``model.eval()`` + ``torch.no_grad()``), deterministic
greedy decoding (AR, free-running) or argmax (parallel/TRM final refinement), freshly reset
recurrent states, no demonstrations, no test-time gradients and no answer-conditioned
stopping. ``y`` is only used *after* prediction for scoring and for the separately
labeled cell-CE diagnostic.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import operations as O
from .utils import atomic_write_bytes, atomic_write_json, atomic_write_text, free_memory, is_oom


def amp_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def to_device(batch: Dict[str, np.ndarray], device: torch.device, sl=slice(None), with_target=True):
    out = {"ops": torch.as_tensor(np.asarray(batch["ops"][sl]), dtype=torch.long, device=device),
           "x": torch.as_tensor(np.asarray(batch["x"][sl]).astype(np.int64), device=device)}
    if with_target:
        out["y"] = torch.as_tensor(np.asarray(batch["y"][sl]).astype(np.int64), device=device)
    return out


def default_eval_batch_size(model, device) -> int:
    if device.type != "cuda":
        return 64
    return 512 if model.kind in ("ar", "trm") else 1024


def run_model_on_set(model, es: Dict[str, np.ndarray], device, amp_dtype=None, batch_size: Optional[int] = None,
                     want_ce: bool = True, want_pred: bool = True) -> Dict[str, np.ndarray]:
    """Run a model over an evaluation set. Predictions are computed from (ops, x) only."""
    was_training = model.training
    model.eval()
    n = len(es["x"])
    bs = batch_size or default_eval_batch_size(model, device)
    out: Dict[str, List[np.ndarray]] = {}
    start = 0
    try:
        with torch.no_grad():
            while start < n:
                sl = slice(start, min(n, start + bs))
                try:
                    with amp_context(device, amp_dtype):
                        if want_pred:
                            # prediction inputs deliberately exclude the target
                            p = model.predict(to_device(es, device, sl, with_target=False))
                            out.setdefault("pred", []).append(p["pred"].to(torch.int8).cpu().numpy())
                            out.setdefault("valid", []).append(p["valid"].cpu().numpy())
                            if "eos_ok" in p:
                                out.setdefault("eos_ok", []).append(p["eos_ok"].cpu().numpy())
                        if want_ce:
                            ce = model.val_cell_ce(to_device(es, device, sl, with_target=True))
                            out.setdefault("cell_ce", []).append(ce.float().cpu().numpy())
                except Exception as exc:  # OOM: retry the same slice with a smaller batch
                    if is_oom(exc) and bs > 1:
                        bs = max(1, bs // 2)
                        free_memory()
                        # drop any partial outputs of this slice
                        out = {k: v[: _n_batches_done(v, start)] for k, v in out.items()}
                        continue
                    raise
                start = sl.stop
    finally:
        model.train(was_training)
    res = {k: np.concatenate(v) for k, v in out.items()}
    if want_pred:
        res.update(score_predictions(res["pred"], res["valid"], es["y"]))
    return res


def _n_batches_done(chunks: List[np.ndarray], n_rows: int) -> int:
    total, k = 0, 0
    for c in chunks:
        if total + len(c) > n_rows:
            break
        total += len(c)
        k += 1
    return k


def score_predictions(pred: np.ndarray, valid: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Invalid cells are encoded as -1 and never match. Exact = all 36 cells valid and equal."""
    tgt = np.asarray(y).reshape(len(y), 36).astype(np.int16)
    p = np.asarray(pred).reshape(len(pred), 36).astype(np.int16)
    cell_correct = (p == tgt).sum(axis=1).astype(np.int16)
    exact = np.asarray(valid, dtype=bool) & (cell_correct == 36)
    return {"cell_correct": cell_correct, "exact": exact}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_string_table(strings: np.ndarray, res: Dict[str, np.ndarray], train_strings: Sequence[str],
                     order: Optional[Sequence[str]] = None):
    import pandas as pd
    df = pd.DataFrame({"string": strings, "exact": res["exact"], "cell_acc": res["cell_correct"] / 36.0,
                       "valid": res["valid"]})
    if "cell_ce" in res:
        df["cell_ce"] = res["cell_ce"]
    agg = {"exact_acc": ("exact", "mean"), "cell_acc": ("cell_acc", "mean"), "valid_rate": ("valid", "mean"),
           "n": ("exact", "size")}
    if "cell_ce" in df:
        agg["cell_ce"] = ("cell_ce", "mean")
    t = df.groupby("string").agg(**agg).reset_index()
    order = list(order or O.TEST_STRINGS)
    t["order"] = t["string"].map({s: i for i, s in enumerate(order)})
    t = t.sort_values("order").drop(columns="order").reset_index(drop=True)
    t["length"] = t["string"].str.len()
    t["group"] = t["string"].map(O.string_group)
    t["seen_in_training"] = t["string"].isin(list(train_strings))
    return t


def group_table(per_string, train_strings: Sequence[str]):
    """Macro averages over strings within groups. Empty groups are reported as N/A (NaN)."""
    import pandas as pd
    train = set(train_strings)
    defs = {
        "singles": lambda s: len(s) == 1,
        "pairs": lambda s: len(s) == 2,
        "pairs_seen": lambda s: len(s) == 2 and s in train,
        "pairs_unseen": lambda s: len(s) == 2 and s not in train,
        "triples_all_distinct": lambda s: O.string_group(s) == "triples_all_distinct" if len(s) == 3 else False,
        "triples_two_types": lambda s: O.string_group(s) == "triples_two_types" if len(s) == 3 else False,
        "triples_one_type": lambda s: O.string_group(s) == "triples_one_type" if len(s) == 3 else False,
        "triples_all": lambda s: len(s) == 3,
    }
    rows = []
    for g, f in defs.items():
        sub = per_string[per_string["string"].map(f)]
        row = {"group": g, "label": O.GROUP_LABELS[g], "n_strings": len(sub)}
        for col in ("exact_acc", "cell_acc", "cell_ce", "valid_rate"):
            if col in per_string:
                row[col] = float(sub[col].mean()) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(v, digits=3) -> str:
    return "N/A" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_eval_outputs(out_dir: Path, es: Dict[str, np.ndarray], res: Dict[str, np.ndarray], meta: dict,
                      train_strings: Sequence[str], prefix: str) -> dict:
    """Per-example arrays (compressed npz) + tidy metadata (csv.gz) + per-string/group CSVs."""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    arrays = {"pred": res["pred"].reshape(-1, 6, 6), "target": es["y"].reshape(-1, 6, 6).astype(np.int8),
              "input": es["x"].reshape(-1, 6, 6).astype(np.int8), "grid_id": es["grid_id"].astype(np.int32),
              "string": np.asarray(es["string"]).astype("U3"), "valid": res["valid"],
              "cell_correct": res["cell_correct"], "exact": res["exact"], "cell_ce": res.get("cell_ce")}
    if "eos_ok" in res:
        arrays["eos_ok"] = res["eos_ok"]
    np.savez_compressed(buf, **{k: v for k, v in arrays.items() if v is not None})
    atomic_write_bytes(out_dir / f"{prefix}_predictions.npz", buf.getvalue())
    tidy = pd.DataFrame({
        "experiment": meta["experiment"], "model": meta["model_id"], "seed": meta["seed"], "split": prefix,
        "grid_id": es["grid_id"], "string": es["string"], "length": [len(s) for s in es["string"]],
        "group": [O.string_group(s) for s in es["string"]],
        "seen_in_training": np.isin(es["string"], list(train_strings)),
        "valid": res["valid"], "cell_correct": res["cell_correct"], "exact": res["exact"],
    })
    if "cell_ce" in res:
        tidy["cell_ce"] = res["cell_ce"]
    if "eos_ok" in res:
        tidy["eos_ok"] = res["eos_ok"]
    tidy.to_csv(out_dir / f"{prefix}_examples.csv.gz", index=False, compression="gzip")
    order = O.TEST_STRINGS if prefix == "test" else list(train_strings)
    ps = per_string_table(np.asarray(es["string"]), res, train_strings, order)
    ps.insert(0, "model", meta["model_id"])
    ps.insert(0, "experiment", meta["experiment"])
    ps.to_csv(out_dir / f"{prefix}_per_string.csv", index=False)
    gt = group_table(ps, train_strings)
    gt.insert(0, "model", meta["model_id"])
    gt.insert(0, "experiment", meta["experiment"])
    gt.to_csv(out_dir / f"{prefix}_groups.csv", index=False)
    summary = {"n_examples": int(len(es["x"])), "exact_acc": float(res["exact"].mean()),
               "cell_acc": float(res["cell_correct"].mean() / 36), "valid_rate": float(res["valid"].mean())}
    if "cell_ce" in res:
        summary["cell_ce"] = float(res["cell_ce"].mean())
    if "eos_ok" in res:
        summary["eos_format_rate"] = float(res["eos_ok"].mean())
    return summary


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
