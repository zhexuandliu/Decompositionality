"""Row-separator suite (notebook 18): preparation, preflight audits, chunked free-running evaluation with completeness
checks, per-run analysis, the sequential 26-job driver, consolidated comparisons / figures, and the review export.

Results: <RESULTS_ROOT>/ar_row_separators_13conditions_v2/
  manifests/                                shared grid manifests (same seed -> same hash as earlier suites)
  <condition>/seed_<seed>/runs/<model>/     checkpoints (best.pt, state.pt), logs, run_config.json
  <condition>/seed_<seed>/eval/<model>/     chunks/, test_predictions.npz (raw 42 tokens per record), ARTIFACT.json,
                                            COMPLETENESS.json, analysis/ (per-run tables)
  audits/seed_<seed>/, status/seed_<seed>.json, report/seed_<seed>/
Comparison/report code reads only saved artifacts; it never trains or builds training models.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import mirror_diag as M
from . import rowsep as RS
from . import tokenization as T
from . import idvar as V
from .words import from_label

STAGES = ["train", "evaluate", "analyze"]
NA = "N/A"
SHORT = {"transformer_ar": "TF-AR", "looped_transformer_ar": "Looped-AR"}
# new condition -> historical no-ROW run id (mirror_report run ids); None = no matching legacy condition
LEGACY = {"identity_s5_train02": "S5/train02", "identity_s5_train023": "S5/train023", "identity_s5_train0234": "S5/train0234",
          "identity_s5_train02345": "S5/train02345", "variable_train02345": "varlen/train02345",
          "identity_s6_train02345": "S6/train02345"}
LEGACY_NOTES = {"identity_s5_train02": "same logical stream & layouts (idext), legacy init; no ROW",
                "identity_s5_train023": "same logical stream & layouts (idext), legacy init; no ROW",
                "identity_s5_train0234": "same logical stream & layouts (idext), legacy init; no ROW",
                "identity_s5_train02345": "legacy run used the notebook-10 sampler: training stream UNMATCHED",
                "variable_train02345": "legacy run used the notebook-10/11 sampler: training stream UNMATCHED",
                "identity_s6_train02345": "same logical stream & S=6 placements as notebook 16; legacy init grown differently"}


# ---------------------------------------------------------------------------
# Preparation and preflight
# ---------------------------------------------------------------------------

def prepare(cfg, repo_dir=None, log=print):
    from .data import load_or_build_manifests
    from .pipeline import Context
    from .utils import atomic_write_json
    e, seed = cfg.experiment, cfg.train.seed
    m = load_or_build_manifests(cfg.manifest_dir, seed)
    val = RS.make_val_set(m.val_grids[: cfg.n_val_grids], seed, e)
    subset = V.subset_by_grids(val, cfg.val_subset_grids)
    test = RS.make_test_set(m.test_grids[: cfg.n_test_grids], seed, e)
    assert len(test["x"]) == RS.expected_test_records(e, cfg.n_test_grids)
    assert set(np.unique(val["k"])) == set(RS.train_ks(e)) and not (val["k"] == 1).any()
    if RS.group_of(e) == "variable_nozero":
        assert not (val["k"] == 0).any() and not (test["k"] == 0).any()
    vb = {g.tobytes() for g in m.val_grids.astype(np.uint8)}
    assert not vb & {g.tobytes() for g in m.test_grids.astype(np.uint8)}
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json",
                      {**cfg.to_dict(), "group": RS.group_of(e), "layout": RS.layout_of(e), "S": RS.n_slots(e),
                       "train_ks": list(RS.train_ks(e)), "row_layout": RS.ROW_LAYOUT_SPEC,
                       "manifest_hash": m.manifest_hash, "n_val_records": len(val["x"]), "n_test_records": len(test["x"]),
                       "data_version": RS.ROWSEP_DATA_VERSION})
    log(f"[{e}] {RS.group_of(e)} | train k={list(RS.train_ks(e))} | val {len(val['x'])} | test {len(test['x']):,} "
        f"| manifests {m.manifest_hash[:12]}")
    return Context(cfg, m, val, subset, test, repo_dir)


def layout_audit():
    """Actual serializer output for m = 0..6: length 87+m, 5 ROW per grid, positions, coordinates, round trip."""
    import pandas as pd
    import torch
    rows = []
    rng = np.random.default_rng(0)
    for m in range(7):
        lay = T.ARRowLayout(m)
        x = torch.as_tensor(rng.integers(0, 10, (2, 6, 6)))
        y = torch.as_tensor(rng.integers(0, 10, (2, 6, 6)))
        ops = torch.as_tensor(rng.integers(0, 5, (2, m)))
        seq = lay.serialize(ops, x, y)
        meta = lay.metadata()
        assert seq.shape[1] == 87 + m
        for start, g in ((lay.in_start, x), (lay.out_start, y)):
            span = seq[:, start:start + 41]
            assert ((span == T.ROW_ID).sum(1) == 5).all() and (span[:, 6::7] == T.ROW_ID).all()
            back = span[:, lay.out_digit_offsets].reshape(2, 6, 6)        # round trip
            assert torch.equal(back, g.long())
        assert (meta["row"][lay.output_digit_indices].numpy() == np.arange(36) // 6).all()
        assert (meta["col"][lay.output_digit_indices].numpy() == np.arange(36) % 6).all()
        assert (meta["row"][lay.output_row_indices] == T.NO_COORD).all() and (meta["col"][lay.input_row_indices] == T.NO_COORD).all()
        assert torch.equal(meta["pos"], torch.arange(87 + m))
        assert T.PAD_ID not in seq.tolist() if m == 0 else True
        rows.append({"m": m, "length": lay.length, "OUTPUT": lay.idx_output, "prefix_len": lay.prefix_len,
                     "first_output_digit": int(lay.output_digit_indices[0]), "output_ROW": list(map(int, lay.output_row_indices)),
                     "EOS": lay.idx_eos, "ROW_per_grid": 5, "positions": f"dense 0..{lay.length - 1}",
                     "ROW coords": "NO_COORD (structural); grid segment"})
    return pd.DataFrame(rows)


def init_audit(seed=42):
    import pandas as pd
    import torch
    from .config import make_experiment_config
    from .models.common import count_parameters
    from .models.registry import build_model
    from .training import model_config_for
    rows = []
    for m in RS.MODELS:
        sds = {}
        for e in RS.CONDITIONS:
            mod = build_model(m, model_config_for(make_experiment_config(e, models=[m]), m), seed=seed)
            sd, prov = RS.common_initial_state(m, seed)
            mod.load_state_dict(sd)
            sds[e] = mod.state_dict()
        ref = next(iter(sds.values()))
        same = all(torch.equal(ref[k], s[k]) for s in sds.values() for k in ref)
        rows.append({"model": m, "unique_params": count_parameters(mod), "block_applications": mod.cfg.depth,
                     "tied": mod.cfg.tied, "vocab": ref["tok_emb.weight"].shape[0], "pos_table": tuple(ref["pos_emb.weight"].shape),
                     "identical_init_across_13": bool(same), "full_init_sha": RS.state_sha(ref), **prov,
                     "grown": json.dumps(prov["grown"])})
    return pd.DataFrame(rows)


def stream_audit(seed=42, B=256, steps=(0, 1, 2, 4999)):
    """(a)/(b) same-support logical identity across S5 / variable / S6; (c) common per-k prefixes across supports
    (incl. with-zero vs nozero); (d) determinism (both architectures read the same stateless stream)."""
    import pandas as pd
    from . import idext as E
    rows = []
    sup = {"train02": ["identity_s5_train02", "variable_train02"], "train023": ["identity_s5_train023", "variable_train023"],
           "train0234": ["identity_s5_train0234", "variable_train0234"],
           "train02345": ["identity_s5_train02345", "variable_train02345", "identity_s6_train02345"]}
    for name, conds in sup.items():
        for s in steps:
            bs = [RS.RowStream(seed, c, B).batch(s) for c in conds]
            same = all(np.array_equal(bs[0][k], b[k]) for b in bs[1:] for k in ("x", "y", "string", "k_index"))
            rows.append({"check": f"same logical stream {name}", "conditions": ", ".join(conds), "update": s, "ok": bool(same)})
    # conditional per-k prefixes: record j of length k identical across supports
    for k in (2, 3):
        a = [E.record(seed, k, j)[0] for j in range(50)]
        rows.append({"check": f"per-k indexed records k={k} shared by every support containing k (incl. nozero)",
                     "conditions": "all", "update": "-", "ok": a == [E.record(seed, k, j)[0] for j in range(50)]})
    for c in RS.CONDITIONS:
        b = RS.RowStream(seed, c, B).batch(0)
        b2 = RS.RowStream(seed, c, B).batch(0)
        det = all(np.array_equal(b[k], b2[k]) for k in ("x", "y", "string"))
        rows.append({"check": "deterministic + only training k", "conditions": c, "update": 0,
                     "ok": bool(det and set(np.unique(b["k"])) <= set(RS.train_ks(c)) and not (b["k"] == 1).any()
                                and (RS.group_of(c) != "variable_nozero" or not (b["k"] == 0).any()))})
    df = pd.DataFrame(rows)
    assert df.ok.all(), df[~df.ok]
    return df


def validation_audit():
    """Code-level audit of the historical S5/train023 and train0234 checkpoint selection (notebook 17 showed the
    generic text 'over k in 0/2/3/4/5')."""
    import pandas as pd
    return pd.DataFrame([
        {"item": "historical selection text", "finding": "evaluate_idvar_model (before commit 12d7cac) wrote the CONSTANT "
         "string 'lowest k-macro validation ... over k in 0/2/3/4/5' for every run; it was a stale label."},
        {"item": "historical validation data", "finding": "prepare_idext built the validation set with idext.make_val_set("
         "train_ks(experiment)) and asserted set(val k) == train_ks; validate() selects on that set only. So the old "
         "train023 / train0234 selection used k in {0,2,3} / {0,2,3,4}: no held-out leakage found."},
        {"item": "new runs", "finding": "rowsep.make_val_set uses ONLY the condition's training k (k=0 excluded for "
         "variable_nozero); prepare() asserts it; the selection rule string is generated from the actual k."}])


# ---------------------------------------------------------------------------
# Evaluation (chunked, resumable) with completeness checks
# ---------------------------------------------------------------------------

def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def eval_dir(cfg, model_id) -> Path:
    return cfg.experiment_dir / "eval" / model_id


def completeness(arr: dict, e: str, n_grids: int) -> dict:
    keys = np.char.add(np.char.add(arr["string"].astype("U8"), "@"),
                       np.char.add(arr["position"].astype(str), np.char.add(":", arr["grid_id"].astype(str))))
    by_k = {int(k): int((arr["k"] == k).sum()) for k in np.unique(arr["k"])}
    req = {k: v * n_grids for k, v in RS.required_heldout(e).items()}
    missing = {k: (by_k.get(k, 0), need) for k, need in req.items() if by_k.get(k, 0) != need}
    out = {"records": int(len(keys)), "expected": RS.expected_test_records(e, n_grids), "duplicates": int(len(keys) - len(np.unique(keys))),
           "records_by_k": by_k, "required_heldout": req, "missing_heldout_sections": missing}
    out["complete"] = out["records"] == out["expected"] and out["duplicates"] == 0 and not missing
    return out


def evaluation_complete(cfg, model_id, rc) -> bool:
    f = eval_dir(cfg, model_id) / "ARTIFACT.json"
    if not f.exists():
        return False
    a = json.loads(f.read_text())
    return bool(a.get("complete") and a.get("config_hash") == rc["config_hash"] and Path(a["predictions"]).exists())


def evaluate_run(model_id: str, ctx, log=print, chunk_grids: int = 100) -> dict:
    import torch
    from .evaluation import run_model_on_set
    from .idvar_pipeline import grid_hashes
    from .models.registry import build_model
    from .training import run_config, run_dir, training_complete
    from .utils import (atomic_write_bytes, atomic_write_json, dtype_name, environment_info, free_memory, hardware_name,
                        resolve_amp_dtype, resolve_device)
    cfg, e = ctx.cfg, ctx.cfg.experiment
    rc = run_config(cfg, model_id, ctx.manifests.manifest_hash)
    ed = eval_dir(cfg, model_id)
    if evaluation_complete(cfg, model_id, rc):
        log(f"[{e}/{model_id}] evaluation already complete -> skipping")
        return json.loads((ed / "ARTIFACT.json").read_text())
    if not training_complete(cfg, model_id, rc):
        raise RuntimeError(f"[{e}/{model_id}] training not complete")
    ck = torch.load(run_dir(cfg, model_id) / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config_hash"] == rc["config_hash"]
    tc = cfg.train
    device = resolve_device(tc.device)
    amp = resolve_amp_dtype(device, tc.dtype)
    model = build_model(model_id, rc["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    es = ctx.test_set
    n_grids = int(es["grid_id"].max()) + 1
    cdir = ed / "chunks"
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
        part = {"raw_tokens": r["raw_tokens"].astype(np.int16), "cell_ce": r["cell_ce"], "n": np.array(int(sel.sum())),
                "config_hash": np.array(rc["config_hash"])}
        atomic_write_bytes(f, _npz_bytes(**part))
        parts.append(part)
        log(f"  [{e}/{model_id}] chunk {ci + 1}/{len(starts)} ({int(sel.sum()):,}) | {(time.time() - t0) / 60:.1f} min")
    t_eval = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    del model
    free_memory()
    raw = np.concatenate([p["raw_tokens"] for p in parts])
    assert raw.shape == (len(es["x"]), 42)
    sc = RS.score(raw, es["y"])
    slots = es["slots"] if "slots" in es else es["slots5"]
    arr = {"raw_tokens": raw, "pred": sc["pred"].reshape(-1, 6, 6), "target": es["y"].reshape(-1, 6, 6).astype(np.int8),
           "input": es["x"].reshape(-1, 6, 6).astype(np.int8), "grid_id": es["grid_id"].astype(np.int32),
           "grid_hash": grid_hashes(es["x"]), "string": es["string"].astype("U8"), "k": es["k"].astype(np.int8),
           "kind": es["kind"].astype("U6"), "position": es["position"].astype(np.int8), "slots5": slots.astype(np.int8),
           "support": es["support"].astype("U16"), "valid": sc["valid"], "exact": sc["exact"],
           "cell_correct": sc["cell_correct"].sum(1).astype(np.int8), "row_ok": sc["row_ok"], "eos_ok": sc["eos_ok"],
           "premature_eos": sc["premature_eos"], "strict": sc["strict"],
           "cell_ce": np.concatenate([p["cell_ce"] for p in parts]).astype(np.float32),
           "meta_condition": np.array(e), "meta_model": np.array(model_id), "meta_seed": np.array(tc.seed),
           "meta_selected_step": np.array(ck["step"])}
    comp = completeness(arr, e, n_grids)
    atomic_write_json(ed / "COMPLETENESS.json", comp)
    pfile = ed / "test_predictions.npz"
    atomic_write_bytes(pfile, _npz_bytes(**arr))
    art = {"experiment": e, "model": model_id, "seed": tc.seed, "complete": comp["complete"], "predictions": str(pfile),
           "predictions_sha256": _sha(pfile), "n_records": len(raw), "expected_records": comp["expected"],
           "selected_step": ck["step"], "selected_val_digit_ce": ck["val_cell_ce"], "config_hash": rc["config_hash"],
           "manifest_hash": ctx.manifests.manifest_hash, "train_ks": list(RS.train_ks(e)),
           "selection_rule": f"lowest k-macro validation answer-DIGIT CE (no ROW/EOS) over k in {list(RS.train_ks(e))}",
           "eval_seconds": t_eval, "peak_gpu_gb_eval": peak, "hardware": hardware_name(device), "dtype": dtype_name(amp),
           "repo_commit": environment_info(ctx.repo_dir)["repo_commit"]}
    atomic_write_json(ed / "ARTIFACT.json", art)
    if not comp["complete"]:
        raise RuntimeError(f"[{e}/{model_id}] evaluation INCOMPLETE: {comp['missing_heldout_sections']}")
    s1 = arr["kind"] == "single"
    log(f"[{e}/{model_id}] step {ck['step']} | singletons exact {int(arr['exact'][s1].sum())}/{int(s1.sum())} | "
        f"strict {arr['strict'].mean():.4f} | eval {t_eval / 60:.1f} min")
    return art


# ---------------------------------------------------------------------------
# Per-run analysis (stage "analyze")
# ---------------------------------------------------------------------------

def load(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _frame(a):
    import pandas as pd
    return pd.DataFrame({"k": a["k"].astype(int), "word": a["string"].astype(str), "kind": a["kind"].astype(str),
                         "position": a["position"].astype(int), "grid_id": a["grid_id"], "grid_hash": a["grid_hash"],
                         "support": a["support"].astype(str), "exact": a["exact"].astype(bool),
                         "cell": a["cell_correct"] / 36, "valid": a["valid"].astype(bool),
                         "rows_ok": a["row_ok"].mean(1), "all_rows": a["row_ok"].all(1), "eos": a["eos_ok"].astype(bool),
                         "premature_eos": a["premature_eos"].astype(bool), "strict": a["strict"].astype(bool),
                         "nC": np.char.count(a["string"].astype(str), "C")})


def _agg(g):
    import pandas as pd
    return pd.Series({"exact_correct": int(g.exact.sum()), "n": len(g), "exact_acc": g.exact.mean(), "cell_acc": g.cell.mean(),
                      "row_token_acc": g.rows_ok.mean(), "all5_row_acc": g.all_rows.mean(), "eos_acc": g.eos.mean(),
                      "premature_eos_rate": g.premature_eos.mean(), "strict_acc": g.strict.mean(),
                      "invalid_digit_rate": (~g.valid).mean()})


def analyze_run(cfg, model_id, log=print) -> dict:
    """Per-run tables saved under eval/<model>/analysis/ (resumable; keyed by the predictions sha)."""
    import pandas as pd
    from .utils import atomic_write_json
    e = cfg.experiment
    ed = eval_dir(cfg, model_id)
    art = json.loads((ed / "ARTIFACT.json").read_text())
    if not art.get("complete"):
        raise RuntimeError("evaluation incomplete")
    ad = ed / "analysis"
    done = ad / "DONE.json"
    if done.exists() and json.loads(done.read_text()).get("sha") == art["predictions_sha256"]:
        log(f"[{e}/{model_id}] analysis already complete -> skipping")
        return json.loads(done.read_text())
    ad.mkdir(parents=True, exist_ok=True)
    a = load(art["predictions"])
    from .diagnosis.group import oracle_apply_string
    for s in np.unique(a["string"]):          # independent oracle check of every stored target
        m = a["string"] == s
        assert (oracle_apply_string(a["input"][m], from_label(s)) == a["target"][m]).all(), s
    df = _frame(a)
    tid = {}
    G = M.group()
    by_k = df.groupby(["k", "support"]).apply(_agg).reset_index()
    by_word = df.groupby(["word", "k", "kind", "support"]).apply(_agg).reset_index()
    s = df[df.kind == "single"]
    by_single = s.groupby(["word", "position"]).apply(_agg).reset_index()
    # geometry on the 36 extracted digit cells (structural positions ignored), ambiguity kept
    sel = np.flatnonzero((a["kind"] == "single") | (a["kind"] == "six") | np.isin(a["string"], ["ACC", "BCC", "CCC", "CCCCC", "CC", "CCCC", "BBBB", "EMPTY"]))
    P = a["pred"].reshape(-1, 36)[sel]
    X = a["input"].reshape(-1, 36)[sel]
    labels = a["string"][sel].astype(str)
    for w in np.unique(labels):
        tid[w] = G.word_id(from_label(w))
    t_ids = np.array([tid[w] for w in labels])
    matches = M.Matcher(G).match(P, X, a["grid_hash"][sel], a["valid"][sel])
    c = M.classify(G, matches, t_ids, a["valid"][sel], a["exact"][sel])
    geo = pd.DataFrame({"rec": sel, "word": labels, "kind": a["kind"][sel].astype(str), "position": a["position"][sel],
                        "grid_hash": a["grid_hash"][sel], "exact": a["exact"][sel], "valid": a["valid"][sel],
                        "outcome": c["outcome"], "match_identity": c["match_identity"], "residual": c["residual"],
                        "mirror_wrong": c["mirror_wrong"], "n_match": c["n_match"]})
    gs = geo[geo.kind == "single"]
    geo_t = gs.groupby("word").apply(lambda g: pd.Series({
        "n": len(g), "exact_correct": int(g.exact.sum()),
        "identity_copy": int((~g.exact & g.match_identity).sum()),
        "other_legal_transform": int((~g.exact & (g.n_match > 0) & ~g.match_identity).sum()),
        "nonlegal_valid_grid": int((g.outcome == "wrong: valid, no legal match").sum()),
        "invalid_digit_output": int((~g.valid).sum()),
        "extra_C_residual": int(g.residual.isin(["extra output-C only", "extra input-C only", "both extra-C hypotheses"]).sum())})).reset_index()
    # spatial error profile of singletons (6x6), first column vs others, wrong-cell counts
    si = np.flatnonzero(a["kind"] == "single")
    wrong = (a["pred"].reshape(-1, 36)[si] != a["target"].reshape(-1, 36)[si])
    sp = []
    for op in "ABC":
        m = a["string"][si] == op
        w = wrong[m]
        sp.append({"op": op, **{f"cell_err_r{r}c{cc}": float(w[:, 6 * r + cc].mean()) for r in range(6) for cc in range(6)},
                   **{f"row{r}_err": float(w[:, 6 * r:6 * r + 6].mean()) for r in range(6)},
                   **{f"col{cc}_err": float(w[:, cc::6].mean()) for cc in range(6)},
                   "first_col_err": float(w[:, 0::6].mean()), "other_cols_err": float(np.delete(w.reshape(-1, 6, 6), 0, axis=2).mean()),
                   **{f"wrong_cells_{b}": int(((w.sum(1) >= lo) & (w.sum(1) <= hi)).sum())
                      for b, (lo, hi) in {"0": (0, 0), "1-3": (1, 3), "4-12": (4, 12), "13-35": (13, 35), "36": (36, 36)}.items()}})
    spatial = pd.DataFrame(sp)
    # digit errors in row r conditional on the preceding generated ROW being correct (descriptive)
    held = np.flatnonzero(a["support"] == "HELD OUT")
    cond = []
    if len(held):
        W = (a["pred"].reshape(-1, 36)[held] != a["target"].reshape(-1, 36)[held])
        R = a["row_ok"][held]
        for r in range(1, 6):
            e_r = W[:, 6 * r:6 * r + 6].mean(1)
            ok = R[:, r - 1]
            cond.append({"row": r, "records": len(held), "preceding_ROW_correct": int(ok.sum()),
                         "digit_err | ROW correct": float(e_r[ok].mean()) if ok.any() else np.nan,
                         "digit_err | ROW wrong": float(e_r[~ok].mean()) if (~ok).any() else np.nan})
    cond = pd.DataFrame(cond)
    # singleton consistency across slots (fixed layouts)
    cons = []
    if RS.layout_of(e) == "fixed":
        for op in "ABC":
            m = np.flatnonzero((a["kind"] == "single") & (a["string"] == op))
            sub = pd.DataFrame({"g": a["grid_hash"][m], "exact": a["exact"][m], "i": m})
            same = sub.groupby("g").i.apply(lambda ii: bool((a["pred"][ii.values] == a["pred"][ii.values[0]]).all())).mean()
            allc = sub.groupby("g").exact.all().mean()
            cons.append({"op": op, "grids_identical_prediction_all_slots": float(same), "grids_all_slots_correct": float(allc),
                         "grids_all_slots_wrong": float((~sub.groupby("g").exact.any()).mean())})
    cons = pd.DataFrame(cons)
    # same-target pairs (only words in this manifest)
    pairs = []
    for sing, comp in (("A", "ACC"), ("B", "BCC"), ("C", "CCC"), ("C", "CCCCC")):
        cm = df[(df.word == comp) & (df.kind != "single")]
        if not len(cm):
            pairs.append({"singleton": sing, "composite": comp, "status": "unavailable (not in this condition's manifest)"})
            continue
        sm = df[(df.kind == "single") & (df.word == sing)].groupby("grid_hash").exact.mean()
        j = pd.concat([sm.rename("single_mean"), cm.set_index("grid_hash").exact.rename("comp")], axis=1, join="inner")
        pairs.append({"singleton": sing, "composite": comp, "status": "available",
                      "composite_in_training": bool(len(comp) in RS.train_ks(e)), "grids": len(j),
                      "composite_correct": int(j.comp.sum()),
                      "grids comp correct & singleton wrong at all slots": int((j.comp & (j.single_mean == 0)).sum()),
                      "grids comp correct & singleton correct at all slots": int((j.comp & (j.single_mean == 1)).sum()),
                      "grids comp wrong & singleton correct at all slots": int((~j.comp.astype(bool) & (j.single_mean == 1)).sum())})
    pairs = pd.DataFrame(pairs)
    # six-step
    six = pd.DataFrame()
    six_word = pd.DataFrame()
    if (a["kind"] == "six").any():
        from .mirror_report import training_transforms
        tt = training_transforms("train02345")
        x = df[df.kind == "six"].copy()
        x["transform_in_training"] = [tid.get(w, G.word_id(w)) in tt for w in x.word]
        gx = geo[geo.kind == "six"]
        x["mirror_wrong"] = gx.mirror_wrong.values
        x["nonlegal"] = (gx.outcome == "wrong: valid, no legal match").values
        six = x.groupby(["nC"]).apply(lambda g: pd.Series({"words": g.word.nunique(), "records": len(g),
                                                           "exact_correct": int(g.exact.sum()), "exact_acc": g.exact.mean(),
                                                           "cell_acc": g.cell.mean(), "strict_acc": g.strict.mean(),
                                                           "wrong_mirror": int(g.mirror_wrong.sum()),
                                                           "wrong_nonlegal": int(g.nonlegal.sum())})).reset_index()
        par = x.groupby([x.nC % 2, "transform_in_training"]).apply(lambda g: pd.Series({
            "words": g.word.nunique(), "exact_correct": int(g.exact.sum()), "n": len(g)})).reset_index()
        par.columns = ["nC_parity_odd", "transform_in_training", "words", "exact_correct", "n"]
        six = pd.concat([six.assign(table="by n_C"), par.assign(table="parity x transform")], ignore_index=True)
        six_word = x.groupby("word").agg(nC=("nC", "first"), transform_in_training=("transform_in_training", "first"),
                                         exact_correct=("exact", "sum"), n=("exact", "size"), cell_acc=("cell", "mean")).reset_index()
    tables = {"by_k": by_k, "by_word": by_word, "singletons": by_single, "singleton_geometry": geo_t, "singleton_spatial": spatial,
              "row_conditional": cond, "singleton_consistency": cons, "same_target_pairs": pairs, "six_step": six,
              "six_per_word": six_word}
    for k, t in tables.items():
        t.to_csv(ad / f"{k}.csv", index=False)
    # compact per-(op, grid) singleton means for cross-condition pairing
    s.groupby(["word", "grid_hash"]).exact.mean().reset_index().to_csv(ad / "singleton_grid_means.csv", index=False)
    info = {"sha": art["predictions_sha256"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(done, info)
    log(f"[{e}/{model_id}] analysis written")
    return info


# ---------------------------------------------------------------------------
# Status and sequential driver
# ---------------------------------------------------------------------------

def family_root(results_root: str, mode: str) -> Path:
    r = Path(results_root) / RS.ROWSEP_RESULTS_SUBDIR
    return r if mode == "full" else r / "smoke"


def status_table(root: Path, seed: int = 42, jobs=None):
    import pandas as pd
    rows = []
    sel = set(jobs or RS.select_jobs())
    for c, m in RS.select_jobs():
        d = Path(root) / c / f"seed_{seed}"
        art = d / "eval" / m / "ARTIFACT.json"
        a = json.loads(art.read_text()) if art.exists() else {}
        st = {"train": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(), "evaluate": bool(a.get("complete")),
              "analyze": (d / "eval" / m / "analysis" / "DONE.json").exists()}
        rows.append({"condition": c, "group": RS.group_of(c), "model": m, "selected": (c, m) in sel,
                     **{k: ("done" if v else "pending") for k, v in st.items()}, "complete": all(st.values()),
                     "selected_step": a.get("selected_step"), "records": a.get("n_records")})
    return pd.DataFrame(rows)


def _runtime_healthy():
    import torch
    if not torch.cuda.is_available():
        return True
    try:
        torch.ones(4, device="cuda").sum().item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def run_queue(results_root: str, mode: str, seed: int, jobs: Sequence[tuple], stages: Sequence[str], repo_dir=None,
              chunk_grids=100, device="auto", dtype="auto", microbatch=None, model_overrides=None, train_overrides=None,
              log=print):
    """Sequential: for each job every selected stage completes before the next job (one model on the GPU)."""
    from .config import make_experiment_config
    from .training import train_model
    from .utils import atomic_write_json, free_memory
    root = family_root(results_root, mode)
    log(f"queue ({len(jobs)} jobs): " + " -> ".join(f"{c}/{SHORT[m]}" for c, m in jobs) + f" | stages {list(stages)}")
    by_cond: Dict[str, List[str]] = {}
    for c, m in jobs:
        by_cond.setdefault(c, []).append(m)
    status_f = root / "status" / f"seed_{seed}.json"
    for c, ms in by_cond.items():
        cfg = make_experiment_config(c, mode, str(Path(results_root) / RS.ROWSEP_RESULTS_SUBDIR), ms, seed=seed,
                                     device=device, dtype=dtype, microbatch_size=microbatch, **(train_overrides or {}))
        cfg.resume = True
        if model_overrides:
            cfg.model_overrides.update(model_overrides)
        ctx = prepare(cfg, repo_dir, log)
        for m in ms:
            log(f"\n=== {c} / {m} ===")
            t0 = time.time()
            for stage in stages:
                try:
                    if stage == "train":
                        r = train_model(m, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, log=log, repo_dir=repo_dir)
                        if "interrupted_at_step" in r:
                            raise RuntimeError("training interrupted")
                    elif stage == "evaluate":
                        evaluate_run(m, ctx, log, chunk_grids)
                    elif stage == "analyze":
                        analyze_run(cfg, m, log)
                    st = json.loads(status_f.read_text()) if status_f.exists() else {}
                    st.setdefault(f"{c}/{m}", {})[stage] = {"status": "done", "seconds": round(time.time() - t0, 1)}
                    atomic_write_json(status_f, st)
                except Exception as exc:
                    traceback.print_exc()
                    st = json.loads(status_f.read_text()) if status_f.exists() else {}
                    st.setdefault(f"{c}/{m}", {})[stage] = {"status": f"FAILED: {exc!r}"[:300]}
                    atomic_write_json(status_f, st)
                    log(f"[{c}/{m}] stage {stage} FAILED -> later stages of this job skipped")
                    free_memory()
                    if not _runtime_healthy():
                        log("CUDA runtime unhealthy -> stopping queue; restart runtime and rerun to resume")
                        return status_table(root, seed, jobs)
                    break
                finally:
                    free_memory()
        del ctx
        free_memory()
    return status_table(root, seed, jobs)


# ---------------------------------------------------------------------------
# Consolidated report (saved artifacts only)
# ---------------------------------------------------------------------------

def _read(root, c, m, seed, name):
    import pandas as pd
    p = Path(root) / c / f"seed_{seed}" / "eval" / m / "analysis" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def consolidated(root: Path, results_root: str, seed: int = 42, jobs=None, show=True, log=print) -> Path:
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from .mirror_report import Report, discover, write_review
    root = Path(root)
    out = root / "report" / f"seed_{seed}"
    figd = out / "figures"
    R = Report(show)
    st = status_table(root, seed, jobs)
    n_sel = int(st.selected.sum())
    R.md(f"# Row-separator suite: 13 conditions x 2 AR models (seed {seed})\n"
         f"{int(st[st.selected].complete.sum())}/{n_sel} selected jobs complete; "
         f"{int(st.complete.sum())}/26 of the full suite complete. "
         + ("**FULL SUITE COMPLETE**" if st.complete.all() else "**PARTIAL - not a complete suite**"))
    R.table("(a) Status of all 26 jobs", st, 30)
    T_ = {"status": st}
    avail = [(r.condition, r.model) for r in st.itertuples() if r.analyze == "done"]

    def cat(name):
        L = []
        for c, m in avail:
            t = _read(root, c, m, seed, name)
            if t is not None and len(t):
                t.insert(0, "model", m)
                t.insert(0, "condition", c)
                L.append(t)
        return pd.concat(L, ignore_index=True) if L else pd.DataFrame()

    by_k = cat("by_k")
    T_["by_k"] = by_k
    # (b) curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    curves = []
    for c, m in [(r.condition, r.model) for r in st.itertuples() if r.train == "done"]:
        rd = root / c / f"seed_{seed}" / "runs" / m
        if (rd / "train_log.csv").exists():
            t = pd.read_csv(rd / "train_log.csv")
            ls = "-" if m == "transformer_ar" else "--"
            for ax, col in zip(axes.flat[:3], ("answer_cell_nll", "row_nll", "eos_nll")):
                if col in t:
                    ax.plot(t.step, t[col].clip(lower=1e-7), ls=ls, lw=0.8)
            for _, r in t.iterrows():
                curves.append({"condition": c, "model": m, "step": r.step, "loss": r.loss, "digit_ce": r.get("answer_cell_nll"),
                               "row_ce": r.get("row_nll"), "eos_ce": r.get("eos_nll")})
        if (rd / "val_log.csv").exists():
            v = pd.read_csv(rd / "val_log.csv")
            axes[1, 1].plot(v.step, v.val_cell_ce.clip(lower=1e-7), ls="-" if m == "transformer_ar" else "--", lw=0.8)
    for ax, ttl in zip(axes.flat, ("train digit CE (36)", "train ROW CE (5)", "train EOS CE", "validation digit CE (selection)")):
        ax.set(yscale="log", title=ttl, xlabel="update")
    fig.suptitle("Training curves, all runs (solid TF-AR, dashed Looped-AR); per-run values in curves.csv")
    fig.tight_layout()
    R.fig(fig, "(b) Training and validation curves", figd / "curves.png")
    T_["curves"] = pd.DataFrame(curves)
    if len(by_k):
        # (c) Group 1 support x test-k heatmaps and per-k tables
        for m in RS.MODELS:
            g1 = by_k[(by_k.model == m) & by_k.condition.str.startswith("identity_s5_")]
            if not len(g1):
                continue
            conds = [c for c in RS.CONDITIONS if c.startswith("identity_s5_")]
            mat = np.full((len(conds), 6), np.nan)
            txt = [["N/A"] * 6 for _ in conds]
            for i, c in enumerate(conds):
                for kk in range(6):
                    q = g1[(g1.condition == c) & (g1.k == kk)]
                    if len(q):
                        mat[i, kk] = q.exact_acc.iloc[0]
                        txt[i][kk] = f"{int(q.exact_correct.iloc[0])}/{int(q.n.iloc[0])}" + ("*" if q.support.iloc[0] == "train-supported" else "")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(len(conds)):
                for kk in range(6):
                    ax.text(kk, i, txt[i][kk], ha="center", va="center", fontsize=7)
            ax.set_xticks(range(6)); ax.set_xticklabels([f"k={kk}" for kk in range(6)])
            ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=7)
            ax.set_title(f"{SHORT[m]}: fixed S=5 exact grid by training support x test k (* = train-supported; N/A = not "
                         "representable)")
            R.fig(fig, f"(c) Group 1 support x k heatmap ({SHORT[m]})", figd / f"group1_heatmap_{m}.png")
        R.table("(c) Accuracy by condition, model, k (exact / cell / structural / strict)", by_k, 400)
    sing = cat("singletons")
    T_["singletons"] = sing
    if len(sing):
        R.table("(c) Singletons by operation x slot", sing, 400)
    geo = cat("singleton_geometry")
    T_["singleton_geometry"] = geo
    if len(geo):
        R.table("(g) Singleton outcomes: exact / identity copy / other legal / non-legal / invalid / extra-C residual",
                geo, 200)
    # (d) cross-condition singleton comparisons, averaged within (op, grid) over each condition's own slots
    def means(c, m):
        t = _read(root, c, m, seed, "singleton_grid_means")
        return t.set_index(["word", "grid_hash"]).exact if t is not None else None

    pairs_spec = []
    for sup in ("train02", "train023", "train0234", "train02345"):
        pairs_spec.append((f"identity_s5_{sup}", f"variable_{sup}", "fixed S5 vs true variable (same logical stream)"))
    for sup, nz in (("train02", "train2"), ("train023", "train23"), ("train0234", "train234"), ("train02345", "train2345")):
        pairs_spec.append((f"variable_{sup}", f"variable_nozero_{nz}", "variable with k=0 vs without (exposure redistributed)"))
    pairs_spec.append(("identity_s5_train02345", "identity_s6_train02345", "S5 vs S6 (same logical stream)"))
    rows = []
    for a_, b_, label in pairs_spec:
        for m in RS.MODELS:
            ma, mb = means(a_, m), means(b_, m)
            if ma is None or mb is None:
                rows.append({"comparison": label, "A": a_, "B": b_, "model": m, "status": "not yet available"})
                continue
            for op in "ABC":
                j = pd.concat([ma.xs(op, level=0).rename("A"), mb.xs(op, level=0).rename("B")], axis=1, join="inner")
                rows.append({"comparison": label, "A": a_, "B": b_, "model": m, "op": op, "status": "paired by grid hash",
                             "grids": len(j), "A_mean": j.A.mean(), "B_mean": j.B.mean(), "B_better": int((j.B > j.A).sum()),
                             "tied": int((j.B == j.A).sum()), "B_worse": int((j.B < j.A).sum())})
    T_["comparisons"] = pd.DataFrame(rows)
    R.table("(d) Singleton comparisons between new conditions (within-grid mean over each condition's own slots)",
            T_["comparisons"], 200)
    # (d)/(j) old no-ROW vs new ROW, only where a matching legacy condition exists
    legacy_rows = []
    try:
        found, _ = discover([str(Path(results_root) / s) for s in ("identity_varlen_02345_all_models",
                                                                    "identity_ar_count_extrapolation", "identity_ar_slot_count")],
                            verify_sha=False, log=lambda *x: None)
    except Exception as exc:
        found = {}
        log(f"legacy discovery failed: {exc!r}")
    for c in RS.CONDITIONS:
        for m in RS.MODELS:
            lid = LEGACY.get(c)
            base = {"condition": c, "model": m, "legacy_run": f"{lid}/{m}" if lid else "none"}
            if lid is None:
                legacy_rows.append({**base, "status": "no matching legacy no-ROW condition (new condition)"})
                continue
            info = found.get(f"{lid}/{m}")
            new = means(c, m)
            if info is None or not str(info.get("status", "")).startswith("available"):
                legacy_rows.append({**base, "status": "legacy predictions unavailable (not counted as zero)"})
                continue
            if new is None:
                legacy_rows.append({**base, "status": "new run not yet available"})
                continue
            a = load(info["predictions"])
            s1 = a["kind"] == "single"
            old = pd.DataFrame({"word": a["string"][s1].astype(str), "grid_hash": a["grid_hash"][s1], "exact": a["exact"][s1]}) \
                .groupby(["word", "grid_hash"]).exact.mean()
            for op in "ABC":
                j = pd.concat([old.xs(op, level=0).rename("old"), new.xs(op, level=0).rename("new")], axis=1, join="inner")
                legacy_rows.append({**base, "op": op, "status": "paired by (op, grid hash); " + LEGACY_NOTES[c],
                                    "grids": len(j), "old_noROW_mean": j.old.mean(), "new_ROW_mean": j.new.mean(),
                                    "improved": int((j.new > j.old).sum()), "tied": int((j.new == j.old).sum()),
                                    "worsened": int((j.new < j.old).sum())})
    T_["legacy"] = pd.DataFrame(legacy_rows)
    R.table("(d)/(j) Old no-ROW vs new ROW singletons (only matching legacy conditions; base grids matched, training "
            "streams/inits as noted)", T_["legacy"], 200)
    for name, title in (("singleton_spatial", "(e) Singleton spatial error profile (rows/columns/first column/wrong-cell counts)"),
                        ("row_conditional", "(f) Held-out digit error in row r given the preceding generated ROW (descriptive)"),
                        ("same_target_pairs", "(h) Same-target pairs (only words in each condition's manifest)"),
                        ("six_step", "(i) S=6 six-operation results by n_C and parity x transform membership"),
                        ("singleton_consistency", "(j) Singleton consistency across slots (fixed layouts)")):
        t = cat(name)
        T_[name] = t
        if len(t):
            R.table(title, t.drop(columns=[c for c in t.columns if c.startswith("cell_err_")]) if name == "singleton_spatial" else t, 300)
    sp = T_.get("singleton_spatial")
    if sp is not None and len(sp):
        rowsel = sp[sp.op.isin(["A", "B", "C"])]
        n = len(rowsel)
        fig, axes = plt.subplots(int(np.ceil(n / 6)), 6, figsize=(15, 2.6 * int(np.ceil(n / 6))), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, r) in zip(axes.flat, rowsel.iterrows()):
            mat = np.array([[r[f"cell_err_r{i}c{j}"] for j in range(6)] for i in range(6)])
            ax.axis("on"); ax.imshow(mat, cmap="Reds", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r.condition.replace('identity_', '').replace('variable_', 'var_')}\n{SHORT[r.model]} {r.op}", fontsize=6)
        fig.suptitle("(e) Singleton 6x6 cell error rate (red = wrong)")
        fig.tight_layout()
        R.fig(fig, "(e) Singleton 6x6 error heatmaps", figd / "singleton_error_maps.png")
    # (k) resources
    res = []
    for c, m in [(r.condition, r.model) for r in st.itertuples() if r.train == "done"]:
        d = root / c / f"seed_{seed}"
        ts = json.loads((d / "runs" / m / "train_summary.json").read_text()) if (d / "runs" / m / "train_summary.json").exists() else {}
        rc = json.loads((d / "runs" / m / "run_config.json").read_text()) if (d / "runs" / m / "run_config.json").exists() else {}
        art = json.loads((d / "eval" / m / "ARTIFACT.json").read_text()) if (d / "eval" / m / "ARTIFACT.json").exists() else {}
        S = RS.n_slots(c)
        res.append({"condition": c, "model": m, "params": ts.get("unique_params"), "op_tokens": S if S else "k",
                    "seq_len": f"{87 + S}" if S else "87+k", "supervised_targets": 42, "updates": ts.get("optimizer_steps"),
                    "examples": ts.get("examples_seen"), "train_min": (ts.get("train_seconds") or 0) / 60,
                    "eval_min": (art.get("eval_seconds") or 0) / 60, "peak_gb_train": ts.get("peak_gpu_memory_gb_this_session"),
                    "peak_gb_eval": art.get("peak_gpu_gb_eval"), "microbatch": ts.get("microbatch_size"),
                    "hardware": ts.get("hardware"), "dtype": ts.get("dtype"), "config_hash": rc.get("config_hash"),
                    "model_hash": rc.get("model_hash"), "best_step": ts.get("best_step"), "repo_commit": ts.get("repo_commit")})
    T_["resources"] = pd.DataFrame(res)
    R.table("(k) Resources and provenance", T_["resources"], 30)
    aud = root / "audits" / f"seed_{seed}"
    for name in ("layout", "initialization", "streams", "validation_audit", "registry"):
        p = aud / f"{name}.csv"
        if p.exists():
            R.table(f"Audit: {name}", pd.read_csv(p), 60)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, t in T_.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    six_w = cat("six_per_word")
    six_w.to_csv(out / "tables" / "six_per_word.csv", index=False)
    rp = write_review(R.items, out, "Row-separator suite report",
                      "# Row-separator suite — review snapshot\n\nAll tables and figures are baked in; nothing to run. "
                      f"{int(st.complete.sum())}/26 jobs complete.", "AR_Row_Separators_Review.ipynb")
    log(f"report -> {out}\nREVIEW FILE: {rp} ({rp.stat().st_size / 2**20:.2f} MiB)")
    return rp
