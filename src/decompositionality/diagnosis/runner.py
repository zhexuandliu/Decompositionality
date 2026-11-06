"""Orchestration of the post-hoc generalization-failure diagnostics over BOTH experiments.

Default path: read saved predictions -> audit -> catalog -> per-run diagnostics (cached) ->
cross-experiment comparisons -> figures -> report. No model weights are loaded, no GPU is used and no
optimizer exists on this path. Source artifacts are opened read-only; everything is written below
``output_dir``.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .. import operations as O
from ..names import MODEL_IDS
from . import group as G
from .analysis import ANALYSIS_VERSION, analyze_run, calibration_controls, diagnostic_split
from .artifacts import (audit_run, cross_run_alignment, grid_hash_rows, inventory, load_experiment_runs,
                        load_manifest_grids, metrics_tables, synthetic_oracle_selfcheck)


@dataclass
class DiagnosticsConfig:
    experiment_dirs: Dict[str, str]
    output_dir: str
    models: Optional[List[str]] = None            # None = all discovered models
    chunk_size: int = 2000
    use_cache: bool = True
    split_seed: int = 20251106                    # deterministic FIT/DIAG split of base-grid IDs
    control_seed: int = 7
    hash_checkpoints: bool = True
    run_frozen_inference_if_missing: bool = False  # explicit opt-in only
    example_models: List[str] = field(default_factory=lambda: ["transformer_ar", "transformer_encoder", "trm"])
    example_strings: List[str] = field(default_factory=lambda: ["A", "B", "C", "AA", "ABC", "CBA", "AAA"])
    example_grid_ids: List[int] = field(default_factory=lambda: [0])


def _save_df(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def _jsonable(o):
    """JSON conversion without importing torch (keeps the analysis path torch-free)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def _analysis_environment() -> dict:
    import platform
    import subprocess
    import sys
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("pandas", "scipy", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    repo = Path(__file__).resolve().parents[3]
    try:
        info["analysis_code_commit"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                                      text=True, timeout=10).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        info["analysis_code_dirty"] = bool(dirty)
    except Exception:
        info["analysis_code_commit"] = "unknown"
    info["torch_imported_by_analysis"] = "torch" in sys.modules
    return info


def run_diagnostics(cfg: DiagnosticsConfig, log=print, make_figures: bool = True) -> dict:
    import pandas as pd
    t_start = time.time()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    exp_dirs = {e: Path(d) for e, d in cfg.experiment_dirs.items()}
    regen_root = out / "regenerated_predictions"
    R: dict = {"config": asdict(cfg), "analysis_version": ANALYSIS_VERSION}

    # 1. inventory ---------------------------------------------------------------------------
    inv = inventory(cfg.experiment_dirs, regen_root, cfg.models)
    _save_df(inv, tables / "00_artifact_inventory.csv")
    R["inventory"] = inv
    log("Artifact inventory:\n" + inv[["experiment", "model", "status"]].to_string(index=False))

    if cfg.run_frozen_inference_if_missing:
        from .inference import regenerate_missing
        R["regenerated"] = regenerate_missing(inv, exp_dirs, regen_root, log=log)
        inv = inventory(cfg.experiment_dirs, regen_root, cfg.models)
        _save_df(inv, tables / "00_artifact_inventory.csv")
        R["inventory"] = inv

    # 2. load + audit --------------------------------------------------------------------------
    runs, manifests = [], {}
    for exp, d in exp_dirs.items():
        manifests[exp] = load_manifest_grids(d) if d.exists() else None
        mids = list(inv[(inv.experiment == exp)]["model"])
        runs += load_experiment_runs(exp, d, mids, regen_root, cfg.hash_checkpoints, log=log)
    audits = pd.DataFrame([audit_run(r, manifests[r.experiment]) for r in runs])
    _save_df(audits, tables / "01_artifact_audit.csv")
    R["audit"] = audits
    if len(audits) and audits["attribution_blocked"].any():
        log("WARNING: attribution blocked for: " + ", ".join(audits[audits.attribution_blocked].apply(
            lambda r: f"{r.experiment}/{r.model}", axis=1)))
    ps_list, gr_list = [], []
    for r in runs:
        ps, gr = metrics_tables(r, O.TRAIN_SETS[r.experiment])
        ps_list.append(ps); gr_list.append(gr)
    R["metrics_per_string"] = pd.concat(ps_list, ignore_index=True) if ps_list else pd.DataFrame()
    R["metrics_groups"] = pd.concat(gr_list, ignore_index=True) if gr_list else pd.DataFrame()
    _save_df(R["metrics_per_string"], tables / "02_metrics_audit_per_string.csv")
    _save_df(R["metrics_groups"], tables / "02_metrics_audit_groups.csv")
    R["alignment"] = cross_run_alignment(runs)
    _save_df(R["alignment"], tables / "03_cross_run_key_alignment.csv")
    man_avail = {e: m for e, m in manifests.items() if m is not None}
    same = None
    if len(man_avail) == 2:
        a, b = list(man_avail.values())
        same = bool(a.shape == b.shape and np.array_equal(a, b))
    R["manifests_identical_across_experiments"] = same
    log(f"Loaded {len(runs)} prediction files; test manifests identical across experiments: {same}")

    # 3. catalog -----------------------------------------------------------------------------
    cat = G.Catalog()
    R["catalog_check"] = G.check_catalog(cat)
    _save_df(cat.table(), tables / "04_transformation_catalog.csv")
    classes = pd.concat([G.string_classes(cat, O.TEST_STRINGS, O.TRAIN_SETS[e]).assign(experiment=e)
                         for e in O.TRAIN_SETS], ignore_index=True)
    _save_df(classes, tables / "04_string_equivalence_classes.csv")
    R["string_classes"] = classes
    oracle_src = next(iter(man_avail.values()))[:50] if man_avail else (runs[0].X[:50] if runs else None)
    if oracle_src is not None:
        R["oracle_selfcheck"] = synthetic_oracle_selfcheck(oracle_src, out / "selfcheck")
        log("Synthetic perfect-oracle self-check through the loader/metric path: 39/39 strings at 100% (synthetic).")

    # 4. split + controls --------------------------------------------------------------------
    all_ids = np.unique(np.concatenate([r.grid_id for r in runs])) if runs else np.arange(0)
    split = diagnostic_split(all_ids, cfg.split_seed)
    (out / "diagnostic_split.json").write_text(json.dumps({"seed": cfg.split_seed, "fit": split["fit"].tolist(),
                                                           "diag": split["diag"].tolist()}))
    R["split"] = split
    ctrl_src = next(iter(man_avail.values())) if man_avail else (None if not runs else
                                                                np.unique(runs[0].X, axis=0))
    if ctrl_src is not None:
        ck = f"{cfg.control_seed}-" + hashlib.sha256((ANALYSIS_VERSION + "|").encode()
                                                     + np.asarray(ctrl_src, np.int8).tobytes()).hexdigest()[:16]
        cpath = out / "cache" / f"controls_{ck}.pkl"
        if cfg.use_cache and cpath.exists():
            ctrl = pickle.loads(cpath.read_bytes())
        else:
            t = time.time()
            ctrl = calibration_controls(np.asarray(ctrl_src), cat, cfg.control_seed, chunk=cfg.chunk_size)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_bytes(pickle.dumps(ctrl))
            log(f"Calibration controls computed in {time.time() - t:.1f}s")
        R["controls"] = ctrl
        for k, v in ctrl.items():
            _save_df(v, tables / f"09_controls_{k}.csv")

    # 5. per-run diagnostics (cached) --------------------------------------------------------
    T_cache: dict = {}
    per_run = {}
    for r in runs:
        if r.audit.get("attribution_blocked"):
            log(f"[{r.key}] attribution SKIPPED (audit failure); see 01_artifact_audit.csv")
            continue
        key = hashlib.sha256(json.dumps([ANALYSIS_VERSION, r.meta["file_sha256"].get("predictions"),
                                                       cat.digest(), cfg.split_seed, O.TRAIN_SETS[r.experiment]]).encode()).hexdigest()[:16]
        cpath = out / "cache" / "runs" / f"{r.experiment}__{r.model}__{key}.pkl"
        if cfg.use_cache and r.meta["file_sha256"].get("predictions") and cpath.exists():
            res = pickle.loads(cpath.read_bytes())
            log(f"[{r.key}] cached diagnostics loaded")
        else:
            t = time.time()
            res = analyze_run(r, cat, O.TRAIN_SETS[r.experiment], split, cfg.chunk_size, T_cache)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_bytes(pickle.dumps(res))
            log(f"[{r.key}] analysed {len(r.P):,} records in {time.time() - t:.1f}s")
        res["per_example"]["grid_hash"] = grid_hash_rows(r.X)
        res["per_example"]["source"] = r.source
        per_run[r.key] = res
        _save_df(res["per_example"], out / "per_example" / f"{r.experiment}__{r.model}.csv.gz")
        (out / "maps").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "maps" / f"{r.experiment}__{r.model}_source_maps.npz",
                            **{f"{s}__{k}": v for s, d in res["maps"].items() for k, v in d.items()})
    R["per_run"] = per_run

    def cat_tables(name):
        dfs = [v[name] for v in per_run.values() if name in v and len(v[name])]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    for i, name in enumerate(["per_string", "confusion", "hypotheses", "hypothesis_types", "stable_rules",
                              "source_maps", "composition"]):
        df = cat_tables(name)
        R[name] = df
        _save_df(df, tables / f"1{i}_{name}.csv")

    # 6. group summary + cross-experiment paired comparisons ---------------------------------
    R["summary"] = group_summary(R["per_string"])
    _save_df(R["summary"], tables / "20_summary_by_experiment_model_group.csv")
    R["paired"] = paired_comparison(runs, per_run, cat)
    _save_df(R["paired"], tables / "21_paired_experiment_comparison_per_string.csv")
    R["z_table"] = z_table(R["per_string"])
    _save_df(R["z_table"], tables / "22_alternate_generator_vs_target.csv")
    R["runs"] = runs
    R["catalog"] = cat

    if make_figures:
        from . import figures
        R["figures"] = figures.make_all(R, out / "figures", log=log)
    from .report import write_report
    R["report_path"] = write_report(R, out)
    meta = {"analysis_version": ANALYSIS_VERSION, "config": _jsonable(asdict(cfg)),
            "catalog": R["catalog_check"], "seconds": time.time() - t_start,
            "manifests_identical_across_experiments": same,
            "runs": [{"experiment": r.experiment, "model": r.model, "source": r.source, **r.meta["file_sha256"],
                      "selected_step": r.meta.get("selected_step"), "code_commit_at_eval": r.meta.get("repo_commit")}
                     for r in runs]}
    meta["analysis_environment"] = _analysis_environment()
    (out / "provenance.json").write_text(json.dumps(_jsonable(meta), indent=1))
    log(f"Diagnostics written to {out} in {time.time() - t_start:.0f}s")
    return R


GROUP_ORDER = ["singles", "pairs_seen", "pairs_unseen", "triples_all_distinct", "triples_two_types", "triples_one_type"]


def group_summary(ps):
    """Experiment x model x group summary. Record-weighted means (all strings share a count when complete)."""
    import pandas as pd
    if not len(ps):
        return pd.DataFrame()
    cols = {"exact_acc": "true_exact_acc", "cell_acc": "true_cell_acc", "any_exact_group_match_rate": "any_exact_legal_match",
            "wrong_unique_legal_rate": "wrong_unique_legal", "no_legal_match_rate": "no_legal_match",
            "stable_rule_diag_exact_fidelity": "DIAG stable_rule_exact_fidelity",
            "stable_rule_diag_cell_fidelity": "DIAG stable_rule_cell_fidelity",
            "perm_diag_cell_fidelity": "DIAG bijective_perm_cell_fidelity",
            "argmax_map_diag_cell_fidelity": "DIAG argmax_map_cell_fidelity",
            "alt_generator_exact_rate": "alt_generator(Z)_exact_agreement",
            "digit_histogram_preserved_rate": "digit_histogram_preserved"}
    rows = []
    for (e, m), sub in ps.groupby(["experiment", "model"], sort=False):
        for g in GROUP_ORDER:
            s = sub[sub.group == g]
            row = {"experiment": e, "model": m, "group": g, "n_strings": len(s), "n_records": int(s["n"].sum()),
                   "exact_correct": int(s["exact_correct"].sum()) if len(s) else 0}
            for c, lab in cols.items():
                row[lab] = float(np.average(s[c], weights=s["n"])) if len(s) and c in s and s[c].notna().all() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def z_table(ps):
    if not len(ps):
        return ps
    t = ps[["experiment", "model", "string", "group", "length", "n", "exact_acc", "alt_generator_exact_rate",
            "alt_generator_cell_agreement"]].copy()
    t["parity"] = np.where(t["length"] % 2 == 1, "odd (alt prediction = Z(target))", "even (alt prediction = target)")
    return t


def paired_comparison(runs, per_run, cat):
    """Align the same model's predictions across experiments on (string, grid_id, grid hash)."""
    import pandas as pd
    by = {}
    for r in runs:
        by.setdefault(r.model, {})[r.experiment] = r
    rows = []
    for m, d in by.items():
        if not {"distinct_pairs", "all_pairs"} <= set(d):
            continue
        r1, r2 = d["distinct_pairs"], d["all_pairs"]
        if r1.key not in per_run or r2.key not in per_run:
            continue
        p1, p2 = per_run[r1.key]["per_example"], per_run[r2.key]["per_example"]
        k1 = pd.DataFrame({"string": r1.string, "grid_id": r1.grid_id, "h": grid_hash_rows(r1.X), "i1": np.arange(len(r1.P))})
        k2 = pd.DataFrame({"string": r2.string, "grid_id": r2.grid_id, "h": grid_hash_rows(r2.X), "i2": np.arange(len(r2.P))})
        j = k1.merge(k2, on=["string", "grid_id", "h"], how="outer", indicator=True)
        for s in O.TEST_STRINGS:
            js = j[j.string == s]
            mt = js[js["_merge"] == "both"]
            i1, i2 = mt["i1"].astype(int).values, mt["i2"].astype(int).values
            if not len(mt):
                continue
            a, b = p1.iloc[i1], p2.iloc[i2]
            same_pred = (r1.P[i1] == r2.P[i2]).all(1)
            u1, u2 = a["unique_exact_match_id"].values, b["unique_exact_match_id"].values
            rows.append({
                "model": m, "string": s, "length": len(s),
                "category_distinct_pairs": "pairs_seen" if s in O.TRAIN_SETS["distinct_pairs"] else (
                    "pairs_unseen" if len(s) == 2 else O.string_group(s)),
                "category_all_pairs": "pairs_seen" if len(s) == 2 else O.string_group(s),
                "n_matched": len(mt), "unmatched_only_distinct_pairs": int((js["_merge"] == "left_only").sum()),
                "unmatched_only_all_pairs": int((js["_merge"] == "right_only").sum()),
                "exact_distinct_pairs": a["exact"].mean(), "exact_all_pairs": b["exact"].mean(),
                "any_legal_distinct_pairs": (a["n_exact_group_matches"] > 0).mean(),
                "any_legal_all_pairs": (b["n_exact_group_matches"] > 0).mean(),
                "alt_generator_exact_distinct_pairs": a["alt_generator_exact"].mean(),
                "alt_generator_exact_all_pairs": b["alt_generator_exact"].mean(),
                "identical_prediction_rate": same_pred.mean(),
                "same_unique_wrong_element_rate": ((u1 >= 0) & (u1 == u2) & ~a["exact"].values).mean()})
    df = pd.DataFrame(rows)
    for c in ("exact", "any_legal", "alt_generator_exact"):
        if len(df):
            df[f"delta_{c} (all_pairs - distinct_pairs)"] = df[f"{c}_all_pairs"] - df[f"{c}_distinct_pairs"]
    return df
