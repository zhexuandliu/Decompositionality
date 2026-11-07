"""Notebook 17: diagnostic-only analysis of SAVED Transformer AR / Looped Transformer AR predictions (identity format,
fixed S=5 supports 02/023/0234/02345, slot-count runs S=2/3/4/6, optional variable_length history).

Nothing here trains, fine-tunes or re-evaluates a model. Every stage reads saved artifacts, adds tables / figures to a
``Report`` (shown inline and later baked into a self-contained review notebook + HTML), and writes CSV exports.
Behavioural diagnosis only: outputs are described, no internal mechanism is inferred.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import mirror_diag as M

VERSION = "ar-mirror-diag-v1"
MODELS = ["transformer_ar", "looped_transformer_ar"]
SHORT = {"transformer_ar": "TF-AR", "looped_transformer_ar": "Looped-AR"}
DEFAULT_SUBDIRS = ["identity_varlen_02345_all_models", "identity_ar_count_extrapolation", "identity_ar_slot_count"]
SUPPORT_KS = {"train02": (0, 2), "train023": (0, 2, 3), "train0234": (0, 2, 3, 4), "train02345": (0, 2, 3, 4, 5)}
CONTROL_WORDS = ["EMPTY", "CC", "CCCC", "BBBB", "ACC", "BCC", "CCC", "CCCCC", "AAAAAA", "CCCCCC"]
PAIRS = [("A", "ACC"), ("B", "BCC"), ("C", "CCC"), ("C", "CCCCC")]
SPLIT_SEED = 20251106
REPORTED = {   # user-reported observations, reconciled (never used as computed outcomes)
    "S2/train02/transformer_ar": {"singleton": (0, 6000)}, "S2/train02/looped_transformer_ar": {"singleton": (0, 6000)},
    "S3/train023/transformer_ar": {"singleton": (0, 9000)}, "S3/train023/looped_transformer_ar": {"singleton": (0, 9000)},
    "S5/train023/transformer_ar": {"singleton": (0, 15000)}, "S5/train023/looped_transformer_ar": {"singleton": (0, 15000)},
    "S4/train0234/transformer_ar": {"singleton": (12000, 12000)}, "S4/train0234/looped_transformer_ar": {"singleton": (0, 12000)},
    "S5/train0234/transformer_ar": {"singleton": (3796, 15000)}, "S5/train0234/looped_transformer_ar": {"singleton": (0, 15000)},
    "S5/train02345/transformer_ar": {"singleton": (15000, 15000)},
    "S5/train02345/looped_transformer_ar": {"singleton": (14858, 15000)},
    "S6/train02345/transformer_ar": {"singleton": (11011, 18000), "A": (5265, 6000), "B": (5746, 6000), "C": (0, 6000),
                                     "k6": (342221, 729000), "k6_seen_transform": (253188, 591000),
                                     "k6_unseen_transform": (89033, 138000)},
    "S6/train02345/looped_transformer_ar": {"singleton": (9665, 18000), "A": (3679, 6000), "B": (5986, 6000),
                                            "C": (0, 6000), "k6": (137160, 729000),
                                            "k6_unseen_transform": (30567, 138000)},
}


def label_of(word: str) -> str:
    return word or "EMPTY"


def word_of(label: str) -> str:
    return "" if label == "EMPTY" else label


# ---------------------------------------------------------------------------
# Report container (inline display + baked review notebook / HTML)
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, show: bool = True):
        self.items: List[tuple] = []
        self.show = show

    def _display(self, obj):
        if not self.show:
            return
        try:
            from IPython.display import display
            display(obj)
        except Exception:
            print(obj)

    def md(self, text: str):
        self.items.append(("md", text))
        if self.show:
            try:
                from IPython.display import Markdown, display
                display(Markdown(text))
            except Exception:
                print(text)

    def table(self, title: str, df, max_rows: int = 60, note: str = ""):
        import pandas as pd
        d = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        shown = d.head(max_rows)
        extra = f" (first {max_rows} of {len(d)} rows; full table in CSV export)" if len(d) > max_rows else ""
        self.items.append(("table", f"{title}{extra}", shown, note))
        if self.show:
            print(f"\n### {title}{extra}" + (f"\n{note}" if note else ""))
            self._display(shown)

    def fig(self, fig, title: str, path: Path, dpi: int = 80):
        import matplotlib.pyplot as plt
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        self.items.append(("fig", title, base64.b64encode(buf.getvalue()).decode()))
        if self.show:
            try:
                from IPython.display import Image, display
                print(f"\n### {title}")
                display(Image(data=buf.getvalue()))
            except Exception:
                pass
        plt.close(fig)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _infer_experiment(art_path: Path, a: dict) -> Optional[str]:
    e = a.get("experiment")
    if e:
        return e
    parts = art_path.parts
    if "identity_varlen_02345_all_models" in parts:
        cond = a.get("condition") or parts[parts.index("identity_varlen_02345_all_models") + 1]
        return {"identity": "idvar_identity", "variable_length": "idvar_variable"}.get(cond)
    for p in parts:
        if p in ("train02", "train023", "train0234"):
            return f"idext_{p}"
        if p.startswith("slots") and "_train" in p:
            return p
    return None


def _describe(experiment: str):
    """experiment id -> (format, S, support)."""
    if experiment is None:
        return None
    if experiment.startswith("idext_"):
        return "fixed", 5, experiment[len("idext_"):]
    if experiment == "idvar_identity":
        return "fixed", 5, "train02345"
    if experiment == "idvar_variable":
        return "variable", None, "train02345"
    if experiment.startswith("slots"):
        S = int(experiment[5:experiment.index("_")])
        return "fixed", S, experiment.split("_", 1)[1]
    return None


def run_id(fmt, S, support, model):
    return f"{'varlen' if fmt == 'variable' else f'S{S}'}/{support}/{model}"


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {"_corrupt": True}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def discover(artifact_roots: List[str], overrides: Optional[Dict[str, str]] = None, include_varlen: bool = True,
             verify_sha: bool = True, log=print):
    """Read-only scan for ARTIFACT.json files (and unindexed test_predictions.npz) under each root. Never writes."""
    import pandas as pd
    found: Dict[str, dict] = {}
    rows = []

    def add(art_path: Optional[Path], npz: Optional[Path], a: dict, origin: str):
        e = _infer_experiment(art_path or npz, a)
        desc = _describe(e)
        model = a.get("model") or (npz or art_path).parent.name
        if desc is None or model not in MODELS:
            return
        fmt, S, support = desc
        if fmt == "variable" and not include_varlen:
            return
        rid = run_id(fmt, S, support, model)
        edir = (art_path or npz).parent        # eval/<model>
        exp_dir = edir.parent.parent
        info = {"run_id": rid, "model": model, "format": fmt, "S": S, "support": support,
                "train_k": SUPPORT_KS.get(support), "experiment": e, "seed": a.get("seed", "unknown"),
                "artifact": str(art_path) if art_path else None, "origin": origin, "artifact_keys": sorted(a),
                "selected_step": a.get("selected_step"), "config_hash": a.get("config_hash"),
                "manifest_hash": a.get("manifest_hash"), "expected_records": a.get("expected_records"),
                "n_records_artifact": a.get("n_records"), "complete_flag": a.get("complete")}
        cand = []
        if a.get("predictions"):
            cand.append(Path(a["predictions"]))
        cand.append(edir / "test_predictions.npz")
        if npz:
            cand.insert(0, npz)
        pred = next((c for c in cand if c.exists()), None)
        info["predictions"] = str(pred) if pred else None
        rc = _json(exp_dir / "runs" / model / "run_config.json")
        es = _json(edir / "eval_summary.json")
        info.update({"run_config_found": bool(rc), "model_hash": rc.get("model_hash"), "data_version": rc.get("data_version"),
                     "initialization": rc.get("initialization", "unknown (not recorded)"),
                     "selection_rule": es.get("selection_rule", rc.get("checkpoint_selection", "unknown")),
                     "selected_val_ce": es.get("selected_val_cell_ce"), "eval_repo_commit": es.get("repo_commit"),
                     "train_recipe": json.dumps(rc.get("train", {}), sort_keys=True) if rc else None,
                     "best_ckpt_exists": (exp_dir / "runs" / model / "best.pt").exists()})
        status = "available"
        if pred is None:
            status = "missing predictions"
        elif a.get("_corrupt"):
            status = "corrupt ARTIFACT.json"
        elif a and a.get("complete") is False:
            status = "partial (ARTIFACT complete=False)"
        if pred is not None and verify_sha and a.get("predictions_sha256"):
            sh = _sha(pred)
            info["predictions_sha256"] = sh
            if sh != a["predictions_sha256"]:
                status = "corrupt (sha mismatch)"
        elif pred is not None:
            info["predictions_sha256"] = _sha(pred) if verify_sha else f"unverified:{pred.stat().st_size}"
        if origin == "unindexed npz":
            status = f"{status}; metadata unknown (no ARTIFACT.json)"
        info["status"] = status
        if rid in found and found[rid]["status"].startswith("available") and not status.startswith("available"):
            return
        if rid in found and origin == "unindexed npz":
            return
        found[rid] = info

    for root in artifact_roots:
        r = Path(root)
        if not r.exists():
            rows.append({"root": root, "exists": False})
            continue
        n0 = len(found)
        arts = sorted(set(r.glob("*/eval/*/ARTIFACT.json")) | set(r.glob("*/seed_*/eval/*/ARTIFACT.json")))
        for f in arts:
            add(f, None, _json(f), "ARTIFACT.json")
        for f in sorted(set(r.glob("*/eval/*/test_predictions.npz")) | set(r.glob("*/seed_*/eval/*/test_predictions.npz"))):
            if not (f.parent / "ARTIFACT.json").exists():
                add(None, f, {}, "unindexed npz")
        rows.append({"root": root, "exists": True, "artifact_files": len(arts), "runs_added": len(found) - n0})
    for rid, path in (overrides or {}).items():
        p = Path(path)
        if p.name == "ARTIFACT.json":
            add(p, None, _json(p), "override")
        else:
            a = _json(p.parent / "ARTIFACT.json")
            add(p.parent / "ARTIFACT.json" if a else None, p, a, "override")
        if rid not in found:
            log(f"override {rid} -> {path} did not resolve to a recognised AR run")
    return found, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loading, audits, classification
# ---------------------------------------------------------------------------

def grid_hash16(x: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(np.asarray(x).reshape(len(x), 36).astype(np.uint8))
    return np.array([hashlib.sha256(r.tobytes()).hexdigest()[:16] for r in flat])


_WID: Dict[str, int] = {}


def true_id(label: str) -> int:
    if label not in _WID:
        _WID[label] = M.group().word_id(word_of(label))
    return _WID[label]


def training_transforms(support: str) -> set:
    from .words import words_of_length
    return {M.group().word_id(w) for k in SUPPORT_KS[support] for w in words_of_length(k)}


def load_and_audit(info: dict) -> tuple:
    """Authoritative records + audit dict. Source arrays are never modified."""
    with np.load(info["predictions"]) as z:
        a = {k: z[k] for k in z.files}
    n = len(a["pred"])
    labels = a["string"].astype(str)
    X = a["input"].reshape(n, 36).astype(np.int16)
    Y = a["target"].reshape(n, 36).astype(np.int16)
    P = a["pred"].reshape(n, 36).astype(np.int16)
    valid = a["valid"].astype(bool)
    tids = np.array([true_id(s) for s in labels], dtype=np.int64)
    G = M.group()
    oracle = np.take_along_axis(X, G.perms[tids], axis=1)
    keys = np.char.add(np.char.add(labels, "@"), np.char.add(a["position"].astype(str), np.char.add(":", a["grid_id"].astype(str))))
    gh = a.get("grid_hash")
    gh_re = grid_hash16(a["input"])
    exact = valid & (P == Y).all(1)
    perfect = (Y == Y).all(1)    # perfect predictor through the same scoring rule (pred := target, valid)
    k = a["k"].astype(int)
    kinds = a["kind"].astype(str)
    singles = kinds == "single"
    au = {"run_id": info["run_id"], "records": n, "expected_records": info.get("expected_records"),
          "count_ok": info.get("expected_records") in (None, n), "duplicate_keys": int(n - len(np.unique(keys))),
          "oracle_target_mismatches": int((oracle != Y).any(1).sum()),
          "grid_hash_mismatches": int((gh != gh_re).sum()) if gh is not None else "no stored hash",
          "stored_exact_mismatches": int((exact != a["exact"].astype(bool)).sum()),
          "invalid_records": int((~valid).sum()), "invalid_coded_as_minus1": bool(((P == -1).any(1) == ~valid).all()),
          "perfect_predictor_exact": float(perfect.mean()),
          "singleton_records": int(singles.sum()), "k6_records": int((k == 6).sum()),
          "decoding": ("greedy free-running from prefix through OUTPUT (evaluate_idvar_model)"
                       if "raw_tokens" in a else "UNVERIFIABLE (no raw tokens saved)"),
          "label_spans": "36 answer digits + EOS after OUTPUT (ARLayout; verified in notebook 16 layout audit)"}
    au["ok"] = (au["duplicate_keys"] == 0 and au["oracle_target_mismatches"] == 0 and au["stored_exact_mismatches"] == 0
                and au["grid_hash_mismatches"] in (0, "no stored hash") and au["perfect_predictor_exact"] == 1.0)
    sel = singles | (kinds == "six") | np.isin(labels, CONTROL_WORDS)
    rec = {"labels": labels, "kind": kinds, "k": k, "position": a["position"].astype(int), "grid_id": a["grid_id"].astype(int),
           "grid_hash": gh_re, "true_id": tids, "valid": valid, "exact": exact, "P": P, "X": X, "Y": Y,
           "slots": a.get("slots5"), "sel": np.flatnonzero(sel)}
    return rec, au


def classify_run(info: dict, rec: dict, matcher: M.Matcher, cache_dir: Optional[Path] = None):
    """Per-record diagnostic frame for singletons, six-step records and named controls (cached by predictions sha)."""
    import pandas as pd
    key = f"{info.get('predictions_sha256', 'nosha')[:24]}_{VERSION}"
    cf = Path(cache_dir) / f"{key}.pkl" if cache_dir else None
    if cf is not None and cf.exists():
        try:
            return pd.read_pickle(cf), True
        except Exception:
            pass
    G = M.group()
    s = rec["sel"]
    P, X, Y = rec["P"][s], rec["X"][s], rec["Y"][s]
    valid, exact = rec["valid"][s], rec["exact"][s]
    matches = matcher.match(P, X, rec["grid_hash"][s], valid)
    c = M.classify(G, matches, rec["true_id"][s], valid, exact)
    tt = training_transforms(info["support"])
    wrong = (P != Y)
    first_wrong = np.where(wrong.any(1), wrong.argmax(1), -1)
    labels = rec["labels"][s]
    df = pd.DataFrame({"rec": s, "kind": rec["kind"][s], "word": labels, "k": rec["k"][s], "position": rec["position"][s],
                       "grid_id": rec["grid_id"][s], "grid_hash": rec["grid_hash"][s], "true_id": rec["true_id"][s],
                       "nC": np.char.count(labels, "C"), "valid": valid, "correct": exact,
                       "cells_correct": (~wrong).sum(1), "first_wrong_cell": first_wrong,
                       "transform_in_training": np.isin(rec["true_id"][s], sorted(tt)),
                       "matches": ["|".join(map(str, m)) for m in matches], **c})
    df.loc[df.word == "EMPTY", "nC"] = 0
    for col in ("outcome", "pred_mirror", "residual", "kind", "word"):
        df[col] = df[col].astype("category")
    if cf is not None:
        cf.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cf)
    return df, False


# ---------------------------------------------------------------------------
# Diagnostics workflow
# ---------------------------------------------------------------------------

class Diagnostics:
    def __init__(self, results_root: str, artifact_roots: Optional[List[str]] = None, run_overrides=None,
                 out_root: Optional[str] = None, include_varlen: bool = True, verify_sha: bool = True,
                 nearest_per_word: int = 20, show: bool = True, log=print):
        self.results_root = Path(results_root)
        self.roots = artifact_roots or [str(self.results_root / s) for s in DEFAULT_SUBDIRS]
        self.overrides = run_overrides or {}
        self.out_root = Path(out_root or self.results_root / "ar_identity_mirror_diagnostics")
        self.include_varlen, self.verify_sha, self.nearest_per_word = include_varlen, verify_sha, nearest_per_word
        self.R = Report(show)
        self.log = log
        self.matcher = M.Matcher()
        self.frames: Dict[str, "pd.DataFrame"] = {}
        self.recs: Dict[str, dict] = {}
        self.T: Dict[str, "pd.DataFrame"] = {}     # every exported table
        self.summary: Dict[str, dict] = {}
        self.t0 = time.time()

    # ---- stage 0: conventions ------------------------------------------------------------------------------
    def conventions(self):
        v = M.verify_conventions()
        self.R.md("## 0. Task and group conventions (verified by independent array operations)\n"
                  "* A(X)[r,c]=X[r,(c-1) mod 6], B(X)[r,c]=X[5-c,r], C(X)[r,c]=X[r,5-c], words left to right "
                  "(AB = B(A(X))). Output-to-source permutations p_g with g(x)=x[p_g]; p_AB = p_A[p_B].\n"
                  "* 288 transforms g = T(dr,dc) o B^r o C^m (mirror first, rotation, translation last), all distinct.\n"
                  "* **Mirror class m = (#C tokens) mod 2**, not the cell-permutation sign (C is an EVEN and B an ODD "
                  "permutation of the 36 cells).\n"
                  "* Residuals: Delta_out = p_true_inv[p_pred], Delta_in = p_pred[p_true_inv]; extra output-C "
                  "<=> Delta_out = C (P_hat = C o P); extra input-C <=> Delta_in = C (P_hat = P o C). For the "
                  "singleton C both equal the identity output (alias kept, never double counted).")
        self.R.table("Convention checks", [{"check": k, "result": str(v_)} for k, v_ in v.items()])
        self.summary["conventions"] = {k: str(x) for k, x in v.items()}

    # ---- stage 1: discovery / audit ------------------------------------------------------------------------
    def discover(self):
        import pandas as pd
        found, roots = discover(self.roots, self.overrides, self.include_varlen, self.verify_sha, self.log)
        self.found = found
        self.R.md("## 1. Run discovery (read-only) and source audit")
        self.R.table("Artifact roots scanned", roots)
        core = [run_id("fixed", S, sup, m) for (S, sup) in [(5, "train02"), (5, "train023"), (5, "train0234"),
                                                            (5, "train02345"), (2, "train02"), (3, "train023"),
                                                            (4, "train0234"), (6, "train02345")] for m in MODELS]
        opt = [run_id("variable", None, "train02345", m) for m in MODELS]
        rows = []
        for rid in core + opt:
            i = found.get(rid)
            rows.append({"run_id": rid, "role": "core" if rid in core else "optional (variable_length)",
                         **({k: i.get(k) for k in ("status", "model", "format", "S", "support", "train_k", "seed",
                                                   "selected_step", "selection_rule", "config_hash", "model_hash",
                                                   "manifest_hash", "data_version", "initialization",
                                                   "expected_records", "origin", "predictions")}
                            if i else {"status": "NOT FOUND in scanned roots (unavailable; not retrained)"})})
        inv = pd.DataFrame(rows)
        inv["train_k"] = inv["train_k"].astype(str)
        self.T["inventory"] = inv
        self.R.table("Run inventory (16 core + 2 optional)", inv[["run_id", "role", "status", "S", "train_k", "seed",
                                                                 "selected_step", "config_hash", "manifest_hash"]], 30)
        self.R.table("Run metadata", inv[["run_id", "selection_rule", "model_hash", "data_version", "initialization",
                                          "expected_records", "origin"]], 30)
        man = inv.manifest_hash.dropna().unique()
        self.R.md(f"Test-manifest hashes across found runs: {sorted(map(str, man))} "
                  f"({'one shared manifest' if len(man) == 1 else 'MORE THAN ONE - joins use actual grid content hashes'}).")
        self.runs = [r for r in core + opt if r in found and found[r]["status"].startswith("available")]
        self.summary["inventory"] = inv[["run_id", "status"]].to_dict("records")

    def load(self, cache: bool = True):
        import pandas as pd
        self.R.md("## 2. Load saved predictions, audit, exact legal-transform matching")
        audits, failed = [], []
        cache_dir = self.out_root / "cache" if cache else None
        for rid in list(self.runs):
            info = self.found[rid]
            t = time.time()
            try:
                rec, au = load_and_audit(info)
                if not au["ok"]:
                    info["status"] = "QUARANTINED (audit failed)"
                    failed.append(rid)
                    audits.append(au)
                    continue
                df, cached = classify_run(info, rec, self.matcher, cache_dir)
                df.insert(0, "run_id", rid)
                df["model"] = info["model"]
                df["S"] = info["S"]
                df["support"] = info["support"]
                df["format"] = info["format"]
                self.frames[rid] = df
                keep = rec["sel"]
                self.recs[rid] = {"P": rec["P"][keep].astype(np.int8), "X": rec["X"][keep].astype(np.int8),
                                  "Y": rec["Y"][keep].astype(np.int8), "valid": rec["valid"][keep]}
                self._full_counts(rid, rec)
                au["matching"] = "cache" if cached else "computed"
                au["seconds"] = round(time.time() - t, 1)
                audits.append(au)
                self.log(f"  [{rid}] {len(df):,} diagnostic records ({'cached' if cached else 'matched'}) in {au['seconds']}s")
            except Exception as exc:
                traceback.print_exc()
                info["status"] = f"QUARANTINED (load error: {exc!r})"[:200]
                failed.append(rid)
        self.runs = [r for r in self.runs if r not in failed]
        self.T["audit"] = pd.DataFrame(audits)
        self.R.table("Prediction audits (perfect predictor scored through the same exact-grid rule)", self.T["audit"])
        if failed:
            self.R.md(f"**Quarantined / failed sources (excluded, not counted as zero): {failed}**")
        self._reconcile()

    def _full_counts(self, rid, rec):
        f = self.summary.setdefault("full_counts", {})
        s = rec["kind"] == "single"
        d = {"singleton": (int(rec["exact"][s].sum()), int(s.sum()))}
        for op in "ABC":
            m = s & (rec["labels"] == op)
            d[op] = (int(rec["exact"][m].sum()), int(m.sum()))
        six = rec["k"] == 6
        if six.any():
            d["k6"] = (int(rec["exact"][six].sum()), int(six.sum()))
            tt = np.isin(rec["true_id"], sorted(training_transforms(self.found[rid]["support"])))
            d["k6_seen_transform"] = (int(rec["exact"][six & tt].sum()), int((six & tt).sum()))
            d["k6_unseen_transform"] = (int(rec["exact"][six & ~tt].sum()), int((six & ~tt).sum()))
        f[rid] = d

    def _reconcile(self):
        import pandas as pd
        rows = []
        for rid, exp in REPORTED.items():
            obs = self.summary.get("full_counts", {}).get(rid)
            for key, (c, n) in exp.items():
                if obs is None:
                    rows.append({"run_id": rid, "quantity": key, "reported": f"{c}/{n}", "observed": "unavailable",
                                 "delta": "", "status": "cannot reconcile (run unavailable)"})
                    continue
                oc, on = obs.get(key, (None, None))
                st = "match" if (oc, on) == (c, n) else ("DIFFERENT DENOMINATOR (different run/manifest?)" if on != n
                                                          else "DIFFERENT COUNT (same denominator)")
                rows.append({"run_id": rid, "quantity": key, "reported": f"{c}/{n}", "observed": f"{oc}/{on}",
                             "delta": (oc - c) if oc is not None else "", "status": st,
                             "selected_step": self.found.get(rid, {}).get("selected_step")})
        self.T["reconciliation"] = pd.DataFrame(rows)
        self.R.table("Reconciliation with user-reported counts (reported values are NOT used as results)",
                     self.T["reconciliation"], 40)

    def overview(self):
        import pandas as pd
        rows = []
        for rid in self.runs:
            f = self.summary["full_counts"][rid]
            i = self.found[rid]
            rows.append({"run_id": rid, "S": i["S"], "train_k": str(i["train_k"]),
                         "singleton exact": "{}/{}".format(*f["singleton"]),
                         **{op: "{}/{}".format(*f[op]) for op in "ABC"},
                         "k6 exact": "{}/{}".format(*f["k6"]) if "k6" in f else "N/A",
                         "selected_step": i["selected_step"]})
        self.T["overview"] = pd.DataFrame(rows)
        self.R.md("## Overview")
        self.R.table("Overview of analysed runs (full-population exact counts)", self.T["overview"], 30)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _cat(self, pred):
        import pandas as pd
        L = [df[pred(df)] for df in self.frames.values()]
        L = [x for x in L if len(x)]
        return pd.concat(L, ignore_index=True) if L else pd.DataFrame()

    @staticmethod
    def _top(series, n=3):
        vc = series.value_counts()
        return "; ".join(f"{k} ({v})" for k, v in vc.head(n).items() if v)

    # ---- stage 3: singletons -------------------------------------------------------------------------------
    def singletons(self):
        import pandas as pd
        G = M.group()
        self.R.md("## 3. Singleton diagnostics (held out in every run)\nOutcome categories are disjoint. Hypothesis "
                  "match counts (identity / true / extra output-C / extra input-C) can OVERLAP; for X=C the two "
                  "extra-C hypotheses and identity are the same transform (alias).")
        s = self._cat(lambda d: d.kind == "single")
        if not len(s):
            self.R.md("No singleton predictions available.")
            return
        s["delta_label"] = [G.label(int(x)) if x >= 0 else "" for x in s.delta_out_id]
        s["unique_label"] = [G.label(int(x)) if x >= 0 else "" for x in s.unique_id]
        rows = []
        for (rid, op, pos), g in s.groupby(["run_id", "word", "position"], observed=True, sort=False):
            w = g[~g.correct]
            ident = g.mirror_identifiable
            rows.append({"run_id": rid, "op": op, "position": pos, "exact": f"{int(g.correct.sum())}/{len(g)}",
                         "exact_acc": g.correct.mean(), "cell_acc": g.cells_correct.mean() / 36,
                         "invalid_rate": (~g.valid).mean(), "legal_match_coverage": (g.n_match > 0).mean(),
                         "mirror_identifiable_coverage": ident.mean(),
                         "match_identity": int(g.match_identity.sum()), "match_true": int(g.match_true.sum()),
                         "match_extra_out_C": int(g.match_extra_out_C.sum()), "match_extra_in_C": int(g.match_extra_in_C.sum()),
                         "mirror_wrong_on_identifiable": (g.mirror_wrong[ident].mean() if ident.any() else np.nan),
                         "mirror_wrong_of_all_records": g.mirror_wrong.mean(),
                         "top_wrong_unique_transforms": self._top(w.unique_label[w.unique_label != ""]),
                         "top_residual_delta_out": self._top(w.delta_label[w.delta_label != ""])})
        pos_t = pd.DataFrame(rows)
        self.T["singleton_by_position"] = pos_t
        op_rows = []
        for (rid, op), g in s.groupby(["run_id", "word"], observed=True, sort=False):
            oc = g.outcome.value_counts()
            w = g[~g.correct & (g.n_match > 0)]
            op_rows.append({"run_id": rid, "op": op, "n": len(g), "correct": int(g.correct.sum()),
                            **{o: int(oc.get(o, 0)) for o in M.OUTCOMES[1:]},
                            "wrong legal: identity": int((w.match_identity).sum()),
                            "wrong legal: extra out-C": int(w.match_extra_out_C.sum()),
                            "wrong legal: extra in-C": int(w.match_extra_in_C.sum()),
                            "wrong legal: union(out-C, in-C)": int((w.match_extra_out_C | w.match_extra_in_C).sum()),
                            **{f"residual: {r}": int((w.residual == r).sum()) for r in M.RESIDUALS},
                            "correct but ambiguous transform": int((g.correct & (g.n_match > 1)).sum())})
        op_t = pd.DataFrame(op_rows)
        self.T["singleton_by_op"] = op_t
        self.R.table("Singleton outcomes by run x operation (disjoint outcome columns sum to n)",
                     op_t[["run_id", "op", "n", "correct"] + M.OUTCOMES[1:] + ["correct but ambiguous transform"]], 60)
        self.R.table("Wrong legal singleton predictions: hypothesis matches (overlapping) and residual classes (disjoint)",
                     op_t[["run_id", "op"] + [c for c in op_t if c.startswith("wrong legal") or c.startswith("residual")]], 60)
        self.R.table("Singleton by operation x position", pos_t.drop(columns=["top_wrong_unique_transforms",
                                                                            "top_residual_delta_out"]), 120)
        self.R.table("Most frequent wrong legal transforms / output residuals by position",
                     pos_t[["run_id", "op", "position", "exact", "top_wrong_unique_transforms", "top_residual_delta_out"]]
                     [pos_t.exact_acc < 1], 80)
        # mirror confusion
        mc = s.groupby(["run_id", "word", "true_mirror", "pred_mirror"], observed=True).size().unstack(fill_value=0).reset_index()
        self.T["singleton_mirror_confusion"] = mc
        self.R.table("Singletons: true geometric mirror class vs predicted (ambiguous / no legal match / invalid kept)", mc, 60)
        self._singleton_figures(s, pos_t, op_t)
        self._cell_maps(s)

    def _singleton_figures(self, s, pos_t, op_t):
        import matplotlib.pyplot as plt
        figd = self.out_root / "figures"
        runs = list(dict.fromkeys(pos_t.run_id))
        for metric, title in (("exact_acc", "exact-grid accuracy"), ("cell_acc", "cell accuracy")):
            nc = 4
            nr = int(np.ceil(len(runs) / nc))
            fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 1.9 * nr), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            for ax, rid in zip(axes.flat, runs):
                t = pos_t[pos_t.run_id == rid]
                P = sorted(t.position.unique())
                mat = np.array([[t[(t.op == op) & (t.position == p)][metric].iloc[0] if len(t[(t.op == op) & (t.position == p)]) else np.nan
                                 for p in P] for op in "ABC"])
                ax.axis("on")
                ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                for i in range(3):
                    for j in range(len(P)):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.6 else "black")
                ax.set_xticks(range(len(P))); ax.set_xticklabels([f"s{p}" for p in P], fontsize=7)
                ax.set_yticks(range(3)); ax.set_yticklabels(list("ABC"), fontsize=7)
                ax.set_title(rid.replace("transformer_ar", "TF").replace("looped_TF", "Looped"), fontsize=8)
            fig.suptitle(f"Singleton {title} by operation x physical slot (actual S per run)")
            fig.tight_layout()
            self.R.fig(fig, f"Singleton {title} heatmaps", figd / f"singleton_{metric}_heatmaps.png")
        # outcome stacked bars
        cols = ["correct"] + M.OUTCOMES[1:]
        colors = ["#1baf7a", "#999999", "#eda100", "#2a78d6", "#8e6bd6"]
        t = op_t.copy()
        t["label"] = t.run_id.str.replace("transformer_ar", "TF").str.replace("looped_TF", "Looped") + " " + t.op.astype(str)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(t))))
        left = np.zeros(len(t))
        for c, col in zip(cols, colors):
            v = t[c].values / t.n.values
            ax.barh(range(len(t)), v, left=left, color=col, label=c)
            left += v
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.label, fontsize=6); ax.invert_yaxis()
        ax.set_xlabel("fraction of singleton records"); ax.legend(fontsize=7, loc="lower right")
        ax.set_title("Singleton outcome categories (disjoint)")
        self.R.fig(fig, "Singleton outcome categories", figd / "singleton_outcomes.png")
        # residual summary
        rc = [f"residual: {r}" for r in M.RESIDUALS]
        sub = t[t[rc].sum(1) > 0]
        if len(sub):
            fig, ax = plt.subplots(figsize=(10, max(3, 0.22 * len(sub))))
            left = np.zeros(len(sub))
            tot = sub[rc].sum(1).values
            for c, col in zip(rc, ["#e34948", "#eb6834", "#b8336a", "#8e6bd6", "#2a78d6", "#bbbbbb"]):
                v = sub[c].values / tot
                ax.barh(range(len(sub)), v, left=left, color=col, label=c.replace("residual: ", ""))
                left += v
            ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.label, fontsize=6); ax.invert_yaxis()
            ax.set_xlabel("fraction of WRONG predictions that match a legal transform")
            ax.legend(fontsize=7, loc="lower right"); ax.set_title("Residual class of wrong legal singleton predictions")
            self.R.fig(fig, "Singleton residual classes", figd / "singleton_residuals.png")
        # mirror confusion figure
        cats = ["0", "1", "ambiguous", "no legal match", "invalid"]
        g = s.groupby(["run_id", "true_mirror", "pred_mirror"], observed=True).size().unstack(fill_value=0)
        g = g.reindex(columns=cats, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(g))))
        frac = g.div(g.sum(1), axis=0)
        ax.imshow(frac.values, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        for i in range(frac.shape[0]):
            for j in range(frac.shape[1]):
                ax.text(j, i, str(int(g.values[i, j])), ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"pred m={c}" if c in "01" else c for c in cats], fontsize=7)
        ax.set_yticks(range(len(g))); ax.set_yticklabels([f"{r[0].replace('transformer_ar','TF')} | true m={r[1]}" for r in g.index], fontsize=6)
        ax.set_title("Singletons: true mirror class (A,B: m=0; C: m=1) vs predicted (counts)")
        self.R.fig(fig, "Singleton mirror-class confusion", figd / "singleton_mirror_confusion.png")

    def _cell_maps(self, s):
        import matplotlib.pyplot as plt
        import pandas as pd
        sel = []
        for rid in self.runs:
            for op in "ABC":
                g = s[(s.run_id == rid) & (s.word == op)]
                if len(g) and g.correct.mean() < 1 and (g.cells_correct.mean() / 36) > 0.2:
                    sel.append((rid, op))
        prio = [x for x in sel if x[0].startswith(("S3/train023", "S5/train023")) and x[1] in "AB"] + \
               [x for x in sel if x[0].startswith("S6/") and x[1] == "C"]
        sel = list(dict.fromkeys(prio + sel))[:12]
        if not sel:
            return
        fig, axes = plt.subplots(2, len(sel), figsize=(2.3 * len(sel), 4.6), squeeze=False)
        rows = []
        for j, (rid, op) in enumerate(sel):
            df = self.frames[rid]
            m = (df.kind == "single").values & (df.word == op).values
            r = self.recs[rid]
            corr = (r["P"][m] == r["Y"][m]) & r["valid"][m][:, None]
            ax = axes[0, j]
            ax.imshow(corr.mean(0).reshape(6, 6), cmap="Blues", vmin=0, vmax=1)
            for c in range(36):
                ax.text(c % 6, c // 6, f"{corr.mean(0)[c]:.2f}", ha="center", va="center", fontsize=5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{rid.replace('transformer_ar','TF')}\n{op}: per-cell acc", fontsize=6)
            nw = 36 - corr.sum(1)
            axes[1, j].hist(nw, bins=np.arange(38) - 0.5, color="#2a78d6")
            axes[1, j].set_title("# wrong cells", fontsize=6)
            axes[1, j].tick_params(labelsize=5)
            fw = df[m].first_wrong_cell
            rows.append({"run_id": rid, "op": op, "records": int(m.sum()), "mean_wrong_cells": float(nw.mean()),
                         "records_with_1-3_wrong_cells": int(((nw >= 1) & (nw <= 3)).sum()),
                         "records_with_>=30_wrong_cells": int((nw >= 30).sum()),
                         "per-cell acc min/median/max": f"{corr.mean(0).min():.2f}/{np.median(corr.mean(0)):.2f}/{corr.mean(0).max():.2f}",
                         "first wrong generated cell (top, descriptive)": self._top(fw[fw >= 0], 4)})
        fig.tight_layout()
        self.R.fig(fig, "Per-cell accuracy maps and wrong-cell counts for selected failing singleton groups",
                   self.out_root / "figures" / "singleton_cell_maps.png")
        self.T["singleton_cell_error_profile"] = pd.DataFrame(rows)
        self.R.table("Spatial error profile of selected failing singleton groups (first wrong cell is descriptive only)",
                     self.T["singleton_cell_error_profile"])

    # ---- stage 4: same-target comparisons ------------------------------------------------------------------
    def equivalence(self):
        import pandas as pd
        self.R.md("## 4. Same-target instruction comparisons (same input grids)\nSingletons have S placements per grid, "
                  "the composite one saved layout per grid; per-position rows plus a within-grid summary (grids are the "
                  "unit; positions are not independent replications). These comparisons change word, k and I layout "
                  "together; they are not causal interventions on I.")
        rows, grid_rows, ctrl = [], [], []
        for rid in self.runs:
            df = self.frames[rid]
            ks = SUPPORT_KS[self.found[rid]["support"]]
            r = self.recs[rid]
            for sing, comp in PAIRS:
                cm = (df.word == comp).values & (df.kind != "single").values
                sm = (df.word == sing).values & (df.kind == "single").values
                base = {"run_id": rid, "singleton": sing, "composite": comp,
                        "composite_in_training_support": len(comp) in ks}
                if not cm.any():
                    rows.append({**base, "position": "N/A", "status": "N/A (composite not in this run's manifest; k>S or absent)"})
                    continue
                C_ = df[cm].set_index("grid_hash")
                Ci = np.flatnonzero(cm)
                cpos = {h: i for h, i in zip(C_.index, Ci)}
                same_target = 0
                gmeans = []
                for pos, g in df[sm].groupby("position"):
                    idx_s = g.index.values
                    hs = g.grid_hash.values
                    ic = np.array([cpos.get(h, -1) for h in hs])
                    ok = ic >= 0
                    tgt_same = (r["Y"][idx_s[ok]] == r["Y"][ic[ok]]).all(1)
                    same_target += int(tgt_same.sum())
                    sc = g.correct.values[ok]
                    cc = df.correct.values[ic[ok]]
                    agree = (r["P"][idx_s[ok]] == r["P"][ic[ok]]).all(1)
                    dis = g[ok][cc & ~sc]
                    rows.append({**base, "position": int(pos), "paired_grids": int(ok.sum()),
                                 "targets_identical": int(tgt_same.sum()),
                                 "composite correct / singleton wrong": int((cc & ~sc).sum()),
                                 "both correct": int((cc & sc).sum()), "singleton correct / composite wrong": int((~cc & sc).sum()),
                                 "both wrong": int((~cc & ~sc).sum()), "exact prediction agreement": int(agree.sum()),
                                 "singleton outcome when composite correct & singleton wrong": self._top(dis.outcome.astype(str), 2),
                                 "singleton residual (same subset)": self._top(dis.residual.astype(str)[dis.residual.astype(str) != ""], 2)})
                # within-grid summary
                sg = df[sm].groupby("grid_hash").correct.mean()
                cj = C_.correct.reindex(sg.index)
                okg = cj.notna()
                sgm, cjm = sg[okg], cj[okg].astype(bool)
                grid_rows.append({**base, "grids": int(okg.sum()), "composite correct": int(cjm.sum()),
                                  "mean singleton acc | composite correct": float(sgm[cjm].mean()) if cjm.any() else np.nan,
                                  "mean singleton acc | composite wrong": float(sgm[~cjm].mean()) if (~cjm).any() else np.nan,
                                  "grids: composite correct & singleton wrong at ALL positions": int((cjm & (sgm == 0)).sum()),
                                  "grids: composite correct & singleton correct at ALL positions": int((cjm & (sgm == 1)).sum()),
                                  "grids: composite wrong & singleton correct at ALL positions": int((~cjm & (sgm == 1)).sum())})
            # identity-target controls
            for w in ["EMPTY", "CC", "CCCC", "BBBB", "AAAAAA", "CCCCCC"]:
                g = df[(df.word == w).values]
                if not len(g):
                    continue
                ctrl.append({"run_id": rid, "word": w, "k": len(word_of(w)), "kind": str(g.kind.iloc[0]),
                             "in_training_support": len(word_of(w)) in ks,
                             "exact": f"{int(g.correct.sum())}/{len(g)}", "exact_acc": g.correct.mean(),
                             "wrong outcome (top)": self._top(g[~g.correct].outcome.astype(str), 2)})
        self.T["equivalence_by_position"] = pd.DataFrame(rows)
        self.T["equivalence_within_grid"] = pd.DataFrame(grid_rows)
        self.T["identity_controls"] = pd.DataFrame(ctrl)
        self.R.table("Same-target pairs per singleton position (A~ACC, B~BCC, C~CCC, C~CCCCC)", self.T["equivalence_by_position"], 120)
        self.R.table("Same-target pairs: within-grid summary (grid = unit)", self.T["equivalence_within_grid"], 60)
        self.R.table("Identity-target controls (EMPTY, CC, CCCC, BBBB; S=6 held-out AAAAAA, CCCCCC)", self.T["identity_controls"], 80)

    # ---- stage 5: source maps -------------------------------------------------------------------------------
    def source_maps(self):
        import pandas as pd
        from .diagnosis.analysis import diagnostic_split
        self.R.md("## 5. Fixed source-position copying maps (post-hoc, descriptive)\nFit on a fixed 500-grid half "
                  f"(diagnostic_split seed {SPLIT_SEED}, same split for every run), evaluated on the other 500 grids, "
                  "using the group's FULL prediction population (invalid predictions count as mismatches). Ties: point "
                  "map uses the lowest source index; tied cells are counted. Not evidence of an internal copying circuit.")
        rows = []
        for rid in self.runs:
            df = self.frames[rid]
            r = self.recs[rid]
            split = diagnostic_split(np.unique(df.grid_id), SPLIT_SEED)
            fit = set(split["fit"].tolist())
            for (op, pos), g in df[df.kind == "single"].groupby(["word", "position"], observed=True):
                if g.correct.all():
                    continue
                ix = g.index.values
                f = np.array([gid in fit for gid in g.grid_id])
                res = M.source_map(r["P"][ix[f]], r["X"][ix[f]], r["valid"][ix[f]], r["P"][ix[~f]], r["X"][ix[~f]], r["valid"][ix[~f]])
                res.pop("map")
                rows.append({"run_id": rid, "op": op, "position": pos, "exact_acc": g.correct.mean(),
                             "cell_acc": g.cells_correct.mean() / 36, **res})
        self.T["source_maps"] = pd.DataFrame(rows)
        if rows:
            self.R.table("Source maps for singleton groups with failures (held-out fidelity)", self.T["source_maps"], 120)
        else:
            self.R.md("No failing singleton groups.")

    # ---- stage 6: six-step ---------------------------------------------------------------------------------
    def six_step(self):
        import pandas as pd
        G = M.group()
        self.R.md("## 6. S=6 six-operation diagnosis (all 729 words x 1000 grids per model)\nStrata differ in A/B "
                  "composition and word structure; differences are descriptive, not causal effects of C or novelty. "
                  "One C inside a longer word WAS trained; what was held out is total non-identity count 1 (and 6).")
        six_runs = [rid for rid in self.runs if (self.frames[rid].kind == "six").any()]
        if not six_runs:
            self.R.md("**S=6 predictions not available - six-step section unavailable.**")
            return
        # independent recomputation of stratum sizes / transform membership
        from math import comb
        tt = training_transforms("train02345")
        words6 = ["".join(w) for w in product("ABC", repeat=6)]
        nC = np.array([w.count("C") for w in words6])
        seen = np.array([G.word_id(w) in tt for w in words6])
        chk = {"training transforms": len(tt), "six-op words seen-transform / unseen": f"{seen.sum()}/{(~seen).sum()}",
               "words by n_C": [int((nC == c).sum()) for c in range(7)],
               "expected": [comb(6, c) * 2 ** (6 - c) for c in range(7)]}
        assert chk["words by n_C"] == chk["expected"] and len(tt) == 111 and seen.sum() == 591
        self.R.table("Recomputed stratum sizes (records = words x 1000)", [{"check": k, "value": str(v)} for k, v in chk.items()])
        strata, cross, perword, sub_rows, near_rows = [], [], [], [], []
        for rid in six_runs:
            df = self.frames[rid]
            x = df[df.kind == "six"].copy()
            x["delta_label"] = [G.label(int(v)) if v >= 0 else "" for v in x.delta_out_id]
            assert len(x) == 729 * x.grid_id.nunique()
            for c, g in x.groupby("nC"):
                pw = g.groupby("word", observed=True).correct.mean()
                ident = g.mirror_identifiable
                strata.append({"run_id": rid, "n_C": int(c), "mirror parity": "odd" if c % 2 else "even", "words": int(g.word.nunique()),
                               "records": len(g), "exact": f"{int(g.correct.sum())}/{len(g)}", "exact_acc": g.correct.mean(),
                               "cell_acc": g.cells_correct.mean() / 36, "invalid_rate": (~g.valid).mean(),
                               "legal_coverage": (g.n_match > 0).mean(), "mirror_identifiable_coverage": ident.mean(),
                               "wrong & mirror correct (identifiable)": int((~g.correct & ident & ~g.mirror_wrong).sum()),
                               "wrong & mirror WRONG (identifiable)": int((~g.correct & g.mirror_wrong).sum()),
                               "wrong, mirror ambiguous": int((~g.correct & (g.pred_mirror == "ambiguous")).sum()),
                               "wrong, no legal match": int((g.outcome == "wrong: valid, no legal match").sum()),
                               "wrong, invalid": int((~g.valid).sum()),
                               "per-word acc median [min,max]": f"{pw.median():.3f} [{pw.min():.3f},{pw.max():.3f}]",
                               "dominant residuals (unique wrong)": self._top(g.delta_label[g.delta_label != ""], 3)})
            for (par, st), g in x.groupby([x.nC % 2, x.transform_in_training]):
                cross.append({"run_id": rid, "parity of n_C": "odd" if par else "even",
                              "target transform in training": bool(st), "words": int(g.word.nunique()),
                              "exact": f"{int(g.correct.sum())}/{len(g)}", "exact_acc": g.correct.mean(),
                              "mirror wrong (all records)": g.mirror_wrong.mean()})
            pw = x.groupby("word", observed=True).agg(n_C=("nC", "first"), transform_in_training=("transform_in_training", "first"),
                                                      exact=("correct", "sum"), n=("correct", "size"),
                                                      cell_acc=("cells_correct", "mean"),
                                                      mirror_wrong=("mirror_wrong", "sum"),
                                                      no_legal=("n_match", lambda v: int((v == 0).sum()))).reset_index()
            pw["cell_acc"] /= 36
            pw.insert(0, "run_id", rid)
            perword.append(pw)
            # truncation / deletion checks via match sets (aliases kept)
            fail = x[~x.correct]
            names = {"first five": lambda w: w[:5], "last five": lambda w: w[1:]}
            names.update({f"delete op {i + 1}": (lambda w, i=i: w[:i] + w[i + 1:]) for i in range(6)})
            hit = {k: np.zeros(len(fail), bool) for k in names}
            for j, (w, ms) in enumerate(zip(fail.word.astype(str), fail.matches)):
                if not ms:
                    continue
                s = set(map(int, ms.split("|")))
                for k, fn in names.items():
                    hit[k][j] = G.word_id(fn(w)) in s
            union = np.zeros(len(fail), bool)
            for k in names:
                union |= hit[k]
                sub_rows.append({"run_id": rid, "check": k, "failed": len(fail), "matches": int(hit[k].sum())})
            sub_rows.append({"run_id": rid, "check": "UNION (each record once)", "failed": len(fail), "matches": int(union.sum())})
            # nearest transforms on a deterministic per-word subset of failures
            r = self.recs[rid]
            fs = fail.copy()
            fs["h"] = [hashlib.sha256(f"{rid}|{w}|{h}".encode()).hexdigest() for w, h in zip(fs.word.astype(str), fs.grid_hash)]
            pick = fs.sort_values("h").groupby("word", observed=True).head(self.nearest_per_word)
            fcount = fail.groupby("word", observed=True).size()
            for i, row in zip(pick.index, pick.itertuples()):
                nb = M.nearest(G, r["P"][i], r["X"][i])
                near_rows.append({"run_id": rid, "word": row.word, "weight": fcount[row.word] / (pick.word == row.word).sum(),
                                  "best_cells": nb["best_cells"], "n_ties": nb["n_ties"], "margin": nb["margin"],
                                  "nearest_mirror_matches_true": (str(int(G.mirror[row.true_id])) in nb["tied_mirror"]) and len(nb["tied_mirror"]) == 1,
                                  "nearest_mirror_ambiguous": len(nb["tied_mirror"]) > 1})
        st = pd.DataFrame(strata)
        self.T["six_strata"] = st
        self.T["six_cross"] = pd.DataFrame(cross)
        self.T["six_per_word"] = pd.concat(perword, ignore_index=True)
        self.T["six_subsequence"] = pd.DataFrame(sub_rows)
        nr = pd.DataFrame(near_rows)
        self.T["six_nearest_sample"] = nr
        self.R.table("Six-step strata by number of C tokens", st, 20)
        self.R.table("Six-step: n_C parity x target transform in training support", self.T["six_cross"])
        if len(nr):
            agg = nr.groupby("run_id").apply(lambda g: pd.Series({
                "sampled failures": len(g), "words sampled": g.word.nunique(),
                "weighted mean best cell agreement": np.average(g.best_cells, weights=g.weight),
                "weighted frac nearest mirror == true": np.average(g.nearest_mirror_matches_true, weights=g.weight),
                "weighted frac nearest mirror ambiguous": np.average(g.nearest_mirror_ambiguous, weights=g.weight),
                "weighted mean margin": np.average(g.margin, weights=g.weight)})).reset_index()
            self.T["six_nearest_summary"] = agg
            self.R.table(f"Nearest legal transform on failed six-step records (<= {self.nearest_per_word} per word, stable "
                         "hash selection; weights = per-word failures / sampled). Nearest != exact legality.", agg)
        self.R.table("Truncation / single-deletion checks on failed six-step records (descriptive; overlaps kept)",
                     self.T["six_subsequence"], 30)
        # side-by-side mirror evidence
        side = []
        for rid in six_runs:
            df = self.frames[rid]
            groups = {"singleton C": (df.kind == "single") & (df.word == "C"), "seen CCC": (df.word == "CCC") & (df.kind == "seen"),
                      "seen CCCCC": (df.word == "CCCCC") & (df.kind == "seen"),
                      "six-step mixed odd-C (n_C 1,3,5)": (df.kind == "six") & df.nC.isin([1, 3, 5]),
                      "six-step even-C (n_C 0,2,4,6)": (df.kind == "six") & df.nC.isin([0, 2, 4, 6])}
            for name, m in groups.items():
                g = df[m.values]
                if not len(g):
                    side.append({"run_id": rid, "group": name, "status": "N/A"})
                    continue
                side.append({"run_id": rid, "group": name, "records": len(g), "exact_acc": g.correct.mean(),
                             "mirror wrong (all records)": g.mirror_wrong.mean(),
                             "mirror wrong | identifiable": g.mirror_wrong[g.mirror_identifiable].mean() if g.mirror_identifiable.any() else np.nan,
                             "identifiable coverage": g.mirror_identifiable.mean()})
        self.T["mirror_side_by_side"] = pd.DataFrame(side)
        self.R.table("Mirror evidence side by side", self.T["mirror_side_by_side"], 20)
        self._six_figures(st, six_runs)
        self._gallery(six_runs)

    def _six_figures(self, st, six_runs):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)
        for rid, ls in zip(six_runs, ("-", "--")):
            t = st[st.run_id == rid]
            axes[0, 0].plot(t.n_C, t.exact_acc, ls=ls, marker="o", label=rid)
            wr = t["wrong & mirror WRONG (identifiable)"] / t.records
            axes[0, 1].plot(t.n_C, wr, ls=ls, marker="o", label=rid)
        axes[0, 0].set(xlabel="number of C tokens", ylabel="exact accuracy", title="Six-step exact accuracy by n_C", ylim=(0, 1))
        axes[0, 1].set(xlabel="number of C tokens", ylabel="fraction of ALL records",
                       title="Six-step: wrong with identifiably wrong mirror class", ylim=(0, 1))
        for a in axes[0]:
            a.legend(fontsize=7)
        fig.tight_layout()
        self.R.fig(fig, "Six-step accuracy and mirror errors by n_C", self.out_root / "figures" / "six_step_by_nC.png")

    def _gallery(self, six_runs):
        import matplotlib.pyplot as plt
        G = M.group()
        cats = {"correct": lambda d: d.correct,
                "extra mirror (wrong mirror class, legal)": lambda d: ~d.correct & d.mirror_wrong,
                "other legal (mirror correct)": lambda d: ~d.correct & (d.n_match > 0) & ~d.mirror_wrong,
                "no legal match": lambda d: d.outcome == "wrong: valid, no legal match"}
        items = []
        for rid in six_runs:
            df = self.frames[rid]
            x = df[df.kind == "six"]
            for name, fn in cats.items():
                g = x[fn(x).values]
                if not len(g):
                    continue
                h = [hashlib.sha256(f"gallery|{rid}|{w}|{gh}".encode()).hexdigest() for w, gh in zip(g.word.astype(str), g.grid_hash)]
                i = g.index[int(np.argmin(h))]
                items.append((rid, name, i, len(g), len(x)))
        if not items:
            return
        fig, axes = plt.subplots(len(items), 3, figsize=(6.5, 2.1 * len(items)), squeeze=False)
        for row, (rid, name, i, cnt, tot) in enumerate(items):
            df = self.frames[rid]
            r = self.recs[rid]
            meta = df.loc[i]
            for col, (arr, ttl) in enumerate(((r["X"][i], "input"), (r["Y"][i], "target"), (r["P"][i], "prediction"))):
                ax = axes[row, col]
                ax.imshow(np.where(arr < 0, -1, arr).reshape(6, 6), cmap="tab10", vmin=-1, vmax=9)
                for c in range(36):
                    ax.text(c % 6, c // 6, str(int(arr[c])) if arr[c] >= 0 else "x", ha="center", va="center", fontsize=6)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(ttl if col else f"{rid.replace('transformer_ar','TF')}\n{name} ({cnt}/{tot})\n{meta.word}", fontsize=6)
            uid = int(meta.unique_id)
            axes[row, 2].set_xlabel(f"pred transform: {G.label(uid) if uid >= 0 else meta.pred_mirror}", fontsize=6)
        fig.suptitle("Deterministic gallery (lowest stable hash per category; counts = category size, not frequency claims)", fontsize=7)
        fig.tight_layout()
        self.R.fig(fig, "Six-step example gallery", self.out_root / "figures" / "six_step_gallery.png")

    # ---- stage 7: placement / slot / format comparisons ----------------------------------------------------
    def comparisons(self):
        import pandas as pd
        from . import idext as E, idslot as I
        self.R.md("## 7. Placement, slot-count and format comparisons\n(i) fixed visible-I vs genuinely variable-length "
                  "input and (ii) fixed-slot designs with vs without zero-I training instructions are kept separate. "
                  "At fixed S, n_I = S - k is determined by k; no separate causal effects are estimated.")
        place = []
        for rid in self.runs:
            df = self.frames[rid]
            r = self.recs[rid]
            s = df[df.kind == "single"]
            for op, g in s.groupby("word", observed=True):
                acc = g.groupby("position").correct.mean()
                byg = g.groupby("grid_hash")
                allc = byg.correct.all()
                allw = ~byg.correct.any()
                ident = []
                for h, gg in byg:
                    ps = r["P"][gg.index.values]
                    ident.append(bool((ps == ps[0]).all()))
                place.append({"run_id": rid, "op": op, "positions": int(g.position.nunique()), "mean_acc": acc.mean(),
                              "min_acc": acc.min(), "max_acc": acc.max(), "grids all positions correct": float(allc.mean()),
                              "grids identical prediction across positions": float(np.mean(ident)),
                              "grids all positions wrong": float(allw.mean())})
        self.T["placement"] = pd.DataFrame(place)
        self.R.table("Placement invariance within each run (agreement separate from correctness)", self.T["placement"], 60)
        # n_I / no-I training fractions
        nrows = []
        for rid in self.runs:
            i = self.found[rid]
            if i["format"] == "variable":
                frac = "N/A (no I tokens)"
            elif i["S"] == 5:
                frac = 0.2 if i["support"] == "train02345" else 0.0
            else:
                e = f"slots{i['S']}_{i['support']}"
                frac = I.full_run_counts(e).get(i["S"], 0) / 1_280_000
            nrows.append({"run_id": rid, "S": i["S"], "train_k": str(i["train_k"]), "no-I training fraction": frac,
                          "singleton n_I": (i["S"] - 1) if i["S"] else "N/A",
                          "singleton n_I seen in training?": (any(i["S"] - k == i["S"] - 1 for k in i["train_k"]) if i["S"] else "N/A"),
                          "source": "allocation definitions matching run_config data_version " + str(i.get("data_version"))})
        self.T["identity_counts"] = pd.DataFrame(nrows)
        self.R.table("Identity counts and zero-I training exposure (k and n_I perfectly dependent at fixed S)", self.T["identity_counts"])
        # same-support S5 vs new S
        rows, shift = [], []
        for sup, Snew in (("train02", 2), ("train023", 3), ("train0234", 4), ("train02345", 6)):
            for m in MODELS:
                a, b = f"S5/{sup}/{m}", f"S{Snew}/{sup}/{m}"
                if a not in self.frames or b not in self.frames:
                    rows.append({"pair": f"{a} vs {b}", "status": "not available: " + ", ".join(x for x in (a, b) if x not in self.frames)})
                    continue
                ia, ib = self.found[a], self.found[b]
                pairing = ("logical stream matched by construction (idext vs idslot)" if ia.get("data_version") == "decomp-idext-v1"
                           and ib.get("data_version") == "decomp-idslot-v1" else
                           f"stream UNMATCHED or unknown ({ia.get('data_version')} vs {ib.get('data_version')})")
                for op in "ABC":
                    sa = self.frames[a]; sb = self.frames[b]
                    ga = sa[(sa.kind == "single") & (sa.word == op)].groupby("grid_hash").correct.mean()
                    gb = sb[(sb.kind == "single") & (sb.word == op)].groupby("grid_hash").correct.mean()
                    j = pd.concat([ga.rename("old"), gb.rename("new")], axis=1, join="inner")
                    rows.append({"pair": f"{a} vs {b}", "op": op, "paired grids": len(j), "old mean (S=5)": j.old.mean(),
                                 "new mean": j.new.mean(), "mean diff": (j.new - j.old).mean(),
                                 "grids improved": int((j.new > j.old).sum()), "tied": int((j.new == j.old).sum()),
                                 "worsened": int((j.new < j.old).sum()), "pairing": pairing,
                                 "manifest": "same" if ia.get("manifest_hash") == ib.get("manifest_hash") else "different/unknown"})
                    for name, df in (("S=5", sa), (f"S={Snew}", sb)):
                        g = df[(df.kind == "single") & (df.word == op)]
                        oc = g.outcome.value_counts(normalize=True)
                        rs = g[~g.correct & (g.n_match > 0)].residual.astype(str).value_counts()
                        shift.append({"support": sup, "model": m, "op": op, "layout": name,
                                      **{o: round(float(oc.get(o, 0)), 4) for o in M.OUTCOMES},
                                      "wrong-legal residuals": self._top(rs[rs.index != ""], 3)})
        self.T["slot_comparison"] = pd.DataFrame(rows)
        self.T["slot_error_shift"] = pd.DataFrame(shift)
        self.R.table("Same support, S=5 vs new S (paired by operation and grid hash; mean over each run's own positions)",
                     self.T["slot_comparison"], 40)
        self.R.table("Error-class shift across S (fractions of singleton records)", self.T["slot_error_shift"], 60)
        # optional variable-length vs visible-I
        vrows = []
        for m in MODELS:
            a, b = f"S5/train02345/{m}", f"varlen/train02345/{m}"
            if a not in self.frames or b not in self.frames:
                vrows.append({"model": m, "status": "optional history not available"})
                continue
            for op in "ABC":
                for name, rid in (("visible-I S=5", a), ("variable_length", b)):
                    g = self.frames[rid]
                    g = g[(g.kind == "single") & (g.word == op)]
                    oc = g.outcome.value_counts()
                    vrows.append({"model": m, "op": op, "format": name, "exact": f"{int(g.correct.sum())}/{len(g)}",
                                  **{o: int(oc.get(o, 0)) for o in M.OUTCOMES[1:]},
                                  "match identity": int(g.match_identity.sum())})
        self.T["format_comparison"] = pd.DataFrame(vrows)
        self.R.table("Optional: visible-I fixed S=5 vs variable-length (train02345)", self.T["format_comparison"])

    # ---- stage 8: evidence ---------------------------------------------------------------------------------
    def evidence(self):
        import pandas as pd
        T = self.T
        rows = []
        op = T.get("singleton_by_op", pd.DataFrame())

        def status(frac, cov=1.0):
            if frac is None or np.isnan(frac) or cov < 0.5:
                return "unresolved (missing/ambiguous data)"
            return "supported pattern" if frac >= 0.8 else ("mixed pattern" if frac >= 0.2 else "pattern not observed")

        # Q1 singleton C
        c = op[op.op == "C"] if len(op) else op
        if len(c):
            wrong = (c.n - c.correct).sum()
            ident = c["wrong legal: identity"].sum()
            other = (c["wrong: valid, unique legal transform"] + c["wrong: valid, multiple legal transforms"]).sum() - ident
            nolegal = c["wrong: valid, no legal match"].sum()
            inval = c["wrong: invalid output"].sum()
            per = "; ".join(f"{r.run_id}: id {r['wrong legal: identity']}/{r.n - r.correct}" for _, r in c.iterrows() if r.n > r.correct)
            rows.append({"question": "1. Singleton C failures: identity, other legal, or non-legal?",
                         "observed (counts / denominators)": f"wrong C={wrong}: identity {ident}, other legal {other}, "
                                                             f"no legal {nolegal}, invalid {inval}. Per run: {per}",
                         "status": status(ident / wrong if wrong else np.nan) + " (identity)",
                         "competing explanations": "identity output = both extra-C aliases for C; cannot distinguish "
                                                   "'ignores C' from 'applies C twice' from outputs alone"})
        # Q2 train023 A/B vs C
        q2 = op[op.run_id.str.contains("/train023/")] if len(op) else op
        if len(q2):
            ab = q2[q2.op.isin(["A", "B"])]
            w = (ab.n - ab.correct).sum()
            u = ab["wrong legal: union(out-C, in-C)"].sum()
            nl = ab["wrong: valid, no legal match"].sum()
            rows.append({"question": "2. train023: do A/B failures share an extra-mirror error with C?",
                         "observed (counts / denominators)": f"A/B wrong {w}: extra-C (out or in) {u}, no legal match {nl}; "
                                                             + "; ".join(f"{r.run_id} {r.op}: extraC {r['wrong legal: union(out-C, in-C)']}, "
                                                                         f"nolegal {r['wrong: valid, no legal match']}/{r.n - r.correct}"
                                                                         for _, r in ab.iterrows()),
                         "status": status(u / w if w else np.nan) + " (shared extra-C)",
                         "competing explanations": "non-legal A/B outputs are not explained by any single transform; see "
                                                   "source-map fidelity and cell maps"})
        # Q3 composite vs singleton
        eq = T.get("equivalence_within_grid", pd.DataFrame())
        if len(eq):
            tot = eq["grids: composite correct & singleton wrong at ALL positions"].sum()
            cc = eq["composite correct"].sum()
            rows.append({"question": "3. Correct ACC/BCC/CCC but wrong A/B/C on the same grid?",
                         "observed (counts / denominators)": f"grids with composite correct and singleton wrong at all positions: {tot} "
                                                             f"of {cc} composite-correct grid pairs (all runs/pairs pooled; per-run table above)",
                         "status": status(tot / cc if cc else np.nan),
                         "competing explanations": "composite and singleton differ in k, I layout and training support "
                                                   "status; not a causal test of I"})
        # Q4 six-step mirror
        st = T.get("six_strata", pd.DataFrame())
        if len(st):
            parts = []
            for rid, g in st.groupby("run_id"):
                wrong = g.records.sum() - g.exact.str.split("/").str[0].astype(int).sum()
                mw = g["wrong & mirror WRONG (identifiable)"].sum()
                odd = g[g.n_C % 2 == 1]
                even = g[g.n_C % 2 == 0]
                parts.append(f"{rid}: odd-C acc {np.average(odd.exact_acc, weights=odd.records):.3f}, even-C acc "
                             f"{np.average(even.exact_acc, weights=even.records):.3f}; wrong-mirror {mw}/{wrong} failures")
                frac = mw / wrong if wrong else np.nan
            rows.append({"question": "4. Six-step failures concentrated on odd C / wrong mirror class?",
                         "observed (counts / denominators)": " | ".join(parts),
                         "status": status(frac) + " (wrong mirror among failures, last run listed)",
                         "competing explanations": "n_C strata differ in A/B composition; mirror-correct failures can be "
                                                   "rotation/translation errors or non-legal outputs"})
        else:
            rows.append({"question": "4. Six-step failures concentrated on odd C / wrong mirror class?",
                         "observed (counts / denominators)": "S=6 predictions unavailable", "status": "unresolved",
                         "competing explanations": ""})
        # Q5 slots
        sc = T.get("slot_comparison", pd.DataFrame())
        if len(sc) and "mean diff" in sc:
            ok = sc.dropna(subset=["mean diff"])
            rows.append({"question": "5. How do I placement / slot count change the error patterns?",
                         "observed (counts / denominators)": "; ".join(f"{r.pair} {r.op}: {r['old mean (S=5)']:.3f}->{r['new mean']:.3f} "
                                                                     f"(+{r['grids improved']}/={r.tied}/-{r.worsened})" for _, r in ok.iterrows()),
                         "status": "descriptive (see error-class shift table)",
                         "competing explanations": "changing S also changes sequence length, absolute positions, layout "
                                                   "counts and zero-I exposure together; one seed"})
        # Q6 source maps
        sm = T.get("source_maps", pd.DataFrame())
        if len(sm):
            hc = sm[(sm.cell_acc > 0.5) & (sm.exact_acc < 0.5)]
            if len(hc):
                stable = (hc.heldout_exact_fidelity >= 0.8).sum()
                rows.append({"question": "6. High cell-acc / exact-wrong: stable wrong copying maps or input-dependent?",
                             "observed (counts / denominators)": f"{stable}/{len(hc)} groups (cell acc>0.5, exact<0.5) have a "
                                                                 f"source map with held-out exact fidelity>=0.8; median exact fidelity "
                                                                 f"{hc.heldout_exact_fidelity.median():.3f}, median cell fidelity "
                                                                 f"{hc.heldout_cell_fidelity.median():.3f}",
                             "status": status(stable / len(hc)) + " (stable fixed map)",
                             "competing explanations": "a fixed map can fit cells by digit coincidence (10 digits); low "
                                                       "fidelity means the outputs depend on input content beyond a fixed copy"})
            else:
                rows.append({"question": "6. High cell-acc / exact-wrong: stable wrong copying maps or input-dependent?",
                             "observed (counts / denominators)": "no singleton group with cell acc>0.5 and exact<0.5",
                             "status": "N/A", "competing explanations": ""})
        self.T["evidence"] = pd.DataFrame(rows)
        self.R.md("## 8. Evidence table (behavioural; thresholds: >=80% supported, 20-80% mixed, <20% not observed; "
                  "coverage <50% -> unresolved)\nThese describe outputs only; no claim that a model implements a mirror "
                  "bit, a sequential executor or any internal algorithm.")
        self.R.table("Evidence for the six questions", self.T["evidence"], 10)
        st_ = self.T["inventory"]
        self.R.md(f"**Coverage:** {len(self.runs)} runs analysed; unavailable: "
                  f"{', '.join(st_[~st_.run_id.isin(self.runs)].run_id) or 'none'}. Elapsed {(time.time() - self.t0) / 60:.1f} min.")

    # ---- stage 9: export -----------------------------------------------------------------------------------
    def export(self) -> Path:
        import nbformat as nbf
        shas = sorted((rid, self.found[rid].get("predictions_sha256", "")[:16]) for rid in self.runs)
        aid = hashlib.sha256(json.dumps([VERSION, shas]).encode()).hexdigest()[:12]
        out = self.out_root / aid
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for k, df in self.T.items():
            df.to_csv(out / "tables" / f"{k}.csv", index=False)
        figs = self.out_root / "figures"
        if figs.exists():
            (out / "figures").mkdir(exist_ok=True)
            for p in figs.glob("*.png"):
                (out / "figures" / p.name).write_bytes(p.read_bytes())
        summ = {"analysis_id": aid, "version": VERSION, "runs": shas, "analysed": self.runs,
                "inventory": self.summary.get("inventory"), "full_counts": self.summary.get("full_counts"),
                "evidence": self.T.get("evidence").to_dict("records") if "evidence" in self.T else None}
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        lines = [f"AR identity / mirror diagnostics {VERSION} ({aid})", f"analysed runs: {len(self.runs)}", ""]
        for r in summ["evidence"] or []:
            lines += [r["question"], "  observed: " + r["observed (counts / denominators)"], "  status: " + r["status"],
                      "  competing: " + r["competing explanations"], ""]
        (out / "diagnostic_summary.txt").write_text("\n".join(lines))
        # HTML
        html = ["<html><head><meta charset='utf-8'><title>AR identity diagnostics</title><style>body{font-family:sans-serif;"
                "max-width:1400px;margin:auto}table{border-collapse:collapse;font-size:11px}td,th{border:1px solid #ccc;"
                "padding:2px 4px}</style></head><body>", f"<h1>AR identity / mirror diagnostics ({aid})</h1>"]
        for it in self.R.items:
            if it[0] == "md":
                html.append("<div>" + it[1].replace("\n", "<br>") + "</div>")
            elif it[0] == "table":
                html.append(f"<h3>{it[1]}</h3>" + (f"<p>{it[3]}</p>" if it[3] else "") + it[2].to_html(index=False))
            else:
                html.append(f"<h3>{it[1]}</h3><img src='data:image/png;base64,{it[2]}'/>")
        html.append("</body></html>")
        (out / "diagnostic_report.html").write_text("\n".join(html))
        # self-contained review notebook (baked outputs; nothing to execute)
        nb = nbf.v4.new_notebook()
        cells = [nbf.v4.new_markdown_cell(f"# AR identity / mirror diagnostics — review snapshot\n\nAnalysis `{aid}`, "
                                          f"version `{VERSION}`. All outputs are baked in; do NOT run cells (no Drive "
                                          f"needed). Runs analysed: {len(self.runs)}.")]
        for it in self.R.items:
            if it[0] == "md":
                cells.append(nbf.v4.new_markdown_cell(it[1]))
            elif it[0] == "table":
                c = nbf.v4.new_code_cell(f"# table: {it[1]}")
                c.outputs = [nbf.v4.new_output("display_data", data={"text/html": it[2].to_html(index=False),
                                                                     "text/plain": it[2].to_string(index=False)[:20000]})]
                cells.append(nbf.v4.new_markdown_cell(f"### {it[1]}" + (f"\n{it[3]}" if it[3] else "")))
                cells.append(c)
            else:
                c = nbf.v4.new_code_cell(f"# figure: {it[1]}")
                c.outputs = [nbf.v4.new_output("display_data", data={"image/png": it[2], "text/plain": it[1]})]
                cells.append(nbf.v4.new_markdown_cell(f"### {it[1]}"))
                cells.append(c)
        nb.cells = cells
        nbf.validate(nb)
        rp = out / "AR_Identity_Diagnostics_Review.ipynb"
        nbf.write(nb, rp)
        self.review_path = rp
        self.log(f"exports -> {out}\nreview notebook: {rp} ({rp.stat().st_size / 2**20:.2f} MiB)")
        return rp
