"""Notebook-19 supplement: fixed S=2/3/4 identity-slot conditions WITH <ROW> separators, Transformer AR + Looped AR
(3 x 2 = 6 runs). Thin addon over the notebook-18 implementation (same serializer, vocabulary, model config, common
initialization, trainer, evaluator and scoring); results live in their own root and the main 26-job registry is
untouched. Comparisons with the main ROW suite and the notebook-16 no-ROW runs are READ-ONLY and only use artifacts
whose completion markers exist.

Pairing:
  * logical stream (k, word, grid, target, order) = main S5/variable stream at the same support (idext.record);
  * placements = the notebook-16 S=2/3/4 placement streams (idslot STREAM_SLOT_TRAIN / STREAM_SLOT_VAL);
  * test manifests = the notebook-16 S=2/3/4 manifests (16,000 / 46,000 / 130,000 records);
  * initialization = rowsep.common_initial_state (identical tensors to the main ROW suite, same function / seed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from . import rowsep as RS
from . import rowsep_reports as RR
from . import tokenization as T

SUBDIR = "ar_row_slots234_supplement_v1"
SUITE_VERSION = "row-slots234-supplement-v1"
CONDITIONS = list(RS.SUPPLEMENT_CONDITIONS)
MAIN_SAME_SUPPORT = {"identity_s2_train02_row": ("identity_s5_train02", "variable_train02"),
                     "identity_s3_train023_row": ("identity_s5_train023", "variable_train023"),
                     "identity_s4_train0234_row": ("identity_s5_train0234", "variable_train0234")}
NOROW_SAME = {"identity_s2_train02_row": "S2/train02", "identity_s3_train023_row": "S3/train023",
              "identity_s4_train0234_row": "S4/train0234"}


def jobs(conditions=None, models=None) -> List[tuple]:
    conds = [c for c in CONDITIONS if conditions is None or c in conditions]
    unknown = set(conditions or []) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown supplement conditions {sorted(unknown)}")
    return [(c, m) for c in conds for m in RS.MODELS if models is None or m in models]


def preflight(seed: int = 42, B: int = 256, main_root: Path = None):
    """Layout lengths, counts, stream matching, init identity with the main suite (recorded hash when available)."""
    import pandas as pd
    from . import idslot as IS
    from .data import make_rng
    rows = []
    for c in CONDITIONS:
        S, ks = RS.n_slots(c), RS.train_ks(c)
        lay = T.ARRowLayout(S)
        assert (lay.length, lay.idx_output, lay.idx_eos) == (87 + S, 44 + S, 86 + S) and lay.length == {2: 89, 3: 90, 4: 91}[S]
        counts = {k: RS.E.records_before(ks, B, 5000, k) for k in ks}
        exp = RS.expected_test_records(c, 1000)
        assert exp == {2: 16000, 3: 46000, 4: 130000}[S] and sum(counts.values()) == 1_280_000
        assert max(counts.values()) - min(counts.values()) <= 1 and 1 not in ks
        main_s5, main_var = MAIN_SAME_SUPPORT[c]
        same, place16 = True, True
        for step in (0, 1, 2499, 4999):
            b = RS.RowStream(seed, c, B).batch(step)
            for ref in (main_s5, main_var):
                r = RS.RowStream(seed, ref, B).batch(step)
                same &= all(np.array_equal(b[k_], r[k_]) for k_ in ("x", "y", "string", "k_index", "k"))
            h = IS.SlotStream(seed, f"slots{S}_{c.split('_')[2]}", B).batch(step)   # notebook-16 no-ROW stream
            place16 &= np.array_equal(h["slots"], b["slots"]) and np.array_equal(h["string"], b["string"])
            nid = (b["slots"] == RS.V.ID_OP).sum(1)
            assert (nid == S - b["k"]).all() and not (b["k"] == 1).any()
        rows.append({"condition": c, "S": S, "train_k": str(list(ks)), "seq_len": lay.length, "OUTPUT": lay.idx_output,
                     "EOS": lay.idx_eos, "per_k_training_records": str(counts),
                     "no-I training records (k=S)": counts.get(S, 0), "val_records": 300 * len(ks),
                     "test_records_per_model": exp, "singleton_records": 3 * S * 1000,
                     "same logical stream as main S5 & variable (same support)": bool(same),
                     "same logical stream + placements as notebook-16 no-ROW": bool(place16)})
    df = pd.DataFrame(rows)
    assert df["same logical stream as main S5 & variable (same support)"].all()
    ini = []
    rec = {}
    if main_root is not None:
        p = Path(main_root) / "audits" / f"seed_{seed}" / "initialization.csv"
        t = RR.read_csv_safe(p)
        if t is not None:
            rec = t.set_index("model").full_init_sha.to_dict()
    for m in RS.MODELS:
        sd, prov = RS.common_initial_state(m, seed)
        ini.append({"model": m, "full_init_sha": prov["full_sha"], "legacy_shared_sha": prov["legacy_shared_sha"],
                    "vocab": sd["tok_emb.weight"].shape[0], "pos_table": tuple(sd["pos_emb.weight"].shape),
                    "main_suite_recorded_sha": rec.get(m, "not available (main audits not found)"),
                    "matches_main_suite": (rec[m] == prov["full_sha"]) if m in rec else "unverified against a "
                    "recorded hash; same function/seed/config (model_hash identical)"})
    return df, pd.DataFrame(ini)


def _complete(root: Path, c: str, m: str, seed: int) -> bool:
    d = Path(root) / c / f"seed_{seed}" / "eval" / m
    if not (d / "analysis" / "DONE.json").exists() or not (d / "ARTIFACT.json").exists():
        return False
    try:
        return bool(json.loads((d / "ARTIFACT.json").read_text()).get("complete"))
    except Exception:
        return False


def _means(root, c, m, seed):
    import pandas as pd
    if not _complete(root, c, m, seed):
        return None
    p = Path(root) / c / f"seed_{seed}" / "eval" / m / "analysis" / "singleton_grid_means.csv"
    t = RR.read_csv_safe(p)
    return t.set_index(["word", "grid_hash"]).exact if t is not None and len(t) else None


def report(results_root: str, mode: str, seed: int = 42, sel_jobs=None, show=True, log=print) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd
    from .mirror_report import Report, discover, write_review
    root = RR.family_root(results_root, mode, SUBDIR)
    main_root = RR.family_root(results_root, mode)
    out = root / "report" / f"seed_{seed}"
    figd = out / "figures"
    R = Report(show)
    st = RR.status_table(root, seed, sel_jobs or jobs(), jobs())
    R.md(f"# ROW supplement: fixed S=2/3/4 x 2 AR models ({SUITE_VERSION}, seed {seed})\n"
         f"{int(st.complete.sum())}/6 jobs complete" + (" — **ALL SIX COMPLETE**" if st.complete.all() else " — partial"))
    R.table("Status of the six supplement jobs", st, 10)
    T_ = {"status": st}
    done = [(r.condition, r.model) for r in st.itertuples() if r.analyze == "done"]

    def cat(name, base=root, pairs=None):
        L = []
        for c, m in (pairs or done):
            t = RR.read_csv_safe(Path(base) / c / f"seed_{seed}" / "eval" / m / "analysis" / f"{name}.csv")
            if t is not None:
                if len(t):
                    t.insert(0, "model", m); t.insert(0, "condition", c)
                    L.append(t)
        return pd.concat(L, ignore_index=True) if L else pd.DataFrame()

    # curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for c, m in [(r.condition, r.model) for r in st.itertuples() if r.train == "done"]:
        rd = root / c / f"seed_{seed}" / "runs" / m
        ls = "-" if m == "transformer_ar" else "--"
        t = RR.read_csv_safe(rd / "train_log.csv")
        if t is not None:
            for ax, col in zip(axes.flat[:3], ("answer_cell_nll", "row_nll", "eos_nll")):
                if col in t:
                    ax.plot(t.step, t[col].clip(lower=1e-7), ls=ls, lw=0.9, label=f"{c} {RR.SHORT[m]}")
        v = RR.read_csv_safe(rd / "val_log.csv")
        if v is not None:
            axes[1, 1].plot(v.step, v.val_cell_ce.clip(lower=1e-7), ls=ls, lw=0.9, label=f"{c} {RR.SHORT[m]}")
    for ax, ttl in zip(axes.flat, ("train digit CE", "train ROW CE", "train EOS CE", "validation digit CE (selection)")):
        ax.set(yscale="log", title=ttl, xlabel="update")
        ax.legend(fontsize=6)
    fig.tight_layout()
    R.fig(fig, "Training / validation curves", figd / "curves.png")
    for name, title in (("by_k", "Accuracy by k (exact / cell / ROW / EOS / strict)"),
                        ("singletons", "Singletons by operation x slot (every slot)"),
                        ("singleton_geometry", "Singleton outcomes (exact / identity copy / other legal / non-legal / invalid)"),
                        ("singleton_spatial", "Singleton spatial error profile (rows, columns, first column, wrong-cell counts)"),
                        ("row_conditional", "Held-out digit error given the preceding generated ROW (descriptive)"),
                        ("singleton_consistency", "Same-grid singleton consistency across slots"),
                        ("same_target_pairs", "Same-target pairs (A~ACC, B~BCC, C~CCC within each manifest)")):
        t = cat(name)
        T_[name] = t
        if len(t):
            R.table(title, t.drop(columns=[c_ for c_ in t.columns if c_.startswith("cell_err_")]) if name == "singleton_spatial" else t, 400)
    sp = T_.get("singleton_spatial")
    if sp is not None and len(sp):
        fig, axes = plt.subplots(len(done), 3, figsize=(7, 2.3 * len(done)), squeeze=False)
        for i, (c, m) in enumerate(done):
            for j, op in enumerate("ABC"):
                r = sp[(sp.condition == c) & (sp.model == m) & (sp.op == op)]
                ax = axes[i, j]
                if not len(r):
                    ax.axis("off"); continue
                mat = np.array([[r.iloc[0][f"cell_err_r{a}c{b}"] for b in range(6)] for a in range(6)])
                ax.imshow(mat, cmap="Reds", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{c.replace('identity_', '')} {RR.SHORT[m]} {op}", fontsize=6)
        fig.tight_layout()
        R.fig(fig, "Singleton 6x6 cell error heatmaps", figd / "singleton_error_maps.png")
    # comparison 1: same-support main ROW suite (S5 and variable), read-only
    rows = []
    for c in CONDITIONS:
        s5, var = MAIN_SAME_SUPPORT[c]
        for m in RS.MODELS:
            new = _means(root, c, m, seed)
            for ref, label in ((s5, "main ROW S5"), (var, "main ROW variable")):
                old = _means(main_root, ref, m, seed)
                base = {"new": c, "reference": ref, "reference_kind": label, "model": m,
                        "matching": "same logical stream + common init (by construction); same test grids; layouts differ"}
                if new is None or old is None:
                    rows.append({**base, "status": "pending/unavailable: " + ("new run" if new is None else "")
                                 + (" main-suite artifact" if old is None else "")})
                    continue
                for op in "ABC":
                    j = pd.concat([old.xs(op, level=0).rename("ref"), new.xs(op, level=0).rename("new")], axis=1, join="inner")
                    rows.append({**base, "op": op, "status": "paired by (op, grid hash); mean over each condition's own slots",
                                 "grids": len(j), "reference_mean": j.ref.mean(), "new_mean": j.new.mean(),
                                 "improved": int((j.new > j.ref).sum()), "tied": int((j.new == j.ref).sum()),
                                 "worsened": int((j.new < j.ref).sum())})
    T_["comparison_main_row"] = pd.DataFrame(rows)
    R.table("Comparison 1 (read-only): new short-slot ROW vs main ROW suite at the same support", T_["comparison_main_row"], 60)
    # comparison 2: historical no-ROW notebook-16 runs at the same S / support
    rows = []
    try:
        found, _ = discover([str(Path(results_root) / "identity_ar_slot_count")], verify_sha=False, log=lambda *a: None)
    except Exception:
        found = {}
    for c in CONDITIONS:
        for m in RS.MODELS:
            info = found.get(f"{NOROW_SAME[c]}/{m}")
            new = _means(root, c, m, seed)
            base = {"new": c, "historical_noROW": f"{NOROW_SAME[c]}/{m}", "model": m,
                    "matching": "same logical stream, placements, test manifest; no ROW; different (grown) init"}
            if info is None or not str(info.get("status", "")).startswith("available") or new is None:
                rows.append({**base, "status": "unavailable" + (" (historical predictions not found)" if info is None else "")
                             + (" (new run pending)" if new is None else "")})
                continue
            a = RR.load(info["predictions"])
            s1 = a["kind"] == "single"
            old = pd.DataFrame({"word": a["string"][s1].astype(str), "grid_hash": a["grid_hash"][s1],
                                "exact": a["exact"][s1]}).groupby(["word", "grid_hash"]).exact.mean()
            for op in "ABC":
                j = pd.concat([old.xs(op, level=0).rename("old"), new.xs(op, level=0).rename("new")], axis=1, join="inner")
                rows.append({**base, "op": op, "status": "paired by (op, grid hash)", "grids": len(j),
                             "noROW_mean": j.old.mean(), "ROW_mean": j.new.mean(), "improved": int((j.new > j.old).sum()),
                             "tied": int((j.new == j.old).sum()), "worsened": int((j.new < j.old).sum())})
    T_["comparison_noROW"] = pd.DataFrame(rows)
    R.table("Comparison 2 (read-only): new ROW vs historical no-ROW (notebook 16) at the same S / support",
            T_["comparison_noROW"], 60)
    R.md("**Interpretation limits.**\n"
         "* All three supplement conditions train some zero-I examples (k=S), unlike S5/train02/023/0234.\n"
         "* Changing S also changes sequence length, positions, layouts, filler count and compute: this is not a pure "
         "I-presence intervention.\n"
         "* S2/train02 trains only even operation counts, so the even-count ambiguity remains.\n"
         "* One seed: descriptive only.")
    aud = root / "audits" / f"seed_{seed}"
    for name in ("preflight", "initialization", "layout"):
        p = aud / f"{name}.csv"
        t = RR.read_csv_safe(p)
        if t is not None:
            R.table(f"Audit: {name}", t, 30)
    res = []
    for c, m in [(r.condition, r.model) for r in st.itertuples() if r.train == "done"]:
        d = root / c / f"seed_{seed}"
        ts = json.loads((d / "runs" / m / "train_summary.json").read_text()) if (d / "runs" / m / "train_summary.json").exists() else {}
        art = json.loads((d / "eval" / m / "ARTIFACT.json").read_text()) if (d / "eval" / m / "ARTIFACT.json").exists() else {}
        res.append({"condition": c, "model": m, "params": ts.get("unique_params"), "seq_len": 87 + RS.n_slots(c),
                    "train_min": (ts.get("train_seconds") or 0) / 60, "eval_min": (art.get("eval_seconds") or 0) / 60,
                    "best_step": ts.get("best_step"), "peak_gb": ts.get("peak_gpu_memory_gb_this_session"),
                    "hardware": ts.get("hardware"), "repo_commit": ts.get("repo_commit")})
    T_["resources"] = pd.DataFrame(res)
    R.table("Resources and provenance", T_["resources"], 10)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, t in T_.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    rp = write_review(R.items, out, "ROW S2/S3/S4 supplement",
                      f"# ROW S2/S3/S4 supplement — review snapshot\n\nBaked outputs; nothing to run. "
                      f"{int(st.complete.sum())}/6 jobs complete.", "AR_Row_Slots234_Review.ipynb")
    log(f"report -> {out}\nREVIEW FILE: {rp}")
    return rp
