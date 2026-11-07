"""Pipeline glue, audits, post-hoc analyses and figures for the identity count-extrapolation experiments (12/13).

All analyses read saved predictions through ``artifact_index.json`` (rebuilt by scanning per-run ARTIFACT.json).
Post-hoc analyses (transformation coverage, H ambiguity) never influence training, sampling or selection.
One seed: descriptive only; the five singleton placements of a grid are NOT independent replicates.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import idext as E
from . import idvar as V
from .names import DISPLAY_NAMES
from .plotstyle import _plt, save_fig
from .words import from_label, words_of_length

EXPS = list(E.IDEXT_EXPERIMENTS)


# ---------------------------------------------------------------------------
# Prepare / index / load
# ---------------------------------------------------------------------------

def prepare_idext(cfg, repo_dir=None, legacy_manifest: Optional[str] = None, idvar_manifest: Optional[str] = None,
                  log=print):
    from .data import load_or_build_manifests
    from .pipeline import Context
    from .utils import atomic_write_json
    m = load_or_build_manifests(cfg.manifest_dir, cfg.train.seed)
    val = E.make_val_set(m.val_grids[: cfg.n_val_grids], cfg.train.seed, cfg.experiment)
    subset = V.subset_by_grids(val, cfg.val_subset_grids)
    test = V.make_test_set(m.test_grids[: cfg.n_test_grids], cfg.train.seed, "identity")   # notebook-10 manifest
    assert len(test["x"]) == V.expected_test_records("identity", cfg.n_test_grids)
    assert set(np.unique(val["k"])) == set(E.train_ks(cfg.experiment))
    cmp = {}
    for name, path in (("legacy", legacy_manifest), ("identity_02345 (notebook 10)", idvar_manifest)):
        if path and Path(path).exists():
            h = json.loads(Path(path).read_text()).get("combined_hash")
            cmp[name] = "identical" if h == m.manifest_hash else "DIFFERENT"
        else:
            cmp[name] = "unavailable"
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json",
                      {**cfg.to_dict(), "train_ks": list(E.train_ks(cfg.experiment)), "manifest_hash": m.manifest_hash,
                       "manifest_comparison": cmp, "n_val_records": len(val["x"]), "n_test_records": len(test["x"]),
                       "test_manifest": "identical construction to notebook 10 (idvar.make_test_set, identity, same seed)"})
    log(f"[{cfg.experiment}] manifests {cfg.manifest_dir} (hash {m.manifest_hash[:12]}; {cmp}) | validation "
        f"{len(val['x'])} (k={list(E.train_ks(cfg.experiment))}) | test {len(test['x']):,}")
    return Context(cfg, m, val, subset, test, repo_dir)


def build_index(root: Path) -> dict:
    from .utils import atomic_write_json
    root = Path(root)
    idx = {}
    for f in sorted(root.glob("*/seed_*/eval/*/ARTIFACT.json")):
        a = json.loads(f.read_text())
        idx[f"{a['experiment']}/seed_{a['seed']}/{a['model']}"] = a
    atomic_write_json(root / "artifact_index.json", idx)
    return idx


def load_run(root: Path, experiment: str, model: str, seed: int = 42) -> Optional[Dict[str, np.ndarray]]:
    a = build_index(root).get(f"{experiment}/seed_{seed}/{model}")
    if not a or not a.get("complete"):
        return None
    with np.load(a["predictions"]) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def init_audit(seed: int = 42):
    import pandas as pd
    import torch
    from .idvar_reports import model_config
    from .models.registry import build_model
    rows = []
    for m in E.IDEXT_MODELS:
        sds = [{k: v.detach().clone() for k, v in build_model(m, model_config(m, "identity"), seed=seed).named_parameters()}
               for _ in EXPS]
        same = sds[0].keys() == sds[1].keys() and all(torch.equal(sds[0][k], sds[1][k]) for k in sds[0])
        rows.append({"model": m, "unique_params": sum(v.numel() for v in sds[0].values()),
                     "identical_initial_state_train02_vs_train023": bool(same),
                     "init_sha256": hashlib.sha256(b"".join(v.numpy().tobytes() for v in sds[0].values())).hexdigest()[:16]})
    return pd.DataFrame(rows)


def allocation_audit(B: int = 256, steps: int = 5000):
    import pandas as pd
    rows = []
    for e in EXPS:
        full = E.full_run_counts(e, B, steps)
        for k in range(6):
            rows.append({"experiment": e, "k": k, "status": "train" if k in full else "HELD OUT",
                         "records_update_0": E.per_update_counts(E.train_ks(e), B, 0).get(k, 0),
                         "records_full_run": full.get(k, 0), "logical_words": 3 ** k,
                         "five_slot_patterns": V.pattern_counts()[k]})
    df = pd.DataFrame(rows)
    for e in EXPS:
        assert df[df.experiment == e].records_full_run.sum() == B * steps
    return df


def serialization_audit():
    """AR spans for the fixed five-token identity serialization (derived from the layout, not hardcoded)."""
    import pandas as pd
    import torch
    from . import tokenization as T
    lay = T.ARLayout(5)
    assert (lay.length, lay.idx_output, lay.out_start, lay.out_start + 35, lay.idx_eos) == (82, 44, 45, 80, 81)
    rows = []
    for w, slots in (("", ()), ("A", (1,)), ("AB", (0, 2)), ("ACB", (1, 2, 4)), ("ABCA", (0, 1, 3, 4)), ("ABCAB", tuple(range(5)))):
        s5 = V.slots5_for(w, slots)
        toks = T.op_slots_to_tokens(torch.as_tensor(s5[None]))[0].tolist()
        assert T.PAD_ID not in toks and len(toks) == 5
        rows.append({"word": w or "EMPTY", "k": len(w), "op_tokens": " ".join("I" if t == T.IDENTITY_ID else "ABC"[t - 10]
                                                                              for t in toks),
                     "teacher_forced_tokens": lay.length, "output_index": lay.idx_output,
                     "answer_span": f"{lay.out_start}..{lay.out_start + 35}", "eos_index": lay.idx_eos})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Post-hoc analyses
# ---------------------------------------------------------------------------

def transformation_coverage():
    """Per experiment: training-transform set (permutations of 36 indexed cells) and per-test-word membership."""
    import pandas as pd
    from .diagnosis import group as G
    cat = G.Catalog()
    rows = []
    for e in EXPS:
        S = {cat.word_id(w) for w in E.training_words(e)}
        for k in range(6):
            for w in words_of_length(k):
                gid = cat.word_id(w)
                rows.append({"experiment": e, "k": k, "string": w or "EMPTY", "perm_id": gid, "perm_word": cat.label(gid),
                             "word_in_training": k in E.train_ks(e), "transform_in_training": gid in S,
                             "n_training_transforms": len(S)})
    df = pd.DataFrame(rows)
    summ = df.groupby(["experiment", "k"]).agg(seen_transform_words=("transform_in_training", "sum"),
                                               words=("string", "size")).reset_index()
    exp = {"idext_train02": [1, 0, 9, 0, 38, 0], "idext_train023": [1, 3, 9, 27, 38, 162]}
    for e, v in exp.items():
        got = summ[summ.experiment == e].sort_values("k").seen_transform_words.tolist()
        assert got == v, (e, got)
    assert df[df.experiment == "idext_train02"].n_training_transforms.iloc[0] == 9
    assert df[df.experiment == "idext_train023"].n_training_transforms.iloc[0] == 30
    return df, summ


def h_ambiguity_check():
    from .diagnosis import group as G
    H = G.Z_SRC
    assert not np.array_equal(H, G.IDENTITY) and np.array_equal(G.compose(H, H), G.IDENTITY)
    for g in G.GENERATORS.values():
        assert np.array_equal(G.compose(H, g), G.compose(g, H))
    tilde = {ch: G.compose(G.GENERATORS[ch], H) for ch in "ABC"}
    n = 0
    for k in range(6):
        for w in words_of_length(k):
            p = G.IDENTITY.copy()
            for ch in w:
                p = G.compose(p, tilde[ch])
            assert np.array_equal(p, G.word_perm(w)) == (k % 2 == 0)
            n += 1
    return {"H": "shift 3 rows and 3 columns", "involution": True, "commutes_with_A_B_C": True,
            "tilde_equals_truth_iff_k_even": True, "words_checked": n}


def run_tables(a: Dict[str, np.ndarray], experiment: str, model: str, cov):
    """Per-k (with transform present/absent split), per-word, singleton tables, odd-k H agreement."""
    import pandas as pd
    from .diagnosis import group as G
    n = len(a["pred"])
    P, Y, X = (a[k].reshape(n, 36).astype(np.int16) for k in ("pred", "target", "input"))
    exact = a["exact"].astype(bool)
    cell = (P == Y).mean(1)
    tks = E.train_ks(experiment)
    c = cov[cov.experiment == experiment].set_index("string")
    present = c.loc[a["string"].astype(str), "transform_in_training"].values
    df = pd.DataFrame({"k": a["k"], "string": a["string"].astype(str), "position": a["position"], "kind": a["kind"],
                       "grid_id": a["grid_id"], "exact": exact, "cell": cell, "invalid": ~a["valid"], "present": present})
    by_k = df.groupby("k").agg(exact_correct=("exact", "sum"), n=("exact", "size"), exact_acc=("exact", "mean"),
                               cell_acc=("cell", "mean"), invalid_rate=("invalid", "mean")).reset_index()
    by_k["status"] = np.where(by_k.k.isin(tks), "seen support", "HELD OUT")
    split = df.groupby(["k", "present"]).agg(exact_correct=("exact", "sum"), n=("exact", "size")).reset_index()
    split["exact_acc"] = split.exact_correct / split.n
    rows = []
    for k in range(6):
        for pres in (True, False):
            s = split[(split.k == k) & (split.present == pres)]
            rows.append({"k": k, "transform_in_training": pres,
                         "exact_correct": int(s.exact_correct.sum()) if len(s) else 0, "n": int(s.n.sum()) if len(s) else 0,
                         "exact_acc": float(s.exact_acc.iloc[0]) if len(s) else np.nan})
    split = pd.DataFrame(rows)
    by_word = df[df.kind == "seen"].groupby("string", sort=False).agg(
        k=("k", "first"), exact_correct=("exact", "sum"), n=("exact", "size"), exact_acc=("exact", "mean"),
        cell_acc=("cell", "mean")).reset_index()
    by_word = by_word.merge(c[["perm_id", "perm_word", "transform_in_training"]], left_on="string", right_index=True)
    single = df[df.kind == "single"].groupby(["string", "position"]).agg(
        exact_correct=("exact", "sum"), n=("exact", "size"), exact_acc=("exact", "mean"), cell_acc=("cell", "mean")).reset_index()
    # odd-k: agreement with truth vs with H(truth), with overlap counts
    HY = Y[:, G.Z_SRC]
    h_exact = (P == HY).all(1) & a["valid"].astype(bool)
    overlap = (Y == HY).all(1)
    hrows = []
    for k in (1, 3, 5):
        m = df.k.values == k
        hrows.append({"k": k, "n": int(m.sum()), "status": "seen support" if k in tks else "HELD OUT",
                      "matches_true_target": int(exact[m].sum()), "matches_H_of_target": int(h_exact[m].sum()),
                      "targets_where_H_equals_truth_on_this_grid": int(overlap[m].sum()),
                      "matches_both (ambiguous)": int((exact & h_exact)[m].sum())})
    hdf = pd.DataFrame(hrows)
    for t in (by_k, split, by_word, single, hdf):
        t.insert(0, "model", model)
        t.insert(0, "experiment", experiment)
    return {"by_k": by_k, "transform_split": split, "by_word": by_word, "singletons": single, "h_agreement": hdf}


def singleton_consistency(a, experiment, model):
    import pandas as pd
    rows = []
    s = a["kind"] == "single"
    for op in "ABC":
        P = []
        for p in range(1, 6):
            m = s & (a["string"] == op) & (a["position"] == p)
            o = np.argsort(a["grid_id"][m])
            P.append(a["pred"][m][o].reshape(-1, 36))
        P = np.stack(P)
        same = (P == P[0:1]).all(axis=(0, 2))
        rows.append({"experiment": experiment, "model": model, "op": op, "n_grids": P.shape[1],
                     "grids_identical_across_5_placements": int(same.sum()), "rate": float(same.mean())})
    return pd.DataFrame(rows)


def singleton_error_diagnostics(a, experiment, model, split_seed=20251106):
    """Reuse the verified 288-transform singleton analysis per placement (XC/CX/identity/pairs/maps)."""
    import pandas as pd
    from .diagnosis import group as G
    from .diagnosis.analysis import analyze_run, diagnostic_split, hypothesis_population_summary
    from .diagnosis.artifacts import Run
    from .idvar_reports import singles_spec
    spec = singles_spec()
    spec.train_labels = lambda e, ex=experiment: E.training_labels(ex)
    cat = G.Catalog()
    split = diagnostic_split(np.unique(a["grid_id"]), split_seed)
    ps, hs = [], []
    for p in range(1, 6):
        m = (a["kind"] == "single") & (a["position"] == p)
        r = Run(f"identity@slot{p}", model, "original", {}, a["input"][m].reshape(-1, 36).astype(np.int16),
                a["target"][m].reshape(-1, 36).astype(np.int16), a["pred"][m].reshape(-1, 36).astype(np.int16),
                a["grid_id"][m].astype(np.int64), a["string"][m].astype(str), {}, {"file_sha256": {}})
        res = analyze_run(r, cat, spec, split)
        x = res["per_string"].copy(); x.insert(0, "position", p); ps.append(x)
        h = hypothesis_population_summary(res["hypothesis_types"], res["per_string"], spec); h.insert(0, "position", p); hs.append(h)
    out = pd.concat(ps, ignore_index=True), pd.concat(hs, ignore_index=True)
    for t in out:
        t["experiment"] = experiment
        t["model"] = model
    return out


def analyse_all(root: Path, seed: int = 42, log=print) -> dict:
    """Run every analysis on the available runs; missing runs are reported as 'not yet available'."""
    import pandas as pd
    root = Path(root)
    cov, cov_summ = transformation_coverage()
    acc = {k: [] for k in ("by_k", "transform_split", "by_word", "singletons", "h_agreement", "consistency",
                           "singleton_per_string", "singleton_hypotheses")}
    status, runs = [], {}
    for e in EXPS:
        for m in E.IDEXT_MODELS:
            a = load_run(root, e, m, seed)
            if a is None:
                status.append({"experiment": e, "model": m, "status": "not yet available"})
                continue
            status.append({"experiment": e, "model": m, "status": "analysed"})
            runs[(e, m)] = a
            t = run_tables(a, e, m, cov)
            for k, v in t.items():
                acc[k].append(v)
            acc["consistency"].append(singleton_consistency(a, e, m))
            ps, hs = singleton_error_diagnostics(a, e, m)
            acc["singleton_per_string"].append(ps)
            acc["singleton_hypotheses"].append(hs)
    res = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in acc.items()}
    res["coverage"], res["coverage_summary"], res["status"] = cov, cov_summ, pd.DataFrame(status)
    res["paired"] = paired_transitions(runs)
    out = root / "comparisons" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    log("Run status: " + "; ".join(f"{r.experiment}/{r.model}: {r.status}" for r in res["status"].itertuples()))
    res["runs"] = runs
    res["out"] = out
    return res


def paired_transitions(runs):
    """Same model, identical test inputs: train02 -> train023 correctness transitions per k."""
    import pandas as pd
    rows = []
    for m in E.IDEXT_MODELS:
        if ("idext_train02", m) not in runs or ("idext_train023", m) not in runs:
            rows.append({"model": m, "k": "all", "status": "not yet available"})
            continue
        a, b = runs[("idext_train02", m)], runs[("idext_train023", m)]
        ka = pd.DataFrame({"s": a["string"], "g": a["grid_id"], "p": a["position"], "h": a["grid_hash"], "ea": a["exact"], "k": a["k"]})
        kb = pd.DataFrame({"s": b["string"], "g": b["grid_id"], "p": b["position"], "h": b["grid_hash"], "eb": b["exact"]})
        j = ka.merge(kb, on=["s", "g", "p", "h"], validate="one_to_one")
        assert len(j) == len(ka) == len(kb)
        for k, g in j.groupby("k"):
            rows.append({"model": m, "k": int(k),
                         "status_train02": "seen" if k in E.train_ks("idext_train02") else "HELD OUT",
                         "status_train023": "seen" if k in E.train_ks("idext_train023") else "HELD OUT",
                         "n": len(g), "train02_exact": int(g.ea.sum()), "train023_exact": int(g.eb.sum()),
                         "both_correct": int((g.ea & g.eb).sum()), "only_train02_correct": int((g.ea & ~g.eb).sum()),
                         "only_train023_correct": int((~g.ea & g.eb).sum()), "both_wrong": int((~g.ea & ~g.eb).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_main(res, out: Path):
    plt = _plt()
    t = res["by_k"]
    if not len(t):
        return []
    paths = []
    for metric, lab in (("exact_acc", "exact-grid accuracy"), ("cell_acc", "cell accuracy")):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), squeeze=False)
        for ax, m in zip(axes[0], E.IDEXT_MODELS):
            for e, col in (("idext_train02", "#2a78d6"), ("idext_train023", "#eb6834")):
                s = t[(t.model == m) & (t.experiment == e)].sort_values("k")
                if not len(s):
                    continue
                ax.plot(s.k, s[metric], color=col, lw=1.6, label=f"{E.IDEXT_DIRS[e]} (train k={list(E.train_ks(e))})")
                seen = s.k.isin(E.train_ks(e))
                ax.scatter(s.k[seen], s[metric][seen], color=col, s=55, zorder=3)
                ax.scatter(s.k[~seen], s[metric][~seen], facecolors="white", edgecolors=col, s=55, zorder=3, lw=1.6)
            ax.set(xticks=range(6), ylim=(-0.02, 1.05), xlabel="k = number of A/B/C tokens (always 5 visible op tokens)",
                   ylabel=lab, title=f"{DISPLAY_NAMES[m]} (seed 42, best ckpt)\nfilled = seen support, open = HELD OUT")
            ax.legend(fontsize=8)
        fig.tight_layout()
        paths += save_fig(fig, out, f"main_{metric}_by_k", t)
    return paths


def plot_singletons(res, out: Path):
    plt = _plt()
    s = res["singletons"]
    if not len(s):
        return []
    runs = [(e, m) for e in EXPS for m in E.IDEXT_MODELS if len(s[(s.experiment == e) & (s.model == m)])]
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.4), squeeze=False)
    for ax, (e, m) in zip(axes[0], runs):
        r = s[(s.experiment == e) & (s.model == m)]
        mat = np.array([[r[(r.string == op) & (r.position == p)].exact_correct.iloc[0] for p in range(1, 6)] for op in "ABC"])
        ax.grid(False)
        ax.imshow(mat / 1000, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for jx in range(5):
                ax.text(jx, i, str(mat[i, jx]), ha="center", va="center", fontsize=8, color="white" if mat[i, jx] > 600 else "#0b0b0b")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"slot {p}" for p in range(1, 6)], fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(list("ABC"))
        ax.set_title(f"{E.IDEXT_DIRS[e]} / {DISPLAY_NAMES[m]}", fontsize=9)
    fig.suptitle("Singletons (HELD OUT in both experiments): exact successes of 1000 per operation x placement")
    fig.tight_layout()
    return save_fig(fig, out, "singletons_op_by_position", s)


def plot_transform_split_and_h(res, out: Path):
    plt = _plt()
    paths = []
    sp = res["transform_split"]
    if len(sp):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), squeeze=False)
        for ax, m in zip(axes[0], E.IDEXT_MODELS):
            for e, col in (("idext_train02", "#2a78d6"), ("idext_train023", "#eb6834")):
                for pres, ls in ((True, "-"), (False, "--")):
                    s = sp[(sp.model == m) & (sp.experiment == e) & (sp.transform_in_training == pres)].sort_values("k")
                    ax.plot(s.k, s.exact_acc, color=col, ls=ls, marker="o",
                            label=f"{E.IDEXT_DIRS[e]}: transform {'IN' if pres else 'NOT in'} training")
            ax.set(xticks=range(6), ylim=(-0.02, 1.05), xlabel="k", ylabel="exact accuracy",
                   title=f"{DISPLAY_NAMES[m]}: by training-transformation membership (N/A strata omitted)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        paths += save_fig(fig, out, "transform_membership_split", sp)
    h = res["h_agreement"]
    if len(h):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.axis("off")
        tbl = h[["experiment", "model", "k", "status", "n", "matches_true_target", "matches_H_of_target",
                 "targets_where_H_equals_truth_on_this_grid"]]
        ax.table(cellText=tbl.values, colLabels=[c.replace("_", " ") for c in tbl.columns], loc="center", fontsize=7)
        ax.set_title("POST-HOC odd-k diagnostic: exact agreement with true target vs H(target) (not a training signal)")
        paths += save_fig(fig, out, "odd_k_H_agreement", h)
    return paths


def status_table(root: Path, seed: int = 42):
    import pandas as pd
    idx = build_index(root)
    rows = []
    for e in EXPS:
        for m in E.IDEXT_MODELS:
            d = Path(root) / E.IDEXT_DIRS[e] / f"seed_{seed}"
            a = idx.get(f"{e}/seed_{seed}/{m}", {})
            rows.append({"experiment": e, "model": m, "trained": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(),
                         "evaluated": bool(a.get("complete")), "records": a.get("n_records"),
                         "selected_step": a.get("selected_step")})
    return pd.DataFrame(rows)
