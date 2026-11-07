"""Per-run behavioral diagnostics on frozen, saved predictions (torch-free; no model is loaded).

All measures compare a model's stored prediction ``Yhat`` for input ``X`` and instruction ``w`` with
post-hoc candidate explanations. Agreement is counted over all 36 cells; invalid cells (-1) never
match. "DIAG" measures fit an explanatory rule to MODEL PREDICTIONS on the FIT half of the base grids and
report its fidelity on the disjoint DIAG half; they describe frozen behaviour and are not a benchmark.

Hypothesis accounting (v2): for every (word, rule) the table records whether the rule is APPLICABLE,
whether it has at least one candidate transformation that differs from the truth (ELIGIBLE as an error
explanation), and the number of UNIQUE records matched by ANY non-truth candidate of that rule (union;
each record counted once). Population summaries then report eligible counts, matched counts,
conditional rates (matched / eligible) and unconditional coverage (matched / full declared population).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..words import from_label
from . import group as G
from .artifacts import Run
from .spec import LEGACY, DiagSpec

ANALYSIS_VERSION = "generalization-failure-diagnostics-v2"


def diagnostic_split(grid_ids: Sequence[int], seed: int) -> Dict[str, np.ndarray]:
    """Deterministic disjoint split of base-grid IDs into equal FIT and DIAG halves."""
    ids = np.unique(np.asarray(grid_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    half = len(ids) // 2
    fit, diag = np.sort(ids[perm[:half]]), np.sort(ids[perm[half:]])
    assert not (set(fit.tolist()) & set(diag.tolist()))
    return {"fit": fit, "diag": diag}


def transformed_inputs(Xu: np.ndarray, cat: G.Catalog) -> np.ndarray:
    """[n_grids, 288, 36] int8: every group element applied to every base grid (computed once per manifest)."""
    return np.asarray(Xu, dtype=np.int8)[:, cat.perms]


def agreement_matrix(P: np.ndarray, grid_index: np.ndarray, T: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """agree[i, g] = #cells where prediction i equals group element g applied to its input (uint8 0..36)."""
    P = np.asarray(P, dtype=np.int8)
    out = np.empty((len(P), T.shape[1]), dtype=np.uint8)
    for s in range(0, len(P), chunk):
        sl = slice(s, min(len(P), s + chunk))
        out[sl] = (T[grid_index[sl]] == P[sl, None, :]).sum(-1, dtype=np.uint8)
    return out


def _unique_grids(grid_id: np.ndarray, X: np.ndarray):
    uid, first, inv = np.unique(grid_id, return_index=True, return_inverse=True)
    return uid, X[first], inv


def _label_word(w):
    return "N/A (inapplicable)" if w is None else (w or "I")


def analyze_run(run: Run, cat: G.Catalog, spec: DiagSpec, split: Dict[str, np.ndarray], chunk: int = 2000,
                T_cache: Optional[dict] = None) -> dict:
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    exp = run.experiment
    if exp not in spec.experiments:
        raise ValueError(f"experiment {exp!r} is not part of diagnostic spec {spec.name!r}")
    train_labels = list(spec.train_labels(exp))
    pair_words = list(spec.pair_words(exp))
    gfn = lambda s: spec.group_fn(exp, s)  # noqa: E731

    uid, Xu, ginv = _unique_grids(run.grid_id, run.X)
    key = (uid.tobytes(), Xu.astype(np.int8).tobytes())
    if T_cache is not None and key in T_cache:
        T = T_cache[key]
    else:
        T = transformed_inputs(Xu, cat)
        if T_cache is not None:
            T_cache.clear()
            T_cache[key] = T
    agree = agreement_matrix(run.P, ginv, T, chunk)
    n = len(run.P)
    valid = (run.P >= 0).all(1)
    cell_correct = (run.P == run.Y).sum(1)
    exact = valid & (cell_correct == 36)
    words_present = [s for s in spec.words if s in set(run.string)] or sorted(set(run.string))
    truth_of = {s: cat.word_id(from_label(s)) for s in words_present}
    alt_of = {s: cat.alt_generator_id(from_label(s)) for s in words_present}
    rec_truth = np.array([truth_of[s] for s in run.string])
    rec_alt = np.array([alt_of[s] for s in run.string])
    train_mask = np.zeros(cat.size, bool)
    train_mask[[cat.word_id(from_label(s)) for s in train_labels]] = True
    pair_mask = np.zeros(cat.size, bool)
    pair_mask[[cat.word_id(w) for w in pair_words]] = True

    full = agree == 36
    n_exact = full.sum(1)
    idx = np.arange(n)
    correct_in = full[idx, rec_truth]
    has_train = (full & train_mask[None]).any(1)
    has_pair = (full & pair_mask[None]).any(1)
    best = agree.max(1)
    n_best = (agree == best[:, None]).sum(1)
    second = np.where(agree < best[:, None], agree.astype(np.int16), -1).max(1)
    off = agree.astype(np.int16).copy()
    off[idx, rec_truth] = -1
    best_off = off.max(1)
    unique_id = np.where(n_exact == 1, full.argmax(1), -1)
    alt_cells = agree[idx, rec_alt]
    hp = valid & ((run.P[:, :, None] == np.arange(10)).sum(1) == (run.X[:, :, None] == np.arange(10)).sum(1)).all(1)
    category = np.where(correct_in, "correct", np.where(n_exact == 1, "wrong_unique_legal",
                                                         np.where(n_exact > 1, "wrong_ambiguous_legal", "no_legal_match")))
    fit_set, diag_set = set(split["fit"].tolist()), set(split["diag"].tolist())
    rec_split = np.array(["fit" if g in fit_set else "diag" if g in diag_set else "unassigned" for g in run.grid_id])
    group_of = {s: gfn(s) for s in words_present}
    groups = np.array([group_of[s] for s in run.string])
    pe = pd.DataFrame({
        "experiment": exp, "model": run.model, "grid_id": run.grid_id, "string": run.string,
        "length": [len(from_label(s)) for s in run.string], "group": groups, "split": rec_split, "valid": valid,
        "cell_correct": cell_correct, "exact": exact, "n_exact_group_matches": n_exact,
        "correct_in_exact_set": correct_in, "exact_set_includes_training_transform": has_train,
        "exact_set_includes_original_pair_transform": has_pair,
        "only_incorrect_exact_matches": (n_exact > 0) & ~correct_in, "unique_exact_match_id": unique_id,
        "best_agreement": best, "n_best_ties": n_best, "second_best_agreement": second,
        "best_offtarget_agreement": best_off, "alt_generator_cells": alt_cells, "alt_generator_exact": alt_cells == 36,
        "digit_histogram_preserved": hp, "category": category})
    res = {"per_example": pe}

    by_word = {s: np.nonzero(run.string == s)[0] for s in words_present}
    rows, conf_rows = [], []
    for s in words_present:
        m = by_word[s]
        tid = truth_of[s]
        k = len(m)
        cats = Counter(category[m])
        wrong_ids = unique_id[m][category[m] == "wrong_unique_legal"]
        wc = Counter(wrong_ids.tolist())
        dom, dom_n = (wc.most_common(1)[0] if wc else (-1, 0))
        rows.append({
            "experiment": exp, "model": run.model, "string": s, "group": group_of[s], "length": len(from_label(s)),
            "n": k, "truth_id": tid, "seen_in_training": s in set(train_labels),
            "transformation_in_training": bool(train_mask[tid]), "transformation_is_original_pair": bool(pair_mask[tid]),
            "exact_correct": int(exact[m].sum()), "exact_acc": exact[m].mean(), "cell_acc": cell_correct[m].mean() / 36,
            "invalid_rate": (~valid[m]).mean(), "any_exact_group_match_count": int((n_exact[m] > 0).sum()),
            "any_exact_group_match_rate": (n_exact[m] > 0).mean(),
            "wrong_unique_legal_rate": cats["wrong_unique_legal"] / k,
            "wrong_ambiguous_legal_rate": cats["wrong_ambiguous_legal"] / k,
            "no_legal_match_rate": cats["no_legal_match"] / k,
            "exact_set_includes_training_transform_rate": has_train[m].mean(),
            "exact_set_includes_original_pair_rate": has_pair[m].mean(),
            "wrong_matching_original_pair_count": int((has_pair[m] & ~exact[m]).sum()),
            "mean_best_group_cell_agreement": best[m].mean() / 36,
            "mean_best_offtarget_cell_agreement": best_off[m].mean() / 36, "mean_best_ties": n_best[m].mean(),
            "dominant_wrong_unique_id": dom, "dominant_wrong_unique_word": cat.label(dom) if dom >= 0 else "",
            "dominant_wrong_unique_share": dom_n / k, "dominant_is_training_transform": bool(dom >= 0 and train_mask[dom]),
            "dominant_is_original_pair": bool(dom >= 0 and pair_mask[dom]),
            "alt_generator_id": alt_of[s], "alt_generator_exact_rate": (alt_cells[m] == 36).mean(),
            "alt_generator_cell_agreement": alt_cells[m].mean() / 36, "digit_histogram_preserved_rate": hp[m].mean()})
        for c in ("correct", "wrong_ambiguous_legal", "no_legal_match"):
            conf_rows.append({"string": s, "bin": c, "group_id": -1, "word": "", "count": cats[c], "n": k})
        for gid, cnt in wc.most_common():
            conf_rows.append({"string": s, "bin": "wrong_unique_legal", "group_id": gid, "word": cat.label(gid),
                              "is_training_transform": bool(train_mask[gid]), "is_original_pair": bool(pair_mask[gid]),
                              "count": cnt, "n": k})
    ps = pd.DataFrame(rows)
    conf = pd.DataFrame(conf_rows)
    conf.insert(0, "model", run.model)
    conf.insert(0, "experiment", exp)
    res["confusion"] = conf

    # ---- hypotheses: per candidate (long) and per rule with union counting ----------------------
    hyp_rows, type_rows = [], []
    for s in words_present:
        m = by_word[s]
        w = from_label(s)
        tid = truth_of[s]
        hyps = G.hypotheses_for(w, pair_words)
        for h in hyps:
            h["group_id"] = cat.word_id(h["executed_word"]) if h["executed_word"] is not None else -1
        by_gid: Dict[int, List[str]] = {}
        for h in hyps:
            if h["group_id"] >= 0:
                by_gid.setdefault(h["group_id"], []).append(f"{h['hypothesis_type']}:{_label_word(h['executed_word'])}")
        for h in hyps:
            gid = h["group_id"]
            app = gid >= 0
            c = full[m, gid] if app else np.zeros(len(m), bool)
            me = f"{h['hypothesis_type']}:{_label_word(h['executed_word'])}"
            hyp_rows.append({"string": s, "group": group_of[s], "hypothesis_type": h["hypothesis_type"],
                             "executed_word": _label_word(h["executed_word"]), "applicable": app, "group_id": gid,
                             "group_word": cat.label(gid) if app else "", "truth_equivalent": bool(app and gid == tid),
                             "aliases": "; ".join(a for a in by_gid.get(gid, []) if a != me),
                             "compatible_count": int(c.sum()), "n": len(m), "compatible_rate": c.mean() if len(m) else np.nan,
                             "candidate_cell_agreement_with_model": (agree[m, gid].mean() / 36) if app else np.nan,
                             "candidate_cell_agreement_with_target": (
                                 (T[ginv[m], gid] == run.Y[m]).mean() if app else np.nan)})
        for t in G.HYPOTHESIS_TYPES:
            cands = [h for h in hyps if h["hypothesis_type"] == t]
            app = [h for h in cands if h["group_id"] >= 0]
            nontruth = sorted({h["group_id"] for h in app if h["group_id"] != tid})
            truth_eq = any(h["group_id"] == tid for h in app)
            union = np.zeros(len(m), bool)
            for gid in nontruth:
                union |= full[m, gid]
            type_rows.append({"string": s, "group": group_of[s], "length": len(w), "hypothesis_type": t,
                              "applicable": bool(app), "has_truth_equivalent_candidate": truth_eq,
                              "n_distinct_non_truth_transforms": len(nontruth), "eligible": bool(nontruth),
                              "n": len(m), "matched_unique": int(union.sum()),
                              "compatible_rate_any": (union.mean() if nontruth else np.nan)})
    for name, rows_ in (("hypotheses", hyp_rows), ("hypothesis_types", type_rows)):
        df = pd.DataFrame(rows_)
        df.insert(0, "model", run.model)
        df.insert(0, "experiment", exp)
        res[name] = df

    # ---- stable rules + source maps (fit on FIT grids from MODEL predictions; score on DIAG) ------
    stable_rows, map_rows, maps = [], [], {}
    for s in words_present:
        mw = by_word[s]
        mfit = mw[rec_split[mw] == "fit"]
        mdiag = mw[rec_split[mw] == "diag"]
        if not len(mfit) or not len(mdiag):
            continue
        tid = truth_of[s]
        sums = agree[mfit].sum(0, dtype=np.int64)
        ties = np.nonzero(sums == sums.max())[0]
        sel = int(ties[0])
        stable_rows.append({
            "string": s, "group": group_of[s], "n_fit": len(mfit), "n_diag": len(mdiag), "selected_id": sel,
            "selected_word": cat.label(sel), "n_tied_rules": len(ties), "tied_words": " ".join(cat.label(t) for t in ties[:6]),
            "selected_is_truth": sel == tid, "selected_is_alt_generator": sel == alt_of[s],
            "selected_is_training_transform": bool(train_mask[sel]), "selected_is_original_pair": bool(pair_mask[sel]),
            "positions_shared_with_truth": int((cat.perms[sel] == cat.perms[tid]).sum()),
            "expected_cell_acc_vs_target_if_rule": G.expected_wrong_perm_cell_accuracy(cat.perms[sel], cat.perms[tid]),
            "fit_cell_fidelity": agree[mfit, sel].mean() / 36, "fit_exact_fidelity": (agree[mfit, sel] == 36).mean(),
            "diag_cell_fidelity": agree[mdiag, sel].mean() / 36, "diag_exact_fidelity": (agree[mdiag, sel] == 36).mean(),
            "diag_model_exact_acc": exact[mdiag].mean()})
        Pf, Xf = run.P[mfit], run.X[mfit]
        M = (Pf[:, :, None] == Xf[:, None, :]).mean(0)
        srt = np.sort(M, axis=1)
        argmap = M.argmax(1)
        row_margin = srt[:, -1] - srt[:, -2]
        r_, c_ = linear_sum_assignment(M, maximize=True)
        hung = np.empty(36, np.int64)
        hung[r_] = c_
        Pd, Xd = run.P[mdiag], run.X[mdiag]
        tperm = cat.perms[tid]
        for kind, mp in (("argmax_source_map", argmap), ("bijective_permutation", hung)):
            fit_match = Pf == Xf[:, mp]
            diag_match = Pd == Xd[:, mp]
            bij = len(set(mp.tolist())) == 36
            gid = cat.id_of(mp) if bij else -1
            map_rows.append({
                "string": s, "group": group_of[s], "map_kind": kind, "is_bijection": bij, "group_id": gid,
                "group_word": cat.label(gid), "equals_truth": bool(np.array_equal(mp, tperm)),
                "positions_agreeing_with_truth": int((mp == tperm).sum()),
                "fit_cell_fidelity": fit_match.mean(), "fit_exact_fidelity": fit_match.all(1).mean(),
                "diag_cell_fidelity": diag_match.mean(), "diag_exact_fidelity": diag_match.all(1).mean(),
                "min_position_diag_fidelity": diag_match.mean(0).min(),
                "mean_row_margin": float(row_margin.mean()), "tied_rows": int((row_margin <= 1e-12).sum()),
                "weak_or_tie_driven": bool(diag_match.mean() < 0.9 or (row_margin <= 1e-12).any())})
        maps[s] = {"M": M.astype(np.float32), "argmax_map": argmap, "bijective_map": hung,
                   "diag_position_fidelity_bijective": (Pd == Xd[:, hung]).mean(0).astype(np.float32)}
    for name, rows_ in (("stable_rules", stable_rows), ("source_maps", map_rows)):
        df = pd.DataFrame(rows_)
        if len(df):
            df.insert(0, "model", run.model)
            df.insert(0, "experiment", exp)
        res[name] = df
    res["maps"] = maps

    # ---- composition of fitted single-operation maps (behavioural consistency) ----------------
    comp_rows = []
    sm = res["source_maps"]
    if all(c in maps for c in "ABC") and len(sm):
        for kind, key_ in (("bijective_permutation", "bijective_map"), ("argmax_source_map", "argmax_map")):
            single_fid = {c: float(sm[(sm.string == c) & (sm.map_kind == kind)]["diag_cell_fidelity"].iloc[0]) for c in "ABC"}
            for s in words_present:
                w = from_label(s)
                if len(w) < 2:
                    continue
                md = by_word[s][rec_split[by_word[s]] == "diag"]
                if not len(md):
                    continue
                comp = maps[w[0]][key_]
                for ch in w[1:]:
                    comp = G.compose(comp, maps[ch][key_])
                Yc = run.X[md][:, comp]
                minf = min(single_fid[c] for c in set(w))
                comp_rows.append({
                    "string": s, "group": group_of[s], "length": len(w), "map_kind": kind, "n_diag": len(md),
                    "composed_group_word": cat.label(cat.id_of(comp)) if len(set(comp.tolist())) == 36 else "not-in-group",
                    "composed_equals_truth": bool(np.array_equal(comp, cat.perms[truth_of[s]])),
                    "agreement_with_model_cells": (Yc == run.P[md]).mean(),
                    "agreement_with_model_exact": (Yc == run.P[md]).all(1).mean(),
                    "agreement_with_target_cells": (Yc == run.Y[md]).mean(),
                    "agreement_with_target_exact": (Yc == run.Y[md]).all(1).mean(),
                    "min_single_map_diag_fidelity": minf,
                    "interpretation": "single maps stable (diag cell fidelity >= 0.9)" if minf >= 0.9
                    else "WEAK: single-operation behaviour has no stable map"})
    df = pd.DataFrame(comp_rows)
    if len(df):
        df.insert(0, "model", run.model)
        df.insert(0, "experiment", exp)
    res["composition"] = df

    merged = ps.copy()
    if len(res["stable_rules"]):
        st = res["stable_rules"].drop(columns=["experiment", "model", "group"]).add_prefix("stable_rule_").rename(
            columns={"stable_rule_string": "string"})
        merged = merged.merge(st, on="string", how="left")
    if len(sm):
        bij = sm[sm.map_kind == "bijective_permutation"][
            ["string", "diag_cell_fidelity", "diag_exact_fidelity", "is_bijection", "group_word", "equals_truth"]]
        bij = bij.add_prefix("perm_").rename(columns={"perm_string": "string"})
        arg = sm[sm.map_kind == "argmax_source_map"][["string", "diag_cell_fidelity"]].rename(
            columns={"diag_cell_fidelity": "argmax_map_diag_cell_fidelity"})
        merged = merged.merge(bij, on="string", how="left").merge(arg, on="string", how="left")
    for c in ("stable_rule_diag_exact_fidelity", "stable_rule_diag_cell_fidelity", "stable_rule_selected_word",
              "stable_rule_selected_is_training_transform", "stable_rule_selected_is_original_pair",
              "stable_rule_selected_is_alt_generator", "perm_diag_cell_fidelity", "argmax_map_diag_cell_fidelity"):
        if c not in merged:
            merged[c] = np.nan
    res["per_string"] = merged
    slim = ["experiment", "model", "grid_id", "string", "length", "group", "split", "exact", "cell_correct",
            "n_exact_group_matches", "unique_exact_match_id", "alt_generator_exact", "best_agreement", "category"]
    res["per_example_slim"] = pe[slim]
    return res


def hypothesis_population_summary(hyp_types, per_string, spec: DiagSpec):
    """For each experiment/model/declared population/rule: population size, eligible records, UNIQUE matched
    records (union within a rule), conditional rate (matched / eligible) and unconditional coverage
    (matched / population). Also reports the any-legal-match count on the same population as an upper bound
    for unconditional coverage of these legal-transform hypotheses."""
    import pandas as pd
    if not len(hyp_types):
        return pd.DataFrame()
    rows = []
    for (e, mdl), ht in hyp_types.groupby(["experiment", "model"], sort=False):
        ps = per_string[(per_string.experiment == e) & (per_string.model == mdl)]
        for pop, groups in spec.populations.items():
            ps_pop = ps[ps.group.isin(groups)]
            pop_n = int(ps_pop["n"].sum())
            any_legal = int(ps_pop["any_exact_group_match_count"].sum()) if len(ps_pop) else 0
            wrong = int((ps_pop["n"] - ps_pop["exact_correct"]).sum()) if len(ps_pop) else 0
            for t in G.HYPOTHESIS_TYPES:
                h = ht[(ht.hypothesis_type == t) & ht.group.isin(groups)]
                elig = h[h.eligible]
                e_n, m_n = int(elig["n"].sum()), int(elig["matched_unique"].sum())
                rows.append({"experiment": e, "model": mdl, "population": pop, "hypothesis_type": t,
                             "population_records": pop_n, "population_wrong_records": wrong,
                             "population_any_legal_match_records": any_legal,
                             "applicable_records": int(h[h.applicable]["n"].sum()),
                             "eligible_records": e_n, "matched_unique_records": m_n,
                             "conditional_rate (matched/eligible)": (m_n / e_n) if e_n else np.nan,
                             "unconditional_coverage (matched/population)": (m_n / pop_n) if pop_n else np.nan,
                             "truth_equivalent_strings": int(h["has_truth_equivalent_candidate"].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration controls (once per base-grid manifest)
# ---------------------------------------------------------------------------

def calibration_controls(Xu: np.ndarray, cat: G.Catalog, seed: int, strings: Sequence[str] = None,
                         chunk: int = 2000) -> dict:
    """Matcher behaviour on reproducible synthetic outputs over the SAME base grids: iid uniform grids,
    per-grid value shuffles (histogram preserved), and fixed wrong legal transforms."""
    import pandas as pd
    strings = list(strings if strings is not None else LEGACY.words)
    rng = np.random.default_rng(seed)
    n_g = len(Xu)
    T = transformed_inputs(Xu, cat)
    gi = np.repeat(np.arange(n_g), len(strings))
    truth = np.array([cat.word_id(from_label(s)) for s in strings])[np.tile(np.arange(len(strings)), n_g)]
    rand_elem = int(rng.integers(1, cat.size))
    a_id = cat.word_id("A")
    controls = {
        "iid_uniform_random_grid": rng.integers(0, 10, (len(gi), 36)),
        "input_value_shuffle": rng.permuted(Xu[gi], axis=1),
        "wrong_legal: truth then extra A": T[gi, [cat.compose_ids(t, a_id) for t in truth]],
        "wrong_legal: Z o truth": T[gi, [cat.compose_ids(t, cat.z_id) for t in truth]],
        f"wrong_legal: truth then fixed element {cat.label(rand_elem)}": T[gi, [cat.compose_ids(t, rand_elem) for t in truth]],
    }
    summ, hist = [], []
    for name, Pc in controls.items():
        ag = agreement_matrix(Pc, gi, T, chunk)
        b = ag.max(1)
        ne = (ag == 36).sum(1)
        tcell = ag[np.arange(len(gi)), truth] / 36
        summ.append({"control": name, "n": len(gi), "exact_group_match_rate": (ne > 0).mean(),
                     "ambiguous_exact_match_rate": (ne > 1).mean(), "mean_best_group_cell_agreement": b.mean() / 36,
                     "p05_best": np.quantile(b, 0.05) / 36, "median_best": np.median(b) / 36,
                     "p95_best": np.quantile(b, 0.95) / 36, "cell_acc_vs_true_target": tcell.mean()})
        cnt = np.bincount(b, minlength=37)
        hist += [{"control": name, "best_agreement_cells": k, "fraction": cnt[k] / len(b)} for k in range(37)]
    tA = cat.word_id("A")
    cands = {"A vs A+extra A": cat.perms[cat.compose_ids(tA, a_id)],
             "A vs Z o A": cat.perms[cat.compose_ids(tA, cat.z_id)],
             "A vs fixed random element": cat.perms[rand_elem]}
    for k in (6, 12, 18, 30):
        q = cat.perms[tA].copy()
        pos = rng.choice(36, 36 - k, replace=False)
        while True:
            perm = rng.permutation(pos)
            if (perm != pos).all():
                break
        q[pos] = cat.perms[tA][perm]
        cands[f"A vs arbitrary map sharing {k} positions"] = q
    form = []
    p = cat.perms[tA]
    for name, q in cands.items():
        form.append({"comparison": name, "positions_shared_k": int((p == q).sum()),
                     "expected_cell_acc_formula": G.expected_wrong_perm_cell_accuracy(p, q),
                     "empirical_cell_acc_on_base_grids": (Xu[:, p] == Xu[:, q]).mean(),
                     "log10_exact_agreement_probability": G.exact_agreement_log10_probability(p, q)})
    return {"summary": pd.DataFrame(summ), "histogram": pd.DataFrame(hist), "formula_check": pd.DataFrame(form)}
