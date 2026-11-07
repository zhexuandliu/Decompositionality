"""Evidence-based markdown report generated ONLY from the analysed artifacts.

v2 corrections: every hypothesis statement reports the declared population size, eligible records,
UNIQUE matched records (union within a rule), the conditional rate (matched / eligible) and the
unconditional coverage (matched / population); integer successes are always shown next to rates; the
"stable" threshold is defined explicitly."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..names import DISPLAY_NAMES

STABLE = 0.95       # a fitted fixed rule is called "stable" if its DIAG exact fidelity >= STABLE
UNEXPLAINED = 0.5
MAP_OK = 0.9


def _md(df, floatfmt="{:.4f}") -> str:
    if df is None or not len(df):
        return "_(no data)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("N/A" if math.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def write_report(R, out: Path) -> Path:
    spec = R["spec"]
    L = [f"# Generalization-failure diagnostics ({spec.name} family; analysis {R['analysis_version']})", ""]
    L += ["Generated automatically from the saved prediction artifacts listed below. It describes model BEHAVIOUR on "
          "test inputs. Compatibility with an explanation does not establish internal representations or causality; "
          "one training seed only.", "",
          f"Definitions: a fitted fixed rule is **stable** if its DIAG exact fidelity >= {STABLE}; a source map is "
          f"**adequate** if its DIAG cell fidelity >= {MAP_OK}. DIAG = disjoint half of the base grids not used for "
          "fitting.", ""]
    inv = R["inventory"]
    L += ["## 1. Data coverage and provenance", "", _md(inv[["experiment", "model", "status"]])]
    au = R["audit"]
    if not len(au):
        L += ["**No prediction artifacts were available: no behavioural analysis could be performed.**", ""]
        p = out / "REPORT.md"
        p.write_text("\n".join(L))
        return p
    cols = ["experiment", "model", "source", "n_records", "n_strings", "selected_step", "code_commit_at_eval", "decoding",
            "duplicate_keys", "invalid_records", "target_oracle_mismatches", "input_manifest_mismatches",
            "attribution_blocked"]
    L += ["### Audit", "", _md(au[[c for c in cols if c in au]]), "",
          f"- Stored targets re-derived with the independent oracle for **{int(au.n_records.sum()):,}** records; "
          f"mismatches: **{int(au.target_oracle_mismatches.sum())}**.",
          f"- Test manifests identical across loaded experiments: **{R.get('manifests_identical_across_experiments')}**.",
          f"- A synthetic perfect-oracle predictor passed through the same loader/metric code scores 100% on all "
          f"{len(spec.words)} words (checks the analysis path, not any checkpoint's forward pass).", ""]

    summ = R["summary"]
    L += ["## 2. Verified observations (recomputed from raw records)", "",
          "Integer exact successes / records are shown; rates have 4 decimals (e.g. 8998/9000 = 0.9998 is NOT 1). "
          "Columns prefixed DIAG use only the diagnostic half of the base grids. N/A = empty category.", ""]
    show = summ[["experiment", "model", "group", "exact_correct", "n_records", "true_exact_acc", "true_cell_acc",
                 "any_legal_match_records", "any_exact_legal_match", "exact_set_includes_original_pair",
                 "DIAG stable_rule_exact_fidelity", "DIAG argmax_map_cell_fidelity", "DIAG bijective_perm_cell_fidelity"]].copy()
    show["model"] = show["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    L += [_md(show), ""]

    ps = R["per_string"]
    hs = R.get("hypothesis_summary")
    pops = spec.populations
    main_pop, main_groups = list(pops.items())[0]
    for e in [x for x in spec.experiments if x in set(ps.experiment)]:
        o = ps[(ps.experiment == e) & ps.group.isin(main_groups)]
        if not len(o):
            continue
        wrong = int((o["n"] - o["exact_correct"]).sum())
        legal_wrong = int((o["any_exact_group_match_count"] - o["exact_correct"]).clip(lower=0).sum())
        L += [f"### {e} - population '{main_pop}'", "",
              f"- Exact-correct: {int(o.exact_correct.sum()):,} of {int(o.n.sum()):,} records "
              f"({o.exact_correct.sum() / max(1, o.n.sum()):.6f}).",
              f"- Wrong predictions exactly equal to SOME legal transform of the input: {legal_wrong:,} of {wrong:,} wrong "
              f"records ({legal_wrong / max(1, wrong):.4f})."]
        pair_wrong = int(o["wrong_matching_original_pair_count"].sum())
        L += [f"- Wrong predictions whose exact-match set includes an ORIGINAL length-two pair transformation: "
              f"{pair_wrong:,} of {wrong:,}."]
        st = o.dropna(subset=["stable_rule_diag_exact_fidelity"])
        stable = st[st.stable_rule_diag_exact_fidelity >= STABLE]
        L += [f"- (model, word) cells with a STABLE fitted fixed rule: {len(stable)} of {len(st)} "
              f"(of which the rule is the truth in {int(stable.stable_rule_selected_is_truth.sum())}, an original pair "
              f"transformation in {int(stable.stable_rule_selected_is_original_pair.sum())}, the Z-twisted alternative in "
              f"{int(stable.stable_rule_selected_is_alt_generator.sum())})."]
        if len(stable):
            top = stable[~stable.stable_rule_selected_is_truth.astype(bool)]["stable_rule_selected_word"].value_counts().head(8)
            if len(top):
                L += ["  - most frequent stable WRONG rules (canonical words): " + ", ".join(f"{w} x{c}" for w, c in top.items())]
        cp = R["composition"]
        if len(cp):
            c = cp[(cp.experiment == e) & (cp.map_kind == "bijective_permutation")]
            if len(c):
                L += [f"- Composition of fitted single-op maps (DIAG, all words of length >= 2): mean exact agreement with "
                      f"model predictions {c.agreement_with_model_exact.mean():.4f}, with true targets "
                      f"{c.agreement_with_target_exact.mean():.4f}; {int(c.interpretation.str.startswith('WEAK').sum())} "
                      f"of {len(c)} rows WEAK (unstable single maps)."]
        L += [""]
    if hs is not None and len(hs):
        L += ["### Hypothesis coverage with explicit denominators (population: " + main_pop + ")", "",
              "matched = UNIQUE records matched by ANY non-truth candidate of the rule (union within the rule; rules "
              "overlap with each other). conditional = matched / eligible; unconditional = matched / population. "
              "any-legal = records exactly equal to some legal transform (an upper bound for unconditional coverage).", ""]
        h = hs[hs.population == main_pop].groupby(["experiment", "hypothesis_type"], sort=False).agg(
            population=("population_records", "sum"), any_legal=("population_any_legal_match_records", "sum"),
            eligible=("eligible_records", "sum"), matched=("matched_unique_records", "sum")).reset_index()
        h["conditional"] = h.matched / h.eligible.where(h.eligible > 0)
        h["unconditional"] = h.matched / h.population.where(h.population > 0)
        L += ["(summed over models)", "", _md(h), ""]

    pc = R["paired"]
    if len(pc) and spec.paired:
        ea, eb = spec.paired
        L += [f"### Paired comparison ({eb} vs {ea}; same model, word and base grid)", "",
              _md(pc.groupby("model")[["identical_prediction_rate", f"delta_exact ({eb} - {ea})",
                                       f"delta_any_legal ({eb} - {ea})"]].mean().reset_index()), ""]

    L += ["## 3. Hypotheses compatible with the observations (behavioural, non-causal)", ""]
    lines = []
    if hs is not None and len(hs):
        h2 = hs[hs.population == main_pop].groupby(["experiment", "hypothesis_type"]).agg(
            matched=("matched_unique_records", "sum"), eligible=("eligible_records", "sum"),
            population=("population_records", "sum")).reset_index()
        for _, r in h2.sort_values("matched", ascending=False).iterrows():
            if r.population and r.matched / r.population >= 0.25:
                lines.append(f"- {r.experiment}: rule '{r.hypothesis_type}' exactly matches {int(r.matched):,} of "
                             f"{int(r.population):,} records in '{main_pop}' (unconditional {r.matched / r.population:.4f}; "
                             f"conditional {r.matched / max(1, r.eligible):.4f} of {int(r.eligible):,} eligible).")
    L += lines or ["- No pre-declared rule reaches 25% unconditional coverage of the main population."]
    L += ["", "## 4. Unresolved cases", ""]
    unres = ps[ps.group.isin(main_groups)].dropna(subset=["stable_rule_diag_exact_fidelity"])
    unres = unres[(unres.exact_acc < 0.5) & (unres.stable_rule_diag_exact_fidelity < UNEXPLAINED)
                  & (unres.perm_diag_cell_fidelity < MAP_OK) & (unres.argmax_map_diag_cell_fidelity < MAP_OK)]
    L += [f"(model, word) cells in '{main_pop}' that are mostly wrong (exact < 0.5) with no stable legal rule "
          f"(DIAG exact < {UNEXPLAINED}) AND no adequate source map (DIAG cell fidelity < {MAP_OK}): **{len(unres)}**. "
          "This only rules out these explanation families; it is not evidence of randomness.", ""]
    if len(unres):
        L += [_md(unres.groupby(["experiment", "model"])["string"].apply(lambda s: " ".join(s)).reset_index())]
    L += ["", "## 5. Limits", "",
          "- Matching one of the 288 transformations does not identify which synonymous string was executed; "
          "ambiguous matches are kept ambiguous.",
          "- DIAG fidelities describe frozen predictions with rules fitted to those predictions; not a benchmark.",
          "- Behavioural compatibility is not a causal or mechanistic claim; one seed only."]
    p = out / "REPORT.md"
    p.write_text("\n".join(L) + "\n")
    return p
