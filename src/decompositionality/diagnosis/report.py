"""Evidence-based markdown report generated ONLY from the analysed artifacts."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..names import DISPLAY_NAMES

STABLE = 0.95     # DIAG exact fidelity above which a fixed rule is called "stable"
UNEXPLAINED = 0.5


def _md(df, floatfmt="{:.3f}") -> str:
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
    L = ["# Generalization-failure diagnostics (post-hoc, frozen predictions)", ""]
    L += ["This report is generated automatically from the saved prediction artifacts listed below. It describes model "
          "BEHAVIOUR on test inputs. Compatibility with an explanation does not establish internal representations or "
          "causality, and only one training seed exists, so nothing here is a cross-seed robustness claim.", ""]
    inv = R["inventory"]
    L += ["## 1. Data coverage and provenance", "", _md(inv[["experiment", "model", "status"]])]
    au = R["audit"]
    if len(au):
        cols = ["experiment", "model", "source", "n_records", "n_strings", "selected_step", "code_commit_at_eval",
                "decoding", "duplicate_keys", "invalid_records", "target_oracle_mismatches", "input_manifest_mismatches",
                "attribution_blocked"]
        L += ["### Audit", "", _md(au[[c for c in cols if c in au]]), ""]
        L += [f"- Stored targets re-derived with the independent oracle for **{int(au.n_records.sum()):,}** records; "
              f"mismatches: **{int(au.target_oracle_mismatches.sum())}**.",
              f"- Test manifests identical across experiments: **{R.get('manifests_identical_across_experiments')}**.",
              "- A synthetic perfect-oracle predictor passed through the same loader/metric code scores 100% on all 39 "
              "strings (this checks the analysis path, not any checkpoint's forward pass).", ""]
    else:
        L += ["**No prediction artifacts were available: no behavioural analysis could be performed.**", ""]
        p = out / "REPORT.md"
        p.write_text("\n".join(L))
        return p

    summ = R["summary"]
    L += ["## 2. Verified observations (recomputed from raw records)", ""]
    show = summ[["experiment", "model", "group", "exact_correct", "n_records", "true_exact_acc", "true_cell_acc",
                 "any_exact_legal_match", "wrong_unique_legal", "no_legal_match", "DIAG stable_rule_exact_fidelity",
                 "DIAG bijective_perm_cell_fidelity", "alt_generator(Z)_exact_agreement"]].copy()
    show["model"] = show["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    L += ["Columns prefixed DIAG use only the diagnostic half of the base grids (rules fitted on the other half). "
          "N/A = empty category (e.g. unseen pairs in all_pairs).", "", _md(show), ""]

    ps = R["per_string"]
    ood = ps[ps.group != "pairs_seen"]
    for e in [x for x in ("distinct_pairs", "all_pairs") if x in set(ps.experiment)]:
        o = ood[ood.experiment == e]
        wrong = o["n"] - o["exact_correct"]
        L += [f"### {e}", ""]
        L += [f"- Unseen strings: {int(o.exact_correct.sum()):,} exact-correct of {int(o.n.sum()):,} records "
              f"({o.exact_correct.sum() / max(1, o.n.sum()):.6f})."]
        legal = (o["any_exact_group_match_rate"] - o["exact_acc"]).clip(lower=0) * o["n"]
        L += [f"- Wrong predictions on unseen strings that EXACTLY equal some legal transform of the input: "
              f"{legal.sum() / max(1, wrong.sum()):.4f} of {int(wrong.sum()):,}."]
        st = o.dropna(subset=["stable_rule_diag_exact_fidelity"])
        stable = st[st.stable_rule_diag_exact_fidelity >= STABLE]
        L += [f"- (model, unseen string) cells where ONE fixed legal transform fitted on FIT grids reproduces >= {STABLE:.0%} "
              f"of DIAG predictions exactly: {len(stable)} of {len(st)}."]
        if len(stable):
            k_train = int(stable["stable_rule_selected_is_training_transform"].sum())
            k_alt = int(stable["stable_rule_selected_is_alt_generator"].sum())
            top = stable["stable_rule_selected_word"].value_counts().head(8)
            L += [f"  - of these, the fitted rule is a training-pair transformation in {k_train} and equals the "
                  f"alternate-generator (Z-twisted) prediction in {k_alt}; most frequent fitted rules (canonical words): "
                  + ", ".join(f"{w} x{c}" for w, c in top.items())]
        odd = o[o.length % 2 == 1]
        if len(odd):
            L += [f"- Odd-length strings: exact agreement with Z(target) = "
                  f"{np.average(odd.alt_generator_exact_rate, weights=odd.n):.4f} (record-weighted over models/strings)."]
        cp = R["composition"]
        if len(cp):
            c = cp[(cp.experiment == e) & (cp.map_kind == "bijective_permutation")]
            if len(c):
                L += [f"- Composition of fitted single-operation maps (DIAG): mean exact agreement with model predictions "
                      f"{c.agreement_with_model_exact.mean():.4f}, with true targets {c.agreement_with_target_exact.mean():.4f}; "
                      f"{int((c.interpretation != 'single maps stable').sum())} of {len(c)} rows flagged WEAK (unstable single maps)."]
        L += [""]
    pc = R["paired"]
    if len(pc):
        L += ["### Cross-experiment (same model, string and base grid)", "",
              _md(pc.groupby("model")[["identical_prediction_rate", "delta_exact (all_pairs - distinct_pairs)",
                                       "delta_any_legal (all_pairs - distinct_pairs)"]].mean().reset_index()), ""]

    L += ["## 3. Hypotheses compatible with the observations", ""]
    hyp_lines = []
    ht = R["hypothesis_types"]
    if len(ht):
        h = ht[ht.group != "pairs_seen"].groupby(["experiment", "hypothesis_type"]).apply(
            lambda d: np.average(d.compatible_rate_any, weights=d.n)).reset_index(name="rate")
        for _, r in h.sort_values("rate", ascending=False).iterrows():
            if r.rate >= 0.25:
                hyp_lines.append(f"- {r.experiment}: {r.rate:.3f} of unseen-string examples are exactly compatible with a "
                                 f"non-truth '{r.hypothesis_type}' hypothesis (hypothesis types overlap).")
    for e in [x for x in ("distinct_pairs", "all_pairs") if x in set(ps.experiment)]:
        odd = ood[(ood.experiment == e) & (ood.length % 2 == 1)]
        if len(odd) and np.average(odd.alt_generator_exact_rate, weights=odd.n) >= 0.5:
            hyp_lines.append(f"- {e}: odd-length behaviour is largely compatible with the alternate generators ZA, ZB, ZC "
                             "(an identifiability-consistent reading of the length-two training data).")
    L += hyp_lines or ["- No pre-registered hypothesis type reaches 25% exact compatibility on unseen strings."]
    L += ["", "## 4. Unresolved cases", ""]
    unres = ood.dropna(subset=["stable_rule_diag_exact_fidelity"])
    unres = unres[(unres.stable_rule_diag_exact_fidelity < UNEXPLAINED) & (unres.perm_diag_cell_fidelity < 0.9)
                  & (unres.argmax_map_diag_cell_fidelity < 0.9)]
    L += [f"(model, unseen string) cells with no stable legal rule (DIAG exact < {UNEXPLAINED}) AND no stable source map "
          f"(DIAG cell fidelity < 0.9): **{len(unres)}**. This only rules out these specific explanation families; it is "
          "not evidence of randomness.", ""]
    if len(unres):
        u = unres.groupby(["experiment", "model"])["string"].apply(lambda s: " ".join(s)).reset_index()
        L += [_md(u)]
    L += ["", "## 5. Limits", "",
          "- Exact legal-transform matching uses all 288 transformations; matching one does not identify which symbolic "
          "string the model 'executed', and ambiguous matches are kept as ambiguous.",
          "- DIAG-split fidelities describe frozen predictions with rules fitted to those predictions; they are not a new "
          "held-out benchmark of the neural models.",
          "- Behavioural compatibility is not a causal or mechanistic claim; one seed only."]
    p = out / "REPORT.md"
    p.write_text("\n".join(L) + "\n")
    return p
