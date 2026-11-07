"""Tests for the post-hoc generalization-failure diagnostics (synthetic fixtures only)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from decompositionality import operations as O
from decompositionality.diagnosis import group as G
from decompositionality.diagnosis.analysis import (analyze_run, calibration_controls, diagnostic_split,
                                                   hypothesis_population_summary)
from decompositionality.diagnosis.spec import LEGACY
from decompositionality.diagnosis.artifacts import Run, audit_run, load_run, metrics_tables, synthetic_oracle_selfcheck
from decompositionality.diagnosis.fixtures import write_fixture_results

CAT = G.Catalog()
TRAIN1 = O.TRAIN_SETS["distinct_pairs"]
TRAIN2 = O.TRAIN_SETS["all_pairs"]
SRC = Path(__file__).resolve().parents[1] / "src"


def grids(n=60, seed=0):
    return np.random.default_rng(seed).integers(0, 10, (n, 36)).astype(np.int16)


def make_run(behaviour, n=60, seed=0, experiment="distinct_pairs", model="m", strings=None):
    X0 = grids(n, seed)
    rng = np.random.default_rng(seed + 1)
    X, Y, P, gid, S = [], [], [], [], []
    for g in range(n):
        for s in strings or O.TEST_STRINGS:
            x = X0[g]
            X.append(x); Y.append(G.oracle_apply_string(x, s)); gid.append(g); S.append(s)
            P.append(behaviour(x, s, g, rng))
    return Run(experiment, model, "synthetic", {"predictions": "synthetic"}, np.array(X), np.array(Y),
               np.array(P, dtype=np.int16), np.array(gid), np.array(S), {}, {"file_sha256": {}})


def split_for(run):
    return diagnostic_split(run.grid_id, 3)


# ---------------------------------------------------------------- geometry
def test_generator_directions_and_convention():
    lab = np.arange(36)
    assert G.apply_perm(G.A_SRC, lab).reshape(6, 6)[0].tolist() == [5, 0, 1, 2, 3, 4]
    assert G.apply_perm(G.B_SRC, lab).reshape(6, 6)[0].tolist() == [30, 24, 18, 12, 6, 0]
    assert G.apply_perm(G.C_SRC, lab).reshape(6, 6)[0].tolist() == [5, 4, 3, 2, 1, 0]
    x = grids(1)[0].reshape(6, 6)
    for s in O.TEST_STRINGS:  # independent oracle == sequential array operations of the training code
        assert np.array_equal(G.oracle_apply_string(x, s), O.apply_string(x, s)), s
    # composition convention: compose(first, second) == apply first, then second
    p, q = G.word_perm("AB"), G.word_perm("C")
    assert np.array_equal(G.apply_perm(G.compose(p, q), x), G.apply_perm(q, G.apply_perm(p, x)))


def test_catalog_properties_and_z():
    info = G.check_catalog(CAT)
    assert info["size"] == 288
    Z = G.Z_SRC
    assert np.array_equal(G.compose(Z, Z), G.IDENTITY)
    for g in G.GENERATORS.values():
        assert np.array_equal(G.compose(Z, g), G.compose(g, Z))
    assert len({CAT.inverse_ids[i] for i in range(288)}) == 288


def test_alternate_generators():
    for s in O.all_strings(2):
        assert CAT.alt_generator_id(s) == CAT.word_id(s)          # pairs unchanged
    for s in O.all_strings(1) + O.all_strings(3):
        assert CAT.alt_generator_id(s) == CAT.compose_ids(CAT.word_id(s), CAT.z_id)
        # literally executing A'=ZA, B'=ZB, C'=ZC
        p = G.IDENTITY.copy()
        for ch in s:
            p = G.compose(p, G.compose(G.GENERATORS[ch], G.Z_SRC))
        assert CAT.id_of(p) == CAT.alt_generator_id(s)


def test_hypotheses_and_formula():
    h = {(d["hypothesis_type"], d["executed_word"]) for d in G.hypotheses_for("ABA", TRAIN1)}
    assert {("first_op", "A"), ("last_op", "A"), ("first_and_last", "AA"), ("first_two", "AB"), ("last_two", "BA"),
            ("drop_one", "BA"), ("drop_one", "AA"), ("drop_one", "AB"), ("reverse", "ABA"),
            ("collapse_adjacent", "ABA"), ("dedupe_global", "AB"), ("original_pair", "CB")} <= h
    hc = {(d["hypothesis_type"], d["executed_word"]) for d in G.hypotheses_for("C", TRAIN1)}
    assert ("first_and_last", "CC") in hc and ("duplicate_single", "CC") in hc and ("reverse", None) in hc
    he = {(d["hypothesis_type"], d["executed_word"]) for d in G.hypotheses_for("", TRAIN1)}
    assert ("identity", "") in he and ("first_op", None) in he and ("duplicate_single", None) in he
    assert {d["executed_word"] for d in G.hypotheses_for("AAB", TRAIN1) if d["hypothesis_type"] == "collapse_adjacent"} == {"AB"}
    assert G.exact_agreement_log10_probability(G.word_perm("A"), G.compose(G.word_perm("A"), G.A_SRC)) == -30
    assert G.exact_agreement_log10_probability(G.word_perm("A"), G.compose(G.word_perm("A"), G.Z_SRC)) == -18
    ctrl = calibration_controls(grids(400), CAT, 1)
    f = ctrl["formula_check"]
    assert np.allclose(f["expected_cell_acc_formula"], f["empirical_cell_acc_on_base_grids"], atol=0.03)
    s = ctrl["summary"].set_index("control")
    assert s.loc["iid_uniform_random_grid", "exact_group_match_rate"] == 0
    assert s.loc["input_value_shuffle", "exact_group_match_rate"] < 0.01
    assert (s.filter(like="wrong_legal", axis=0)["exact_group_match_rate"] == 1).all()
    assert (s.filter(like="wrong_legal", axis=0)["cell_acc_vs_true_target"] < 0.2).all()


# ---------------------------------------------------------------- matching
def test_perfect_identity_wrong_and_z_and_prefix():
    perfect = analyze_run(make_run(lambda x, s, g, r: G.oracle_apply_string(x, s)), CAT, LEGACY, split_for(make_run(lambda *a: a[0], n=60)))
    ps = perfect["per_string"]
    assert (ps.exact_acc == 1).all() and (ps.any_exact_group_match_rate == 1).all()
    run = make_run(lambda x, s, g, r: x.copy())
    res = analyze_run(run, CAT, LEGACY, split_for(run))
    ps = res["per_string"].set_index("string")
    assert ps.loc["ABC", "dominant_wrong_unique_word"] == "I" and ps.loc["ABC", "stable_rule_selected_word"] == "I"
    assert ps.loc["CC", "exact_acc"] == 1  # CC is the identity: copying is correct there
    ht = res["hypothesis_types"].set_index(["string", "hypothesis_type"])
    assert ht.loc[("ABC", "identity"), "compatible_rate_any"] == 1
    assert not ht.loc[("CC", "identity"), "eligible"]           # identity is the TRUTH for CC, not an error rule
    zr = make_run(lambda x, s, g, r: G.apply_perm(G.Z_SRC, G.oracle_apply_string(x, s)) if len(s) % 2 else G.oracle_apply_string(x, s))
    zres = analyze_run(zr, CAT, LEGACY, split_for(zr))["per_string"].set_index("string")
    assert zres.loc["A", "alt_generator_exact_rate"] == 1 and zres.loc["A", "exact_acc"] == 0
    assert zres.loc["AB", "alt_generator_exact_rate"] == 1 and zres.loc["AB", "exact_acc"] == 1
    assert zres.loc["ABC", "stable_rule_selected_is_alt_generator"]
    pr = make_run(lambda x, s, g, r: G.oracle_apply_string(x, s[:2]))
    pres = analyze_run(pr, CAT, LEGACY, split_for(pr))
    h = pres["hypothesis_types"].set_index(["string", "hypothesis_type"])
    assert h.loc[("ABC", "first_two"), "compatible_rate_any"] == 1
    assert h.loc[("CBA", "original_pair"), "compatible_rate_any"] == 1   # CB is an original training pair
    assert pres["per_string"].set_index("string").loc["ABC", "exact_set_includes_training_transform_rate"] == 1


def test_arbitrary_permutation_and_many_to_one_maps_recovered():
    perm = np.random.default_rng(5).permutation(36)
    assert CAT.id_of(perm) == -1
    many = np.random.default_rng(6).integers(0, 36, 36)
    def beh(x, s, g, r):
        return x[perm] if s == "A" else x[many] if s == "B" else G.oracle_apply_string(x, s)
    run = make_run(beh, n=80)
    res = analyze_run(run, CAT, LEGACY, split_for(run))
    sm = res["source_maps"].set_index(["string", "map_kind"])
    assert sm.loc[("A", "bijective_permutation"), "diag_cell_fidelity"] == 1
    assert sm.loc[("A", "bijective_permutation"), "group_id"] == -1
    assert np.array_equal(res["maps"]["A"]["bijective_map"], perm)
    assert sm.loc[("B", "argmax_source_map"), "diag_cell_fidelity"] == 1
    assert np.array_equal(res["maps"]["B"]["argmax_map"], many)
    assert sm.loc[("B", "bijective_permutation"), "diag_cell_fidelity"] < 1   # many-to-one is not a permutation
    assert res["per_string"].set_index("string").loc["B", "digit_histogram_preserved_rate"] < 0.2
    comp = res["composition"].set_index(["string", "map_kind"])
    # B's behaviour has no stable PERMUTATION, so compositions involving B are flagged weak
    assert comp.loc[("AB", "bijective_permutation"), "min_single_map_diag_fidelity"] < 0.9
    assert comp.loc[("AB", "bijective_permutation"), "interpretation"].startswith("WEAK")
    # AC: stable single maps (perm, true C); composed prediction = perm then C, which the model does NOT do on AC
    assert comp.loc[("AC", "bijective_permutation"), "min_single_map_diag_fidelity"] == 1
    assert comp.loc[("AC", "bijective_permutation"), "agreement_with_model_exact"] == 0
    split = split_for(run)
    assert not set(split["fit"]) & set(split["diag"]) and len(split["fit"]) == len(split["diag"]) == 40


def test_composition_consistency_of_wrong_primitives():
    # every op replaced by ZX: compositions of the fitted single maps reproduce the model on all words
    run = make_run(lambda x, s, g, r: G.apply_perm(CAT.perms[CAT.alt_generator_id(s)], x))
    comp = analyze_run(run, CAT, LEGACY, split_for(run))["composition"]
    c = comp[comp.map_kind == "bijective_permutation"].set_index("string")
    assert (c.agreement_with_model_exact == 1).all()
    assert c.loc["ABC", "agreement_with_target_exact"] == 0 and c.loc["AB", "agreement_with_target_exact"] == 1


def test_input_dependent_switching():
    ids = [CAT.word_id("A"), CAT.word_id("B"), CAT.word_id("C")]
    run = make_run(lambda x, s, g, r: G.apply_perm(CAT.perms[ids[g % 3]], x), n=90)
    ps = analyze_run(run, CAT, LEGACY, split_for(run))["per_string"].set_index("string")
    assert ps.loc["ABC", "any_exact_group_match_rate"] == 1
    assert ps.loc["ABC", "stable_rule_diag_exact_fidelity"] < 0.5


def test_symmetric_grid_ambiguity_kept():
    const = np.full(36, 4, np.int16)
    X = np.stack([const] * 39)
    Y = np.stack([G.oracle_apply_string(const, s) for s in O.TEST_STRINGS])
    run = Run("distinct_pairs", "m", "synthetic", {}, X, Y, Y.copy(), np.zeros(39, int), np.array(O.TEST_STRINGS),
              {}, {"file_sha256": {}})
    pe = analyze_run(run, CAT, LEGACY, {"fit": np.array([0]), "diag": np.array([1])})["per_example"]
    assert (pe.n_exact_group_matches == 288).all() and (pe.unique_exact_match_id == -1).all()
    assert (pe.category == "correct").all()


# ---------------------------------------------------------------- artifacts & audit
def _write(tmp, run, shuffle=False, extra=None):
    o = np.random.default_rng(0).permutation(len(run.P)) if shuffle else np.arange(len(run.P))
    p = tmp / "test_predictions.npz"
    d = dict(pred=run.P[o].reshape(-1, 6, 6).astype(np.int8), target=run.Y[o].reshape(-1, 6, 6).astype(np.int8),
             input=run.X[o].reshape(-1, 6, 6).astype(np.int8), grid_id=run.grid_id[o], string=run.string[o].astype("U3"))
    d.update(extra or {})
    np.savez_compressed(p, **d)
    return p


def test_invalid_tokens_shuffle_duplicates_and_manifest(tmp_path):
    def beh(x, s, g, r):
        y = G.oracle_apply_string(x, s).copy()
        if g == 0:
            y[10:] = -1   # early EOS / non-digit tokens
        if g == 1:
            y[0] = 15     # an out-of-range token id that must NOT be clipped into 0..9
        return y
    run = make_run(beh, n=20)
    a = load_run("distinct_pairs", "m", _write(tmp_path, run, shuffle=False))
    tmp2 = tmp_path / "s"; tmp2.mkdir()
    b = load_run("distinct_pairs", "m", _write(tmp2, run, shuffle=True))
    assert np.array_equal(a.P, b.P) and np.array_equal(a.string, b.string)   # file order irrelevant
    au = audit_run(a, run.X[::39])
    assert au["out_of_range_prediction_values"] == 39 and au["invalid_records"] == 2 * 39
    assert au["target_oracle_mismatches"] == 0 and au["input_manifest_mismatches"] == 0 and not au["attribution_blocked"]
    ps, gr = metrics_tables(a, LEGACY)
    assert (ps.exact_correct == 18).all() and (ps.n == 20).all()
    a.experiment = "all_pairs"
    assert np.isnan(metrics_tables(a, LEGACY)[1].set_index("group").loc["pairs_unseen", "exact_acc"])
    # duplicates, a wrong target and a mismatched manifest are all flagged
    bad = make_run(lambda x, s, g, r: G.oracle_apply_string(x, s), n=5)
    bad.Y[3] = (bad.Y[3] + 1) % 10
    bad.grid_id[5] = bad.grid_id[44]; bad.string[5] = bad.string[44]
    tmp3 = tmp_path / "d"; tmp3.mkdir()
    c = load_run("distinct_pairs", "m", _write(tmp3, bad))
    au = audit_run(c, np.zeros((5, 36), np.int16))
    assert au["duplicate_keys"] >= 1 and au["target_oracle_mismatches"] >= 1 and au["input_manifest_mismatches"] > 0
    assert au["attribution_blocked"]
    assert len(au["missing_strings"]) == 0


def test_oracle_selfcheck(tmp_path):
    ps = synthetic_oracle_selfcheck(grids(5), tmp_path)
    assert len(ps) == 39 and (ps.exact_acc == 1).all()


# ---------------------------------------------------------------- end to end, torch-free, no optimizer
def test_end_to_end_fixture_torch_free(tmp_path):
    root = tmp_path / "results"
    write_fixture_results(root, n_grids=20)
    code = f"""
import sys; sys.path.insert(0, {str(SRC)!r})
from decompositionality.diagnosis.runner import DiagnosticsConfig, run_diagnostics
cfg = DiagnosticsConfig({{e: {str(root)!r} + '/' + e for e in ('distinct_pairs', 'all_pairs')}}, {str(tmp_path / 'diag')!r})
R = run_diagnostics(cfg, log=lambda *a: None)
assert 'torch' not in sys.modules, 'analysis path imported torch'
print(len(R['per_run']))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("8")
    d = tmp_path / "diag"
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["analysis_environment"]["torch_imported_by_analysis"] is False
    for f in ("REPORT.md", "tables/20_summary_by_experiment_model_group.csv", "tables/22_alternate_generator_vs_target.csv",
              "figures/07_alternate_generator_vs_target.png", "figures/11_paired_experiment_comparison.png"):
        assert (d / f).exists(), f
    import pandas as pd
    s = pd.read_csv(d / "tables/20_summary_by_experiment_model_group.csv")
    z = s[(s.model == "transformer_ar") & (s.group == "triples_all_distinct")]
    assert (z["alt_generator(Z)_exact_agreement"] > 0.85).all()
    assert s[(s.experiment == "all_pairs") & (s.group == "pairs_unseen")]["true_exact_acc"].isna().all()
    # source artifacts untouched (hashes before/after)
    from decompositionality.diagnosis.artifacts import sha256_file
    before = {p: sha256_file(p) for p in root.rglob("test_predictions.npz")}
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert before == {p: sha256_file(p) for p in root.rglob("test_predictions.npz")}


def test_missing_artifacts_and_frozen_inference_parity(tmp_path, monkeypatch):
    """Missing predictions are inventoried as missing; opt-in frozen inference from best.pt reproduces the
    original evaluation exactly, without constructing any optimizer."""
    import torch
    from conftest import TINY
    from decompositionality import pipeline
    from decompositionality.config import make_experiment_config
    from decompositionality.diagnosis.artifacts import inventory
    from decompositionality.diagnosis.runner import DiagnosticsConfig, run_diagnostics

    cfg = make_experiment_config("distinct_pairs", "smoke", str(tmp_path / "res"), ["transformer_ar", "unet_film"], device="cpu")
    cfg.model_overrides = {m: TINY[m] for m in cfg.models}
    ctx = pipeline.prepare(cfg, log=lambda *_: None)
    pipeline.train_and_evaluate(ctx, log=lambda *_: None)
    exp_dir = cfg.experiment_dir
    orig = {m: np.load(exp_dir / "eval" / m / "test_predictions.npz") for m in cfg.models}
    orig = {m: {k: z[k] for k in z.files} for m, z in orig.items()}
    for m in cfg.models:
        (exp_dir / "eval" / m / "test_predictions.npz").unlink()
    dirs = {"distinct_pairs": str(exp_dir), "all_pairs": str(tmp_path / "res" / "smoke" / "all_pairs")}  # 2nd missing
    inv = inventory(dirs)
    assert inv[inv.experiment == "distinct_pairs"].status.str.contains("frozen inference possible").all()
    assert inv[inv.experiment == "all_pairs"].status.str.startswith("MISSING: no predictions").all()

    def no_optimizer(*a, **k):
        raise AssertionError("an optimizer was constructed during diagnostics")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", no_optimizer)
    out = tmp_path / "diag"
    R = run_diagnostics(DiagnosticsConfig(dirs, str(out), run_frozen_inference_if_missing=True), log=lambda *_: None,
                        make_figures=False)
    assert set(R["audit"]["source"]) == {"regenerated"} and len(R["audit"]) == 2
    for m in cfg.models:
        with np.load(out / "regenerated_predictions" / "distinct_pairs" / m / "test_predictions.npz") as z:
            assert np.array_equal(z["pred"], orig[m]["pred"]) and np.array_equal(z["string"], orig[m]["string"])
        prov = json.loads((out / "regenerated_predictions" / "distinct_pairs" / m / "regeneration_provenance.json").read_text())
        assert prov["optimizer_created"] is False and prov["kind"].startswith("REGENERATED")
    assert not (exp_dir / "eval" / "transformer_ar" / "test_predictions.npz").exists()  # originals never written


def test_hypothesis_denominators_union_and_bounds():
    """Union counting within a rule, explicit eligibility, and unconditional coverage <= any-legal matching."""
    # model: drop the FIRST op for triples (AAB -> AB: two drop_one candidates map to AB -> counted once),
    # identity for singles (identity is eligible), truth elsewhere
    def beh(x, s, g, r):
        if len(s) == 3:
            return G.oracle_apply_string(x, s[1:])
        if len(s) == 1:
            return x.copy()
        return G.oracle_apply_string(x, s)
    run = make_run(beh, n=30)
    res = analyze_run(run, CAT, LEGACY, split_for(run))
    ht = res["hypothesis_types"].set_index(["string", "hypothesis_type"])
    assert ht.loc[("AAB", "drop_one"), "matched_unique"] == 30           # not 60: union, each record once
    assert not ht.loc[("A", "first_op"), "eligible"] and not ht.loc[("A", "drop_one"), "applicable"]
    hs = hypothesis_population_summary(res["hypothesis_types"], res["per_string"], LEGACY)
    assert (hs["matched_unique_records"] <= hs["eligible_records"]).all()
    assert (hs["matched_unique_records"] <= hs["population_any_legal_match_records"]).all()
    assert (hs["unconditional_coverage (matched/population)"].fillna(0)
            <= hs["conditional_rate (matched/eligible)"].fillna(0) + 1e-12).all()
    row = hs[(hs.population.str.startswith("unseen")) & (hs.hypothesis_type == "duplicate_single")].iloc[0]
    assert row.eligible_records == 3 * 30 and row.population_records == 33 * 30      # singles only eligible
    empty = hs[(hs.population == "singles") & (hs.hypothesis_type == "first_two")].iloc[0]
    assert empty.eligible_records == 0 and np.isnan(empty["conditional_rate (matched/eligible)"])
