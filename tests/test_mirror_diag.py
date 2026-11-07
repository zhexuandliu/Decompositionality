"""Focused behavioural checks for the notebook-17 diagnostics (conventions, exact matching, residuals, source maps,
discovery / audits, end-to-end export on a synthetic saved-prediction fixture)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from decompositionality import mirror_diag as M

G = M.group()


def test_conventions_and_group():
    v = M.verify_conventions(G)
    assert v["group_size"] == 288
    # canonical parameters unique and mirror class independent of cell parity
    assert len({tuple(p) for p in G.params}) == 288
    assert G.mirror[G.C] == 1 and G.mirror[G.B] == 0 and G.mirror[G.word_id("CBC")] == 0 and G.mirror[G.word_id("CB")] == 1
    # AB means B(A(X))
    x = np.arange(36)
    assert np.array_equal(x[M.word_perm("AB")], x[M.P_A][M.P_B])


def _rows(x, perms):
    return np.stack([np.asarray(x)[p] for p in perms]).astype(np.int8)


def test_classification_categories():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 10, 36)
    t = G.word_id("A")
    true = x[G.perms[t]]
    out_c = x[G.perms[G.mult[t, G.C]]]           # C(A(x))
    in_c = x[G.perms[G.mult[G.C, t]]]            # A(C(x))
    trans = x[M.translation(2, 3)]
    rot = x[M.word_perm("B")]
    other_m = x[G.perms[G.word_id("BC")]]
    nonlegal = true.copy(); nonlegal[[0, 1]] = nonlegal[[1, 0]] if nonlegal[0] != nonlegal[1] else (nonlegal[[0, 1]] + 1) % 10
    invalid = true.copy().astype(np.int16); invalid[5] = -1
    preds = np.stack([true, out_c, in_c, trans, rot, other_m, nonlegal, invalid]).astype(np.int8)
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 0], bool)
    correct = (preds == true).all(1) & valid
    mt = M.Matcher(G)
    ms = mt.match(preds, np.tile(x, (8, 1)), np.array(["g"] * 8), valid)
    for i in range(7):
        assert ms[i] == M.brute_force_match(G, preds[i], x)
    c = M.classify(G, ms, np.full(8, t), valid, correct)
    assert list(c["outcome"][[0, 6, 7]]) == ["correct", "wrong: valid, no legal match", "wrong: invalid output"]
    assert c["residual"][1] == "extra output-C only" and c["residual"][2] == "extra input-C only"
    assert c["residual"][3] == "orientation-preserving nonidentity" and c["residual"][4] == "orientation-preserving nonidentity"
    assert c["residual"][5] == "other orientation-reversing"
    assert c["match_extra_out_C"][1] and not c["match_extra_in_C"][1] and c["match_extra_in_C"][2]
    assert list(c["pred_mirror"][[1, 2, 5, 6, 7]]) == ["1", "1", "1", "no legal match", "invalid"]
    assert c["mirror_wrong"][1] and not c["mirror_wrong"][3]


def test_singleton_C_identity_alias_and_ambiguity():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 10, 36)
    t = G.C
    pred = x.astype(np.int8)[None]                  # predicts the input = identity
    ms = M.Matcher(G).match(pred, x[None], np.array(["g"]), np.array([True]))
    c = M.classify(G, ms, np.array([t]), np.array([True]), np.array([False]))
    assert c["match_identity"][0] and c["match_extra_out_C"][0] and c["match_extra_in_C"][0]
    assert c["residual"][0] == "both extra-C hypotheses"
    # constant grid: every transform matches; correct grid still scores correct, mirror attribute ambiguous
    z = np.full(36, 7)
    ms = M.Matcher(G).match(z[None].astype(np.int8), z[None], np.array(["z"]), np.array([True]))
    assert len(ms[0]) == 288
    c = M.classify(G, ms, np.array([G.A]), np.array([True]), np.array([True]))
    assert c["outcome"][0] == "correct" and c["pred_mirror"][0] == "ambiguous"
    # left-right symmetric grid: A and A o C produce the same grid -> multiple matches, mirror ambiguous
    sym = np.tile(np.array([1, 2, 3, 3, 2, 1]), 6) + np.repeat(np.arange(6), 6) % 3
    p = sym[G.perms[G.word_id("AB")]].astype(np.int8)
    ms = M.Matcher(G).match(p[None], sym[None], np.array(["s"]), np.array([True]))
    assert len(ms[0]) >= 2 and M.classify(G, ms, np.array([G.I]), np.array([True]), np.array([False]))["pred_mirror"][0] == "ambiguous"


def test_matcher_equals_brute_force_random():
    rng = np.random.default_rng(3)
    mt = M.Matcher(G, max_grids=2)
    for j in range(30):
        x = rng.integers(0, 3, 36)       # few digits -> frequent ambiguity
        p = x[G.perms[rng.integers(0, 288)]] if j % 3 else rng.integers(0, 3, 36)
        got = mt.match(p[None].astype(np.int8), x[None], np.array([f"h{j}"]), np.array([True]))[0]
        assert got == M.brute_force_match(G, p, x)


def test_source_map_fidelity_ties_and_invalid():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 10, (200, 36))
    mp = G.perms[G.word_id("AC")]
    P = X[:, mp].astype(np.int16)
    valid = np.ones(200, bool)
    valid[::10] = False
    P[~valid, 0] = -1
    r = M.source_map(P[:100], X[:100], valid[:100], P[100:], X[100:], valid[100:], G)
    assert r["bijective"] and r["legal_transform"] == G.label(G.word_id("AC"))
    assert abs(r["heldout_exact_fidelity"] - 0.9) < 1e-9          # invalid held-out records count as mismatches
    # nonlegal copying map (two outputs copy the same source) -> not bijective, still high fidelity
    mp2 = mp.copy(); mp2[1] = mp2[0]
    P2 = X[:, mp2]
    r2 = M.source_map(P2[:100], X[:100], np.ones(100, bool), P2[100:], X[100:], np.ones(100, bool), G)
    assert not r2["bijective"] and r2["legal_transform"] == "not legal" and r2["heldout_exact_fidelity"] == 1.0
    # constant inputs -> every source ties
    Z = np.zeros((10, 36), int)
    r3 = M.source_map(Z, Z, np.ones(10, bool), Z, Z, np.ones(10, bool), G)
    assert r3["cells_with_tied_sources"] == 36


def test_six_strata_and_membership():
    from itertools import product
    from decompositionality.mirror_report import training_transforms
    tt = training_transforms("train02345")
    w6 = ["".join(w) for w in product("ABC", repeat=6)]
    assert len(tt) == 111
    assert sum(G.word_id(w) in tt for w in w6) == 591
    assert [sum(w.count("C") == c for w in w6) for c in range(7)] == [64, 192, 240, 160, 60, 12, 1]


# ---------------------------------------------------------------------------
# synthetic saved-prediction fixture through every stage
# ---------------------------------------------------------------------------

def _write_run(root: Path, sub: str, experiment: str, model: str, test: dict, corrupt_fn, complete=True):
    from decompositionality.idvar_pipeline import grid_hashes
    n = len(test["x"])
    Y = test["y"].reshape(n, 36).astype(np.int16)
    X = test["x"].reshape(n, 36).astype(np.int16)
    P = corrupt_fn(X, Y, test).astype(np.int16)
    valid = (P >= 0).all(1)
    arr = {"pred": P.reshape(-1, 6, 6).astype(np.int8), "target": Y.reshape(-1, 6, 6).astype(np.int8),
           "input": X.reshape(-1, 6, 6).astype(np.int8), "grid_id": test["grid_id"].astype(np.int32),
           "grid_hash": grid_hashes(test["x"]), "string": test["string"].astype("U8"), "k": test["k"].astype(np.int8),
           "kind": test["kind"].astype("U6"), "position": test["position"].astype(np.int8),
           "slots5": test["slots"] if "slots" in test else test["slots5"], "valid": valid,
           "exact": valid & (P == Y).all(1), "raw_tokens": P.astype(np.int16)}
    ed = root / sub / "eval" / model
    ed.mkdir(parents=True, exist_ok=True)
    f = ed / "test_predictions.npz"
    np.savez_compressed(f, **arr)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    (ed / "ARTIFACT.json").write_text(json.dumps({"experiment": experiment, "model": model, "seed": 42, "complete": complete,
                                                  "predictions": str(f), "predictions_sha256": sha, "n_records": n,
                                                  "expected_records": n, "selected_step": 5000, "config_hash": "c",
                                                  "manifest_hash": "m"}))
    rd = root / sub / "runs" / model
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "run_config.json").write_text(json.dumps({"model_hash": "h", "data_version": "decomp-idslot-v1" if "slots" in experiment
                                                    else "decomp-idext-v1", "train": {"lr": 3e-4}, "train_ks": [0, 2, 3]}))


def _corrupt(X, Y, test):
    P = Y.copy()
    s = test["kind"] == "single"
    labels = test["string"].astype(str)
    c = s & (labels == "C")
    P[c] = X[c]                                         # C -> identity
    a = s & (labels == "A") & (test["position"] == 1)
    P[a] = Y[a][:, M.P_C]                               # A -> extra output-C
    b = s & (labels == "B")
    P[b, :3] = (P[b, :3] + 1) % 10                      # non-legal
    six = test["kind"] == "six"
    P[six & (np.char.count(labels, "C") % 2 == 1)] = Y[six & (np.char.count(labels, "C") % 2 == 1)][:, M.P_C]
    inv = np.flatnonzero(six)[:5]
    P[inv, 0] = -1
    return P


def test_end_to_end_synthetic(tmp_path):
    import nbformat
    from decompositionality import idslot as I, idvar as V
    from decompositionality.mirror_report import Diagnostics
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 10, (4, 6, 6))
    root = tmp_path / "results"
    ext = root / "identity_ar_count_extrapolation"
    slot = root / "identity_ar_slot_count"
    t5 = V.make_test_set(grids, 42, "identity")
    _write_run(ext, "train023/seed_42", "idext_train023", "transformer_ar", t5, _corrupt)
    t3 = I.make_test_set(grids, 42, "slots3_train023")
    _write_run(slot, "slots3_train023/seed_42", "slots3_train023", "transformer_ar", t3, _corrupt)
    t6 = I.make_test_set(grids, 42, "slots6_train02345")
    _write_run(slot, "slots6_train02345/seed_42", "slots6_train02345", "looped_transformer_ar", t6, _corrupt)
    _write_run(slot, "slots4_train0234/seed_42", "slots4_train0234", "transformer_ar",
               I.make_test_set(grids, 42, "slots4_train0234"), _corrupt, complete=False)        # partial source
    D = Diagnostics(str(root), out_root=str(tmp_path / "diag"), show=False, log=lambda *a: None)
    D.conventions(); D.discover(); D.load(); D.overview(); D.singletons(); D.equivalence(); D.source_maps()
    D.six_step(); D.comparisons(); D.evidence()
    inv = D.T["inventory"].set_index("run_id")
    assert inv.loc["S4/train0234/transformer_ar", "status"].startswith("partial")
    assert inv.loc["S2/train02/transformer_ar", "status"].startswith("NOT FOUND")
    assert set(D.runs) == {"S5/train023/transformer_ar", "S3/train023/transformer_ar", "S6/train02345/looped_transformer_ar"}
    assert D.T["audit"].ok.all()
    op = D.T["singleton_by_op"].set_index(["run_id", "op"])
    r = op.loc[("S3/train023/transformer_ar", "C")]
    assert r["wrong legal: identity"] == r.n == 3 * 4
    r = op.loc[("S5/train023/transformer_ar", "A")]
    assert r["wrong legal: extra out-C"] == 4 and r.correct == 16
    assert op.loc[("S5/train023/transformer_ar", "B"), "wrong: valid, no legal match"] == 20
    st = D.T["six_strata"].set_index("n_C")
    assert (st.records == np.array([64, 192, 240, 160, 60, 12, 1]) * 4).all()
    assert (st.loc[[1, 3, 5], "exact_acc"] == 0).all() and st.loc[0, "exact_acc"] > 0.95
    eq = D.T["equivalence_by_position"]
    row = eq[(eq.run_id == "S3/train023/transformer_ar") & (eq.singleton == "C") & (eq.composite == "CCC")]
    assert row["composite correct / singleton wrong"].sum() == 12 and row["targets_identical"].sum() == 12
    assert (eq[(eq.run_id == "S3/train023/transformer_ar") & (eq.composite == "CCCCC")].status.str.startswith("N/A")).all()
    rp = D.export()
    nb = nbformat.read(rp, as_version=4)
    outs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs]
    assert any("image/png" in o.data for o in outs) and any("text/html" in o.data for o in outs)
    assert (rp.parent / "diagnostic_report.html").exists() and (rp.parent / "summary.json").exists()
    # cache reuse: second run loads cached classification
    D2 = Diagnostics(str(root), out_root=str(tmp_path / "diag"), show=False, log=lambda *a: None)
    D2.discover(); D2.load()
    assert set(D2.T["audit"].matching) == {"cache"}
