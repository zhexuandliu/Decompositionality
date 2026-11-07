"""Targeted checks for the row-separator suite (notebook 18)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from decompositionality import rowsep as RS
from decompositionality import rowsep_reports as RR
from decompositionality import tokenization as T
from decompositionality.models.registry import build_model

TINY = {"d_model": 32, "n_heads": 2, "d_ff": 64, "depth": 2}


def test_registry_counts_and_heldout_sections():
    r = RS.check_registry()
    assert r["jobs"] == 26 and r["records_per_arch"] == 4_602_000 and r["total_records"] == 9_204_000
    held = [sorted(RS.required_heldout(c)) for c in RS.CONDITIONS if RS.group_of(c) == "fixed5"]
    assert held == [[1, 3, 4, 5], [1, 4, 5], [1, 5], [1]]
    assert RS.required_heldout("identity_s5_train02") == {1: 15, 3: 27, 4: 81, 5: 243}
    assert RS.required_heldout("identity_s6_train02345") == {1: 18, 6: 729}
    assert RS.required_heldout("variable_nozero_train2") == {1: 3}
    assert len(RS.select_jobs("variable_nozero")) == 8 and len(RS.select_jobs("fixed6")) == 2


def test_layout_positions_round_trip():
    df = RR.layout_audit()
    assert list(df.length) == [87 + m for m in range(7)] and list(df.EOS) == [86 + m for m in range(7)]
    lay = T.ARRowLayout(5)
    assert lay.idx_output == 49 and lay.output_row_indices.tolist() == [56, 63, 70, 77, 84]
    assert T.ROW_ID not in (T.PAD_ID, T.IDENTITY_ID, T.EOS_ID) and T.VOCAB_SIZE_ROW == T.ROW_ID + 1


def test_common_init_preserves_legacy_random_state():
    for m in RS.MODELS:
        sd, prov = RS.common_initial_state(m, 42)
        legacy = build_model(m, RS.legacy_random_config(m), seed=42).state_dict()
        for k, v in legacy.items():
            if k in prov["grown"]:
                assert torch.equal(sd[k][: v.shape[0]], v)       # old rows/slices preserved
            else:
                assert torch.equal(sd[k], v)
        assert sd["tok_emb.weight"].shape[0] == 21 and sd["pos_emb.weight"].shape[0] == 93
        assert torch.equal(sd["head.bias"][20:], torch.zeros(1))
        sd2, _ = RS.common_initial_state(m, 42)
        assert RS.state_sha(sd) == RS.state_sha(sd2)


def _tiny(m="transformer_ar"):
    mod = build_model(m, RS.model_config(m, TINY), seed=0)
    return mod.eval()


@pytest.mark.parametrize("k", [0, 2, 5])
def test_teacher_forcing_targets_shift_and_causality(k):
    mod = _tiny()
    rng = np.random.default_rng(k)
    x = torch.as_tensor(rng.integers(0, 10, (3, 6, 6)))
    y = torch.as_tensor(rng.integers(0, 10, (3, 6, 6)))
    ops = torch.as_tensor(rng.integers(0, 3, (3, k)))          # variable: exactly k tokens (k=0 -> empty)
    lg, tg = mod._teacher_forced({"ops": ops, "x": x, "y": y})
    assert lg.shape[1] == 42
    lay = T.ARRowLayout(k)
    seq = lay.serialize(ops, x, y)
    assert torch.equal(tg, seq[:, lay.out_start:lay.idx_eos + 1])     # first target = first digit (from OUTPUT)
    assert (tg[:, T.ARRowLayout.out_row_offsets] == T.ROW_ID).all() and (tg[:, 41] == T.EOS_ID).all()
    assert torch.equal(tg[:, T.ARRowLayout.out_digit_offsets], y.reshape(3, 36))
    full = mod.forward_tokens(seq[:, :-1], k)
    seq2 = seq.clone()
    seq2[:, lay.out_start:] = T.ROW_ID                                   # change every answer token
    f2 = mod.forward_tokens(seq2[:, :-1], k)
    assert torch.allclose(full[:, :lay.idx_output + 1], f2[:, :lay.idx_output + 1], atol=1e-5)
    mod.train()
    loss, mets = mod.training_loss({"ops": ops, "x": x, "y": y})
    loss.backward()
    assert torch.isfinite(loss) and set(mets) == {"answer_cell_nll", "eos_nll", "row_nll"}
    out = mod.predict({"ops": ops, "x": x})
    assert out["raw_tokens"].shape == (3, 42) and out["pred"].shape == (3, 36)


def test_scoring_does_not_repair_and_perfect_predictor():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, (4, 36))
    perfect = T.ARRowLayout.grid_tokens(torch.as_tensor(y)).numpy()
    raw = np.concatenate([perfect, np.full((4, 1), T.EOS_ID)], 1)
    s = RS.score(raw, y)
    assert s["exact"].all() and s["strict"].all() and not s["premature_eos"].any() and s["nondigit_at_digit_pos"].sum() == 0
    bad = raw.copy()
    bad[0, 6] = 7                                   # wrong ROW only: digits still exact, strict fails
    bad[1, 10] = T.EOS_ID                           # premature EOS at a digit position: invalid, not exact
    bad[2] = np.concatenate([bad[2, 1:], [T.EOS_ID]])    # deleted first token: shifted digits are NOT realigned
    s = RS.score(bad, y)
    assert s["exact"][0] and not s["strict"][0] and not s["all_rows_ok"][0]
    assert not s["exact"][1] and s["premature_eos"][1] and not s["valid"][1]
    assert not s["exact"][2]
    assert s["exact"][3] and s["strict"][3]


def test_streams_and_validation_sets():
    df = RR.stream_audit(42, 64, steps=(0, 3))
    assert df.ok.all()
    a = RS.RowStream(42, "variable_train02", 64).batch(0)
    b = RS.RowStream(42, "variable_nozero_train2", 64).batch(0)
    assert not (b["k"] == 0).any() and (a["k"] == 0).sum() == 32
    a2 = a["string"][a["k"] == 2]
    # common per-k records by within-k index: the first k=2 records coincide across with-zero / nozero supports
    idx_a = a["k_index"][a["k"] == 2]
    idx_b = b["k_index"]
    common = np.intersect1d(idx_a, idx_b)
    assert len(common) and all(a2[idx_a == j][0] == b["string"][idx_b == j][0] for j in common)
    assert {"CC", "BBBB"} <= set(RS.training_labels("variable_nozero_train234"))
    grids = np.random.default_rng(0).integers(0, 10, (6, 6, 6))
    for c in RS.CONDITIONS:
        v = RS.make_val_set(grids, 42, c)
        assert set(np.unique(v["k"])) == set(RS.train_ks(c)) and len(v["x"]) == 6 * len(RS.train_ks(c))
        t = RS.make_test_set(grids, 42, c)
        assert len(t["x"]) == RS.expected_test_records(c, 6)
        comp = RR.completeness({"string": t["string"], "position": t["position"], "grid_id": t["grid_id"], "k": t["k"]}, c, 6)
        assert comp["complete"], (c, comp)
    t = RS.make_test_set(grids, 42, "identity_s5_train02")
    keep = t["k"] != 4                               # drop a longer-OOD section -> completeness must fail
    comp = RR.completeness({k: t[k][keep] for k in ("string", "position", "grid_id", "k")}, "identity_s5_train02", 6)
    assert not comp["complete"] and 4 in comp["missing_heldout_sections"]


def test_end_to_end_tiny(tmp_path):
    jobs = [("identity_s5_train02", "transformer_ar"), ("variable_train02", "looped_transformer_ar"),
            ("identity_s6_train02345", "transformer_ar"), ("variable_nozero_train2", "looped_transformer_ar")]
    over = {m: TINY for m in RS.MODELS}
    st = RR.run_queue(str(tmp_path), "smoke", 42, jobs, RR.STAGES, model_overrides=over, chunk_grids=1, log=lambda *a: None)
    sel = st[st.selected]
    assert sel.complete.all(), sel
    root = RR.family_root(str(tmp_path), "smoke")
    art = json.loads((root / "variable_train02" / "seed_42" / "eval" / "looped_transformer_ar" / "ARTIFACT.json").read_text())
    with np.load(art["predictions"]) as z:
        assert z["raw_tokens"].shape[1] == 42 and z["row_ok"].shape[1] == 5
    # interrupted evaluation: remove the artifact and one chunk -> rerun recomputes only what is missing
    ed = root / "identity_s5_train02" / "seed_42" / "eval" / "transformer_ar"
    (ed / "ARTIFACT.json").unlink()
    (ed / "chunks" / "chunk_001.npz").unlink()
    (ed / "analysis" / "DONE.json").unlink()
    st2 = RR.run_queue(str(tmp_path), "smoke", 42, jobs[:1], RR.STAGES, model_overrides=over, chunk_grids=1, log=lambda *a: None)
    assert st2[st2.selected].complete.all() and (ed / "chunks" / "chunk_001.npz").exists()
    rp = RR.consolidated(root, str(tmp_path), 42, jobs, show=False, log=lambda *a: None)
    import nbformat
    nb = nbformat.read(rp, as_version=4)
    outs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs]
    assert any("image/png" in o.data for o in outs) and any("text/html" in o.data for o in outs)
