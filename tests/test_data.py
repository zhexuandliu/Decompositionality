import numpy as np

from decompositionality import data as D
from decompositionality import operations as O


def test_stream_determinism_strings_and_targets():
    s = D.TrainStream(42, O.TRAIN_SETS["distinct_pairs"], 64)
    a, b = s.batch(7), s.batch(7)
    for k in ("x", "y", "ops", "string", "placement"):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(s.batch(8)["x"], a["x"])
    assert set(a["string"]) <= set(O.TRAIN_SETS["distinct_pairs"])
    assert all(len(x) == 2 for x in a["string"])
    for i in range(64):
        assert np.array_equal(a["y"][i], O.apply_string(a["x"][i], a["string"][i]))
        assert O.slots_to_string(a["ops"][i]) == a["string"][i]
        assert (a["ops"][i] == O.PAD_OP).sum() == 1
    assert set(a["placement"]) == {0, 1, 2}


def test_balanced_schedule_and_same_grids_across_experiments():
    for exp in ("distinct_pairs", "all_pairs"):
        s = D.TrainStream(42, O.TRAIN_SETS[exp], 256)
        allstr = np.concatenate([s.batch(i)["string"] for i in range(13)])
        _, counts = np.unique(allstr, return_counts=True)
        assert len(counts) == len(O.TRAIN_SETS[exp])
        assert counts.max() - counts.min() <= 1
        for n in (5, 100, 1001):  # every prefix is balanced too
            _, c = np.unique(allstr[:n], return_counts=True)
            assert c.max() - c.min() <= 1
    s1 = D.TrainStream(42, O.TRAIN_SETS["distinct_pairs"], 256)
    s2 = D.TrainStream(42, O.TRAIN_SETS["all_pairs"], 256)
    assert np.array_equal(s1.batch(3)["x"], s2.batch(3)["x"])


def test_stream_independent_of_batching_order():
    s = D.TrainStream(42, O.TRAIN_SETS["all_pairs"], 32)
    seq = [s.batch(i) for i in range(5)]
    again = s.batch(3)  # random access == sequential access (resume equivalence)
    assert np.array_equal(again["x"], seq[3]["x"]) and np.array_equal(again["ops"], seq[3]["ops"])


def test_heldout_exclusion():
    s0 = D.TrainStream(42, O.TRAIN_SETS["distinct_pairs"], 16)
    g = s0.batch(2)["x"]
    held = {D.grid_bytes(g[3]), D.grid_bytes(g[9])}
    s = D.TrainStream(42, O.TRAIN_SETS["distinct_pairs"], 16, held)
    b = s.batch(2)
    assert not ({D.grid_bytes(x) for x in b["x"]} & held)
    assert np.array_equal(np.delete(b["x"], [3, 9], 0), np.delete(g, [3, 9], 0))
    assert np.array_equal(b["x"], s.batch(2)["x"])


def test_manifests(tmp_path):
    m = D.load_or_build_manifests(tmp_path, 42)
    assert m.val_grids.shape == (300, 6, 6) and m.test_grids.shape == (1000, 6, 6)
    D.validate_manifests(m)
    m2 = D.load_or_build_manifests(tmp_path, 42)  # reload + verify
    assert m2.manifest_hash == m.manifest_hash
    assert (tmp_path / "manifest.json").exists()
    # training stream never produces held-out grids (checked on a sample of batches)
    s = D.TrainStream(42, O.TRAIN_SETS["all_pairs"], 256, m.heldout_bytes)
    for i in range(20):
        assert not ({D.grid_bytes(x) for x in s.batch(i)["x"]} & m.heldout_bytes)


def test_eval_set_sizes():
    m = D.load_or_build_manifests(None, 42)
    v1 = D.make_eval_set(m.val_grids, O.TRAIN_SETS["distinct_pairs"], "val")
    v2 = D.make_eval_set(m.val_grids, O.TRAIN_SETS["all_pairs"], "val")
    t = D.make_eval_set(m.test_grids, O.TEST_STRINGS, "test")
    assert len(v1["x"]) == 1800 and len(v2["x"]) == 2700 and len(t["x"]) == 39000
    assert (t["ops"][t["string"] == "A"] == [0, 3, 3]).all()  # canonical right padding
    assert (t["ops"][t["string"] == "CBA"] == [2, 1, 0]).all()
    sub = D.subset_eval_set(v1, 32)
    assert len(sub["x"]) == 32 * 6 and sub["grid_id"].max() == 31
