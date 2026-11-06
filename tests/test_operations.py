import itertools

import numpy as np

from decompositionality import operations as O

LAB = np.arange(36).reshape(6, 6)


def ref_A(x):  # independent reference implementations by explicit coordinates
    return np.array([[x[r][(c - 1) % 6] for c in range(6)] for r in range(6)])


def ref_B(x):
    return np.array([[x[5 - c][r] for c in range(6)] for r in range(6)])


def ref_C(x):
    return np.array([[x[r][5 - c] for c in range(6)] for r in range(6)])


REF = {"A": ref_A, "B": ref_B, "C": ref_C}


def test_manual_examples():
    a = O.translate_right(LAB)
    assert a[0].tolist() == [5, 0, 1, 2, 3, 4]            # shift right, last column wraps to first
    assert a[:, 0].tolist() == LAB[:, 5].tolist()
    b = O.rotate_clockwise(LAB)
    assert b[0].tolist() == [30, 24, 18, 12, 6, 0]        # clockwise: first row = old first column, bottom-up
    assert b[0, 5] == LAB[0, 0] and b[5, 5] == LAB[0, 5]  # top-left -> top-right, top-right -> bottom-right
    c = O.reflect_left_right(LAB)
    assert c[0].tolist() == [5, 4, 3, 2, 1, 0]
    assert c[3].tolist() == LAB[3][::-1].tolist()


def test_reference_agreement_and_orders():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, (6, 6))
    for k, f in REF.items():
        assert np.array_equal(O.apply_string(x, k), f(x))
    def power(s, n):
        y = x
        for _ in range(n):
            y = O.apply_string(y, s)
        return y
    assert np.array_equal(power("A", 6), x) and not np.array_equal(power("A", 3), x)
    assert np.array_equal(power("B", 4), x) and not np.array_equal(power("B", 2), x)
    assert np.array_equal(power("C", 2), x) and not np.array_equal(power("C", 1), x)


def test_left_to_right_composition():
    assert not np.array_equal(O.apply_string(LAB, "AB"), O.apply_string(LAB, "BA"))  # witness
    rng = np.random.default_rng(1)
    for s in O.all_strings(1) + O.all_strings(2) + O.all_strings(3):
        x = rng.integers(0, 10, (6, 6))
        y = x
        for ch in s:  # left to right: AB means B(A(X))
            y = REF[ch](y)
        assert np.array_equal(O.apply_string(x, s), y), s
    assert np.array_equal(O.apply_string(LAB, "ABC"), ref_C(ref_B(ref_A(LAB))))


def test_batch_apply():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 10, (20, 6, 6))
    strs = [O.TEST_STRINGS[i % 39] for i in range(20)]
    out = O.apply_strings_batch(x, strs)
    for i in range(20):
        assert np.array_equal(out[i], O.apply_string(x[i], strs[i]))


def test_string_sets_and_groups():
    counts = O.check_test_string_counts()
    assert counts == {"singles": 3, "pairs": 9, "triples_all_distinct": 6, "triples_two_types": 18, "triples_one_type": 3}
    assert O.TRAIN_SETS["distinct_pairs"] == ["AB", "AC", "BA", "BC", "CA", "CB"]
    assert O.TRAIN_SETS["all_pairs"] == ["AA", "AB", "AC", "BA", "BB", "BC", "CA", "CB", "CC"]
    assert set(O.TEST_STRINGS) == {"".join(p) for n in (1, 2, 3) for p in itertools.product("ABC", repeat=n)}


def test_slots():
    assert O.slots_for_string("A") == [0, O.PAD_OP, O.PAD_OP]
    assert O.slots_for_string("AB") == [0, 1, O.PAD_OP]
    assert O.slots_for_string("ABC") == [0, 1, 2]
    assert [O.pair_slots("BC", i) for i in range(3)] == [[1, 2, 3], [1, 3, 2], [3, 1, 2]]
    for i in range(3):
        assert O.slots_to_string(O.pair_slots("CA", i)) == "CA"  # order preserved
