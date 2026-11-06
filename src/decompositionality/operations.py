"""Grid operations, operation strings and evaluation string groups.

CONVENTION (important): operation strings execute LEFT TO RIGHT.
``"AB"`` means "first apply A, then apply B", i.e. ``Y = B(A(X))``;
``"ABC"`` means ``Y = C(B(A(X)))``.

Atomic operations on a 6x6 periodic grid with array coordinates ``r, c``:

* ``A = <translation>``: shift the whole grid RIGHT by one column with wrap-around,
  ``A(X)[r, c] = X[r, (c - 1) mod 6]``.
* ``B = <rotate>``: rotate the whole grid 90 degrees CLOCKWISE,
  ``B(X)[r, c] = X[5 - c, r]``.
* ``C = <reflection>``: mirror LEFT-TO-RIGHT (reverse column order),
  ``C(X)[r, c] = X[r, 5 - c]``.

These are discrete transformations of a periodic grid (a finite permutation group
acting on cells); they are *not* the continuous SE(2) group, and reflection is
included. The letters A/B/C are analysis aliases only; models receive the three
dedicated atomic operation IDs defined here.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

GRID_SIZE = 6
NUM_CELLS = GRID_SIZE * GRID_SIZE
NUM_DIGITS = 10
NUM_SLOTS = 3

# Atomic operation IDs (the model-facing vocabulary of operations).
OP_TRANSLATION = 0
OP_ROTATE = 1
OP_REFLECTION = 2
PAD_OP = 3  # distinct PAD id for empty operation slots; NOT an operation
NUM_OPS = 3

OP_TOKEN_NAMES = ("<translation>", "<rotate>", "<reflection>")
ALIASES = ("A", "B", "C")
ALIAS_TO_OP = {"A": OP_TRANSLATION, "B": OP_ROTATE, "C": OP_REFLECTION}
OP_TO_ALIAS = {v: k for k, v in ALIAS_TO_OP.items()}


def translate_right(x: np.ndarray) -> np.ndarray:
    """A: shift right by one column with wrap; works on [..., 6, 6]."""
    return np.roll(x, 1, axis=-1)


def rotate_clockwise(x: np.ndarray) -> np.ndarray:
    """B: rotate 90 degrees clockwise; works on [..., 6, 6]."""
    return np.ascontiguousarray(np.rot90(x, k=-1, axes=(-2, -1)))


def reflect_left_right(x: np.ndarray) -> np.ndarray:
    """C: reverse column order; works on [..., 6, 6]."""
    return np.ascontiguousarray(x[..., ::-1])


OP_FUNCTIONS = {
    OP_TRANSLATION: translate_right,
    OP_ROTATE: rotate_clockwise,
    OP_REFLECTION: reflect_left_right,
}


def parse_string(s: str) -> Tuple[int, ...]:
    """Convert an alias string such as ``"AB"`` into atomic op IDs (left to right)."""
    try:
        return tuple(ALIAS_TO_OP[ch] for ch in s)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid operation string {s!r}") from exc


def apply_string(x: np.ndarray, s: str) -> np.ndarray:
    """Apply an operation string LEFT TO RIGHT: apply_string(X, "AB") == B(A(X))."""
    y = np.asarray(x)
    for op in parse_string(s):
        y = OP_FUNCTIONS[op](y)
    return y


def apply_strings_batch(x: np.ndarray, strings: Sequence[str]) -> np.ndarray:
    """Apply per-example strings to a batch [N, 6, 6], vectorized by unique string."""
    x = np.asarray(x)
    out = np.empty_like(x)
    strings = np.asarray(strings)
    for s in np.unique(strings):
        idx = np.nonzero(strings == s)[0]
        out[idx] = apply_string(x[idx], str(s))
    return out


def slots_for_string(s: str, placement: str = "right") -> List[int]:
    """Place the (ordered) operations of ``s`` into three slots.

    ``placement="right"`` is canonical right padding used for validation/test:
    ``[A, PAD, PAD]``, ``[A, B, PAD]``, ``[A, B, C]``.
    """
    ops = list(parse_string(s))
    if not 1 <= len(ops) <= NUM_SLOTS:
        raise ValueError(f"string length must be 1..3, got {s!r}")
    if placement != "right":
        raise ValueError("only canonical right padding is defined here")
    return ops + [PAD_OP] * (NUM_SLOTS - len(ops))


# Order-preserving placements of a length-two string into three slots (training).
PAIR_PLACEMENTS = ((0, 1), (0, 2), (1, 2))  # [o1,o2,PAD], [o1,PAD,o2], [PAD,o1,o2]


def pair_slots(s: str, placement_index: int) -> List[int]:
    ops = parse_string(s)
    assert len(ops) == 2
    slots = [PAD_OP] * NUM_SLOTS
    a, b = PAIR_PLACEMENTS[placement_index]
    slots[a], slots[b] = ops
    return slots


def slots_to_string(slots: Sequence[int]) -> str:
    return "".join(OP_TO_ALIAS[int(o)] for o in slots if int(o) != PAD_OP)


# ---------------------------------------------------------------------------
# String sets
# ---------------------------------------------------------------------------

def all_strings(length: int) -> List[str]:
    return ["".join(p) for p in itertools.product(ALIASES, repeat=length)]


TRAIN_SETS: Dict[str, List[str]] = {
    "distinct_pairs": ["AB", "AC", "BA", "BC", "CA", "CB"],
    "all_pairs": ["AA", "AB", "AC", "BA", "BB", "BC", "CA", "CB", "CC"],
}

GROUP_ORDER = (
    "singles",
    "pairs",
    "triples_all_distinct",
    "triples_two_types",
    "triples_one_type",
)
GROUP_LABELS = {
    "singles": "Singles (length 1)",
    "pairs": "Pairs (length 2)",
    "pairs_seen": "Pairs: seen in training",
    "pairs_unseen": "Pairs: unseen in training",
    "triples_all_distinct": "Triples: 3 distinct ops",
    "triples_two_types": "Triples: 2 op types",
    "triples_one_type": "Triples: 1 op type",
    "triples_all": "Triples: all 27",
}


def string_group(s: str) -> str:
    n = len(s)
    if n == 1:
        return "singles"
    if n == 2:
        return "pairs"
    if n == 3:
        k = len(set(s))
        return {3: "triples_all_distinct", 2: "triples_two_types", 1: "triples_one_type"}[k]
    raise ValueError(s)


def test_strings() -> List[str]:
    """All 39 test strings, sorted by group then lexicographically."""
    candidates = all_strings(1) + all_strings(2) + all_strings(3)
    out: List[str] = []
    for g in GROUP_ORDER:
        out.extend(sorted(s for s in candidates if string_group(s) == g))
    return out


EXPECTED_GROUP_COUNTS = {
    "singles": 3,
    "pairs": 9,
    "triples_all_distinct": 6,
    "triples_two_types": 18,
    "triples_one_type": 3,
}


def check_test_string_counts() -> Dict[str, int]:
    strings = test_strings()
    counts = {g: sum(string_group(s) == g for s in strings) for g in GROUP_ORDER}
    assert counts == EXPECTED_GROUP_COUNTS, counts
    assert len(strings) == 39 and len(set(strings)) == 39
    assert set(all_strings(3)) <= set(strings)
    return counts


TEST_STRINGS: List[str] = test_strings()
check_test_string_counts()
