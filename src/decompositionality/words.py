"""Instruction-word labels (torch-free).

Internally a word is a plain string over {A, B, C}; the EMPTY word is ``""`` (zero effective
operations, target = input grid). Saved tables, filenames and plots use the display LABEL ``"EMPTY"``
for it, because an empty string is read back as missing by CSV/Pandas loaders. The label is only
metadata: models never see it (they receive five PAD operation slots).
"""
from __future__ import annotations

import itertools
from typing import Iterable, List

EMPTY_LABEL = "EMPTY"


def to_label(word: str) -> str:
    return EMPTY_LABEL if word == "" else word


def from_label(label: str) -> str:
    label = str(label)
    if label == EMPTY_LABEL:
        return ""
    if label in ("", "nan", "None"):
        raise ValueError(f"ambiguous/missing word label {label!r}; the empty word must be written as {EMPTY_LABEL!r}")
    if any(ch not in "ABC" for ch in label):
        raise ValueError(f"invalid word label {label!r}")
    return label


def word_length(label: str) -> int:
    return len(from_label(label))


def words_of_length(n: int) -> List[str]:
    return ["".join(p) for p in itertools.product("ABC", repeat=n)]


def labels(words: Iterable[str]) -> List[str]:
    return [to_label(w) for w in words]
