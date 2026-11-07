"""Experiment-family specifications for the diagnostics (torch-free).

``legacy``: the original length-two-only experiments (39 test words, 3 slots).
``mixed``: the mixed-length 0/2/3/4/5 experiments (364 test words incl. EMPTY, 5 slots).
Words are LABELS everywhere ("EMPTY" for the empty word).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .. import operations as O
from ..words import EMPTY_LABEL, from_label


@dataclass
class DiagSpec:
    name: str
    experiments: List[str]
    words: List[str]                              # all evaluated words (labels), display order
    common_panel: List[str]                       # readable heatmap panel
    panel_groups: List[Tuple[str, int]]           # (title, count) blocks of the common panel
    group_order: List[str]
    group_labels: Dict[str, str]
    train_labels: Callable[[str], List[str]]      # experiment -> training words (labels)
    pair_words: Callable[[str], List[str]]        # experiment -> ORIGINAL length-two training pairs
    group_fn: Callable[[str, str], str]           # (experiment, label) -> group
    populations: Dict[str, List[str]]             # name -> groups (declared populations for summaries)
    extra_groups: Dict[str, Callable[[str], bool]] = field(default_factory=dict)
    paired: Optional[Tuple[str, str]] = None
    exp_labels: Dict[str, str] = field(default_factory=dict)
    controls_words: List[str] = field(default_factory=list)
    n_slots: int = 3


def _legacy_group(exp: str, s: str) -> str:
    if len(s) == 2:
        return "pairs_seen" if s in set(O.TRAIN_SETS[exp]) else "pairs_unseen"
    return O.string_group(s)


LEGACY = DiagSpec(
    name="legacy", experiments=["distinct_pairs", "all_pairs"], words=list(O.TEST_STRINGS),
    common_panel=list(O.TEST_STRINGS), panel_groups=[(O.GROUP_LABELS[g], O.EXPECTED_GROUP_COUNTS[g]) for g in O.GROUP_ORDER],
    group_order=["singles", "pairs_seen", "pairs_unseen", "triples_all_distinct", "triples_two_types", "triples_one_type"],
    group_labels={**O.GROUP_LABELS},
    train_labels=lambda e: list(O.TRAIN_SETS[e]), pair_words=lambda e: list(O.TRAIN_SETS[e]), group_fn=_legacy_group,
    populations={"unseen strings (all groups except seen pairs)":
                 ["singles", "pairs_unseen", "triples_all_distinct", "triples_two_types", "triples_one_type"],
                 "singles": ["singles"], "pairs_unseen": ["pairs_unseen"],
                 "triples (all 27)": ["triples_all_distinct", "triples_two_types", "triples_one_type"]},
    extra_groups={"triples_all": lambda s: len(s) == 3},
    paired=("distinct_pairs", "all_pairs"),
    exp_labels={"distinct_pairs": "Exp. 1 distinct_pairs", "all_pairs": "Exp. 2 all_pairs"},
    controls_words=list(O.TEST_STRINGS), n_slots=3)


def _mixed_spec() -> DiagSpec:
    from .. import mixed as M
    common = [EMPTY_LABEL] + list(O.TEST_STRINGS)
    return DiagSpec(
        name="mixed", experiments=list(M.MIXED_EXPERIMENTS), words=M.test_labels(), common_panel=common,
        panel_groups=[("EMPTY", 1), ("Singles", 3), ("Pairs", 9), ("Length 3 (all trained)", 27)],
        group_order=list(M.MIXED_GROUP_ORDER), group_labels=dict(M.MIXED_GROUP_LABELS),
        train_labels=M.training_labels, pair_words=M.pair_set, group_fn=lambda e, s: M.mixed_group(s, e),
        populations={"OOD: singles + unseen pairs": ["singles", "pairs_unseen"], "singles": ["singles"],
                     "pairs_unseen": ["pairs_unseen"], "EMPTY": ["EMPTY"], "len3_seen": ["len3_seen"],
                     "len4_seen": ["len4_seen"], "len5_seen": ["len5_seen"]},
        extra_groups={"single_A": lambda s: s == "A", "single_B": lambda s: s == "B", "single_C": lambda s: s == "C"},
        paired=tuple(M.MIXED_EXPERIMENTS),
        exp_labels={"mixed_02345_distinct_pairs": "Mixed 0/2-5, distinct pairs",
                    "mixed_02345_all_pairs": "Mixed 0/2-5, all pairs"},
        controls_words=common, n_slots=M.N_SLOTS)


def _compact_spec() -> DiagSpec:
    from .. import mixed as M
    common = [EMPTY_LABEL] + list(O.TEST_STRINGS)
    return DiagSpec(
        name="compact", experiments=list(M.COMPACT_EXPERIMENTS), words=M.test_labels(), common_panel=common,
        panel_groups=[("EMPTY", 1), ("Singles", 3), ("Pairs (all seen)", 9), ("Length 3 (all trained)", 27)],
        group_order=["EMPTY", "singles", "pairs_seen", "len3_seen", "len4_seen", "len5_seen"],
        group_labels=dict(M.MIXED_GROUP_LABELS, pairs_seen="Pairs (all 9 trained)"),
        train_labels=M.training_labels, pair_words=M.pair_set, group_fn=lambda e, s: M.mixed_group(s, e),
        populations={"singletons A/B/C (never trained)": ["singles"], "EMPTY": ["EMPTY"], "pairs_seen": ["pairs_seen"],
                     "len3_seen": ["len3_seen"]},
        extra_groups={"single_A": lambda s: s == "A", "single_B": lambda s: s == "B", "single_C": lambda s: s == "C"},
        paired=("compact_02345_rope", "compact_02345_sinusoidal"),
        exp_labels={"compact_02345_rope": "Compact, RoPE", "compact_02345_sinusoidal": "Compact, sinusoidal",
                    "compact_02345_learned": "Compact, learned 1D PE",
                    "compact_02345_cnn": "Compact, shared CNN controls (no PE switch)"},
        controls_words=common, n_slots=M.N_SLOTS)


def get_spec(name: str) -> DiagSpec:
    if name == "legacy":
        return LEGACY
    if name == "mixed":
        return _mixed_spec()
    if name == "compact":
        return _compact_spec()
    raise KeyError(name)


def spec_for_experiment(experiment: str) -> DiagSpec:
    from ..mixed import is_compact, is_mixed
    return get_spec("compact" if is_compact(experiment) else "mixed" if is_mixed(experiment) else "legacy")


def word(label: str) -> str:
    return from_label(label)
