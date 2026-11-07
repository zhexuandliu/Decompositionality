"""Row-separator suite (notebook 18): 13 conditions x {Transformer AR, Looped AR} = 26 runs, all with learned absolute
sequence positions and a real <ROW> token between the rows of BOTH the input and the generated output grid.

  group               condition                    operation segment             train k        held-out tests
  fixed5              identity_s5_train02 ...      5 slots, visible <identity>  0,2 / 023 / 0234 / 02345   k=1 (all 5 slots) + longer k<=5
  variable_with_zero  variable_train02 ...         exactly k A/B/C tokens        same supports  k=1 + longer k<=5
  fixed6              identity_s6_train02345       6 slots, visible <identity>  0,2,3,4,5      k=1 (all 6 slots) + all 729 k=6
  variable_nozero     variable_nozero_train2 ...   exactly k tokens, no k=0      2 / 23 / 234 / 2345  k=1 (+ seen-support checks)

Logical records come from the shared per-k indexed stream ``idext.record(seed, k, j)`` (word, grid), with the rotating
near-balanced per-update allocation and batch order of ``idext``. Hence, for a given support, the S=5, variable and S=6
conditions see the IDENTICAL ordered (k, word, grid, target) stream; across supports (incl. with-zero vs nozero) records
at a common k coincide by within-k index. S=5 slot subsets are the historical idext subsets; S=6 subsets come from the
notebook-16 placement stream; variable conditions discard the layout. Both architectures read the same stream.
"""
from __future__ import annotations

import dataclasses
import hashlib
from math import comb
from typing import Dict, List

import numpy as np

from . import idext as E
from . import idslot as IS
from . import idvar as V
from .data import make_rng
from .words import to_label, words_of_length

ROWSEP_RESULTS_SUBDIR = "ar_row_separators_13conditions_v2"
ROWSEP_DATA_VERSION = "decomp-rowsep-v2"
ROW_LAYOUT_SPEC = ("rowsep-layout-v1: BOS OPS [m op tokens] GRID [r0 ROW r1 ROW r2 ROW r3 ROW r4 ROW r5] OUTPUT "
                   "[same 41] EOS; length 87+m; OUTPUT 44+m; digit(r,c) 45+m+7r+c; ROW_r 51+m+7r; EOS 86+m")
MODELS = ["transformer_ar", "looped_transformer_ar"]
POSITION_CAPACITY_SLOTS = 6      # 93-token table (m <= 6) shared by all 13 conditions
INIT_SEED_OFFSET = 211

# condition -> (group, layout, S, train ks)
CONDITIONS: Dict[str, tuple] = {}
for _sup, _ks in (("train02", (0, 2)), ("train023", (0, 2, 3)), ("train0234", (0, 2, 3, 4)), ("train02345", (0, 2, 3, 4, 5))):
    CONDITIONS[f"identity_s5_{_sup}"] = ("fixed5", "fixed", 5, _ks)
for _sup, _ks in (("train02", (0, 2)), ("train023", (0, 2, 3)), ("train0234", (0, 2, 3, 4)), ("train02345", (0, 2, 3, 4, 5))):
    CONDITIONS[f"variable_{_sup}"] = ("variable_with_zero", "variable", None, _ks)
CONDITIONS["identity_s6_train02345"] = ("fixed6", "fixed", 6, (0, 2, 3, 4, 5))
for _sup, _ks in (("train2", (2,)), ("train23", (2, 3)), ("train234", (2, 3, 4)), ("train2345", (2, 3, 4, 5))):
    CONDITIONS[f"variable_nozero_{_sup}"] = ("variable_nozero", "variable", None, _ks)
GROUPS = ["fixed5", "variable_with_zero", "fixed6", "variable_nozero"]
assert len(CONDITIONS) == 13
# Supplement (notebook 19): fixed S=2/3/4 with ROW. Kept OUT of CONDITIONS so the main 26-job registry, its selectors
# and checks are unchanged; every per-condition helper below accepts both dictionaries.
SUPPLEMENT_CONDITIONS: Dict[str, tuple] = {
    "identity_s2_train02_row": ("fixed_short", "fixed", 2, (0, 2)),
    "identity_s3_train023_row": ("fixed_short", "fixed", 3, (0, 2, 3)),
    "identity_s4_train0234_row": ("fixed_short", "fixed", 4, (0, 2, 3, 4)),
}
ALL_CONDITIONS: Dict[str, tuple] = {**CONDITIONS, **SUPPLEMENT_CONDITIONS}


def is_rowsep(experiment: str) -> bool:
    return experiment in ALL_CONDITIONS


def group_of(e):
    return ALL_CONDITIONS[e][0]


def layout_of(e):
    return ALL_CONDITIONS[e][1]


def n_slots(e):
    return ALL_CONDITIONS[e][2]


def train_ks(e):
    return ALL_CONDITIONS[e][3]


def _non5_fixed(e) -> bool:
    """Fixed layouts whose placements come from the notebook-16 slot streams (S=6 main; S=2/3/4 supplement)."""
    return layout_of(e) == "fixed" and n_slots(e) != 5


def support_name(e):
    return e.split("_")[-1]


def condition_format(e) -> str:
    """Batch 'condition' for serialization: fixed slots keep every <identity>; variable strips them."""
    return "identity" if layout_of(e) == "fixed" else "variable_length"


def training_words(e) -> List[str]:
    return [w for k in train_ks(e) for w in words_of_length(k)]


def training_labels(e) -> List[str]:
    return [to_label(w) for w in training_words(e)]


def select_jobs(run_group="all", conditions=None, models=None) -> List[tuple]:
    """Ordered (condition, model) jobs. Default: all 13 conditions x 2 models = 26."""
    if conditions:
        unknown = set(conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}")
        conds = [c for c in CONDITIONS if c in conditions]
    elif run_group == "all":
        conds = list(CONDITIONS)
    else:
        if run_group not in GROUPS:
            raise ValueError(f"RUN_GROUP must be 'all' or one of {GROUPS}")
        conds = [c for c in CONDITIONS if group_of(c) == run_group]
    ms = [m for m in MODELS if (models is None or m in models)]
    return [(c, m) for c in conds for m in ms]


# ---------------------------------------------------------------------------
# Streams and sets
# ---------------------------------------------------------------------------

def _layout_record(e, seed, k, j, word, slots5):
    lay = layout_of(e)
    if _non5_fixed(e):
        S = n_slots(e)
        return IS.slots_for(word, IS.subset(make_rng(seed, IS.STREAM_SLOT_TRAIN, S, k, j), S, k), S)
    return V.slots5_for(word, slots5)          # S=5 layout (variable conditions strip the fillers at serialization)


class RowStream:
    """Stateless: batch(step) depends only on (seed, step, B, condition, held-out grids)."""

    def __init__(self, seed: int, experiment: str, effective_batch_size: int, heldout_bytes=None):
        self.seed, self.e, self.B = int(seed), experiment, int(effective_batch_size)
        self.ks = train_ks(experiment)
        self.heldout = set(heldout_bytes or [])

    def batch(self, step: int) -> Dict[str, np.ndarray]:
        step = int(step)
        cnt = E.per_update_counts(self.ks, self.B, step)
        recs = []
        for k in self.ks:
            j0 = E.records_before(self.ks, self.B, step, k)
            for j in range(j0, j0 + cnt[k]):
                word, slots5, g = E.record(self.seed, k, j, self.heldout)
                recs.append((k, j, word, _layout_record(self.e, self.seed, k, j, word, slots5), g))
        recs = [recs[i] for i in make_rng(self.seed, E.STREAM_EXT_ORDER, step).permutation(len(recs))]
        ks = np.array([r[0] for r in recs])
        words = [r[2] for r in recs]
        x = np.stack([r[4] for r in recs]).astype(np.uint8)
        key = "slots" if _non5_fixed(self.e) else "slots5"
        return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), key: np.stack([r[3] for r in recs]),
                "k": ks, "group": ks, "length": ks, "string": np.array([to_label(w) for w in words]),
                "k_index": np.array([r[1] for r in recs]), "sample_id": np.array([f"k{r[0]}:{r[1]}" for r in recs]),
                "condition": condition_format(self.e)}


def make_val_set(val_grids: np.ndarray, seed: int, e: str) -> Dict[str, np.ndarray]:
    """300 records per TRAINING k (k=0 only where trained); balanced words; S=5 placements identical to the idext
    validation sets, S=6 identical to notebook 16; variable conditions use the same logical records."""
    n = len(val_grids)
    rows = []
    for k in train_ks(e):
        words = V._balanced_words(seed, k, n)
        if _non5_fixed(e):
            S = n_slots(e)
            rng = make_rng(seed, IS.STREAM_SLOT_VAL, S, k)
            lays = [IS.slots_for(words[g], IS.subset(rng, S, k), S) for g in range(n)]
        else:
            keys = make_rng(seed, E.STREAM_EXT_VALSLOT, k).random((n, 5))
            lays = [V.slots5_for(words[g], tuple(sorted(np.argsort(keys[g])[:k].tolist()))) for g in range(n)]
        for g in range(n):
            rows.append((g, k, words[g], lays[g]))
    gid = np.array([r[0] for r in rows])
    ks = np.array([r[1] for r in rows])
    words = [r[2] for r in rows]
    x = val_grids[gid].astype(np.uint8)
    key = "slots" if _non5_fixed(e) else "slots5"
    return {"x": x, "y": V.apply_words(x, words).astype(np.uint8), key: np.stack([r[3] for r in rows]), "k": ks,
            "group": ks, "length": ks, "grid_id": gid, "string": np.array([to_label(w) for w in words]),
            "condition": condition_format(e), "split": "val"}


def make_test_set(test_grids: np.ndarray, seed: int, e: str) -> Dict[str, np.ndarray]:
    """Exact historical evaluation manifests: S=5 = notebook-10 identity manifest (376,000 per 1000 grids);
    variable-with-zero = notebook-11 variable-length manifest (364,000); S=6 = notebook-16 manifest (1,108,000);
    variable-nozero = the variable manifest restricted to its trained k plus singletons."""
    lay, S = layout_of(e), n_slots(e)
    if S == 6:
        t = IS.make_test_set(test_grids, seed, "slots6_train02345")
    elif e in SUPPLEMENT_CONDITIONS:          # the notebook-16 S=2/3/4 manifests (same words / layouts / grids)
        t = IS.make_test_set(test_grids, seed, {2: "slots2_train02", 3: "slots3_train023", 4: "slots4_train0234"}[S])
    elif lay == "fixed":
        t = V.make_test_set(test_grids, seed, "identity")
    else:
        t = V.make_test_set(test_grids, seed, "variable_length")
        if group_of(e) == "variable_nozero":
            keep = np.isin(t["k"], list(train_ks(e)) + [1])
            t = {k_: (v[keep] if isinstance(v, np.ndarray) else v) for k_, v in t.items()}
    t["condition"] = condition_format(e)
    t["support"] = np.where(np.isin(t["k"], list(train_ks(e))), "train-supported", "HELD OUT")
    return t


def expected_test_records(e: str, n_grids: int) -> int:
    g, lay, S, ks = ALL_CONDITIONS[e]
    if g == "fixed_short":
        return n_grids * (sum(3 ** k for k in ks) + 3 * S)
    if S == 6:
        return n_grids * (361 + 18 + 729)
    if lay == "fixed":
        return n_grids * 376
    if g == "variable_with_zero":
        return n_grids * 364
    return n_grids * (sum(3 ** k for k in ks) + 3)


def required_heldout(e: str) -> Dict[int, int]:
    """Held-out k -> required records per grid in the primary manifest (completion checks reject missing sections)."""
    g, lay, S, ks = ALL_CONDITIONS[e]
    out = {1: 3 * (S if lay == "fixed" else 1)}
    if g in ("fixed5", "variable_with_zero"):
        for k in range(max(ks) + 1, 6):
            out[k] = 3 ** k
    if S == 6:
        out[6] = 729
    return out


def check_registry() -> dict:
    jobs = select_jobs()
    assert len(jobs) == 26
    tot = {m: sum(expected_test_records(c, 1000) for c in CONDITIONS) for m in MODELS}
    assert tot["transformer_ar"] == 4_602_000
    held = {c: sorted(required_heldout(c)) for c in CONDITIONS if group_of(c) == "fixed5"}
    assert list(held.values()) == [[1, 3, 4, 5], [1, 4, 5], [1, 5], [1]]
    for c in CONDITIONS:
        assert 1 not in train_ks(c)
        if group_of(c) == "variable_nozero":
            assert 0 not in train_ks(c)
    counts = {c: {k: E.records_before(train_ks(c), 256, 5000, k) for k in train_ks(c)} for c in CONDITIONS}
    for c, d in counts.items():
        assert sum(d.values()) == 1_280_000 and max(d.values()) - min(d.values()) <= 1
    return {"jobs": len(jobs), "records_per_arch": tot["transformer_ar"], "total_records": 2 * tot["transformer_ar"],
            "per_k_training_counts": counts}


# ---------------------------------------------------------------------------
# Model config and common initialization
# ---------------------------------------------------------------------------

def model_config(model_id: str, overrides=None) -> dict:
    from .models.registry import default_config
    c = dataclasses.asdict(default_config(model_id))
    c.update(overrides or {})
    c.update(n_slots=POSITION_CAPACITY_SLOTS, op_format="variable", identity_token=True, row_sep=True)
    return c


def legacy_random_config(model_id: str, overrides=None) -> dict:
    """The historical five-slot identity AR configuration (no ROW) whose seeded random init is preserved."""
    from .models.registry import default_config
    c = dataclasses.asdict(default_config(model_id))
    c.update(overrides or {})
    c.update(n_slots=5, identity_token=True)
    return c


def common_initial_state(model_id: str, seed: int, overrides=None):
    """ONE common fresh state for all 13 conditions of an architecture: the legacy seeded RANDOM init (no ROW),
    with only genuinely new rows (<ROW> embedding / output-head row, position rows 82..92) drawn from a separate
    generator. Never uses trained weights. Returns (state_dict, provenance)."""
    import torch
    from .models.registry import build_model
    src = build_model(model_id, legacy_random_config(model_id, overrides), seed=seed).state_dict()
    dst = build_model(model_id, model_config(model_id, overrides), seed=seed).state_dict()
    gen = torch.Generator().manual_seed(int(seed) * 1000 + INIT_SEED_OFFSET)
    out, grown = {}, {}
    for k in sorted(dst):
        v, s = dst[k], src.get(k)
        if s is not None and s.shape == v.shape:
            out[k] = s.clone()
            continue
        assert s is not None and v.shape[1:] == s.shape[1:] and v.shape[0] > s.shape[0], (k, v.shape)
        assert k in ("tok_emb.weight", "pos_emb.weight", "head.weight", "head.bias"), k
        n_new = v.shape[0] - s.shape[0]
        extra = (torch.zeros(n_new, *v.shape[1:]) if k.endswith("bias")
                 else torch.randn(n_new, *v.shape[1:], generator=gen) * 0.02)
        out[k] = torch.cat([s.clone(), extra], dim=0)
        grown[k] = f"{tuple(s.shape)} -> {tuple(v.shape)}"
    return out, {"grown": grown, "legacy_shared_sha": state_sha({k: v for k, v in src.items() if k not in grown}),
                 "full_sha": state_sha(out), "new_rows_generator": f"torch.Generator(seed*1000+{INIT_SEED_OFFSET})"}


def state_sha(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scoring from raw generated tokens (42 = 41 output-content tokens + 1 EOS step)
# ---------------------------------------------------------------------------

def score(raw: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    """Score by the predeclared target layout. Digits are read ONLY at the 36 expected digit positions (no filtering,
    no shifting, no repair); structural positions are scored separately."""
    from . import tokenization as T
    lay = T.ARRowLayout
    raw = np.asarray(raw)
    n = len(raw)
    Y = np.asarray(target).reshape(n, 36)
    dig = raw[:, lay.out_digit_offsets]
    is_digit = dig < 10
    cell = (dig == Y) & is_digit
    rows = raw[:, lay.out_row_offsets] == T.ROW_ID
    eos = raw[:, 41] == T.EOS_ID
    premature = (raw[:, :41] == T.EOS_ID).any(1)
    exact = cell.all(1)
    return {"pred": np.where(is_digit, dig, -1).astype(np.int8), "valid": is_digit.all(1), "cell_correct": cell,
            "exact": exact, "row_ok": rows, "all_rows_ok": rows.all(1), "eos_ok": eos, "premature_eos": premature,
            "strict": exact & rows.all(1) & eos, "nondigit_at_digit_pos": ~is_digit, "nonrow_at_row_pos": ~rows}
