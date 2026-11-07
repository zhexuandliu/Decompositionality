"""Explicit token vocabulary and the fixed autoregressive serialization layout.

Vocabulary (shared IDs; non-AR models only use IDs 0..13):

=========  =======================================
IDs 0-9    digit categories 0..9 (0 is a real value)
10         <translation>   (A)
11         <rotate>        (B)
12         <reflection>    (C)
13         <PAD>           empty operation slot (never a digit)
14         <BOS>
15         <OPS>
16         <GRID>
17         <OUTPUT>
18         <EOS>
=========  =======================================

AR layout (80 tokens, zero-based indices)::

    0 <BOS> | 1 <OPS> | 2-4 op slots | 5 <GRID> | 6-41 input x00..x55 |
    42 <OUTPUT> | 43-78 output y00..y55 | 79 <EOS>

The first answer digit (index 43) is predicted from the hidden state at <OUTPUT>
(index 42). Position metadata is a function of the layout only.
"""
from __future__ import annotations

import numpy as np
import torch

from . import operations as O

NUM_DIGIT_TOKENS = 10
OP_TOKEN_BASE = 10
PAD_ID = 13
BOS_ID, OPS_ID, GRID_ID, OUTPUT_ID, EOS_ID = 14, 15, 16, 17, 18
VOCAB_SIZE = 19
NONAR_VOCAB_SIZE = 14  # digits + 3 ops + PAD
TOKEN_NAMES = [str(i) for i in range(10)] + list(O.OP_TOKEN_NAMES) + ["<PAD>", "<BOS>", "<OPS>", "<GRID>", "<OUTPUT>", "<EOS>"]

# op-slot value (0,1,2 or PAD_OP=3) -> token ID
_OP_SLOT_TO_TOKEN = torch.tensor([OP_TOKEN_BASE + 0, OP_TOKEN_BASE + 1, OP_TOKEN_BASE + 2, PAD_ID])

AR_LEN = 80
IDX_BOS, IDX_OPS, IDX_SLOTS, IDX_GRID = 0, 1, (2, 3, 4), 5
IDX_IN_START = 6
IDX_OUTPUT = 42
IDX_OUT_START = 43
IDX_EOS = 79
PREFIX_LEN = 43  # tokens 0..42 inclusive (ends with <OUTPUT>)

class ARLayout:
    """Fixed AR serialization layout derived from the number of operation slots ``n_slots``::

        <BOS> <OPS> s_1..s_n <GRID> x00..x55 <OUTPUT> y00..y55 <EOS>

    n_slots=3 (legacy): 80 tokens, <OUTPUT> at 42, answers 43..78, <EOS> at 79.
    n_slots=5 (mixed-length experiments): 82 tokens, <OUTPUT> at 44, answers 45..80, <EOS> at 81.
    """

    def __init__(self, n_slots: int = 3):
        self.n_slots = int(n_slots)
        self.idx_bos, self.idx_ops = 0, 1
        self.idx_slots = tuple(range(2, 2 + self.n_slots))
        self.idx_grid = 2 + self.n_slots
        self.in_start = self.idx_grid + 1
        self.idx_output = self.in_start + 36
        self.out_start = self.idx_output + 1
        self.idx_eos = self.out_start + 36
        self.length = self.idx_eos + 1
        self.prefix_len = self.idx_output + 1

    def metadata(self) -> dict:
        L = self.length
        seg = np.empty(L, dtype=np.int64)
        row = np.full(L, NO_COORD, dtype=np.int64)
        col = np.full(L, NO_COORD, dtype=np.int64)
        seg[self.idx_bos] = SEG_BOS
        seg[self.idx_ops] = SEG_OPS_MARK
        seg[list(self.idx_slots)] = SEG_OP_SLOT
        seg[self.idx_grid] = SEG_GRID_MARK
        seg[self.in_start:self.in_start + 36] = SEG_IN
        seg[self.idx_output] = SEG_OUT_MARK
        seg[self.out_start:self.out_start + 36] = SEG_OUT
        seg[self.idx_eos] = SEG_EOS
        rc = np.arange(36)
        for start in (self.in_start, self.out_start):
            row[start:start + 36] = rc // 6
            col[start:start + 36] = rc % 6
        return {"pos": torch.arange(L), "seg": torch.from_numpy(seg), "row": torch.from_numpy(row),
                "col": torch.from_numpy(col)}

    def prefix(self, ops: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert ops.shape[1] == self.n_slots, f"expected {self.n_slots} operation slots, got {ops.shape[1]}"
        B, dev = x.shape[0], x.device
        out = torch.cat([
            torch.full((B, 1), BOS_ID, device=dev, dtype=torch.long),
            torch.full((B, 1), OPS_ID, device=dev, dtype=torch.long),
            op_slots_to_tokens(ops),
            torch.full((B, 1), GRID_ID, device=dev, dtype=torch.long),
            x.reshape(B, 36).long(),
            torch.full((B, 1), OUTPUT_ID, device=dev, dtype=torch.long)], dim=1)
        assert out.shape[1] == self.prefix_len
        return out

    def serialize(self, ops: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = torch.cat([self.prefix(ops, x), y.reshape(B, 36).long(),
                         torch.full((B, 1), EOS_ID, device=x.device, dtype=torch.long)], dim=1)
        assert out.shape[1] == self.length
        return out


# segments
SEG_BOS, SEG_OPS_MARK, SEG_OP_SLOT, SEG_GRID_MARK, SEG_IN, SEG_OUT_MARK, SEG_OUT, SEG_EOS = range(8)
NUM_SEGMENTS = 8
NO_COORD = 6  # row/col index for non-grid tokens


def op_slots_to_tokens(ops: torch.Tensor) -> torch.Tensor:
    return _OP_SLOT_TO_TOKEN.to(ops.device)[ops.long()]


_LEGACY = None


def _legacy() -> "ARLayout":
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = ARLayout(3)
    return _LEGACY


def ar_layout_metadata() -> dict:
    """Legacy 3-slot layout metadata."""
    return _legacy().metadata()


def ar_prefix(ops: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """[B,3] op slots + [B,6,6] grid -> [B,43] prefix ending at <OUTPUT> (legacy 3-slot layout)."""
    return _legacy().prefix(ops, x)


def ar_serialize(ops: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Full 80-token teacher-forcing sequence (legacy 3-slot layout)."""
    return _legacy().serialize(ops, x, y)


def grid_token_inputs(ops: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Non-AR input tokens: n_slots op-slot tokens followed by 36 row-major digit tokens -> [B, n_slots+36]."""
    B = x.shape[0]
    return torch.cat([op_slots_to_tokens(ops), x.reshape(B, 36).long()], dim=1)
