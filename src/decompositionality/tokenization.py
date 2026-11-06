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

# segments
SEG_BOS, SEG_OPS_MARK, SEG_OP_SLOT, SEG_GRID_MARK, SEG_IN, SEG_OUT_MARK, SEG_OUT, SEG_EOS = range(8)
NUM_SEGMENTS = 8
NO_COORD = 6  # row/col index for non-grid tokens


def op_slots_to_tokens(ops: torch.Tensor) -> torch.Tensor:
    return _OP_SLOT_TO_TOKEN.to(ops.device)[ops.long()]


def ar_layout_metadata() -> dict:
    seg = np.empty(AR_LEN, dtype=np.int64)
    row = np.full(AR_LEN, NO_COORD, dtype=np.int64)
    col = np.full(AR_LEN, NO_COORD, dtype=np.int64)
    seg[IDX_BOS] = SEG_BOS
    seg[IDX_OPS] = SEG_OPS_MARK
    seg[list(IDX_SLOTS)] = SEG_OP_SLOT
    seg[IDX_GRID] = SEG_GRID_MARK
    seg[IDX_IN_START:IDX_IN_START + 36] = SEG_IN
    seg[IDX_OUTPUT] = SEG_OUT_MARK
    seg[IDX_OUT_START:IDX_OUT_START + 36] = SEG_OUT
    seg[IDX_EOS] = SEG_EOS
    rc = np.arange(36)
    for start in (IDX_IN_START, IDX_OUT_START):
        row[start:start + 36] = rc // 6
        col[start:start + 36] = rc % 6
    return {"pos": torch.arange(AR_LEN), "seg": torch.from_numpy(seg), "row": torch.from_numpy(row),
            "col": torch.from_numpy(col)}


def ar_prefix(ops: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """[B,3] op slots + [B,6,6] grid -> [B,43] prefix ending at <OUTPUT> (no answer tokens)."""
    B = x.shape[0]
    dev = x.device
    cols = [
        torch.full((B, 1), BOS_ID, device=dev, dtype=torch.long),
        torch.full((B, 1), OPS_ID, device=dev, dtype=torch.long),
        op_slots_to_tokens(ops),
        torch.full((B, 1), GRID_ID, device=dev, dtype=torch.long),
        x.reshape(B, 36).long(),
        torch.full((B, 1), OUTPUT_ID, device=dev, dtype=torch.long),
    ]
    out = torch.cat(cols, dim=1)
    assert out.shape[1] == PREFIX_LEN
    return out


def ar_serialize(ops: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Full 80-token teacher-forcing sequence."""
    B = x.shape[0]
    out = torch.cat([ar_prefix(ops, x), y.reshape(B, 36).long(),
                     torch.full((B, 1), EOS_ID, device=x.device, dtype=torch.long)], dim=1)
    assert out.shape[1] == AR_LEN
    return out


def grid_token_inputs(ops: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Non-AR input tokens: 3 op-slot tokens followed by 36 row-major digit tokens -> [B,39]."""
    B = x.shape[0]
    return torch.cat([op_slots_to_tokens(ops), x.reshape(B, 36).long()], dim=1)
