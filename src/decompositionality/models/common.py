"""Shared model building blocks and the common model interface.

Every model consumes the same raw batch dict (torch tensors on the device)::

    {"ops": LongTensor[B, 3]  (0=<translation>, 1=<rotate>, 2=<reflection>, 3=PAD),
     "x":   LongTensor[B, 6, 6] digits 0..9,
     "y":   LongTensor[B, 6, 6] digits 0..9   (targets; only used by loss functions)}

and implements:

* ``training_loss(batch) -> (loss, metrics)``: mean per-example training objective.
* ``val_cell_ce(batch) -> Tensor[B]``: per-example final-answer cell cross-entropy
  (teacher-forced for AR models; final refinement for TRM).
* ``predict(batch) -> dict(pred=Long[B,36] (-1 = invalid), valid=Bool[B], ...)``
  deterministic greedy/argmax prediction that never receives ``y``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GridModel(nn.Module):
    model_id: str = "base"
    kind: str = "parallel"  # "ar" | "parallel" | "trm"
    loss_label: str = "cell CE"

    def __init__(self):
        super().__init__()
        self.activation_checkpointing = False

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def val_cell_ce(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def cost_info(self) -> dict:
        raise NotImplementedError

    def shared_core_parameters(self) -> Optional[int]:
        return None

    def untied_equivalent_parameters(self) -> Optional[int]:
        return None

    def dummy_batch(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(0)
        return {
            "ops": torch.randint(0, 3, (batch_size, 3), generator=g).to(device),
            "x": torch.randint(0, 10, (batch_size, 6, 6), generator=g).to(device),
            "y": torch.randint(0, 10, (batch_size, 6, 6), generator=g).to(device),
        }


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    seen, total = set(), 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def cell_ce_per_example(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logits [B,36,C], y [B,36] -> mean cell CE per example [B] (computed in fp32)."""
    B = logits.shape[0]
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1).long(), reduction="none")
    return ce.view(B, -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Standard pre-norm Transformer block (used by AR, encoder and ViT baselines)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """attn_mask: bool, broadcastable to [B, H, Lq, Lk]; True = may attend."""
        B, L, D = h.shape
        q, k, v = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class PreNormBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, bias=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = h + self.drop(self.attn(self.ln1(h), attn_mask))
        return h + self.drop(self.ffn(self.ln2(h)))


class BlockStack(nn.Module):
    """``depth`` block applications using ``depth`` independent blocks (untied) or one
    shared block applied ``depth`` times to the evolving hidden state (tied).
    No step embedding and no hidden-state reset between loops."""

    def __init__(self, depth: int, tied: bool, make_block):
        super().__init__()
        self.depth, self.tied = depth, tied
        self.blocks = nn.ModuleList([make_block() for _ in range(1 if tied else depth)])
        self.activation_checkpointing = False
        self.block_applications = 0  # counter of executed block applications (last forward)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        self.block_applications = 0
        for i in range(self.depth):
            blk = self.blocks[0 if self.tied else i]
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                h = checkpoint(blk, h, attn_mask, use_reentrant=False)
            else:
                h = blk(h, attn_mask)
            self.block_applications += 1
        return h

    def block_parameters(self) -> int:
        return count_parameters(self.blocks[0])


def gpt_init(module: nn.Module, std: float = 0.02) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def key_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B,L] -> bool [B,1,1,L], True for keys that may be attended."""
    return (tokens != pad_id)[:, None, None, :]
