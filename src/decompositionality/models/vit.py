"""Simple patch Transformer (ViT) on the categorical 6x6 grid with 2x2 patches, and its
weight-tied looped variant.

``looped_vit`` is OUR controlled weight-tied baseline ("Looped ViT (tied baseline)"):
the same ViT with one shared block applied six times. It is NOT the published hybrid
LoopViT system (no ConvGLU, local convolutions, test-time training or entropy-based
stopping).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import tokenization as T
from .common import BlockStack, GridModel, PreNormBlock, SequencePE, cell_ce_per_example, gpt_init, key_padding_mask
from .transformer import TransformerAR, TransformerConfig

PATCH = 2
NP = 3  # patches per side


def patchify(t: torch.Tensor) -> torch.Tensor:
    """[B,6,6,*F] -> [B,9,4,*F]; patches row-major over the 3x3 patch grid,
    cells within a patch row-major (dr, dc)."""
    B, rest = t.shape[0], t.shape[3:]
    t = t.reshape(B, NP, PATCH, NP, PATCH, *rest)          # B, pr, dr, pc, dc, ...
    t = t.permute(0, 1, 3, 2, 4, *range(5, 5 + len(rest)))  # B, pr, pc, dr, dc, ...
    return t.reshape(B, NP * NP, PATCH * PATCH, *rest)


def unpatchify(t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`patchify`: [B,9,4,*F] -> [B,6,6,*F]."""
    B, rest = t.shape[0], t.shape[3:]
    t = t.reshape(B, NP, NP, PATCH, PATCH, *rest)           # B, pr, pc, dr, dc, ...
    t = t.permute(0, 1, 3, 2, 4, *range(5, 5 + len(rest)))  # B, pr, dr, pc, dc, ...
    return t.reshape(B, 6, 6, *rest)


class ViT(GridModel):
    kind = "parallel"
    loss_label = "parallel cell CE"

    def __init__(self, cfg: TransformerConfig, model_id: str = "vit"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        d = cfg.d_model
        # 3 atomic ops + PAD (masked) [+ the visible, learnable <identity> op, slot value 4, when enabled]
        self.op_emb = nn.Embedding(5 if cfg.identity_token else 4, d)
        self.patch_proj = nn.Linear(PATCH * PATCH * 10, d)  # 2x2x10 one-hot, order preserved
        self.type_emb = nn.Embedding(2, d)
        if cfg.sequence_pe == "learned":
            self.slot_emb = nn.Embedding(cfg.n_slots, d)
        self.prow_emb = nn.Embedding(NP, d)
        self.pcol_emb = nn.Embedding(NP, d)
        self.stack = BlockStack(cfg.depth, cfg.tied, lambda: PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, PATCH * PATCH * 10)  # shared head: 4 cells x 10 logits per patch
        gpt_init(self)
        if cfg.sequence_pe != "learned":
            self.seq_pe = SequencePE(cfg.sequence_pe, cfg.n_slots + NP * NP, d, d // cfg.n_heads, cfg.pe_base)
        self.register_buffer("seq_pos", torch.arange(cfg.n_slots + NP * NP), persistent=False)
        p = torch.arange(NP * NP)
        self.register_buffer("prow", p // NP, persistent=False)
        self.register_buffer("pcol", p % NP, persistent=False)
        self.register_buffer("slots", torch.arange(cfg.n_slots), persistent=False)
        self.register_buffer("types", torch.tensor([0] * cfg.n_slots + [1] * 9), persistent=False)

    def logits(self, batch) -> torch.Tensor:
        ops, x = batch["ops"].long(), batch["x"].long()
        n = self.n_ops_of(ops)
        N = self.cfg.n_slots
        B = x.shape[0]
        types = torch.cat([self.types[:n], self.types[N:]])
        seq_pos = self.seq_pos[:n + NP * NP]
        onehot = F.one_hot(x, 10).to(self.patch_proj.weight.dtype)        # [B,6,6,10]
        patches = patchify(onehot).reshape(B, NP * NP, -1)                # [B,9,40]
        cos = sin = None
        h_ops = self.op_emb(ops) + self.slot_emb(self.slots[:n]) if self.cfg.sequence_pe == "learned" else self.op_emb(ops)
        h_patch = self.patch_proj(patches) + self.prow_emb(self.prow) + self.pcol_emb(self.pcol)
        h = torch.cat([h_ops, h_patch], dim=1) + self.type_emb(types)  # [B, n+9, d]
        if self.cfg.sequence_pe != "learned":
            add = self.seq_pe.additive(seq_pos)
            if add is not None:
                h = h + add.to(h.dtype)
            cos, sin = self.seq_pe.rotary(seq_pos)
        valid = torch.cat([ops != 3, torch.ones(B, NP * NP, dtype=torch.bool, device=x.device)], dim=1)
        h = self.stack(h, valid[:, None, None, :], cos, sin)
        out = self.head(self.ln_f(h[:, n:])).view(B, NP * NP, PATCH * PATCH, 10)
        return unpatchify(out).reshape(B, 36, 10)

    def training_loss(self, batch):
        return cell_ce_per_example(self.logits(batch), batch["y"].reshape(-1, 36)).mean(), {}

    def val_cell_ce(self, batch):
        return cell_ce_per_example(self.logits(batch), batch["y"].reshape(-1, 36))

    @torch.no_grad()
    def predict(self, batch):
        pred = self.logits(batch).argmax(dim=-1)
        return {"pred": pred, "valid": torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)}

    def cost_info(self):
        return {"block_applications_per_forward": self.cfg.depth, "core_calls_per_forward": self.cfg.depth,
                "input_tokens": f"{self.cfg.n_slots + 9} ({self.cfg.n_slots} op slots + 9 patches)", "decoded_output_steps": 1,
                "decoding": "parallel argmax"}

    shared_core_parameters = TransformerAR.shared_core_parameters
    untied_equivalent_parameters = TransformerAR.untied_equivalent_parameters
