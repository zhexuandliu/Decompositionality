"""Decoder-only autoregressive Transformer and bidirectional parallel-output encoder,
each with an untied (6 independent blocks) and a weight-tied looped variant
(one shared block applied 6 times)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import tokenization as T
from .common import (BlockStack, GridModel, PreNormBlock, cell_ce_per_example, count_parameters,
                     gpt_init, key_padding_mask)


@dataclass
class TransformerConfig:
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    depth: int = 6           # number of block applications
    tied: bool = False       # True -> one shared block applied `depth` times
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Autoregressive decoder
# ---------------------------------------------------------------------------

class TransformerAR(GridModel):
    kind = "ar"
    loss_label = "AR teacher-forced NLL (36 answer cells + EOS)"

    def __init__(self, cfg: TransformerConfig, model_id: str = "transformer_ar"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        d = cfg.d_model
        self.tok_emb = nn.Embedding(T.VOCAB_SIZE, d)
        self.pos_emb = nn.Embedding(T.AR_LEN, d)
        self.seg_emb = nn.Embedding(T.NUM_SEGMENTS, d)
        self.row_emb = nn.Embedding(7, d)  # 6 = not a grid token
        self.col_emb = nn.Embedding(7, d)
        self.stack = BlockStack(cfg.depth, cfg.tied, lambda: PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, T.VOCAB_SIZE)  # full vocabulary: non-digit outputs are possible (invalid)
        gpt_init(self)
        meta = T.ar_layout_metadata()
        for k, v in meta.items():
            self.register_buffer(f"meta_{k}", v, persistent=False)
        self.register_buffer("causal", torch.tril(torch.ones(T.AR_LEN, T.AR_LEN, dtype=torch.bool)), persistent=False)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, L<=80] laid out per the fixed layout -> logits [B, L, vocab].
        Position metadata depends only on the index in the layout."""
        B, L = tokens.shape
        h = (self.tok_emb(tokens) + self.pos_emb(self.meta_pos[:L]) + self.seg_emb(self.meta_seg[:L])
             + self.row_emb(self.meta_row[:L]) + self.col_emb(self.meta_col[:L]))
        mask = self.causal[:L, :L][None, None] & key_padding_mask(tokens, T.PAD_ID)
        h = self.stack(h, mask)
        return self.head(self.ln_f(h))

    def _teacher_forced(self, batch):
        """THE single next-token shift: logits at index t predict token t+1.
        Returns logits/targets for the 37 supervised positions (36 answer cells + EOS)."""
        seq = T.ar_serialize(batch["ops"], batch["x"], batch["y"])
        logits = self.forward_tokens(seq[:, :-1])          # positions 0..78
        targets = seq[:, 1:]                               # tokens 1..79
        sl = slice(T.IDX_OUTPUT, T.IDX_EOS)                # logits 42..78 -> tokens 43..79
        return logits[:, sl], targets[:, sl]

    def training_loss(self, batch):
        lg, tg = self._teacher_forced(batch)
        ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none").view(lg.shape[0], -1)
        loss = ce.mean(dim=1).mean()
        return loss, {"answer_cell_nll": ce[:, :36].mean().detach(), "eos_nll": ce[:, 36].mean().detach()}

    def val_cell_ce(self, batch):
        lg, tg = self._teacher_forced(batch)
        return cell_ce_per_example(lg[:, :36], tg[:, :36])  # teacher-forced answer-cell CE, EOS excluded

    @torch.no_grad()
    def predict(self, batch, predict_eos: bool = True):
        """Greedy free-running decoding from the prefix ending at <OUTPUT>.
        The model never sees ground-truth answer tokens. Full-vocabulary argmax:
        any non-digit token among the 36 grid positions makes the grid invalid."""
        seq = T.ar_prefix(batch["ops"], batch["x"])
        n_steps = 36 + (1 if predict_eos else 0)
        for _ in range(n_steps):
            nxt = self.forward_tokens(seq)[:, -1].argmax(dim=-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        gen = seq[:, T.PREFIX_LEN:T.PREFIX_LEN + 36]
        is_digit = gen < 10
        out = {"pred": torch.where(is_digit, gen, torch.full_like(gen, -1)), "valid": is_digit.all(dim=1),
               "raw_tokens": gen}
        if predict_eos:
            out["eos_ok"] = seq[:, T.PREFIX_LEN + 36] == T.EOS_ID
        return out

    def cost_info(self):
        return {"block_applications_per_forward": self.cfg.depth, "core_calls_per_forward": self.cfg.depth,
                "input_tokens": f"{T.AR_LEN - 1} (teacher forcing); prompt {T.PREFIX_LEN}",
                "decoded_output_steps": 36, "decoding": "greedy free-running (+1 EOS diagnostic step)"}

    def shared_core_parameters(self):
        return self.stack.block_parameters() if self.cfg.tied else None

    def untied_equivalent_parameters(self):
        if not self.cfg.tied:
            return None
        return count_parameters(self) + (self.cfg.depth - 1) * self.stack.block_parameters()


# ---------------------------------------------------------------------------
# Bidirectional encoder, parallel grid prediction
# ---------------------------------------------------------------------------

class TransformerEncoder(GridModel):
    kind = "parallel"
    loss_label = "parallel cell CE"

    def __init__(self, cfg: TransformerConfig, model_id: str = "transformer_encoder"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        d = cfg.d_model
        self.tok_emb = nn.Embedding(T.NONAR_VOCAB_SIZE, d)
        self.type_emb = nn.Embedding(2, d)   # 0 = operation slot, 1 = grid cell
        self.slot_emb = nn.Embedding(3, d)   # ordered operation-slot positions
        self.row_emb = nn.Embedding(6, d)
        self.col_emb = nn.Embedding(6, d)
        self.stack = BlockStack(cfg.depth, cfg.tied, lambda: PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)  # shared per-cell classifier
        gpt_init(self)
        cells = torch.arange(36)
        self.register_buffer("rows", cells // 6, persistent=False)
        self.register_buffer("cols", cells % 6, persistent=False)
        self.register_buffer("slots", torch.arange(3), persistent=False)
        self.register_buffer("types", torch.tensor([0] * 3 + [1] * 36), persistent=False)

    def logits(self, batch) -> torch.Tensor:
        tokens = T.grid_token_inputs(batch["ops"], batch["x"])          # [B,39]
        pos = torch.cat([self.slot_emb(self.slots), self.row_emb(self.rows) + self.col_emb(self.cols)], dim=0)
        h = self.tok_emb(tokens) + self.type_emb(self.types) + pos
        h = self.stack(h, key_padding_mask(tokens, T.PAD_ID))         # no causal mask
        return self.head(self.ln_f(h[:, 3:]))                          # [B,36,10]

    def training_loss(self, batch):
        loss = cell_ce_per_example(self.logits(batch), batch["y"].reshape(-1, 36)).mean()
        return loss, {}

    def val_cell_ce(self, batch):
        return cell_ce_per_example(self.logits(batch), batch["y"].reshape(-1, 36))

    @torch.no_grad()
    def predict(self, batch):
        pred = self.logits(batch).argmax(dim=-1)
        return {"pred": pred, "valid": torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)}

    def cost_info(self):
        return {"block_applications_per_forward": self.cfg.depth, "core_calls_per_forward": self.cfg.depth,
                "input_tokens": "39 (3 op slots + 36 cells)", "decoded_output_steps": 1,
                "decoding": "parallel argmax"}

    shared_core_parameters = TransformerAR.shared_core_parameters
    untied_equivalent_parameters = TransformerAR.untied_equivalent_parameters
