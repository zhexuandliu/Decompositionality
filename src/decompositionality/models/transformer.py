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
from .common import (SEQUENCE_PE_CHOICES, BlockStack, GridModel, PreNormBlock, SequencePE, cell_ce_per_example,
                     count_parameters, gpt_init, key_padding_mask)


@dataclass
class TransformerConfig:
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    depth: int = 6           # number of block applications
    tied: bool = False       # True -> one shared block applied `depth` times
    dropout: float = 0.0
    n_slots: int = 3         # operation slots (legacy 3; mixed-length experiments 5)
    sequence_pe: str = "learned"   # 1D sequence PE: "learned" (legacy) | "rope" | "sinusoidal" (both nontrainable)
    pe_base: float = 10000.0
    identity_token: bool = False   # vocabulary includes the visible, learnable <identity> operation
    op_format: str = "fixed"       # "fixed" (exactly n_slots op tokens) | "variable" (0..n_slots real tokens, no PAD)
    row_sep: bool = False          # AR only: <ROW> separators between grid rows in input AND output (41-token grids)
    aux_vocab: bool = False        # vocabulary incl. the auxiliary-task tokens (D, E, <main>, <aux>; IDs up to 24)
    out_mode: str = "inplace"      # encoder only: "inplace" (bidirectional; answers read at the input cells) |
                                   # "causal" (36 output query positions after the input; the WHOLE sequence is
                                   # causal; all 36 answers predicted in parallel, nothing fed back)

    def __post_init__(self):
        assert self.sequence_pe in SEQUENCE_PE_CHOICES, self.sequence_pe
        assert self.op_format in ("fixed", "variable"), self.op_format
        assert self.out_mode in ("inplace", "causal"), self.out_mode


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
        self._layout_cls = T.ARRowLayout if cfg.row_sep else T.ARLayout
        self.layout = self._layout_cls(cfg.n_slots)
        L = self.layout.length
        if cfg.row_sep:
            assert cfg.identity_token, "row-separator models extend the identity vocabulary"
        V = T.VOCAB_SIZE_ROW if cfg.row_sep else (T.VOCAB_SIZE_WITH_IDENTITY if cfg.identity_token else T.VOCAB_SIZE)
        if cfg.aux_vocab:
            assert not cfg.row_sep, "aux vocabulary and ROW separators are not combined"
            V = T.VOCAB_SIZE_AUX
        self.tok_emb = nn.Embedding(V, d)
        if cfg.sequence_pe == "learned":          # legacy learned absolute 1D position table (max-size)
            self.pos_emb = nn.Embedding(L, d)
        self.seg_emb = nn.Embedding(T.NUM_SEGMENTS, d)
        self.row_emb = nn.Embedding(7, d)  # 6 = not a grid token
        self.col_emb = nn.Embedding(7, d)
        self.stack = BlockStack(cfg.depth, cfg.tied, lambda: PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)  # full vocabulary: non-digit outputs are possible (invalid)
        gpt_init(self)
        if cfg.sequence_pe != "learned":          # nontrainable; built after init so it consumes no RNG
            self.seq_pe = SequencePE(cfg.sequence_pe, L, d, d // cfg.n_heads, cfg.pe_base)
        meta = self.layout.metadata()
        for k, v in meta.items():
            self.register_buffer(f"meta_{k}", v, persistent=False)
        self.register_buffer("causal", torch.tril(torch.ones(L, L, dtype=torch.bool)), persistent=False)
        self._meta_cache = {}

    def layout_for(self, n_ops: int):
        return self.layout if n_ops == self.cfg.n_slots else self._layout_cls(n_ops)

    def _meta(self, n_ops: int, device):
        """Layout metadata (contiguous positions 0..L-1, segments, row/col) for an instruction of n_ops tokens."""
        if n_ops == self.cfg.n_slots:
            return {"pos": self.meta_pos, "seg": self.meta_seg, "row": self.meta_row, "col": self.meta_col}
        key = (n_ops, str(device))
        if key not in self._meta_cache:
            self._meta_cache[key] = {k: v.to(device) for k, v in self._layout_cls(n_ops).metadata().items()}
        return self._meta_cache[key]

    def forward_tokens(self, tokens: torch.Tensor, n_ops: int = None) -> torch.Tensor:
        """tokens [B, L] laid out per the layout of an ``n_ops``-token instruction (default: n_slots)
        -> logits [B, L, vocab]. Position metadata depends only on the index in that layout."""
        B, L = tokens.shape
        meta = self._meta(self.cfg.n_slots if n_ops is None else n_ops, tokens.device)
        cos = sin = None
        if self.cfg.sequence_pe == "learned":
            h = (self.tok_emb(tokens) + self.pos_emb(meta["pos"][:L]) + self.seg_emb(meta["seg"][:L])
                 + self.row_emb(meta["row"][:L]) + self.col_emb(meta["col"][:L]))
        else:
            pos = meta["pos"][:L]              # absolute layout positions, identical in training and decoding
            h = (self.tok_emb(tokens) + self.seg_emb(meta["seg"][:L])
                 + self.row_emb(meta["row"][:L]) + self.col_emb(meta["col"][:L]))
            add = self.seq_pe.additive(pos)
            if add is not None:
                h = h + add.to(h.dtype)
            cos, sin = self.seq_pe.rotary(pos)
        mask = self.causal[:L, :L][None, None] & key_padding_mask(tokens, T.PAD_ID)
        h = self.stack(h, mask, cos, sin)
        return self.head(self.ln_f(h))

    def _teacher_forced(self, batch):
        """THE single next-token shift: logits at index t predict token t+1.
        Returns logits/targets for the supervised positions: 36 answer cells + EOS (legacy), or 36 digits + 5 <ROW>
        + EOS = 42 (row-separator layout), in sequence order."""
        n = self.n_ops_of(batch["ops"])
        lay = self.layout_for(n)
        seq = lay.serialize(batch["ops"], batch["x"], batch["y"])
        logits = self.forward_tokens(seq[:, :-1], n)       # positions 0..len-2
        targets = seq[:, 1:]                               # tokens 1..len-1
        sl = slice(lay.idx_output, lay.idx_eos)            # logits OUTPUT..EOS-1 -> tokens first answer..EOS
        return logits[:, sl], targets[:, sl]

    def training_loss(self, batch):
        lg, tg = self._teacher_forced(batch)
        ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none").view(lg.shape[0], -1)
        loss = ce.mean(dim=1).mean()       # plain mean over ALL supervised targets (37 legacy, 42 with <ROW>)
        lay = self.layout
        dig = torch.as_tensor(lay.out_digit_offsets, device=ce.device)
        out = {"answer_cell_nll": ce[:, dig].mean().detach(), "eos_nll": ce[:, lay.n_out].mean().detach()}
        if lay.row_sep:
            out["row_nll"] = ce[:, torch.as_tensor(lay.out_row_offsets, device=ce.device)].mean().detach()
        return loss, out

    def val_cell_ce(self, batch):
        lg, tg = self._teacher_forced(batch)
        dig = torch.as_tensor(self.layout.out_digit_offsets, device=lg.device)
        return cell_ce_per_example(lg[:, dig], tg[:, dig])  # teacher-forced answer-DIGIT CE (no EOS / ROW)

    @torch.no_grad()
    def predict(self, batch, predict_eos: bool = True):
        """Greedy free-running decoding from the prefix ending at <OUTPUT>.
        The model never sees ground-truth answer tokens. Full-vocabulary argmax:
        any non-digit token among the 36 grid positions makes the grid invalid."""
        n = self.n_ops_of(batch["ops"])
        lay = self.layout_for(n)
        seq = lay.prefix(batch["ops"], batch["x"])        # the ACTUAL (possibly shorter) prefix ending at <OUTPUT>
        P0 = lay.prefix_len
        n_out = lay.n_out
        n_steps = n_out + (1 if predict_eos else 0)
        for _ in range(n_steps):      # fixed-length rollout; every generated token (incl. ROW/EOS) is fed back
            nxt = self.forward_tokens(seq, n)[:, -1].argmax(dim=-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        gen = seq[:, P0:P0 + n_out][:, torch.as_tensor(lay.out_digit_offsets, device=seq.device)]
        is_digit = gen < 10
        out = {"pred": torch.where(is_digit, gen, torch.full_like(gen, -1)), "valid": is_digit.all(dim=1),
               "raw_tokens": seq[:, P0:P0 + n_steps] if lay.row_sep else gen}
        if predict_eos:
            out["eos_ok"] = seq[:, P0 + n_out] == T.EOS_ID
        return out

    def cost_info(self):
        return {"block_applications_per_forward": self.cfg.depth, "core_calls_per_forward": self.cfg.depth,
                "input_tokens": f"{self.layout.length - 1} (teacher forcing); prompt {self.layout.prefix_len}",
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
        self.tok_emb = nn.Embedding(T.VOCAB_SIZE_AUX if cfg.aux_vocab else
                                    (T.VOCAB_SIZE_WITH_IDENTITY if cfg.identity_token else T.NONAR_VOCAB_SIZE), d)
        self.causal_out = cfg.out_mode == "causal"
        # 0 = operation slot, 1 = grid cell [, 2 = output query]
        self.type_emb = nn.Embedding(3 if self.causal_out else 2, d)
        n = cfg.n_slots
        if cfg.sequence_pe == "learned":
            self.slot_emb = nn.Embedding(n, d)   # legacy learned operation-slot positions
        self.row_emb = nn.Embedding(6, d)
        self.col_emb = nn.Embedding(6, d)
        self.stack = BlockStack(cfg.depth, cfg.tied, lambda: PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)  # shared per-cell classifier
        if self.causal_out:
            self.query_emb = nn.Embedding(1, d)   # one learned query vector (+ its output cell's row/col embeddings)
        gpt_init(self)
        if cfg.sequence_pe != "learned":
            self.seq_pe = SequencePE(cfg.sequence_pe, n + 36, d, d // cfg.n_heads, cfg.pe_base)
        self.register_buffer("seq_pos", torch.arange(n + 36), persistent=False)
        cells = torch.arange(36)
        self.register_buffer("rows", cells // 6, persistent=False)
        self.register_buffer("cols", cells % 6, persistent=False)
        self.register_buffer("slots", torch.arange(n), persistent=False)
        self.register_buffer("types", torch.tensor([0] * n + [1] * 36), persistent=False)

    def logits(self, batch) -> torch.Tensor:
        n = self.n_ops_of(batch["ops"])                               # actual op tokens (= n_slots unless variable)
        N = self.cfg.n_slots
        tokens = T.grid_token_inputs(batch["ops"], batch["x"])          # [B, n+36]
        types = torch.cat([self.types[:n], self.types[N:]])           # 0 for the n op tokens, 1 for the 36 cells
        seq_pos = self.seq_pos[:n + 36]
        cos = sin = None
        if self.cfg.sequence_pe == "learned":
            pos = torch.cat([self.slot_emb(self.slots[:n]), self.row_emb(self.rows) + self.col_emb(self.cols)], dim=0)
            h = self.tok_emb(tokens) + self.type_emb(types) + pos
        else:
            rc = self.row_emb(self.rows) + self.col_emb(self.cols)
            pos = torch.cat([rc.new_zeros(n, rc.shape[-1]), rc], dim=0)  # spatial row/col for grid tokens only
            h = self.tok_emb(tokens) + self.type_emb(types) + pos
            add = self.seq_pe.additive(seq_pos)
            if add is not None:
                h = h + add.to(h.dtype)
            cos, sin = self.seq_pe.rotary(seq_pos)
        if self.causal_out:
            # 36 output queries after [ops, input cells]; causal attention over the whole sequence; read the answers
            # at the queries (parallel, no feedback). Queries carry only their output cell's (row, col).
            B = tokens.shape[0]
            q = (self.query_emb.weight[0] + self.row_emb(self.rows) + self.col_emb(self.cols)
                 + self.type_emb.weight[2])                                 # [36, d]
            h = torch.cat([h, q[None].expand(B, -1, -1).to(h.dtype)], dim=1)  # [B, n+72, d]
            L = h.shape[1]
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=h.device))
            keys = torch.cat([key_padding_mask(tokens, T.PAD_ID)[:, 0, 0],
                              torch.ones(B, 36, dtype=torch.bool, device=h.device)], dim=1)
            h = self.stack(h, causal[None, None] & keys[:, None, None, :], cos, sin)
            return self.head(self.ln_f(h[:, n + 36:]))                     # [B,36,10] at the queries
        h = self.stack(h, key_padding_mask(tokens, T.PAD_ID), cos, sin)  # no causal mask
        return self.head(self.ln_f(h[:, n:]))                          # [B,36,10]

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
                "input_tokens": f"{self.cfg.n_slots + 36} ({self.cfg.n_slots} op slots + 36 cells)", "decoded_output_steps": 1,
                "decoding": "parallel argmax"}

    shared_core_parameters = TransformerAR.shared_core_parameters
    untied_equivalent_parameters = TransformerAR.untied_equivalent_parameters
