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
    ar_seq_pos: bool = True        # AR only. False = NO 1-D position table pos(t): op tokens get a learned op-SLOT
                                   # embedding instead, every other token keeps only segment + row/col (ablation)
    digit_head: bool = False       # AR only. True = 10-way digit output head, loss on the 36 answer digits only
                                   # (no EOS target), 36 decoding steps (ablation)
    trace: bool = False            # AR only (Aligned AR). Output = the grid after EVERY slot (S x 36 digits, last = the
                                   # answer); all trace digits are supervised; requires digit_head + target_coords
    step_emb: str = "slot"         # trace only: embedding of the trace step s added to its digits: "slot" = the op-slot
                                   # embedding of slot s (shared), "separate" = an independent learned step table
    aux3_vocab: bool = False       # AR only. Token table extended to the D/E/F auxiliary vocabulary (26 IDs); the
                                   # extra rows are appended AFTER the standard init from a separate generator, so all
                                   # other initial values equal the model without it (notebook 40)
    target_coords: bool = False    # AR only. True = in the output part, every position carries the (row, col) of the
                                   # cell it PREDICTS instead of its own token's: <OUTPUT> -> cell (0,0), answer token
                                   # of cell j-1 -> cell j, answer token of cell 35 (predicts EOS) -> no coordinate
    head_eos: bool = False         # AR only, with digit_head: 11-way head (digits 0-9 + EOS); loss on 36 digits + EOS
                                   # as in the original AR, 37 decoding steps (ablation)
    out_mode: str = "inplace"      # encoder only: "inplace" (bidirectional; answers read at the input cells) |
                                   # "causal" (36 output query positions after the input; the WHOLE sequence is
                                   # causal; all 36 answers predicted in parallel, nothing fed back) |
                                   # "causal_ar" (identical to "causal", except that output position j ALSO receives
                                   # the digit of cell j-1: the true one in training, its own prediction at test time;
                                   # decoded cell by cell) |
                                   # "causal_nexttoken" (as "causal_ar", but ordinary next-token AR at the output
                                   # positions: position j holds the digit of cell j-1 with cell j-1's OWN row/col and
                                   # no query vector; position 0 holds a start vector; position j predicts cell j)

    def __post_init__(self):
        assert self.sequence_pe in SEQUENCE_PE_CHOICES, self.sequence_pe
        assert self.op_format in ("fixed", "variable"), self.op_format
        assert self.out_mode in ("inplace", "causal", "causal_ar", "causal_nexttoken"), self.out_mode


# ---------------------------------------------------------------------------
# Autoregressive decoder
# ---------------------------------------------------------------------------

def trace_targets(ops: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """[B, S] op slots + [B, 6, 6] input -> [B, S x 36]: the grid after each slot (A/B/C applied, anything else kept)."""
    from ..mirror_diag import P_A, P_B, P_C
    dev = x.device
    perms = {0: torch.as_tensor(P_A, device=dev), 1: torch.as_tensor(P_B, device=dev), 2: torch.as_tensor(P_C, device=dev)}
    g = x.reshape(x.shape[0], 36).long()
    out = []
    for s in range(ops.shape[1]):
        o = ops[:, s].long()[:, None]
        g = torch.where(o == 0, g[:, perms[0]], torch.where(o == 1, g[:, perms[1]], torch.where(o == 2, g[:, perms[2]], g)))
        out.append(g)
    return torch.cat(out, dim=1)


class TransformerAR(GridModel):
    kind = "ar"
    loss_label = "AR teacher-forced NLL (36 answer cells + EOS)"

    def __init__(self, cfg: TransformerConfig, model_id: str = "transformer_ar"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        d = cfg.d_model
        self._layout_cls = T.ARTraceLayout if cfg.trace else (T.ARRowLayout if cfg.row_sep else T.ARLayout)
        self.layout = self._layout_cls(cfg.n_slots)
        L = self.layout.length
        L_pos = T.ARLayout(cfg.n_slots).length if cfg.trace else L   # trace: same table size as Aligned AR -> same init
        if cfg.row_sep:
            assert cfg.identity_token, "row-separator models extend the identity vocabulary"
        V = T.VOCAB_SIZE_ROW if cfg.row_sep else (T.VOCAB_SIZE_WITH_IDENTITY if cfg.identity_token else T.VOCAB_SIZE)
        if cfg.aux_vocab:
            assert not cfg.row_sep, "aux vocabulary and ROW separators are not combined"
            V = T.VOCAB_SIZE_AUX
        self.tok_emb = nn.Embedding(V, d)
        if cfg.sequence_pe == "learned":          # legacy learned absolute 1D position table (max-size)
            self.pos_emb = nn.Embedding(L_pos, d)
        self.seg_emb = nn.Embedding(T.NUM_SEGMENTS, d)
        self.row_emb = nn.Embedding(7, d)  # 6 = not a grid token
        self.col_emb = nn.Embedding(7, d)
        self.stack = BlockStack(cfg.depth, cfg.tied, lambda: PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)  # full vocabulary: non-digit outputs are possible (invalid)
        gpt_init(self)
        # Ablations. Everything is created exactly as in the original AR first, so every shared tensor has the
        # original initial values under the same seed; the replacements are then carved out of the original tensors.
        if not cfg.ar_seq_pos:
            assert cfg.sequence_pe == "learned" and not cfg.row_sep
            self.slot_emb = nn.Embedding(cfg.n_slots, d)          # init = the original pos(t) rows of the op slots
            with torch.no_grad():
                self.slot_emb.weight.copy_(self.pos_emb.weight[list(self.layout.idx_slots)])
            del self.pos_emb
        if cfg.digit_head:
            assert not cfg.row_sep
            rows = list(range(10)) + ([T.EOS_ID] if cfg.head_eos else [])
            head = nn.Linear(d, len(rows))                         # init = the original head's digit (+ EOS) rows
            with torch.no_grad():
                head.weight.copy_(self.head.weight[rows]); head.bias.copy_(self.head.bias[rows])
            self.head = head
            self.loss_label = ("AR teacher-forced NLL (36 answer digits + EOS; 11-way head)" if cfg.head_eos else
                               "AR teacher-forced NLL (36 answer digits; 10-way head, no EOS)")
        assert not cfg.head_eos or cfg.digit_head
        if cfg.aux3_vocab:
            assert cfg.digit_head and not cfg.aux_vocab
            old = self.tok_emb.weight.data
            g = torch.Generator().manual_seed(4040)
            extra = torch.randn(T.VOCAB_SIZE_AUX3 - old.shape[0], d, generator=g) * 0.02
            self.tok_emb = nn.Embedding(T.VOCAB_SIZE_AUX3, d)
            with torch.no_grad():
                self.tok_emb.weight.copy_(torch.cat([old, extra.to(old.dtype)], 0))
        if cfg.trace:
            assert cfg.digit_head and cfg.target_coords and not cfg.ar_seq_pos and not cfg.head_eos
            assert cfg.step_emb in ("slot", "separate"), cfg.step_emb
            self.loss_label = "AR teacher-forced NLL over the whole trace (S x 36 digits; 10-way head)"
            if cfg.step_emb == "separate":       # after the standard init, from its own generator
                g = torch.Generator().manual_seed(5050)
                self.step_table = nn.Embedding(cfg.n_slots, d)
                with torch.no_grad():
                    self.step_table.weight.copy_(torch.randn(cfg.n_slots, d, generator=g) * 0.02)
        if cfg.sequence_pe != "learned":          # nontrainable; built after init so it consumes no RNG
            self.seq_pe = SequencePE(cfg.sequence_pe, L, d, d // cfg.n_heads, cfg.pe_base)
        meta = self.layout.metadata()
        if cfg.target_coords:
            assert not cfg.row_sep
            meta = self._target_coord_meta(self.layout)
        self._meta_keys = list(meta)
        for k, v in meta.items():
            self.register_buffer(f"meta_{k}", v, persistent=False)
        self.register_buffer("causal", torch.tril(torch.ones(L, L, dtype=torch.bool)), persistent=False)
        self._meta_cache = {}

    @staticmethod
    def _target_coord_meta(lay):
        """Layout metadata with the output part's row/col (and, for a trace, step) shifted to the PREDICTED cell."""
        meta = lay.metadata()
        row, col = meta["row"].clone(), meta["col"].clone()
        n = lay.n_out                                              # 36, or S x 36 for a trace
        t = torch.arange(n)
        row[lay.idx_output:lay.idx_output + n] = (t % 36) // 6     # <OUTPUT> and output tokens 0..n-2 -> next cell
        col[lay.idx_output:lay.idx_output + n] = t % 6
        row[lay.out_start + n - 1] = T.NO_COORD                    # the last output token predicts EOS (unused)
        col[lay.out_start + n - 1] = T.NO_COORD
        out = {**meta, "row": row, "col": col}
        if "step" in meta:
            step = torch.full_like(meta["step"], -1)
            step[lay.idx_output:lay.idx_output + n] = t // 36
            step[lay.out_start + n - 1] = -1
            out["step"] = step
        return out

    def layout_for(self, n_ops: int):
        return self.layout if n_ops == self.cfg.n_slots else self._layout_cls(n_ops)

    def _meta(self, n_ops: int, device):
        """Layout metadata (contiguous positions 0..L-1, segments, row/col) for an instruction of n_ops tokens."""
        if n_ops == self.cfg.n_slots:
            return {k: getattr(self, f"meta_{k}") for k in self._meta_keys}
        key = (n_ops, str(device))
        if key not in self._meta_cache:
            lay = self._layout_cls(n_ops)
            meta = self._target_coord_meta(lay) if self.cfg.target_coords else lay.metadata()
            self._meta_cache[key] = {k: v.to(device) for k, v in meta.items()}
        return self._meta_cache[key]

    def forward_tokens(self, tokens: torch.Tensor, n_ops: int = None) -> torch.Tensor:
        """tokens [B, L] laid out per the layout of an ``n_ops``-token instruction (default: n_slots)
        -> logits [B, L, vocab]. Position metadata depends only on the index in that layout."""
        B, L = tokens.shape
        meta = self._meta(self.cfg.n_slots if n_ops is None else n_ops, tokens.device)
        cos = sin = None
        if self.cfg.sequence_pe == "learned" and not self.cfg.ar_seq_pos:     # no pos(t); op-slot embedding
            h = (self.tok_emb(tokens) + self.seg_emb(meta["seg"][:L])
                 + self.row_emb(meta["row"][:L]) + self.col_emb(meta["col"][:L]))
            sl = list(self.layout_for(self.cfg.n_slots if n_ops is None else n_ops).idx_slots)
            sl = [i for i in sl if i < L]
            h = h.clone()
            h[:, sl] = h[:, sl] + self.slot_emb.weight[: len(sl)].to(h.dtype)
            if "step" in meta:                     # trace: the step (slot) whose result this position predicts
                st = meta["step"][:L]
                idx = torch.nonzero(st >= 0).squeeze(1)
                table = self.slot_emb.weight if self.cfg.step_emb == "slot" else self.step_table.weight
                h[:, idx] = h[:, idx] + table[st[idx]].to(h.dtype)
        elif self.cfg.sequence_pe == "learned":
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
        y = trace_targets(batch["ops"], batch["x"]) if self.cfg.trace else batch["y"]
        seq = lay.serialize(batch["ops"], batch["x"], y)
        logits = self.forward_tokens(seq[:, :-1], n)       # positions 0..len-2
        targets = seq[:, 1:]                               # tokens 1..len-1
        sl = slice(lay.idx_output, lay.idx_eos)            # logits OUTPUT..EOS-1 -> tokens first answer..EOS
        return logits[:, sl], targets[:, sl]

    def training_loss(self, batch):
        lg, tg = self._teacher_forced(batch)
        if self.cfg.head_eos:                        # 36 digits + EOS; EOS token -> class 10
            tg = torch.where(tg == T.EOS_ID, torch.full_like(tg, 10), tg)
            ce = F.cross_entropy(lg.float().reshape(-1, 11), tg.reshape(-1), reduction="none").view(lg.shape[0], -1)
            return ce.mean(dim=1).mean(), {"answer_cell_nll": ce[:, :36].mean().detach(), "eos_nll": ce[:, 36].mean().detach()}
        if self.cfg.digit_head:                      # answer digits only (36, or the whole S x 36 trace); no EOS
            return cell_ce_per_example(lg[:, :-1], tg[:, :-1]).mean(), {}
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
        lay = self.layout_for(self.n_ops_of(batch["ops"]))
        dig = torch.as_tensor(lay.out_digit_offsets, device=lg.device)   # the (final) answer grid
        return cell_ce_per_example(lg[:, dig], tg[:, dig])  # teacher-forced answer-DIGIT CE (no EOS / ROW)

    @torch.no_grad()
    def predict(self, batch, predict_eos: bool = True):
        """Greedy free-running decoding from the prefix ending at <OUTPUT>.
        The model never sees ground-truth answer tokens. Full-vocabulary argmax:
        any non-digit token among the 36 grid positions makes the grid invalid."""
        n = self.n_ops_of(batch["ops"])
        lay = self.layout_for(n)
        seq = lay.prefix(batch["ops"], batch["x"])        # the ACTUAL (possibly shorter) prefix ending at <OUTPUT>
        if self.cfg.head_eos:                             # 37 steps over {0..9, EOS}; EOS inside the grid = invalid
            P0 = lay.prefix_len
            for _ in range(37):
                c = self.forward_tokens(seq, n)[:, -1].argmax(dim=-1)
                nxt = torch.where(c == 10, torch.full_like(c, T.EOS_ID), c)
                seq = torch.cat([seq, nxt[:, None]], dim=1)
            gen = seq[:, P0:P0 + 36]
            is_digit = gen < 10
            return {"pred": torch.where(is_digit, gen, torch.full_like(gen, -1)), "valid": is_digit.all(dim=1),
                    "raw_tokens": gen, "eos_ok": seq[:, P0 + 36] == T.EOS_ID}
        if self.cfg.digit_head:                           # 36 steps, digits only (always a valid grid); EOS is not
            P0 = lay.prefix_len                           # predicted: it is simply appended (forced) at the end
            for _ in range(lay.n_out):                    # 36, or S x 36 for a trace
                nxt = self.forward_tokens(seq, n)[:, -1].argmax(dim=-1)
                seq = torch.cat([seq, nxt[:, None]], dim=1)
            seq = torch.cat([seq, torch.full_like(seq[:, :1], T.EOS_ID)], dim=1)
            gen = seq[:, P0:P0 + lay.n_out][:, torch.as_tensor(lay.out_digit_offsets, device=seq.device)]
            return {"pred": gen, "valid": torch.ones(gen.shape[0], dtype=torch.bool, device=gen.device)}
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
        self.causal_out = cfg.out_mode in ("causal", "causal_ar", "causal_nexttoken")
        self.feedback = cfg.out_mode in ("causal_ar", "causal_nexttoken")
        self.nexttoken = cfg.out_mode == "causal_nexttoken"
        if self.feedback:
            self.kind = "ar_query"
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
        if self.feedback:
            # created AFTER gpt_init so every other tensor has exactly the causal-encoder initial values (same seed)
            self.start_emb = nn.Parameter(torch.randn(d) * 0.02)   # "no previous digit" for output position 0
        if self.nexttoken:
            del self.query_emb      # unused; it was created (and initialized) only so all other tensors match
        if cfg.sequence_pe != "learned":
            self.seq_pe = SequencePE(cfg.sequence_pe, n + 36, d, d // cfg.n_heads, cfg.pe_base)
        self.register_buffer("seq_pos", torch.arange(n + 36), persistent=False)
        cells = torch.arange(36)
        self.register_buffer("rows", cells // 6, persistent=False)
        self.register_buffer("cols", cells % 6, persistent=False)
        self.register_buffer("slots", torch.arange(n), persistent=False)
        self.register_buffer("types", torch.tensor([0] * n + [1] * 36), persistent=False)

    def logits(self, batch, prev=None) -> torch.Tensor:
        """``prev`` (causal_ar only): [B, 36] digit of cell j-1 fed to output position j (index 0 unused)."""
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
            if self.nexttoken:
                # ordinary next-token AR: position j = token "digit of cell j-1" with cell j-1's OWN (row, col);
                # position 0 = start vector (no row/col, like <OUTPUT>); the readout at position j predicts cell j
                assert prev is not None
                rc = self.row_emb(self.rows) + self.col_emb(self.cols)     # [36, d], cell order
                fb = self.tok_emb(prev.long())[:, 1:] + rc[None, :-1]       # positions 1..35 <- cells 0..34
                q = torch.cat([self.start_emb.expand(B, 1, -1), fb], dim=1) + self.type_emb.weight[2]
            else:
                q = (self.query_emb.weight[0] + self.row_emb(self.rows) + self.col_emb(self.cols)
                     + self.type_emb.weight[2])                             # [36, d]
                q = q[None].expand(B, -1, -1)
            if self.feedback and not self.nexttoken:                       # + digit of the PREVIOUS cell
                assert prev is not None
                fb = self.tok_emb(prev.long())                              # [B, 36, d]
                fb = torch.cat([self.start_emb.expand(B, 1, -1), fb[:, 1:]], dim=1)
                q = q + fb
            h = torch.cat([h, q.to(h.dtype)], dim=1)                         # [B, n+72, d]
            L = h.shape[1]
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=h.device))
            keys = torch.cat([key_padding_mask(tokens, T.PAD_ID)[:, 0, 0],
                              torch.ones(B, 36, dtype=torch.bool, device=h.device)], dim=1)
            h = self.stack(h, causal[None, None] & keys[:, None, None, :], cos, sin)
            return self.head(self.ln_f(h[:, n + 36:]))                     # [B,36,10] at the queries
        h = self.stack(h, key_padding_mask(tokens, T.PAD_ID), cos, sin)  # no causal mask
        return self.head(self.ln_f(h[:, n:]))                          # [B,36,10]

    def _teacher_prev(self, batch):
        """prev[:, j] = TRUE digit of cell j-1 (j >= 1); index 0 is replaced by the start vector."""
        y = batch["y"].reshape(-1, 36).long()
        return torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)

    def training_loss(self, batch):
        lg = self.logits(batch, self._teacher_prev(batch) if self.feedback else None)
        loss = cell_ce_per_example(lg, batch["y"].reshape(-1, 36)).mean()
        return loss, {}

    def val_cell_ce(self, batch):
        lg = self.logits(batch, self._teacher_prev(batch) if self.feedback else None)
        return cell_ce_per_example(lg, batch["y"].reshape(-1, 36))   # teacher-forced for causal_ar

    @torch.no_grad()
    def predict(self, batch):
        if not self.feedback:
            pred = self.logits(batch).argmax(dim=-1)
        else:                                       # greedy, cell by cell; position j sees its own previous outputs
            B = batch["x"].shape[0]
            prev = torch.zeros(B, 36, dtype=torch.long, device=batch["x"].device)
            pred = torch.zeros_like(prev)
            for j in range(36):
                pj = self.logits(batch, prev)[:, j].argmax(dim=-1)
                pred[:, j] = pj
                if j + 1 < 36:
                    prev[:, j + 1] = pj
        return {"pred": pred, "valid": torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)}

    def cost_info(self):
        return {"block_applications_per_forward": self.cfg.depth, "core_calls_per_forward": self.cfg.depth,
                "input_tokens": f"{self.cfg.n_slots + 36} ({self.cfg.n_slots} op slots + 36 cells)", "decoded_output_steps": 1,
                "decoding": "parallel argmax"}

    shared_core_parameters = TransformerAR.shared_core_parameters
    untied_equivalent_parameters = TransformerAR.untied_equivalent_parameters
