"""TRM-adapted: a Tiny Recursive Model adaptation for the instruction-conditioned grid task.

Method reference: Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny
Networks" (arXiv:2510.04871); official code
https://github.com/SamsungSAILMontreal/TinyRecursiveModels (MIT license).
The recurrent core below re-implements, in plain PyTorch, the update structure of
``models/recursive_reasoning/trm.py`` in that repository:

* two carried states: answer ``y`` (official ``z_H``) and latent ``z`` (official ``z_L``),
  plus the input embedding ``x``;
* ONE shared network ``net`` (``L_level``: input injection by addition, then
  ``n_core_layers`` post-norm blocks: self-attention -> RMSNorm, SwiGLU -> RMSNorm,
  bias-free linears, weightless RMSNorm);
* ``latent_recursion``: ``n`` updates ``z <- net(z, y + x)``, then one ``y <- net(y, z)``;
* per supervised refinement: ``T-1`` latent_recursion cycles without gradients, one with
  gradients; decode ``y`` with the output head; then detach the carried ``(y, z)``;
* fixed (non-learned, seeded) initial-state buffers ``y_init``/``z_init`` broadcast to every
  position, reset for every new batch; embeddings scaled by sqrt(d) as in the official code.

Adaptations (documented, deliberate):

* instruction conditioning: inputs are 3 ordered operation slots + 36 grid cells
  (39 positions) with learned slot/row/column/type positions; PAD op slots are
  masked as attention keys. No puzzle-identifier embedding.
* scale: width 640, 8 heads, SwiGLU hidden 1728, 2-block core (~9.94M parameters).
* fixed compute: n=2, T=2, S=2 at both training and evaluation; NO adaptive halting
  (no Q/halting head, no ACT exploration).
* optimizer scheduling: all S refinements of a logical batch are compared with the SAME
  final target; their losses are averaged and ONE optimizer step is taken per logical
  batch (the official code takes an optimizer step after each supervision step and
  refills halted slots with new samples). Refinements are not counted as new samples.

Compute per forward example: S*T*(n+1) = 12 shared-core calls = 24 block applications,
of which S*(n+1) = 6 core calls (12 block applications) carry gradients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .. import tokenization as T
from .common import SEQUENCE_PE_CHOICES, GridModel, SequencePE, apply_rope, cell_ce_per_example, count_parameters


@dataclass
class TRMConfig:
    d_model: int = 640
    n_heads: int = 8
    d_ff: int = 1728            # SwiGLU hidden width
    n_core_layers: int = 2      # blocks in the shared core network
    n_latent: int = 2           # n: latent (z) updates per latent_recursion
    T_cycles: int = 2           # T: latent_recursion cycles per refinement (T-1 without grad)
    S_refinements: int = 2      # S: supervised answer refinements
    rms_eps: float = 1e-5
    n_slots: int = 3            # operation slots (legacy 3; mixed-length experiments 5)
    sequence_pe: str = "learned"   # "learned" (legacy) | "rope" | "sinusoidal" (both nontrainable)
    pe_base: float = 10000.0

    def __post_init__(self):
        assert self.sequence_pe in SEQUENCE_PE_CHOICES, self.sequence_pe


def trunc_normal_(t: torch.Tensor, std: float, generator=None) -> torch.Tensor:
    with torch.no_grad():
        t.normal_(0, 1, generator=generator).clamp_(-2, 2).mul_(std / 0.8796256610342398)
    return t


def rms_norm(h: torch.Tensor, eps: float) -> torch.Tensor:
    dt = h.dtype
    h = h.float()
    return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)).to(dt)


class Linear(nn.Module):
    """Bias-free linear with truncated LeCun-normal init (as the official CastedLinear)."""

    def __init__(self, fan_in: int, fan_out: int):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_(torch.empty(fan_out, fan_in), 1.0 / math.sqrt(fan_in)))

    def forward(self, x):
        return F.linear(x, self.weight)


class TRMBlock(nn.Module):
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads, self.head_dim, self.eps = cfg.n_heads, d // cfg.n_heads, cfg.rms_eps
        self.qkv = Linear(d, 3 * d)
        self.o = Linear(d, d)
        self.gate_up = Linear(d, 2 * cfg.d_ff)
        self.down = Linear(cfg.d_ff, d)

    def forward(self, h, attn_mask, cos=None, sin=None):
        B, L, D = h.shape
        q, k, v = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if cos is not None:  # RoPE on freshly projected Q/K at EVERY core call (never on the carried y/z states)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2).reshape(B, L, D)
        h = rms_norm(h + self.o(a), self.eps)                     # post-norm
        gate, up = self.gate_up(h).chunk(2, dim=-1)
        return rms_norm(h + self.down(F.silu(gate) * up), self.eps)


class TRM(GridModel):
    kind = "trm"
    loss_label = "TRM deep-supervision loss (mean cell CE over S refinements)"

    def __init__(self, cfg: TRMConfig, model_id: str = "trm"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        d = cfg.d_model
        std = 1.0 / math.sqrt(d)
        self.embed_scale = math.sqrt(d)
        self.tok_emb = nn.Parameter(trunc_normal_(torch.empty(T.NONAR_VOCAB_SIZE, d), std))
        self.type_emb = nn.Parameter(trunc_normal_(torch.empty(2, d), std))
        if cfg.sequence_pe == "learned":
            self.slot_emb = nn.Parameter(trunc_normal_(torch.empty(cfg.n_slots, d), std))
        self.row_emb = nn.Parameter(trunc_normal_(torch.empty(6, d), std))
        self.col_emb = nn.Parameter(trunc_normal_(torch.empty(6, d), std))
        self.core = nn.ModuleList([TRMBlock(cfg) for _ in range(cfg.n_core_layers)])
        self.head = Linear(d, 10)
        g = torch.Generator().manual_seed(12345)
        self.register_buffer("y_init", trunc_normal_(torch.empty(d), 1.0, g), persistent=True)
        self.register_buffer("z_init", trunc_normal_(torch.empty(d), 1.0, g), persistent=True)
        if cfg.sequence_pe != "learned":
            self.seq_pe = SequencePE(cfg.sequence_pe, cfg.n_slots + 36, d, d // cfg.n_heads, cfg.pe_base)
        self.register_buffer("seq_pos", torch.arange(cfg.n_slots + 36), persistent=False)
        cells = torch.arange(36)
        self.register_buffer("rows", cells // 6, persistent=False)
        self.register_buffer("cols", cells % 6, persistent=False)
        self.register_buffer("types", torch.tensor([0] * cfg.n_slots + [1] * 36), persistent=False)
        self.reset_counters()
        self._rope = None

    # -- bookkeeping ---------------------------------------------------------
    def reset_counters(self):
        self.core_calls = 0
        self.block_applications = 0
        self.grad_core_calls = 0
        self.grad_block_applications = 0

    def _embed(self, batch):
        assert batch["ops"].shape[1] == self.cfg.n_slots, f"expected {self.cfg.n_slots} operation slots"
        tokens = T.grid_token_inputs(batch["ops"], batch["x"])
        rc = self.row_emb[self.rows] + self.col_emb[self.cols]
        if self.cfg.sequence_pe == "learned":
            pos = torch.cat([self.slot_emb, rc], dim=0)
        else:
            pos = torch.cat([rc.new_zeros(self.cfg.n_slots, rc.shape[-1]), rc], dim=0)
        emb = self.tok_emb[tokens] + pos + self.type_emb[self.types]
        x = self.embed_scale * 0.7071067811865476 * emb
        if self.cfg.sequence_pe == "sinusoidal":
            x = x + self.seq_pe.additive(self.seq_pos).to(x.dtype)   # added ONCE to the input embedding x, scale 1
        mask = (tokens != T.PAD_ID)[:, None, None, :]
        return x, mask

    def _core_apply(self, h, mask):
        cos, sin = self._rope if self._rope is not None else (None, None)
        for blk in self.core:
            h = blk(h, mask, cos, sin)
        return h

    def net(self, h, injection, mask):
        """The single shared network: h <- core(h + injection)."""
        self.core_calls += 1
        self.block_applications += len(self.core)
        if torch.is_grad_enabled():
            self.grad_core_calls += 1
            self.grad_block_applications += len(self.core)
            if self.activation_checkpointing and self.training:
                return checkpoint(self._core_apply, h + injection, mask, use_reentrant=False)
        return self._core_apply(h + injection, mask)

    def latent_recursion(self, x, y, z, mask):
        for _ in range(self.cfg.n_latent):
            z = self.net(z, y + x, mask)   # z update conditioned on x, y, z
        y = self.net(y, z, mask)           # y update from y, z
        return y, z

    def refinement_logits(self, batch):
        """Returns the list of S answer logits [B,36,10] (one per supervised refinement).
        States are freshly initialized here for every call (no state carried across inputs)."""
        self.reset_counters()
        self._rope = self.seq_pe.rotary(self.seq_pos) if self.cfg.sequence_pe == "rope" else None
        x, mask = self._embed(batch)
        B, L, D = x.shape
        y = self.y_init.to(x.dtype).expand(B, L, D)
        z = self.z_init.to(x.dtype).expand(B, L, D)
        outs = []
        for _ in range(self.cfg.S_refinements):
            with torch.no_grad():
                for _ in range(self.cfg.T_cycles - 1):
                    y, z = self.latent_recursion(x, y, z, mask)
            y, z = self.latent_recursion(x, y, z, mask)
            outs.append(self.head(y[:, self.cfg.n_slots:]))          # differentiable decode BEFORE detaching
            y, z = y.detach(), z.detach()             # detach carried states between refinements
        return outs

    def training_loss(self, batch):
        yt = batch["y"].reshape(-1, 36)
        ces = [cell_ce_per_example(lg, yt).mean() for lg in self.refinement_logits(batch)]
        loss = torch.stack(ces).mean()
        metrics = {f"refinement_{i + 1}_ce": c.detach() for i, c in enumerate(ces)}
        metrics["final_refinement_ce"] = ces[-1].detach()
        return loss, metrics

    def val_cell_ce(self, batch):
        return cell_ce_per_example(self.refinement_logits(batch)[-1], batch["y"].reshape(-1, 36))

    @torch.no_grad()
    def predict(self, batch):
        pred = self.refinement_logits(batch)[-1].argmax(dim=-1)   # final refinement
        return {"pred": pred, "valid": torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)}

    def expected_calls(self):
        c = self.cfg
        calls = c.S_refinements * c.T_cycles * (c.n_latent + 1)
        grad_calls = c.S_refinements * (c.n_latent + 1)
        return {"core_calls": calls, "block_applications": calls * c.n_core_layers,
                "grad_core_calls": grad_calls, "grad_block_applications": grad_calls * c.n_core_layers}

    def cost_info(self):
        e = self.expected_calls()
        return {"block_applications_per_forward": e["block_applications"],
                "core_calls_per_forward": e["core_calls"],
                "grad_block_applications_per_forward": e["grad_block_applications"],
                "input_tokens": f"{self.cfg.n_slots + 36} ({self.cfg.n_slots} op slots + 36 cells)",
                "decoded_output_steps": 1,
                "decoding": f"parallel argmax of final refinement (S={self.cfg.S_refinements} refinements)"}

    def shared_core_parameters(self):
        return sum(count_parameters(b) for b in self.core)
