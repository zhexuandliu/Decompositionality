"""Attention-free conditional CNN baselines with FiLM conditioning (Perez et al., 2018)
and CoordConv-style coordinate channels (Liu et al., 2018).

Input: 10 one-hot digit channels + normalized row and column coordinates -> [B,12,6,6].
Ordered operation conditioning: three slot embeddings (PAD embedding fixed at zero) are
CONCATENATED (never averaged) and passed through a small MLP; per-block linear heads
produce FiLM scale/shift vectors broadcast over space:  h <- (1 + dgamma(c)) * h + beta(c),
zero-initialized so modulation starts as the identity. All spatial convolutions use
circular padding; normalization is GroupNorm (independent of microbatch size).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import GridModel, cell_ce_per_example


@dataclass
class OpConditionerConfig:
    op_emb_dim: int = 64
    hidden: int = 256


@dataclass
class ResCNNConfig:
    channels: int = 288
    n_blocks: int = 6
    groups: int = 8
    cond: OpConditionerConfig = None

    def __post_init__(self):
        if self.cond is None:
            self.cond = OpConditionerConfig()
        elif isinstance(self.cond, dict):
            self.cond = OpConditionerConfig(**self.cond)


@dataclass
class UNetConfig:
    c_high: int = 192         # channels at 6x6
    c_low: int = 384          # channels at 3x3
    enc_blocks: int = 2
    mid_blocks: int = 2
    dec_blocks: int = 2
    groups: int = 8
    cond: OpConditionerConfig = None

    def __post_init__(self):
        if self.cond is None:
            self.cond = OpConditionerConfig()
        elif isinstance(self.cond, dict):
            self.cond = OpConditionerConfig(**self.cond)


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, padding_mode="circular")


def grid_input(x: torch.Tensor, dtype) -> torch.Tensor:
    """[B,6,6] digits -> [B,12,6,6]: one-hot digits + row/col coordinates in [-1,1]."""
    B = x.shape[0]
    onehot = F.one_hot(x.long(), 10).permute(0, 3, 1, 2).to(dtype)
    lin = torch.linspace(-1, 1, 6, device=x.device, dtype=dtype)
    rows = lin.view(1, 1, 6, 1).expand(B, 1, 6, 6)
    cols = lin.view(1, 1, 1, 6).expand(B, 1, 6, 6)
    return torch.cat([onehot, rows, cols], dim=1)


class OpConditioner(nn.Module):
    """Ordered 3-slot operation encoder -> FiLM parameters for each conditioned block."""

    def __init__(self, cfg: OpConditionerConfig, film_channels: List[int]):
        super().__init__()
        self.emb = nn.Embedding(4, cfg.op_emb_dim, padding_idx=3)  # PAD -> fixed zero vector
        self.mlp = nn.Sequential(nn.Linear(3 * cfg.op_emb_dim, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.film_channels = list(film_channels)
        self.film = nn.Linear(cfg.hidden, 2 * sum(film_channels))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, ops: torch.Tensor):
        c = self.mlp(self.emb(ops.long()).flatten(1))   # concatenation preserves slot order
        params = self.film(c)
        out, i = [], 0
        for ch in self.film_channels:
            out.append((params[:, i:i + ch], params[:, i + ch:i + 2 * ch]))
            i += 2 * ch
        return out


class FiLMResBlock(nn.Module):
    """Pre-activation residual block: GN-GELU-conv-GN-FiLM-GELU-conv, plus identity."""

    def __init__(self, ch: int, groups: int):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, ch)
        self.c1 = conv3x3(ch, ch)
        self.n2 = nn.GroupNorm(groups, ch)
        self.c2 = conv3x3(ch, ch)

    def forward(self, h, film):
        dg, beta = film
        u = self.c1(F.gelu(self.n1(h)))
        u = (1 + dg[:, :, None, None]) * self.n2(u) + beta[:, :, None, None]
        return h + self.c2(F.gelu(u))


class _CNNBase(GridModel):
    kind = "parallel"
    loss_label = "parallel cell CE"

    def training_loss(self, batch):
        return cell_ce_per_example(self.logits(batch), batch["y"].reshape(-1, 36)).mean(), {}

    def val_cell_ce(self, batch):
        return cell_ce_per_example(self.logits(batch), batch["y"].reshape(-1, 36))

    @torch.no_grad()
    def predict(self, batch):
        pred = self.logits(batch).argmax(dim=-1)
        return {"pred": pred, "valid": torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)}

    @staticmethod
    def _to_cells(out):  # [B,10,6,6] -> [B,36,10] row-major
        return out.permute(0, 2, 3, 1).reshape(out.shape[0], 36, 10)


class ResCNNFiLM(_CNNBase):
    def __init__(self, cfg: ResCNNConfig, model_id: str = "rescnn_film"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        C = cfg.channels
        self.cond = OpConditioner(cfg.cond, [C] * cfg.n_blocks)
        self.stem = conv3x3(12, C)
        self.blocks = nn.ModuleList([FiLMResBlock(C, cfg.groups) for _ in range(cfg.n_blocks)])
        self.out_norm = nn.GroupNorm(cfg.groups, C)
        self.classifier = nn.Conv2d(C, 10, 1)

    def logits(self, batch):
        films = self.cond(batch["ops"])
        h = self.stem(grid_input(batch["x"], self.stem.weight.dtype))
        for blk, f in zip(self.blocks, films):
            h = blk(h, f)
        return self._to_cells(self.classifier(F.gelu(self.out_norm(h))))  # spatial 6x6 throughout

    def cost_info(self):
        return {"block_applications_per_forward": self.cfg.n_blocks, "core_calls_per_forward": self.cfg.n_blocks,
                "input_tokens": "6x6 grid, 12 channels", "decoded_output_steps": 1, "decoding": "parallel argmax"}


class UNetFiLM(_CNNBase):
    def __init__(self, cfg: UNetConfig, model_id: str = "unet_film"):
        super().__init__()
        self.cfg, self.model_id = cfg, model_id
        Ch, Cl, g = cfg.c_high, cfg.c_low, cfg.groups
        film_ch = [Ch] * cfg.enc_blocks + [Cl] * cfg.mid_blocks + [Ch] * cfg.dec_blocks
        self.cond = OpConditioner(cfg.cond, film_ch)
        self.stem = conv3x3(12, Ch)
        self.enc = nn.ModuleList([FiLMResBlock(Ch, g) for _ in range(cfg.enc_blocks)])
        self.down = conv3x3(Ch, Cl, stride=2)                         # 6x6 -> 3x3
        self.mid = nn.ModuleList([FiLMResBlock(Cl, g) for _ in range(cfg.mid_blocks)])
        self.up_conv = conv3x3(Cl, Ch)                                # after nearest 3x3 -> 6x6
        self.merge = nn.Conv2d(2 * Ch, Ch, 1)                         # skip connection fusion
        self.dec = nn.ModuleList([FiLMResBlock(Ch, g) for _ in range(cfg.dec_blocks)])
        self.out_norm = nn.GroupNorm(g, Ch)
        self.classifier = nn.Conv2d(Ch, 10, 1)

    def logits(self, batch):
        films = iter(self.cond(batch["ops"]))
        h = self.stem(grid_input(batch["x"], self.stem.weight.dtype))
        for blk in self.enc:
            h = blk(h, next(films))
        skip = h
        h = self.down(h)
        assert h.shape[-1] == 3
        for blk in self.mid:
            h = blk(h, next(films))
        h = self.up_conv(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.merge(torch.cat([h, skip], dim=1))
        for blk in self.dec:
            h = blk(h, next(films))
        return self._to_cells(self.classifier(F.gelu(self.out_norm(h))))

    def cost_info(self):
        c = self.cfg
        n = c.enc_blocks + c.mid_blocks + c.dec_blocks
        return {"block_applications_per_forward": n, "core_calls_per_forward": n,
                "input_tokens": "6x6 grid, 12 channels (one 3x3 level)", "decoded_output_steps": 1,
                "decoding": "parallel argmax"}
