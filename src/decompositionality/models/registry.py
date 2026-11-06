"""Stable model identifiers, explicit default configurations, and a builder."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

import torch

from .cnn import ResCNNConfig, ResCNNFiLM, UNetConfig, UNetFiLM
from .common import GridModel, count_parameters
from .transformer import TransformerAR, TransformerConfig, TransformerEncoder
from .trm import TRM, TRMConfig
from .vit import ViT

from ..names import DISPLAY_NAMES, MODEL_IDS, TIED_PAIRS  # noqa: F401  (re-exported)

_SPECS: Dict[str, tuple] = {
    "transformer_ar": (TransformerAR, lambda: TransformerConfig(tied=False)),
    "looped_transformer_ar": (TransformerAR, lambda: TransformerConfig(tied=True)),
    "transformer_encoder": (TransformerEncoder, lambda: TransformerConfig(tied=False)),
    "looped_transformer_encoder": (TransformerEncoder, lambda: TransformerConfig(tied=True)),
    "vit": (ViT, lambda: TransformerConfig(tied=False)),
    "looped_vit": (ViT, lambda: TransformerConfig(tied=True)),
    "trm": (TRM, TRMConfig),
    "rescnn_film": (ResCNNFiLM, ResCNNConfig),
    "unet_film": (UNetFiLM, UNetConfig),
}


def default_config(model_id: str):
    return _SPECS[model_id][1]()


def config_from_dict(model_id: str, d: Dict[str, Any]):
    cls = type(default_config(model_id))
    return cls(**d)


def build_model(model_id: str, config=None, seed: Optional[int] = None) -> GridModel:
    if model_id not in _SPECS:
        raise KeyError(f"unknown model id {model_id!r}; choose from {MODEL_IDS}")
    cls, _ = _SPECS[model_id]
    cfg = config if config is not None else default_config(model_id)
    if isinstance(cfg, dict):
        cfg = config_from_dict(model_id, cfg)
    if seed is not None:
        torch.manual_seed(seed)
    m = cls(cfg, model_id=model_id)
    return m


def config_dict(model: GridModel) -> dict:
    return dataclasses.asdict(model.cfg)


def model_summary(model_id: str, config=None) -> dict:
    """Parameter accounting on CPU (builds then discards the model)."""
    m = build_model(model_id, config)
    total = count_parameters(m)
    shared = m.shared_core_parameters()
    untied = m.untied_equivalent_parameters()
    info = {
        "model_id": model_id,
        "display_name": DISPLAY_NAMES[model_id],
        "kind": m.kind,
        "unique_params": total,
        "shared_core_params": shared,
        "untied_equivalent_params": untied,
        "loss_label": m.loss_label,
        **m.cost_info(),
        "config": config_dict(m),
    }
    del m
    return info
