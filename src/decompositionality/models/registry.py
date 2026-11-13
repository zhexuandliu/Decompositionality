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
    "causal_encoder": (TransformerEncoder, lambda: TransformerConfig(tied=False, out_mode="causal")),
    "looped_causal_encoder": (TransformerEncoder, lambda: TransformerConfig(tied=True, out_mode="causal")),
    "query_ar": (TransformerEncoder, lambda: TransformerConfig(tied=False, out_mode="causal_ar")),
    "ar_slotpos_digiteos_target": (TransformerAR, lambda: TransformerConfig(tied=False, ar_seq_pos=False, digit_head=True,
                                                                            head_eos=True, target_coords=True)),
    "ar_digithead_target": (TransformerAR, lambda: TransformerConfig(tied=False, digit_head=True, target_coords=True)),
    "ar_slotpos_digithead_target": (TransformerAR, lambda: TransformerConfig(tied=False, ar_seq_pos=False,
                                                                             digit_head=True, target_coords=True)),
    "nexttoken_ar": (TransformerEncoder, lambda: TransformerConfig(tied=False, out_mode="causal_nexttoken")),
    # ablations of the original Transformer AR (notebook 31)
    "ar_slotpos": (TransformerAR, lambda: TransformerConfig(tied=False, ar_seq_pos=False)),
    "ar_digithead": (TransformerAR, lambda: TransformerConfig(tied=False, digit_head=True)),
    "ar_slotpos_digithead": (TransformerAR, lambda: TransformerConfig(tied=False, ar_seq_pos=False, digit_head=True)),
    "ar_digiteos": (TransformerAR, lambda: TransformerConfig(tied=False, digit_head=True, head_eos=True)),
    "ar_slotpos_digiteos": (TransformerAR, lambda: TransformerConfig(tied=False, ar_seq_pos=False, digit_head=True,
                                                                     head_eos=True)),
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
    """Model config as a dict. ``n_slots`` is omitted when it equals the legacy default (3) so legacy
    config hashes -- and therefore resume/skip of the completed legacy runs -- are unchanged."""
    d = dataclasses.asdict(model.cfg)
    if d.get("n_slots") == 3:
        d.pop("n_slots")
    if d.get("sequence_pe") == "learned":      # legacy default: omit so legacy hashes are unchanged
        d.pop("sequence_pe")
        d.pop("pe_base", None)
    for key, legacy in (("identity_token", False), ("op_format", "fixed"), ("conditioner", "concat"), ("row_sep", False),
                         ("out_mode", "inplace"), ("aux_vocab", False), ("ar_seq_pos", True),
                         ("digit_head", False), ("head_eos", False), ("target_coords", False),
                         ("aux3_vocab", False)):
        if d.get(key) == legacy:
            d.pop(key)
    return d


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
