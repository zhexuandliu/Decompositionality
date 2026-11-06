"""Stable model identifiers and display names (torch-free, shared by training and analysis)."""

MODEL_IDS = [
    "transformer_ar",
    "looped_transformer_ar",
    "transformer_encoder",
    "looped_transformer_encoder",
    "vit",
    "looped_vit",
    "trm",
    "rescnn_film",
    "unet_film",
]

DISPLAY_NAMES = {
    "transformer_ar": "Transformer AR",
    "looped_transformer_ar": "Looped Transformer AR (tied)",
    "transformer_encoder": "Transformer encoder",
    "looped_transformer_encoder": "Looped Transformer encoder (tied)",
    "vit": "ViT (2x2 patches)",
    "looped_vit": "Looped ViT (tied baseline)",
    "trm": "TRM-adapted",
    "rescnn_film": "ResCNN-FiLM",
    "unet_film": "U-Net-FiLM",
}

# untied counterpart of each tied baseline (matched width and unrolled depth)
TIED_PAIRS = {
    "looped_transformer_ar": "transformer_ar",
    "looped_transformer_encoder": "transformer_encoder",
    "looped_vit": "vit",
}
