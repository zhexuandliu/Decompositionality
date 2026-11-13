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

# Extra models accepted by experiment configs but NOT part of the default nine-model queue (so legacy notebooks'
# default queues are unchanged).
EXTRA_MODEL_IDS = ["causal_encoder", "looped_causal_encoder", "query_ar", "ar_slotpos", "ar_digithead",
                   "ar_slotpos_digithead", "ar_digiteos", "ar_slotpos_digiteos", "nexttoken_ar",
                   "ar_slotpos_digiteos_target", "ar_slotpos_digithead_target", "ar_digithead_target",
                   "ar_trace_slotstep", "ar_trace_sepstep"]

DISPLAY_NAMES = {
    "causal_encoder": "Causal encoder (parallel output queries)",
    "looped_causal_encoder": "Looped causal encoder (tied)",
    "query_ar": "Query AR (causal encoder + previous-digit feedback, decoded cell by cell)",
    "ar_slotpos": "Transformer AR without pos(t) (op-slot embedding instead)",
    "ar_digithead": "Transformer AR with a 10-way digit head (no EOS)",
    "ar_slotpos_digithead": "Transformer AR without pos(t) + 10-way digit head",
    "ar_digiteos": "Transformer AR with an 11-way head (digits + EOS)",
    "ar_slotpos_digiteos_target": "Transformer AR without pos(t) + 11-way head + target-cell coordinates",
    "ar_digithead_target": "Transformer AR (pos(t) kept) + 10-way head (36 digits) + target coords",
    "ar_slotpos_digithead_target": "Transformer AR without pos(t) + 10-way head (36 digits, forced EOS) + target coords",
    "ar_trace_slotstep": "Aligned AR + intermediate grids (step = op-slot embedding)",
    "ar_trace_sepstep": "Aligned AR + intermediate grids (separate step embedding)",
    "nexttoken_ar": "Query AR with ordinary next-token feedback (position holds cell j-1 with its own row/col)",
    "ar_slotpos_digiteos": "Transformer AR without pos(t) + 11-way head (digits + EOS)",
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
    "looped_causal_encoder": "causal_encoder",
}
