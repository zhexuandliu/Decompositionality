import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

# Tiny architecture fixtures (fast unit tests only; NOT a substitute for full-size checks,
# which live in test_full_models.py).
TINY = {
    "transformer_ar": dict(d_model=32, n_heads=4, d_ff=64, depth=3, tied=False),
    "looped_transformer_ar": dict(d_model=32, n_heads=4, d_ff=64, depth=3, tied=True),
    "transformer_encoder": dict(d_model=32, n_heads=4, d_ff=64, depth=3, tied=False),
    "looped_transformer_encoder": dict(d_model=32, n_heads=4, d_ff=64, depth=3, tied=True),
    "vit": dict(d_model=32, n_heads=4, d_ff=64, depth=3, tied=False),
    "looped_vit": dict(d_model=32, n_heads=4, d_ff=64, depth=3, tied=True),
    "trm": dict(d_model=32, n_heads=4, d_ff=48),
    "rescnn_film": dict(channels=16, n_blocks=2, groups=4, cond=dict(op_emb_dim=8, hidden=16)),
    "unet_film": dict(c_high=16, c_low=32, enc_blocks=1, mid_blocks=1, dec_blocks=1, groups=4,
                      cond=dict(op_emb_dim=8, hidden=16)),
}


@pytest.fixture
def tiny():
    return TINY
