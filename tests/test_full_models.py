"""Full-size (benchmark configuration) checks: parameter counts and forward/backward with
finite gradients for ALL NINE models. Models are built one at a time and released."""
import gc

import pytest
import torch

from decompositionality.models.common import count_parameters
from decompositionality.models.registry import MODEL_IDS, TIED_PAIRS, build_model


@pytest.mark.parametrize("mid", MODEL_IDS)
def test_full_model_counts_and_gradients(mid):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(mid, seed=42).to(device).train()
    n = count_parameters(m)
    if mid in TIED_PAIRS:
        assert 1.7e6 <= n <= 1.9e6, n
        assert 10.5e6 <= m.untied_equivalent_parameters() <= 10.8e6
    else:
        assert 9e6 <= n <= 11e6, n
    b = m.dummy_batch(2, device)
    loss, _ = m.training_loss(b)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    with_grad = sum(p.numel() for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert with_grad / n > 0.95, (mid, with_grad / n)
    m.eval()
    p = m.predict({k: v for k, v in b.items() if k != "y"})
    assert p["pred"].shape == (2, 36)
    del m, loss, grads
    gc.collect()
