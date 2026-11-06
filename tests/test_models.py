import copy

import pytest
import torch

from decompositionality import tokenization as T
from decompositionality.models.common import count_parameters
from decompositionality.models.registry import MODEL_IDS, TIED_PAIRS, build_model
from decompositionality.models.vit import patchify, unpatchify
from conftest import TINY


def batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ops": torch.tensor([[0, 1, 3], [2, 3, 0], [3, 1, 2], [0, 2, 1]])[:B],
            "x": torch.randint(0, 10, (B, 6, 6), generator=g), "y": torch.randint(0, 10, (B, 6, 6), generator=g)}


def tiny(mid, seed=0):
    return build_model(mid, TINY[mid], seed=seed).eval()


@pytest.mark.parametrize("mid", MODEL_IDS)
def test_tiny_interface(mid):
    m = tiny(mid).train()
    b = batch()
    loss, _ = m.training_loss(b)
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    p = m.predict({k: v for k, v in b.items() if k != "y"})  # prediction never receives the target
    assert p["pred"].shape == (4, 36) and p["valid"].shape == (4,)
    assert m.val_cell_ce(b).shape == (4,)


def test_ar_causal_mask_future_tokens_cannot_change_earlier_logits():
    m = tiny("transformer_ar")
    b = batch()
    seq = T.ar_serialize(b["ops"], b["x"], b["y"])
    seq2 = seq.clone()
    seq2[:, 60:] = (seq2[:, 60:] + 3) % 10
    with torch.no_grad():
        l1, l2 = m.forward_tokens(seq[:, :-1]), m.forward_tokens(seq2[:, :-1])
    assert torch.allclose(l1[:, :60], l2[:, :60], atol=1e-6)
    assert not torch.allclose(l1[:, 60:], l2[:, 60:])


def test_ar_loss_alignment():
    m = tiny("transformer_ar")
    b = batch()
    lg, tg = m._teacher_forced(b)
    assert lg.shape[1] == 37
    assert torch.equal(tg[:, :36], b["y"].reshape(4, 36))  # answer digits
    assert (tg[:, 36] == T.EOS_ID).all()
    # first answer digit is predicted from the hidden state at <OUTPUT> (index 42)
    seq = T.ar_serialize(b["ops"], b["x"], b["y"])
    with torch.no_grad():
        direct = m.forward_tokens(seq[:, :43])[:, 42]
    assert torch.allclose(direct, lg[:, 0], atol=1e-5)


def test_ar_generation_ignores_target_and_matches_manual_greedy():
    m = tiny("looped_transformer_ar")
    b = batch()
    p1 = m.predict(b)
    b2 = dict(b, y=torch.zeros_like(b["y"]))
    p2 = m.predict(b2)
    assert torch.equal(p1["raw_tokens"], p2["raw_tokens"])
    seq = T.ar_prefix(b["ops"], b["x"])
    with torch.no_grad():
        for _ in range(36):
            seq = torch.cat([seq, m.forward_tokens(seq)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(seq[:, 43:79], p1["raw_tokens"])


def test_ar_invalid_tokens_marked():
    m = tiny("transformer_ar")
    with torch.no_grad():
        m.head.bias.zero_()
        m.head.bias[T.EOS_ID] = 100.0  # force non-digit outputs
    p = m.predict(batch())
    assert not p["valid"].any() and (p["pred"] == -1).all()


@pytest.mark.parametrize("mid", ["transformer_ar", "transformer_encoder", "vit", "trm", "looped_vit"])
def test_pad_keys_do_not_leak(mid):
    m = tiny(mid)
    b = batch()
    with torch.no_grad():
        out1 = m.val_cell_ce(b)
        if mid == "trm":
            m.tok_emb[T.PAD_ID] += 5.0
        elif mid == "vit" or mid == "looped_vit":
            m.op_emb.weight[3] += 5.0
        else:
            m.tok_emb.weight[T.PAD_ID] += 5.0
        out2 = m.val_cell_ce(b)
    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize("mid", ["rescnn_film", "unet_film"])
def test_cnn_pad_embedding_zero(mid):
    m = tiny(mid)
    assert torch.all(m.cond.emb.weight[3] == 0)
    m.train()
    m.training_loss(batch())[0].backward()
    assert m.cond.emb.weight.grad[3].abs().sum() == 0


@pytest.mark.parametrize("mid", MODEL_IDS)
def test_operation_order_matters(mid):
    m = tiny(mid)
    b = batch()
    b2 = dict(b, ops=b["ops"].flip(1))  # reversed order of the same operations
    with torch.no_grad():
        if mid in ("rescnn_film", "unet_film"):  # zero-init FiLM: perturb to non-identity first
            m.cond.film.weight.normal_(0, 0.5)
        assert not torch.allclose(m.val_cell_ce(b), m.val_cell_ce(b2))


def test_patchify_roundtrip_and_order():
    x = torch.arange(36).view(1, 6, 6)
    p = patchify(x)
    assert p.shape == (1, 9, 4)
    assert p[0, 0].tolist() == [0, 1, 6, 7] and p[0, 1].tolist() == [2, 3, 8, 9] and p[0, 3].tolist() == [12, 13, 18, 19]
    assert torch.equal(unpatchify(p), x)
    f = torch.randn(2, 6, 6, 10)
    assert torch.equal(unpatchify(patchify(f)), f)


@pytest.mark.parametrize("tied_id", list(TIED_PAIRS))
def test_tied_untied_equivalence(tied_id):
    untied_id = TIED_PAIRS[tied_id]
    tied = tiny(tied_id, seed=1)
    untied = tiny(untied_id, seed=2)
    assert len(tied.stack.blocks) == 1 and len(untied.stack.blocks) == 3
    sd = tied.state_dict()
    new = {}
    for k, v in untied.state_dict().items():
        if k.startswith("stack.blocks."):
            parts = k.split(".")
            parts[2] = "0"
            new[k] = sd[".".join(parts)]
        else:
            new[k] = sd[k]
    untied.load_state_dict(new)
    b = batch()
    with torch.no_grad():
        assert torch.allclose(tied.val_cell_ce(b), untied.val_cell_ce(b), atol=1e-5)
        if tied.kind == "ar":
            assert torch.equal(tied.predict(b)["raw_tokens"], untied.predict(b)["raw_tokens"])
    # parameter sharing: one shared block applied depth times
    assert count_parameters(untied) - count_parameters(tied) == 2 * count_parameters(tied.stack.blocks[0])
    assert tied.stack.block_applications == 3


def test_trm_calls_detachment_and_reset():
    m = build_model("trm", TINY["trm"], seed=0).train()
    b = batch()
    loss, _ = m.training_loss(b)
    e = m.expected_calls()
    assert (e["core_calls"], e["block_applications"], e["grad_core_calls"]) == (12, 24, 6)
    assert (m.core_calls, m.block_applications, m.grad_core_calls, m.grad_block_applications) == (12, 24, 6, 12)
    # backward through the final refinement only reaches its own gradient-bearing core calls
    counter = {"n": 0}
    hooks = [blk.register_full_backward_hook(lambda *a: counter.__setitem__("n", counter["n"] + 1)) for blk in m.core]
    outs = m.refinement_logits(b)
    outs[-1].float().pow(2).mean().backward()
    assert counter["n"] == 3 * 2  # (n+1) core calls x 2 blocks; earlier refinement is detached
    counter["n"] = 0
    m.zero_grad()
    outs = m.refinement_logits(b)
    sum(o.float().pow(2).mean() for o in outs).backward()
    assert counter["n"] == 2 * 3 * 2  # every refinement head is differentiable
    for h in hooks:
        h.remove()
    # no-grad cycles at eval; deterministic; no state carried across inputs/batches
    m.eval()
    with torch.no_grad():
        m.val_cell_ce(b)
    assert m.core_calls == 12 and m.grad_core_calls == 0
    p_all = m.predict(b)["pred"]
    p_one = m.predict({k: v[:1] for k, v in b.items()})["pred"]
    assert torch.equal(p_all[:1], p_one) and torch.equal(p_all, m.predict(b)["pred"])
