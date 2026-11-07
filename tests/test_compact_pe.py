"""Compact left-aligned mixed-length experiments with switchable 1D sequence PE (RoPE vs sinusoidal)."""
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from decompositionality import mixed as M
from decompositionality import tokenization as T
from decompositionality.config import make_experiment_config
from decompositionality.models import common as C
from decompositionality.models.common import count_parameters
from decompositionality.models.registry import build_model, default_config
from conftest import TINY

ATTN = M.ATTENTION_MODELS
GOLDEN_LEGACY = json.loads((Path(__file__).parent / "golden_legacy_hashes.json").read_text())


def cfg_for(m, pe, tiny=True, n_slots=5):
    c = dataclasses.asdict(default_config(m))
    if tiny:
        c.update(TINY[m])
    c["n_slots"] = n_slots
    if pe and m not in M.CNN_MODELS:
        c["sequence_pe"] = pe
    return c


def batch(B=4, empty_row=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    ops = torch.tensor([[0, 1, 3, 3, 3], [2, 0, 1, 3, 3], [1, 1, 1, 1, 1], [3, 3, 3, 3, 3]])[:B]
    return {"ops": ops, "x": torch.randint(0, 10, (B, 6, 6), generator=g), "y": torch.randint(0, 10, (B, 6, 6), generator=g)}


def test_legacy_config_hashes_unchanged():
    from decompositionality.training import run_config
    for e in ("distinct_pairs", "all_pairs"):
        cfg = make_experiment_config(e)
        for m in cfg.models:
            rc = run_config(cfg, m, "bdf24c327a65-test")
            assert GOLDEN_LEGACY[f"{e}/{m}"] == [rc["config_hash"], rc["model_hash"]], m


def test_supports_layout_and_streams():
    sup = M.support("compact_02345_rope")
    assert {L: len(w) for L, w in sup.items()} == {0: 1, 2: 9, 3: 27, 4: 81, 5: 243}
    assert len(M.training_words("compact_02345_rope")) == 361 and len(M.test_words()) == 364
    assert not any(len(w) == 1 for w in M.training_words("compact_02345_sinusoidal"))
    lay = T.ARLayout(5)
    assert (lay.length, lay.idx_output, lay.out_start, lay.idx_eos, lay.idx_slots) == (82, 44, 45, 81, (2, 3, 4, 5, 6))
    assert (T.ARLayout(3).length, T.ARLayout(3).idx_output) == (80, 42)
    streams = [M.MixedTrainStream(42, e, 64) for e in M.COMPACT_EXPERIMENTS]
    for step in (0, 3):
        bs = [s.batch(step) for s in streams]
        for k in ("x", "y", "ops", "string", "length", "sample_index"):
            assert all(np.array_equal(bs[0][k], b[k]) for b in bs[1:]), k
        for ops, L, w in zip(bs[0]["ops"], bs[0]["length"], bs[0]["string"]):
            assert (ops[:L] != 3).all() and (ops[L:] == 3).all()            # compact left aligned
            assert M.canonical_slots("" if w == "EMPTY" else w) == list(ops)
    again = M.MixedTrainStream(42, "compact_02345_rope", 64).batch(3)     # stateless resume
    assert np.array_equal(again["ops"], bs[0]["ops"]) and np.array_equal(again["x"], bs[0]["x"])
    lens = np.concatenate([streams[0].batch(i)["length"] for i in range(20)])
    assert {int(L): int((lens == L).sum()) for L in M.LENGTHS} == {L: 256 for L in M.LENGTHS}
    v = M.make_val_set(np.zeros((300, 6, 6), np.uint8), "compact_02345_rope", 42)
    v2 = M.make_val_set(np.zeros((300, 6, 6), np.uint8), "compact_02345_sinusoidal", 42)
    assert np.array_equal(v["string"], v2["string"]) and len(v["x"]) == 1500
    assert set(v["string"]) == set(M.training_labels("compact_02345_rope"))  # every allowed word at least once


@pytest.mark.parametrize("m", ATTN)
def test_pe_pairs_identical_init_no_learned_1d_tables(m):
    p = {}
    for pe in ("rope", "sinusoidal"):
        mod = build_model(m, cfg_for(m, pe, tiny=False), seed=42)
        p[pe] = dict(mod.named_parameters())
        assert not any(k.endswith(("pos_emb.weight", "slot_emb.weight", "slot_emb")) for k in p[pe])
        assert all("seq_pe" not in k for k in p[pe])              # PE tables are buffers, not parameters
        assert hasattr(mod, "seq_pe") and not any(b.requires_grad for b in mod.seq_pe.buffers())
    assert p["rope"].keys() == p["sinusoidal"].keys()
    assert all(torch.equal(p["rope"][k], p["sinusoidal"][k]) for k in p["rope"])
    assert count_parameters(build_model(m, cfg_for(m, "rope", tiny=False))) == \
        count_parameters(build_model(m, cfg_for(m, "sinusoidal", tiny=False)))


def test_rope_math():
    cos, sin = C.rope_tables(16, 8)
    q, k = torch.randn(1, 1, 16, 8), torch.randn(1, 1, 16, 8)
    rq = C.apply_rope(q, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)       # norm preserving
    q0, k0 = torch.randn(8), torch.randn(8)
    Q = C.apply_rope(q0.expand(1, 1, 16, 8), cos, sin)[0, 0]
    K = C.apply_rope(k0.expand(1, 1, 16, 8), cos, sin)[0, 0]
    # <R(p) q, R(p') k> depends only on p - p'
    assert torch.allclose(Q[5] @ K[2], Q[10] @ K[7], atol=1e-4) and torch.allclose(Q[3] @ K[3], Q[0] @ K[0], atol=1e-4)
    pe = C.sinusoidal_table(4, 6)
    assert torch.allclose(pe[1, 0], torch.sin(torch.tensor(1.0))) and torch.allclose(pe[1, 1], torch.cos(torch.tensor(1.0)))


@pytest.mark.parametrize("m", ATTN)
def test_pe_wired_every_attention_call_and_sinusoid_once(m, monkeypatch):
    from decompositionality.models import trm as trm_mod
    calls = {"rope": 0, "add": 0}
    orig = C.apply_rope

    def counting(x, cos, sin):
        calls["rope"] += 1
        return orig(x, cos, sin)
    monkeypatch.setattr(C, "apply_rope", counting)
    monkeypatch.setattr(trm_mod, "apply_rope", counting)
    mod = build_model(m, cfg_for(m, "rope"), seed=0).eval()
    with torch.no_grad():
        mod.val_cell_ce(batch())
    attn_calls = mod.block_applications if m == "trm" else mod.stack.block_applications
    assert calls["rope"] == 2 * attn_calls > 0                 # Q and K rotated in every attention invocation
    mod = build_model(m, cfg_for(m, "sinusoidal"), seed=0).eval()
    orig_add = mod.seq_pe.additive

    def add(pos):
        calls["add"] += 1
        return orig_add(pos)
    mod.seq_pe.additive = add
    calls["rope"] = 0
    with torch.no_grad():
        mod.val_cell_ce(batch())
    assert calls["add"] == 1 and calls["rope"] == 0            # sinusoid added once at the input, never rotated


@pytest.mark.parametrize("pe", ["rope", "sinusoidal"])
@pytest.mark.parametrize("m", ATTN + M.CNN_MODELS)
def test_empty_ops_finite_and_pad_masking(m, pe):
    mod = build_model(m, cfg_for(m, pe), seed=0).train()
    b = batch()
    loss, _ = mod.training_loss(b)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in mod.parameters() if p.grad is not None)
    mod.eval()
    with torch.no_grad():
        p = mod.predict({k: v for k, v in b.items() if k != "y"})
        assert p["pred"].shape == (4, 36)
        if m in ATTN:   # perturbing the PAD embedding must not change any valid output
            before = mod.val_cell_ce(b)
            if m == "trm":
                mod.tok_emb[T.PAD_ID] += 3.0
            elif m in ("vit", "looped_vit"):
                mod.op_emb.weight[3] += 3.0
            else:
                mod.tok_emb.weight[T.PAD_ID] += 3.0
            assert torch.allclose(before, mod.val_cell_ce(b), atol=1e-5)


@pytest.mark.parametrize("pe", ["rope", "sinusoidal"])
def test_ar_causality_and_free_running(pe):
    mod = build_model("transformer_ar", cfg_for("transformer_ar", pe), seed=0).eval()
    b = batch()
    seq = mod.layout.serialize(b["ops"], b["x"], b["y"])
    seq2 = seq.clone()
    seq2[:, 60:] = (seq2[:, 60:] + 3) % 10
    with torch.no_grad():
        l1, l2 = mod.forward_tokens(seq[:, :-1]), mod.forward_tokens(seq2[:, :-1])
    assert torch.allclose(l1[:, :60], l2[:, :60], atol=1e-5)
    p1 = mod.predict(b)
    p2 = mod.predict(dict(b, y=torch.zeros_like(b["y"])))
    assert torch.equal(p1["raw_tokens"], p2["raw_tokens"])
    # uncached decoding keeps absolute positions: last-position logits equal the full-prefix forward
    s = mod.layout.prefix(b["ops"], b["x"])
    with torch.no_grad():
        full = mod.forward_tokens(seq[:, :50])[:, 44:50]
        inc = torch.stack([mod.forward_tokens(seq[:, :t + 1])[:, -1] for t in range(44, 50)], 1)
    assert torch.allclose(full, inc, atol=1e-5)


def test_singleton_hypotheses_include_xc_cx():
    from decompositionality.diagnosis import group as G
    h = {(d["hypothesis_type"], d["executed_word"]) for d in G.hypotheses_for("A", ["AA", "AB"])}
    assert ("X_then_C", "AC") in h and ("C_then_X", "CA") in h and ("duplicate_single", "AA") in h
    cat = G.Catalog()
    assert cat.word_id("CC") == cat.word_id("") == 0          # C+C and identity coincide (aliases, not independent)


def test_end_to_end_smoke_resume_reuse_and_rejection(tmp_path):
    from decompositionality.compact_reports import combined_table, paired_pe_comparison
    from decompositionality.diagnosis.runner import DiagnosticsConfig, run_diagnostics
    from decompositionality.diagnosis.spec import get_spec
    from decompositionality.mixed_pipeline import evaluate_mixed_model, prepare_mixed
    from decompositionality.mixed_reports import build_mixed_reports, make_compact_archive
    from decompositionality.training import train_model
    root = tmp_path / "compact_02345_pe"
    q = lambda *a: None  # noqa: E731
    attn = ["transformer_ar", "trm"]
    cfgs = {}
    for e, models in (("compact_02345_rope", attn), ("compact_02345_sinusoidal", attn), ("compact_02345_cnn", ["unet_film"])):
        c = make_experiment_config(e, "smoke", str(root), models, device="cpu")
        c.model_overrides = {m: TINY[m] for m in models}
        cfgs[e] = (c, prepare_mixed(c, log=q))
    c, ctx = cfgs["compact_02345_rope"]
    # interrupted + resumed run equals an uninterrupted one is covered for the shared trainer elsewhere; here:
    r = train_model("trm", c, ctx.manifests, ctx.val_set, ctx.subset_set, log=q, stop_after_steps=2)
    assert r["interrupted_at_step"] == 2
    # a RoPE checkpoint must be rejected by the sinusoidal run
    cs, ctxs = cfgs["compact_02345_sinusoidal"]
    (cs.experiment_dir / "runs" / "trm").mkdir(parents=True, exist_ok=True)
    shutil.copy(c.experiment_dir / "runs" / "trm" / "state.pt", cs.experiment_dir / "runs" / "trm" / "state.pt")
    with pytest.raises(RuntimeError):
        train_model("trm", cs, ctxs.manifests, ctxs.val_set, ctxs.subset_set, log=q)
    (cs.experiment_dir / "runs" / "trm" / "state.pt").unlink()
    for e, (c_, ctx_) in cfgs.items():
        for m in c_.models:
            train_model(m, c_, ctx_.manifests, ctx_.val_set, ctx_.subset_set, log=q)
            evaluate_mixed_model(m, ctx_, log=q)
        build_mixed_reports(c_.experiment_dir, e, log=q)
        make_compact_archive(c_.experiment_dir, log=q)
    # shared CNN control is reused (skipped) when a second notebook asks for it with the same config
    cc, cctx = cfgs["compact_02345_cnn"]
    msgs = []
    train_model("unet_film", cc, cctx.manifests, cctx.val_set, cctx.subset_set, log=msgs.append)
    assert any("skipping" in m for m in msgs)
    assert (root / "smoke" / "manifests" / "heldout_grids.npz").exists()
    ct = combined_table(root / "smoke")
    assert (ct.status == "complete").sum() == 5
    pc = paired_pe_comparison(root / "smoke", log=q)
    assert pc is not None and {"A", "B", "C"} <= set(pc.word_or_group)
    row = pc[(pc.model == "trm") & (pc.word_or_group == "A")].iloc[0]
    assert row.both_correct + row.only_rope_correct + row.only_sinusoidal_correct + row.both_wrong == row.n
    dirs = {e: str(c_.experiment_dir) for e, (c_, _) in cfgs.items()}
    R = run_diagnostics(DiagnosticsConfig(dirs, str(tmp_path / "diag"), spec="compact",
                                          analysis_words=get_spec("compact").common_panel), log=q)
    assert R["audit"].target_oracle_mismatches.sum() == 0 and (R["audit"].empty_word_records > 0).all()
    assert set(R["per_string"].string) == set(get_spec("compact").common_panel)
    hs = R["hypothesis_summary"]
    assert (hs.matched_unique_records <= hs.population_any_legal_match_records).all()
    assert len(R["paired"]) and {"both_correct", "both_wrong"} <= set(R["paired"].columns)
