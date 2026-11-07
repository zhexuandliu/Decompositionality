"""Slot-count experiment (one notebook, eight runs): shared preparation, audits, the sequential run driver, and the
comparison / figure / export layer. Comparison functions read ONLY saved artifacts (artifact index, CSV/JSON logs,
prediction npz files); they never build or train models.

Results tree (under <RESULTS_ROOT>/identity_ar_slot_count/):
  manifests/                      shared grid manifests (same seed -> same hash as the five-slot experiments)
  audits/seed_<seed>/             support / layout / initialization / stream-matching audits
  <condition>/seed_<seed>/runs/<model>/   checkpoints (best.pt, state.pt), logs, run_config.json
  <condition>/seed_<seed>/eval/<model>/   prediction chunks, test_predictions.npz, ARTIFACT.json, metric CSVs,
                                          diagnostics/, export/
  status/seed_<seed>.json         per-run stage status (persisted after every stage)
  comparisons/seed_<seed>/        consolidated tables and figures
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import idslot as I
from . import idvar as V
from .names import DISPLAY_NAMES
from .plotstyle import _plt, save_fig
from .words import from_label, words_of_length

CONDITIONS = list(I.IDSLOT_EXPERIMENTS)
MODELS = list(I.IDSLOT_MODELS)
STAGES = ["train", "evaluate", "diagnose", "export"]
COLORS = {"slots2_train02": "#2a78d6", "slots3_train023": "#eb6834", "slots4_train0234": "#1baf7a",
          "slots6_train02345": "#6250d6"}
HIST_NAME = {"slots2_train02": "train02", "slots3_train023": "train023", "slots4_train0234": "train0234",
             "slots6_train02345": "train02345"}
NA = "N/A"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def prepare_condition(cfg, repo_dir=None, compare_manifests: Optional[Dict[str, str]] = None, log=print):
    """Shared manifests (same seed as the five-slot experiments) + this condition's validation / test sets."""
    from .data import load_or_build_manifests
    from .pipeline import Context
    from .utils import atomic_write_json
    assert cfg.is_idslot
    e, seed = cfg.experiment, cfg.train.seed
    m = load_or_build_manifests(cfg.manifest_dir, seed)
    val = I.make_val_set(m.val_grids[: cfg.n_val_grids], seed, e)
    subset = V.subset_by_grids(val, cfg.val_subset_grids)
    test = I.make_test_set(m.test_grids[: cfg.n_test_grids], seed, e)
    assert len(test["x"]) == I.expected_test_records(e, cfg.n_test_grids)
    assert len(val["x"]) == cfg.n_val_grids * len(I.train_ks(e))
    assert set(np.unique(val["k"])) == set(I.train_ks(e)) and not (val["k"] == 1).any()
    val_b = {g.tobytes() for g in m.val_grids.astype(np.uint8)}
    test_b = {g.tobytes() for g in m.test_grids.astype(np.uint8)}
    assert not (val_b & test_b), "validation and test grids overlap"
    cmp = {}
    for name, path in (compare_manifests or {}).items():
        if path and Path(path).exists():
            h = json.loads(Path(path).read_text()).get("combined_hash")
            cmp[name] = "identical" if h == m.manifest_hash else "DIFFERENT"
        else:
            cmp[name] = "unavailable"
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.experiment_dir / "experiment_config.json",
                      {**cfg.to_dict(), "S": I.n_slots(e), "train_ks": list(I.train_ks(e)),
                       "manifest_hash": m.manifest_hash, "manifest_comparison": cmp, "n_val_records": len(val["x"]),
                       "n_test_records": len(test["x"]), "data_version": I.IDSLOT_DATA_VERSION})
    log(f"[{e}] S={I.n_slots(e)} | manifests hash {m.manifest_hash[:12]} ({cmp}) | validation {len(val['x'])} | "
        f"test {len(test['x']):,}")
    return Context(cfg, m, val, subset, test, repo_dir)


# ---------------------------------------------------------------------------
# Audits (run once in the notebook's audit cell; written to audits/seed_<seed>/)
# ---------------------------------------------------------------------------

def support_audit(B: int = 256, steps: int = 5000):
    import pandas as pd
    from math import comb
    from . import idext as E
    I.check_supports()
    rows = []
    for e in CONDITIONS:
        S, ks = I.IDSLOT_EXPERIMENTS[e]
        full = I.full_run_counts(e, B, steps)
        for k in range(S + 1):
            rows.append({"condition": e, "S": S, "k": k, "identity_tokens": S - k,
                         "status": "train" if k in ks else "HELD OUT",
                         "records_update_0": E.per_update_counts(ks, B, 0).get(k, 0),
                         "records_update_1": E.per_update_counts(ks, B, 1).get(k, 0),
                         "records_full_run": full.get(k, 0), "logical_words": 3 ** k,
                         "physical_patterns": comb(S, k) * 3 ** k})
    df = pd.DataFrame(rows)
    for e in CONDITIONS:
        d = df[df.condition == e]
        assert d.records_full_run.sum() == B * steps
        assert d[d.status == "HELD OUT"].records_full_run.sum() == 0
    return df


def layout_audit(seed: int = 42, decode_check: bool = True):
    """Actual tensors through the real serializer and model code for every condition and every relevant k:
    length 77+S, OUTPUT/answer/EOS indices, contiguous positions, all identity tokens visible (no PAD), causal
    mask, exactly one teacher-forcing shift, no answer tokens in the decoding prefix. With ``decode_check``, a
    freshly initialized (untrained) model of each architecture runs a teacher-forced forward pass and greedy
    decoding at S=6 / k=6 and k=1 within the same 83-position table used in training."""
    import pandas as pd
    import torch
    from . import tokenization as T
    from .evaluation import to_device
    from .models.registry import build_model
    from .config import make_experiment_config
    from .training import model_config_for
    expect = {2: (79, 41, 42, 77, 78), 3: (80, 42, 43, 78, 79), 4: (81, 43, 44, 79, 80), 6: (83, 45, 46, 81, 82)}
    rows = []
    rng = np.random.default_rng(0)
    models = {}
    if decode_check:
        for m in MODELS:
            mc = model_config_for(make_experiment_config("slots6_train02345", models=[m]), m)
            mod = build_model(m, mc)
            mod.load_state_dict(I.grown_initial_state(m, seed))
            models[m] = mod.eval()
    for e in CONDITIONS:
        S, ks = I.IDSLOT_EXPERIMENTS[e]
        lay = T.ARLayout(S)
        got = (lay.length, lay.idx_output, lay.out_start, lay.out_start + 35, lay.idx_eos)
        assert lay.length == 77 + S and lay.idx_output == 39 + S and got == expect[S], (S, got)
        meta = lay.metadata()
        assert np.array_equal(np.asarray(meta["pos"]), np.arange(lay.length)), "positions must be contiguous"
        test_ks = sorted(set(ks) | {1} | ({6} if S == 6 else set()))
        for k in test_ks:
            w = "".join("ABC"[c] for c in rng.integers(0, 3, k))
            pos = I.subset(rng, S, k)
            slots = I.slots_for(w, pos, S)[None]
            x = rng.integers(0, 10, (1, 6, 6))
            y = V.apply_words(x, [w])
            tb = to_device({"x": x, "y": y, "slots": slots, "condition": "identity"}, torch.device("cpu"))
            seq = lay.serialize(tb["ops"], tb["x"], tb["y"])[0]
            pre = lay.prefix(tb["ops"], tb["x"])[0]
            ops_tok = seq[2:2 + S].tolist()
            n_id = sum(t == T.IDENTITY_ID for t in ops_tok)
            assert T.PAD_ID not in seq.tolist() and len(seq) == lay.length and n_id == S - k
            assert seq[lay.idx_output] == T.OUTPUT_ID and seq[lay.idx_eos] == T.EOS_ID
            assert len(pre) == lay.prefix_len == lay.idx_output + 1 and pre[-1] == T.OUTPUT_ID
            assert torch.equal(seq[lay.out_start:lay.out_start + 36], torch.as_tensor(y.reshape(36)))
            row = {"condition": e, "S": S, "k": k, "word": w or "EMPTY", "op_tokens":
                   " ".join("I" if t == T.IDENTITY_ID else "ABC"[t - 10] for t in ops_tok), "identity_tokens": n_id,
                   "pad_tokens": 0, "total_tokens": lay.length, "output_index": lay.idx_output,
                   "answer_indices": f"{lay.out_start}..{lay.out_start + 35}", "eos_index": lay.idx_eos,
                   "prefix_tokens": lay.prefix_len, "positions": f"contiguous 0..{lay.length - 1}"}
            if decode_check:
                for m, mod in models.items():
                    with torch.no_grad():
                        lg, tg = mod._teacher_forced(tb)
                        assert lg.shape[1] == 37 and torch.equal(tg[0], seq[lay.out_start:lay.idx_eos + 1])
                        full = mod.forward_tokens(seq[None, :-1], S)
                        assert full.shape[1] == lay.length - 1
                        # causal: changing the answer tokens must not change logits at the prefix positions
                        seq2 = seq.clone()
                        seq2[lay.out_start:lay.idx_eos] = (seq2[lay.out_start:lay.idx_eos] + 1) % 10
                        f2 = mod.forward_tokens(seq2[None, :-1], S)
                        assert torch.allclose(full[0, :lay.idx_output + 1], f2[0, :lay.idx_output + 1], atol=1e-5)
                        if (S == 6 and k == 6) or k == 1:
                            out = mod.predict(tb)
                            assert out["pred"].shape == (1, 36)
                            row[f"{m}_greedy_decode_ok"] = True
                    row[f"{m}_teacher_forced_ok"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def _sha(tensors) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        h.update(k.encode())
        h.update(tensors[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def init_audit(seed: int = 42):
    """Common fresh initialization (identical for all four conditions by construction AND checked via the actual
    model-construction path), parameter counts, block applications, and agreement of the shared tensors with the
    historical five-slot construction (idvar_reports.model_config(m, 'identity'), same seed)."""
    import pandas as pd
    import torch
    from .config import make_experiment_config
    from .idvar_reports import model_config as hist_config
    from .models.common import count_parameters
    from .models.registry import build_model
    from .training import model_config_for
    rows = []
    for m in MODELS:
        sds = {}
        for e in CONDITIONS:
            mod = build_model(m, model_config_for(make_experiment_config(e, models=[m]), m), seed=seed)
            mod.load_state_dict(I.grown_initial_state(m, seed))
            sds[e] = {k: v.detach().clone() for k, v in mod.state_dict().items()}
        ref = sds[CONDITIONS[0]]
        same = all(s.keys() == ref.keys() and all(torch.equal(ref[k], s[k]) for k in ref) for s in sds.values())
        hist = build_model(m, hist_config(m, "identity"), seed=seed)
        hsd = hist.state_dict()
        shared = [k for k in ref if k != "pos_emb.weight"]
        shared_same = all(torch.equal(ref[k], hsd[k]) for k in shared)
        pos_prefix_same = torch.equal(ref["pos_emb.weight"][: hsd["pos_emb.weight"].shape[0]], hsd["pos_emb.weight"])
        hist_param_sha = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for _, v in hist.named_parameters())).hexdigest()[:16]
        cfgm = mod.cfg
        rows.append({"model": m, "unique_params": count_parameters(mod), "historical_S5_params": count_parameters(hist),
                     "tied": bool(cfgm.tied), "block_applications": int(cfgm.depth), "d_model": cfgm.d_model,
                     "n_heads": cfgm.n_heads, "d_ff": cfgm.d_ff, "dropout": cfgm.dropout,
                     "pos_table_shape": tuple(ref["pos_emb.weight"].shape),
                     "historical_pos_table_shape": tuple(hsd["pos_emb.weight"].shape),
                     "identical_init_across_4_conditions": bool(same), "full_init_sha": _sha(ref),
                     "shared_tensors_sha": _sha({k: ref[k] for k in shared}),
                     "shared_tensors_equal_historical_S5_construction": bool(shared_same),
                     "pos_rows_0_81_equal_historical": bool(pos_prefix_same),
                     "added_rows": "row 82 from torch.Generator(seed*1000+83), std 0.02",
                     "historical_named_param_sha (idext init_audit convention)": hist_param_sha})
    return pd.DataFrame(rows)


def stream_audit(seed: int = 42, B: int = 256, steps: Sequence[int] = (0, 1, 2, 3, 2499, 4999), heldout=None):
    """Supports 02/023/0234: the logical (word, grid, target) stream and batch order equal the historical five-slot
    runs (idext.ExtStream) at the same update; only the slot layout changes. 02345: the historical run (notebook 10)
    used idvar.IdVarStream, so the S=6 stream is NEW (labelled unmatched). Also: the two architectures read the
    same stateless stream (batch is a pure function of the step), and placement subsets are uniform."""
    import pandas as pd
    from math import comb
    from itertools import combinations
    from . import idext as E
    from .data import make_rng
    rows = []
    for e in CONDITIONS:
        st = I.SlotStream(seed, e, B, heldout)
        hist = I.HISTORICAL_S5[e]
        for s in steps:
            b = st.batch(s)
            b2 = I.SlotStream(seed, e, B, heldout).batch(s)
            deterministic = all(np.array_equal(b[k], b2[k]) for k in ("x", "y", "slots", "string"))
            ks_ok = set(np.unique(b["k"])) <= set(I.train_ks(e)) and not (b["k"] == 1).any()
            nid = (b["slots"] == V.ID_OP).sum(1)
            row = {"condition": e, "update": s, "records": len(b["x"]), "deterministic_replay": deterministic,
                   "only_training_k": bool(ks_ok), "identity_tokens_equal_S_minus_k": bool((nid == I.n_slots(e) - b["k"]).all()),
                   "no_identity_records": int((nid == 0).sum())}
            if hist[0] == "idext":
                h = E.ExtStream(seed, hist[1], B, heldout).batch(s)
                row["historical_S5_stream"] = hist[1]
                row["same_words_grids_targets_order_as_S5"] = bool(
                    np.array_equal(h["string"], b["string"]) and np.array_equal(h["x"], b["x"])
                    and np.array_equal(h["y"], b["y"]) and np.array_equal(h["k_index"], b["k_index"]))
            else:
                row["historical_S5_stream"] = f"{hist[1]} (notebook 10, IdVarStream)"
                row["same_words_grids_targets_order_as_S5"] = "UNMATCHED (different historical sampler)"
            rows.append(row)
    df = pd.DataFrame(rows)
    # uniformity of the ordered slot subset within S (the same subset() routine is used for train/val/test)
    urows = []
    for S, k in ((3, 1), (4, 2), (6, 3), (6, 5)):
        cnt = {}
        n = 30000
        for j in range(n):
            c = I.subset(make_rng(seed, I.STREAM_SLOT_TRAIN, S, k, j), S, k)
            cnt[c] = cnt.get(c, 0) + 1
        exp = n / comb(S, k)
        chi2 = sum((v - exp) ** 2 / exp for v in cnt.values()) + exp * (comb(S, k) - len(cnt))
        urows.append({"S": S, "k": k, "subsets": comb(S, k), "observed_subsets": len(cnt), "draws": n,
                      "chi2": round(chi2, 1), "dof": comb(S, k) - 1})
    return df, pd.DataFrame(urows)


def test_manifest_audit(ctxs: dict):
    """Exact per-condition evaluation counts, exhaustive singleton position coverage, the complete 729-word
    six-operation set, independent oracle targets, and identical seen-control records across the two models
    (one manifest per condition, shared by both)."""
    import pandas as pd
    from .diagnosis.group import oracle_apply_string
    rows = []
    for e, ctx in ctxs.items():
        t = ctx.test_set
        S = I.n_slots(e)
        n_grids = int(t["grid_id"].max()) + 1
        kinds = t["kind"]
        tgt = np.empty_like(t["y"])
        for s in np.unique(t["string"]):
            m = t["string"] == s
            tgt[m] = oracle_apply_string(t["x"][m], from_label(s))
        sing = kinds == "single"
        cover = {(o, p) for o, p in zip(t["string"][sing], t["position"][sing])}
        six = kinds == "six"
        r = {"condition": e, "S": S, "grids": n_grids, "seen_records": int((kinds == "seen").sum()),
             "singleton_records": int(sing.sum()), "six_op_records": int(six.sum()), "total": len(t["x"]),
             "expected_total": I.expected_test_records(e, n_grids),
             "singleton_cells_covered": len(cover), "singleton_cells_expected": 3 * S,
             "six_op_words": len(set(t["string"][six])), "oracle_target_mismatches": int((tgt != t["y"]).any((1, 2)).sum()),
             "seen_words": len(set(t["string"][kinds == "seen"])), "seen_words_expected": sum(3 ** k for k in I.train_ks(e)),
             "pad_tokens": 0, "identity_count_mismatches": int(((t["slots"] == V.ID_OP).sum(1) != S - t["k"]).sum())}
        if S == 6:
            assert r["six_op_words"] == 729 and (t["slots"][six] != V.ID_OP).all()
        assert r["total"] == r["expected_total"] and r["oracle_target_mismatches"] == 0
        assert r["singleton_cells_covered"] == 3 * S and r["identity_count_mismatches"] == 0
        assert r["seen_words"] == r["seen_words_expected"]
        rows.append(r)
    return pd.DataFrame(rows)


def microbatch_audit(seed: int = 0):
    """One logical update with microbatch 16 vs 4 on a tiny AR model (S=3 identity format): identical gradients
    (loss weight = n/B), exactly one optimizer and scheduler step per logical update."""
    import torch
    from .config import TrainConfig
    from .models.transformer import TransformerAR, TransformerConfig
    from .training import StepRunner, make_optimizer, make_scheduler, make_scaler
    from dataclasses import replace
    st = I.SlotStream(seed, "slots3_train023", 16)
    batch = st.batch(0)
    tc = replace(TrainConfig(), effective_batch_size=16, optimizer_steps=10, warmup_steps=2)
    grads, steps = [], []
    for mb in (16, 4):
        torch.manual_seed(0)
        model = TransformerAR(TransformerConfig(d_model=32, n_heads=2, d_ff=64, depth=2, n_slots=6, identity_token=True,
                                                op_format="variable"))
        opt = make_optimizer(model, tc)
        sched = make_scheduler(opt, tc)
        before = [p.detach().clone() for p in model.parameters()]
        calls = {"opt": 0, "sched": 0}
        o_step, s_step = opt.step, sched.step
        opt.step = lambda *a, **k: (calls.__setitem__("opt", calls["opt"] + 1), o_step(*a, **k))[1]
        sched.step = lambda *a, **k: (calls.__setitem__("sched", calls["sched"] + 1), s_step(*a, **k))[1]
        run = StepRunner(model, opt, sched, make_scaler(None), tc, torch.device("cpu"), None, mb, log=lambda *_: None)
        run(batch)
        grads.append(torch.cat([(a - p.detach()).flatten() for a, p in zip(before, model.parameters())]))
        steps.append(dict(calls))
    diff = float((grads[0] - grads[1]).abs().max())
    return {"max_abs_param_update_difference_mb16_vs_mb4": diff, "equal_within_1e-6": diff < 1e-6,
            "optimizer_steps_per_update": [s["opt"] for s in steps], "scheduler_steps_per_update": [s["sched"] for s in steps]}


# ---------------------------------------------------------------------------
# Status persistence
# ---------------------------------------------------------------------------

def status_path(slot_root: Path, seed: int) -> Path:
    return Path(slot_root) / "status" / f"seed_{seed}.json"


def load_status(slot_root: Path, seed: int) -> dict:
    p = status_path(slot_root, seed)
    return json.loads(p.read_text()) if p.exists() else {}


def _set_status(slot_root, seed, cond, model, stage, value):
    from .utils import atomic_write_json
    st = load_status(slot_root, seed)
    st.setdefault(f"{cond}/{model}", {})[stage] = {"status": value, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(status_path(slot_root, seed), st)


def status_table(slot_root: Path, seed: int = 42):
    """Eight rows (condition x model) x stages, verified against artifacts on disk (never only the status file)."""
    import pandas as pd
    st = load_status(slot_root, seed)
    idx = build_index(slot_root)
    rows = []
    for e in CONDITIONS:
        for m in MODELS:
            d = Path(slot_root) / e / f"seed_{seed}"
            a = idx.get(f"{e}/seed_{seed}/{m}", {})
            on_disk = {"train": (d / "runs" / m / "TRAINING_COMPLETE.json").exists(),
                       "evaluate": bool(a.get("complete")),
                       "diagnose": (d / "eval" / m / "diagnostics" / "DONE.json").exists(),
                       "export": (d / "eval" / m / "export" / "DONE.json").exists()}
            r = {"condition": e, "S": I.n_slots(e), "model": m}
            for s in STAGES:
                logged = st.get(f"{e}/{m}", {}).get(s, {}).get("status", "pending")
                r[s] = "done" if on_disk[s] else (logged if logged != "done" else "MISSING ON DISK")
            r["complete"] = all(on_disk.values())
            r["selected_step"] = a.get("selected_step")
            r["records"] = a.get("n_records")
            rows.append(r)
    return pd.DataFrame(rows)


def build_index(slot_root: Path) -> dict:
    from .utils import atomic_write_json
    root = Path(slot_root)
    idx = {}
    for f in sorted(root.glob("*/seed_*/eval/*/ARTIFACT.json")):
        a = json.loads(f.read_text())
        if a.get("experiment") in I.IDSLOT_EXPERIMENTS:
            idx[f"{a['experiment']}/seed_{a['seed']}/{a['model']}"] = a
    if root.exists():
        atomic_write_json(root / "artifact_index.json", idx)
    return idx


# ---------------------------------------------------------------------------
# Per-run diagnostics and export (stages "diagnose" and "export")
# ---------------------------------------------------------------------------

def _singles_spec(S: int, experiment: str):
    from . import operations as O
    from .diagnosis.spec import DiagSpec
    return DiagSpec(
        name=f"idslot_singles_S{S}", experiments=[f"identity@slot{p}" for p in range(1, S + 1)], words=["A", "B", "C"],
        common_panel=["A", "B", "C"], panel_groups=[("Singletons (held out)", 3)], group_order=["singles"],
        group_labels={"singles": "Singletons A/B/C (held out)"}, train_labels=lambda e: I.training_labels(experiment),
        pair_words=lambda e: list(O.TRAIN_SETS["all_pairs"]), group_fn=lambda e, s: "singles",
        populations={"singletons": ["singles"]}, controls_words=["A", "B", "C"], n_slots=S)


def singleton_consistency(a, experiment, model):
    import pandas as pd
    S = I.n_slots(experiment)
    rows = []
    s = a["kind"] == "single"
    for op in "ABC":
        P, E_ = [], []
        for p in range(1, S + 1):
            m = s & (a["string"] == op) & (a["position"] == p)
            o = np.argsort(a["grid_id"][m])
            P.append(a["pred"][m][o].reshape(-1, 36))
            E_.append(a["exact"][m][o])
        P, E_ = np.stack(P), np.stack(E_)
        same = (P == P[0:1]).all(axis=(0, 2))
        rows.append({"condition": experiment, "model": model, "op": op, "S": S, "n_grids": P.shape[1],
                     "grids_identical_prediction_across_all_S_placements": int(same.sum()),
                     "consistency_rate": float(same.mean()),
                     "grids_correct_at_all_placements": int(E_.all(0).sum()),
                     "grids_correct_at_some_placement": int(E_.any(0).sum())})
    return pd.DataFrame(rows)


def singleton_error_diagnostics(a, experiment, model, split_seed=20251106):
    """The verified 288-transform singleton analysis (correct / XC / CX / identity / legal-permutation matches,
    hypotheses with union counts) per physical position. Describes behavior, not internal execution."""
    import pandas as pd
    from .diagnosis import group as G
    from .diagnosis.analysis import analyze_run, diagnostic_split, hypothesis_population_summary
    from .diagnosis.artifacts import Run
    S = I.n_slots(experiment)
    spec = _singles_spec(S, experiment)
    cat = G.Catalog()
    split = diagnostic_split(np.unique(a["grid_id"]), split_seed)
    ps, hs = [], []
    for p in range(1, S + 1):
        m = (a["kind"] == "single") & (a["position"] == p)
        r = Run(f"identity@slot{p}", model, "original", {}, a["input"][m].reshape(-1, 36).astype(np.int16),
                a["target"][m].reshape(-1, 36).astype(np.int16), a["pred"][m].reshape(-1, 36).astype(np.int16),
                a["grid_id"][m].astype(np.int64), a["string"][m].astype(str), {}, {"file_sha256": {}})
        res = analyze_run(r, cat, spec, split)
        x = res["per_string"].copy(); x.insert(0, "position", p); ps.append(x)
        h = hypothesis_population_summary(res["hypothesis_types"], res["per_string"], spec); h.insert(0, "position", p)
        hs.append(h)
    out = pd.concat(ps, ignore_index=True), pd.concat(hs, ignore_index=True)
    for t in out:
        for col, val in (("model", model), ("condition", experiment)):
            if col in t:
                t[col] = val
            else:
                t.insert(0, col, val)
    return out


def six_op_subsequence_diagnostics(a, model):
    """S=6 only, FAILED k=6 records: does the prediction equal executing only the first five operations, only the
    last five, or the word with one operation deleted? Every match is kept (different subwords can induce the
    same permutation); union coverage counts each record once. Post-hoc checks, not supervision."""
    import pandas as pd
    from .diagnosis import group as G
    six = a["kind"] == "six"
    X = a["input"].reshape(len(a["input"]), 36)
    P = a["pred"].reshape(len(a["pred"]), 36)
    fail = six & ~a["exact"].astype(bool)
    names = ["first_five", "last_five"] + [f"delete_{i + 1}" for i in range(6)]
    match = {n: np.zeros(len(P), bool) for n in names}
    ambiguous_words = 0
    for w in np.unique(a["string"][fail]):
        m = fail & (a["string"] == w)
        cands = {"first_five": w[:5], "last_five": w[1:], **{f"delete_{i + 1}": w[:i] + w[i + 1:] for i in range(6)}}
        perms = {n: G.word_perm(s) for n, s in cands.items()}
        if len({p.tobytes() for p in perms.values()}) < len(perms):
            ambiguous_words += 1
        Xm, Pm = X[m], P[m]
        for n, p in perms.items():
            match[n][m] = (Xm[:, p] == Pm).all(1)
    nf = int(fail.sum())
    union = np.zeros(len(P), bool)
    for n in names:
        union |= match[n]
    rows = [{"model": model, "check": n, "failed_six_op_records": nf, "matches": int(match[n][fail].sum()),
             "rate_of_failed": (match[n][fail].mean() if nf else np.nan)} for n in names]
    multi = sum(match[n].astype(int) for n in names)
    rows.append({"model": model, "check": "UNION (each record once)", "failed_six_op_records": nf,
                 "matches": int(union[fail].sum()), "rate_of_failed": (union[fail].mean() if nf else np.nan)})
    rows.append({"model": model, "check": "records matching >1 check", "failed_six_op_records": nf,
                 "matches": int((multi[fail] > 1).sum()), "rate_of_failed": ((multi[fail] > 1).mean() if nf else np.nan)})
    rows.append({"model": model, "check": "failed words whose candidate subwords share a permutation",
                 "failed_six_op_records": nf, "matches": ambiguous_words, "rate_of_failed": np.nan})
    return pd.DataFrame(rows)


def load_predictions(path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def diagnose_run(cfg, model_id, log=print) -> dict:
    from .idvar_pipeline import audit_predictions, eval_dir, sha256_file
    from .utils import atomic_write_json
    e = cfg.experiment
    ed = eval_dir(cfg, model_id)
    art = json.loads((ed / "ARTIFACT.json").read_text())
    if not art.get("complete"):
        raise RuntimeError(f"[{e}/{model_id}] evaluation incomplete; cannot diagnose")
    dd = ed / "diagnostics"
    done = dd / "DONE.json"
    if done.exists() and json.loads(done.read_text()).get("predictions_sha256") == art["predictions_sha256"]:
        log(f"[{e}/{model_id}] diagnostics already complete -> skipping")
        return json.loads(done.read_text())
    dd.mkdir(parents=True, exist_ok=True)
    from .data import load_or_build_manifests
    man = load_or_build_manifests(cfg.manifest_dir, cfg.train.seed)
    aud = audit_predictions(Path(art["predictions"]), "identity", man.test_grids, experiment=e)
    atomic_write_json(dd / "prediction_audit.json", aud)
    if not aud["ok"]:
        raise RuntimeError(f"[{e}/{model_id}] prediction audit FAILED: {aud}")
    a = load_predictions(art["predictions"])
    singleton_consistency(a, e, model_id).to_csv(dd / "singleton_consistency.csv", index=False)
    ps, hs = singleton_error_diagnostics(a, e, model_id)
    ps.to_csv(dd / "singleton_per_string.csv", index=False)
    hs.to_csv(dd / "singleton_hypotheses.csv", index=False)
    if I.n_slots(e) == 6:
        six_op_subsequence_diagnostics(a, model_id).to_csv(dd / "six_op_subsequence_matches.csv", index=False)
    info = {"predictions_sha256": art["predictions_sha256"], "audit_ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(done, info)
    log(f"[{e}/{model_id}] diagnostics written to {dd}")
    return info


def export_run(cfg, model_id, log=print) -> Path:
    """Small per-run archive (logs, configs, metric/diagnostic CSVs, summaries). Checkpoints and prediction
    files stay in place on Drive and are referenced by the artifact index."""
    from .idvar_pipeline import eval_dir
    from .training import run_dir
    from .utils import atomic_write_json
    ed, rd = eval_dir(cfg, model_id), run_dir(cfg, model_id)
    xd = ed / "export"
    xd.mkdir(parents=True, exist_ok=True)
    dst = xd / f"{cfg.experiment}_{model_id}_seed{cfg.train.seed}.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for base in (rd, ed):
            for p in sorted(base.rglob("*")):
                rel = p.relative_to(base)
                if (p.is_file() and p.suffix in (".json", ".csv") and "chunks" not in rel.parts
                        and "export" not in rel.parts):
                    zf.write(p, Path(base.name if base == rd else "eval") / rel)
    atomic_write_json(xd / "DONE.json", {"archive": str(dst), "bytes": dst.stat().st_size})
    log(f"[{cfg.experiment}/{model_id}] export {dst} ({dst.stat().st_size / 2**20:.2f} MiB)")
    return dst


# ---------------------------------------------------------------------------
# Sequential eight-run driver
# ---------------------------------------------------------------------------

def _runtime_healthy() -> bool:
    import torch
    if not torch.cuda.is_available():
        return True
    try:
        x = torch.ones(8, device="cuda") * 2
        torch.cuda.synchronize()
        return bool(x.sum().item() == 16)
    except Exception:
        return False


def run_queue(slot_root: str, mode: str, seed: int, conditions: Sequence[str], models: Sequence[str],
              stages: Sequence[str], repo_dir=None, compare_manifests=None, chunk_grids: int = 100,
              device: str = "auto", dtype: str = "auto", microbatch=None, log=print) -> "pd.DataFrame":
    """Runs the queue SEQUENTIALLY: for each condition (in order) and each model (in order), every selected stage
    finishes before the next run starts. One model/optimizer on the GPU at a time; memory is freed between runs.
    Completed, verified stages are skipped; interrupted training resumes from state.pt; missing evaluation chunks
    are recomputed. A recoverable per-run failure is recorded and the queue continues while the runtime is
    healthy; data-integrity / audit failures stop the affected run's later stages."""
    from .config import make_experiment_config
    from .idvar_pipeline import evaluate_idvar_model
    from .training import train_model
    from .utils import free_memory
    unknown = (set(conditions) - set(CONDITIONS)) | (set(models) - set(MODELS)) | (set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown queue entries: {sorted(unknown)}")
    queue = [(e, m) for e in CONDITIONS if e in conditions for m in MODELS if m in models]
    log("Resolved queue (sequential): " + " -> ".join(f"{e}/{m}" for e, m in queue) + f" | stages {list(stages)}")
    for e in [c for c in CONDITIONS if c in conditions]:
        cfg = make_experiment_config(e, mode, slot_root, list(models), seed=seed, device=device, dtype=dtype,
                                     microbatch_size=microbatch)
        cfg.resume = True
        ctx = prepare_condition(cfg, repo_dir, compare_manifests, log)   # integrity failures raise -> stop
        root = cfg.manifest_dir.parent
        for m in [x for x in MODELS if x in models]:
            log(f"\n=== {e} (S={I.n_slots(e)}) / {m} ===")
            for stage in [s for s in STAGES if s in stages]:
                try:
                    if stage == "train":
                        r = train_model(m, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, log=log,
                                        repo_dir=repo_dir)
                        if "interrupted_at_step" in r:
                            raise RuntimeError(f"training interrupted at step {r['interrupted_at_step']}")
                    elif stage == "evaluate":
                        evaluate_idvar_model(m, ctx, log=log, chunk_grids=chunk_grids)
                    elif stage == "diagnose":
                        diagnose_run(cfg, m, log)
                    elif stage == "export":
                        export_run(cfg, m, log)
                    _set_status(root, seed, e, m, stage, "done")
                except Exception as exc:
                    traceback.print_exc()
                    _set_status(root, seed, e, m, stage, f"FAILED: {exc!r}"[:300])
                    log(f"[{e}/{m}] stage {stage} FAILED -> later stages of this run skipped")
                    free_memory()
                    if not _runtime_healthy():
                        log("Runtime unhealthy (CUDA) -> stopping the queue; restart the runtime and rerun to resume")
                        return status_table(root, seed)
                    break
                finally:
                    free_memory()
        del ctx
        free_memory()
    root = Path(slot_root) if mode == "full" else Path(slot_root) / "smoke"
    return status_table(root, seed)


# ---------------------------------------------------------------------------
# Consolidated comparison (reads saved artifacts only)
# ---------------------------------------------------------------------------

def _frame(a, experiment):
    import pandas as pd
    n = len(a["pred"])
    P, Y = a["pred"].reshape(n, 36), a["target"].reshape(n, 36)
    return pd.DataFrame({"k": a["k"].astype(int), "string": a["string"].astype(str), "kind": a["kind"].astype(str),
                         "position": a["position"].astype(int), "grid_id": a["grid_id"], "grid_hash": a["grid_hash"],
                         "exact": a["exact"].astype(bool), "cell": (P == Y).mean(1), "invalid": ~a["valid"].astype(bool),
                         "n_identity": a["n_identity"].astype(int) if "n_identity" in a else I.n_slots(experiment) - a["k"],
                         "present": a["transform_in_training"].astype(bool) if "transform_in_training" in a else np.nan})


def _agg(g):
    import pandas as pd
    return pd.Series({"exact_correct": int(g.exact.sum()), "n": len(g), "exact_acc": g.exact.mean(),
                      "cell_acc": g.cell.mean(), "invalid_rate": g.invalid.mean()})


def _logs(slot_root, e, m, seed):
    import pandas as pd
    rd = Path(slot_root) / e / f"seed_{seed}" / "runs" / m
    t = pd.read_csv(rd / "train_log.csv") if (rd / "train_log.csv").exists() else None
    v = pd.read_csv(rd / "val_log.csv") if (rd / "val_log.csv").exists() else None
    s = json.loads((rd / "train_summary.json").read_text()) if (rd / "train_summary.json").exists() else None
    rc = json.loads((rd / "run_config.json").read_text()) if (rd / "run_config.json").exists() else None
    return t, v, s, rc


def historical_lookup(results_root: Path, seed: int = 42):
    """Historical S=5 predictions for the same supports and architectures, found through the five-slot artifact
    indices (support_comparison.locate); never guessed."""
    from .support_comparison import locate
    found = locate(Path(results_root), seed)
    return {(c, m): found.get((HIST_NAME[c], m)) for c in CONDITIONS for m in MODELS}


def analyse(slot_root: str, results_root: str, seed: int = 42, log=print) -> dict:
    import pandas as pd
    slot_root = Path(slot_root)
    out = slot_root / "comparisons" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(slot_root)
    status = status_table(slot_root, seed)
    cov = I.transformation_coverage()
    by_k, single, main, six_words, icount, curves, timing, six_sub, consist = [], [], [], [], [], [], [], [], []
    frames = {}
    for e in CONDITIONS:
        S, ks = I.IDSLOT_EXPERIMENTS[e]
        no_i = I.full_run_counts(e).get(S, 0) / 1_280_000
        for m in MODELS:
            t, v, s, rc = _logs(slot_root, e, m, seed)
            if t is not None:
                for _, r in t.iterrows():
                    curves.append({"condition": e, "model": m, "step": r["step"], "series": "train answer-digit CE",
                                   "value": r.get("answer_cell_nll", np.nan)})
                    curves.append({"condition": e, "model": m, "step": r["step"], "series": "train EOS CE",
                                   "value": r.get("eos_nll", np.nan)})
            if v is not None:
                for _, r in v.iterrows():
                    curves.append({"condition": e, "model": m, "step": r["step"],
                                   "series": "validation answer-digit CE (k-macro, EOS excluded)", "value": r["val_cell_ce"]})
            a_ = idx.get(f"{e}/seed_{seed}/{m}", {})
            esum_p = slot_root / e / f"seed_{seed}" / "eval" / m / "eval_summary.json"
            esum = json.loads(esum_p.read_text()) if esum_p.exists() else {}
            if s:
                timing.append({"condition": e, "S": S, "model": m, "unique_params": s.get("unique_params"),
                               "train_minutes": s["train_seconds"] / 60, "examples_per_second":
                               s["examples_seen"] / max(1e-9, s["train_seconds"]),
                               "peak_gpu_gb_train_session": s.get("peak_gpu_memory_gb_this_session"),
                               "microbatch": s.get("microbatch_size"), "accumulation": s.get("accumulation_steps"),
                               "oom_retries": s.get("oom_retries"), "eval_minutes": (esum.get("test_eval_seconds") or np.nan) / 60,
                               "peak_gpu_gb_eval": esum.get("peak_gpu_memory_gb_eval"), "hardware": s.get("hardware"),
                               "best_step": s.get("best_step"), "best_val_digit_ce": s.get("best_val_cell_ce")})
            row = {"condition": e, "S": S, "model": m, "train_k": str(list(ks)), "no_identity_training_fraction": round(no_i, 4)}
            if not a_.get("complete"):
                row.update({"status": "not yet available", "seen_exact": NA, "seen_cell": NA, "singleton_exact": NA,
                            "singleton_cell": NA, "singleton_denominator": NA, "six_op_exact": NA})
                main.append(row)
                continue
            a = load_predictions(a_["predictions"])
            df = _frame(a, e)
            frames[(e, m)] = df
            g = df.groupby("k").apply(_agg).reset_index()
            g["status"] = np.where(g.k.isin(ks), "trained k (new grids)", "HELD OUT")
            g.insert(0, "model", m); g.insert(0, "S", S); g.insert(0, "condition", e)
            by_k.append(g)
            sg = df[df.kind == "single"].groupby(["string", "position"]).apply(_agg).reset_index()
            sg.insert(0, "model", m); sg.insert(0, "S", S); sg.insert(0, "condition", e)
            single.append(sg)
            ic = df.groupby(["n_identity", "k"]).apply(_agg).reset_index()
            ic["status"] = np.where(ic.k.isin(ks), "trained k", "HELD OUT")
            ic.insert(0, "model", m); ic.insert(0, "S", S); ic.insert(0, "condition", e)
            icount.append(ic)
            seen = df[df.kind == "seen"]
            sing = df[df.kind == "single"]
            row.update({"status": "analysed", "seen_exact": f"{int(seen.exact.sum())}/{len(seen)} ({seen.exact.mean():.4f})",
                        "seen_cell": round(seen.cell.mean(), 4),
                        "singleton_exact": f"{int(sing.exact.sum())}/{len(sing)} ({sg.exact_acc.mean():.4f} macro over {3 * S} cells)",
                        "singleton_cell": round(sing.cell.mean(), 4), "singleton_denominator": f"3*{S}*1000 = {len(sing)}"})
            if S == 6:
                sx = df[df.kind == "six"]
                pw = sx.groupby("string").agg(exact_correct=("exact", "sum"), n=("exact", "size"), cell_acc=("cell", "mean"),
                                              transform_in_training=("present", "first")).reset_index()
                pw["exact_acc"] = pw.exact_correct / pw.n
                pw.insert(0, "model", m)
                six_words.append(pw)
                row["six_op_exact"] = f"{int(sx.exact.sum())}/{len(sx)} ({sx.exact.mean():.4f})"
                for pres in (True, False):
                    q = sx[sx.present == pres]
                    row[f"six_op_exact_transform_{'IN' if pres else 'NOT_in'}_training"] = f"{int(q.exact.sum())}/{len(q)}"
                f = slot_root / e / f"seed_{seed}" / "eval" / m / "diagnostics" / "six_op_subsequence_matches.csv"
                if f.exists():
                    six_sub.append(pd.read_csv(f))
            else:
                row["six_op_exact"] = NA
            f = slot_root / e / f"seed_{seed}" / "eval" / m / "diagnostics" / "singleton_consistency.csv"
            if f.exists():
                consist.append(pd.read_csv(f))
            main.append(row)
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()  # noqa: E731
    res = {"status": status, "main": pd.DataFrame(main), "by_k": cat(by_k), "singletons": cat(single),
           "identity_count_view": cat(icount), "six_op_per_word": cat(six_words), "curves": pd.DataFrame(curves),
           "timing": pd.DataFrame(timing), "six_op_subsequence": cat(six_sub), "singleton_consistency": cat(consist),
           "coverage": cov}
    res["historical"], res["historical_meta"], res["historical_paired_seen"] = historical_comparison(
        frames, slot_root, Path(results_root), seed)
    aud = slot_root / "audits" / f"seed_{seed}"
    for name in ("initialization", "layout", "support", "stream_match", "placement_uniformity", "test_manifest"):
        p = aud / f"{name}.csv"
        res[f"audit_{name}"] = pd.read_csv(p) if p.exists() else pd.DataFrame([{"audit": name, "status": "not yet available"}])
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    res["out"] = out
    log("Run status: " + "; ".join(f"{r.condition}/{r.model}: {r.status}" for r in res["main"].itertuples()))
    return res


def _selected_step(slot_root, e, m, seed):
    p = Path(slot_root) / e / f"seed_{seed}" / "eval" / m / "eval_summary.json"
    return json.loads(p.read_text()).get("selected_step") if p.exists() else None


def historical_comparison(frames, slot_root: Path, results_root: Path, seed: int):
    """Same support, same architecture: old S=5 vs new S. Singletons use each condition's own position-average
    (denominators 3*5*1000 vs 3*S*1000); paired by (operation, grid hash) on within-grid mean correctness. Seen
    controls paired by logical (word, grid hash). Metadata (manifest, model config, recipe, stream, init)
    exposed explicitly."""
    import pandas as pd
    hist = historical_lookup(results_root, seed)
    rows, meta, paired = [], [], []
    for c in CONDITIONS:
        S = I.n_slots(c)
        for m in MODELS:
            h = hist.get((c, m))
            new = frames.get((c, m))
            base = {"condition": c, "support": HIST_NAME[c], "model": m, "S_new": S, "S_old": 5}
            if h is None or new is None:
                rows.append({**base, "status": "not yet available" + ("" if new is not None else " (new run)")
                             + ("" if h is not None else " (historical S=5 run)")})
                continue
            a = load_predictions(h["predictions"])
            old = pd.DataFrame({"string": a["string"].astype(str), "kind": a["kind"].astype(str), "k": a["k"].astype(int),
                                "position": a["position"].astype(int), "grid_hash": a["grid_hash"],
                                "exact": a["exact"].astype(bool),
                                "cell": (a["pred"].reshape(len(a["pred"]), 36) == a["target"].reshape(len(a["pred"]), 36)).mean(1)})
            os_, ns = old[old.kind == "single"], new[new.kind == "single"]
            go = os_.groupby(["string", "grid_hash"]).exact.mean().rename("old_mean")
            gn = ns.groupby(["string", "grid_hash"]).exact.mean().rename("new_mean")
            j = pd.concat([go, gn], axis=1, join="inner").reset_index()
            for op in ["A", "B", "C", "all"]:
                q = j if op == "all" else j[j.string == op]
                qo = os_ if op == "all" else os_[os_.string == op]
                qn = ns if op == "all" else ns[ns.string == op]
                rows.append({**base, "status": "compared", "op": op,
                             "old_singleton_exact": f"{int(qo.exact.sum())}/{len(qo)}", "old_exact_acc": qo.exact.mean(),
                             "old_cell_acc": qo.cell.mean(),
                             "new_singleton_exact": f"{int(qn.exact.sum())}/{len(qn)}", "new_exact_acc": qn.exact.mean(),
                             "new_cell_acc": qn.cell.mean(),
                             "old_first_position_exact": f"{int(qo[qo.position == 1].exact.sum())}/{int((qo.position == 1).sum())}",
                             "new_first_position_exact": f"{int(qn[qn.position == 1].exact.sum())}/{int((qn.position == 1).sum())}",
                             "paired_op_grid_units": len(q), "mean_within_grid_diff_new_minus_old": (q.new_mean - q.old_mean).mean(),
                             "units_new_higher": int((q.new_mean > q.old_mean).sum()),
                             "units_old_higher": int((q.new_mean < q.old_mean).sum()),
                             "units_equal": int((q.new_mean == q.old_mean).sum())})
            # seen controls paired by logical input
            sn = new[new.kind == "seen"][["string", "grid_hash", "k", "exact"]].rename(columns={"exact": "new"})
            so = old[old.kind == "seen"][["string", "grid_hash", "exact"]].rename(columns={"exact": "old"})
            jj = sn.merge(so, on=["string", "grid_hash"], validate="one_to_one")
            for k, q in jj.groupby("k"):
                paired.append({**base, "k": int(k), "paired_logical_inputs": len(q), "old_exact": int(q.old.sum()),
                               "new_exact": int(q.new.sum()), "both_correct": int((q.old & q.new).sum()),
                               "gained": int((~q.old & q.new).sum()), "lost": int((q.old & ~q.new).sum()),
                               "both_wrong": int((~q.old & ~q.new).sum())})
            # metadata
            rc_new = _logs(slot_root, c, m, seed)[3] or {}
            rc_old = json.loads(Path(h["run_config"]).read_text()) if Path(h["run_config"]).exists() else {}
            man_old = (json.loads(Path(h["manifest_json"]).read_text()).get("combined_hash")
                       if Path(h["manifest_json"]).exists() else None)
            tr_old, tr_new = rc_old.get("train", {}), rc_new.get("train", {})
            diffs = {k: (tr_old.get(k), tr_new.get(k)) for k in sorted(set(tr_old) | set(tr_new)) if tr_old.get(k) != tr_new.get(k)}
            old_dv = rc_old.get("data_version")
            stream = ("matched by construction: same per-k indexed record stream and order (verified in stream audit)"
                      if old_dv == "decomp-idext-v1" and rc_old.get("train_ks") == list(I.train_ks(c))
                      else f"UNMATCHED (historical data_version {old_dv}; different sampler)")
            meta.append({**base, "manifest_old": (man_old or "?")[:12], "manifest_new": (rc_new.get("manifest_hash") or "?")[:12],
                         "same_manifest": man_old == rc_new.get("manifest_hash"),
                         "test_grids_identical": set(old.grid_hash) == set(new.grid_hash),
                         "model_hash_old": rc_old.get("model_hash"), "model_hash_new": rc_new.get("model_hash"),
                         "model_config_differences": json.dumps({k: (rc_old.get("model_config", {}).get(k), v)
                                                                 for k, v in rc_new.get("model_config", {}).items()
                                                                 if rc_old.get("model_config", {}).get(k) != v}),
                         "recipe_differences": json.dumps(diffs) if diffs else "none",
                         "params_old": rc_old.get("model_params"), "params_new": rc_new.get("model_params"),
                         "selected_step_old": h.get("selected_step"),
                         "selected_step_new": _selected_step(slot_root, c, m, seed), "training_stream": stream,
                         "initialization": "shared tensors = historical construction (see audit_initialization); "
                                           "position table grown 82->83"})
    return pd.DataFrame(rows), pd.DataFrame(meta), pd.DataFrame(paired)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _na_panel(ax, text="not yet available"):
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666")


def figures(res, out: Path) -> List[Path]:
    import pandas as pd
    plt = _plt()
    out = Path(out)
    paths = []
    # b. training objective + validation CE
    cv = res["curves"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 7.5), squeeze=False)
    for j, e in enumerate(CONDITIONS):
        for i, series in enumerate([("train answer-digit CE", "train EOS CE"), ("validation answer-digit CE (k-macro, EOS excluded)",)]):
            ax = axes[i, j]
            q = cv[(cv.condition == e)] if len(cv) else cv
            if not len(q):
                _na_panel(ax); continue
            for m, ls in zip(MODELS, ("-", "--")):
                for sname in series:
                    r = q[(q.model == m) & (q.series == sname)]
                    if len(r):
                        ax.plot(r.step, r.value.clip(lower=1e-6), ls=ls, lw=1.3,
                                color=COLORS[e] if "EOS" not in sname else "#999",
                                label=f"{DISPLAY_NAMES[m][:16]}: {sname}")
            ax.set(yscale="log", xlabel="optimizer update", title=f"{e} (S={I.n_slots(e)})")
            ax.legend(fontsize=6)
    fig.suptitle("Training objective (answer digits and EOS reported separately) and checkpoint-selection validation CE")
    fig.tight_layout()
    paths += save_fig(fig, out, "training_and_validation_curves", cv if len(cv) else None)
    # c. accuracy by k
    bk = res["by_k"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), squeeze=False)
    for ax, m in zip(axes[0], MODELS):
        q = bk[bk.model == m] if len(bk) else bk
        if not len(q):
            _na_panel(ax); continue
        for e in CONDITIONS:
            s = q[q.condition == e].sort_values("k")
            if not len(s):
                continue
            ax.plot(s.k, s.exact_acc, color=COLORS[e], lw=1.5, label=f"{e} (S={I.n_slots(e)})")
            tr = s.status != "HELD OUT"
            ax.scatter(s.k[tr], s.exact_acc[tr], color=COLORS[e], s=50, zorder=3)
            ax.scatter(s.k[~tr], s.exact_acc[~tr], facecolors="white", edgecolors=COLORS[e], s=50, lw=1.5, zorder=3)
        ax.set(xticks=range(7), ylim=(-0.02, 1.05), xlabel="k (A/B/C tokens); evaluated k <= S only",
               ylabel="exact-grid accuracy", title=f"{DISPLAY_NAMES[m]}: filled = trained k, open = HELD OUT")
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths += save_fig(fig, out, "accuracy_by_k", bk if len(bk) else None)
    # d. singleton heatmaps at each actual S
    sg = res["singletons"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 5.6), squeeze=False, gridspec_kw={"width_ratios": [2, 3, 4, 6]})
    for i, m in enumerate(MODELS):
        for j, e in enumerate(CONDITIONS):
            ax = axes[i, j]
            S = I.n_slots(e)
            r = sg[(sg.condition == e) & (sg.model == m)] if len(sg) else sg
            if not len(r):
                _na_panel(ax, f"{e}/{m}\nnot yet available"); continue
            mat = np.array([[int(r[(r.string == op) & (r.position == p)].exact_correct.iloc[0]) for p in range(1, S + 1)]
                            for op in "ABC"])
            den = int(r.n.iloc[0])
            ax.grid(False)
            ax.imshow(mat / den, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for a_ in range(3):
                for b_ in range(S):
                    ax.text(b_, a_, str(mat[a_, b_]), ha="center", va="center", fontsize=7,
                            color="white" if mat[a_, b_] > 0.6 * den else "#0b0b0b")
            ax.set_xticks(range(S)); ax.set_xticklabels([f"s{p}" for p in range(1, S + 1)], fontsize=7)
            ax.set_yticks(range(3)); ax.set_yticklabels(list("ABC"))
            ax.set_title(f"{DISPLAY_NAMES[m][:18]} / S={S}", fontsize=8)
    fig.suptitle("Singletons (held out everywhere): exact successes of 1000 per operation x physical slot (actual S)")
    fig.tight_layout()
    paths += save_fig(fig, out, "singleton_heatmaps", sg if len(sg) else None)
    # e. historical S=5 vs new S at the same support
    h = res["historical"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), squeeze=False)
    for ax, metric in zip(axes[0], ("exact_acc", "cell_acc")):
        q = h[(h.get("status", pd.Series(dtype=str)) == "compared") & (h.op == "all")] if len(h) and "op" in h else pd.DataFrame()
        if not len(q):
            _na_panel(ax, "historical or new runs not yet available"); continue
        lab = [f"{r.support}\n{DISPLAY_NAMES[r.model][:12]}" for r in q.itertuples()]
        x = np.arange(len(q))
        ax.bar(x - 0.2, q[f"old_{metric}"], 0.4, color="#bbbbbb", label="old S=5 (denominator 15000)")
        ax.bar(x + 0.2, q[f"new_{metric}"], 0.4, color=[COLORS[c] for c in q.condition],
               label="new S (denominator 3*S*1000)")
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7)
        ax.set(ylim=(0, 1.05), ylabel=f"singleton {metric.replace('_', ' ')} (position-averaged)",
               title=f"Same support, same architecture: S=5 vs new S ({metric})")
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths += save_fig(fig, out, "historical_S5_vs_new_singletons", h if len(h) else None)
    # f. six-operation per-word distribution
    sw = res["six_op_per_word"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), squeeze=False)
    for ax, m in zip(axes[0], MODELS):
        q = sw[sw.model == m] if len(sw) else sw
        if not len(q):
            _na_panel(ax, "S=6 run not yet available"); continue
        bins = np.linspace(0, 1, 21)
        for pres, col in ((True, "#2a78d6"), (False, "#eb6834")):
            r = q[q.transform_in_training == pres]
            ax.hist(r.exact_acc, bins=bins, color=col, alpha=0.75,
                    label=f"transform {'IN' if pres else 'NOT in'} training ({len(r)} words; "
                          f"{int(r.exact_correct.sum())}/{int(r.n.sum())})")
        ax.set(xlabel="per-word exact accuracy over 1000 grids", ylabel="words",
               title=f"{DISPLAY_NAMES[m]}: S=6, all 729 six-operation words (no identity)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths += save_fig(fig, out, "six_op_per_word_distribution", sw if len(sw) else None)
    # g. identity-count view (one line per condition; supports not merged)
    ic = res["identity_count_view"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), squeeze=False)
    for ax, m in zip(axes[0], MODELS):
        q = ic[ic.model == m] if len(ic) else ic
        if not len(q):
            _na_panel(ax); continue
        for e in CONDITIONS:
            s = q[q.condition == e].sort_values("n_identity")
            if not len(s):
                continue
            ax.plot(s.n_identity, s.exact_acc, color=COLORS[e], lw=1.3, label=f"{e}")
            tr = s.status == "trained k"
            ax.scatter(s.n_identity[tr], s.exact_acc[tr], color=COLORS[e], s=40, zorder=3)
            ax.scatter(s.n_identity[~tr], s.exact_acc[~tr], facecolors="white", edgecolors=COLORS[e], s=40, zorder=3)
        ax.set(xlabel="identity tokens in the instruction (S - k)", ylabel="exact accuracy", ylim=(-0.02, 1.05),
               title=f"{DISPLAY_NAMES[m]}: by identity count (descriptive; open = held out)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths += save_fig(fig, out, "identity_count_view", ic if len(ic) else None)
    return paths


def export_comparisons(res, log=print) -> Path:
    out = Path(res["out"])
    dst = out / "comparisons.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p != dst and p.suffix in (".csv", ".png", ".pdf", ".json"):
                zf.write(p, p.relative_to(out))
    log(f"comparison archive: {dst} ({dst.stat().st_size / 2**20:.1f} MiB)")
    return dst
