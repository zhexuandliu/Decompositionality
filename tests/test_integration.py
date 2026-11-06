"""End-to-end: train -> checkpoint -> interrupt/resume -> evaluate -> plots -> compare,
with tiny model fixtures and tiny budgets (CPU)."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from decompositionality import pipeline, reports
from decompositionality.compare import compare
from decompositionality.config import make_experiment_config
from decompositionality.training import train_model
from conftest import TINY

MODELS = ["transformer_ar", "trm", "unet_film"]


def cfg_for(root, exp, **kw):
    cfg = make_experiment_config(exp, "smoke", str(root), MODELS, device="cpu", **kw)
    cfg.model_overrides = {m: TINY[m] for m in MODELS}
    return cfg


def test_resume_equivalence(tmp_path):
    # uninterrupted run
    a = cfg_for(tmp_path / "a", "distinct_pairs")
    ctx_a = pipeline.prepare(a, log=lambda *_: None)
    train_model("transformer_ar", a, ctx_a.manifests, ctx_a.val_set, ctx_a.subset_set, log=lambda *_: None)
    # interrupted after 2 steps, resumed with a DIFFERENT microbatch size
    b = cfg_for(tmp_path / "b", "distinct_pairs", microbatch_size=16)
    ctx_b = pipeline.prepare(b, log=lambda *_: None)
    r = train_model("transformer_ar", b, ctx_b.manifests, ctx_b.val_set, ctx_b.subset_set, log=lambda *_: None,
                    stop_after_steps=2)
    assert r["interrupted_at_step"] == 2
    b.train = replace(b.train, microbatch_size=4)
    train_model("transformer_ar", b, ctx_b.manifests, ctx_b.val_set, ctx_b.subset_set, log=lambda *_: None)
    sa = torch.load(tmp_path / "a/smoke/distinct_pairs/runs/transformer_ar/state.pt", weights_only=False)
    sb = torch.load(tmp_path / "b/smoke/distinct_pairs/runs/transformer_ar/state.pt", weights_only=False)
    assert sa["global_step"] == sb["global_step"] == a.train.optimizer_steps
    for k in sa["model"]:
        assert torch.allclose(sa["model"][k], sb["model"][k], atol=1e-5, rtol=1e-4), k
    assert [v["step"] for v in sa["logs"]["val"]] == [v["step"] for v in sb["logs"]["val"]]
    # a mismatched configuration must refuse to resume
    c = cfg_for(tmp_path / "b", "distinct_pairs", lr=1e-3)
    with pytest.raises(RuntimeError):
        train_model("transformer_ar", c, ctx_b.manifests, ctx_b.val_set, ctx_b.subset_set, log=lambda *_: None)


def test_end_to_end_both_experiments_and_compare(tmp_path):
    for exp in ("distinct_pairs", "all_pairs"):
        cfg = cfg_for(tmp_path, exp)
        ctx = pipeline.prepare(cfg, log=lambda *_: None)
        pipeline.train_and_evaluate(ctx, log=lambda *_: None)
        # second call skips completed runs after verifying configs/artifacts
        out = pipeline.train_and_evaluate(ctx, log=lambda *_: None)
        assert set(out) == set(MODELS)
        d = cfg.experiment_dir
        for m in MODELS:
            assert (d / "runs" / m / "best.pt").exists() and (d / "runs" / m / "state.pt").exists()
            ex = pd.read_csv(d / "eval" / m / "test_examples.csv.gz")
            assert len(ex) == cfg.n_test_grids * 39
            assert set(ex.columns) >= {"experiment", "model", "seed", "grid_id", "string", "valid", "cell_correct", "exact"}
            ps = pd.read_csv(d / "eval" / m / "test_per_string.csv")
            assert len(ps) == 39
            val = json.loads((d / "runs" / m / "train_summary.json").read_text())
            assert val["examples_seen"] == cfg.train.optimizer_steps * cfg.train.effective_batch_size
        reports.build_reports(d, cfg.train_strings, log=lambda *_: None)
        for f in ("a_training_curves.png", "b_group_exact_accuracy.pdf", "b_group_cell_accuracy.png",
                  "c_per_string_exact_heatmap.png", "d_shared_examples.png", "d_failure_examples.png"):
            assert (d / "figures" / f).exists(), f
        assert (d / "tables" / "e_resource_table.csv").exists() and (d / "tables" / "f_numerical_summary.md").exists()
        groups = pd.read_csv(d / "eval" / "trm" / "test_groups.csv").set_index("group")
        if exp == "all_pairs":
            assert np.isnan(groups.loc["pairs_unseen", "exact_acc"])
        pipeline.make_archive(cfg, log=lambda *_: None)
    res = compare(tmp_path / "smoke", log=lambda *_: None)
    assert res is not None
    assert len(res["per_string"]) == 39 * len(MODELS)
    assert (res["out_dir"] / "compare_group_delta.png").exists()


def test_compare_missing_folder(tmp_path):
    msgs = []
    assert compare(tmp_path, log=msgs.append) is None
    assert "missing" in msgs[0]
