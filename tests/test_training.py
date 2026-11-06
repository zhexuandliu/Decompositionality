from dataclasses import replace

import numpy as np
import pytest
import torch

from decompositionality import operations as O
from decompositionality.config import TrainConfig, make_experiment_config
from decompositionality.data import TrainStream
from decompositionality.evaluation import group_table, per_string_table, score_predictions, to_device
from decompositionality.models.registry import build_model
from decompositionality.training import lr_factor
from conftest import TINY


def test_lr_schedule():
    tc = TrainConfig()
    assert lr_factor(0, tc) == pytest.approx(1 / 250)
    assert lr_factor(249, tc) == pytest.approx(1.0)
    assert lr_factor(250, tc) == pytest.approx(1.0)
    assert lr_factor(4999, tc) * tc.lr == pytest.approx(3e-5)
    vals = [lr_factor(u, tc) for u in range(250, 5000)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("mid", ["transformer_ar", "trm", "rescnn_film"])
def test_gradient_accumulation_independent_of_microbatch(mid):
    tc = replace(TrainConfig(), effective_batch_size=16)
    batch = TrainStream(42, O.TRAIN_SETS["all_pairs"], 16).batch(0)
    grads = []
    for mb in (16, 4, 5):
        m = build_model(mid, TINY[mid], seed=0)
        if mid == "rescnn_film":
            torch.nn.init.normal_(m.cond.film.weight, std=0.1)
        # same weighting as StepRunner._attempt, without the optimizer step
        B = 16
        for s in range(0, B, mb):
            sl = slice(s, min(B, s + mb))
            loss, _ = m.training_loss(to_device(batch, torch.device("cpu"), sl))
            (loss * (sl.stop - sl.start) / B).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6, rtol=1e-4)
    assert torch.allclose(grads[0], grads[2], atol=1e-6, rtol=1e-4)


def test_metrics_invalid_and_empty_groups():
    y = np.zeros((3, 6, 6), dtype=np.uint8)
    pred = np.zeros((3, 36), dtype=np.int8)
    pred[1, 5] = -1  # invalid token
    pred[2, 0] = 3   # wrong digit
    valid = np.array([True, False, True])
    s = score_predictions(pred, valid, y)
    assert s["exact"].tolist() == [True, False, False]
    assert s["cell_correct"].tolist() == [36, 35, 35]
    res = {"exact": s["exact"], "cell_correct": s["cell_correct"], "valid": valid}
    ps = per_string_table(np.array(["AA", "AA", "B"]), res, O.TRAIN_SETS["all_pairs"])
    gt = group_table(ps, O.TRAIN_SETS["all_pairs"]).set_index("group")
    assert np.isnan(gt.loc["pairs_unseen", "exact_acc"])  # empty group -> N/A, not 0
    assert gt.loc["pairs_seen", "exact_acc"] == pytest.approx(0.5)
    assert gt.loc["singles", "exact_acc"] == 0.0


def test_smoke_config_keeps_full_architecture():
    cfg = make_experiment_config("all_pairs", "smoke", "/tmp/x")
    assert cfg.model_overrides == {} and cfg.train.optimizer_steps < 50
    full = make_experiment_config("distinct_pairs")
    assert full.train.optimizer_steps == 5000 and full.train.effective_batch_size == 256 and full.train.seed == 42
    assert full.n_val_grids == 300 and full.n_test_grids == 1000 and full.val_subset_grids == 32
    assert len(full.models) == 9
