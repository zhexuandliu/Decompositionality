"""Sanity checks and the bounded small-dataset overfit DIAGNOSTIC.

The overfit diagnostic trains repeatedly on a small fixed set of allowed length-two
examples drawn from a separate RNG stream (``STREAM_OVERFIT``). It never touches the
benchmark training stream or checkpoints and writes to its own directory. Its purpose is
to distinguish a broken pipeline from a generalization result; it is not a benchmark.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import operations as O
from .config import TrainConfig
from .data import overfit_examples
from .evaluation import run_model_on_set
from .models.registry import build_model
from .training import StepRunner, make_optimizer, make_scaler, make_scheduler
from .utils import atomic_write_json, free_memory, resolve_amp_dtype, resolve_device


def geometry_checks() -> dict:
    """Fast checks of the operation definitions (also covered by the unit tests)."""
    lab = np.arange(36).reshape(6, 6)
    x = np.random.default_rng(0).integers(0, 10, (6, 6))
    A, B, C = (O.OP_FUNCTIONS[i] for i in range(3))
    for r in range(6):
        for c in range(6):
            assert A(lab)[r, c] == lab[r, (c - 1) % 6]
            assert B(lab)[r, c] == lab[5 - c, r]
            assert C(lab)[r, c] == lab[r, 5 - c]
    y = x.copy()
    for _ in range(6):
        y = A(y)
    assert np.array_equal(y, x)
    assert np.array_equal(B(B(B(B(x)))), x) and np.array_equal(C(C(x)), x)
    assert not np.array_equal(O.apply_string(lab, "AB"), O.apply_string(lab, "BA"))
    assert np.array_equal(O.apply_string(lab, "AB"), B(A(lab)))
    O.check_test_string_counts()
    return {"ok": True, "A(lab)[0]": A(lab)[0].tolist(), "B(lab)[0]": B(lab)[0].tolist(), "C(lab)[0]": C(lab)[0].tolist()}


def overfit_diagnostic(model_id: str, experiment: str = "distinct_pairs", n_examples: int = 64, steps: int = 300,
                       lr: float = 1e-3, seed: int = 42, device: str = "auto", dtype: str = "auto",
                       out_dir: Optional[str] = None, heldout_bytes: Optional[set] = None, log=print) -> dict:
    """Full-batch training on ``n_examples`` fixed length-two examples; reports training-set
    exact accuracy (free-running for AR). Short warmup then constant LR."""
    strings = O.TRAIN_SETS[experiment]
    ex = overfit_examples(seed, strings, n_examples, heldout_bytes or set())
    tc = replace(TrainConfig(), seed=seed, optimizer_steps=steps, effective_batch_size=n_examples, lr=lr,
                 min_lr=lr, warmup_steps=min(20, steps), device=device, dtype=dtype)
    dev = resolve_device(device)
    amp = resolve_amp_dtype(dev, dtype)
    model = build_model(model_id, seed=seed).to(dev)
    opt = make_optimizer(model, tc)
    runner = StepRunner(model, opt, make_scheduler(opt, tc), make_scaler(amp), tc, dev, amp,
                        microbatch=n_examples if dev.type == "cuda" else min(32, n_examples), log=log)
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        m = runner(ex)
        if step % max(1, steps // 10) == 0 or step == steps:
            r = run_model_on_set(model, ex, dev, amp, want_ce=False)
            hist.append({"step": step, "loss": m["loss"], "train_exact": float(r["exact"].mean()),
                         "train_cell_acc": float(r["cell_correct"].mean() / 36)})
            log(f"  [{model_id} overfit] step {step} loss {m['loss']:.4f} exact {hist[-1]['train_exact']:.3f} "
                f"cell {hist[-1]['train_cell_acc']:.3f}")
    res = {"model_id": model_id, "experiment": experiment, "n_examples": n_examples, "steps": steps, "lr": lr,
           "final": hist[-1], "history": hist, "seconds": time.time() - t0, "device": str(dev),
           "note": "debugging diagnostic on repeated examples; NOT part of the benchmark"}
    if out_dir:
        atomic_write_json(Path(out_dir) / f"overfit_{experiment}_{model_id}.json", res)
    del model, opt, runner
    free_memory()
    return res
