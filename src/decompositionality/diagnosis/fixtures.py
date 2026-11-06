"""SYNTHETIC result folders with planted behaviours, for tests and local integration runs.

These files mimic the saved-artifact schema of the real pipeline but contain NO real model output.
Never confuse fixture results with analysis of actual trained models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .. import operations as O
from . import group as G

PLANTED = {
    # model id (reused only to mimic folder layout) -> planted behaviour
    "transformer_ar": "seen strings correct; odd-length strings -> Z o truth; unseen pairs -> identity; 3% invalid tokens",
    "transformer_encoder": "first operation only (singles correct)",
    "trm": "seen strings correct; everything else iid random digits",
    "rescnn_film": "perfect oracle on all strings",
}


def _behaviour(model: str, s: str, x: np.ndarray, train: set, rng) -> np.ndarray:
    cat_word = s
    if model == "rescnn_film":
        return G.oracle_apply_string(x, s)
    if model == "transformer_encoder":
        return G.oracle_apply_string(x, s[0])
    if s in train:
        return G.oracle_apply_string(x, s)
    if model == "transformer_ar":
        if len(s) % 2 == 1:
            return G.apply_perm(G.Z_SRC, G.oracle_apply_string(x, s))
        return x.copy()
    return rng.integers(0, 10, size=36)


def write_fixture_results(root: Path, n_grids: int = 40, seed: int = 0) -> Dict[str, str]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    grids = rng.integers(0, 10, (n_grids, 36)).astype(np.int8)
    dirs = {}
    for exp, train in O.TRAIN_SETS.items():
        d = root / exp
        (d / "manifests").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "manifests" / "heldout_grids.npz", val_grids=grids[:5].reshape(-1, 6, 6),
                            test_grids=grids.reshape(-1, 6, 6))
        (d / "manifests" / "manifest.json").write_text(json.dumps({"combined_hash": "synthetic-fixture"}))
        for m in PLANTED:
            mrng = np.random.default_rng([seed, len(exp), len(m)])
            X, Y, P, gid, S = [], [], [], [], []
            for g in range(n_grids):
                for s in O.TEST_STRINGS:
                    x = grids[g]
                    X.append(x); Y.append(G.oracle_apply_string(x, s)); gid.append(g); S.append(s)
                    P.append(_behaviour(m, s, x, set(train), mrng))
            P = np.array(P, dtype=np.int8)
            if m == "transformer_ar":
                bad = mrng.random(len(P)) < 0.03
                P[bad, mrng.integers(0, 36, bad.sum())] = -1
            X, Y = np.array(X, np.int8), np.array(Y, np.int8)
            order = mrng.permutation(len(P))  # shuffled file order on purpose
            valid = (P >= 0).all(1)
            cc = (P == Y).sum(1)
            ev = d / "eval" / m
            ev.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ev / "test_predictions.npz", pred=P[order].reshape(-1, 6, 6), target=Y[order].reshape(-1, 6, 6),
                                input=X[order].reshape(-1, 6, 6), grid_id=np.array(gid, np.int32)[order],
                                string=np.array(S, dtype="U3")[order], valid=valid[order],
                                cell_correct=cc[order].astype(np.int16), exact=(valid & (cc == 36))[order],
                                cell_ce=np.zeros(len(P), np.float32)[order])
            dec = "greedy free-running (+1 EOS diagnostic step)" if m == "transformer_ar" else "parallel argmax"
            (ev / "eval_summary.json").write_text(json.dumps({"model_id": m, "selected_step": 5000, "decoding": dec,
                                                               "repo_commit": "synthetic-fixture", "config_hash": "fixture"}))
            rn = d / "runs" / m
            rn.mkdir(parents=True, exist_ok=True)
            (rn / "run_config.json").write_text(json.dumps({"model_id": m, "synthetic_fixture": True}))
        dirs[exp] = str(d)
    return dirs
