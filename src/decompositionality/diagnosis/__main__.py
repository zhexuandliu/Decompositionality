"""CLI: python -m decompositionality.diagnosis --results-root <root> [--output-dir DIR]

Reads <root>/distinct_pairs and <root>/all_pairs (saved predictions) and writes the diagnostics to
<root>/diagnostics_generalization_failures (default). Never trains; frozen inference only with
--run-frozen-inference-if-missing."""
from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DiagnosticsConfig, run_diagnostics


def main(argv=None):
    p = argparse.ArgumentParser(prog="decompositionality.diagnosis")
    p.add_argument("--results-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--split-seed", type=int, default=20251106)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-checkpoint-hash", action="store_true")
    p.add_argument("--run-frozen-inference-if-missing", action="store_true")
    a = p.parse_args(argv)
    root = Path(a.results_root)
    cfg = DiagnosticsConfig(
        experiment_dirs={e: str(root / e) for e in ("distinct_pairs", "all_pairs")},
        output_dir=a.output_dir or str(root / "diagnostics_generalization_failures"),
        models=a.models or None, chunk_size=a.chunk_size, use_cache=not a.no_cache, split_seed=a.split_seed,
        hash_checkpoints=not a.no_checkpoint_hash, run_frozen_inference_if_missing=a.run_frozen_inference_if_missing)
    run_diagnostics(cfg, make_figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
