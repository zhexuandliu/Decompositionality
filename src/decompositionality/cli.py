"""Command-line entry points.

Examples::

    python -m decompositionality.cli counts
    python -m decompositionality.cli smoke --experiment distinct_pairs --models vit trm
    python -m decompositionality.cli train --experiment all_pairs --model transformer_ar --output-root results
    python -m decompositionality.cli evaluate --experiment all_pairs --model transformer_ar
    python -m decompositionality.cli plot --experiment all_pairs
    python -m decompositionality.cli compare --results-root results
    python -m decompositionality.cli overfit --model transformer_encoder --steps 300
"""
from __future__ import annotations

import argparse
import json
import sys

from .models.registry import MODEL_IDS, model_summary


def _cfg(args, mode=None):
    from .config import make_experiment_config
    overrides = {}
    if getattr(args, "steps", None):
        overrides["optimizer_steps"] = args.steps
    if getattr(args, "device", None):
        overrides["device"] = args.device
    if getattr(args, "dtype", None):
        overrides["dtype"] = args.dtype
    if getattr(args, "microbatch", None):
        overrides["microbatch_size"] = args.microbatch
    models = getattr(args, "models", None) or ([args.model] if getattr(args, "model", None) else None)
    cfg = make_experiment_config(args.experiment, mode or args.mode, args.output_root, models, **overrides)
    cfg.resume = not getattr(args, "no_resume", False)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="decompositionality")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, multi=False):
        sp.add_argument("--experiment", default="distinct_pairs", choices=["distinct_pairs", "all_pairs"])
        sp.add_argument("--output-root", default="results")
        sp.add_argument("--mode", default="full", choices=["full", "smoke"])
        sp.add_argument("--device", default=None)
        sp.add_argument("--dtype", default=None, choices=["auto", "bf16", "fp16", "fp32"])
        if multi:
            sp.add_argument("--models", nargs="+", choices=MODEL_IDS, default=None)
        else:
            sp.add_argument("--model", required=True, choices=MODEL_IDS)

    sub.add_parser("counts", help="print parameter counts of all nine models")
    sp = sub.add_parser("smoke", help="tiny end-to-end run with the FULL architectures")
    common(sp, multi=True)
    sp = sub.add_parser("train", help="train one model (resumable), then evaluate it")
    common(sp)
    sp.add_argument("--steps", type=int, default=None)
    sp.add_argument("--microbatch", type=int, default=None)
    sp.add_argument("--no-resume", action="store_true")
    sp.add_argument("--no-eval", action="store_true")
    sp = sub.add_parser("run", help="train + evaluate all selected models, then build reports")
    common(sp, multi=True)
    sp = sub.add_parser("evaluate", help="evaluate the selected checkpoint of one model")
    common(sp)
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("plot", help="build figures/tables for one experiment")
    common(sp, multi=True)
    sp = sub.add_parser("compare", help="cross-experiment comparison")
    sp.add_argument("--results-root", default="results")
    sp = sub.add_parser("overfit", help="small-dataset overfit diagnostic (not a benchmark)")
    sp.add_argument("--model", required=True, choices=MODEL_IDS)
    sp.add_argument("--experiment", default="distinct_pairs")
    sp.add_argument("--n-examples", type=int, default=64)
    sp.add_argument("--steps", type=int, default=300)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--out-dir", default="results/diagnostics")
    args = p.parse_args(argv)

    if args.cmd == "counts":
        for m in MODEL_IDS:
            s = model_summary(m)
            print(f"{m:28s} unique={s['unique_params']:>12,}  shared_core={s['shared_core_params']}  "
                  f"untied_equiv={s['untied_equivalent_params']}")
        return 0
    from . import pipeline, reports
    if args.cmd in ("smoke", "run"):
        cfg = _cfg(args, "smoke" if args.cmd == "smoke" else None)
        ctx = pipeline.prepare(cfg)
        pipeline.train_and_evaluate(ctx)
        reports.build_reports(cfg.experiment_dir, cfg.train_strings)
        pipeline.make_archive(cfg)
        return 0
    if args.cmd == "train":
        cfg = _cfg(args)
        ctx = pipeline.prepare(cfg)
        pipeline.train_and_evaluate(ctx, [args.model], evaluate=not args.no_eval)
        return 0
    if args.cmd == "evaluate":
        cfg = _cfg(args)
        ctx = pipeline.prepare(cfg)
        print(json.dumps(pipeline.evaluate_model(args.model, ctx, force=args.force)["test"], indent=1))
        return 0
    if args.cmd == "plot":
        cfg = _cfg(args)
        reports.build_reports(cfg.experiment_dir, cfg.train_strings)
        return 0
    if args.cmd == "compare":
        from .compare import compare
        compare(args.results_root)
        return 0
    if args.cmd == "overfit":
        from .diagnostics import overfit_diagnostic
        overfit_diagnostic(args.model, args.experiment, args.n_examples, args.steps, args.lr,
                           device=args.device, out_dir=args.out_dir)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
