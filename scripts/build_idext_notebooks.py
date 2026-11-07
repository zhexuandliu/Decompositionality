"""Generate notebooks/12_identity_train02_ar_extrapolation.ipynb and 13_identity_train023_ar_extrapolation.ipynb from
ONE template (they differ only in the training support). Refuses to overwrite notebooks with saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = {"idext_train02": ("12_identity_train02_ar_extrapolation.ipynb", "{0, 2}", "{1, 3, 4, 5}",
                        "128 + 128 per update (640,000 each)"),
      "idext_train023": ("13_identity_train023_ar_extrapolation.ipynb", "{0, 2, 3}", "{1, 4, 5}",
                         "85 each + 1 extra rotating 0,2,3 (426,667 / 426,667 / 426,666)"),
      "idext_train0234": ("14_identity_train0234_ar_extrapolation.ipynb", "{0, 2, 3, 4}", "{1, 5}",
                          "64 per k per update (320,000 each)")}

INTRO = """# Operation-count extrapolation in the five-slot identity format — train k ∈ {train}, held out k ∈ {held}

Transformer AR and Looped AR only, configured exactly as in the successful identity_02345 run (notebook 10). Every
instruction has FIVE visible operation tokens: k A/B/C tokens in random ordered slots, with the other 5-k slots
filled by the learnable `<identity>` token. There is no PAD.

* **Extrapolation is in k**, the number of non-identity operations, NOT in sequence length or position indices. The
  AR sequence is always 82 tokens.
* **Budget:** seed 42, 5000 updates x 256 = 1,280,000 records. Allocation: {alloc}. Within k, words and slot subsets
  are uniform.
* **Selection:** validation uses ONLY the trained k (300 grids each); singletons and held-out k never influence
  selection.
* **Test:** the shared notebook-10 manifest (376,000 records, including all 15 singleton placements). AR decoding is
  free-running.
* **Comparing 12, 13 and 14:** they have the same total budget but different per-k exposure (each larger support
  replaces some exposure of the smaller k). Initial weights are identical per architecture across all three.
* **Post-hoc analyses:**
  * training-transformation coverage, since a held-out word can still induce a seen transformation;
  * the even-count ambiguity: with H = shift (3,3), H∘A, H∘B, H∘C reproduce every even-k mapping exactly, so train02
    data alone cannot rule this alternative out, and k = 3 supervision rejects it.

One seed: descriptive only.
"""

CONFIG = r'''# --- CONFIGURATION (the only cell to edit) --------------------------------------------------------------------
EXPERIMENT = "{exp}"            # fixed for this notebook
MODE = os.environ.get("DECOMP_MODE", "full")   # "full" | "smoke" (tiny CPU check, separate paths, NOT results)
SEED = int(os.environ.get("DECOMP_SEED", 42))  # configurable for later replication; no automatic seed sweep
MODELS = ["transformer_ar", "looped_transformer_ar"]
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE
    else "/content/decompositionality_results" if IN_COLAB else str(REPO_DIR / "results"))
EXT_ROOT = f"{{RESULTS_ROOT}}/identity_ar_count_extrapolation"
RESUME = True
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
MICROBATCH = None
EVAL_CHUNK_GRIDS = 100
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
from decompositionality.config import make_experiment_config
from decompositionality import idext as E
cfg = make_experiment_config(EXPERIMENT, MODE, EXT_ROOT, MODELS, seed=SEED, device=DEVICE, dtype=DTYPE,
                             microbatch_size=MICROBATCH)
cfg.resume = RESUME
FAMILY_ROOT = cfg.manifest_dir.parent
expected = dict(lr=3e-4, weight_decay=0.01, warmup_steps=250, min_lr=3e-5, grad_clip=1.0, val_every=250)
print("resolved recipe:", cfg.train.semantic_dict())
print("discrepancies from expected:", {{k: (getattr(cfg.train, k), v) for k, v in expected.items() if getattr(cfg.train, k) != v}} or "none")
if MODE == "smoke":
    print("*** SMOKE MODE: tiny integration run; NOT scientific results ***")
print(f"{{EXPERIMENT}}: train k={{list(E.train_ks(EXPERIMENT))}} | steps={{cfg.train.optimizer_steps}} | batch={{cfg.train.effective_batch_size}}"
      f" | seed={{SEED}} | runs -> {{cfg.experiment_dir}}")
'''

ENV = r'''import json, torch
from decompositionality.utils import environment_info
print(json.dumps(environment_info(REPO_DIR), indent=1))
'''

AUDIT = r'''# --- Audits: supports, allocation, serialization, initialization, transformation coverage, H ambiguity -----------
import pandas as pd
from IPython.display import Image, Markdown, display
from decompositionality.idext_reports import (allocation_audit, h_ambiguity_check, init_audit, prepare_idext,
                                              serialization_audit, transformation_coverage)
AUD = cfg.experiment_dir / "audits"; AUD.mkdir(parents=True, exist_ok=True)
print(E.check_supports())
for name, df in (("sample_allocation", allocation_audit(cfg.train.effective_batch_size, cfg.train.optimizer_steps)),
                 ("serialization", serialization_audit()), ("initialization", init_audit(SEED))):
    display(df); df.to_csv(AUD / f"{name}.csv", index=False)
cov, cov_summ = transformation_coverage(); display(cov_summ)
print(h_ambiguity_check())
ctx = prepare_idext(cfg, repo_dir=str(REPO_DIR), legacy_manifest=f"{RESULTS_ROOT}/distinct_pairs/manifests/manifest.json",
                    idvar_manifest=f"{RESULTS_ROOT}/identity_varlen_02345_all_models/manifests/manifest.json")
'''

TRAIN = r'''# --- Train both AR models sequentially (fresh identical init per architecture; resumable) -----------------------
from decompositionality.training import train_model
from decompositionality.utils import free_memory
failures = {}
for mid in cfg.models:
    print(f"\n=== training {EXPERIMENT}/{mid} ===")
    try:
        train_model(mid, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, repo_dir=str(REPO_DIR))
    except Exception as exc:
        import traceback; traceback.print_exc(); failures[mid] = repr(exc)
    finally:
        free_memory()
if failures:
    print("TRAINING FAILED:", failures, "-> rerun to resume")
'''

EVAL = r'''# --- Free-running evaluation of the SELECTED checkpoints on the common 376,000-record test manifest -------------
from decompositionality.idvar_pipeline import evaluate_idvar_model, audit_predictions
from decompositionality.idext_reports import build_index
for mid in cfg.models:
    try:
        evaluate_idvar_model(mid, ctx, chunk_grids=EVAL_CHUNK_GRIDS)
    except Exception as exc:
        import traceback; traceback.print_exc()
    finally:
        free_memory()
idx = build_index(FAMILY_ROOT)
aud = pd.DataFrame([{"model": m, **audit_predictions(pathlib.Path(idx[f"{EXPERIMENT}/seed_{SEED}/{m}"]["predictions"]),
                                                     "identity", ctx.manifests.test_grids)}
                    for m in cfg.models if idx.get(f"{EXPERIMENT}/seed_{SEED}/{m}", {}).get("complete")])
display(aud); aud.to_csv(AUD / "prediction_audit.csv", index=False)
'''

ANALYSE = r'''# --- Curves, per-k results, singletons, transformation coverage, H diagnostic, paired transitions ---------------
# (rerun this cell alone after both notebooks finish to get the full joint comparison; no retraining)
from decompositionality.reports import plot_training_curves
from decompositionality.mixed_reports import plot_length_curves
from decompositionality.idext_reports import analyse_all, plot_main, plot_singletons, plot_transform_split_and_h, status_table
paths = (plot_training_curves(cfg.experiment_dir, cfg.experiment_dir / "figures") or []) + \
        (plot_length_curves(cfg.experiment_dir, cfg.experiment_dir / "figures") or [])
res = analyse_all(FAMILY_ROOT, SEED)
FIG = res["out"] / "figures"
paths += plot_main(res, FIG) + plot_singletons(res, FIG) + plot_transform_split_and_h(res, FIG)
for p in paths:
    if str(p).endswith(".png"):
        display(Image(filename=str(p)))
display(res["status"]); display(res["by_k"]); display(res["singletons"]); display(res["consistency"])
display(res["transform_split"]); display(res["h_agreement"]); display(res["paired"])
'''

STATUS = r'''# --- Export + per-run status (complete only if BOTH models are trained and evaluated) ---------------------------
from decompositionality.mixed_reports import make_compact_archive
st = status_table(FAMILY_ROOT, SEED); display(st)
mine = st[st.experiment == EXPERIMENT]
make_compact_archive(cfg.experiment_dir)
print("COMPLETE" if (mine.trained.all() and mine.evaluated.all()) else
      f"INCOMPLETE: {mine[~(mine.trained & mine.evaluated)].model.tolist()} -> rerun Run all to resume")
'''


def build(exp: str) -> Path:
    fname, train, held, alloc = NB[exp]
    p = ROOT / "notebooks" / fname
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{fname} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(INTRO.format(train=train, held=held, alloc=alloc)), code(BOOTSTRAP), code(CONFIG.format(exp=exp)),
                code(ENV), code(AUDIT), code(TRAIN), code(EVAL), code(ANALYSE), code(STATUS),
                md("Outputs: `<RESULTS_ROOT>/identity_ar_count_extrapolation/` -> `manifests/`, "
                   "`train02/seed_42/<model>`, `train023/seed_42/<model>` (runs, eval + ARTIFACT.json, audits), "
                   "`artifact_index.json`, `comparisons/seed_42/` (tables + figures).")]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for e in (sys.argv[1:] or list(NB)):
        print(build(e))
