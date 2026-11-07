"""Generate notebooks/10_identity_02345_all_models.ipynb and 11_variable_length_02345_all_models.ipynb from ONE template.
They differ ONLY in CONDITION. Refuses to overwrite notebooks that contain saved outputs.
Run:  python scripts/build_idvar_notebooks.py"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = {"identity": ("10", "10_identity_02345_all_models.ipynb",
                   "five physical slots; A/B/C randomly placed in order, every other slot filled with a VISIBLE, "
                   "learnable `<identity>` operation (no PAD anywhere)"),
      "variable_length": ("11", "11_variable_length_02345_all_models.ipynb",
                          "genuinely variable-length instructions: only the k A/B/C tokens, no PAD, no `<identity>`, "
                          "no reserved slots")}

INTRO = """# All nine models — input format: **{cond}**

**Question.** Can a model learn individually usable A/B/C operations from FINAL outputs of composite instructions,
without any single-operation instruction in training or checkpoint-selection validation?

**This notebook's input format:** {desc}.

**Shared data (identical in notebooks 10 and 11).**
* Words have k A/B/C tokens with k in {{0, 2, 3, 4, 5}}: 361 logical words, all repetitions allowed; k = 1 is never
  trained or validated.
* Each update of 256 records holds 51 records per k plus one rotating extra, so each k gets exactly 256,000 of the
  1,280,000 records (seed 42, 5000 updates).
* Within k the word is uniform and the ordered slot subset is uniform.
* Both formats serialize the SAME indexed (word, slots, grid) records: identity keeps the 5-slot layout with
  `<identity>` fillers; variable_length drops the fillers.
* Batches are bucketed by k, so no padding is ever needed.

`A` = shift right one cell, `B` = rotate 90° clockwise, `C` = mirror left-right; words run left to right.
`<identity>` is a learnable input token; the predictor has no identity-return branch.

**Models.** All nine families are trained from scratch, separately in each format (18 runs in total).
* Attention models use learned absolute 1D positions: identity uses physical slots 1-5; variable_length uses
  contiguous positions, so downstream offsets move with k.
* The CNNs use a new order-sensitive conditioner shared by both formats: a phi-MLP over (token, position), a sum
  over the actual tokens, and a rho-MLP feeding FiLM.
* For each architecture, both formats start from identical initial weights.

**Test** uses the selected checkpoint (lowest k-macro validation CE):
* all 361 seen words x 1000 grids (identity: one sampled placement per word/grid);
* singletons: identity A/B/C x 5 placements x 1000; variable_length A/B/C x 1000.

**Limits.** One seed. The two formats differ in filler semantics, positions, visible length and downstream offsets;
this is not a pure mask or length ablation. Success with one op among four `<identity>` tokens is not success on a
bare one-token instruction. Historical PAD runs (04-09) are NOT matched (different serialization and CNN adapter).
"""

CONFIG = r'''# --- CONFIGURATION (the only cell to edit) --------------------------------------------------------------------
CONDITION = "{cond}"           # fixed for this notebook
MODE = os.environ.get("DECOMP_MODE", "full")   # "full" = all nine models, full budget | "smoke" = tiny CPU check (NOT results)
SEED = 42
MODELS = os.environ.get("DECOMP_MODELS", "transformer_ar,looped_transformer_ar,transformer_encoder,"
                        "looped_transformer_encoder,vit,looped_vit,trm,rescnn_film,unet_film").split(",")
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE
    else "/content/decompositionality_results" if IN_COLAB else str(REPO_DIR / "results"))
IDVAR_ROOT = f"{{RESULTS_ROOT}}/identity_varlen_02345_all_models"   # NEW subtree
RESUME = True
OPTIMIZER_STEPS = None          # None = mode default (full: 5000)
EFFECTIVE_BATCH = None          # None = mode default (full: 256)
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
MICROBATCH = None
EVAL_CHUNK_GRIDS = 100
RUN_AUDITS = True
RUN_PROFILE = True
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
from decompositionality.config import make_experiment_config
from decompositionality.models.registry import MODEL_IDS, DISPLAY_NAMES
from decompositionality.idvar import IDVAR_EXPERIMENTS
EXPERIMENT = {{v: k for k, v in IDVAR_EXPERIMENTS.items()}}[CONDITION]
_over = dict(seed=SEED, device=DEVICE, dtype=DTYPE, microbatch_size=MICROBATCH)
if OPTIMIZER_STEPS: _over["optimizer_steps"] = OPTIMIZER_STEPS
if EFFECTIVE_BATCH: _over["effective_batch_size"] = EFFECTIVE_BATCH
cfg = make_experiment_config(EXPERIMENT, MODE, IDVAR_ROOT, MODELS, **_over)
cfg.resume = RESUME
FAMILY_ROOT = cfg.experiment_dir.parent
print("model name -> registry entry:", {{m: DISPLAY_NAMES[m] for m in MODELS}})
assert all(m in MODEL_IDS for m in MODELS)
if MODE == "smoke":
    print("*** SMOKE MODE: tiny integration run; NOT scientific results ***")
print(f"condition={{CONDITION}} | experiment={{EXPERIMENT}} | mode={{MODE}} | steps={{cfg.train.optimizer_steps}} | "
      f"batch={{cfg.train.effective_batch_size}} | seed={{SEED}} | models={{len(MODELS)}}")
print("runs ->", cfg.experiment_dir, "| shared manifests ->", cfg.manifest_dir)
'''

ENV = r'''import json, torch
from decompositionality.utils import environment_info
ENV_INFO = environment_info(REPO_DIR)
print(json.dumps(ENV_INFO, indent=1))
'''

AUDIT = r'''# --- Audits before training: supports, allocation, serialization/shapes, initialization, manifests -------------
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
from decompositionality import idvar as V
from decompositionality.idvar_pipeline import prepare_idvar
from decompositionality.idvar_reports import allocation_audit, init_audit, shape_audit
print(V.check_supports())
s_id, s_var = V.IdVarStream(SEED, 256, "identity"), V.IdVarStream(SEED, 256, "variable_length")
for step in (0, 1, 7):
    a, b = s_id.batch(step), s_var.batch(step)
    assert all(np.array_equal(a[k], b[k]) for k in ("x", "y", "slots5", "k", "string", "sample_index")), "formats differ in data"
    assert not (a["k"] == 1).any()
print("identical logical records in both formats; no k=1 training records")
AUD = cfg.experiment_dir / "audits"; AUD.mkdir(parents=True, exist_ok=True)
alloc = allocation_audit(SEED, cfg.train.effective_batch_size, cfg.train.optimizer_steps); display(alloc)
alloc.to_csv(AUD / "sample_allocation.csv", index=False)
if RUN_AUDITS:
    shp = shape_audit(MODELS); display(shp); shp.to_csv(AUD / "serialization_and_shapes.csv", index=False)
    ini = init_audit(MODELS, SEED); display(ini); ini.to_csv(AUD / "initialization_and_parameters.csv", index=False)
    assert ini.identical_initial_tensors_across_formats.all()
ctx = prepare_idvar(cfg, repo_dir=str(REPO_DIR), legacy_manifest=f"{RESULTS_ROOT}/distinct_pairs/manifests/manifest.json")
print("validation records per k:", pd.Series(ctx.val_set["k"]).value_counts().sort_index().to_dict(),
      "| test records:", len(ctx.test_set["x"]))
'''

PROFILE = r'''# --- Memory/throughput profile on DUMMY data (does not touch the training stream) --------------------------------
from decompositionality.training import profile_model
if RUN_PROFILE and torch.cuda.is_available():
    prof = pd.DataFrame([profile_model(m, cfg.train, n_steps=2, n_slots=5) for m in MODELS])
    display(prof); prof.to_csv(AUD / "profile_dummy.csv", index=False)
'''

TRAIN = r'''# --- Train all models sequentially (fresh init; resumable; one model on the GPU at a time) ---------------------
from decompositionality.training import train_model
from decompositionality.utils import free_memory
failures = {}
for mid in cfg.models:
    print(f"\n=== training {CONDITION}/{mid} ===")
    try:
        train_model(mid, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, repo_dir=str(REPO_DIR))
    except Exception as exc:
        import traceback; traceback.print_exc(); failures[mid] = repr(exc)
    finally:
        free_memory()
if failures:
    print("TRAINING FAILED for:", failures, "-> rerun Run all to resume; later cells report per-model status")
'''

EVAL = r'''# --- Evaluate the SELECTED checkpoints (chunked, restartable; AR free-running) -----------------------------------
from decompositionality.idvar_pipeline import evaluate_idvar_model, audit_predictions, build_artifact_index
eval_fail = {}
for mid in cfg.models:
    try:
        evaluate_idvar_model(mid, ctx, chunk_grids=EVAL_CHUNK_GRIDS)
    except Exception as exc:
        import traceback; traceback.print_exc(); eval_fail[mid] = repr(exc)
    finally:
        free_memory()
idx = build_artifact_index(FAMILY_ROOT)
audits = []
for mid in cfg.models:
    a = idx.get(f"{CONDITION}/{mid}")
    if a and a.get("complete"):
        audits.append({"model": mid, **audit_predictions(pathlib.Path(a["predictions"]), CONDITION, ctx.manifests.test_grids)})
aud = pd.DataFrame(audits); display(aud); aud.to_csv(AUD / "prediction_audit.csv", index=False)
'''

REPORTS = r'''# --- Training curves, accuracy by k, singleton figures ------------------------------------------------------------
from decompositionality.reports import plot_training_curves
from decompositionality.mixed_reports import plot_length_curves
from decompositionality.idvar_reports import plot_by_k, plot_singletons
FIG = FAMILY_ROOT / "comparisons" / "figures"
paths = (plot_training_curves(cfg.experiment_dir, cfg.experiment_dir / "figures") or []) + \
        (plot_length_curves(cfg.experiment_dir, cfg.experiment_dir / "figures") or []) + plot_by_k(FAMILY_ROOT, FIG) + \
        plot_singletons(FAMILY_ROOT, FIG)
for p in paths:
    if str(p).endswith(".png"):
        display(Image(filename=str(p)))
'''

DIAG = r'''# --- Singleton diagnostics (288-transform matching, hypotheses with denominators, FIT/DIAG maps, placement
#     consistency) for every model available in EITHER format (missing runs are listed, never zero-filled) ------------
from decompositionality.idvar_reports import singleton_diagnostics, plot_dominant_and_coverage
diag = singleton_diagnostics(FAMILY_ROOT, MODELS)
display(diag["status"])
cols = ["condition", "position", "model", "string", "exact_correct", "n", "cell_acc", "any_exact_group_match_count",
        "dominant_wrong_unique_word", "dominant_wrong_unique_share", "stable_rule_selected_word",
        "stable_rule_diag_exact_fidelity", "argmax_map_diag_cell_fidelity"]
display(diag["per_string"][[c for c in cols if c in diag["per_string"]]])
display(diag["consistency"])
for p in plot_dominant_and_coverage(diag, FIG):
    if str(p).endswith(".png"):
        display(Image(filename=str(p)))
'''

COMPARE = r'''# --- Paired format comparison (rerun this cell alone once both notebooks are done; no retraining) -----------------
from decompositionality.idvar_reports import compare_formats, status_table
cmp = compare_formats(FAMILY_ROOT, MODELS)
display(cmp["status"])
if cmp["summary"] is not None:
    display(cmp["summary"]); display(cmp["singletons"]); display(cmp["seen"])
'''

STATUS = r'''# --- Export + per-model status (this notebook is complete only if EVERY model is trained and evaluated) -----------
from decompositionality.mixed_reports import make_compact_archive
st = status_table(FAMILY_ROOT, MODELS); display(st)
mine = st[st.condition == CONDITION]
make_compact_archive(cfg.experiment_dir)
if not (mine.trained.all() and mine.evaluated.all()):
    print("INCOMPLETE:", mine[~(mine.trained & mine.evaluated)].model.tolist(), "-> rerun Run all to resume")
else:
    print(f"{CONDITION}: all {len(mine)} models trained and evaluated.")
'''


def build(cond: str) -> Path:
    num, fname, desc = NB[cond]
    p = ROOT / "notebooks" / fname
    if p.exists():
        old = nbf.read(p, as_version=4)
        if any(c.get("outputs") for c in old.cells if c.cell_type == "code"):
            raise SystemExit(f"{fname} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(INTRO.format(cond=cond, desc=desc)), code(BOOTSTRAP), code(CONFIG.format(cond=cond)), code(ENV),
                code(AUDIT), code(PROFILE), code(TRAIN), code(EVAL), code(REPORTS), code(DIAG), code(COMPARE), code(STATUS),
                md("Outputs: `<RESULTS_ROOT>/identity_varlen_02345_all_models/` -> `manifests/`, `identity/<model>`, "
                   "`variable_length/<model>` (runs, eval with `ARTIFACT.json`, audits, figures), `artifact_index.json`, "
                   "`comparisons/` (figures, singleton diagnostics, format comparison).")]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for c in (sys.argv[1:] or list(NB)):
        print(build(c))
