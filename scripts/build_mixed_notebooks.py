"""Generate notebooks/04_train_mixed_02345_distinct_pairs.ipynb and 05_train_mixed_02345_all_pairs.ipynb
from ONE template (clean, no outputs). Notebooks 01-03 are NOT touched. Run:
    python scripts/build_mixed_notebooks.py"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402  (same private-repo bootstrap; token never printed/stored)

ROOT = Path(__file__).resolve().parents[1]
NB = {
    "mixed_02345_distinct_pairs": ("04", "AB, AC, BA, BC, CA, CB", 358),
    "mixed_02345_all_pairs": ("05", "AA, AB, AC, BA, BB, BC, CA, CB, CC", 361),
}

INTRO = """# Mixed-length training (lengths 0, 2, 3, 4, 5) — `{exp}`

**Question.** Models are trained FROM SCRATCH on instructions with 0, 2, 3, 4 or 5 operations but **never** on a
single operation. Can they execute the individual operations A, B, C at test time?

**Task.** 6x6 grids of digits 0-9; `A` = shift right by one with wrap, `B` = rotate 90° clockwise, `C` = mirror
left-right. Words run **left to right** (`AB` = B(A(X))). The EMPTY word has zero operations (target = input).

**Training support of this notebook ({n_words} words):** EMPTY; length 2: **{pairs}**; all 27 length-3, 81 length-4 and
243 length-5 words (repeats allowed). Length 1 is never trained or used for validation.
The sibling experiment differs **only** in its length-two support.

**Sampling.** Each length (0/2/3/4/5) gets exactly 20% of the 1,280,000 training examples (5000 steps x batch 256,
seed 42); words are balanced within a length; operations occupy a uniformly random order-preserving subset of
**five operation slots** (others PAD). Pairs now receive only 20% of the budget, unlike the original length-two-only
experiments, so old/new differences are not attributable to any single factor.

**Checkpoint selection** uses 300 validation grids x one word at each of the 5 trained lengths (1,500 records), by the
length-macro final-answer cell CE (AR: teacher-forced digit CE). **Test**: the selected checkpoint on all **364 words**
(lengths 0-5, incl. EMPTY and the never-trained singles) x 1000 held-out grids = 364,000 records/model; AR decoding is
free-running. Lengths 3-5 are *seen strings on new grids*, not unseen-combination generalization.

After evaluation this notebook runs the post-hoc diagnostics (288-transformation matching, explicit error rules with
denominators, fitted rules/maps on a FIT/DIAG split, composition, controls). One seed: descriptive only.

**Run:** GPU runtime -> *Run all*. Authentication as in notebooks 01-03 (`GITHUB_TOKEN` Colab secret or hidden prompt).
Results persist on Drive; rerun *Run all* to resume (completed stages are verified and skipped).
"""

CONFIG = r'''# --- CONFIGURATION ------------------------------------------------------------------------------------
EXPERIMENT = "{exp}"   # fixed for this notebook
MODE = os.environ.get("DECOMP_MODE", "full")   # "full" = benchmark | "smoke" = tiny integration run (NOT results)
SEED = 42
MODELS = os.environ.get("DECOMP_MODELS", "transformer_ar,looped_transformer_ar,transformer_encoder,"
                        "looped_transformer_encoder,vit,looped_vit,trm,rescnn_film,unet_film").split(",")
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE
    else "/content/decompositionality_results" if IN_COLAB else str(REPO_DIR / "results"))
MIXED_ROOT = f"{{RESULTS_ROOT}}/mixed_length_02345"   # NEW subtree; legacy results are never touched
RESUME = True
OPTIMIZER_STEPS = None          # None = mode default (full: 5000)
EFFECTIVE_BATCH = None          # None = mode default (full: 256)
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"                  # bf16 if supported, else fp16 + GradScaler; "fp32" disables AMP
MICROBATCH = None               # None = probe with dummy data (effective batch never reduced)
ACTIVATION_CHECKPOINTING = False
N_SLOTS = 5                     # five operation slots (fixed for these experiments; legacy models use 3)
EVAL_CHUNK_GRIDS = 100          # test evaluation is saved in restartable chunks of this many base grids
ANALYSIS_CHUNK_SIZE = 2000      # records per vectorized matching chunk in the diagnostics
USE_ANALYSIS_CACHE = True
INCLUDE_SIBLING_IN_DIAGNOSTICS = True   # add the other mixed experiment (read-only) for paired comparisons if present
RUN_PROFILE = True
DOWNLOAD_ARCHIVE = False
# -----------------------------------------------------------------------------------------------------------

if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

from decompositionality.config import make_experiment_config
_over = dict(seed=SEED, device=DEVICE, dtype=DTYPE, microbatch_size=MICROBATCH,
             activation_checkpointing=ACTIVATION_CHECKPOINTING)
if OPTIMIZER_STEPS: _over["optimizer_steps"] = OPTIMIZER_STEPS
if EFFECTIVE_BATCH: _over["effective_batch_size"] = EFFECTIVE_BATCH
cfg = make_experiment_config(EXPERIMENT, MODE, MIXED_ROOT, MODELS, **_over)
cfg.resume = RESUME
assert cfg.n_slots == N_SLOTS == 5
if MODE == "smoke":
    print("*** SMOKE MODE: full architectures, a few updates, tiny eval sets. NOT research results. ***")
print(f"experiment={{cfg.experiment}} mode={{cfg.mode}} steps={{cfg.train.optimizer_steps}} "
      f"batch={{cfg.train.effective_batch_size}} seed={{cfg.train.seed}} slots={{cfg.n_slots}}")
print("output directory:", cfg.experiment_dir.resolve())
'''

ENV = r'''import json, torch
from decompositionality.utils import environment_info
ENV_INFO = environment_info(REPO_DIR)
print(json.dumps(ENV_INFO, indent=1))
'''

VERIFY = r'''# --- Verify supports, length balance, slot placement and manifests -------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from decompositionality import mixed as M
from decompositionality.mixed_pipeline import prepare_mixed
from decompositionality.diagnosis.group import Catalog, check_catalog
from decompositionality.plotstyle import draw_grid
print("word counts:", M.check_supports())
sup = M.support(EXPERIMENT)
print({L: len(w) for L, w in sup.items()}, "| length-2 words:", sup[2])
ctx = prepare_mixed(cfg, repo_dir=str(REPO_DIR),
                    legacy_manifest=f"{RESULTS_ROOT}/distinct_pairs/manifests/manifest.json", log=print)
_s = M.MixedTrainStream(cfg.train.seed, EXPERIMENT, 256, ctx.manifests.heldout_bytes)
_b = [_s.batch(i) for i in range(20)]          # stateless: inspecting batches does not consume anything
_len = np.concatenate([b["length"] for b in _b]); _ops = np.concatenate([b["ops"] for b in _b])
print("length shares in the first 20 logical batches:", {int(L): round(float((_len == L).mean()), 4) for L in M.LENGTHS})
print("fraction of examples with a real operation in each slot:", ((_ops != 3).mean(0)).round(3).tolist())
assert not (_len == 1).any() and (_ops[_len == 0] == 3).all() and (_ops[_len == 5] != 3).all()
print("validation records per length:", pd.Series(ctx.val_set["length"]).value_counts().sort_index().to_dict())
print("test records:", len(ctx.test_set["x"]), "| words:", len(set(ctx.test_set["string"])))
print("catalog:", check_catalog(Catalog()))
x0 = ctx.test_set["x"][0].reshape(6, 6)
show = ["EMPTY", "A", "AB", "ABCBA"]
fig, axes = plt.subplots(1, len(show) + 1, figsize=(2.1 * (len(show) + 1), 2.6))
draw_grid(axes[0], x0, "input X")
for ax, s in zip(axes[1:], show):
    w = "" if s == "EMPTY" else s
    draw_grid(ax, M.apply_word_batch(x0[None], [w])[0], f"{s}\nslots {M.canonical_slots(w)}")
plt.tight_layout(); plt.show()
'''

COUNTS = r'''# --- Parameter counts with FIVE operation slots, and a time estimate ---------------------------------
from decompositionality.mixed_reports import resource_table
from decompositionality.training import profile_model
rt = resource_table(cfg.experiment_dir, cfg.experiment_dir / "tables")
display(rt[["model", "display_name", "n_slots", "unique_learned_params", "shared_core_params", "untied_equivalent_params",
            "block_applications_per_forward", "input_tokens", "decoded_output_steps"]])
if RUN_PROFILE and torch.cuda.is_available():
    prof = pd.DataFrame([profile_model(m, cfg.train, n_steps=2, n_slots=N_SLOTS) for m in cfg.models])
    display(prof)
    print(f"Estimated TRAINING time for {len(cfg.models)} models: {prof.est_train_minutes.sum():.0f} min, plus evaluation of "
          "364,000 records/model (AR free-running decoding is the slowest). Rerun to resume if the session ends.")
'''

TRAIN = r'''# --- Train all selected models sequentially (fresh initialization; resumable; checkpoints every 250 steps) ---
from decompositionality.training import train_model
from decompositionality.utils import free_memory
failures = {}
for mid in cfg.models:
    print(f"\n=== training {mid} ===")
    try:
        train_model(mid, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, repo_dir=str(REPO_DIR))
    except Exception as exc:
        import traceback; traceback.print_exc(); failures[mid] = repr(exc)
    finally:
        free_memory()
if failures:
    raise RuntimeError(f"training failed for: {failures}")
'''

EVAL = r'''# --- Evaluate each SELECTED checkpoint on all 364 words x test grids (chunked, restartable) ---------------
from decompositionality.mixed_pipeline import evaluate_mixed_model
failures = {}
for mid in cfg.models:
    try:
        evaluate_mixed_model(mid, ctx, chunk_grids=EVAL_CHUNK_GRIDS)
    except Exception as exc:
        import traceback; traceback.print_exc(); failures[mid] = repr(exc)
    finally:
        free_memory()
if failures:
    raise RuntimeError(f"evaluation failed for: {failures}")
'''

REPORTS = r'''# --- Training/evaluation figures and tables (saved before the diagnostics run) ------------------------------
from decompositionality.mixed_reports import build_mixed_reports
rep = build_mixed_reports(cfg.experiment_dir, EXPERIMENT)
FIGS = cfg.experiment_dir / "figures"
for name in ["a_training_curves", "a2_validation_by_length", "b1_singles_primary_and_empty",
             "b2_accuracy_by_length_and_seen_status", "b3_unseen_pairs_AA_BB_CC"]:
    p = FIGS / f"{name}.png"
    if p.exists():
        display(Image(filename=str(p)))
'''

DIAG = r'''# --- Post-hoc diagnostics on the SAVED predictions of this experiment (+ sibling, read-only, if present) ---
from decompositionality.diagnosis.runner import DiagnosticsConfig, run_diagnostics
from decompositionality.mixed import MIXED_EXPERIMENTS
dirs = {EXPERIMENT: str(cfg.experiment_dir)}
sibling = [e for e in MIXED_EXPERIMENTS if e != EXPERIMENT][0]
sib_dir = cfg.experiment_dir.parent / sibling
if INCLUDE_SIBLING_IN_DIAGNOSTICS and (sib_dir / "eval").exists():
    dirs[sibling] = str(sib_dir)
    print("including sibling experiment (read-only) for paired comparisons:", sib_dir)
else:
    print("sibling experiment not available -> paired comparison will be N/A (this notebook does not need it)")
dcfg = DiagnosticsConfig(experiment_dirs=dirs, output_dir=str(cfg.experiment_dir / "diagnostics"), models=cfg.models,
                         chunk_size=ANALYSIS_CHUNK_SIZE, use_cache=USE_ANALYSIS_CACHE, spec="mixed",
                         example_strings=["EMPTY", "A", "B", "C", "AA", "BB", "CC", "ABC"])
R = run_diagnostics(dcfg)
DFIG = pathlib.Path(R["output_dir"]) / "figures"
def show(prefix):
    for p in sorted(DFIG.glob(prefix + "*.png")):
        display(Image(filename=str(p)))
'''

DIAG_SHOW = r'''# Audit, then the out-of-support words (singles and unseen pairs) first
display(R["audit"][["experiment", "model", "n_records", "n_strings", "empty_word_records", "missing_strings",
                    "duplicate_keys", "invalid_records", "target_oracle_mismatches", "input_manifest_mismatches",
                    "decoding", "attribution_blocked"]])
ood = R["per_string"][R["per_string"].group.isin(["singles", "pairs_unseen"])]
display(ood[["experiment", "model", "string", "exact_correct", "n", "exact_acc", "cell_acc", "any_exact_group_match_count",
             "dominant_wrong_unique_word", "dominant_wrong_unique_share", "dominant_is_original_pair",
             "stable_rule_selected_word", "stable_rule_diag_exact_fidelity", "argmax_map_diag_cell_fidelity"]])
hs = R["hypothesis_summary"]
display(hs[hs.population == "OOD: singles + unseen pairs"])
show("01_true"); show("03_explanation"); show("04_dominant"); show("05_hypothesis")
'''

DIAG_SHOW2 = r'''# Stable rules / source maps / composition / controls / paired comparison / examples
show("08_stable"); show("13_exact"); show("10_source"); show("09_composition"); show("06_nearest"); show("07_alternate")
display(R["controls"]["summary"])
show("11_paired"); show("12_examples")
'''

EXPORT = r'''# --- Export and concise summary of ACTUAL results --------------------------------------------------------
from decompositionality.mixed_reports import make_compact_archive
arch = make_compact_archive(cfg.experiment_dir)
display(Markdown((cfg.experiment_dir / "tables" / "f_summary.md").read_text()))
display(Markdown(pathlib.Path(R["report_path"]).read_text()))
if DOWNLOAD_ARCHIVE and IN_COLAB:
    from google.colab import files
    files.download(str(arch))
'''

FOOTER = """## Outputs
`<RESULTS_ROOT>/mixed_length_02345/{exp}/`: `manifests/`, `runs/<model>/` (`state.pt` last, `best.pt` selected,
logs), `eval/<model>/` (`test_predictions.npz` with inputs/targets/predictions/slots/lengths for 364,000 records,
`test_per_word.csv`, `test_groups.csv`, `eval_summary.json`, `chunks/`), `figures/`, `tables/`, `diagnostics/`
(tables, figures, per-example diagnostics, fitted maps, `REPORT.md`, `provenance.json`) and
`results_{exp}.zip` (no checkpoints, chunks or caches). Smoke runs go to `.../mixed_length_02345/smoke/`.
"""


def build(exp: str) -> Path:
    num, pairs, n_words = NB[exp]
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(INTRO.format(exp=exp, pairs=pairs, n_words=n_words)), code(BOOTSTRAP), code(CONFIG.format(exp=exp)),
                code(ENV), code(VERIFY), code(COUNTS), code(TRAIN), code(EVAL), code(REPORTS),
                md("## Diagnostics\nBehavioural, post-hoc and non-causal. Hypothesis tables report eligible records, unique "
                   "matched records, conditional and unconditional rates. DIAG measures use a disjoint half of the base grids."),
                code(DIAG), code(DIAG_SHOW), code(DIAG_SHOW2), code(EXPORT), md(FOOTER.format(exp=exp))]
    p = ROOT / "notebooks" / f"{num}_train_{exp}.ipynb"
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for e in NB:
        print(build(e))
