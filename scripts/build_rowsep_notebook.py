"""Generate notebooks/18_ar_row_separators_all_experiments.ipynb: ONE notebook, 13 conditions x 2 AR models = 26 runs,
all with learned positions and a real <ROW> token in input and output grids. Refuses to overwrite saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "18_ar_row_separators_all_experiments.ipynb"

INTRO = """# Row separators: 13 conditions x {Transformer AR, Looped AR} = 26 fresh runs (one Run all)

Every input AND output grid is serialized as 41 tokens: six digits per row, with a real `<ROW>` token between rows
(5 per grid). Sequence length is 87+m, where m = number of operation tokens. Every condition uses learned absolute
positions plus the existing row/column/segment features.

| group | conditions | operation segment | train k | held-out tests |
|---|---|---|---|---|
| fixed5 | identity_s5_train02 / 023 / 0234 / 02345 | 5 slots, visible `<identity>` fillers | 0,2 / 0,2,3 / 0,2,3,4 / 0,2,3,4,5 | k=1 at all 5 slots + every longer k ≤ 5 (all 3^k words) |
| variable_with_zero | variable_train02 / 023 / 0234 / 02345 | exactly k A/B/C tokens (k=0 = empty segment) | same | k=1 + longer k ≤ 5 |
| fixed6 | identity_s6_train02345 | 6 slots, visible `<identity>` | 0,2,3,4,5 | k=1 at all 6 slots + all 729 k=6 words |
| variable_nozero | variable_nozero_train2 / 23 / 234 / 2345 | exactly k tokens, no k=0 in training or validation | 2 / 2,3 / 2,3,4 / 2,3,4,5 | k=1 (+ seen-support checks) |

**Training and selection.**
* seed 42; 5000 updates × 256 per run; AdamW 3e-4 with the established schedule.
* Loss: mean CE over 42 output targets (36 digits + 5 ROW + EOS).
* Checkpoint selection: answer-DIGIT CE only, macro-averaged over the trained k.

**Data and initialization.**
* For a given support, the S5, variable and S6 conditions see the identical ordered logical stream.
* With-zero and nozero conditions share per-k records by index. Removing k=0 redistributes exposure to positive k; it
  is not a pure removal of identity supervision.
* Each architecture has ONE common fresh initialization for all 13 conditions.

**Evaluation.** Free-running decoding of 41 content tokens + 1 EOS step over the full vocabulary; nothing is repaired.
* Exact-grid accuracy = the 36 expected digit positions only.
* Strict accuracy = digits + 5 ROW + EOS.

**Workflow.**
1. Run all: setup → preflight audits → sequential 26-job queue (train → evaluate → analyze each) → consolidated report.
2. If interrupted, Run all again; it resumes.
3. **Send back `AR_Row_Separators_Review.ipynb`**, printed and downloaded by the last cell.

One seed: descriptive only.
"""

CONFIG = r'''# --- CONFIGURATION (the only cell to edit) --------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")   # "full" | "smoke" (tiny CPU integration run, separate root, NOT results)
SEED = 42
RUN_GROUP = "all"          # "all" | "fixed5" | "variable_with_zero" | "fixed6" | "variable_nozero"
RUN_CONDITIONS = None      # optional explicit list of condition IDs (overrides RUN_GROUP)
RUN_MODELS = ["transformer_ar", "looped_transformer_ar"]
RUN_STAGES = ["train", "evaluate", "analyze"]
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
MICROBATCH = None
EVAL_CHUNK_GRIDS = 100
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import json, pathlib
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import rowsep as RS, rowsep_reports as RR
from decompositionality.config import make_experiment_config
JOBS = RS.select_jobs(RUN_GROUP, RUN_CONDITIONS, RUN_MODELS)
ROOT = RR.family_root(RESULTS_ROOT, MODE)
_c = make_experiment_config(JOBS[0][0], MODE, str(pathlib.Path(RESULTS_ROOT) / RS.ROWSEP_RESULTS_SUBDIR), RUN_MODELS, seed=SEED)
expected = dict(lr=3e-4, weight_decay=0.01, warmup_steps=250, min_lr=3e-5, grad_clip=1.0, val_every=250,
                optimizer_steps=5000, effective_batch_size=256)
print("resolved recipe:", _c.train.semantic_dict())
print("discrepancies from expected:", {k: (getattr(_c.train, k), v) for k, v in expected.items() if getattr(_c.train, k) != v} or "none")
if MODE == "smoke":
    print("*** SMOKE MODE: tiny integration run under", ROOT, "- NOT results ***")
print(f"{len(JOBS)} jobs selected" + ("" if len(JOBS) == 26 else " (PARTIAL selection - never reported as the complete suite)"))
print("results ->", ROOT)
'''

ENV = r'''from decompositionality.utils import environment_info
print(json.dumps(environment_info(REPO_DIR), indent=1))
'''

PREFLIGHT = r'''# --- Preflight audits (stop on failure): registry, layout/positions, initialization, streams, validation ----------
AUD = ROOT / "audits" / f"seed_{SEED}"; AUD.mkdir(parents=True, exist_ok=True)
reg = RS.check_registry()
reg_t = pd.DataFrame([{"condition": c, "group": RS.group_of(c), "layout": RS.layout_of(c), "S": RS.n_slots(c),
                       "train_k": str(list(RS.train_ks(c))), "per_k_training_records": str(reg["per_k_training_counts"][c]),
                       "test_records_per_model": RS.expected_test_records(c, 1000),
                       "required_heldout_k": str(sorted(RS.required_heldout(c)))} for c in RS.CONDITIONS])
print(f"{reg['jobs']} jobs; {reg['records_per_arch']:,} test records per architecture; {reg['total_records']:,} total")
audits = {"registry": reg_t, "layout": RR.layout_audit(), "initialization": RR.init_audit(SEED),
          "streams": RR.stream_audit(SEED), "validation_audit": RR.validation_audit()}
for name, df in audits.items():
    df.to_csv(AUD / f"{name}.csv", index=False); print(f"\n[{name}]"); display(df)
assert audits["initialization"].identical_init_across_13.all()
'''

DRIVER = r'''# --- Sequential driver (one model on the GPU at a time; resumable; verified stages skipped) -------------------------
status = RR.run_queue(RESULTS_ROOT, MODE, SEED, JOBS, RUN_STAGES, repo_dir=str(REPO_DIR), chunk_grids=EVAL_CHUNK_GRIDS,
                      device=DEVICE, dtype=DTYPE, microbatch=MICROBATCH)
display(status)
'''

REPORT = r'''# --- Consolidated report + review export (reads saved artifacts only; rerun alone at any time) ------------------
rp = RR.consolidated(ROOT, RESULTS_ROOT, SEED, JOBS)
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB:
    from google.colab import files
    files.download(str(rp))
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(ENV), nbf.v4.new_code_cell(PREFLIGHT), nbf.v4.new_code_cell(DRIVER),
                nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
