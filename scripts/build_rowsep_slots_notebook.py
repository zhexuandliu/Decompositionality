"""Generate notebooks/19_ar_row_identity_slots234_supplement.ipynb (6 runs: S=2/3/4 with ROW x 2 AR models).
Refuses to overwrite a notebook with saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "19_ar_row_identity_slots234_supplement.ipynb"

INTRO = """# ROW supplement: fixed S = 2 / 3 / 4 identity slots x {Transformer AR, Looped AR} = 6 runs

Standalone supplement to notebook 18. It does NOT touch notebook 18's 26-job queue or its results.

| condition | S | train k | held-out test | test records / model |
|---|---|---|---|---|
| identity_s2_train02_row | 2 | 0,2 | A/B/C at both slots | 16,000 |
| identity_s3_train023_row | 3 | 0,2,3 | A/B/C at all 3 slots | 46,000 |
| identity_s4_train0234_row | 4 | 0,2,3,4 | A/B/C at all 4 slots | 130,000 |

**Format.** Real `<ROW>` tokens between the rows of the input AND output grids, visible `<identity>` fillers, and
learned positions. Sequences are 89 / 90 / 91 tokens.

**Same machinery as notebook 18.** Serializer, vocabulary, model config, the common fresh initialization, the
42-target loss, digit-CE selection, and free-running unrepaired scoring.

**Pairing.**
* The logical stream equals the main S5 / variable stream at the same support.
* Placements and test manifests equal notebook 16's S=2/3/4 no-ROW runs.

**Comparisons.** Comparisons with notebook 18 and notebook 16 are read-only, and only use completed artifacts. Missing
ones show as pending.

**Send back `AR_Row_Slots234_Review.ipynb`** (printed and downloaded by the last cell). One seed: descriptive only.
"""

CONFIG = r'''# --- CONFIGURATION ------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")      # "full" | "smoke" (tiny CPU check, separate root, NOT results)
SEED = 42
SUPPLEMENT_CONDITIONS = ["identity_s2_train02_row", "identity_s3_train023_row", "identity_s4_train0234_row"]
MODEL_IDS = ["transformer_ar", "looped_transformer_ar"]
RUN_STAGES = ["train", "evaluate", "analyze"]
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
EVAL_CHUNK_GRIDS = 100
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import json, pathlib
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import rowsep as RS, rowsep_reports as RR, rowsep_slots as SP
from decompositionality.config import make_experiment_config
JOBS = SP.jobs(SUPPLEMENT_CONDITIONS, MODEL_IDS)
OUTPUT_ROOT = RR.family_root(RESULTS_ROOT, MODE, SP.SUBDIR)
MAIN_ROOT = RR.family_root(RESULTS_ROOT, MODE)          # notebook-18 root: READ-ONLY
_c = make_experiment_config(JOBS[0][0], MODE, str(pathlib.Path(RESULTS_ROOT) / SP.SUBDIR), MODEL_IDS, seed=SEED)
print("resolved recipe:", _c.train.semantic_dict())
print(f"{len(JOBS)} supplement jobs:", JOBS)
print("output ->", OUTPUT_ROOT, "| main suite (read-only) ->", MAIN_ROOT)
if MODE == "smoke":
    print("*** SMOKE MODE - NOT results ***")
'''

PREFLIGHT = r'''# --- Preflight (stop on failure): layouts, counts, stream / placement / init matching -------------------------------
AUD = OUTPUT_ROOT / "audits" / f"seed_{SEED}"; AUD.mkdir(parents=True, exist_ok=True)
pf, ini = SP.preflight(SEED, main_root=MAIN_ROOT)
lay = RR.layout_audit()
for name, df in (("preflight", pf), ("initialization", ini), ("layout", lay)):
    df.to_csv(AUD / f"{name}.csv", index=False); print(f"\n[{name}]"); display(df)
'''

DRIVER = r'''# --- Sequential six-job driver (resumable; one model on the GPU at a time) ------------------------------------------
status = RR.run_queue(RESULTS_ROOT, MODE, SEED, JOBS, RUN_STAGES, repo_dir=str(REPO_DIR), chunk_grids=EVAL_CHUNK_GRIDS,
                      device=DEVICE, dtype=DTYPE, subdir=SP.SUBDIR, universe=SP.jobs())
display(status)
'''

REPORT = r'''# --- Report + read-only comparisons + review export (rerunnable alone; never trains) ---------------------------
rp = SP.report(RESULTS_ROOT, MODE, SEED, JOBS)
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
                nbf.v4.new_code_cell(PREFLIGHT), nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
