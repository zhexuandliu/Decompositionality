"""Generate notebooks/23_S7_support_sweep_20k.ipynb (4 runs, 10000 updates). Refuses to overwrite outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "23_S7_support_sweep_20k.ipynb"

INTRO = """# S=7 support sweep: Transformer AR vs causal encoder, 20,000 updates, sampled evaluation (12 runs)

Seven visible operation slots with `<identity>` fillers. The training support varies; the format, models and recipe
do not.

| condition | train k | held-out tests |
|---|---|---|
| S7_k02 | 0,2 | k=1 at every slot; k=3–7 |
| S7_k023 | 0,2,3 | k=1; k=4–7 |
| S7_k0234 | 0,2,3,4 | k=1; k=5–7 |
| S7_k02345 | 0,2,3,4,5 | k=1; k=6, 7 |
| S7_k023456 | 0,2,3,4,5,6 | k=1; k=7 |
| S7_k0234567 | 0,2,3,4,5,6,7 | k=1 only (k=7 fills every slot: trained zero-identity instructions) |

**Training.** 20,000 updates × 256 (5.12M examples).
* One common initialization per architecture: all conditions share capacity 7.
* Records at a common k are shared across conditions by index.

**Evaluation: the LAST checkpoint, on a SAMPLED test set** (about 50–100k records per run, instead of millions):
* Singletons: A/B/C × all 7 slots × 1000 grids = 21,000. Exhaustive.
* Each trained k: 5000 (word, grid) records.
* Each held-out longer k: 20,000 records. Words are cycled so each appears almost equally often.

The sizes are set in the config cell.

**Results.** A separate folder. **Send back `S7_Sweep_20k_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_CONDITIONS = ["S7_k02", "S7_k023", "S7_k0234", "S7_k02345", "S7_k023456", "S7_k0234567"]
RUN_MODELS = ["transformer_ar", "causal_encoder"]
STEPS = 20000
SUBDIR = "S7_support_sweep_20k_v1"
EVAL_PLAN = {"n_single_grids": 1000, "n_seen_per_k": 5000, "n_long_per_k": 20000}
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import longk as LK, longk_exp as LX
_c = LX.make_cfg(RUN_CONDITIONS[0], RESULTS_ROOT, MODE, RUN_MODELS, SEED, SUBDIR,
                 **({"optimizer_steps": STEPS} if MODE == "full" else {}))
print("resolved recipe:", _c.train.semantic_dict())
print("results ->", LX.family_root(RESULTS_ROOT, MODE, SUBDIR))
'''

DRIVER = r'''# --- Train (20,000 updates) + sampled LAST-checkpoint evaluation, sequentially (resumable) ---------------------------
status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR), device=DEVICE,
                      subdir=SUBDIR, steps=STEPS if MODE == "full" else None, sampled_plan=EVAL_PLAN)
display(status)
'''

REPORT = r'''# --- Report + review export (reads saved artifacts only) -----------------------------------------------------------
rp = LX.report(RESULTS_ROOT, MODE, SEED, conditions=RUN_CONDITIONS, subdir=SUBDIR,
               title_note=f"Budget: {STEPS} updates; sampled evaluation {EVAL_PLAN}.", review_name="S7_Sweep_20k_Review.ipynb")
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
                nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
