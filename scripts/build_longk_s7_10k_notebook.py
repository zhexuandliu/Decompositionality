"""Generate notebooks/22_ar_vs_causal_encoder_S7_10k.ipynb (4 runs, 10000 updates). Refuses to overwrite outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "22_ar_vs_causal_encoder_S7_10k.ipynb"

INTRO = """# S=7 rerun with a longer budget: Transformer AR vs causal encoder, 10,000 updates (4 runs)

The same two S=7 conditions and models as notebook 21. Only the training budget changes: **10,000 updates × 256**
(2.56M examples) instead of 5000. Warmup is still 250 updates, and the cosine schedule now spans 10,000. In notebook
21 the models did not fit the trained k perfectly (AR 98.3–98.7%, causal encoder 99.6–99.8%).

| condition | train k | held-out tests |
|---|---|---|
| S7_k02345 | 0,2,3,4,5 | k=1 at every slot; k=6 (729 words); k=7 (2187 words) |
| S7_k023456 | 0,2,3,4,5,6 | k=1; k=7 |

**Unchanged from notebook 21.** Initialization, logical training stream (the first 5000 batches are identical) and
test sets.

**Evaluation.** The LAST checkpoint, i.e. the model after update 10,000.

**Results.** A new folder, so the 5000-update results are kept. The report includes a 5k vs 10k comparison table.

**Send back `LongK_S7_10k_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_CONDITIONS = ["S7_k02345", "S7_k023456"]
RUN_MODELS = ["transformer_ar", "causal_encoder"]
STEPS = 10000
SUBDIR = "longk_ar_vs_causal_encoder_S7_10k_v1"     # separate from notebook 21's results
N_TEST_GRIDS = 1000
K7_GRIDS = 300
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

DRIVER = r'''# --- Train (10,000 updates) + evaluate the LAST checkpoint (resumable) ----------------------------------------------
status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR),
                      n_grids=N_TEST_GRIDS, k7_grids=K7_GRIDS, device=DEVICE, subdir=SUBDIR,
                      steps=STEPS if MODE == "full" else None)
display(status)
'''

REPORT = r'''# --- Report, incl. 5k (notebook 21) vs 10k comparison; reads saved artifacts only ---------------------------------
rp = LX.report(RESULTS_ROOT, MODE, SEED, conditions=RUN_CONDITIONS, subdir=SUBDIR, compare_subdir=LK.SUBDIR,
               title_note=f"Budget: {STEPS} updates.", review_name="LongK_S7_10k_Review.ipynb")
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
