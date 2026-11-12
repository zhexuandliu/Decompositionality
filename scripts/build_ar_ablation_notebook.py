"""Generate notebooks/31_ar_ablation_S7_k02345_seed42.ipynb: ablations of the original Transformer AR on S7_k02345."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "31_ar_ablation_S7_k02345_seed42.ipynb"

INTRO = """# What makes the original Transformer AR worse? Ablations on S7_k02345 (seed 42)

Three variants of the ORIGINAL Transformer AR (same sequence, delimiters, segment embeddings, teacher forcing and
token-by-token decoding). Each changes one thing, or both:

| model | change |
|---|---|
| `ar_slotpos` | **no 1-D position table pos(t)**; op tokens get an op-slot embedding instead; every other token keeps segment + row/col |
| `ar_digithead` | **10-way digit output head**; loss on the 36 answer digits only (no EOS target); 36 decoding steps |
| `ar_slotpos_digithead` | both |

Under the same seed, every other tensor starts from exactly the original AR's initial values. The slot embedding
starts from the original pos(t) rows of the op slots, and the digit head starts from the original head's digit rows.
Same data stream, 20,000 updates, and sampled evaluation of the LAST checkpoint on the same test records as notebooks
23 / 29.

The final cell compares them with the original AR (23b), query AR (29b, if finished) and causal encoder (23b):
* accuracy by test k;
* singletons by operation and slot;
* k = 6 failures that equal "one operation dropped".

The runs are resumable. To go faster, open a second copy of this notebook and give the two copies disjoint
`RUN_MODELS` lists.

**Send back `AR_Ablation_S7_k02345_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_CONDITIONS = ["S7_k02345"]
RUN_MODELS = ["ar_slotpos", "ar_digithead", "ar_slotpos_digithead"]
STEPS = 20000
SUBDIR = "S7_support_sweep_20k_v1"                # same folder as notebooks 23 / 27 / 29 (model subfolders differ)
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
from decompositionality import longk as LK, longk_exp as LX, ar_ablation as AB
print("results ->", LX.family_root(RESULTS_ROOT, MODE, SUBDIR))
'''

DRIVER = r'''# --- Train (20,000 updates) + sampled LAST-checkpoint evaluation (resumable) ---------------------------------------
status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR), device=DEVICE,
                      subdir=SUBDIR, steps=STEPS if MODE == "full" else None, sampled_plan=EVAL_PLAN)
display(status)
'''

REPORT = r'''# --- Comparison (reads saved predictions only; rerun after all runs finish) ---------------------------------------
rp = AB.report(RESULTS_ROOT, MODE, SEED)
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
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
