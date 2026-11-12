"""Generate notebooks/29a_/29b_query_ar_S7_sweep_seed42_*.ipynb: the encoder-aligned AR ("query_ar") on the S=7 sweep,
seed 42, split into two halves that run in parallel (3 runs each)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"a": ["S7_k02", "S7_k023", "S7_k0234"], "b": ["S7_k02345", "S7_k023456", "S7_k0234567"]}

INTRO = """# Query AR on the S=7 sweep — seed 42, part __PART__ of 2 (__CONDS_TXT__)

**Model `query_ar`.** It is identical to the causal encoder in every respect except autoregressive prediction:
* sequence `[ops][36 input cells][36 output positions]`, no delimiters, no 1-D position encoding;
* slot embeddings for the ops; (row, col) embeddings for the cells and output positions; type embeddings;
* causal attention over the whole sequence; 6 untied blocks; a 10-way digit head; 36-cell cross-entropy.

**The one difference.** Output position j also receives the digit of cell j−1: the true digit in training, and its
OWN prediction at test time, decoded cell by cell. Position 0 gets a learned start vector.

**Initialization.** Under the same seed, every shared tensor has exactly the causal encoder's initial value.

**Setup.** Same as notebooks 23 / 27: S=7 with identity fillers, 20,000 updates, and sampled evaluation of the LAST
checkpoint on the same test records. Seed 42.

**This notebook = part __PART__** (__CONDS_TXT__). The other half is notebook __OTHER__; run both at the same
time. The last cell (rerun after both finish) compares query_ar with the existing Transformer AR and causal
encoder (seed 42). **Send back `QueryAR_vs_AR_vs_Encoder_seed42_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_CONDITIONS = __CONDS__
RUN_MODELS = ["query_ar"]
STEPS = 20000
SUBDIR = "S7_support_sweep_20k_v1"                # same folder as notebooks 23 / 27 (model subfolders differ)
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
print("results ->", LX.family_root(RESULTS_ROOT, MODE, SUBDIR))
'''

DRIVER = r'''# --- Train (20,000 updates) + sampled LAST-checkpoint evaluation (resumable) ---------------------------------------
status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR), device=DEVICE,
                      subdir=SUBDIR, steps=STEPS if MODE == "full" else None, sampled_plan=EVAL_PLAN)
display(status)
'''

REPORT = r'''# --- Comparison with the existing Transformer AR and causal encoder (seed 42; rerun after both halves finish) -----
rp = LX.multiseed_report(RESULTS_ROOT, MODE, [42], LK.S7_SWEEP, SUBDIR, models=["transformer_ar", "causal_encoder", "query_ar"],
                         out_name="query_ar_seed42", review_name="QueryAR_vs_AR_vs_Encoder_seed42_Review.ipynb")
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build(part):
    conds = PARTS[part]
    other = "29b" if part == "a" else "29a"
    name = f"29{part}_query_ar_S7_sweep_seed42_{'_'.join(c.replace('S7_', '') for c in conds)}.ipynb"
    sub = lambda t: (t.replace("__CONDS__", repr(conds)).replace("__PART__", part).replace("__OTHER__", other)  # noqa: E731
                     .replace("__CONDS_TXT__", ", ".join(conds)))
    p = ROOT / "notebooks" / name
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(sub(INTRO)), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(sub(CONFIG)),
                nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for part in PARTS:
        print(build(part))
