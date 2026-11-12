"""Generate notebooks/27{a,b,c,d}_S7_sweep_seed{43,44}_*.ipynb: extra seeds for the notebook-23 S=7 sweep.
Each notebook = one seed x 3 conditions x 2 models (6 runs); all four can run at the same time."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HALF = {"first": ["S7_k02", "S7_k023", "S7_k0234"], "second": ["S7_k02345", "S7_k023456", "S7_k0234567"]}
PARTS = {"a": (43, "first"), "b": (43, "second"), "c": (44, "first"), "d": (44, "second")}

INTRO = """# S=7 support sweep — extra seed __SEED__, part __PART__ (__CONDS_TXT__)

The notebook-23 setup again, with training seed __SEED__.
* Identity fillers, Transformer AR and causal encoder.
* 20,000 updates.
* Sampled evaluation of the LAST checkpoint.

**What the seed changes.** The training seed changes the initialization and the training stream. The held-out
grids, the validation set and the sampled test records are the same as for seed 42 (fixed evaluation seed), so the
three seeds are evaluated on identical records.

**Parallel notebooks.**

| notebook | seed | conditions |
|---|---|---|
| 27a | 43 | k02, k023, k0234 |
| 27b | 43 | k02345, k023456, k0234567 |
| 27c | 44 | k02, k023, k0234 |
| 27d | 44 | k02345, k023456, k0234567 |

All four can run at the same time. Seed 42 is the existing notebook-23 runs, which are reused and not retrained.

**Report.** The last cell, rerun in any notebook once everything is finished, makes the multi-seed report
`S7_Sweep_MultiSeed_Review.ipynb`:
* accuracy vs test k per training support (mean over seeds, min–max band, individual seeds);
* singleton / +1 / +2 step accuracy vs longest trained k;
* mean ± std tables.
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = __SEED__                                   # TRAINING seed (initialization + training stream)
RUN_CONDITIONS = __CONDS__
RUN_MODELS = ["transformer_ar", "causal_encoder"]
STEPS = 20000
SUBDIR = "S7_support_sweep_20k_v1"                # same folder as notebook 23 (runs live in seed_<SEED> subfolders)
EVAL_PLAN = {"n_single_grids": 1000, "n_seen_per_k": 5000, "n_long_per_k": 20000}
ALL_SEEDS = [42, 43, 44]
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
print("training seed", SEED, "| evaluation seed (fixed)", LX.EVAL_SEED, "| results ->", LX.family_root(RESULTS_ROOT, MODE, SUBDIR))
'''

DRIVER = r'''# --- Train (20,000 updates) + sampled LAST-checkpoint evaluation (resumable) ---------------------------------------
status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR), device=DEVICE,
                      subdir=SUBDIR, steps=STEPS if MODE == "full" else None, sampled_plan=EVAL_PLAN)
display(status)
'''

REPORT = r'''# --- Multi-seed report over everything finished so far (rerun once all four notebooks are done) --------------------
rp = LX.multiseed_report(RESULTS_ROOT, MODE, ALL_SEEDS, LK.S7_SWEEP, SUBDIR)
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build(part):
    seed, half = PARTS[part]
    conds = HALF[half]
    name = f"27{part}_S7_sweep_seed{seed}_{'_'.join(c.replace('S7_', '') for c in conds)}.ipynb"
    sub = lambda t: (t.replace("__SEED__", str(seed)).replace("__CONDS__", repr(conds)).replace("__PART__", part)  # noqa: E731
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
