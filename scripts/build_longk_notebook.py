"""Generate notebooks/21_ar_vs_causal_encoder_last_ckpt_longk.ipynb (8 runs). Refuses to overwrite saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "21_ar_vs_causal_encoder_last_ckpt_longk.ipynb"

INTRO = """# Transformer AR vs causal encoder — last checkpoint, singletons and longer instructions (8 runs)

| condition | slots S | train k | held-out tests |
|---|---|---|---|
| S5_k0234 | 5 | 0,2,3,4 | k=1 at every slot; k=5 (243 words) |
| S6_k02345 | 6 | 0,2,3,4,5 | k=1; k=6 (729 words) |
| S7_k02345 | 7 | 0,2,3,4,5 | k=1; k=6 (729); k=7 (2187 words) |
| S7_k023456 | 7 | 0,2,3,4,5,6 | k=1; k=7 (2187 words) |

**Format.** Visible `<identity>` fillers, learned positions. Every longer test stays within S, so it adds no new
physical positions.

**Models.** `transformer_ar` (autoregressive, free-running) and `causal_encoder` (causal attention with 36 parallel
output queries, no feedback).

**Training.** seed 42, 5000 × 256.
* One common initialization per architecture: all conditions share positional capacity 7.
* S6_k02345 and S7_k02345 see the identical logical training stream.

**Evaluation: the LAST checkpoint, i.e. the model after update 5000.** The best checkpoint is still saved, and its
step is reported for reference.

**Test size.** 1000 test grids, except k=7, which uses `K7_GRIDS` grids (default 300, i.e. 656,100 records per run).

**Send back `LongK_AR_vs_CausalEnc_Review.ipynb`** (printed and downloaded by the last cell).
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_CONDITIONS = ["S5_k0234", "S6_k02345", "S7_k02345", "S7_k023456"]
RUN_MODELS = ["transformer_ar", "causal_encoder"]
N_TEST_GRIDS = 1000
K7_GRIDS = 300                 # grids used for the 2187 k=7 words (raise to 1000 for the full set; ~3x the time)
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
for e in RUN_CONDITIONS:
    print(e, "expected test records per model:", f"{LK.expected_records(e, N_TEST_GRIDS, K7_GRIDS):,}")
print("results ->", LX.family_root(RESULTS_ROOT, MODE))
'''

DRIVER = r'''# --- Train + evaluate the LAST checkpoint, sequentially (resumable) --------------------------------------------------
status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR),
                      n_grids=N_TEST_GRIDS, k7_grids=K7_GRIDS, device=DEVICE)
display(status)
'''

REPORT = r'''# --- Report + review export (reads saved artifacts only) -----------------------------------------------------------
rp = LX.report(RESULTS_ROOT, MODE, SEED)
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
