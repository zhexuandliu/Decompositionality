"""Generate notebooks/20_causal_encoder_identity_settings.ipynb (8 runs). Refuses to overwrite saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "20_causal_encoder_identity_settings.ipynb"

INTRO = """# Causal encoder in four identity-filler settings (8 runs)

**Model.** `[operation slots] + [36 input cells] + [36 output queries]`.
* Causal attention over the WHOLE sequence: every position sees only earlier positions.
* All 36 answers are read at the output queries in ONE parallel pass. Nothing is generated and fed back, unlike
  Transformer AR.
* The queries carry only their output cell's (row, col) embeddings plus a learned query vector.
* Two variants: `causal_encoder` (6 untied blocks, 10.7M) and `looped_causal_encoder` (one shared block × 6, 1.8M).

| setting | slots | train k | paired with (same stream, validation and test set) |
|---|---|---|---|
| S5_k02345 | 5 | 0,2,3,4,5 | notebook 10 (AR + bidirectional encoder) |
| S5_k0234 | 5 | 0,2,3,4 | notebook 14 (AR) |
| S4_k0234 | 4 | 0,2,3,4 | notebook 16 (AR) |
| S6_k02345 | 6 | 0,2,3,4,5 | notebook 16 (AR) |

**Fixed across all runs.** Visible `<identity>` fillers, learned positions, seed 42, 5000 × 256, the same recipe as
the AR runs, and checkpoint selection on validation digit CE over the trained k. Singletons are tested at every slot.

**Initialization.** One common initialization per architecture across the four settings. The S4/S6 slot table has
one extra row, drawn from a separate generator.

**Read-only comparison.** The AR / encoder runs are only read, never modified.

**Send back `Causal_Encoder_Review.ipynb`.** The last cell prints and downloads it.
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_SETTINGS = ["S5_k02345", "S5_k0234", "S4_k0234", "S6_k02345"]
RUN_MODELS = ["causal_encoder", "looped_causal_encoder"]
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
EVAL_CHUNK_GRIDS = 100
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import causal_encoder_exp as CE
print("results ->", CE.family_root(RESULTS_ROOT, MODE))
display(CE.init_audit(SEED))
'''

DRIVER = r'''# --- Train + evaluate sequentially (resumable) ---------------------------------------------------------------------
status = CE.run_queue(RESULTS_ROOT, MODE, SEED, RUN_SETTINGS, RUN_MODELS, repo_dir=str(REPO_DIR),
                      chunk_grids=EVAL_CHUNK_GRIDS, device=DEVICE)
display(status)
'''

REPORT = r'''# --- Report vs existing AR / bidirectional-encoder runs (read-only) + review export --------------------------------
rp = CE.report(RESULTS_ROOT, MODE, SEED)
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
