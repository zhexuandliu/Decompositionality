"""Generate notebooks/37_shortcut_analysis_S7_short_k.ipynb (read-only analysis of saved predictions)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "37_shortcut_analysis_S7_short_k.ipynb"

INTRO = """# What do the models compute when max trained k is short? (S=7, read-only)

With S = 7 slots, every model fails on held-out singletons and longer instructions when trained on k02, k023 or
k0234, and succeeds from k02345 on. This notebook reads the saved predictions (seed 42, LAST checkpoint) of:
* original AR;
* AR34 (no pos(t), target coordinates, 10-way head);
* query AR;
* causal encoder.

It asks what transform the models actually apply. No training is involved; a CPU runtime is enough.

1. **Singletons.** Outcome shares: correct, identity copy (input unchanged), another legal transform, no legal
   transform, invalid.
2. **Wrong singletons.** The transform applied, written as its shortest A/B/C word, and whether that transform is
   produced by some TRAINED instruction.
3. **Per operation and slot.** The modal transform and how consistently it is used across grids (systematic rule vs
   noise).
4. **First held-out longer k.** The same outcome classes, plus "equals the word with one operation dropped".

S7_k02345 is included as the success reference. **Send back `Shortcut_Analysis_S7_short_k_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = "full"
SEED = 42
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import shortcut_analysis as SA
'''

RUN = r'''# --- Analysis (reads saved predictions only) ---------------------------------------------------------------------
rp = SA.analyse(RESULTS_ROOT, MODE, SEED)
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
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(RUN)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
