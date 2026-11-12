"""Generate notebooks/24_S7_longer_k_analysis.ipynb (read-only analysis of notebook 23a/23b results)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "24_S7_longer_k_analysis.ipynb"

INTRO = """# S=7 longer-instruction analysis (read-only; uses the saved notebook 23a/23b predictions)

No training and no re-evaluation; a CPU runtime is enough. Results are for the LAST checkpoint.

**Analysed:** S7_k02345 (held-out k=6, which leaves one `<identity>`, and k=7, which leaves none) and S7_k023456
(held-out k=7), for Transformer AR and the causal encoder.

**What it reports:**
1. k=6 accuracy by WHICH slot holds the single `<identity>`.
2. k=6 / k=7 accuracy by number of C tokens and by whether the overall transform occurs in training. Also per-word
   accuracy (full per-word CSV).
3. Failure types: identity copy, wrong legal transform (and how many have the wrong mirror class), non-legal grid,
   invalid output.
4. Whether a wrong prediction equals the word with ONE operation removed (as if that slot were `<identity>`), by
   the removed operation's position.

**Check.** The sampled test set is rebuilt deterministically and verified record-by-record against the saved
predictions.

**Send back `S7_LongerK_Analysis_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = "full"
SEED = 42
SUBDIR = "S7_support_sweep_20k_v1"          # notebook 23a/23b results
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import longk_analysis as LA
'''

RUN = r'''# --- Analysis + review export ---------------------------------------------------------------------------------------
rp = LA.analyse(RESULTS_ROOT, MODE, SEED, SUBDIR)
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
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(RUN)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
