"""Generate notebooks/15_compare_identity_supports.ipynb (read-only comparison; no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTRO = """# Comparing training supports in the five-slot identity format (no training)

Transformer AR and Looped AR trained on k ∈ {0,2} (nb 12), {0,2,3} (nb 13), {0,2,3,4} (nb 14) and {0,2,3,4,5} (nb 10).
This notebook only READS saved predictions.

**Checked before comparing:**
* the same test manifest;
* identical test records and targets;
* the same model configuration.

**Notebook 10 caveat:** it used a different training stream (20% per k, not the per-k indexed stream), so its run is
not stream-matched to 12–14.

**Reported:**
* exact accuracy by k, with trained vs held-out marked;
* singletons by operation × placement;
* k=5 split by whether the word's transformation appears in training;
* correctness transitions between consecutive supports on identical test records.

One seed: descriptive only.
"""
CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
SEED = 42
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
'''
RUN = r'''# --- Compare (read-only) ----------------------------------------------------------------------------------------
import pandas as pd
from IPython.display import Image, display
from decompositionality.support_comparison import run
pd.set_option("display.max_rows", 200); pd.set_option("display.width", 250)
R = run(RESULTS_ROOT, SEED)
if "compatibility" in R:
    display(R["compatibility"])
    display(R["by_k"].pivot_table(index=["model", "support"], columns="k", values="exact_correct", aggfunc="first"))
    display(R["transform_split"][R["transform_split"].k.isin([1, 3, 4, 5])])
    display(R["transitions"])
    for p in R["figures"]:
        if str(p).endswith(".png"):
            display(Image(filename=str(p)))
'''


def build() -> Path:
    p = ROOT / "notebooks" / "15_compare_identity_supports.ipynb"
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(RUN),
                nbf.v4.new_markdown_cell("Outputs: `<RESULTS_ROOT>/support_comparison/seed_42/` (CSV tables + figures).")]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
