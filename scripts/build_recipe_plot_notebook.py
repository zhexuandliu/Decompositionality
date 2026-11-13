"""Generate notebooks/41_plot_ar_recipe_S7_k02345.ipynb (read-only figures)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_shortcut_notebook import CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "41_plot_ar_recipe_S7_k02345.ipynb"

INTRO = """# Figures: which changes make the Transformer AR generalize? (S7_k02345, seed 42, read-only)

Four models trained on S7_k02345 (trained k = 0,2,3,4,5 in 7 slots):

| model | pos(t) | output head | answer-position coordinates | notebook |
|---|---|---|---|---|
| Original AR | yes | 20-way, + EOS | own token's cell | 23b |
| No pos(t) + 10-way head | no | 10-way digits | own token's cell | 35b |
| pos(t) + 10-way head + target coords | yes | 10-way digits | cell being predicted | 35c |
| **AR34** | no | 10-way digits | cell being predicted | 34 |

Figures:
1. accuracy by test length k;
2. held-out accuracy (singletons A/B/C, k = 6, k = 7);
3. a table.

PNGs are saved next to the review file. A CPU runtime is enough.
**Send back `AR_Recipe_Figures_Review.ipynb`.**
"""

RUN = r'''# --- Figures (reads saved predictions only) --------------------------------------------------------------------------
from decompositionality import ar_ablation as AB
rp = AB.plot_recipe(RESULTS_ROOT, MODE, SEED)
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
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP),
                nbf.v4.new_code_cell(CONFIG.replace("from decompositionality import shortcut_analysis as SA", "")),
                nbf.v4.new_code_cell(RUN)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
