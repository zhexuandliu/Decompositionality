"""Generate notebooks/32_nexttoken_ar_S7_k02345_seed42.ipynb: query AR with ordinary next-token feedback."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_ar_ablation_notebook import CONFIG, DRIVER, REPORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "32_nexttoken_ar_S7_k02345_seed42.ipynb"

INTRO = """# Query AR with ordinary next-token feedback (`nexttoken_ar`), S7_k02345, seed 42

Everything is as in query AR: no delimiters, no 1-D position, slot / row-col / type embeddings, causal attention,
10-way head, 36-cell loss, cell-by-cell decoding. The one exception is what sits at the output positions.

| output position j | query AR | nexttoken_ar (ordinary AR) |
|---|---|---|
| j = 0 | query + type + row/col of cell 0 + start vector | start vector + type (no row/col, like `<OUTPUT>`) |
| j ≥ 1 | query + type + row/col of **cell j** + digit of cell j−1 | digit of cell j−1 + type + row/col of **cell j−1** |
| predicts | cell j | cell j |

So in nexttoken_ar each answer digit is an ordinary token carrying its OWN coordinates, and the position predicts the
NEXT cell, exactly as in the Transformer AR. There is no query vector. Under the same seed, all tensors equal query
AR's initial values.

**Comparison isolated:** nexttoken_ar vs query AR = "coordinates of the current token" vs "coordinates of the target
cell" (plus the query vector).

The last cell reruns the notebook-31 comparison table with this model added.
**Send back `AR_Ablation_S7_k02345_Review.ipynb`.**
"""


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = CONFIG.replace('RUN_MODELS = ["ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos"]', 'RUN_MODELS = ["nexttoken_ar"]')
    assert cfg != CONFIG
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(cfg),
                nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
