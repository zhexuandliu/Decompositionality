"""Generate notebooks/33_ar_target_coords_S7_k02345_seed42.ipynb: ar_slotpos_digiteos + target-cell coordinates."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_ar_ablation_notebook import CONFIG, DRIVER, REPORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "33_ar_target_coords_S7_k02345_seed42.ipynb"

INTRO = """# Transformer AR with target-cell coordinates (`ar_slotpos_digiteos_target`), S7_k02345, seed 42

Starting point: `ar_slotpos_digiteos` from notebook 31. It is the original Transformer AR (same sequence with
delimiters, segment embeddings, teacher forcing, 37-step decoding) with no pos(t) (op-slot embedding instead) and an
11-way head (digits + EOS).

**One change.** In the output part, each position's (row, col) embedding is that of the cell it PREDICTS, not that
of its own token:

| sequence position | `ar_slotpos_digiteos` | this model | predicts |
|---|---|---|---|
| `<OUTPUT>` | none | (0,0) | cell (0,0) |
| answer token of cell (0,0) | (0,0) | (0,1) | cell (0,1) |
| … | … | … | … |
| answer token of cell (5,4) | (5,4) | (5,5) | cell (5,5) |
| answer token of cell (5,5) | (5,5) | none | EOS |

Input cells keep their own coordinates, and nothing else changes. Under the same seed, all initial values are
identical to `ar_slotpos_digiteos`.

After this change, the position that predicts cell j holds "digit of cell j−1 + coordinates of cell j", exactly as in
query AR. What still differs from query AR: the `<BOS>`/`<OPS>`/`<GRID>` delimiters, the EOS target and 11- vs
10-way head, and the 8 segment vs 3 type embeddings.

The last cell reruns the notebook-31 comparison with this model added.
**Send back `AR_Ablation_S7_k02345_Review.ipynb`.**
"""


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = CONFIG.replace('RUN_MODELS = ["ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos"]',
                         'RUN_MODELS = ["ar_slotpos_digiteos_target"]')
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
