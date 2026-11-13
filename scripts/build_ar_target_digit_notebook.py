"""Generate notebooks/34_ar_target_coords_digit_head_S7_k02345_seed42.ipynb."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_ar_ablation_notebook import CONFIG, DRIVER, REPORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "34_ar_target_coords_digit_head_S7_k02345_seed42.ipynb"

INTRO = """# Transformer AR + target coordinates, 10-way head, 36 digits (`ar_slotpos_digithead_target`), S7_k02345, seed 42

Starting point: `ar_slotpos_digiteos_target` (notebook 33). It has no pos(t) (op-slot embedding instead), an
11-way head (digits + EOS), and target-cell coordinates in the output part.

**One change: the loss and the output head.**

| | notebook 33 | this model |
|---|---|---|
| output head | 11-way (digits + EOS) | **10-way (digits only)** |
| loss | 36 digits + EOS (37 targets) | **36 digits** |
| decoding | 37 steps (EOS predicted) | **36 steps; EOS is appended (forced), not predicted** |

The sequence, delimiters (incl. the final `<EOS>` token), embeddings and target coordinates are unchanged. Under the
same seed, all initial values are identical to notebook 33 (the 10-way head = the digit rows of the 11-way head).

What still differs from query AR: the `<BOS>`/`<OPS>`/`<GRID>` delimiters, segment vs type embeddings, and the
initialization (AR family vs causal-encoder family).

The last cell reruns the notebook-31 comparison with this model added.
**Send back `AR_Ablation_S7_k02345_Review.ipynb`.**
"""


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = CONFIG.replace('RUN_MODELS = ["ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos"]',
                         'RUN_MODELS = ["ar_slotpos_digithead_target"]')
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
