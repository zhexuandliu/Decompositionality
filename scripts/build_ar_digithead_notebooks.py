"""Generate notebooks/35a_/35b_: 10-way digit head (36-digit loss, no EOS target) WITHOUT target coordinates."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_ar_ablation_notebook import CONFIG, DRIVER, REPORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARTS = {
    "a": ("ar_digithead", "35a_ar_digithead_S7_k02345_seed42.ipynb",
          "the ORIGINAL Transformer AR (pos(t) kept, own-token coordinates) with ONLY the output changed"),
    "b": ("ar_slotpos_digithead", "35b_ar_slotpos_digithead_S7_k02345_seed42.ipynb",
          "the Transformer AR without pos(t) (op-slot embedding), own-token coordinates, with the output changed"),
}

INTRO = """# `__MODEL__` on S7_k02345, seed 42 (notebook __NB__)

This is __DESC__:
* **10-way digit head** (initialized from the original head's digit rows);
* **loss on the 36 answer digits only** (no EOS target);
* **36 decoding steps**; `<EOS>` is appended, not predicted.

No target coordinates: every output token keeps its OWN cell's (row, col), as in the ordinary AR.

| model | pos(t) | head / loss | target coords | notebook |
|---|---|---|---|---|
| original AR | yes | 20-way, 36 + EOS | no | 23b |
| `ar_digithead` | yes | **10-way, 36** | no | 35a |
| `ar_slotpos_digithead` | no | **10-way, 36** | no | 35b |
| `ar_slotpos_digithead_target` | no | 10-way, 36 | yes | 34 (singletons 1.00, k=6 1.00, k=7 0.985) |

**Questions.** 35a: does removing the EOS target alone fix the original AR? 35b vs 34: are target coordinates still
needed once the EOS target is gone?

Run 35a and 35b at the same time. The last cell (rerun after both finish) shows the full comparison table.
**Send back `AR_Ablation_S7_k02345_Review.ipynb`.**
"""


def build(part):
    model, name, desc = PARTS[part]
    p = ROOT / "notebooks" / name
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = CONFIG.replace('RUN_MODELS = ["ar_slotpos", "ar_digiteos", "ar_slotpos_digiteos"]', f'RUN_MODELS = ["{model}"]')
    assert cfg != CONFIG
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    intro = INTRO.replace("__MODEL__", model).replace("__NB__", "35" + part).replace("__DESC__", desc)
    nb.cells = [nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(cfg),
                nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for part in PARTS:
        print(build(part))
