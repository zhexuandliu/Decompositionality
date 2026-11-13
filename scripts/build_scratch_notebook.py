"""Generate notebooks/44_aligned_ar_scratch_S5_k023_seed42.ipynb: unsupervised scratch grids, final-answer loss only."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG, DRIVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "44_aligned_ar_scratch_S5_k023_seed42.ipynb"

INTRO = """# Aligned AR with unsupervised scratch grids — 5 slots, trained k = 0, 2, 3 (seed 42)

**Why.** Supervising every intermediate grid (notebook 43) solves this setting, but it shows the model each single
operation applied to the input, so single operations are no longer held out. Here the intermediate grids are
**not supervised** and the model never sees them. It only gets *positions* for them, and the loss is on the final
answer only.

**Model.** The same sequence as notebook 43:

`<BOS> <OPS> [5 slots] <GRID> [36 input] <OUTPUT> [scratch for step 1] … [scratch for step 4] [final grid] <EOS>`

* Each scratch position holds one learned scratch vector (no digit), plus the **step = op-slot embedding of slot s**
  and the (row, col) of its cell. These are the same position encodings as notebook 43.
* There is no loss on the scratch positions and nothing is fed back into them. They are free "thinking space",
  computed in parallel, and the whole model is trained end to end by ordinary backpropagation (no RL).
* Only the final grid is predicted, autoregressively, with target-cell coordinates and a 10-way head. The loss is on
  its 36 digits only.

Initialization is identical to Aligned AR. The test records are the same as in notebook 42 (Aligned AR without
scratch: single operations 0.00, k = 4 0.00, k = 5 0.002); notebook 43 with supervised grids reached 1.00 / 1.00 /
0.9998.

**Send back `AlignedAR_scratch_S5_k023_Review.ipynb`.**
"""

REPORT = r'''# --- Comparison: no scratch (42), supervised grids (43), unsupervised scratch (this notebook) -------------------------
from decompositionality import shortcut_analysis as SA
models = ["ar_slotpos_digithead_target", "ar_trace_slotstep", "ar_scratch_slotstep"]
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, [SEED], ["S5_k023"], SUBDIR, models)
ok = df[df.status == "available"]
print("Exact accuracy of the final grid (rows: model, columns: test k; k = 1 = single operations; trained k = 0, 2, 3):")
display(ok.pivot_table(index="model", columns="k", values="exact_acc").round(4))
rp = SA.analyse(RESULTS_ROOT, MODE, SEED, conditions=["S5_k023"], models=models,
                review_name="AlignedAR_scratch_S5_k023_Review.ipynb", out_name=f"seed_{SEED}_scratch_S5_k023")
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = (CONFIG.replace("__CONDS__", repr(["S5_k023"]))
           .replace('RUN_MODELS = ["query_ar"]', 'RUN_MODELS = ["ar_scratch_slotstep"]'))
    assert '"ar_scratch_slotstep"' in cfg
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
