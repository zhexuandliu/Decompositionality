"""Generate notebooks/43_aligned_ar_trace_sft_S5_k023_seed42.ipynb: Aligned AR that writes every intermediate grid."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG, DRIVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "43_aligned_ar_trace_sft_S5_k023_seed42.ipynb"

INTRO = """# Intermediate-grid supervision (SFT) for Aligned AR — 5 slots, trained k = 0, 2, 3 (seed 42)

**Why.** Aligned AR fails on this setting (notebook 42): no single operation is correct (A → AA in 43%, C returned
unchanged = CC in 24%, the rest non-legal grids), and 4-operation instructions drop a middle operation. Our explanation is an ordinal template
("1st … last" roles) that fits every final answer. Supervising the grid after EVERY slot rules that template out:
the model must execute slot by slot, and `<identity>` must leave the grid unchanged.

**Model.** Aligned AR (no sequence position, op-slot embedding, target-cell coordinates, 10-way digit head), but the
output is the trace:

`<BOS> <OPS> [5 slots] <GRID> [36 input] <OUTPUT> [grid after slot 1] … [grid after slot 5] <EOS>`

The last grid is the answer, and there are no separators between grids. Every output position carries the
coordinates of the cell it predicts **and the step it belongs to**. All 5 × 36 digits are supervised, with no EOS
target. Initialization is identical to Aligned AR.

**Step embedding.** Step s uses the op-slot embedding of slot s, so the grid of step s is aligned with the operation
it executes (`ar_trace_slotstep`). A control with a separate step table (`ar_trace_sepstep`) exists but is not run
here.

**Evaluation.** Free-running generation of the whole trace; only the final grid is scored. This uses a smaller
sampled test set than before (the trace is 5× longer to decode). Tests: single operations at every slot, k = 4, k = 5.
The reference is Aligned AR without a trace on the same setting (notebook 42: single operations 0.00, k = 4 0.00,
k = 5 0.002).

**Send back `AlignedAR_trace_SFT_S5_k023_Review.ipynb`.**
"""

REPORT = r'''# --- Comparison with Aligned AR without a trace (notebook 42) ---------------------------------------------------------
from decompositionality import shortcut_analysis as SA
models = ["ar_slotpos_digithead_target", "ar_trace_slotstep"]
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, [SEED], ["S5_k023"], SUBDIR, models)
ok = df[df.status == "available"]
print("Exact accuracy of the FINAL grid (rows: model, columns: test k; k = 1 = single operations; trained k = 0, 2, 3):")
display(ok.pivot_table(index="model", columns="k", values="exact_acc").round(4))
rp = SA.analyse(RESULTS_ROOT, MODE, SEED, conditions=["S5_k023"], models=models,
                review_name="AlignedAR_trace_SFT_S5_k023_Review.ipynb", out_name=f"seed_{SEED}_trace_sft_S5_k023")
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
           .replace('RUN_MODELS = ["query_ar"]', 'RUN_MODELS = ["ar_trace_slotstep"]')
           .replace('EVAL_PLAN = {"n_single_grids": 1000, "n_seen_per_k": 5000, "n_long_per_k": 20000}',
                    'EVAL_PLAN = {"n_single_grids": 500, "n_seen_per_k": 1000, "n_long_per_k": 4000}   # smaller: 4x longer decode'))
    assert '"ar_trace_slotstep"' in cfg and '"n_single_grids": 500' in cfg
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
