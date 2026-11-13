"""Generate notebooks/43_aligned_ar_trace_sft_S4_k02_seed42.ipynb: Aligned AR that writes every intermediate grid."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG, DRIVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "43_aligned_ar_trace_sft_S4_k02_seed42.ipynb"

INTRO = """# Intermediate-grid supervision (SFT) for Aligned AR — 4 slots, trained k = 0, 2 (seed 42)

**Why.** Aligned AR fails on this setting (notebook 42): it applies a single operation twice (A → AA 100% at
slots 1–3) and drops the middle operation of a 3-operation instruction. Our explanation is an ordinal template
("1st … last" roles) that fits every final answer. Supervising the grid after EVERY slot rules that template out:
the model must execute slot by slot, and `<identity>` must leave the grid unchanged.

**Model.** Aligned AR (no sequence position, op-slot embedding, target-cell coordinates, 10-way digit head), but the
output is the trace:

`<BOS> <OPS> [4 slots] <GRID> [36 input] <OUTPUT> [grid after slot 1] [after slot 2] [after slot 3] [after slot 4] <EOS>`

The last grid is the answer, and there are no separators between grids. Every output position carries the
coordinates of the cell it predicts **and the step it belongs to**. All 4 × 36 digits are supervised, with no EOS
target. Initialization is identical to Aligned AR.

| run | step embedding |
|---|---|
| `ar_trace_slotstep` (main) | the op-slot embedding of slot s (step s aligned with the operation it executes) |
| `ar_trace_sepstep` (control) | an independent learned step table |

**Evaluation.** Free-running generation of the whole trace; only the final grid is scored. This uses a smaller
sampled test set than before (the trace is 4× longer to decode). The reference is Aligned AR without a trace on the
same setting (notebook 42: single operations 0.00, k = 3 0.00, k = 4 0.06).

**Send back `AlignedAR_trace_SFT_S4_k02_Review.ipynb`.**
"""

REPORT = r'''# --- Comparison with Aligned AR without a trace (notebook 42) ---------------------------------------------------------
from decompositionality import shortcut_analysis as SA
models = ["ar_slotpos_digithead_target", "ar_trace_slotstep", "ar_trace_sepstep"]
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, [SEED], ["S4_k02"], SUBDIR, models)
ok = df[df.status == "available"]
print("Exact accuracy of the FINAL grid (rows: model, columns: test k; k = 1 = single operations; trained k = 0, 2):")
display(ok.pivot_table(index="model", columns="k", values="exact_acc").round(4))
rp = SA.analyse(RESULTS_ROOT, MODE, SEED, conditions=["S4_k02"], models=models,
                review_name="AlignedAR_trace_SFT_S4_k02_Review.ipynb", out_name=f"seed_{SEED}_trace_sft_S4_k02")
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = (CONFIG.replace("__CONDS__", repr(["S4_k02"]))
           .replace('RUN_MODELS = ["query_ar"]', 'RUN_MODELS = ["ar_trace_slotstep", "ar_trace_sepstep"]')
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
