"""Generate notebooks/36a_/36b_: the notebook-34 AR (ar_slotpos_digithead_target) on the rest of the S=7 sweep, seed 42."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG, DRIVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ar_slotpos_digithead_target"
PARTS = {"a": ["S7_k02", "S7_k023", "S7_k0234"], "b": ["S7_k023456", "S7_k0234567"]}

INTRO = """# Notebook-34 AR on the S=7 sweep — seed 42, part __PART__ of 2 (__CONDS_TXT__)

**Model `ar_slotpos_digithead_target`** (notebook 34). It is the original Transformer AR (same sequence with
delimiters, segment embeddings, teacher forcing, token-by-token decoding) with three changes:
1. **no pos(t)**: op tokens get an op-slot embedding; grid tokens only (row, col);
2. **target coordinates**: every output-part position carries the (row, col) of the cell it PREDICTS
   (`<OUTPUT>` → (0,0), answer of cell j−1 → cell j);
3. **10-way digit head**: loss on the 36 answer digits only (no EOS target), 36 decoding steps.

On S7_k02345 (notebook 34) it reached singletons 1.00, k=6 0.9996, k=7 0.985 — the same level as query AR.

**Setup.** Same as notebooks 23 / 29: S=7 with identity fillers, 20,000 updates, sampled evaluation of the LAST
checkpoint on the same test records, seed 42. S7_k02345 is already done (notebook 34).

**This notebook = part __PART__** (__CONDS_TXT__). The other half is notebook __OTHER__; run both at the same time.
The last cell (rerun after both finish) plots this model against the original Transformer AR, query AR and the
causal encoder on all six training supports. **Send back `AR34_vs_QueryAR_vs_Encoder_seed42_Review.ipynb`.**
"""

REPORT = r'''# --- Comparison on the whole S=7 sweep, seed 42 (rerun after both halves finish) -------------------------------------
MODELS_CMP = ["transformer_ar", "ar_slotpos_digithead_target", "query_ar", "causal_encoder"]
rp = LX.multiseed_report(RESULTS_ROOT, MODE, [42], LK.S7_SWEEP, SUBDIR, models=MODELS_CMP,
                         out_name="ar34_seed42", review_name="AR34_vs_QueryAR_vs_Encoder_seed42_Review.ipynb")
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, [42], LK.S7_SWEEP, SUBDIR, MODELS_CMP)
ok = df[df.status == "available"]
summary = ok[ok.role != "trained"].pivot_table(index=["condition", "role"], columns="model", values="exact_acc", sort=False)
print("Held-out accuracy (seed 42):")
display(summary[[m for m in MODELS_CMP if m in summary.columns]].round(4))
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build(part):
    conds = PARTS[part]
    other = "36b" if part == "a" else "36a"
    name = f"36{part}_ar34_S7_sweep_seed42_{'_'.join(c.replace('S7_', '') for c in conds)}.ipynb"
    p = ROOT / "notebooks" / name
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = CONFIG.replace("__CONDS__", repr(conds)).replace('RUN_MODELS = ["query_ar"]', f'RUN_MODELS = ["{MODEL}"]')
    assert f'"{MODEL}"' in cfg
    intro = (INTRO.replace("__PART__", part).replace("__OTHER__", other).replace("__CONDS_TXT__", ", ".join(conds)))
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(cfg),
                nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for part in PARTS:
        print(build(part))
