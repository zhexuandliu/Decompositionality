"""Generate notebooks/42a_/42b_: Aligned AR (ar_slotpos_digithead_target) on short supports with 2 empty slots."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG, DRIVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"a": ("S5_k023", "5 slots, trained k = 0, 2, 3"), "b": ("S4_k02", "4 slots, trained k = 0, 2")}

INTRO = """# Aligned AR on a short support with only 2 empty slots: __DESC__ (seed 42, notebook __NB__)

**Question.** Aligned AR (no sequence position, target-cell coordinates, 10-way digit head) learns single
operations whenever every training instruction leaves at most 2 slots empty. So far, however, it has only been
tested with a longest training instruction of 4 or more. Here the instructions are short (longest __K__) but still
leave only 2 slots empty.
* **Success** → the number of empty slots is what matters.
* **Failure** → a short longest instruction (≤ 3) is itself a problem.

Standard AR failed on 5 slots / k ≤ 3 (0/15000 singletons, earlier experiments).

**Setup.** Same model, data stream, 20,000 updates and sampled LAST-checkpoint evaluation as notebooks 34 / 36 / 39.
The other notebook (__OTHER__) runs the second setting; run both at the same time. The last cell (rerun after both
finish) shows accuracy by k and the failure analysis for both settings. **Send back `AlignedAR_gap2_short_Review.ipynb`.**
"""

REPORT = r'''# --- Accuracy and outcome analysis for both settings (rerun after both notebooks finish) ------------------------------
from decompositionality import shortcut_analysis as SA
conds = LK.GAP2_SHORT
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, [SEED], conds, SUBDIR, RUN_MODELS)
ok = df[df.status == "available"]
print("Exact accuracy (rows: setting, columns: test k; k = 1 = single operations):")
display(ok.pivot_table(index="condition", columns="k", values="exact_acc").round(4))
rp = SA.analyse(RESULTS_ROOT, MODE, SEED, conditions=conds, models=RUN_MODELS,
                review_name="AlignedAR_gap2_short_Review.ipynb", out_name=f"seed_{SEED}_aligned_gap2_short")
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build(part):
    cond, desc = PARTS[part]
    other = "42b" if part == "a" else "42a"
    name = f"42{part}_aligned_ar_{cond}_seed42.ipynb"
    p = ROOT / "notebooks" / name
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = (CONFIG.replace("__CONDS__", repr([cond]))
           .replace('RUN_MODELS = ["query_ar"]', 'RUN_MODELS = ["ar_slotpos_digithead_target"]'))
    assert '"ar_slotpos_digithead_target"' in cfg
    intro = (INTRO.replace("__DESC__", desc).replace("__NB__", "42" + part).replace("__OTHER__", other)
             .replace("__K__", "3" if cond == "S5_k023" else "2"))
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
