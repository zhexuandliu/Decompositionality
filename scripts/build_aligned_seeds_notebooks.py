"""Generate notebooks/45_aligned_ar_S7_k02345_seeds43_44.ipynb: Aligned AR on S7_k02345 with seeds 43 and 44 (run one
after the other) + the 3-seed plot."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "45_aligned_ar_S7_k02345_seeds43_44.ipynb"

INTRO = """# Aligned AR on S7_k02345 — seeds 43 and 44, plus the 3-seed plot

Aligned AR (no sequence position, target-cell coordinates, 10-way digit head) on 7 slots, trained k = 0, 2, 3, 4, 5.
This is the same setting as notebook 34 (seed 42: single operations 1.00, k = 6 0.9996, k = 7 0.985). Only the
seed changes, which affects the initialization and the training stream; the test records are identical.

The two seeds run one after the other (resumable). The last cell plots all three seeds (42, 43, 44). The Standard AR
is included as a reference if its seed runs exist (notebooks 23 / 27).
**Send back `AlignedAR_S7_k02345_3seeds_Review.ipynb`.**
"""

DRIVER = r'''# --- Train (20,000 updates) + sampled LAST-checkpoint evaluation, seeds 43 and 44 (resumable) -------------------------
for SEED in SEEDS:
    print(f"\n######## seed {SEED} ########")
    status = LX.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, repo_dir=str(REPO_DIR), device=DEVICE,
                          subdir=SUBDIR, steps=STEPS if MODE == "full" else None, sampled_plan=EVAL_PLAN)
    display(status)
'''

REPORT = r'''# --- Three seeds (42, 43, 44): plot + per-seed table ---------------------------------------------------------------
from decompositionality import ar_ablation as AB
rp = AB.plot_seeds(RESULTS_ROOT, MODE, seeds=[42, 43, 44])
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    cfg = (CONFIG.replace("__CONDS__", repr(["S7_k02345"]))
           .replace('RUN_MODELS = ["query_ar"]', 'RUN_MODELS = ["ar_slotpos_digithead_target"]')
           .replace("SEED = 42", "SEEDS = [43, 44]"))
    assert "SEEDS = [43, 44]" in cfg and '"ar_slotpos_digithead_target"' in cfg
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
