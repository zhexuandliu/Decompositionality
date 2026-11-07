"""Generate notebooks/06_placement_probe_mixed_02345.ipynb (clean). Notebooks 01-05 are not touched.
Run:  python scripts/build_probe_notebook.py"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Placement probe: does single-operation behaviour depend on the slot position?

**Inference only.** This notebook loads the already-selected checkpoints (`runs/<model>/best.pt`) of the
mixed-length experiments (notebooks 04 and 05). It does not train, fine-tune or re-select anything, and it uses no
optimizer. AR models decode free-running.

**Motivation.** In the standard test, a single operation is encoded as `[X, PAD, PAD, PAD, PAD]`. Several models
output "X then C" (A→AC, B→BC, C→CC = identity) on it. This probe asks whether that behaviour depends on where the
operation sits:

* **Singles:** A, B and C, each placed in slot 1, 2, 3, 4 and 5 (all other slots PAD), on the 1000 test base grids.
* **Control:** the experiment's trained length-two pairs in all 10 order-preserving placements (all of which occur
  in training), on 200 grids. This shows whether non-canonical placements are handled correctly for a TRAINED
  length.

Each prediction is checked against the truth and against explicit candidate rules: "X then C", "X twice", identity,
and every original training pair. It is also checked against all 288 legal transformations (the dominant wrong one
is reported). Results are behavioural, from one seed, and non-causal.
"""

CONFIG = r'''# --- CONFIGURATION ---------------------------------------------------------------------------------------
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
MIXED_ROOT = os.environ.get("DECOMP_MIXED_ROOT") or f"{RESULTS_ROOT}/mixed_length_02345"
EXPERIMENTS = ["mixed_02345_distinct_pairs", "mixed_02345_all_pairs"]
MODELS = os.environ.get("DECOMP_MODELS", "transformer_ar,looped_transformer_ar,transformer_encoder,"
                        "looped_transformer_encoder,vit,looped_vit,trm,rescnn_film,unet_film").split(",")
N_SINGLE_GRIDS = int(os.environ.get("DECOMP_PROBE_GRIDS", 1000))   # singles: 3 ops x 5 slots x grids
N_PAIR_GRIDS = int(os.environ.get("DECOMP_PROBE_PAIR_GRIDS", 200))  # control: pairs x 10 placements x grids
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
OUTPUT_DIR = f"{MIXED_ROOT}/placement_probe"
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
print("reading checkpoints from:", MIXED_ROOT)
print("writing probe outputs to:", OUTPUT_DIR)
'''

RUN = r'''# --- Run the probe (cached per experiment/model; inference only) -------------------------------------------
import pandas as pd
from IPython.display import Image, display
from decompositionality.placement_probe import run_probe, plot_probe
pd.set_option("display.max_rows", 400); pd.set_option("display.width", 250)
res, out = run_probe(MIXED_ROOT, EXPERIMENTS, MODELS, OUTPUT_DIR, N_SINGLE_GRIDS, N_PAIR_GRIDS, DEVICE, DTYPE)
paths = plot_probe(res, out / "figures")
for p in paths:
    if str(p).endswith(".png"):
        display(Image(filename=str(p)))
'''

TABLES = r'''# --- Numeric tables -----------------------------------------------------------------------------------------
cols = ["experiment", "model", "string", "positions", "exact_correct", "n", "exact_acc", "match: X then C",
        "match: X twice", "match: identity", "any_legal_match_rate", "dominant_wrong_transform", "dominant_wrong_share"]
singles = res[res.kind == "single"][cols]
display(singles.style.format(precision=3).hide(axis="index"))
ctrl = res[res.kind == "pair_control"]
display(ctrl.groupby(["experiment", "model", "canonical_placement"])[["exact_correct", "n"]].sum()
        .assign(exact_acc=lambda d: d.exact_correct / d.n))
'''

EXPORT = r'''# --- Export -------------------------------------------------------------------------------------------------
import shutil
arch = shutil.make_archive(str(out) + "_export", "zip", root_dir=out, base_dir=".")
print("tables/figures:", out, "| archive:", arch)
'''


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(INTRO), code(BOOTSTRAP), code(CONFIG), code(RUN), code(TABLES), code(EXPORT),
                md("Outputs: `<RESULTS_ROOT>/mixed_length_02345/placement_probe/` (per-model `probe_predictions.npz` "
                   "+ `probe_meta.json` with the checkpoint step, `placement_probe_results.csv`, `figures/`).")]
    p = ROOT / "notebooks" / "06_placement_probe_mixed_02345.ipynb"
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
