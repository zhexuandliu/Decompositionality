"""Generate notebooks/28_S7_sweep_combine_seeds.ipynb: read-only combination of seeds 42 / 43 / 44."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "28_S7_sweep_combine_seeds.ipynb"

INTRO = """# S=7 support sweep — combine seeds 42, 43 and 44 (read-only)

**Sources.**
* Seed 42: notebooks 23a/23b.
* Seed 43: notebook 27a.
* Seed 44: notebook 27b.

**Reads only.** This notebook reads saved predictions; it does not train or re-evaluate, so a CPU runtime is
enough. Unfinished runs show as "not yet available" and are left out of the averages, so it can be rerun at any
time.

**Outputs.**
1. Availability of every (condition, model, seed).
2. Exact accuracy by condition, model and test k: mean ± std over seeds, plus each seed's value.
3. One panel per training support: accuracy vs test k (k=1 = singletons).
   * Thick line = mean over seeds, band = min–max, thin lines = individual seeds.
   * Grey = trained k; open markers = held out.
4. Singleton / +1 step / +2 step accuracy vs longest trained k (line = mean, dots = seeds).
5. A compact summary table of the held-out numbers per support and model.

**Send back `S7_Sweep_MultiSeed_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = "full"
SEEDS = [42, 43, 44]
SUBDIR = "S7_support_sweep_20k_v1"
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import longk as LK, longk_exp as LX
'''

RUN = r'''# --- Combined report (reads saved artifacts only) -------------------------------------------------------------------
rp = LX.multiseed_report(RESULTS_ROOT, MODE, SEEDS, LK.S7_SWEEP, SUBDIR)
print("REVIEW FILE TO SEND BACK:", rp)
'''

SUMMARY = r'''# --- Compact held-out summary: mean [min, max] over seeds per support x model ---------------------------------------
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, SEEDS, LK.S7_SWEEP, SUBDIR)
ok = df[df.status == "available"]
fmt = lambda s: f"{s.mean():.3f} [{s.min():.3f}, {s.max():.3f}] (n={s.count()})"
held = ok[ok.role != "trained"]
summary = held.pivot_table(index=["condition", "model"], columns="role", values="exact_acc", aggfunc=fmt)
trained = ok[ok.role == "trained"].groupby(["condition", "model"]).exact_acc.agg(lambda s: f"{s.min():.4f}")
summary["trained k (min over k, seeds)"] = trained
cols = [c for c in ["trained k (min over k, seeds)", "singleton", "+1 step", "+2 step", "+3 step", "+4 step", "+5 step"]
        if c in summary.columns]
summary = summary[cols].reset_index()
display(summary)
summary.to_csv(LX.family_root(RESULTS_ROOT, MODE, SUBDIR) / "report" / "multiseed" / "tables" / "heldout_summary.csv", index=False)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(RUN), nbf.v4.new_code_cell(SUMMARY)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
