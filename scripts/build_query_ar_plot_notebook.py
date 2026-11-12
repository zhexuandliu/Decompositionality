"""Generate notebooks/30_plot_query_ar_vs_causal_encoder.ipynb (read-only plots of the S=7 sweep results)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "30_plot_query_ar_vs_causal_encoder.ipynb"

INTRO = """# Query AR vs causal encoder on the S=7 sweep (read-only plots)

Run this after notebooks 29a / 29b (query_ar, seed 42) have finished. It reads saved predictions only: no
training and no re-evaluation, so a CPU runtime is enough.

**Models compared by default.**
* `query_ar` (notebooks 29a/29b): identical to the causal encoder except that output position j also receives the
  digit of cell j−1, and it decodes cell by cell.
* `causal_encoder` (notebooks 23a/23b, and 27a/27b for extra seeds): 36 output queries, all predicted in one pass.

Both use the same test records. The default seed is 42, the only seed query_ar has so far. Add `"transformer_ar"`
to `MODELS` to also show the original AR.

**Plots.**
1. One panel per training support (k02 … k0234567): exact accuracy vs test k (k=1 = singletons). Grey = trained k,
   open markers = held out, dashed line = longest trained k.
2. Singleton / +1 step / +2 step accuracy vs longest trained k.
3. Tables: accuracy by condition, model and k; and a compact held-out summary.

**Send back `QueryAR_vs_CausalEncoder_Review.ipynb`.**
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = "full"
SEEDS = [42]                                         # query_ar has seed 42 only (so far)
MODELS = ["query_ar", "causal_encoder"]              # add "transformer_ar" to include the original AR
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

RUN = r'''# --- Plots + tables (reads saved artifacts only) ---------------------------------------------------------------------
rp = LX.multiseed_report(RESULTS_ROOT, MODE, SEEDS, LK.S7_SWEEP, SUBDIR, models=MODELS,
                         out_name="query_ar_vs_causal_encoder", review_name="QueryAR_vs_CausalEncoder_Review.ipynb")
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, SEEDS, LK.S7_SWEEP, SUBDIR, MODELS)
ok = df[df.status == "available"]
held = ok[ok.role != "trained"]
summary = held.pivot_table(index=["condition", "model"], columns="role", values="exact_acc", aggfunc="mean").round(4)
summary["trained k (min)"] = ok[ok.role == "trained"].groupby(["condition", "model"]).exact_acc.min().round(4)
cols = [c for c in ["trained k (min)", "singleton", "+1 step", "+2 step", "+3 step", "+4 step", "+5 step"] if c in summary.columns]
print("Held-out summary (mean over SEEDS):")
display(summary[cols].reset_index())
print("REVIEW FILE TO SEND BACK:", rp)
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
                nbf.v4.new_code_cell(RUN)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
