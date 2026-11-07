"""Generate notebooks/17_ar_identity_mirror_diagnostics.ipynb (diagnostic only; reads saved predictions; no training).
Refuses to overwrite a notebook with saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "17_ar_identity_mirror_diagnostics.ipynb"

INTRO = """# AR identity / mirror diagnostics (notebook 17): what are Transformer AR and Looped AR doing wrong?

**Diagnostic only.**
* Reads the SAVED free-running predictions of the identity-format AR runs: fixed S=5 supports 02/023/0234/02345 and
  the notebook-16 slot-count runs S=2/3/4/6, plus optional variable_length history.
* No training, fine-tuning, re-evaluation or new inputs. CPU is enough.

**Method.**
* Every singleton, the selected same-target controls and all 729,000 six-step records per S=6 model are matched EXACTLY
  to the 288 legal spatial transforms. Ambiguous match sets are kept, never broken arbitrarily.
* Wrong legal outputs are split into extra output-C / extra input-C / other residual classes.
* Also: same-target pairs (A~ACC, B~BCC, C~CCC, C~CCCCC), held-out source-position copying maps, six-step strata by
  number of C tokens, and placement / slot-count / format comparisons.

**Questions answered in the final evidence table.**
1. Is singleton C predicted as identity, another legal transform, or a non-legal grid?
2. Under train023, do A/B failures share an extra-mirror error with C?
3. Can ACC/BCC/CCC be correct while A/B/C fail on the same grid?
4. Do S=6 six-step failures concentrate on odd C count / the wrong mirror class?
5. How do I placement and slot count change these error patterns?
6. Are high-cell / exact-wrong outputs stable wrong copying maps or input-dependent?

These are behavioural descriptions of outputs, NOT mechanistic claims.

**When it finishes, send back `AR_Identity_Diagnostics_Review.ipynb`.** The last cell prints its path and offers a
download. It is a self-contained snapshot with every table and figure baked in.
"""

CONFIG = r'''# --- CONFIGURATION ------------------------------------------------------------------------------------------------
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
ARTIFACT_ROOTS = [f"{RESULTS_ROOT}/identity_varlen_02345_all_models",     # S=5/train02345 (+ variable_length)
                  f"{RESULTS_ROOT}/identity_ar_count_extrapolation",      # S=5/train02, train023, train0234
                  f"{RESULTS_ROOT}/identity_ar_slot_count"]               # S=2/3/4/6 (notebook 16)
RUN_OVERRIDES = {}   # e.g. {"S5/train02/transformer_ar": "/content/drive/MyDrive/.../eval/transformer_ar/ARTIFACT.json"}
INCLUDE_VARIABLE_LENGTH = True        # optional history; never blocks the core analysis
RECOMPUTE_MISSING_PREDICTIONS = False # no training / re-evaluation is performed by this notebook
MAX_NEW_INFERENCE_RECORDS_PER_RUN = 30000
VERIFY_PREDICTION_SHA = True
NEAREST_PER_WORD = 20                 # six-step nearest-transform subset (stable hash), per word per model
OUT_ROOT = f"{RESULTS_ROOT}/ar_identity_mirror_diagnostics"
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
if RECOMPUTE_MISSING_PREDICTIONS:
    raise NotImplementedError("Bounded re-inference is not implemented in this version; missing sources are reported "
                              "as unavailable instead (no checkpoint is run).")
import pandas as pd
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
from decompositionality.mirror_report import Diagnostics
D = Diagnostics(RESULTS_ROOT, ARTIFACT_ROOTS, RUN_OVERRIDES, OUT_ROOT, INCLUDE_VARIABLE_LENGTH, VERIFY_PREDICTION_SHA,
                NEAREST_PER_WORD)
D.conventions()
'''

STAGES = [
    ("# --- 1. Discover runs (read-only) and audit sources -------------------------------------------------------\n"
     "D.discover()"),
    ("# --- 2. Load predictions, audits, exact group matching (cached by prediction sha), overview ------------------\n"
     "D.load()\nD.overview()"),
    "# --- 3. Singleton diagnostics -----------------------------------------------------------------------------\nD.singletons()",
    "# --- 4. Same-target comparisons (A~ACC, B~BCC, C~CCC, C~CCCCC) and identity controls ----------------------\nD.equivalence()",
    "# --- 5. Held-out source-position copying maps ---------------------------------------------------------------\nD.source_maps()",
    "# --- 6. S=6 six-operation diagnosis ------------------------------------------------------------------------\nD.six_step()",
    "# --- 7. Placement / slot-count / optional format comparisons -------------------------------------------------\nD.comparisons()",
    "# --- 8. Evidence table for the six questions ----------------------------------------------------------------\nD.evidence()",
    ("# --- 9. Exports + self-contained review notebook (SEND THIS FILE BACK) -----------------------------------------\n"
     "rp = D.export()\nprint('REVIEW FILE TO SEND BACK:', rp)\n"
     "if IN_COLAB:\n    from google.colab import files\n    files.download(str(rp))"),
]


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = ([nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG)]
                + [nbf.v4.new_code_cell(s) for s in STAGES])
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
