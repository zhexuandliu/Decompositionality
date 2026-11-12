"""Generate notebooks/26a_/26b_aux_task_*.ipynb (2 conditions x 2 models each; run in parallel)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"a": ("26a_aux_task_none_and_k02345.ipynb", ["aux_none", "aux_k02345"]),
         "b": ("26b_aux_task_k0234567_and_k01234567.ipynb", ["aux_k0234567", "aux_k01234567"])}

INTRO = """# Auxiliary-task transfer, part __PART__ of 2: __CONDS_TXT__

**Question.** Can an UNRELATED auxiliary task that covers long instructions, late positions and zero-identity
inputs teach the positional behaviour needed for the main task's longer instructions? The main task's own long
instructions are never trained.

**Main task (all conditions).**
* S=7 with `<identity>` fillers, operations A/B/C (they move cells).
* Trained k ∈ {0,2,3,4,5}.
* Tests: singletons at every slot, k=6, k=7. The test set is the sampled one from notebook 23, record for record.

**Auxiliary task.**
* Value operations D: d → d+1 (mod 10) and E: d → 9−d. They change digits but never move a cell, so no D/E word
  equals any A/B/C composition.
* Same 7 slots, same `<identity>` fillers, same random placement.
* Every instruction starts with a task marker (`<main>` / `<aux>`).

| condition | auxiliary training k | per update (main / aux) |
|---|---|---|
| aux_none | — | 256 / 0 (baseline) |
| aux_k02345 | 0,2,3,4,5 | 192 / 64 (extra task, no extra length) |
| aux_k0234567 | 0,2,3,4,5,6,7 | 192 / 64 (with long and zero-identity auxiliary instructions) |
| aux_k01234567 | 0,1,2,…,7 | 192 / 64 (also single D / E) |

**Shared across conditions.** Transformer AR and the causal encoder, 20,000 updates, one initialization per
architecture. Checkpoint selection uses the main task only; evaluation uses the LAST checkpoint.

**This notebook = part __PART__** (__CONDS_TXT__). The other half is notebook __OTHER__; run both at the same
time. **Send back `AuxTask_part__PART___Review.ipynb`.** After both finish, the optional last cell gives
`AuxTask_ALL_Review.ipynb`.
"""

CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")
SEED = 42
RUN_CONDITIONS = __CONDS__
ALL_CONDITIONS = ["aux_none", "aux_k02345", "aux_k0234567", "aux_k01234567"]
RUN_MODELS = ["transformer_ar", "causal_encoder"]
STEPS = 20000
EVAL_PLAN = {"n_single_grids": 1000, "n_seen_per_k": 5000, "n_long_per_k": 20000, "n_aux_per_k": 1000}
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import auxtask as AX, auxtask_exp as AXE
_c = AXE.make_cfg(RUN_CONDITIONS[0], RESULTS_ROOT, MODE, RUN_MODELS, SEED, **({"optimizer_steps": STEPS} if MODE == "full" else {}))
print("resolved recipe:", _c.train.semantic_dict())
print("results ->", AXE.family_root(RESULTS_ROOT, MODE))
'''

DRIVER = r'''# --- Train (20,000 updates) + LAST-checkpoint evaluation, sequentially (resumable) ----------------------------------
status = AXE.run_queue(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, EVAL_PLAN,
                       steps=STEPS if MODE == "full" else None, repo_dir=str(REPO_DIR), device=DEVICE)
display(status)
'''

REPORT = r'''# --- Report for THIS half -------------------------------------------------------------------------------------------
rp = AXE.report(RESULTS_ROOT, MODE, SEED, RUN_CONDITIONS, out_name=f"seed_{SEED}_part__PART__",
                review_name="AuxTask_part__PART___Review.ipynb")
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB:
    from google.colab import files
    files.download(str(rp))
'''

COMBINED = r'''# --- OPTIONAL: combined report (run after BOTH 26a and 26b finished) ------------------------------------------------
rp_all = AXE.report(RESULTS_ROOT, MODE, SEED, ALL_CONDITIONS, out_name=f"seed_{SEED}_all", review_name="AuxTask_ALL_Review.ipynb")
print("COMBINED REVIEW FILE:", rp_all)
if IN_COLAB:
    from google.colab import files
    files.download(str(rp_all))
'''


def build(part):
    name, conds = PARTS[part]
    other = "26b" if part == "a" else "26a"
    sub = lambda t: (t.replace("__CONDS__", repr(conds)).replace("__OTHER__", other).replace("__PART__", part)  # noqa: E731
                     .replace("__CONDS_TXT__", ", ".join(conds)))
    p = ROOT / "notebooks" / name
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(sub(INTRO)), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(sub(CONFIG)),
                nbf.v4.new_code_cell(DRIVER), nbf.v4.new_code_cell(sub(REPORT)), nbf.v4.new_code_cell(COMBINED)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for part in PARTS:
        print(build(part))
