"""Generate notebook 38 (read-only mechanism analysis) and notebooks 39a/39b (AR34 trained on k0234 at S = 4/5/6)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402
from build_query_ar_notebooks import CONFIG as TRAIN_CONFIG, DRIVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

NB38_INTRO = """# Mechanism analysis: first/last anchoring? (read-only, S=7, seed 42)

Notebook 37 showed a striking pattern: models trained on S7_k02 execute a singleton TWICE (A → AA, B → BB,
C → CC = unchanged), and on k=3 they return a two-operation result. Hypothesis: with short training instructions,
the model organises the computation around the FIRST and the LAST real operation. A singleton has first = last
(executed twice), and a longer instruction loses its middle operation(s).

Two read-only checks; nothing is trained:
1. **Which operation is dropped** (from the saved predictions): for held-out longer instructions whose prediction
   equals the word with one operation removed, the dropped position. The hypothesis predicts MIDDLE.
2. **Attention probe** (loads the LAST checkpoints): how much attention the answer positions and the input cells pay
   to the FIRST / MIDDLE / LAST real operation and to the `<identity>` fillers, per layer and per head. This covers
   S7_k02, S7_k0234 and S7_k02345 (the success reference), for AR34, query AR, causal encoder and original AR.

A CPU runtime works but is slow; a GPU makes the probe much faster (set `DEVICE = "cuda"`).
**Send back `Shortcut_Analysis_S7_short_k_Review.ipynb` and `Attention_Probe_Review.ipynb`.**
"""

NB38_CONFIG = r'''# --- CONFIGURATION -------------------------------------------------------------------------------------------------
MODE = "full"
SEED = 42
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import pandas as pd
pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250)
from decompositionality import shortcut_analysis as SA, attn_probe as AP
print("device:", DEVICE)
'''

NB38_RUN1 = r'''# --- 1. Predictions: outcome classes + which operation is dropped (table 5) ---------------------------------------
rp1 = SA.analyse(RESULTS_ROOT, MODE, SEED)
print("REVIEW FILE TO SEND BACK:", rp1)
if IN_COLAB and rp1:
    from google.colab import files
    files.download(str(rp1))
'''

NB38_RUN2 = r'''# --- 2. Attention probe (loads the LAST checkpoints) ---------------------------------------------------------------
rp2 = AP.probe(RESULTS_ROOT, MODE, SEED, device=DEVICE)
print("REVIEW FILE TO SEND BACK:", rp2)
if IN_COLAB and rp2:
    from google.colab import files
    files.download(str(rp2))
'''

NB39_INTRO = """# AR34 trained on k0234 at S = 4 / 5 / 6 — part __PART__ of 2 (__CONDS_TXT__), seed 42

**Question.** With the same training support k ∈ {0,2,3,4}, the original AR learned singletons at S = 4 (1.00) but
not at S = 7. AR34 (no pos(t), target coordinates, 10-way head) also fails at S7_k0234 (notebook 36a). Does the
success depend on the gap between the number of slots S and the longest training instruction (4)?

Hypothesis: when S = max k, executing slot by slot (each slot = one step, `<identity>` = no step) costs no more
than indexing the real operations as first / second / … / last, and the slot-by-slot rule generalizes. When S ≫
max k, the cheaper first/last indexing wins, and singletons are executed twice.
**Prediction:** singleton accuracy falls as S grows from 4 to 7.

**Setup.** Same model, data stream, 20,000 updates and sampled LAST-checkpoint evaluation as notebooks 23 / 34 / 36;
only S changes. S7_k0234 is already done (notebook 36a). The other half is notebook __OTHER__; run both at the same
time. The last cell (rerun after both finish) shows accuracy by S and the outcome analysis of notebook 37 for all four
S. **Send back `AR34_k0234_by_S_Review.ipynb`.**
"""

NB39_REPORT = r'''# --- Accuracy and outcome analysis for S = 4 / 5 / 6 / 7 (rerun after both halves finish) ----------------------------
from decompositionality import shortcut_analysis as SA
conds = LK.K0234_BY_S
df = LX.per_k_accuracy(RESULTS_ROOT, MODE, [SEED], conds, SUBDIR, RUN_MODELS)
ok = df[df.status == "available"]
print("Exact accuracy (rows: S, columns: test k; trained k = 0,2,3,4):")
display(ok.assign(S=ok.condition.str[1].astype(int)).pivot_table(index="S", columns="k", values="exact_acc").round(4))
rp = SA.analyse(RESULTS_ROOT, MODE, SEED, conditions=conds, models=RUN_MODELS,
                review_name="AR34_k0234_by_S_Review.ipynb", out_name=f"seed_{SEED}_ar34_k0234_by_S")
print("REVIEW FILE TO SEND BACK:", rp)
if IN_COLAB and rp:
    from google.colab import files
    files.download(str(rp))
'''

PARTS = {"a": ["S4_k0234", "S5_k0234"], "b": ["S6_k0234"]}


def _nb(cells, gpu):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], **({"gpuType": "A100"} if gpu else {})},
                   **({"accelerator": "GPU"} if gpu else {}),
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = cells
    nbf.validate(nb)
    return nb


def _write(name, nb):
    p = ROOT / "notebooks" / name
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nbf.write(nb, p)
    return p


def build():
    out = [_write("38_mechanism_analysis_first_last.ipynb", _nb(
        [nbf.v4.new_markdown_cell(NB38_INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(NB38_CONFIG),
         nbf.v4.new_code_cell(NB38_RUN1), nbf.v4.new_code_cell(NB38_RUN2)], gpu=True))]
    for part, conds in PARTS.items():
        other = "39b" if part == "a" else "39a"
        cfg = (TRAIN_CONFIG.replace("__CONDS__", repr(conds))
               .replace('RUN_MODELS = ["query_ar"]', 'RUN_MODELS = ["ar_slotpos_digithead_target"]'))
        assert '"ar_slotpos_digithead_target"' in cfg
        intro = NB39_INTRO.replace("__PART__", part).replace("__OTHER__", other).replace("__CONDS_TXT__", ", ".join(conds))
        name = f"39{part}_ar34_k0234_{'_'.join(c.split('_')[0] for c in conds)}_seed42.ipynb"
        out.append(_write(name, _nb([nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(BOOTSTRAP),
                                     nbf.v4.new_code_cell(cfg), nbf.v4.new_code_cell(DRIVER),
                                     nbf.v4.new_code_cell(NB39_REPORT)], gpu=True)))
    return out


if __name__ == "__main__":
    for p in build():
        print(p)
