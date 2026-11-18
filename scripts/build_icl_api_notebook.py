"""Generate notebooks/46_icl_api_k2_gpt6_luna.ipynb: in-context inference of k = 2 grid transformations via OpenRouter."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "46_icl_api_k2_gpt6_luna.ipynb"

INTRO = """# In-context inference of grid transformations — k = 2, `openai/gpt-6-luna` via OpenRouter

The model sees 3 input/output pairs of 6×6 digit grids, all produced by the same hidden transformation, and must
apply it to a new grid. The transformations are the 6 ordered pairs of distinct operations: AB, AC, BA, BC, CA, CB
(A = shift every row right by one with wrap-around, B = rotate 90° clockwise, C = mirror left–right; left to right).
The prompt lists the three operations but never says which ones are used.

**Two experiments on the same tasks.** They differ only in the final instruction:
* `answer`: give the output grid;
* `ops`: first state the transformation as a sequence of the operations (`<ops>…</ops>`), then give the grid.

**Task construction.** The demos are checked so that exactly one of the 288 transformations fits them. The query
grid has 288 distinct images, so every answer identifies the transformation it corresponds to (or none).

**Before running.**
1. Create an OpenRouter API key (openrouter.ai → Keys; set a credit limit).
2. In Colab, open the key icon (Secrets) on the left and add a secret named `OPENROUTER_API_KEY` with the key as its
   value; allow this notebook to access it. Never paste the key into a cell.

**Order.** Cell 4 runs a dry run (no API calls, no cost). Cell 5 runs a 4-call pilot: check the responses and
the cost. Cell 6 runs everything (resumable: calls already made are skipped). Cell 7 scores and downloads the
results. A CPU runtime is enough.
"""

CONFIG = r'''# --- 3. Configuration ---------------------------------------------------------------------------------------------
MODEL = "openai/gpt-6-luna"
N_PER_WORD = 20                  # tasks per transformation (6 transformations -> 120 tasks, 240 calls for 2 experiments)
N_DEMOS = 3
TASK_SEED = 0
EXPERIMENTS = ["answer", "ops"]
REASONING = None                 # None = the model's default; e.g. {"effort": "high"} or {"max_tokens": 8000}
MAX_TOKENS = None                # cap on output tokens per call (None = provider default)
WORKERS = 4                      # parallel API calls
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE else str(REPO_DIR / "results"))
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
if IN_COLAB and not os.environ.get("OPENROUTER_API_KEY"):
    from google.colab import userdata
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")      # never printed
print("OpenRouter key present:", bool(os.environ.get("OPENROUTER_API_KEY")))
import json, pathlib
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 300); pd.set_option("display.width", 220)
from decompositionality import icl_api as IA
_tag = "default" if not REASONING else "_".join(f"{k}{v}" for k, v in REASONING.items())
OUT_DIR = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"k2_{MODEL.replace('/', '_')}_demos{N_DEMOS}_seed{TASK_SEED}_{_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "responses.jsonl"
print("results ->", OUT_DIR)
'''

DRY = r'''# --- 4. Dry run: build the tasks and show one prompt per experiment (no API calls) ----------------------------------
tasks = IA.make_tasks(IA.K2_WORDS, n_per_word=N_PER_WORD, n_demos=N_DEMOS, seed=TASK_SEED)
(OUT_DIR / "tasks.json").write_text(json.dumps(tasks))
print(len(tasks), "tasks:", {w: sum(t["word"] == w for t in tasks) for w in IA.K2_WORDS})
for e in EXPERIMENTS:
    print(f"\n==================== experiment '{e}' (task {tasks[0]['task_id']}) ====================")
    print(IA.build_prompt(tasks[0], e))
print("\nExpected answer for", tasks[0]["task_id"], ":")
print(IA._fmt(tasks[0]["expected"]))
'''

PILOT = r'''# --- 5. Pilot: 4 calls, then inspect the raw responses and the cost ------------------------------------------------
IA.run(tasks, OUT_JSONL, MODEL, experiments=EXPERIMENTS, reasoning=REASONING, max_tokens=MAX_TOKENS, workers=WORKERS,
       limit=4)
for line in OUT_JSONL.read_text().splitlines()[-4:]:
    r = json.loads(line)
    print(f"\n----- {r['task_id']} / {r['experiment']} | model {r['model']} | {r['seconds']} s | error: {r['error']}")
    print((r["content"] or "")[:1500])
    print("usage:", r["usage"])
display(IA.score(tasks, OUT_JSONL))
'''

FULL = r'''# --- 6. Full run (resumable: calls already in responses.jsonl are skipped) -------------------------------------------
IA.run(tasks, OUT_JSONL, MODEL, experiments=EXPERIMENTS, reasoning=REASONING, max_tokens=MAX_TOKENS, workers=WORKERS)
'''

REPORT = r'''# --- 7. Scores and summary --------------------------------------------------------------------------------------------
df = IA.score(tasks, OUT_JSONL)
S = IA.summarize(df)
for name, tab in S.items():
    print(f"\n### {name}")
    display(tab)
df.to_csv(OUT_DIR / "scores.csv", index=False)
for name, tab in S.items():
    tab.to_csv(OUT_DIR / f"summary_{name}.csv", index=False)
print("saved to", OUT_DIR)
if IN_COLAB:
    import shutil
    from google.colab import files
    shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
    files.download(str(OUT_DIR) + ".zip")
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(DRY), nbf.v4.new_code_cell(PILOT), nbf.v4.new_code_cell(FULL),
                nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
