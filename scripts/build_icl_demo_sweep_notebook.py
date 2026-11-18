"""Generate notebooks/47_icl_api_k2_demo_sweep_gpt6_luna.ipynb: accuracy vs number of demonstrations."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "47_icl_api_k2_demo_sweep_gpt6_luna.ipynb"

INTRO = """# Does in-context accuracy improve with more demonstrations? — k = 2, `openai/gpt-6-luna`

Same task as notebook 46: the 6 transformations AB, AC, BA, BC, CA, CB, the same prompts, and the two experiments
(`answer` = give the grid; `ops` = first state the operation sequence, then the grid). The only change is the
**number of demonstrations: 1, 2, 3, 5, 8, 10**.

**Nested design.** Every task has 10 demonstrations; with n demos the prompt uses the first n. The query grid and
the first demos are therefore identical across n, so only "more demonstrations" changes. The first demo alone
already determines the transformation uniquely among the 288.

**Replicates.** 2 independent replicates (fresh random grids), each with 20 tasks per transformation (120 tasks).
In total: 6 demo counts × 2 experiments × 120 tasks × 2 replicates = **2,880 calls**. At notebook 46's cost
(~$0.0007 per call) this is roughly $2–4; longer prompts with more demos cost a little more.

Uses the same `OPENROUTER_API_KEY` Colab secret as notebook 46. Cell 4 = dry run (no cost). Cell 5 = 4-call pilot
with 10 demos (the longest prompts). Cell 6 = full run (resumable). Cell 7 = plots and tables (downloads a zip).
"""

CONFIG = r'''# --- 3. Configuration ---------------------------------------------------------------------------------------------
MODEL = "openai/gpt-6-luna"
DEMO_COUNTS = [1, 2, 3, 5, 8, 10]
REPLICATES = [0, 1]              # task seeds 100 + r: fresh grids per replicate
N_PER_WORD = 20
EXPERIMENTS = ["answer", "ops"]
REASONING = None                 # None = the model's default (as in notebook 46)
MAX_TOKENS = None
WORKERS = 8
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
OUT_DIR = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"k2_demo_sweep_{MODEL.replace('/', '_')}_{_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("results ->", OUT_DIR)
'''

DRY = r'''# --- 4. Dry run: build the nested tasks; show the prompt sizes (no API calls) ------------------------------------------
BASE = {r: IA.make_nested_tasks(IA.K2_WORDS, n_per_word=N_PER_WORD, max_demos=max(DEMO_COUNTS), seed=100 + r)
        for r in REPLICATES}
(OUT_DIR / "tasks.json").write_text(json.dumps({str(r): t for r, t in BASE.items()}))
for n in DEMO_COUNTS:
    p = IA.build_prompt(IA.with_demos(BASE[REPLICATES[0]], n)[0], "ops")
    print(f"{n:2d} demos: prompt of {len(p)} characters")
print("\nExample prompt with 2 demos, experiment 'ops':\n")
print(IA.build_prompt(IA.with_demos(BASE[REPLICATES[0]], 2)[0], "ops"))
PATHS = IA.sweep_paths(OUT_DIR, DEMO_COUNTS, REPLICATES)
print("\ncalls planned:", len(PATHS) * len(EXPERIMENTS) * len(BASE[REPLICATES[0]]))
'''

PILOT = r'''# --- 5. Pilot: 4 calls with 10 demos (the longest prompts) ----------------------------------------------------------
r0, nmax = REPLICATES[0], max(DEMO_COUNTS)
IA.run(IA.with_demos(BASE[r0], nmax), PATHS[(r0, nmax)], MODEL, experiments=EXPERIMENTS, reasoning=REASONING,
       max_tokens=MAX_TOKENS, workers=WORKERS, limit=4)
display(IA.score(IA.with_demos(BASE[r0], nmax), PATHS[(r0, nmax)]))
'''

FULL = r'''# --- 6. Full sweep (resumable: calls already saved are skipped) ------------------------------------------------------
for r in REPLICATES:
    for n in DEMO_COUNTS:
        print(f"\n=== replicate {r}, {n} demos ===")
        IA.run(IA.with_demos(BASE[r], n), PATHS[(r, n)], MODEL, experiments=EXPERIMENTS, reasoning=REASONING,
               max_tokens=MAX_TOKENS, workers=WORKERS)
'''

REPORT = r'''# --- 7. Plots and tables ----------------------------------------------------------------------------------------------
df = IA.score_sweep(BASE, OUT_DIR, DEMO_COUNTS, REPLICATES)
print("scored calls:", len(df))
acc = df.pivot_table(index="n_demos", columns="experiment", values="correct", aggfunc="mean").round(3)
print("\nAccuracy (mean over transformations and replicates):"); display(acc)
print("\nAccuracy by replicate:"); display(df.pivot_table(index=["n_demos", "replicate"], columns="experiment",
                                                          values="correct", aggfunc="mean").round(3))
print("\nAccuracy by transformation:"); display(df.pivot_table(index="word", columns=["experiment", "n_demos"],
                                                               values="correct", aggfunc="mean").round(2))
print("\nError types (share of answers):")
display(df.groupby(["experiment", "n_demos"]).error_type.value_counts(normalize=True).unstack(fill_value=0).round(3))
print("\nWrong legal transforms (what was applied instead):")
w = df[df.error_type == "wrong transform"]
display(w.groupby(["word", "applied"]).size().rename("count").reset_index().sort_values("count", ascending=False))
o = df[df.experiment == "ops"]
print("\n'ops' experiment: stated operations correct / literal / consistent with own answer, by demos:")
display(o.groupby("n_demos")[["ops_equivalent", "ops_literal", "ops_consistent_with_answer", "correct"]].mean().round(3))
print("\nUsage:"); display(df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]]
                           .apply(pd.to_numeric, errors="coerce").sum().to_frame("total").T)
figs = IA.plot_sweep(df, OUT_DIR)
df.to_csv(OUT_DIR / "scores.csv", index=False)
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
