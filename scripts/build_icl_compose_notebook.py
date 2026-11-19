"""Generate notebooks/51_icl_api_compose_k2_gpt6_luna.ipynb: in-context COMPOSITION test (rules shown separately)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "51_icl_api_compose_k2_gpt6_luna.ipynb"

INTRO = """# Can the model compose rules it learned in context? — experiment 1 (composition), `openai/gpt-6-luna`

**Question.** If the model sees two rules separately, can it apply their composition, which it has never been
shown? If it really treats the rules as reusable parts, composing should be about as accurate as seeing the
composite directly.

**Two conditions on the same query grid** (neutral prompts: nothing about which operations exist):
* `compose`: 3 examples of **Rule P** (the first operation), then 3 examples of **Rule Q** (the second operation),
  then "Apply Rule P and then Rule Q to this grid". The composite is never shown.
* `direct` (holistic baseline): 3 examples of the composite transformation itself, then "Apply the same
  transformation to this grid" (as in notebooks 48-50).

**Pairs.** The 6 ordered pairs of distinct operations (AB, AC, BA, BC, CA, CB; A = shift right, B = rotate 90°
clockwise, C = mirror left-right; hidden from the model). The pairs do not commute, so order matters, and the
composite never equals P alone, Q alone or the reversed order. Every set of 3 examples determines its rule uniquely
among the 288 transformations.

**Size.** 10 tasks per pair × 6 pairs × 2 conditions = **120 calls** (~$0.1-0.2).

**Key measure.** Composition gap = accuracy(`compose`) − accuracy(`direct`). Wrong `compose` answers are classified
by what was applied instead: P only, Q only, the reversed order (Q then P), or something else.

Cell 4 = dry run (no cost). Cell 5 = 4-call pilot. Cell 6 = full run (resumable). Cell 7 = tables and zip.
"""

CONFIG = r'''# --- 3. Configuration ---------------------------------------------------------------------------------------------
MODEL = "openai/gpt-6-luna"
N_PER_PAIR = 10
N_DEMOS = 3                      # examples per rule (and for the direct baseline)
TASK_SEED = 0
CONDITIONS = ["compose", "direct"]
REASONING = None                 # None = default (medium for this model)
MAX_TOKENS = None
WORKERS = 32
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
OUT_DIR = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"compose_k2_{MODEL.replace('/', '_')}_demos{N_DEMOS}_n{N_PER_PAIR}_seed{TASK_SEED}_{_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "responses.jsonl"
print("results ->", OUT_DIR)
'''

DRY = r'''# --- 4. Dry run: build the tasks and show both prompts (no API calls) -----------------------------------------------
tasks = IA.make_compose_tasks(IA.K2_WORDS, n_per_pair=N_PER_PAIR, n_demos=N_DEMOS, seed=TASK_SEED)
(OUT_DIR / "tasks.json").write_text(json.dumps(tasks))
print(len(tasks), "tasks")
for c in CONDITIONS:
    print(f"\n==================== condition '{c}' (task {tasks[0]['task_id']}) ====================")
    print(IA.build_compose_prompt(tasks[0], c))
print("\ncalls planned:", len(tasks) * len(CONDITIONS))
'''

PILOT = r'''# --- 5. Pilot: 4 calls --------------------------------------------------------------------------------------------------
IA.run(tasks[:2], OUT_JSONL, MODEL, experiments=CONDITIONS, hint=False, reasoning=REASONING, max_tokens=MAX_TOKENS,
       workers=WORKERS)
display(IA.score(tasks, OUT_JSONL))
'''

FULL = r'''# --- 6. Full run (resumable: calls already saved are skipped) --------------------------------------------------------
IA.run(tasks, OUT_JSONL, MODEL, experiments=CONDITIONS, hint=False, reasoning=REASONING, max_tokens=MAX_TOKENS,
       workers=WORKERS)
'''

REPORT = r'''# --- 7. Tables ---------------------------------------------------------------------------------------------------------
df = IA.score(tasks, OUT_JSONL)
acc = df.pivot_table(index="word", columns="experiment", values="correct", aggfunc="mean")
acc.loc["ALL"] = df.groupby("experiment").correct.mean()
acc["composition gap (compose - direct)"] = acc["compose"] - acc["direct"]
print("Accuracy by pair and condition:"); display(acc.round(3))
print("\nError types:"); display(df.groupby("experiment").error_type.value_counts(normalize=True).unstack(fill_value=0).round(3))
c = df[(df.experiment == "compose") & ~df.correct]
print(f"\nWrong 'compose' answers ({len(c)}): what was applied instead")
summary = {"P only": int(c.get("applied_is_p_only", pd.Series(dtype=bool)).fillna(False).sum()),
           "Q only": int(c.get("applied_is_q_only", pd.Series(dtype=bool)).fillna(False).sum()),
           "reversed order (Q then P)": int(c.get("applied_is_reversed", pd.Series(dtype=bool)).fillna(False).sum())}
summary["other"] = len(c) - sum(summary.values())
display(pd.Series(summary, name="count").to_frame())
display(c.groupby(["word", "applied"]).size().rename("count").reset_index().sort_values("count", ascending=False))
paired = df.pivot_table(index="task_id", columns="experiment", values="correct")
print("\nPaired outcome on the same query (rows: direct, columns: compose):")
display(pd.crosstab(paired["direct"], paired["compose"]))
print("\nUsage:"); display(df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]]
                           .apply(pd.to_numeric, errors="coerce").groupby(df.experiment).mean().round(1))
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
