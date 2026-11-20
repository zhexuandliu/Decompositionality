"""Generate notebooks/52_icl_api_k2_steps_models.ipynb: two-operation rules, prompt says "sequence of simple geometric
operations", the model writes each step with its grid; three models."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "52_icl_api_k2_steps_models.ipynb"

INTRO = """# Two-operation rules, "sequence of simple geometric operations" prompt — three models

**What changes from notebooks 48-50.** The header now also says:

> The hidden transformation is a sequence of one or more simple geometric operations on the grid. Each operation
> only moves digits to other cells; no digit is ever changed.

It does **not** say which operations, how many, or name any of them.

**Two experiments per task:**
* `answer`: give the transformed grid only.
* `steps`: break the transformation into its sequence of simple operations. For each step, give a short description
  (`<step>`) and the grid after all steps so far (`<grid>`), then the final answer.

Because every intermediate grid is given, each step can be checked exactly: we find which of the 288
transformations maps the previous grid to this one. This tells us how many steps the model uses, whether each step is
a legal transformation, and whether its decomposition matches ours (e.g. `A | B` for AB).

**Tasks.** The 6 two-operation rules AB, AC, BA, BC, CA, CB (A = shift rows right, B = rotate 90° clockwise,
C = mirror left-right); 20 tasks each, 3 examples; the same tasks for every model.

**Models:** `openai/gpt-6-luna`, `openai/gpt-6.2-astra`, `meta/muse-glimmer-30b` (provider-default reasoning).
**Size:** 120 tasks × 2 experiments × 3 models = **720 calls**.

Cell 4 = dry run (no cost). Cell 5 = pilot: 2 calls per model (checks the model IDs). Cell 6 = full run (resumable).
Cell 7 = tables and zip.
"""

CONFIG = r'''# --- 3. Configuration ---------------------------------------------------------------------------------------------
MODELS = ["openai/gpt-6-luna", "openai/gpt-6.2-astra", "meta/muse-glimmer-30b"]
N_PER_WORD = 20
N_DEMOS = 3
TASK_SEED = 0
EXPERIMENTS = ["answer", "steps"]
STYLE = "steps"
REASONING = None                 # None = each model's default
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
pd.set_option("display.max_rows", 300); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 200)
from decompositionality import icl_api as IA
_tag = "default" if not REASONING else "_".join(f"{k}{v}" for k, v in REASONING.items())
OUT_ROOT = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"k2_steps_demos{N_DEMOS}_n{N_PER_WORD}_seed{TASK_SEED}_{_tag}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
out = {m: OUT_ROOT / f"{m.replace('/', '_')}.jsonl" for m in MODELS}
print("results ->", OUT_ROOT)
'''

DRY = r'''# --- 4. Dry run: build the tasks and show both prompts (no API calls) -----------------------------------------------
tasks = IA.make_tasks(IA.K2_WORDS, n_per_word=N_PER_WORD, n_demos=N_DEMOS, seed=TASK_SEED)
(OUT_ROOT / "tasks.json").write_text(json.dumps(tasks))
print(len(tasks), "tasks")
for e in EXPERIMENTS:
    print(f"\n==================== experiment '{e}' (task {tasks[0]['task_id']}) ====================")
    print(IA.build_prompt(tasks[0], e, hint=False, style=STYLE))
print("\ncalls planned:", len(tasks) * len(EXPERIMENTS) * len(MODELS))
'''

PILOT = r'''# --- 5. Pilot: 2 calls per model ----------------------------------------------------------------------------------------
for m in MODELS:
    IA.run(tasks[:1], out[m], m, experiments=EXPERIMENTS, hint=False, style=STYLE, reasoning=REASONING,
           max_tokens=MAX_TOKENS, workers=WORKERS)
    for line in out[m].read_text().splitlines()[-2:]:
        r = json.loads(line)
        print(f"--- {m} / {r['experiment']}  error={r['error']}\n{(r['content'] or '')[:1200]}\n")
'''

FULL = r'''# --- 6. Full run (resumable: calls already saved are skipped) --------------------------------------------------------
for m in MODELS:
    IA.run(tasks, out[m], m, experiments=EXPERIMENTS, hint=False, style=STYLE, reasoning=REASONING,
           max_tokens=MAX_TOKENS, workers=WORKERS)
'''

REPORT = r'''# --- 7. Tables ---------------------------------------------------------------------------------------------------------
df = pd.concat([IA.score(tasks, out[m]).assign(model=m) for m in MODELS if out[m].exists()], ignore_index=True)
acc = df.pivot_table(index="word", columns=["model", "experiment"], values="correct", aggfunc="mean")
acc.loc["ALL"] = df.groupby(["model", "experiment"]).correct.mean()
print("Accuracy:"); display(acc.round(2))
print("\nError types:"); display(df.groupby(["model", "experiment"]).error_type.value_counts(normalize=True).unstack(fill_value=0).round(3))
s = df[df.experiment == "steps"].copy()
s["ours"] = s.step_words == s.word.map(lambda w: " | ".join(w))
print("\nSteps: how the model decomposes (fraction of tasks)")
display(s.groupby("model").agg(n=("task_id", "size"), correct=("correct", "mean"),
                               mean_steps=("n_steps", "mean"), all_steps_legal=("steps_all_legal", "mean"),
                               chain_correct=("steps_correct_chain", "mean"), same_as_ours=("ours", "mean")).round(2))
print("\nNumber of steps:"); display(pd.crosstab(s.model, s.n_steps))
print("\nMost common decompositions per rule (step words: what each step does, as a shortest A/B/C word; ? = not a legal transformation)")
display(s.groupby(["model", "word", "step_words"]).size().rename("count").reset_index()
         .sort_values(["model", "word", "count"], ascending=[True, True, False]).groupby(["model", "word"]).head(3))
print("\nExample step descriptions:")
display(s.groupby(["model", "word"]).head(1)[["model", "word", "correct", "step_words", "steps_text"]])
print("\nUsage:"); display(df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]]
                           .apply(pd.to_numeric, errors="coerce").groupby([df.model, df.experiment]).mean().round(4))
df.to_csv(OUT_ROOT / "scores.csv", index=False)
print("saved to", OUT_ROOT)
if IN_COLAB:
    import shutil
    from google.colab import files
    shutil.make_archive(str(OUT_ROOT), "zip", OUT_ROOT)
    files.download(str(OUT_ROOT) + ".zip")
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
