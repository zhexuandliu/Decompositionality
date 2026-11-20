"""Generate notebooks/52_icl_api_k2_steps_models.ipynb: two-operation rules, "sequence of rigid transformations" prompt,
1/3/5/7/9 nested examples, answer vs. steps, three models."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "52_icl_api_k2_steps_models.ipynb"

INTRO = """# Two-operation rules, "sequence of rigid transformations" prompt — 3 models × 1/3/5/7/9 examples

**Prompt header** (notebooks 48-50 header plus one sentence):

> Each example below shows a 6x6 grid of digits (Input) and the grid obtained by applying the same hidden
> transformation to it (Output). The transformation is the same for every example. The hidden transformation is a
> sequence of one or more rigid transformations (distance-preserving maps) of the grid, where the grid wraps around
> at its edges.

No operation is named, and the number of steps is not given.

**Two experiments per task:**
* `answer`: give the transformed grid only.
* `steps`: "break the transformation down into the simplest steps it is made of", one `<step>` per step, then the
  final grid. No intermediate grids are requested.

**Tasks.** The 6 two-operation rules AB, AC, BA, BC, CA, CB (A = shift rows right, B = rotate 90° clockwise,
C = mirror left-right); 20 tasks each. Examples are **nested**: each task has 9 examples, and the run with n examples
uses the first n, with the same query grid. The first example alone already determines the rule uniquely.

**Models:** `openai/gpt-6-luna`, `openai/gpt-6.2-astra`, `meta/muse-glimmer-30b` (each with its default reasoning).
**Size:** 120 tasks × 5 example counts × 2 experiments × 3 models = **3,600 calls**.

Cell 4 = dry run (no cost). Cell 5 = pilot: 2 calls per model (checks model IDs and output format).
Cell 6 = full run (resumable). Cell 7 = tables, figure and zip.
"""

CONFIG = r'''# --- 3. Configuration ---------------------------------------------------------------------------------------------
MODELS = ["openai/gpt-6-luna", "openai/gpt-6.2-astra", "meta/muse-glimmer-30b"]
DEMO_COUNTS = [1, 3, 5, 7, 9]
N_PER_WORD = 20
TASK_SEED = 0
EXPERIMENTS = ["answer", "steps"]
STYLE = "steps"                  # header: sequence of rigid transformations, wrap-around grid
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
import matplotlib.pyplot as plt
from IPython.display import display
pd.set_option("display.max_rows", 300); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 250)
from decompositionality import icl_api as IA
_tag = "default" if not REASONING else "_".join(f"{k}{v}" for k, v in REASONING.items())
OUT_ROOT = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"k2_rigid_steps_n{N_PER_WORD}_seed{TASK_SEED}_{_tag}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
path = lambda m, n: OUT_ROOT / m.replace("/", "_") / f"demos{n}.jsonl"
print("results ->", OUT_ROOT)
'''

DRY = r'''# --- 4. Dry run: build the tasks and show both prompts (no API calls) -----------------------------------------------
tasks = IA.make_nested_tasks(IA.K2_WORDS, n_per_word=N_PER_WORD, max_demos=max(DEMO_COUNTS), seed=TASK_SEED)
(OUT_ROOT / "tasks.json").write_text(json.dumps(tasks))
print(len(tasks), "tasks")
for e in EXPERIMENTS:
    print(f"\n==================== experiment '{e}' (task {tasks[0]['task_id']}, 1 example) ====================")
    print(IA.build_prompt(IA.with_demos(tasks, 1)[0], e, hint=False, style=STYLE))
print("\ncalls planned:", len(tasks) * len(DEMO_COUNTS) * len(EXPERIMENTS) * len(MODELS))
'''

PILOT = r'''# --- 5. Pilot: 2 calls per model (3 examples) ------------------------------------------------------------------------
for m in MODELS:
    IA.run(IA.with_demos(tasks, 3)[:1], path(m, 3), m, experiments=EXPERIMENTS, hint=False, style=STYLE,
           reasoning=REASONING, max_tokens=MAX_TOKENS, workers=WORKERS)
    for line in path(m, 3).read_text().splitlines()[-2:]:
        r = json.loads(line)
        print(f"--- {m} / {r['experiment']}  error={r['error']}\n{(r['content'] or '')[:1500]}\n")
'''

FULL = r'''# --- 6. Full run (resumable: calls already saved are skipped) --------------------------------------------------------
for m in MODELS:
    for n in DEMO_COUNTS:
        print(f"=== {m}, {n} examples")
        IA.run(IA.with_demos(tasks, n), path(m, n), m, experiments=EXPERIMENTS, hint=False, style=STYLE,
               reasoning=REASONING, max_tokens=MAX_TOKENS, workers=WORKERS)
'''

REPORT = r'''# --- 7. Tables and figure --------------------------------------------------------------------------------------------
df = pd.concat([IA.score(IA.with_demos(tasks, n), path(m, n)).assign(model=m, n_demos=n)
                for m in MODELS for n in DEMO_COUNTS if path(m, n).exists()], ignore_index=True)
print("Accuracy by model, experiment and number of examples:")
display(df.pivot_table(index="n_demos", columns=["model", "experiment"], values="correct").round(3))
print("\nAccuracy by rule (all example counts):")
display(df.pivot_table(index="word", columns=["model", "experiment"], values="correct").round(2))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, e in zip(axes, EXPERIMENTS):
    for m in MODELS:
        q = df[(df.model == m) & (df.experiment == e)].groupby("n_demos").correct.mean()
        ax.plot(q.index, q.values, marker="o", lw=2, label=m)
    ax.set(title=e, xlabel="number of examples", xticks=DEMO_COUNTS, ylim=(0, 1.02)); ax.grid(alpha=.3)
axes[0].set_ylabel("accuracy"); axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT_ROOT / "accuracy.png", dpi=150); plt.show()
print("\nError types:")
display(df.groupby(["model", "experiment"]).error_type.value_counts(normalize=True).unstack(fill_value=0).round(3))
s = df[df.experiment == "steps"]
print("\nNumber of steps the model writes (rows: model, columns: number of steps):")
display(pd.crosstab([s.model, s.word], s.n_steps))
print("\nExample decompositions (3 examples, first 2 tasks per rule):")
display(s[s.n_demos == 3].groupby(["model", "word"]).head(2)[["model", "word", "correct", "n_steps", "steps_text"]])
print("\nUsage per call:")
display(df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]].apply(pd.to_numeric, errors="coerce")
        .groupby([df.model, df.experiment]).mean().round(4))
print("total cost by model ($):", df.assign(c=pd.to_numeric(df.cost_usd, errors="coerce")).groupby("model").c.sum().round(2).to_dict())
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
