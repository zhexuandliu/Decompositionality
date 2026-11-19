"""Generate notebooks/49_icl_api_klen_nohint_gpt6_luna.ipynb: accuracy vs number of operations (k = 1, 3), 20 tasks
per transformation (as for k = 2 in notebook 48), 3 demonstrations, neutral prompt."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "49_icl_api_klen_nohint_gpt6_luna.ipynb"

INTRO = """# In-context inference vs number of operations (k = 1, 3) — 3 demonstrations, neutral prompt

The prompt is the same as in notebook 48: **it says nothing about the operations**, only that the same hidden
transformation is applied in every example. There are two experiments, as before: `answer` (give the grid) and
`describe` (describe the rule in words first, then give the grid).

**Hidden transformations.** For each k, ALL transformations that need **exactly k operations**, i.e. group elements
whose shortest A/B/C word has length k. A word like CCA equals A, so it counts as k = 1, never as 3.
* k = 1: A, B, C (3 transformations)
* k = 3: 18 transformations (AAA, AAB, AAC, ABA, ABB, ABC, ACB, BAA, BAB, BAC, BBA, BBB, BBC, BCA, CAA, CAB, CAC, CBA)

**20 tasks per transformation**, the same as for k = 2 in notebook 48 (one replicate): 60 tasks for k = 1 and 360
for k = 3. Each task has 3 demonstrations, checked to determine the transformation uniquely. 420 tasks × 2
experiments = **840 calls** (~$0.5–1).

The report adds the 3-demo k = 2 results from notebook 48, if that folder is present on Drive.

Cell 4 = dry run (no cost). Cell 5 = 4-call pilot. Cell 6 = full run (resumable). Cell 7 = tables, plot and zip.
"""

CONFIG = r'''# --- 3. Configuration ---------------------------------------------------------------------------------------------
MODEL = "openai/gpt-6-luna"
KS = [1, 3]
N_PER_TRANSFORM = 20             # tasks per transformation (k = 2 in notebook 48 also has 20 per transformation)
N_DEMOS = 3
TASK_SEED = 0
HINT = False                     # never tell the model which operations exist
EXPERIMENTS = ["answer", "describe"]
REASONING = None
MAX_TOKENS = None
WORKERS = 16
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
import numpy as np
import pandas as pd
from IPython.display import display
pd.set_option("display.max_rows", 300); pd.set_option("display.width", 220)
from decompositionality import icl_api as IA
_tag = "default" if not REASONING else "_".join(f"{k}{v}" for k, v in REASONING.items())
OUT_DIR = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"klen13_nohint_{MODEL.replace('/', '_')}_demos{N_DEMOS}_n{N_PER_TRANSFORM}_seed{TASK_SEED}_{_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "responses.jsonl"
print("results ->", OUT_DIR)
'''

DRY = r'''# --- 4. Dry run: build the tasks and show a prompt (no API calls) ---------------------------------------------------
tasks = IA.make_tasks_by_length(KS, n_demos=N_DEMOS, seed=TASK_SEED, n_per_transform=N_PER_TRANSFORM)
(OUT_DIR / "tasks.json").write_text(json.dumps(tasks))
for k in KS:
    tk = [t for t in tasks if t["k"] == k]
    print(f"k = {k}: {len(tk)} tasks, {len({t['word'] for t in tk})} distinct transformations")
print("\nExample prompt (k = 3, experiment 'describe'):\n")
print(IA.build_prompt([t for t in tasks if t["k"] == 3][0], "describe", HINT))
print("\ncalls planned:", len(tasks) * len(EXPERIMENTS))
'''

PILOT = r'''# --- 5. Pilot: 4 calls (k = 5, both experiments) --------------------------------------------------------------------
pilot_tasks = [t for t in tasks if t["k"] == max(KS)][:2]
IA.run(pilot_tasks, OUT_JSONL, MODEL, experiments=EXPERIMENTS, hint=HINT, reasoning=REASONING,
       max_tokens=MAX_TOKENS, workers=WORKERS)
display(IA.score(tasks, OUT_JSONL))
'''

FULL = r'''# --- 6. Full run (resumable: calls already saved are skipped) --------------------------------------------------------
IA.run(tasks, OUT_JSONL, MODEL, experiments=EXPERIMENTS, hint=HINT, reasoning=REASONING, max_tokens=MAX_TOKENS,
       workers=WORKERS)
'''

REPORT = r'''# --- 7. Tables and plot --------------------------------------------------------------------------------------------------
df = IA.score(tasks, OUT_JSONL)
# k = 2 (3 demos) from notebook 48, if available
k2_dir = pathlib.Path(RESULTS_ROOT) / "icl_api" / f"k2_demo_sweep_nohint_{MODEL.replace('/', '_')}_{_tag}"
if (k2_dir / "tasks.json").exists():
    T48 = {int(r): t for r, t in json.loads((k2_dir / "tasks.json").read_text()).items()}
    k2 = IA.score_sweep(T48, k2_dir, [N_DEMOS], sorted(T48))
    if len(k2):
        df = pd.concat([df, k2.drop(columns=["replicate", "n_demos"]).assign(k=2)], ignore_index=True)
        print(f"added {len(k2)} k = 2 calls from notebook 48")
acc = df.pivot_table(index="k", columns="experiment", values="correct", aggfunc=["mean", "count"]).round(3)
print("Accuracy by number of operations k:"); display(acc)
print("\nError types (share of answers):")
display(df.groupby(["experiment", "k"]).error_type.value_counts(normalize=True).unstack(fill_value=0).round(3))
print("\nAccuracy by transformation (k = 1, 2, 3):")
display(df.pivot_table(index=["k", "word"], columns="experiment", values="correct", aggfunc="mean").round(2))
print("\nWrong legal transforms (true shortest word -> applied):")
w = df[df.error_type == "wrong transform"]
display(w.groupby(["k", "word", "applied"]).size().rename("count").reset_index().sort_values("count", ascending=False).head(25))
d = df[df.experiment == "describe"]
print("\nA few stated rules per k:")
for k_, g_ in d.groupby("k"):
    print(f"\nk = {k_} (accuracy {g_.correct.mean():.2f}):")
    for _, r_ in g_.head(5).iterrows():
        print(f"  [{r_.word}, {'correct' if r_.correct else 'wrong'}] {str(r_.rule)[:200]}")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4))
for e, c in (("answer", "#0072B2"), ("describe", "#D55E00")):
    q = df[df.experiment == e].groupby("k").correct.agg(["mean", "count"])
    se = np.sqrt(q["mean"] * (1 - q["mean"]) / q["count"])
    ax.errorbar(q.index, q["mean"], yerr=1.96 * se, marker="o", lw=2, capsize=3, color=c,
                label={"answer": "answer only", "describe": "describe the rule first"}[e])
ax.set(xlabel="number of operations k (shortest)", ylabel="exact accuracy", ylim=(0, 1.02), xticks=sorted(df.k.unique()))
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e6")
ax.legend(frameon=False); ax.set_title(f"{MODEL}, {N_DEMOS} demos, no hint (bars = 95% CI)", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT_DIR / "acc_vs_k.png", dpi=150); plt.show()
print("\nUsage:"); display(df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]]
                           .apply(pd.to_numeric, errors="coerce").sum().to_frame("total").T)
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
