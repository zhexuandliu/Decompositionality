"""Generate notebooks/54_toybox_batch1.ipynb: GPT-6 Luna on 30 Toybox questions (rotation / translation / both)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "54_toybox_batch1.ipynb"

INTRO = """# Toybox batch 1 — 30 questions, `openai/gpt-6-luna`

Same format and prompt as the single pilot question (notebook 53): 3 before/after demonstrations of 3 different
objects, 1 query image of a 4th object, 4 candidates, "Answer with exactly one letter".

| task | n | transformation | candidates |
|---|---|---|---|
| `R` rotation | 10 | 90° counter-clockwise, 90° clockwise or 180°, from recorded frames | 0 / 90 / 180 / 270° |
| `T` translation | 10 | left / right / up / down (crop-window shift = camera pan) | the 4 directions |
| `RT` rotation + translation | 10 | ±90° rotation (recorded) plus a shift | correct, rotation only, translation only, wrong rotation direction + shift |

The questions are already in the repository (`experiments/toybox_rotation_pilot/batch1/`). Answers live in
`batch1/answers.json` and are only read when scoring. One call per question (30 calls, about 2-3 cents). Re-running
skips questions that already have a successful response.

Needs the Colab secrets `GITHUB_TOKEN` and `OPENROUTER_API_KEY` (never printed).
"""

PULL = r'''# --- 3. Update the repository (the question files were added recently) -------------------------------------------
import base64, subprocess
_tok = None
if IN_COLAB:
    from google.colab import userdata
    _tok = userdata.get("GITHUB_TOKEN")
_env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
if _tok:
    _env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                GIT_CONFIG_VALUE_0="AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{_tok}".encode()).decode())
_r = subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"], env=_env, capture_output=True, text=True)
del _tok, _env
print(_r.stdout[-500:] or _r.stderr[-500:])
PILOT = REPO_DIR / "experiments" / "toybox_rotation_pilot"
assert (PILOT / "batch1" / "answers.json").exists(), "batch files missing: pull failed?"
print("batch folder:", PILOT / "batch1")
'''

SHOW = r'''# --- 4. One example question (composites are for viewing only; they are not sent) -----------------------------
from IPython.display import Image as _Img, display
BATCH = PILOT / "batch1"
for qid in ["R00", "T00", "RT00"]:
    print(qid); display(_Img(filename=str(BATCH / qid / "composite.jpg"), width=900))
'''

CALL = r'''# --- 5. Calls: one per question (resumable) ------------------------------------------------------------------------
if IN_COLAB and not os.environ.get("OPENROUTER_API_KEY"):
    from google.colab import userdata
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")      # never printed
MODEL = "openai/gpt-6-luna"
r = subprocess.run([sys.executable, str(PILOT / "eval_batch.py"), "--batch", str(BATCH), "--model", MODEL, "--workers", "8"],
                   capture_output=True, text=True, cwd=str(PILOT))
print(r.stdout[-6000:]); print(r.stderr[-3000:])
'''

REPORT = r'''# --- 6. Results and download -------------------------------------------------------------------------------------
import pandas as pd, shutil
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
df = pd.read_csv(BATCH / f"scores_{MODEL.replace('/', '_')}.csv")
print(df.groupby("task").correct.agg(["mean", "sum", "size"]))
print("\nWhat was chosen when wrong:"); print(df[~df.correct].groupby(["task", "chosen_kind"]).size())
display(df[["qid", "task", "rule", "answer", "choice", "correct", "chosen_kind", "reasoning_tokens", "cost"]])
print("total cost $", round(pd.to_numeric(df.cost, errors="coerce").sum(), 4))
if IN_COLAB:
    from google.colab import files
    out = "/content/toybox_batch1_results"
    os.makedirs(out, exist_ok=True)
    for f in BATCH.glob("*_" + MODEL.replace("/", "_") + ".*"):
        shutil.copy(f, out)
    shutil.make_archive(out, "zip", out)
    files.download(out + ".zip")
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"}, "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(PULL),
                nbf.v4.new_code_cell(SHOW), nbf.v4.new_code_cell(CALL), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
