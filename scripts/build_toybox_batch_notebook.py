"""Generate notebooks/56_toybox_batch1_astra.ipynb (55 = Muse, kept as run): GPT-6 Luna on 30 Toybox questions (rotation / translation / both)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "56_toybox_batch1_astra.ipynb"

INTRO = """# Toybox batch 1 — 60 questions, `openai/gpt-6.2-astra` (GPT-6 Luna and Muse Glimmer results already in the repo)

Same format and prompt as the single pilot question (notebook 53): 3 before/after demonstrations of 3 different
objects, 1 query image of a 4th object, 4 candidates, "Answer with exactly one letter".

| task | n | transformation | candidates |
|---|---|---|---|
| `R` rotation | 10 | 90° counter-clockwise, 90° clockwise or 180°, from recorded frames | 0 / 90 / 180 / 270° |
| `T` translation | 10 | left / right / up / down (crop-window shift = camera pan) | the 4 directions |
| `RT` rotation + translation | 10 | ±90° rotation (recorded) plus a shift | correct, rotation only, translation only, wrong rotation direction + shift |
| `M` mirror | 10 | left-right mirror (digital flip) about the object's own axis | correct, top-bottom mirror, rotate 180°, unchanged |
| `RM` rotation then mirror | 10 | ±90° rotation (recorded) then left-right mirror | correct, rotation only, mirror only, wrong order |
| `TM` translation + mirror | 10 | left-right mirror plus a shift | correct, mirror only, shift only, top-bottom mirror + shift |

The questions are already in the repository (`experiments/toybox_rotation_pilot/batch1/`). Answers live in
`batch1/answers.json` and are only read when scoring. One call per question. GPT-6 Luna and Muse Glimmer have already been run (results in `batch1/model_outputs/`).
This notebook calls `openai/gpt-6.2-astra` ($8 / $40 per 1M input / output tokens): cell 5 is a 2-question pilot that
prints the real token use and the projected cost of all 60; cell 6 runs the rest (estimated $5-15). Re-running
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
for qid in ["R00", "T00", "RT00", "M00", "RM00", "TM00"]:
    print(qid); display(_Img(filename=str(BATCH / qid / "composite.jpg"), width=900))
'''

CALL = r'''# --- 5. Pilot: 2 questions with Astra, then the projected cost of all 60 ----------------------------------------------
if IN_COLAB and not os.environ.get("OPENROUTER_API_KEY"):
    from google.colab import userdata
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")      # never printed
import json
MODEL = "openai/gpt-6.2-astra"
MAX_TOKENS = 48000                     # generous cap so long reasoning is not truncated
PRICE_IN, PRICE_OUT = 8 / 1e6, 40 / 1e6
def run(limit=None):
    cmd = [sys.executable, str(PILOT / "eval_batch.py"), "--batch", str(BATCH), "--model", MODEL, "--workers", "8",
           "--max-tokens", str(MAX_TOKENS)] + (["--limit", str(limit)] if limit else [])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PILOT))
    print(r.stdout[-6000:]); print(r.stderr[-3000:])
run(limit=2)
recs = [json.loads(l) for l in (BATCH / f"responses_{MODEL.replace('/', '_')}.jsonl").read_text().splitlines()]
ok = [r for r in recs if r.get("error") is None and r.get("usage")]
for r in ok:
    u = r["usage"]
    print(r["qid"], "answer:", repr(r["raw_response"]), "| prompt", u.get("prompt_tokens"), "completion", u.get("completion_tokens"),
          "| cost $", u.get("cost"), "| finish", r.get("finish_reason"))
if ok:
    pin = sum(r["usage"]["prompt_tokens"] for r in ok) / len(ok); pout = sum(r["usage"]["completion_tokens"] for r in ok) / len(ok)
    print(f"\\nprojected cost of 60 questions: ${60 * (pin * PRICE_IN + pout * PRICE_OUT):.2f}"
          f"  (mean {pin:.0f} input / {pout:.0f} output tokens per question)")
'''

FULL = r'''# --- 6. Full run (resumable: the pilot questions are skipped) --------------------------------------------------------
run()
'''

REPORT = r'''# --- 6. Results (all models with responses) and download ------------------------------------------------------------
import pandas as pd, shutil, glob
sys.path.insert(0, str(PILOT))
from eval_batch import score
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
frames, seen = [], set()
for f in sorted(glob.glob(str(BATCH / "responses_*.jsonl")) + glob.glob(str(BATCH / "model_outputs" / "responses_*.jsonl"))):
    m = pathlib.Path(f).stem.replace("responses_", "")
    if m in seen:                       # a model's fresh responses (batch1/) take precedence over the saved copy
        continue
    seen.add(m)
    d = score(BATCH, pathlib.Path(f)); d["model"] = m; frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["task"] = df.qid.str.rstrip("0123456789")
order = ["R", "T", "RT", "M", "RM", "TM"]
print("Accuracy by task and model:")
display(df.pivot_table(index="task", columns="model", values="correct").reindex(order).round(2))
print("\\nWhat was chosen when wrong:")
display(df[~df.correct].groupby(["model", "task", "chosen_kind"]).size().rename("n").reset_index())
print("\\nInvalid answers:", df.groupby("model").choice.apply(lambda s: int(s.isna().sum())).to_dict())
print("cost by model ($):", df.groupby("model").cost.sum().round(3).to_dict())
if IN_COLAB:
    from google.colab import files
    out = "/content/toybox_batch1_results"
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    for f in BATCH.glob("responses_*.jsonl"):
        shutil.copy(f, out)
    for f in BATCH.glob("scores_*.csv"):
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
                nbf.v4.new_code_cell(SHOW), nbf.v4.new_code_cell(CALL), nbf.v4.new_code_cell(FULL), nbf.v4.new_code_cell(REPORT)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
