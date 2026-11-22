"""Generate notebooks/53_toybox_rotation_pilot.ipynb: ONE evaluation call of GPT-6 Luna on the Toybox question."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "53_toybox_rotation_pilot.ipynb"

INTRO = """# Toybox in-context rotation pilot — one question, ONE call to `openai/gpt-6-luna`

The question (11 recorded Toybox frames: 3 before/after demonstrations, 1 query, 4 candidates) is already built and
stored in the repository under `experiments/toybox_rotation_pilot/question/`. This notebook only:

1. makes sure the repository is up to date;
2. shows the composite (what the question looks like; the composite itself is **not** sent);
3. makes **exactly one** API call: the 11 images in order, each with a neutral label, then the prompt. The script
   refuses to call again once `results/response.json` exists, and a failed call is saved and not retried;
4. shows the raw response and the comparison with the intended answer (read from `metadata.json` only after the call);
5. downloads the results.

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
assert (PILOT / "question" / "inputs.json").exists(), "question files missing: pull failed?"
print("question folder:", PILOT / "question")
'''

SHOW = r'''# --- 4. Look at the question (composite = neutral labels only; not sent to the model) -------------------------------
from IPython.display import Image as _Img, display
display(_Img(filename=str(PILOT / "question" / "composite.jpg"), width=1100))
print((PILOT / "question" / "prompt.txt").read_text())
'''

CALL = r'''# --- 5. THE single evaluation call (refuses to run again once results/response.json exists) -------------------------
if IN_COLAB and not os.environ.get("OPENROUTER_API_KEY"):
    from google.colab import userdata
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")      # never printed
r = subprocess.run([sys.executable, str(PILOT / "eval_once.py")], capture_output=True, text=True, cwd=str(PILOT))
print(r.stdout[-4000:]); print(r.stderr[-2000:])
'''

REPORT = r'''# --- 6. Raw response and download ---------------------------------------------------------------------------------
import json, shutil
resp = json.loads((PILOT / "results" / "response.json").read_text())
print("model returned:", resp.get("model_returned"), "| finish:", resp.get("finish_reason"), "| error:", resp.get("error"))
print("raw response:", repr(resp.get("raw_response")))
print("usage:", resp.get("usage"))
print("\nsummary:", (PILOT / "results" / "result_summary.json").read_text())
if IN_COLAB:
    from google.colab import files
    shutil.make_archive("/content/toybox_pilot_results", "zip", PILOT / "results")
    files.download("/content/toybox_pilot_results.zip")
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
