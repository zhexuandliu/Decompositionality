"""Generate the two Colab notebooks from ONE template so they stay identical except for
the training distribution. Run:  python scripts/build_notebooks.py"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

TITLES = {
    "distinct_pairs": ("01", "Experiment 1 — `distinct_pairs`", "AB, AC, BA, BC, CA, CB"),
    "all_pairs": ("02", "Experiment 2 — `all_pairs`", "AA, AB, AC, BA, BB, BC, CA, CB, CC"),
}

INTRO = """# Decompositionality on 6×6 grids — {title}

**Research question.** Models are trained *only* on length-two operation strings and tested on
held-out grids with length-one and length-three strings (including repeated operations).
This is a controlled synthetic task inspired by grid reasoning — **not** an ARC reproduction.
All models are trained from random initialization on online-generated synthetic data only (seed 42).

**Operations** on a 6×6 grid of categorical digits 0–9 (array coordinates `r, c`):

| alias | atomic token | definition |
|---|---|---|
| A | `<translation>` | shift RIGHT by one column with wrap: `A(X)[r,c] = X[r,(c-1) mod 6]` |
| B | `<rotate>` | rotate 90° CLOCKWISE: `B(X)[r,c] = X[5-c, r]` |
| C | `<reflection>` | mirror LEFT-TO-RIGHT: `C(X)[r,c] = X[r, 5-c]` |

> **Strings execute LEFT TO RIGHT:** `AB` means *first A, then B*, i.e. `Y = B(A(X))`; `ABC` means `Y = C(B(A(X)))`.

**This notebook's training strings:** {strings} (uniform, deterministic balanced schedule; every training
string has length exactly two; 3 operation slots with random order-preserving PAD placement).
Budget: 5000 optimizer steps × effective batch 256 = 1,280,000 fresh examples per model (same budget in both
experiments, so `all_pairs` sees fewer examples per individual string).

**Test:** the same 1000 held-out base grids × all 39 strings (3 singles, 9 pairs, 27 triples).
**Nine models:** `transformer_ar`, `looped_transformer_ar`, `transformer_encoder`, `looped_transformer_encoder`,
`vit`, `looped_vit` (*Looped ViT (tied baseline)*, not the published hybrid LoopViT), `trm` (*TRM-adapted*, not an
ARC reproduction), `rescnn_film`, `unet_film`. Tied baselines have ≈1.8M unique parameters; all others ≈10M.

**How to run:** Runtime → *Change runtime type* → GPU, then *Run all*. Add a Colab Secret named `GITHUB_TOKEN`
(read access to the private repo) or paste a token when prompted (input hidden). With `USE_DRIVE=True` results
and checkpoints persist on Google Drive; if the runtime disconnects, just *Run all* again — completed models are
skipped after verification and interrupted ones resume from their last checkpoint (every 250 steps).
This notebook does not require the other experiment; the final comparison cell runs only if both result folders exist.
"""

BOOTSTRAP = r'''# --- 2. Bootstrap: locate or clone the PRIVATE repository (credentials are never printed or stored) ---
import base64, importlib.util, os, pathlib, subprocess, sys

REPO_URL = "https://github.com/zhexuandliu/Decompositionality.git"
REPO_BRANCH = "main"
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

def _find_repo():
    candidates = [pathlib.Path("/content/Decompositionality"), pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
    for p in candidates:
        if (p / "src" / "decompositionality").is_dir():
            return p.resolve()
    return None

REPO_DIR = _find_repo()
if REPO_DIR is None:
    token = None
    if IN_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = None
    if not token:
        import getpass
        token = getpass.getpass("GitHub token with read access to the private repo (input hidden): ")
    # Pass the credential through environment-scoped git config (not the URL, not the command line,
    # not persisted in .git/config).
    _hdr = "AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{token}".encode()).decode()
    _env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_CONFIG_COUNT="1",
                GIT_CONFIG_KEY_0="http.https://github.com/.extraheader", GIT_CONFIG_VALUE_0=_hdr)
    REPO_DIR = pathlib.Path("/content/Decompositionality" if IN_COLAB else "Decompositionality").resolve()
    _r = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)],
                        env=_env, capture_output=True, text=True)
    del token, _hdr, _env
    if _r.returncode != 0:
        raise RuntimeError("git clone failed (check the GITHUB_TOKEN secret / repo access):\n" + _r.stderr[-2000:])

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))
for _mod in ("numpy", "pandas", "matplotlib", "torch"):
    if importlib.util.find_spec(_mod) is None:  # Colab already ships these; never reinstall torch/CUDA here
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _mod], check=True)
REPO_COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Repository:", REPO_DIR, "| commit:", REPO_COMMIT or "unknown")
'''

CONFIG = r'''# --- 3. CONFIGURATION (edit here) ---------------------------------------------------------------
EXPERIMENT = "{experiment}"          # fixed for this notebook
MODE = os.environ.get("DECOMP_MODE", "full")   # "full" = benchmark | "smoke" = few updates + tiny eval sets (integration test only)
SEED = 42
MODELS = os.environ.get("DECOMP_MODELS", "transformer_ar,looped_transformer_ar,transformer_encoder,"
                        "looped_transformer_encoder,vit,looped_vit,trm,rescnn_film,unet_film").split(",")
USE_DRIVE = IN_COLAB                 # persist results/checkpoints on Google Drive (recommended in Colab)
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE
    else "/content/decompositionality_results" if IN_COLAB else str(REPO_DIR / "results"))
RESUME = True                        # continue interrupted runs; completed runs are verified and skipped
OPTIMIZER_STEPS = None               # None = mode default (full: 5000)
EFFECTIVE_BATCH = None               # None = mode default (full: 256)
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")   # "auto" -> CUDA if available
DTYPE = "auto"                       # "auto": bf16 if supported, else fp16 + GradScaler; "fp32" to disable AMP
MICROBATCH = None                    # None = probe a safe size with dummy data (effective batch is never reduced)
ACTIVATION_CHECKPOINTING = False     # optional memory saver
RUN_PROFILE = True                   # short dummy-data timing run per model (GPU only) for a time estimate
RUN_OVERFIT_DIAGNOSTIC = False       # optional debugging diagnostic (not part of the benchmark)
DOWNLOAD_ARCHIVE = False             # trigger a browser download of the results zip at the end (Colab)
# -----------------------------------------------------------------------------------------------------

if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

from decompositionality.config import make_experiment_config
_over = dict(seed=SEED, device=DEVICE, dtype=DTYPE, microbatch_size=MICROBATCH,
             activation_checkpointing=ACTIVATION_CHECKPOINTING)
if OPTIMIZER_STEPS: _over["optimizer_steps"] = OPTIMIZER_STEPS
if EFFECTIVE_BATCH: _over["effective_batch_size"] = EFFECTIVE_BATCH
cfg = make_experiment_config(EXPERIMENT, MODE, RESULTS_ROOT, MODELS, **_over)
cfg.resume = RESUME
if MODE == "smoke":
    print("*** SMOKE MODE: full architectures, a few updates, tiny eval sets. NOT benchmark results. ***")
print(f"experiment={{cfg.experiment}} mode={{cfg.mode}} steps={{cfg.train.optimizer_steps}} "
      f"effective_batch={{cfg.train.effective_batch_size}} seed={{cfg.train.seed}}")
print("train strings:", cfg.train_strings)
print("output directory:", cfg.experiment_dir.resolve())
'''

ENV = r'''# Environment record (saved with every run as well)
import json, torch
from decompositionality.utils import environment_info
ENV_INFO = environment_info(REPO_DIR)
print(json.dumps(ENV_INFO, indent=1))
'''

SAMPLES = r'''# --- 4. Sample inputs and their known transformation outputs (left-to-right execution) ---
import matplotlib.pyplot as plt, numpy as np
from decompositionality import operations as O
from decompositionality.reports import draw_grid
x0 = np.random.default_rng(0).integers(0, 10, (6, 6))
lab = np.arange(36).reshape(6, 6) % 10
show = ["A", "B", "C", "AB", "BA", "ABC"]
fig, axes = plt.subplots(2, len(show) + 1, figsize=(2.0 * (len(show) + 1), 4.6))
for row, g in enumerate([x0, lab]):
    draw_grid(axes[row, 0], g, "input X")
    for j, s in enumerate(show):
        draw_grid(axes[row, j + 1], O.apply_string(g, s), f"{s}: " + " then ".join(s))
fig.suptitle("Operation strings run LEFT TO RIGHT (AB = B(A(X)))"); plt.tight_layout(); plt.show()
'''

MANIFESTS = r'''# --- 5. Fixed split manifests + geometry / shape / data-stream checks ---
from decompositionality import pipeline
from decompositionality.data import TrainStream, grid_bytes
from decompositionality.diagnostics import geometry_checks
print(geometry_checks())
ctx = pipeline.prepare(cfg, repo_dir=str(REPO_DIR))
assert len(ctx.test_set["x"]) == cfg.n_test_grids * 39
_s = TrainStream(cfg.train.seed, cfg.train_strings, cfg.train.effective_batch_size, ctx.manifests.heldout_bytes)
_b = _s.batch(0)   # stateless: inspecting batch 0 does not consume anything
assert set(_b["string"]) <= set(cfg.train_strings) and not ({grid_bytes(g) for g in _b["x"]} & ctx.manifests.heldout_bytes)
print("batch 0 shapes:", {k: v.shape for k, v in _b.items()})
print("first op slots (3=PAD):", _b["ops"][:6].tolist(), "strings:", _b["string"][:6].tolist())
'''

COUNTS = r'''# --- 6. Full model parameter counts and a resource estimate ---
import pandas as pd
from decompositionality.models.registry import MODEL_IDS, model_summary
from decompositionality.training import profile_model
counts = pd.DataFrame([model_summary(m) for m in MODEL_IDS]).drop(columns=["config"])
display(counts[["model_id", "display_name", "unique_params", "shared_core_params", "untied_equivalent_params",
                "block_applications_per_forward", "core_calls_per_forward", "decoded_output_steps", "input_tokens"]])
if RUN_PROFILE and torch.cuda.is_available():
    prof = pd.DataFrame([profile_model(m, cfg.train, n_steps=2) for m in cfg.models])
    display(prof)
    print(f"Estimated total TRAINING time for {len(cfg.models)} models: {prof.est_train_minutes.sum():.0f} min "
          "(+ evaluation; AR free-running decoding of 39,000 test examples is the slowest). "
          "This is an estimate only; a free Colab session may not fit all runs — rerun to resume.")
else:
    print("Profiling skipped (no CUDA or RUN_PROFILE=False).")
'''

TRAIN = r'''# --- 7. Train all selected models sequentially (resumable; checkpoints every 250 steps) ---
from decompositionality.training import train_model
from decompositionality.utils import free_memory
train_failures = {}
for mid in cfg.models:
    print(f"\n=== training {mid} ===")
    try:
        train_model(mid, cfg, ctx.manifests, ctx.val_set, ctx.subset_set, repo_dir=str(REPO_DIR))
    except Exception as exc:
        import traceback; traceback.print_exc()
        train_failures[mid] = repr(exc)
    finally:
        free_memory()
if train_failures:
    raise RuntimeError(f"training failed for: {train_failures}")
'''

EVAL = r'''# --- 8. Evaluate the selected (lowest seen-pair validation CE) checkpoints on full val + test ---
# AR models: greedy free-running decoding from the prefix ending at <OUTPUT>; others: argmax (TRM: final refinement).
eval_failures = {}
for mid in cfg.models:
    try:
        pipeline.evaluate_model(mid, ctx)
    except Exception as exc:
        import traceback; traceback.print_exc()
        eval_failures[mid] = repr(exc)
    finally:
        free_memory()
if eval_failures:
    raise RuntimeError(f"evaluation failed for: {eval_failures}")
'''

REPORTS = r'''# --- 9. Figures, tables, per-example predictions and a compact results archive ---
from IPython.display import Image, Markdown
from decompositionality import reports
rep = reports.build_reports(cfg.experiment_dir, cfg.train_strings)
for name in ["a_training_curves", "b_group_exact_accuracy", "b_group_cell_accuracy", "c_per_string_exact_heatmap",
             "d_shared_examples", "d_failure_examples"]:
    p = cfg.experiment_dir / "figures" / f"{name}.png"
    if p.exists():
        display(Image(filename=str(p)))
display(pd.read_csv(cfg.experiment_dir / "tables" / "e_resource_table.csv"))
archive = pipeline.make_archive(cfg)
if DOWNLOAD_ARCHIVE and IN_COLAB:
    from google.colab import files
    files.download(str(archive))
'''

COMPARE = r'''# --- 10. Cross-experiment comparison (only if BOTH experiments' results exist under the same root) ---
from decompositionality.compare import compare
cmp = compare(cfg.experiment_dir.parent)
if cmp is not None:
    display(cmp["groups"])
    for name in ["compare_group_delta", "compare_per_string_delta_heatmap"]:
        display(Image(filename=str(cmp["out_dir"] / f"{name}.png")))
'''

OVERFIT = r'''# --- Optional: small-dataset OVERFIT diagnostic (debugging only; separate data stream and output folder) ---
# Trains each model on 64 fixed length-two examples to distinguish a broken pipeline from a generalization result.
if RUN_OVERFIT_DIAGNOSTIC:
    from decompositionality.diagnostics import overfit_diagnostic
    diag = [overfit_diagnostic(m, EXPERIMENT, n_examples=64, steps=300, out_dir=str(cfg.experiment_dir / "diagnostics"),
                               heldout_bytes=ctx.manifests.heldout_bytes) for m in cfg.models]
    display(pd.DataFrame([{"model": d["model_id"], **d["final"]} for d in diag]))
else:
    print("Overfit diagnostic disabled (set RUN_OVERFIT_DIAGNOSTIC=True in the configuration cell).")
'''

FOOTER = """## Where things are written
`<RESULTS_ROOT>/<experiment>/`: `manifests/` (fixed grids + hashes), `runs/<model>/` (`state.pt` = last resumable
checkpoint, `best.pt` = selected checkpoint, logs, `train_summary.json`), `eval/<model>/` (per-example predictions
`.npz` + tidy `.csv.gz`, per-string and group CSVs), `figures/` (PNG+PDF+CSV), `tables/`, and
`results_<experiment>.zip` (everything except checkpoints). The comparison is written to `<RESULTS_ROOT>/comparison/`.
Only one seed is used: no error bars or significance claims.
"""


def build(experiment: str) -> Path:
    num, title, strings = TITLES[experiment]
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    cells = [
        nbf.v4.new_markdown_cell(INTRO.format(title=title, strings=strings)),
        nbf.v4.new_code_cell(BOOTSTRAP),
        nbf.v4.new_code_cell(CONFIG.format(experiment=experiment)),
        nbf.v4.new_code_cell(ENV),
        nbf.v4.new_code_cell(SAMPLES),
        nbf.v4.new_code_cell(MANIFESTS),
        nbf.v4.new_code_cell(COUNTS),
        nbf.v4.new_code_cell(TRAIN),
        nbf.v4.new_code_cell(EVAL),
        nbf.v4.new_code_cell(REPORTS),
        nbf.v4.new_code_cell(COMPARE),
        nbf.v4.new_code_cell(OVERFIT),
        nbf.v4.new_markdown_cell(FOOTER),
    ]
    nb.cells = cells
    path = ROOT / "notebooks" / f"{num}_train_{experiment}.ipynb"
    nbf.validate(nb)
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    for e in TITLES:
        print(build(e))
