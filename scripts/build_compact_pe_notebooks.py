"""Generate notebooks/07_train_compact_02345_rope.ipynb and 08_train_compact_02345_sinusoidal.ipynb from ONE
template; they differ ONLY in the PE value. Notebooks 01-06 are not touched.
Run:  python scripts/build_compact_pe_notebooks.py"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = {"rope": ("07", "RoPE (rotary Q/K, theta 10,000, every attention call)"),
      "sinusoidal": ("08", "fixed sinusoidal (base 10,000, added once to the input embedding, scale 1)")}

INTRO = """# Compact left-aligned mixed-length training — sequence PE: **{pe_title}**

**Question.** Models are trained FROM SCRATCH only on final outputs of composite instructions (effective lengths
0, 2, 3, 4, 5; **never** a single operation) and tested on the never-trained single operations A, B, C.

**Training support (identical in notebooks 07 and 08; 361 words):** EMPTY; all 9 length-2 words (AA..CC); all 27
length-3, 81 length-4 and 243 length-5 words. Lengths are sampled 20% each and words are balanced within a length.
The budget is seed 42, 5000 updates x batch 256 = 1,280,000 examples per run. Every model and both PE conditions see
the SAME logical example sequence.

**Compact layout.** The j-th operation always occupies slot j; the remaining slots are PAD (masked as attention
keys). Grid and output tokens keep fixed offsets (AR: 82 tokens, `<OUTPUT>` at 44, answers 45..80, `<EOS>` 81).
There is no random placement in these runs.

**The ONLY difference between 07 and 08 is the 1D sequence positional encoding of the seven attention models.**
Both use the same deterministic layout position IDs and keep the same additive grid row/column (ViT: patch
row/column) and segment/type embeddings. Neither has a learned 1D position or slot table. This controls 1D sequence
PE conditional on the shared spatial encoding; it is not a comparison of 2D schemes.

**Shared CNN controls.** ResCNN-FiLM and U-Net-FiLM have no attention PE. They are trained once in `shared_cnn/` and
reused by whichever notebook runs second, but only when the configuration hashes match. They are reported once and
are n/a for the PE contrast.

**Test.** The selected checkpoint (lowest length-macro validation CE on lengths 0/2/3/4/5) is evaluated on all 364
words of length 0..5 x 1000 held-out grids, using compact left alignment. AR decoding is free-running. Lengths 2-5 are
seen strings on new grids; the singles are the only held-out strings.

**Limits.** One seed. Differences from notebooks 04/05 change BOTH the placement distribution and the PE, so they do
not isolate placement. Behavioural diagnostics are non-causal.
"""

CONFIG = r'''# --- CONFIGURATION ---------------------------------------------------------------------------------------
PE = "{pe}"                                   # 1D sequence PE for the attention models of THIS notebook
MODE = os.environ.get("DECOMP_MODE", "full")  # "full" = requested benchmark | "smoke" = tiny CPU integration run
SEED = 42
ATTN_MODELS = os.environ.get("DECOMP_ATTN_MODELS", "transformer_ar,looped_transformer_ar,transformer_encoder,"
                             "looped_transformer_encoder,vit,looped_vit,trm").split(",")
CNN_CONTROLS = os.environ.get("DECOMP_CNN_MODELS", "rescnn_film,unet_film").split(",")
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE
    else "/content/decompositionality_results" if IN_COLAB else str(REPO_DIR / "results"))
COMPACT_ROOT = f"{{RESULTS_ROOT}}/compact_02345_pe"   # NEW subtree (manifests/, rope/, sinusoidal/, shared_cnn/, comparisons/)
RESUME = True
OPTIMIZER_STEPS = None          # None = mode default (full: 5000)
EFFECTIVE_BATCH = None          # None = mode default (full: 256)
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
MICROBATCH = None
EVAL_CHUNK_GRIDS = 100
ANALYSIS_CHUNK_SIZE = 2000
USE_ANALYSIS_CACHE = True
RUN_PROFILE = True
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
from decompositionality.config import make_experiment_config
_over = dict(seed=SEED, device=DEVICE, dtype=DTYPE, microbatch_size=MICROBATCH)
if OPTIMIZER_STEPS: _over["optimizer_steps"] = OPTIMIZER_STEPS
if EFFECTIVE_BATCH: _over["effective_batch_size"] = EFFECTIVE_BATCH
cfg_pe = make_experiment_config(f"compact_02345_{{PE}}", MODE, COMPACT_ROOT, ATTN_MODELS, **_over)
cfg_cnn = make_experiment_config("compact_02345_cnn", MODE, COMPACT_ROOT, CNN_CONTROLS, **_over)
for c in (cfg_pe, cfg_cnn):
    c.resume = RESUME
if MODE == "smoke":
    print("*** SMOKE MODE: tiny integration run; NOT scientific results ***")
print(f"PE={{PE}} | mode={{MODE}} | steps={{cfg_pe.train.optimizer_steps}} | batch={{cfg_pe.train.effective_batch_size}} | "
      f"seed={{SEED}} | slots={{cfg_pe.n_slots}} | placement=compact_left")
print("attention runs ->", cfg_pe.experiment_dir, "| shared CNN controls ->", cfg_cnn.experiment_dir,
      "| shared manifests ->", cfg_pe.manifest_dir)
'''

ENV = r'''import json, torch
from decompositionality.utils import environment_info
ENV_INFO = environment_info(REPO_DIR)
print(json.dumps(ENV_INFO, indent=1))
'''

VERIFY = r'''# --- Validate supports, compact layout, offsets, PE wiring and manifests --------------------------------------
import dataclasses, hashlib
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
from decompositionality import mixed as M, tokenization as T
from decompositionality.mixed_pipeline import prepare_mixed
from decompositionality.models.registry import build_model, default_config
from decompositionality.models.common import count_parameters
sup = M.support(cfg_pe.experiment)
assert {L: len(w) for L, w in sup.items()} == {0: 1, 2: 9, 3: 27, 4: 81, 5: 243} and len(cfg_pe.train_strings) == 361
assert len(M.test_words()) == 364 and not any(len(w) == 1 for w in M.training_words(cfg_pe.experiment))
lay = T.ARLayout(5)
assert (lay.length, lay.idx_output, lay.out_start, lay.out_start + 35, lay.idx_eos) == (82, 44, 45, 80, 81)
_s = M.MixedTrainStream(SEED, cfg_pe.experiment, 256)
_o = M.MixedTrainStream(SEED, "compact_02345_sinusoidal" if PE == "rope" else "compact_02345_rope", 256)
for step in (0, 1, 2):
    a, b = _s.batch(step), _o.batch(step)
    assert all(np.array_equal(a[k], b[k]) for k in ("x", "y", "ops", "string", "length")), "PE conditions differ in data"
    for ops, L in zip(a["ops"], a["length"]):
        assert (ops[:L] != 3).all() and (ops[L:] == 3).all(), "not compact left-aligned"
print("compact left alignment + identical logical data stream across PE conditions: OK")
init_rows = []
for m in ATTN_MODELS:
    tensors = {}
    for pe in ("rope", "sinusoidal"):
        c = dataclasses.asdict(default_config(m)); c.update(n_slots=5, sequence_pe=pe)
        mod = build_model(m, c, seed=SEED)
        tensors[pe] = {k: v.detach().clone() for k, v in mod.named_parameters()}
        assert not any("pos_emb" in k or "slot_emb" in k for k in tensors[pe]), "learned 1D position table present"
    same = tensors["rope"].keys() == tensors["sinusoidal"].keys() and all(
        torch.equal(tensors["rope"][k], tensors["sinusoidal"][k]) for k in tensors["rope"])
    h = hashlib.sha256(b"".join(v.numpy().tobytes() for v in tensors[PE].values())).hexdigest()[:16]
    init_rows.append({"model": m, "unique_params": sum(v.numel() for v in tensors[PE].values()),
                      "identical_initial_trainable_tensors_rope_vs_sinusoidal": same, "init_sha256": h})
    assert same, m
init_df = pd.DataFrame(init_rows); display(init_df)
ctx_pe = prepare_mixed(cfg_pe, repo_dir=str(REPO_DIR), legacy_manifest=f"{RESULTS_ROOT}/distinct_pairs/manifests/manifest.json")
ctx_cnn = prepare_mixed(cfg_cnn, repo_dir=str(REPO_DIR), legacy_manifest=f"{RESULTS_ROOT}/distinct_pairs/manifests/manifest.json")
assert ctx_pe.manifests.manifest_hash == ctx_cnn.manifests.manifest_hash
cfg_pe.experiment_dir.mkdir(parents=True, exist_ok=True)
init_df.to_csv(cfg_pe.experiment_dir / "initialization_checksums.csv", index=False)
print("validation records per length:", pd.Series(ctx_pe.val_set["length"]).value_counts().sort_index().to_dict())
'''

COUNTS = r'''# --- Parameter counts / time estimate -------------------------------------------------------------------------
from decompositionality.mixed_reports import resource_table
from decompositionality.training import profile_model
display(resource_table(cfg_pe.experiment_dir, cfg_pe.experiment_dir / "tables", PE, ATTN_MODELS))
if RUN_PROFILE and torch.cuda.is_available():
    prof = pd.DataFrame([profile_model(m, cfg_pe.train, n_steps=2, n_slots=5, sequence_pe=PE) for m in ATTN_MODELS])
    display(prof)
'''

TRAIN = r'''# --- Train sequentially: 7 attention models (this PE), then the shared CNN controls (skipped if complete) ------
from decompositionality.training import train_model
from decompositionality.utils import free_memory
failures = {}
for cfg_, ctx_ in ((cfg_pe, ctx_pe), (cfg_cnn, ctx_cnn)):
    for mid in cfg_.models:
        print(f"\n=== training {cfg_.experiment}/{mid} ===")
        try:
            train_model(mid, cfg_, ctx_.manifests, ctx_.val_set, ctx_.subset_set, repo_dir=str(REPO_DIR))
        except Exception as exc:
            import traceback; traceback.print_exc(); failures[f"{cfg_.experiment}/{mid}"] = repr(exc)
        finally:
            free_memory()
if failures:
    raise RuntimeError(f"training failed: {failures}")
'''

EVAL = r'''# --- Evaluate the SELECTED checkpoints on all 364 words x 1000 test grids (chunked, restartable) --------------
from decompositionality.mixed_pipeline import evaluate_mixed_model
failures = {}
for cfg_, ctx_ in ((cfg_pe, ctx_pe), (cfg_cnn, ctx_cnn)):
    for mid in cfg_.models:
        try:
            evaluate_mixed_model(mid, ctx_, chunk_grids=EVAL_CHUNK_GRIDS)
        except Exception as exc:
            import traceback; traceback.print_exc(); failures[f"{cfg_.experiment}/{mid}"] = repr(exc)
        finally:
            free_memory()
if failures:
    raise RuntimeError(f"evaluation failed: {failures}")
'''

REPORTS = r'''# --- Training/evaluation figures and tables ---------------------------------------------------------------------
from decompositionality.mixed_reports import build_mixed_reports
from decompositionality.compact_reports import combined_table
for cfg_ in (cfg_pe, cfg_cnn):
    build_mixed_reports(cfg_.experiment_dir, cfg_.experiment)
    for name in ["a_training_curves", "a2_validation_by_length", "b1_singles_primary_and_empty",
                 "b2_accuracy_by_length_and_seen_status"]:
        p = cfg_.experiment_dir / "figures" / f"{name}.png"
        if p.exists():
            display(Image(filename=str(p)))
FAMILY_ROOT = cfg_pe.experiment_dir.parent      # COMPACT_ROOT (full) or COMPACT_ROOT/smoke (smoke)
display(combined_table(FAMILY_ROOT))
'''

DIAG = r'''# --- Diagnostics on saved predictions (this PE + shared CNNs + the other PE if available; read-only) -----------
from decompositionality.diagnosis.runner import DiagnosticsConfig, run_diagnostics
from decompositionality.diagnosis.spec import get_spec
other = "compact_02345_sinusoidal" if PE == "rope" else "compact_02345_rope"
dirs = {cfg_pe.experiment: str(cfg_pe.experiment_dir), cfg_cnn.experiment: str(cfg_cnn.experiment_dir)}
other_dir = cfg_pe.experiment_dir.parent / M.COMPACT_DIRS[other]
if (other_dir / "eval").exists():
    dirs[other] = str(other_dir); print("including the other PE condition (read-only):", other_dir)
else:
    print("other PE condition not available yet -> paired PE comparison will be N/A")
dcfg = DiagnosticsConfig(experiment_dirs=dirs, output_dir=str(cfg_pe.experiment_dir / "diagnostics"),
                         models=ATTN_MODELS + CNN_CONTROLS, chunk_size=ANALYSIS_CHUNK_SIZE, use_cache=USE_ANALYSIS_CACHE,
                         spec="compact", analysis_words=get_spec("compact").common_panel,
                         example_strings=["EMPTY", "A", "B", "C", "AB", "ABC"])
R = run_diagnostics(dcfg)
DFIG = pathlib.Path(R["output_dir"]) / "figures"
def show(prefix):
    for p in sorted(DFIG.glob(prefix + "*.png")):
        display(Image(filename=str(p)))
display(R["audit"][["experiment", "model", "n_records", "n_strings", "empty_word_records", "duplicate_keys",
                    "invalid_records", "target_oracle_mismatches", "input_manifest_mismatches", "decoding"]])
sing = R["per_string"][R["per_string"].group.isin(["singles", "EMPTY"])]
display(sing[["experiment", "model", "string", "exact_correct", "n", "cell_acc", "any_exact_group_match_count",
              "dominant_wrong_unique_word", "dominant_wrong_unique_share", "stable_rule_selected_word",
              "stable_rule_diag_exact_fidelity", "argmax_map_diag_cell_fidelity"]])
hs = R["hypothesis_summary"]
display(hs[hs.population.str.startswith("singletons")])
show("01_true"); show("04_dominant"); show("05_hypothesis"); show("13_exact"); show("08_stable"); show("09_composition")
'''

PAIRED = r'''# --- Paired RoPE minus sinusoidal on identical test examples (needs both notebooks; N/A otherwise) -------------
from decompositionality.compact_reports import paired_pe_comparison
pc = paired_pe_comparison(cfg_pe.experiment_dir.parent)
if pc is not None:
    display(pc[pc.word_or_group.isin(["A", "B", "C"]) | pc.word_or_group.str.startswith("[group]")])
    cmp_dir = cfg_pe.experiment_dir.parent / "comparisons"
    display(Image(filename=str(cmp_dir / "paired_rope_minus_sinusoidal.png")))
    show("11_paired")
'''

EXPORT = r'''# --- Export (no checkpoints, chunks or caches) --------------------------------------------------------------------
from decompositionality.mixed_reports import make_compact_archive
for cfg_ in (cfg_pe, cfg_cnn):
    make_compact_archive(cfg_.experiment_dir)
display(Markdown(pathlib.Path(R["report_path"]).read_text()))
'''


def build(pe: str) -> Path:
    num, title = NB[pe]
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(INTRO.format(pe_title=title)), code(BOOTSTRAP), code(CONFIG.format(pe=pe)), code(ENV), code(VERIFY),
                code(COUNTS), code(TRAIN), code(EVAL), code(REPORTS), code(DIAG), code(PAIRED), code(EXPORT),
                md("Outputs: `<RESULTS_ROOT>/compact_02345_pe/` -> `manifests/`, `rope/`, `sinusoidal/`, `shared_cnn/` "
                   "(each: `runs/`, `eval/`, `figures/`, `tables/`, `diagnostics/`, results zip) and `comparisons/`.")]
    p = ROOT / "notebooks" / f"{num}_train_compact_02345_{pe}.ipynb"
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    for pe in NB:
        print(build(pe))
