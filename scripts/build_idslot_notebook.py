"""Generate notebooks/16_identity_slot_count_all_conditions_ar.ipynb: ONE notebook, eight sequential runs
(4 slot-count conditions x Transformer AR / Looped AR). Refuses to overwrite a notebook with saved outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOTSTRAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "16_identity_slot_count_all_conditions_ar.ipynb"

INTRO = """# Slot-count intervention in the identity format — all four conditions, eight runs, one Run all

**Question.** Five-slot identity training on k ∈ {0,2,3,4,5} generalized to singletons (notebook 10); five-slot
training on {0,2}, {0,2,3}, {0,2,3,4} did not (notebooks 12–14). Every training instruction in those three failing
supports contains at least one `<identity>` (I); train02345 also contains k=5 instructions with no I. This notebook
changes the number of physical slots S while keeping each logical training support fixed:

| condition | S | train k | no-I training records | held-out tests |
|---|---|---|---|---|
| slots2_train02 | 2 | 0, 2 | 640,000 (k=2) | k=1 at both positions |
| slots3_train023 | 3 | 0, 2, 3 | 426,666 (k=3) | k=1 at all 3 positions |
| slots4_train0234 | 4 | 0, 2, 3, 4 | 320,000 (k=4) | k=1 at all 4 positions |
| slots6_train02345 | 6 | 0, 2, 3, 4, 5 | 0 | k=1 at all 6 positions; all 729 six-operation words |

**Format.**
* Every instruction has exactly S VISIBLE tokens: the k A/B/C tokens fill a uniformly random ordered k-subset of the
  slots, and the rest are the learnable `<identity>` token. There is no PAD and no hidden slot.
* Each sequence has 77+S tokens with contiguous learned positions, and the shared position table has 83 rows.

**Recipe.** Notebooks 10/12–14 unchanged:
* Transformer AR and Looped AR, AdamW 3e-4, 5000 updates × 256, seed 42.
* Per-k allocation as in notebooks 12–14.
* For supports 02/023/0234, the logical (word, grid, target) stream and batch order are identical to the S=5 runs;
  only the layout changes. 02345 is a new stream, since notebook 10 used a different sampler.
* Common fresh initialization: the historical S=5 init with one added position row.
* Selection uses validation answer-digit CE (EOS excluded), macro-averaged over the trained k.

**Not a pure identity-token ablation.**
* Changing S also changes sequence length, downstream absolute positions, layouts, filler counts and compute.
* S=2/train02 keeps the even-count H ambiguity.
* Six-operation failure at S=6 confounds a larger k with the first-ever zero-I inputs.

One seed: descriptive only.

**Workflow.**
1. Run all: setup → shared audits → the sequential queue of eight runs (train → evaluate → diagnose → export each) →
   consolidated comparison.
2. If interrupted, Run all again: verified stages are skipped, training resumes from `state.pt`, and missing
   evaluation chunks are recomputed.
3. After any partial progress, the comparison cells can be rerun alone; they never train or build models.
"""

CONFIG = r'''# --- CONFIGURATION (the only cell to edit) --------------------------------------------------------------------
MODE = os.environ.get("DECOMP_MODE", "full")   # "full" | "smoke" (tiny CPU integration check, separate paths, NOT results)
SEED = int(os.environ.get("DECOMP_SEED", 42))   # no automatic seed sweep
RUN_CONDITIONS = ["slots2_train02", "slots3_train023", "slots4_train0234", "slots6_train02345"]   # default: all four
RUN_MODELS = ["transformer_ar", "looped_transformer_ar"]                                          # default: both
RUN_STAGES = ["train", "evaluate", "diagnose", "export"]                                          # default: every stage
USE_DRIVE = IN_COLAB
RESULTS_ROOT = os.environ.get("DECOMP_RESULTS_ROOT") or (
    "/content/drive/MyDrive/decompositionality_results" if USE_DRIVE
    else "/content/decompositionality_results" if IN_COLAB else str(REPO_DIR / "results"))
SLOT_ROOT = f"{RESULTS_ROOT}/identity_ar_slot_count"
DEVICE = os.environ.get("DECOMP_DEVICE", "auto")
DTYPE = "auto"
MICROBATCH = None          # None = probe the largest safe microbatch (effective batch stays 256)
EVAL_CHUNK_GRIDS = 100     # 10 resumable chunks per run
# -----------------------------------------------------------------------------------------------------------------
if USE_DRIVE and IN_COLAB and RESULTS_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
import json, pathlib
import pandas as pd
from IPython.display import Image, display
from decompositionality.config import make_experiment_config
from decompositionality import idslot as I, idslot_reports as R
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120)
FAMILY_ROOT = pathlib.Path(SLOT_ROOT) if MODE == "full" else pathlib.Path(SLOT_ROOT) / "smoke"
_c = make_experiment_config(RUN_CONDITIONS[0], MODE, SLOT_ROOT, RUN_MODELS, seed=SEED)
expected = dict(lr=3e-4, weight_decay=0.01, warmup_steps=250, min_lr=3e-5, grad_clip=1.0, val_every=250,
                optimizer_steps=5000, effective_batch_size=256)
print("resolved recipe:", _c.train.semantic_dict())
print("discrepancies from expected defaults:",
      {k: (getattr(_c.train, k), v) for k, v in expected.items() if getattr(_c.train, k) != v} or "none")
if MODE == "smoke":
    print("*** SMOKE MODE: tiny integration run under", FAMILY_ROOT, "- NOT benchmark results ***")
QUEUE = [(e, m) for e in R.CONDITIONS if e in RUN_CONDITIONS for m in R.MODELS if m in RUN_MODELS]
print(f"queue ({len(QUEUE)} runs, sequential):", " -> ".join(f"{e}/{m}" for e, m in QUEUE), "| stages:", RUN_STAGES)
print("results ->", FAMILY_ROOT)
'''

ENV = r'''from decompositionality.utils import environment_info
print(json.dumps(environment_info(REPO_DIR), indent=1))
'''

AUDIT = r'''# --- Shared audits (stop on failure): supports/counts, stream matching, layouts/masks/decoding, init, manifests -
AUD = FAMILY_ROOT / "audits" / f"seed_{SEED}"; AUD.mkdir(parents=True, exist_ok=True)
from decompositionality.data import load_or_build_manifests
man = load_or_build_manifests(FAMILY_ROOT / "manifests", SEED)
sup = R.support_audit(_c.train.effective_batch_size, _c.train.optimizer_steps)
stream, unif = R.stream_audit(SEED, _c.train.effective_batch_size, heldout=man.heldout_bytes)
lay = R.layout_audit(SEED)
ini = R.init_audit(SEED)
mb = R.microbatch_audit()
assert stream.deterministic_replay.all() and stream.only_training_k.all() and mb["equal_within_1e-6"]
assert (stream[stream.historical_S5_stream.str.startswith("idext")].same_words_grids_targets_order_as_S5 == True).all()
assert ini.identical_init_across_4_conditions.all() and ini.shared_tensors_equal_historical_S5_construction.all()
cfgs = {e: make_experiment_config(e, MODE, SLOT_ROOT, RUN_MODELS, seed=SEED) for e in R.CONDITIONS}
ctxs = {e: R.prepare_condition(c, str(REPO_DIR), {
            "five-slot count extrapolation (nb 12-14)": f"{RESULTS_ROOT}/identity_ar_count_extrapolation/manifests/manifest.json",
            "identity_02345 (nb 10)": f"{RESULTS_ROOT}/identity_varlen_02345_all_models/manifests/manifest.json"})
        for e, c in cfgs.items()}
tm = R.test_manifest_audit(ctxs)
for name, df in (("support", sup), ("stream_match", stream), ("placement_uniformity", unif), ("layout", lay),
                 ("initialization", ini), ("test_manifest", tm)):
    df.to_csv(AUD / f"{name}.csv", index=False); print(f"\n[{name}]"); display(df)
(AUD / "microbatch.json").write_text(json.dumps(mb, indent=1)); print("[microbatch]", mb)
print("training-transformation coverage:"); display(I.transformation_coverage())
del ctxs
'''

DRIVER = r'''# --- Sequential eight-run driver (one model/optimizer on the GPU at a time; resumable; skips verified stages) ---
status = R.run_queue(SLOT_ROOT, MODE, SEED, RUN_CONDITIONS, RUN_MODELS, RUN_STAGES, repo_dir=str(REPO_DIR),
                     compare_manifests={"five-slot count extrapolation (nb 12-14)":
                                        f"{RESULTS_ROOT}/identity_ar_count_extrapolation/manifests/manifest.json"},
                     chunk_grids=EVAL_CHUNK_GRIDS, device=DEVICE, dtype=DTYPE, microbatch=MICROBATCH)
display(status)
'''

COMPARE = r'''# --- Consolidated comparison (reads saved artifacts only; rerun alone at any time) -----------------------------
res = R.analyse(FAMILY_ROOT, RESULTS_ROOT, SEED)
FIG = res["out"] / "figures"
for p in R.figures(res, FIG):
    if str(p).endswith(".png"):
        display(Image(filename=str(p)))
print("(a) main table"); display(res["main"])
print("(c) accuracy by k"); display(res["by_k"])
print("(e) historical S=5 vs new S at the same support"); display(res["historical"]); display(res["historical_meta"])
display(res["historical_paired_seen"])
print("(f) six-operation per-word (S=6)"); display(res["six_op_per_word"].sort_values(["model", "exact_acc"]) if len(res["six_op_per_word"]) else "not yet available")
print("six-operation failure subsequence checks"); display(res["six_op_subsequence"])
print("(g) identity-count view"); display(res["identity_count_view"])
print("singleton consistency across placements"); display(res["singleton_consistency"])
print("(h) audits"); display(res["audit_initialization"]); display(res["timing"])
'''

STATUS = r'''# --- Export + eight-row stage status (a run is complete only when every stage is verified on disk) -------------
arch = R.export_comparisons(res)
st = R.status_table(FAMILY_ROOT, SEED); display(st)
sel = st[st.condition.isin(RUN_CONDITIONS) & st.model.isin(RUN_MODELS)]
print(f"{int(sel.complete.sum())}/{len(sel)} selected runs complete" +
      ("" if sel.complete.all() else " -> rerun Run all to resume the rest"))
'''


def build() -> Path:
    p = ROOT / "notebooks" / NAME
    if p.exists() and any(c.get("outputs") for c in nbf.read(p, as_version=4).cells if c.cell_type == "code"):
        raise SystemExit(f"{p.name} contains saved outputs; refusing to overwrite it")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(BOOTSTRAP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(ENV), nbf.v4.new_code_cell(AUDIT), nbf.v4.new_code_cell(DRIVER),
                nbf.v4.new_code_cell(COMPARE), nbf.v4.new_code_cell(STATUS)]
    nbf.validate(nb)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print(build())
