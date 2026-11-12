# Decompositionality: systematic generalization of operation sequences on 6×6 grids

A controlled synthetic study of **systematic generalization** in an instruction-conditioned grid-transformation
task. Models learn **only length-two** operation strings and are tested on **length-one and length-three**
strings, including repeated operations, on **held-out grids**. Nine models are compared under two training
distributions (18 full training runs, one seed).

This task is *inspired by* grid reasoning. It is **not** a reproduction of the ARC benchmark: no ARC data, no
pretrained weights, no external data, no test-time training, no demonstrations at evaluation time, no
beam-search ensembles, no symbolic execution inside models and no oracle answer selection. Every model is
trained from random initialization on our synthetic data only.

> **Status:** all 18 full runs (9 models × 2 experiments) were executed in Colab (A100, BF16) at code commit
> `9e9fc06`. Their saved outputs are in notebooks 01/02, and their artifacts are on the runner's Google Drive (not in
> git). Notebook 03 analyzes those saved predictions. No result files are committed to this repository.

---

## 1. Task

The input `X` is a 6×6 array of independent, uniformly random **categorical** digits in {0,…,9}. Digits are
categories, not regression targets or images. **0 is a valid value, never padding.**

| alias | atomic token (model-facing ID) | definition (array coordinates `r, c ∈ {0..5}`) |
|---|---|---|
| **A** | `<translation>` (op 0, token 10) | shift the whole grid **right** by one column with wrap-around: `A(X)[r,c] = X[r,(c-1) mod 6]` |
| **B** | `<rotate>` (op 1, token 11) | rotate 90° **clockwise**: `B(X)[r,c] = X[5-c, r]` |
| **C** | `<reflection>` (op 2, token 12) | mirror **left-to-right** (reverse the column order): `C(X)[r,c] = X[r, 5-c]` |

### Strings execute LEFT TO RIGHT

`AB` means **first A, then B**: `Y = B(A(X))`. `ABC` means `Y = C(B(A(X)))`.

A/B/C are analysis aliases only. Models receive the three dedicated atomic operation IDs, with no embedding
for whole programs (`AB`, `ABC`), permutation classes, samples or tasks. These are discrete transformations of
a periodic grid. They are **not** the continuous SE(2) group, and they include a reflection.

A raw example contains only the three operation slots, `input_grid` and `target_grid`. The data generator
computes the target. Models never receive the target (except as the shifted teacher-forcing input of the AR
decoders during training), intermediate grids, transformation matrices, routing indices or lookup tables.

### Operation slots

Every model has **exactly three operation slots** and a distinct `PAD` ID. During training, each length-two
string uses one of the three order-preserving placements, chosen uniformly at random: `[o1,o2,PAD]`,
`[o1,PAD,o2]` or `[PAD,o1,o2]`. Every slot therefore sees real operations. Validation and test use canonical
right padding: `[A,PAD,PAD]`, `[A,B,PAD]`, `[A,B,C]`. PAD is ignored: Transformers mask PAD keys, and the CNNs
use a fixed zero PAD embedding. Grid positions never move. Random PAD placement reduces a trivial
untrained-position confound, but the length-generalization problem remains real.

## 2. Training distributions

| experiment | identifier | training strings |
|---|---|---|
| 1 | `distinct_pairs` | AB, AC, BA, BC, CA, CB |
| 2 | `all_pairs` | AA, AB, AC, BA, BB, BC, CA, CB, CC |

Every training string has length exactly two. There are no single operations, triples, identity examples or
intermediate targets. Strings follow a **deterministic balanced schedule**: each consecutive block of *k* global
samples is a random permutation of the *k* strings, so counts differ by at most one after any prefix.

**Both experiments use the same budget:** 5000 optimizer steps × effective batch 256 = **1,280,000 fresh
examples per run**. Experiment 2 therefore sees *fewer examples per individual string* (≈142k vs ≈213k). This
trade-off is deliberate.

## 3. Evaluation strings (all 39, on the same 1000 test grids)

| group | strings | count |
|---|---|---|
| `singles` | A, B, C | 3 |
| `pairs` | all 3² strings (seen/unseen reported separately) | 9 |
| `triples_all_distinct` | ABC, ACB, BAC, BCA, CAB, CBA | 6 |
| `triples_two_types` | exactly two distinct types (AAB, ABA, BAA, …) | 18 |
| `triples_one_type` | AAA, BBB, CCC | 3 |

The total is 39 strings, including all 27 triples. Strings that happen to implement the same transformation
are still reported separately. AA/BB/CC are *unseen* pairs in experiment 1 and *seen* pairs in experiment 2. An
empty group (unseen pairs in experiment 2) is reported as **N/A**, not 0.

## 4. Data, reproducibility and splits

* **Online data.** Fresh grids come from a stateless generator keyed by `(seed, logical batch index)`, with
  separate SeedSequence streams for grids, operation selection and PAD placement. Model initialization,
  validation, microbatch size, OOM retries, profiling, smoke checks and TRM refinement passes cannot change
  which examples are seen. All models in an experiment see the identical raw sample sequence. Both experiments
  see the identical input-grid sequence.
* **Fixed manifests.** Independent streams generate 300 validation and 1000 test base grids. They are unique,
  mutually disjoint, and excluded from training: a colliding training grid is deterministically resampled.
  They are saved with their generation config, seed, version, ordering and SHA-256 hashes in
  `<results>/<experiment>/manifests/`, and every reload is verified against regeneration.
  * Validation: 300 grids × the experiment's training strings (1800 or 2700 examples).
  * Fixed validation **subset**: the first 32 validation grids × seen strings, used for affordable
    free-running accuracy curves every 250 steps. It is labeled as a subset and never presented as the full
    validation set.
  * Test: 1000 grids × 39 strings = 39,000 examples per model.
* There is no geometric augmentation, since re-framing a grid would change what an operation string means.

## 5. Models

All models consume the same raw samples through model-specific input adapters and share one interface
(`training_loss`, `val_cell_ce`, `predict`). The registry is in `src/decompositionality/models/registry.py`.

| id | display name | unique learned params | notes |
|---|---|---:|---|
| `transformer_ar` | Transformer AR | 10,701,331 | decoder-only, causal, 6 pre-norm blocks, d=384, 6 heads, FFN 1536 (GELU), dropout 0 |
| `looped_transformer_ar` | Looped Transformer AR (tied) | 1,829,011 | one shared block × 6 (untied-equivalent 10,701,331; shared block 1,774,464) |
| `transformer_encoder` | Transformer encoder | 10,663,306 | bidirectional, 3 op slots + 36 cells, parallel per-cell head |
| `looped_transformer_encoder` | Looped Transformer encoder (tied) | 1,790,986 | one shared block × 6 (untied-equivalent 10,663,306) |
| `vit` | ViT (2×2 patches) | 10,684,456 | 9 patches (2×2×10 one-hot → linear), 3 op slots + 9 patches, 4×10 logits per patch |
| `looped_vit` | **Looped ViT (tied baseline)** | 1,812,136 | one shared block × 6 (untied-equivalent 10,684,456) |
| `trm` | **TRM-adapted** | 9,938,560 | 2-block attention core, d=640, 8 heads, SwiGLU 1728; n=2, T=2, S=2 |
| `rescnn_film` | ResCNN-FiLM | 10,006,826 | 6 residual blocks × 288 ch, circular padding, GroupNorm, FiLM per block |
| `unet_film` | U-Net-FiLM | 10,301,642 | 6×6 (192 ch, 2 blocks) → 3×3 (384 ch, 2 blocks) → 6×6 (192 ch, 2 blocks), skip connection |

**Parameter-count convention.** We count unique learned parameters (`requires_grad`, each shared tensor counted
once). The count includes embeddings, input adapters, conditioning networks, norms and output heads. The tied
baselines have ≈1.8M **unique** parameters while matching their untied counterpart's width and unrolled
depth. They are *not* 10M models, and tables report their untied-equivalent counts separately. Non-learned
buffers are not counted: TRM's fixed initial states and the layout/position index buffers. The CNNs' PAD
embedding row is fixed at zero (256 values in total, negligible). Exact counts are in `configs/models.json`
(`python -m decompositionality.cli counts`).

### Model details

* **`transformer_ar` / `looped_transformer_ar`.** An explicit tokenizer (`tokenization.py`) uses 10 digit IDs,
  3 operation IDs, PAD and 5 structural IDs. Row-major layout, 80 tokens:
  `<BOS> <OPS> s1 s2 s3 <GRID> x00..x55 <OUTPUT> y00..y55 <EOS>`. `<OUTPUT>` is at index 42, the answer
  digits are at 43–78 and `<EOS>` is at 79. The next-token shift is implemented once
  (`TransformerAR._teacher_forced`), and the first answer digit is predicted from the hidden state at
  `<OUTPUT>`. The loss covers only the 36 answer digits and EOS. Positions are learned absolute embeddings
  plus segment and row/column embeddings, all defined by the layout. Attention uses a causal mask plus
  PAD-key masking. **Evaluation uses greedy free-running decoding** from the prefix ending at `<OUTPUT>`: 36
  steps over the *full* vocabulary, plus a 37th EOS step as a separate format diagnostic. A non-digit token
  makes the grid invalid and incorrect. Teacher-forced answer-cell NLL and EOS NLL are logged separately and
  labeled as teacher-forced. KV caching is not used: the short prefix is recomputed. The looped variant shares
  one block across all 6 applications, with no step embedding and no hidden-state reset.
* **`transformer_encoder` / `looped_transformer_encoder`.** The input is `[B,39,d]`: ordered operation-slot
  embeddings, row/column embeddings and token-type embeddings. PAD keys are masked and there is no causal
  mask. A shared linear classifier on the 36 final grid states produces `[B,36,10]`, trained with mean
  cell-wise CE.
* **`vit` / `looped_vit`.** Each 2×2 patch's four one-hot cells are flattened in within-patch order (40 values)
  and linearly projected. There is no averaging, no rendering and no resizing. The sequence is 3 op slots + 9
  patches = 12 positions. Unpatchify is the exact inverse of patchify (tested). `looped_vit` is **our
  controlled weight-tied ViT**, *not* the published hybrid LoopViT: it has no ConvGLU, no local convolution,
  no test-time training and no entropy-based stopping.
* **`trm` (TRM-adapted).** This is an adaptation of the Tiny Recursive Model
  ([paper](https://arxiv.org/abs/2510.04871), [official code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels),
  MIT license). We re-implemented the update structure of the official `trm.py` in plain PyTorch; no code was
  copied verbatim.
  * **States and network.** The model keeps an answer state `y` (official `z_H`), a latent state `z`
    (official `z_L`) and the input embedding `x`. One shared network (input injection by addition, then 2
    post-norm blocks: attention→RMSNorm, SwiGLU→RMSNorm) performs `z ← net(z, y + x)` n times, then
    `y ← net(y, z)`.
  * **Refinement schedule.** Each supervised refinement runs T−1 cycles without gradients, then one cycle with
    gradients, decodes `y` with the output head, and then detaches `(y, z)`. `y` and `z` start from fixed
    seeded buffers for every batch.
  * **Adaptations.** Conditioning is instruction-based: 3 op slots + 36 cells, PAD-key masking, learned
    slot/row/column positions, and no puzzle embedding. Scale is d=640. Compute is fixed: n=2, T=2, S=2 at
    training *and* evaluation, with **no adaptive halting/Q head**. Optimizer scheduling: all S refinements
    are compared with the **same final target**, their losses are averaged, and **one optimizer step per
    logical batch** is taken. The official code instead steps after every supervision step and refills halted
    slots. Refinements are not counted as samples.
  * **Compute.** S·T·(n+1) = **12 shared-core calls = 24 block applications per forward example**, of which 6
    calls (12 block applications) carry gradients. The count follows the official convention: `H_cycles ×
    (L_cycles + 1)` `L_level` calls per supervision step. TRM is **not compute-matched** to the 6-block
    baselines. Do not compare it with the paper's ARC scores.
* **`rescnn_film` / `unet_film`.** Inputs are 10 one-hot channels plus normalized row/column coordinates
  (CoordConv), giving `[B,12,6,6]`. Coordinates are required because a circular translation-equivariant CNN
  with spatially constant conditioning cannot implement fixed-center rotation or reflection. Convolutions use
  circular padding and GroupNorm. The 3 ordered op embeddings are **concatenated** (never averaged; PAD is a
  zero vector) → MLP → per-block FiLM `(1 + Δγ)·h + β`, zero-initialized as the identity. The ResCNN keeps
  6×6 resolution throughout with no pooling. The U-Net uses one strided-conv reduction 6×6→3×3, then nearest
  upsampling plus a conv back to 6×6, with a skip connection fused by a 1×1 conv. Both end in a 1×1
  classifier to `[B,10,6,6]` with cell-wise CE.

### What the comparison means

Models are compared **under their own prediction/training schemes**. Autoregressive and parallel models differ
in attention visibility and output factorization. TRM adds iterative deep supervision. Equal parameter and
exposure budgets **do not imply equal training or inference FLOPs**: see the resource table. Training losses of
different factorizations are *not* the same objective, and the plots label them accordingly. Within each
untied/tied pair, everything except weight sharing is matched.

## 6. Training and checkpoint selection

* AdamW (lr 3e-4, β=(0.9, 0.95), weight decay 0.01 on matrices/kernels only), 250 linear warmup steps, then
  cosine decay to 3e-5. Gradients are clipped at 1.0. One LR-schedule step per optimizer update.
* Seed 42, 5000 steps, effective batch 256. The **whole budget always runs**, with no early stopping.
* CUDA mixed precision uses BF16 if supported, otherwise FP16 with a GradScaler (unscaled before clipping).
  CPU runs in FP32. Activation checkpointing is optional. There are no Triton, FlashAttention, W&B or
  multi-GPU dependencies.
* The microbatch size is probed with **dummy data** before training. Gradients accumulate to exactly 256 with
  per-example-correct weighting. On OOM, the **same logical batch** is retried with a smaller microbatch. The
  effective batch and sample budget are never reduced.
* Validation runs every 250 steps and at the final step:
  * full validation **seen-pair cell CE**. For AR models this is **teacher-forced answer-cell CE (EOS
    excluded)**; for TRM it is the final refinement.
  * free-running/argmax exact and cell accuracy on the fixed 32-grid subset.
* **Checkpoint selection:** lowest validation cell CE, with ties going to the earlier checkpoint. The last
  checkpoint is also kept (`state.pt`). Test strings never influence selection, and test evaluation runs only
  after training.
* Resumable checkpoints are saved atomically. They contain model, optimizer, scheduler, scaler, step, RNG
  states, data-stream position, best score, config/model hashes, data version, manifest hash and logs. On
  resume, the hashes are validated and the identical next logical batch is reproduced (tested, including
  with a different microbatch size after resume). Completed runs are skipped only after verifying a matching
  config hash and the existing artifacts.
* Models whose **seen-pair** test accuracy is below 0.90 are flagged *INADEQUATE* in the summary. Their poor
  out-of-distribution accuracy must not be read as a compositionality finding.

## 7. Outputs

`<RESULTS_ROOT>/<experiment>/` contains:

| path | contents |
|---|---|
| `manifests/` | `heldout_grids.npz`, `manifest.json` (hashes, ordering, generation config) |
| `runs/<model>/` | `state.pt` (last, resumable), `best.pt` (selected), `train_log.csv`, `val_log.csv`, `run_config.json` (includes repo commit and package versions), `train_summary.json` |
| `eval/<model>/` | `{val,test}_predictions.npz`, `{val,test}_examples.csv.gz`, `*_per_string.csv`, `*_groups.csv`, `eval_summary.json` |
| `figures/` | (a) training curves; (b) group exact/cell accuracy bars; (c) 9×39 per-string heatmap; (d) shared and failure examples. Each figure is saved as PNG + PDF + CSV of the plotted numbers |
| `tables/` | (e) `e_resource_table.{csv,md}`; (f) `f_numerical_summary.{csv,md}`; all-model per-string/group CSVs |
| `results_<experiment>.zip` | compact archive of everything except checkpoints |

The per-example records contain prediction, target, input, grid ID, operation string, experiment, model, seed,
validity, cell-correct count and exact-match status. **Exact-grid accuracy** (primary metric) requires all 36
cells to be valid and correct. Secondary metrics are cell accuracy and final-output cell CE; the AR cell CE is
the teacher-forced diagnostic and is labeled as such. Only one seed is used, so there are no error bars or
significance claims.

The cross-experiment comparison (`<RESULTS_ROOT>/comparison/`) joins both experiments on
model/string/grid ID. It verifies identical manifests, inputs and targets, then reports `all_pairs −
distinct_pairs` exact-accuracy changes per string and per group.

## 8. Running in Google Colab

1. Open `notebooks/01_train_distinct_pairs.ipynb` or `notebooks/02_train_all_pairs.ipynb` in Colab:
   *File → Open notebook → GitHub* (authorize private repos), or upload the file.
2. *Runtime → Change runtime type → GPU*.
3. **Authentication for the private repository.** Add a Colab Secret named `GITHUB_TOKEN` (key icon in the
   left sidebar) holding a fine-grained token with *Contents: read* on this repo, and enable notebook access.
   Without the secret, the bootstrap cell asks for the token through a hidden prompt. The token is passed to
   `git clone` through environment-scoped git config. It never appears in a URL, the command line, notebook
   output or `.git/config`.
4. **Run all.** By default `USE_DRIVE=True`: Google Drive is mounted and everything is written to
   `/content/drive/MyDrive/decompositionality_results/<experiment>/`. With `USE_DRIVE=False`, output goes to
   `/content/decompositionality_results/`, which is lost when the runtime is recycled.
5. **Resume.** If the session ends, reconnect and *Run all* again with `RESUME=True` (the default). Finished
   models are verified and skipped. An interrupted model continues from its last checkpoint (every 250 steps)
   with the identical next batch.
6. The two notebooks are independent. Run them with the same `RESULTS_ROOT`, and the final cell of either
   notebook produces the cross-experiment comparison once both result folders exist. Otherwise it reports
   which folder is missing and finishes successfully.

**Smoke mode:** set `MODE = "smoke"` in the configuration cell. It keeps the **full architectures** but runs 6
updates of batch 16 and evaluates on 8 validation / 4 test grids. It writes to `<RESULTS_ROOT>/smoke/`. Smoke
mode is for integration testing only and never for reporting.

**Runtime expectations.** The configuration cell's profiling step times each model on dummy data and prints an
estimate. Rough expectations on a T4: the TRM (24 block applications per forward pass, 12 of them in
no-grad cycles) and the AR decoders are the slowest to train. AR free-running evaluation of 39,000 test
examples requires 37 decoder passes each. Nine models may **not** fit in one free Colab session; the pipeline
is resumable by design. An A100/L4 runtime with BF16 is substantially faster.

## 9. Command line

```bash
pip install -e .            # or: export PYTHONPATH=src
python -m decompositionality.cli counts                                   # parameter counts
python -m decompositionality.cli smoke --experiment distinct_pairs        # tiny end-to-end run, full architectures
python -m decompositionality.cli train --experiment all_pairs --model trm --output-root results   # full single-model run (+eval)
python -m decompositionality.cli evaluate --experiment all_pairs --model trm --output-root results
python -m decompositionality.cli run --experiment distinct_pairs --output-root results          # all nine + reports
python -m decompositionality.cli plot --experiment all_pairs --output-root results
python -m decompositionality.cli compare --results-root results
python -m decompositionality.cli overfit --model transformer_encoder --steps 300   # debugging diagnostic
```

Wrapper scripts live in `scripts/`. `scripts/build_notebooks.py` regenerates notebooks 01/02 from one template. Doing so would erase their saved Colab training outputs, so don't rerun it casually. `scripts/build_analysis_notebook.py` builds notebook 03 only.

## 10. Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The tests cover:

* **Operations:** manual coordinate examples, periodic wrap, rotation/reflection directions, A⁶=B⁴=C²=I, an
  AB≠BA witness, and left-to-right composition against an independently written reference.
* **String sets and splits:** training sets, near-balanced schedule, the 39 strings and their group counts,
  held-out exclusion and manifests.
* **Data stream:** determinism, identical grids across experiments, and stateless resume access.
* **Tokenization and AR model:** PAD ≠ digit 0 and the exact AR layout; causal masking (future tokens cannot
  change earlier logits); AR loss alignment including the first answer token and EOS; generation that never
  uses the target and matches a manual greedy loop; invalid AR tokens.
* **Transformer, ViT and CNN components:** PAD-key leakage, operation-order sensitivity, patchify/unpatchify
  round-trip and cell order, and tied/untied forward equivalence plus parameter-sharing counts.
* **TRM:** state reset, detachment boundaries (counted with backward hooks), differentiable heads, and exactly
  12 core calls / 24 block applications / 6 gradient-bearing calls.
* **All nine full-size models:** forward/backward with finite gradients, built one at a time.
* **Training and metrics:** gradient-accumulation invariance, the LR schedule, metrics with invalid outputs and
  empty groups, and the cross-experiment join.
* **End to end:** train → interrupt → resume (bitwise-close to an uninterrupted run, even with a different
  microbatch) → evaluate → plots → compare.

## 11. Post-hoc diagnostics: what do the models output on unseen instructions?

**Notebook:** [`notebooks/03_analyze_generalization_failures.ipynb`](notebooks/03_analyze_generalization_failures.ipynb).
It runs one *Run all* workflow over **both** experiments and all discovered models. It is **analysis only**: it never
trains, fine-tunes or re-selects checkpoints, and by default it does not load model weights or need a GPU. The
implementation is in `src/decompositionality/diagnosis/`. The CLI is
`python -m decompositionality.diagnosis --results-root <RESULTS_ROOT>`.

**Required artifacts** (written by notebooks 01/02), under `<RESULTS_ROOT>/<experiment>/`:

* `eval/<model>/test_predictions.npz` and `eval_summary.json`, the raw per-example test predictions. For AR models
  these come from free-running decoding, and `-1` marks a non-digit token.
* `manifests/heldout_grids.npz`, the fixed base grids.
* Optionally `runs/<model>/best.pt` and `run_config.json`, used only for provenance hashes and opt-in regeneration.

The default `RESULTS_ROOT` is `/content/drive/MyDrive/decompositionality_results`, as used by notebooks 01/02.

**Running in Colab.** Open notebook 03 and authenticate as for notebooks 01/02 (`GITHUB_TOKEN` secret). Check
`RESULTS_ROOT` in the configuration cell, then *Run all*. A CPU runtime is sufficient. Outputs are written to
`<RESULTS_ROOT>/diagnostics_generalization_failures/`:

* `tables/*.csv`
* `figures/*.{png,pdf,csv}`
* `per_example/*.csv.gz` (compact per-record diagnostics)
* `maps/*.npz` (all fitted source maps)
* `REPORT.md`
* `provenance.json` (file hashes, selected checkpoint steps, evaluation code commit, analysis code commit)
* `cache/`
* a compact `diagnostics_generalization_failures_export.zip`

Source predictions and checkpoints are opened read-only.

**Missing artifacts.** An artifact inventory is shown before the analysis. An experiment or model without
predictions is listed as *missing* and never reported as 0. With `RUN_FROZEN_INFERENCE_IF_MISSING=True` (off by
default), missing predictions are regenerated from the already selected `best.pt` and the original manifest. This
uses `model.eval()`, no gradients, frozen parameters and no optimizer. The results go to
`diagnostics_generalization_failures/regenerated_predictions/` with their own provenance file. A test checks that
regenerated predictions equal the originals bit for bit.

**What each diagnostic means**

| diagnostic | meaning |
|---|---|
| audit | Integer exact/cell counts are recomputed from raw records. Targets are re-derived with an independent coordinate oracle, and input grids are checked against the manifest. The audit also flags duplicate or missing keys, out-of-range tokens (never clipped), and free-running vs teacher-forced provenance. A synthetic perfect-oracle predictor must score 100% through the same loader. |
| legal-transform match | Each prediction is compared with **all 288** transformations generated by A, B, C (36 translations × 8 orientations) applied to its input. Exact matches can be unique, ambiguous (e.g. symmetric inputs) or absent. A match does not identify which symbolic string was executed. |
| hypotheses | Pre-declared explanations: identity; first/middle/last op; proper prefixes and suffixes; dropped ops and subsequences; reverse and reorder; collapsed repeats; a single op doubled; each training pair. Types overlap, so rates need not sum to 1. |
| stable rule (DIAG) | One transformation is fitted to the **model's predictions** on the FIT half of the base grids and scored on the disjoint DIAG half. This separates a stable wrong rule from input-dependent switching. |
| source maps (DIAG) | The copy matrix `M[j,k] = P(pred_j == input_k)` is used to fit a many-to-one argmax map and a bijective permutation (Hungarian assignment), plus a digit-histogram check. Tie-driven or low-fidelity maps are flagged. |
| composition | The fitted single-op maps are composed in instruction order and compared with both predictions and targets. Results are marked *weak* when the single-op maps are unstable. |
| alternate generators | With Z = shift right 3 and down 3, A'=ZA, B'=ZB, C'=ZC reproduce every length-2 string exactly but predict Z(target) for odd lengths. This is an explicit identifiability check, not a causal claim. It cannot explain failures on unseen pairs. |
| controls | iid random grids, per-grid value shuffles, and fixed wrong legal transforms, run through the same matcher. A check of the formula `k/36 + (1-k/36)/10` shows that exact accuracy 0 with cell accuracy ≈0.10 can be a perfectly structured error. |

DIAG-split fidelities describe frozen predictions with rules fitted to those same predictions; they are not a new
benchmark. Behavioural compatibility does not establish internal representations or causality, and only one seed
exists. `DECOMP_SYNTHETIC_FIXTURES=1` runs the notebook on clearly labelled **synthetic** planted behaviours; this is
for testing only and is never a result.

## 13. Mixed-length experiments (lengths 0, 2, 3, 4, 5; length 1 never trained)

[![Open 04 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/04_train_mixed_02345_distinct_pairs.ipynb)
[`notebooks/04_train_mixed_02345_distinct_pairs.ipynb`](notebooks/04_train_mixed_02345_distinct_pairs.ipynb) ·
[![Open 05 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/05_train_mixed_02345_all_pairs.ipynb)
[`notebooks/05_train_mixed_02345_all_pairs.ipynb`](notebooks/05_train_mixed_02345_all_pairs.ipynb)

**Question:** can models execute the individual operations A, B, C without ever being trained on a single operation?
Each notebook trains all nine models **from scratch**, then evaluates them, runs the diagnostics, plots and exports.
The two notebooks are independent; neither needs the other to have run.

**Supports.** Training and checkpoint-selection validation use the same support. **Only the length-two support
differs** between the two experiments:

| length | `mixed_02345_distinct_pairs` | `mixed_02345_all_pairs` |
|---|---|---|
| 0 | EMPTY (target = input) | EMPTY |
| 1 | never | never |
| 2 | AB AC BA BC CA CB | AA AB AC BA BB BC CA CB CC |
| 3 / 4 / 5 | all 27 / 81 / 243 words | identical |
| total | 358 words | 361 words |

The legacy identifiers `distinct_pairs` and `all_pairs` keep their original length-two-only behaviour and 3-slot
layout. Their config hashes are unchanged, so completed legacy runs still resume and skip correctly.

**Sampling.** The stream is stateless and keyed by sample index:
* **Lengths:** every block of five samples is a seeded permutation of the lengths 0/2/3/4/5, so each length gets
  exactly 20% of samples.
* **Words:** within a length, words follow a balanced schedule (counts differ by at most one). The schedule is
  identical across the two experiments except at length 2.
* **Shared across experiments:** grids (the same stream as the legacy experiments, excluding held-out grids),
  the length schedule, PAD placement, and the words at lengths 0/3/4/5.
* **Budget:** seed 42, 5000 steps × batch 256 = 1,280,000 examples per run (256,000 per length). Pairs now get only
  20% of the budget, unlike the legacy experiments.

**Five operation slots.** A length-k word occupies a uniformly random order-preserving subset of the 5 slots, and the
remaining slots are PAD. EMPTY is five PADs (zero operations, not five identity operations); length 5 has no PAD.
Evaluation uses canonical right padding. Input sizes become:
* AR: 82 tokens, with `<OUTPUT>` at index 44, answers at 45–80 and `<EOS>` at 81 (`tokenization.ARLayout`).
* Encoder: 41 positions. ViT: 14 positions.
* CNN conditioner: concatenates 5 slot embeddings.

All-PAD inputs stay finite because grid keys are always valid. A 3-slot checkpoint is rejected for a 5-slot
configuration.

Unique parameters with 5 slots:

| model | unique parameters |
|---|---:|
| transformer_ar | 10,702,099 |
| looped_transformer_ar | 1,829,779 |
| transformer_encoder | 10,664,074 |
| looped_transformer_encoder | 1,791,754 |
| vit | 10,685,224 |
| looped_vit | 1,812,904 |
| trm | 9,939,840 |
| rescnn_film | 10,039,594 |
| unet_film | 10,334,410 |

**Validation (checkpoint selection).** The 300 validation grids each get one word at every trained length, giving
1,500 records. Words follow a fixed balanced schedule that covers every allowed word, and it is identical across
experiments except at length 2.
* **Selection:** lowest length-macro final-answer cell CE. For AR models this is teacher-forced digit CE, excluding
  EOS.
* **Also logged:** per-length CE and non-empty-only CE.
* **Decoded-accuracy subset:** a fixed 32-grid validation subset (160 records), reported per length.
* Singles and unseen pairs are never used for selection.

**Test.** The selected checkpoint is evaluated on all **364 words** (lengths 0–5, including EMPTY and the singles) ×
the 1000 held-out test grids, giving 364,000 records per model. Evaluation runs in restartable chunks. AR decoding is
free-running, and invalid tokens count as errors. Reported results:
* **Primary:** singles A, B, C separately and as a macro average.
* EMPTY, reported separately.
* Seen pairs.
* Unseen AA/BB/CC (N/A in `all_pairs`).
* Lengths 3/4/5. These are *seen strings on new grids*, not generalization to unseen combinations.
* Micro and length-macro overall accuracy, both labelled explicitly.

The empty word is stored and displayed as the label `EMPTY`; models never see that text.

**Grids.** The base grids are regenerated deterministically from the seed. They are byte-identical to the legacy
manifests (hash `bdf24c327a65…`), and the notebook checks this when the legacy manifest is on Drive.

**Diagnostics (analysis v2).** Each notebook runs the generalized diagnostics (`DiagnosticsConfig(spec="mixed")`) on
its own saved predictions. It adds the sibling experiment read-only, for paired comparisons, when that is present.
The diagnostics cover:
* the full 288-element catalog;
* explicit rules: identity, first/last op, first-and-last, first two/last two, drop one, reverse, collapse adjacent
  repeats, global dedupe, duplicate single, and the original training pairs;
* FIT/DIAG fitted rules and maps, composition, controls, and the secondary half-period Z check.

**Denominator fix.** The v1 report's hypothesis lines averaged per-string rates only over strings where the rule had a
non-truth candidate. It then described that average as a share of *all* unseen examples. This is why
`collapse_repeats` (52.4%) and `double_single` (58.4%) could exceed the any-legal rate. v2 reports, for every rule and
declared population:
* eligible records;
* UNIQUE matched records, using a union within the rule;
* the conditional rate (matched / eligible);
* unconditional coverage (matched / population);
* the any-legal upper bound.

v2 also shows integer successes (e.g. 8998/9000) and defines its thresholds. If an output folder already contains v1
results, v2 writes to `<dir>/versions/<version>/`, so historical exports are never overwritten.

**Outputs.** `/content/drive/MyDrive/decompositionality_results/mixed_length_02345/<experiment>/` contains:
* `runs/<model>/`: `state.pt`, `best.pt`, logs.
* `eval/<model>/`: `test_predictions.npz`, `test_per_word.csv`, `test_groups.csv`, `eval_summary.json`, `chunks/`.
* `figures/` and `tables/`, including `f_summary.md`.
* `diagnostics/`.
* `results_<experiment>.zip`, with no checkpoints, chunks or caches.

Smoke runs go to `.../mixed_length_02345/smoke/`.

**Interpretation limits.** Several things changed at once relative to the legacy experiments: length diversity,
identity supervision, the slot layout, and the per-pair budget. Old/new differences are therefore not causal
evidence about any one factor. The two new experiments differ only in their length-two support and per-pair
exposure. With one seed, the comparisons are descriptive only. Poor seen-support fit is not a clean decomposition
failure. Single-operation success does not by itself identify an internal algorithm.

## 14. Compact left-aligned PE ablation (notebooks 07/08): RoPE vs sinusoidal

[![Open 07 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/07_train_compact_02345_rope.ipynb)
[`notebooks/07_train_compact_02345_rope.ipynb`](notebooks/07_train_compact_02345_rope.ipynb) ·
[![Open 08 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/08_train_compact_02345_sinusoidal.ipynb)
[`notebooks/08_train_compact_02345_sinusoidal.ipynb`](notebooks/08_train_compact_02345_sinusoidal.ipynb)

**Data (identical in both notebooks).** The training distribution is the all-pairs mixed-length support: EMPTY plus
all 9 / 27 / 81 / 243 words of lengths 2 / 3 / 4 / 5, giving 361 words. Each length gets 20% of samples and words are
balanced within a length. Single operations are **never** used for training or checkpoint-selection validation. The
logical example sequence is the same as in notebook 05, except that operations are **compact left-aligned**: the j-th
operation is always in slot j and the remaining slots are PAD. There is no random placement anywhere: not in training,
validation, or the primary test. The 5 physical slots keep the grid and output offsets fixed (AR: 82 tokens,
`<OUTPUT>` at 44, answers 45–80, `<EOS>` 81). Seed 42, 5000 updates × batch 256.

**The only difference between 07 and 08** is the 1D sequence positional encoding of the seven attention models:
transformer_ar, looped_transformer_ar, transformer_encoder, looped_transformer_encoder, vit, looped_vit and trm. It is
selected by the model-config field `sequence_pe`:
* `rope`: standard 1D rotary embedding with theta 10,000 over the full even head dimension (64 for d=384/6 heads,
  80 for TRM's 640/8). It uses the `rotate_half` pairing, so channel i is paired with i + hd/2. It is applied to the
  freshly projected Q and K, never V, in **every** attention call, including every tied loop and every TRM core call.
  It never rotates the carried hidden state.
* `sinusoidal`: a fixed sin/cos table with base 10,000, added **once** at the input embedding with scale 1 (TRM: added
  once to the input embedding x, which the existing recurrence re-injects as before). It is not re-added inside
  loops.

Both variants use the same deterministic layout position IDs and keep the same additive grid row/column (ViT: patch
row/column) and segment/type embeddings. The learned 1D position/slot tables are removed in both. Corresponding models
have identical trainable parameters and identical seed-42 initial weights; the PE tables are buffers that consume no
RNG, and the notebooks verify this before training. So the comparison isolates 1D sequence PE **conditional on the
shared spatial encoding**; it is not a comparison of 2D schemes.

One caveat: the token embeddings of AR, encoder and ViT are initialised with std 0.02, while sinusoidal vectors have
unit amplitude. That is the standard, documented "scale 1" choice and it is identical across models, but the
sinusoidal signal starts out much larger than the token embedding. Unique parameters:

| model | parameters (both PE variants) |
|---|---:|
| transformer_ar | 10,670,611 |
| looped_transformer_ar | 1,798,291 |
| transformer_encoder | 10,662,154 |
| looped_transformer_encoder | 1,789,834 |
| vit | 10,683,304 |
| looped_vit | 1,810,984 |
| trm | 9,936,640 |

**Shared CNN controls.** `rescnn_film` and `unet_film` have no attention PE. They are trained once under
`compact_02345_cnn` in `shared_cnn/`. Whichever notebook runs second reuses them, but only if the configuration hashes
match. They are reported once and marked n/a for the PE contrast. In total: 14 attention runs + 2 CNN runs = 16 unique
runs.

**Selection and test.** These follow notebooks 04/05:
* **Validation:** 1,500 records (300 grids × lengths 0/2/3/4/5), identical for every run. The best checkpoint has the
  lowest length-macro final-answer CE.
* **Test:** all 364 words × the 1000 shared test grids, giving 364,000 records per run and 5,824,000 across the 16
  unique runs. AR decoding is free-running.
* **Primary metric:** A, B and C separately and as a macro average, with integer counts.
* All pairs are seen, so there is no unseen-pair category. The singles are the only held-out strings.
* `EMPTY` is a display label only.

**Diagnostics** (analysis v3) cover:
* the audit and perfect-predictor check;
* the 288-element catalog;
* singleton hypotheses: correct, XX, XC, CX, identity (note that CC is the same transformation as identity), and every
  original pair, with eligible / unique-matched / conditional / unconditional counts;
* FIT/DIAG fitted transforms and source maps;
* composition of fitted single-op maps.

By default the per-example matching is restricted to the 40-word panel (EMPTY + the 39 original words). Per-word
metrics for all 364 words are always reported.

**Paired RoPE − sinusoidal comparison.** It runs automatically in whichever notebook finds both conditions complete.
It uses identical (grid, word) test examples with verified identical inputs and targets, and reports, per model,
operation and length: exact/cell accuracy differences and both-correct / only-RoPE / only-sinusoidal / both-wrong
counts. It is written to `comparisons/`.

**Outputs.** `/content/drive/MyDrive/decompositionality_results/compact_02345_pe/` contains:
* `manifests/`: shared across all runs, byte-identical to the legacy test grids (hash `bdf24c327a65…`).
* `rope/`, `sinusoidal/`, `shared_cnn/`: each with `runs/`, `eval/`, `figures/`, `tables/`, `diagnostics/` and a zip.
* `comparisons/`.

Smoke mode writes to `compact_02345_pe/smoke/`.

**Limits.**
* One seed; no seed-robust ranking.
* Differences from notebooks 04/05 change both the placement distribution and the PE, so they do not isolate
  placement.
* Behavioural diagnostics do not identify an internal mechanism.

**Notebook 09: compact placement + learned 1D PE**
([`notebooks/09_train_compact_02345_learned.ipynb`](notebooks/09_train_compact_02345_learned.ipynb)). Everything is
the same as 07/08 (data, compact layout, manifests, budget, shared CNN controls) except that the seven attention models
use the legacy *trainable* 1D tables. Results go to `compact_02345_pe/learned/`. Together with notebook 05 (random
placement + learned PE, same 361-word support), 09 separates placement (05 vs 09) from PE type (09 vs 07/08). The
learned tables add parameters, so 09 is not parameter-identical to 07/08. One seed.

## 15. Identity-filler vs variable-length instructions, all nine models (notebooks 10/11)

[![Open 10 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/10_identity_02345_all_models.ipynb)
[`notebooks/10_identity_02345_all_models.ipynb`](notebooks/10_identity_02345_all_models.ipynb) ·
[![Open 11 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/11_variable_length_02345_all_models.ipynb)
[`notebooks/11_variable_length_02345_all_models.ipynb`](notebooks/11_variable_length_02345_all_models.ipynb)

**Shared data.**
* Instructions have k A/B/C tokens with k ∈ {0, 2, 3, 4, 5}: 361 logical words, all repetitions allowed. k = 1 is
  never used for training or checkpoint selection. ACC is allowed even though it equals A.
* Each update of 256 records holds 51 records per k plus one rotating extra, so each k gets exactly 256,000 of the
  1,280,000 records (seed 42, 5000 updates).
* Within k, the word is uniform over {A,B,C}^k and the ordered slot subset is uniform over C(5,k). This gives 1009
  five-slot training patterns; the 15 singleton patterns are held out.
* Both formats serialize the SAME indexed records:
  * **identity**: all 5 slots, with every unused slot filled by a visible, learnable `<identity>` token (token 19).
    `<identity>` is a normal attention key and is never masked or zeroed. There is no PAD anywhere.
  * **variable_length**: only the k A/B/C tokens, with no PAD, no `<identity>` and no reserved slots. EMPTY is a
    zero-length operation segment.
* Batches are bucketed by k (`group`), so neither format is ever padded.
* Gradient accumulation weights each bucket by n/256, giving one optimizer and scheduler step per logical update.

**Models.**
* All nine families are trained separately in each format: 18 runs, with no shared CNN controls.
* For each architecture, both formats start from identical initial tensors.
* Attention models use learned absolute 1D positions:
  * identity: physical slots, with fixed offsets (AR: 82 tokens, `<OUTPUT>` at 44).
  * variable_length: contiguous positions, with offsets that move with k (AR: 77+k tokens, `<OUTPUT>` at 39+k).
* TRM states are sized to the actual sequence length.
* The CNNs use a new conditioner shared by both formats: v_j = φ(concat(E_op(o_j), E_pos(j))), s = Σ_j v_j over the
  ACTUAL tokens (k = 0 → s = 0), then ρ → FiLM. The legacy concatenation adapter remains the default for old
  experiments.
* Unique parameters (identical in both formats):

| model | parameters |
|---|---:|
| transformer_ar | 10,702,868 |
| looped_transformer_ar | 1,830,548 |
| transformer_encoder | 10,666,378 |
| looped_transformer_encoder | 1,794,058 |
| vit | 10,685,608 |
| looped_vit | 1,813,288 |
| trm | 9,943,680 |
| rescnn_film | 10,122,410 |
| unet_film | 10,417,226 |

**Selection and test.**
* **Validation:** 1,500 records, one per (grid, k), balanced over all 361 words. The checkpoint is selected on the
  k-macro final-answer digit CE.
* **Test:** the 361 seen words × 1000 grids, with identity using one deterministic placement per word/grid. Plus the
  singletons:
  * identity: A/B/C × 5 placements × 1000 = 15,000 records;
  * variable_length: A/B/C × 1000 = 3,000 records.
* Totals: 376,000 / 364,000 records per model; 6,660,000 across the 18 runs.
* Identity singletons are reported per operation × placement, with the mean and min/max over placements and slot 1
  shown separately. The best placement is never selected.

**Diagnostics.** These cover singletons:
* 288-transform matching;
* explicit X, XX, XC, CX, identity (= CC) and original-pair hypotheses, with eligible / union-matched / coverage
  counts;
* FIT/DIAG fixed transforms and source maps per (operation, placement);
* identity placement consistency;
* the format comparison, paired by model/word/grid hash, with per-placement transition tables and no inflation of the
  3,000 variable records.

Prediction files are located through `artifact_index.json`, which is rebuilt by scanning the per-run `ARTIFACT.json`
files.

**Outputs.** `/content/drive/MyDrive/decompositionality_results/identity_varlen_02345_all_models/` contains
`manifests/`, `identity/<model>`, `variable_length/<model>` and `comparisons/`. Run each notebook with Run all; rerun it
to resume.

**Limits.** One seed. The two formats differ in filler semantics, positions, visible length and downstream offsets;
this is not a pure mask or length ablation. Historical PAD runs (04–09) use a different serialization and CNN adapter,
so they are not paired with these runs.

## 16. Operation-count extrapolation in the identity format (notebooks 12/13; Transformer AR + Looped AR)

[![Open 12](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/12_identity_train02_ar_extrapolation.ipynb)
[`notebooks/12_identity_train02_ar_extrapolation.ipynb`](notebooks/12_identity_train02_ar_extrapolation.ipynb) ·
[![Open 13](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhexuandliu/Decompositionality/blob/main/notebooks/13_identity_train023_ar_extrapolation.ipynb)
[`notebooks/13_identity_train023_ar_extrapolation.ipynb`](notebooks/13_identity_train023_ar_extrapolation.ipynb)

**Setup.**
* The model, serializer and recipe are exactly those of the successful notebook-10 identity run: five visible
  operation tokens, unused slots filled with the learnable `<identity>`, random ordered placement, no PAD, learned
  positions, 82 AR tokens.
* The model hashes match notebook 10: `b8c7a0906c8420d2` for transformer_ar (10,702,868 parameters) and
  `76a3940b2ffbff78` for looped_transformer_ar (1,830,548 parameters).
* Only the training support over k (the number of A/B/C tokens) changes. Extrapolation is therefore in k, not in
  sequence length.

| experiment | train k | records per update | records over 5000 updates | held-out k | logical words / physical patterns |
|---|---|---|---|---|---|
| train02 | 0, 2 | 128 + 128 | 640,000 each | 1, 3, 4, 5 | 10 / 91 |
| train023 | 0, 2, 3 | 85 each + 1 rotating (0→2→3) | 426,667 / 426,667 / 426,666 | 1, 4, 5 | 37 / 361 |

**Records and initialisation.**
* Records are indexed within k (`record(seed, k, j)`), so the common k=0/2 records of the two experiments come from
  one per-k stream. Their exposure counts and batch positions still differ.
* train023 is NOT train02 plus triples: at an equal total budget it replaces some k=0/2 exposure with k=3.
* Each architecture starts from identical initial weights in both experiments.

**Selection.** Validation uses 300 grids at each trained k (600 / 900 records). The common k=0/2 validation records
are identical across the two experiments. The checkpoint with the lowest answer-digit CE, macro-averaged over the
trained k, is selected.

**Test.** The common notebook-10 manifest is used for all four runs:

| k | words | records |
|---|---|---|
| 0 | 1 | 1,000 |
| 1 | 3 (× 5 placements) | 15,000 |
| 2 | 9 | 9,000 |
| 3 | 27 | 27,000 |
| 4 | 81 | 81,000 |
| 5 | 243 | 243,000 |

That is 376,000 records per run. Each k is reported separately, with integer counts and a seen / HELD OUT label for
that run.

**Transformation coverage (post-hoc).** train02 has 9 distinct training transformations and train023 has 30. Words
whose transformation is in the training support, per k = 0..5:
* train02: 1/1, 0/3, 9/9, 0/27, 38/81, 0/243
* train023: 1/1, 3/3, 9/9, 27/27, 38/81, 162/243

**Even-count ambiguity (post-hoc).** H = a shift by 3 rows and 3 columns is an involution that commutes with A, B and
C. So H∘A, H∘B, H∘C reproduce the true mapping for every even k and differ for every odd k. The train02 data alone
therefore do not rule this alternative out; k = 3 supervision does. The notebooks report odd-k agreement with H(target)
descriptively, with overlap counts. It is never a training signal.

**Outputs.** `/content/drive/MyDrive/decompositionality_results/identity_ar_count_extrapolation/` contains:
* `manifests/`
* `train02/seed_42/<model>` and `train023/seed_42/<model>`
* `artifact_index.json`
* `comparisons/seed_42/`

**Third support: train0234** ([`notebooks/14_identity_train0234_ar_extrapolation.ipynb`](notebooks/14_identity_train0234_ar_extrapolation.ipynb)).
* Train k ∈ {0, 2, 3, 4} at 64 records per k per update (320,000 each). Held out: k ∈ {1, 5}.
* 118 logical words, 766 physical patterns, 62 distinct training transformations.
* Words whose transformation is in training, per k = 0..5: 1/1, 3/3, 9/9, 27/27, 81/81, 162/243.
* Same models, initial weights, format, recipe and test manifest as 12/13. Results go to `train0234/seed_42/`.

**Running.** Each notebook trains both models; rerun it to resume. Once the notebooks you need are finished, rerunning
only the analysis cell gives the joint comparison across all available supports. One seed; descriptive only.

## 17. Slot-count intervention in the identity format (notebook 16; eight runs, one Run all)

[`notebooks/16_identity_slot_count_all_conditions_ar.ipynb`](notebooks/16_identity_slot_count_all_conditions_ar.ipynb)

**Question.**
* Five-slot identity training on k ∈ {0,2,3,4,5} generalized to singletons (notebook 10).
* {0,2}, {0,2,3} and {0,2,3,4} did not (notebooks 12–14).
* In those three failing supports, every training instruction contains at least one `<identity>` (I). train02345 also
  has k=5 instructions with no I.
* Notebook 16 changes the number of physical slots S at fixed logical support. This is NOT a pure identity ablation:
  S also changes sequence length, downstream absolute positions, layouts, filler counts and compute.

| condition | S | train k | per-k training records | words / physical patterns | no-I training records | held-out tests |
|---|---|---|---|---|---|---|
| slots2_train02 | 2 | 0,2 | 640,000 each (128/update) | 10 / 10 | 640,000 | k=1 × 2 positions |
| slots3_train023 | 3 | 0,2,3 | 426,667 / 426,667 / 426,666 (85 + 1 rotating) | 37 / 55 | 426,666 | k=1 × 3 positions |
| slots4_train0234 | 4 | 0,2,3,4 | 320,000 each (64/update) | 118 / 244 | 320,000 | k=1 × 4 positions |
| slots6_train02345 | 6 | 0,2,3,4,5 | 256,000 each (51 + 1 rotating) | 361 / 3349 | 0 | k=1 × 6 positions; all 729 k=6 words |

**Format.**
* Exactly S visible operation tokens. The k A/B/C tokens go in a uniformly sampled ordered k-subset of the S slots;
  the rest are the learnable `<identity>`.
* No PAD, no ghost slots. Sequence length is 77+S tokens: OUTPUT at 39+S, answers 40+S..75+S, EOS at 76+S
  (asserted from the serializer).
* Contiguous learned absolute positions in a shared 83-row table. Inputs keep their true length; conditions are
  never padded to a common S.
* S=6/k=6 extrapolates in the non-identity count (the first inputs with zero I). It does not extrapolate in
  position indices or sequence length.

**Pairing and initialization.**
* For 02/023/0234 the logical (word, grid, target) stream and batch order are identical to the S=5 runs of
  notebooks 12–14; only the layout changes. This is verified in the audit cell.
* S=6/02345 is a new stream, labelled unmatched, because notebook 10 used a different sampler.
* Both architectures read the same stateless stream.
* One common fresh initialization per architecture across all four conditions: the historical S=5 seed-42 init
  (shared tensors verified equal) with position row 82 added from a separate generator.
* Parameter counts: Transformer AR 10,703,252 (S=5: 10,702,868); Looped AR 1,830,932 (S=5: 1,830,548).
  Both use 6 block applications.

**Recipe.** Unchanged from notebooks 10/12–14:
* AdamW 3e-4, weight decay 0.01, 250 warmup, cosine to 3e-5, clip 1.0.
* 5000 updates × 256 = 1,280,000 records per run; validation every 250 updates.
* Selection: lowest validation answer-digit CE (EOS excluded), macro over the trained k, ties to the earlier update.
* Validation: 300 records per trained k (600 / 900 / 1200 / 1500 by condition).

**Evaluation per model.**
* 1000 test grids, free-running greedy decoding.
* Counts per condition (seen + singleton + six-op = total):
  * slots2: 10,000 + 6,000 = 16,000
  * slots3: 37,000 + 9,000 = 46,000
  * slots4: 118,000 + 12,000 = 130,000
  * slots6: 361,000 + 18,000 + 729,000 = 1,108,000
* Total over both models: 2,600,000 records.
* Training transformations: 9 / 30 / 62 / 111. Of the 729 six-op words, 591 induce a training transformation and 138
  do not.

**Analysis.** All of it reads saved artifacts only:
* main table, training/validation curves, accuracy by k;
* A/B/C × slot heatmaps at each actual S;
* same-support S=5 vs new-S singleton comparison, with metadata and paired logical seen controls;
* six-op per-word table, transform-membership strata, and first-five / last-five / single-deletion checks on
  failures;
* identity-count view;
* initialization, layout, stream, timing and status audits.

**Running.**
* Run all executes slots2 → slots3 → slots4 → slots6, Transformer AR then Looped AR, one run at a time: train →
  evaluate → diagnose → export. That is roughly 8–9 A100 hours.
* `RUN_CONDITIONS` / `RUN_MODELS` / `RUN_STAGES` select reruns; the defaults are all eight runs and every stage.
* Interrupted runs: Run all again. Verified stages are skipped, training resumes from `state.pt`, and missing
  evaluation chunks are recomputed.
* The comparison cells can be rerun alone. Missing runs or historical sources show as "not yet available".
* Results go to `<RESULTS_ROOT>/identity_ar_slot_count/`. `DECOMP_MODE=smoke` writes a tiny CPU integration run to
  `.../smoke/` (not results).

**Interpretation limits (one seed, descriptive).**
* Recovery at S=3/4 together with singleton loss at S=6 would be consistent with the zero-I hypothesis, without
  proving a mechanism.
* S=6 singleton success would show that zero-I examples are not necessary.
* S=3/4 failure would show they are not sufficient under this recipe.
* S=2/train02 keeps the even-count H ambiguity.
* S=6 six-op failure confounds larger k with the removal of all I.

## 18. AR identity / mirror diagnostics (notebook 17; diagnostic only, no training)

[`notebooks/17_ar_identity_mirror_diagnostics.ipynb`](notebooks/17_ar_identity_mirror_diagnostics.ipynb)

**Inputs.** Reads SAVED free-running predictions of Transformer AR and Looped AR from three roots under
`<RESULTS_ROOT>`:
* `identity_varlen_02345_all_models` (S=5/train02345, plus optional variable_length);
* `identity_ar_count_extrapolation` (S=5/train02, 023, 0234);
* `identity_ar_slot_count` (S=2/3/4/6).

Runs are discovered read-only from `ARTIFACT.json` files, plus any unindexed `test_predictions.npz`. `RUN_OVERRIDES`
points the notebook at other locations.

**Missing data.** Default `RECOMPUTE_MISSING_PREDICTIONS=False`: nothing is trained, re-evaluated or re-inferred.
* Missing sources are listed as unavailable; corrupt or failed-audit sources are quarantined. Neither is counted as zero.
* User-reported counts are reconciled against the artifacts, never used as results.

**Method** (code in `mirror_diag.py` and `mirror_report.py`, tests in `tests/test_mirror_diag.py`):
* The exact 288-element group in canonical form T(dr,dc) o B^r o C^m.
* Geometric mirror class = #C mod 2, not the cell-permutation sign.
* Per-grid full-byte dictionaries match every singleton, six-step and control prediction to ALL legal transforms.
  Ambiguity is kept.
* Wrong legal outputs are classed as extra output-C (Delta_out = C), extra input-C (Delta_in = C) or other residuals.
* Also: same-target pairs A~ACC, B~BCC, C~CCC, C~CCCCC; held-out (500/500) source-position maps; six-step strata by
  n_C x training-transform membership; placement / slot / format comparisons; an evidence table for six questions.

**Send back** `<RESULTS_ROOT>/ar_identity_mirror_diagnostics/<analysis_id>/AR_Identity_Diagnostics_Review.ipynb`.
* It is a self-contained report snapshot with every table and figure baked in; no cells to run, no Drive needed.
* The last notebook cell prints its path and downloads it.
* CSV tables, PNGs, `diagnostic_report.html`, `diagnostic_summary.txt` and `summary.json` sit next to it.

## 19. Row separators: 13 conditions x 2 AR models in one notebook (notebook 18)

[`notebooks/18_ar_row_separators_all_experiments.ipynb`](notebooks/18_ar_row_separators_all_experiments.ipynb):
26 fresh runs (Transformer AR + Looped AR), seed 42, 5000 x 256 each, learned absolute positions everywhere.

**ROW layout.**
* A new token `<ROW>` = ID 20 (vocabulary 21; older IDs unchanged) is placed between the rows of BOTH grids: 41
  tokens per grid, 5 ROW tokens each.
* For m operation tokens: length 87+m, OUTPUT at 44+m, output digit (r,c) at 45+m+7r+c, output ROW_r at 51+m+7r,
  EOS at 86+m. All asserted by `tokenization.ARRowLayout`.
* Digits keep their true (r,c) features; ROW tokens get the grid segment and NO_COORD; positions are dense.

**Conditions.**

| group | conditions | train k | held-out tests |
|---|---|---|---|
| fixed5 (5 slots, visible I) | identity_s5_train02/023/0234/02345 | 0,2 … 0,2,3,4,5 | k=1 at all slots + every longer k <= 5 |
| variable_with_zero (exactly k tokens, k=0 = empty segment) | variable_train02/023/0234/02345 | same | k=1 + longer k <= 5 |
| fixed6 (6 slots, visible I) | identity_s6_train02345 | 0,2,3,4,5 | k=1 at 6 slots + all 729 k=6 |
| variable_nozero (no k=0 in training or validation) | variable_nozero_train2/23/234/2345 | 2 … 2,3,4,5 | k=1 (+ seen-support checks) |

* Test records per architecture: 4 x 376,000 + 4 x 364,000 + 1,108,000 + 12,000 + 39,000 + 120,000 + 363,000 =
  4,602,000.

**Data and initialization.**
* Logical stream: `idext.record(seed,k,j)` with the rotating per-update allocation. The same support gives an
  identical ordered stream for S5 / variable / S6. Supports share records per k by index.
* Removing k=0 redistributes exposure to positive k (no extra budget-matched runs).
* One common initialization per architecture for all 13 conditions: the legacy seeded random init, with only the new
  ROW-embedding, head and position rows drawn from a separate generator.

**Loss, selection, scoring.**
* Loss: mean CE over 42 targets (36 digits + 5 ROW + EOS).
* Selection: answer-digit CE over the trained k only.
* Free-running decoding of 41 + 1 tokens, scored at the predeclared positions with no repair.
  * Exact grid = the 36 digit positions.
  * Strict = digits + ROW + EOS.

**Historical-selection audit.** The old "over k in 0/2/3/4/5" text was a stale constant label. The code selected on
the trained k only.

**Running.** Run all (`RUN_GROUP`, `RUN_CONDITIONS`, `RUN_MODELS`, `RUN_STAGES` select partial reruns). Results go to
`<RESULTS_ROOT>/ar_row_separators_13conditions_v2/`. Send back `report/seed_42/AR_Row_Separators_Review.ipynb`.

## 20. ROW supplement: fixed S = 2/3/4 identity slots (notebook 19; 6 runs)

[`notebooks/19_ar_row_identity_slots234_supplement.ipynb`](notebooks/19_ar_row_identity_slots234_supplement.ipynb)
is a standalone supplement. Notebook 18's 26-job registry, condition IDs, config hashes and result root are unchanged.

**Conditions.**

| condition | S | train k | sequence length | test records / model |
|---|---|---|---|---|
| identity_s2_train02_row | 2 | 0,2 | 89 | 16,000 |
| identity_s3_train023_row | 3 | 0,2,3 | 90 | 46,000 |
| identity_s4_train0234_row | 4 | 0,2,3,4 | 91 | 130,000 |

Both AR models run in each condition: 6 runs, 384,000 evaluation records in total.

**Shared with notebook 18.** Serializer, vocabulary, model config and common initialization (identical tensors);
42-target loss; digit-CE selection; scoring.

**Pairing.**
* The logical stream equals the main S5 / variable stream at the same support.
* Placements and test manifests equal notebook 16's S=2/3/4 no-ROW runs.

**Results.** `<RESULTS_ROOT>/ar_row_slots234_supplement_v1/`.
* Comparisons with notebook 18 (same support) and notebook 16 (same S, no ROW) are read-only. They use only artifacts
  with completion markers; missing ones show as pending.
* Send back `report/seed_42/AR_Row_Slots234_Review.ipynb`.

## 21. Causal encoder in four identity settings (notebook 20; 8 runs)

[`notebooks/20_causal_encoder_identity_settings.ipynb`](notebooks/20_causal_encoder_identity_settings.ipynb)

**Model.** `causal_encoder` (6 untied blocks) and `looped_causal_encoder` (one shared block × 6).
* Sequence: `[op slots] + [36 input cells] + [36 output queries]`, causal attention over the whole sequence.
* All answers are read at the queries in one parallel pass, with no autoregressive feedback.
* Implementation: `TransformerConfig(out_mode="causal")`. The default `"inplace"` is omitted from config hashes, so
  every existing encoder / AR hash is unchanged.
* The new model ids are accepted by configs but are not part of the default nine-model queue.

**Settings.** Each is paired with an existing AR run: same stream, validation and test manifests.
* S5 k∈{0,2,3,4,5}: notebook 10.
* S5 k∈{0,2,3,4}: notebook 14.
* S4 k∈{0,2,3,4}: notebook 16.
* S6 k∈{0,2,3,4,5}: notebook 16.

One common initialization per architecture across the four settings.

**Results.** `<RESULTS_ROOT>/causal_encoder_identity_v1/`. Send back `report/seed_42/Causal_Encoder_Review.ipynb`.

## 22. AR vs causal encoder, last checkpoint, longer instructions (notebook 21; 8 runs)

[`notebooks/21_ar_vs_causal_encoder_last_ckpt_longk.ipynb`](notebooks/21_ar_vs_causal_encoder_last_ckpt_longk.ipynb)

**Runs.** `transformer_ar` and `causal_encoder` × four conditions:

| condition | train k | held-out tests |
|---|---|---|
| S5_k0234 | 0,2,3,4 | k=1, k=5 |
| S6_k02345 | 0,2,3,4,5 | k=1, k=6 |
| S7_k02345 | 0,2,3,4,5 | k=1, k=6, k=7 |
| S7_k023456 | 0,2,3,4,5,6 | k=1, k=7 |

All runs use the identity-filler format.

**Shared setup.** Positional capacity 7, one initialization per architecture, and the per-k indexed logical stream.

**Evaluation.** The LAST checkpoint (`state.pt` at update 5000). Singletons are tested at every slot; every held-out
longer word is tested. k=7 uses `K7_GRIDS` grids (default 300).

**Results.** `<RESULTS_ROOT>/longk_ar_vs_causal_encoder_v1/`. Send back
`report/seed_42/LongK_AR_vs_CausalEnc_Review.ipynb`.

## 23. S=7 rerun at 10,000 updates (notebook 22; 4 runs)

[`notebooks/22_ar_vs_causal_encoder_S7_10k.ipynb`](notebooks/22_ar_vs_causal_encoder_S7_10k.ipynb)

**Runs.** S7_k02345 and S7_k023456 × {Transformer AR, causal encoder}, trained for 10,000 updates instead of 5000.

**Unchanged from notebook 21.** Initialization, logical stream and test sets. Evaluation is again the last
checkpoint.

**Results.** A separate folder, `longk_ar_vs_causal_encoder_S7_10k_v1/`, so notebook 21's results are kept. The
report includes a 5k vs 10k table.

## 24. S=7 support sweep at 20,000 updates with sampled evaluation (notebooks 23a + 23b; 12 runs)

The sweep is split into two halves that can run at the same time on two runtimes:
* [`notebooks/23a_S7_support_sweep_20k_k02_k023_k0234.ipynb`](notebooks/23a_S7_support_sweep_20k_k02_k023_k0234.ipynb):
  k02, k023, k0234.
* [`notebooks/23b_S7_support_sweep_20k_k02345_k023456_k0234567.ipynb`](notebooks/23b_S7_support_sweep_20k_k02345_k023456_k0234567.ipynb):
  k02345, k023456, k0234567.

Each half writes its own report (`report/seed_42_part{a,b}/`). After both finish, the optional last cell (in either
notebook) writes the combined report (`report/seed_42_all/S7_Sweep_20k_ALL_Review.ipynb`).

**Runs.** S=7 identity-filler format with training supports k ∈ {0,2}, {0,2,3}, {0,2,3,4}, {0,2,3,4,5},
{0,2,3,4,5,6} and {0,2,3,4,5,6,7}, each with {Transformer AR, causal encoder}: 12 runs.
* 20,000 updates each.
* One initialization per architecture.

**Evaluation.** The LAST checkpoint, on a sampled test set (`longk.make_test_sampled`; about 52k–127k records per
run):
* singletons at every slot on 1000 grids;
* 5000 records per trained k;
* 20,000 records per held-out longer k, with every word covered.

**Results.** `<RESULTS_ROOT>/S7_support_sweep_20k_v1/`. Send back the part reviews, or the combined one.

## 25. S=7 longer-instruction analysis (notebook 24; read-only)

[`notebooks/24_S7_longer_k_analysis.ipynb`](notebooks/24_S7_longer_k_analysis.ipynb) reads the saved notebook
23a/23b predictions; it does not train or re-evaluate.

**Analysed:** S7_k02345 (k=6, k=7) and S7_k023456 (k=7), both models.
* k=6 accuracy by `<identity>` slot;
* accuracy by number of C tokens and by training-transform membership;
* per-word accuracy;
* failure types;
* "dropped one operation" matches.

**Output.** Send back `report/seed_42_longer_k_analysis/S7_LongerK_Analysis_Review.ipynb`.

## 26. S=7 longer-instruction follow-up (notebook 25; read-only)

[`notebooks/25_S7_longer_k_followup.ipynb`](notebooks/25_S7_longer_k_followup.ipynb) covers three analyses:
1. Causal encoder k=7 success vs word structure: first/last ops, counts, C runs, length after cancelling
   CC/BBBB/AAAAAA, shortest word for the transform.
2. Transformer AR k=6 "dropped 5th operation": unambiguous counts, and by letter / slot of op #5 and by identity
   slot.
3. Causal encoder k=6 early-identity failures, by number of C and by first operation.

Send back `report/seed_42_longer_k_followup/S7_LongerK_Followup_Review.ipynb`.

## 27. Auxiliary-task transfer (notebooks 26a + 26b; 8 runs, run in parallel)

[`26a`](notebooks/26a_aux_task_none_and_k02345.ipynb) runs aux_none and aux_k02345;
[`26b`](notebooks/26b_aux_task_k0234567_and_k01234567.ipynb) runs aux_k0234567 and aux_k01234567.
Each condition is run with Transformer AR and the causal encoder, for 20,000 updates.

**Main task.** S=7 A/B/C, trained k ∈ {0,2,3,4,5}; tests: singletons, k=6, k=7. The test set is identical to
notebook 23's sampled set.

**Auxiliary task.** Value operations D (d+1 mod 10) and E (9−d), which never move a cell.
* Same slots, fillers and placement as the main task.
* A task marker leads every instruction, so the instruction has 8 operation tokens.
* 64 of 256 records per update are auxiliary.
* Its training k is set per condition, up to 7 and optionally including k=1.

**Vocabulary.** The new token IDs are 21–24. Older IDs and config hashes are unchanged.

**Selection and evaluation.** Checkpoint selection uses the main task only; evaluation uses the LAST checkpoint.

**Results.** `<RESULTS_ROOT>/aux_task_transfer_v1/`.

## 28. Extra seeds for the S=7 sweep (notebooks 27a + 27b; 24 runs, parallel)

27a runs seed 43 and 27b runs seed 44; each covers all six supports × both models (12 runs). Seed 42 reuses the
notebook-23 runs.

**Seeds.** The training seed controls initialization and the training stream. Held-out grids, validation and the
sampled test records use the fixed evaluation seed 42 (`longk_exp.EVAL_SEED`), which leaves the seed-42 runs
unchanged.

**Report.** `longk_exp.multiseed_report` produces:
* accuracy vs test k per support (mean, min–max band, individual seeds);
* singleton / +1 / +2 step accuracy vs longest trained k;
* mean ± std tables.

## 29. Combine the three seeds (notebook 28; read-only)

[`notebooks/28_S7_sweep_combine_seeds.ipynb`](notebooks/28_S7_sweep_combine_seeds.ipynb) combines seed 42
(notebooks 23a/23b), seed 43 (27a) and seed 44 (27b). It reads saved predictions only.

**Outputs.** Availability, mean ± std tables with per-seed values, accuracy-vs-k curves per training support, the
singleton / +1 / +2 summary figure, and a compact held-out summary (mean [min, max]). Unfinished seeds are excluded,
not counted as zero.

## 12. Known limitations

* Only one training seed is used, so there is no cross-seed robustness claim.
* The budgets (5000 steps) are initial choices, not convergence guarantees. There is no hyperparameter search.
* Single-operation identifiability (whether length-two training data determine the meaning of single
  operations) is deliberately out of scope for this first experiment.
* GPU paths were first exercised in the Colab runs recorded in notebooks 01/02 (A100, BF16). OOM retry was not
  triggered there.

## References

* TRM: A. Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*, [arXiv:2510.04871](https://arxiv.org/abs/2510.04871); code [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT).
* VARC (vision-style ARC models): [arXiv:2511.14761](https://arxiv.org/abs/2511.14761).
* FiLM: Perez et al., [arXiv:1709.07871](https://arxiv.org/abs/1709.07871).
* CoordConv: Liu et al., [arXiv:1807.03247](https://arxiv.org/abs/1807.03247).
* ResNet: He et al., [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
* U-Net: Ronneberger et al., [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).

The categorical colour palette for model families in the figures comes from a validated reference data-viz
palette. Tied variants reuse their untied model's hue with a lighter fill and hatching.
