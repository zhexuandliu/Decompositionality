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
