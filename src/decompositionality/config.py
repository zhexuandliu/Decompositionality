"""Experiment / training configuration objects and the full vs smoke presets.

Only ``experiment`` (training pair set) and the output directory differ between the
two notebooks; everything else is shared.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Optional

from . import operations as O
from .data import DATA_VERSION
from .models.registry import MODEL_IDS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class TrainConfig:
    seed: int = 42
    optimizer_steps: int = 5000
    effective_batch_size: int = 256
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 250
    grad_clip: float = 1.0
    val_every: int = 250
    log_every: int = 25
    # --- settings below do NOT change which examples are seen or the objective ---
    device: str = "auto"                 # "auto" | "cuda" | "cpu"
    dtype: str = "auto"                  # "auto" (bf16 if supported, else fp16+scaler) | "bf16" | "fp16" | "fp32"
    microbatch_size: Optional[int] = None  # None = probe a safe size with dummy data
    max_microbatch_size: int = 256
    cpu_microbatch_size: int = 32
    eval_batch_size: Optional[int] = None
    activation_checkpointing: bool = False
    tf32: bool = True

    def semantic_dict(self) -> dict:
        """Fields that define the training run (used for config hashes/resume checks)."""
        d = asdict(self)
        for k in ("device", "dtype", "microbatch_size", "max_microbatch_size", "cpu_microbatch_size",
                  "eval_batch_size", "activation_checkpointing", "tf32", "log_every"):
            d.pop(k)
        d["betas"] = list(d["betas"])
        return d


@dataclass
class ExperimentConfig:
    experiment: str = "distinct_pairs"     # legacy "distinct_pairs" | "all_pairs"; or a mixed-length id
                                           # "mixed_02345_distinct_pairs" | "mixed_02345_all_pairs"
    mode: str = "full"                     # "full" | "smoke"
    output_root: str = "results"
    models: List[str] = field(default_factory=lambda: list(MODEL_IDS))
    n_val_grids: int = 300
    n_test_grids: int = 1000
    val_subset_grids: int = 32
    resume: bool = True
    train: TrainConfig = field(default_factory=TrainConfig)
    # Optional per-model config overrides {model_id: {field: value}}. Empty for the benchmark;
    # used by unit tests (tiny fixtures). Part of the run config hash.
    model_overrides: dict = field(default_factory=dict)

    @property
    def is_mixed(self) -> bool:
        from .mixed import is_mixed
        return is_mixed(self.experiment)

    @property
    def is_idvar(self) -> bool:
        from .idvar import is_idvar
        return is_idvar(self.experiment)

    @property
    def n_slots(self) -> int:
        """Operation slots / max instruction positions: legacy 3; mixed-length, compact and identity/variable 5."""
        from .mixed import N_SLOTS
        return N_SLOTS if (self.is_mixed or self.is_idvar) else 3

    @property
    def train_strings(self) -> List[str]:
        """Training words as LABELS (the empty word is "EMPTY")."""
        if self.is_idvar:
            from .idvar import training_labels as idvar_labels
            return idvar_labels()
        if self.is_mixed:
            from .mixed import training_labels
            return training_labels(self.experiment)
        return list(O.TRAIN_SETS[self.experiment])

    @property
    def experiment_dir(self) -> Path:
        from .idvar import IDVAR_DIRS
        from .mixed import COMPACT_DIRS
        return Path(self.output_root) / {**COMPACT_DIRS, **IDVAR_DIRS}.get(self.experiment, self.experiment)

    @property
    def manifest_dir(self) -> Path:
        """Compact PE experiments share ONE manifest folder at the family root; others keep theirs per experiment."""
        from .idvar import is_idvar
        from .mixed import is_compact
        shared = is_compact(self.experiment) or is_idvar(self.experiment)
        return Path(self.output_root) / "manifests" if shared else self.experiment_dir / "manifests"

    @property
    def sequence_pe(self):
        from .mixed import COMPACT_EXPERIMENTS
        return COMPACT_EXPERIMENTS.get(self.experiment)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_strings"] = self.train_strings
        d["n_slots"] = self.n_slots
        d["data_version"] = DATA_VERSION
        return d


SMOKE_OVERRIDES = dict(optimizer_steps=6, effective_batch_size=16, warmup_steps=2, val_every=3, log_every=1)
SMOKE_SPLITS = dict(n_val_grids=8, n_test_grids=4, val_subset_grids=4)


def make_experiment_config(experiment: str, mode: str = "full", output_root: str = "results",
                           models: Optional[List[str]] = None, **train_overrides) -> ExperimentConfig:
    """Full mode = the benchmark defaults. Smoke mode keeps the FULL model architectures
    and only shrinks steps/batch/eval sets for integration testing (never for reporting)."""
    from .idvar import IDVAR_EXPERIMENTS
    from .mixed import COMPACT_EXPERIMENTS, MIXED_EXPERIMENTS
    known = list(O.TRAIN_SETS) + list(MIXED_EXPERIMENTS) + list(COMPACT_EXPERIMENTS) + list(IDVAR_EXPERIMENTS)
    if experiment not in known:
        raise ValueError(f"experiment must be one of {known}")
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be 'full' or 'smoke'")
    tc = TrainConfig()
    splits = {}
    if mode == "smoke":
        tc = replace(tc, **SMOKE_OVERRIDES)
        splits = dict(SMOKE_SPLITS)
        if experiment in MIXED_EXPERIMENTS or experiment in COMPACT_EXPERIMENTS or experiment in IDVAR_EXPERIMENTS:
            splits["n_test_grids"] = 2   # 2 grids x 364 words = 728 smoke test records
    tc = replace(tc, **train_overrides)
    root = output_root if mode == "full" else str(Path(output_root) / "smoke")
    cfg = ExperimentConfig(experiment=experiment, mode=mode, output_root=root,
                           models=list(models or MODEL_IDS), train=tc, **splits)
    for m in cfg.models:
        if m not in MODEL_IDS:
            raise ValueError(f"unknown model {m}")
    return cfg


def load_experiment_json(name: str) -> dict:
    return json.loads((CONFIG_DIR / "experiments" / f"{name}.json").read_text())
