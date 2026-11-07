"""Training loop: AdamW + warmup/cosine schedule, gradient accumulation to a fixed
effective batch, mixed precision, OOM-safe microbatching, validation-based checkpoint
selection, and exact resume.

Invariants
----------
* One optimizer step and one LR-schedule step per logical batch of
  ``effective_batch_size`` examples; the logical batch is generated in full before being
  split into microbatches.
* Each microbatch loss is the mean per-example objective weighted by
  ``microbatch_examples / effective_batch_size`` so the accumulated gradient equals the
  gradient of the mean over the logical batch, independently of the microbatch size.
* An OOM retries the SAME logical batch with a smaller microbatch; no examples are
  consumed or skipped and the effective batch is never reduced.
* Validation, profiling and microbatch probing use separate data and never touch the
  training stream (which is stateless anyway).
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .config import ExperimentConfig, TrainConfig
from .data import DATA_VERSION, TrainStream
from .evaluation import amp_context, run_model_on_set, to_device
from .models.common import GridModel, count_parameters
from .models.registry import build_model, config_dict
from .utils import (atomic_torch_save, atomic_write_json, atomic_write_text, dtype_name, environment_info,
                    free_memory, hardware_name, is_oom, resolve_amp_dtype, resolve_device, stable_hash)

CHECKPOINT_FORMAT = 1


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------

def make_optimizer(model: torch.nn.Module, tc: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; weight decay on matrices/conv kernels only (not biases, norms, embeddings)."""
    decay, no_decay = [], []
    emb_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and id(p) not in emb_ids and not name.endswith("_emb"):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=tuple(tc.betas))


def lr_factor(update: int, tc: TrainConfig) -> float:
    """Multiplier for optimizer update ``update`` (0-indexed): linear warmup for
    ``warmup_steps`` updates, then cosine decay from lr to min_lr at the final update."""
    if update < tc.warmup_steps:
        return (update + 1) / tc.warmup_steps
    total = max(1, tc.optimizer_steps - tc.warmup_steps - 1)
    progress = min(1.0, (update - tc.warmup_steps) / total)
    floor = tc.min_lr / tc.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def make_scheduler(opt, tc: TrainConfig):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda u: lr_factor(u, tc))


def make_scaler(amp_dtype):
    enabled = amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Microbatch probing
# ---------------------------------------------------------------------------

def probe_microbatch(model: GridModel, tc: TrainConfig, device, amp_dtype, log=print) -> int:
    """Largest power-of-two microbatch (<= effective batch) whose forward+backward fits,
    measured with DUMMY data (never the training stream)."""
    if tc.microbatch_size:
        return min(tc.microbatch_size, tc.effective_batch_size)
    if device.type != "cuda":
        return min(tc.cpu_microbatch_size, tc.effective_batch_size)
    cand = min(tc.max_microbatch_size, tc.effective_batch_size)
    cand = 2 ** int(math.floor(math.log2(cand)))
    model.train()
    while cand >= 1:
        try:
            batch = model.dummy_batch(cand, device)
            with amp_context(device, amp_dtype):
                loss, _ = model.training_loss(batch)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del loss, batch
            free_memory()
            log(f"  microbatch probe: {cand} fits")
            return cand
        except Exception as exc:
            if not is_oom(exc):
                raise
            model.zero_grad(set_to_none=True)
            free_memory()
            log(f"  microbatch probe: {cand} OOM, trying {cand // 2}")
            cand //= 2
    raise RuntimeError("even microbatch size 1 does not fit in memory")


# ---------------------------------------------------------------------------
# One logical optimizer step
# ---------------------------------------------------------------------------

class StepRunner:
    def __init__(self, model, opt, sched, scaler, tc: TrainConfig, device, amp_dtype, microbatch: int, log=print):
        self.model, self.opt, self.sched, self.scaler = model, opt, sched, scaler
        self.tc, self.device, self.amp_dtype, self.mb, self.log = tc, device, amp_dtype, microbatch, log
        self.oom_retries = 0

    def __call__(self, batch_np: Dict[str, np.ndarray]) -> Dict[str, float]:
        B = len(batch_np["x"])
        assert B == self.tc.effective_batch_size
        while True:
            try:
                return self._attempt(batch_np, B)
            except Exception as exc:
                if not is_oom(exc) or self.mb <= 1:
                    raise
                self.opt.zero_grad(set_to_none=True)
                free_memory()
                self.mb = max(1, self.mb // 2)
                self.oom_retries += 1
                self.log(f"  OOM: retrying the SAME logical batch with microbatch {self.mb}")

    def _attempt(self, batch_np, B) -> Dict[str, float]:
        self.opt.zero_grad(set_to_none=True)
        totals: Dict[str, float] = {}
        from .evaluation import index_chunks
        # same-length buckets when the batch carries a "group" field (identity / variable_length experiments);
        # each microbatch contributes (mean loss) * n / B = (sum of per-example losses) / B
        for sl in index_chunks(batch_np, self.mb):
            n = len(sl)
            w = n / B
            tb = to_device(batch_np, self.device, sl)
            with amp_context(self.device, self.amp_dtype):
                loss, metrics = self.model.training_loss(tb)
            if not torch.isfinite(loss.detach()) and not self.scaler.is_enabled():
                # fp16 runs: the GradScaler detects inf/NaN gradients and skips the update instead
                raise FloatingPointError(f"non-finite training loss {loss.item()}")
            self.scaler.scale(loss * w).backward()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * w
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v) * w
        self.scaler.unscale_(self.opt)  # unscale BEFORE clipping (no-op unless fp16)
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
        lr = self.opt.param_groups[0]["lr"]
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()  # exactly once per optimizer update
        totals["grad_norm"] = float(gn)
        totals["lr"] = lr
        return totals


# ---------------------------------------------------------------------------
# Run-level helpers
# ---------------------------------------------------------------------------

def run_dir(cfg: ExperimentConfig, model_id: str) -> Path:
    return cfg.experiment_dir / "runs" / model_id


def model_config_for(cfg: ExperimentConfig, model_id: str):
    """Model config used for a run: registry default + optional overrides, with the experiment's
    operation-slot count (legacy 3 is the registry default; mixed-length experiments use 5)."""
    import dataclasses
    from .models.registry import default_config
    base = dataclasses.asdict(default_config(model_id))
    base.update(cfg.model_overrides.get(model_id) or {})
    if cfg.n_slots != 3:
        base["n_slots"] = cfg.n_slots
    from .idvar import condition_of, is_idvar
    if is_idvar(cfg.experiment):
        # identical parameter shapes in BOTH conditions: <identity> in the vocabulary, max-size position tables,
        # the variable-length CNN conditioner; only the accepted instruction format differs.
        base["op_format"] = "variable" if condition_of(cfg.experiment) == "variable_length" else "fixed"
        if "identity_token" in base:
            base["identity_token"] = True
        if "conditioner" in base:
            base["conditioner"] = "set_mlp"
        return base
    from .mixed import CNN_MODELS, is_compact
    if is_compact(cfg.experiment):
        pe = cfg.sequence_pe
        if model_id in CNN_MODELS:
            if pe is not None:
                raise ValueError("CNN controls have no attention PE switch; train them in compact_02345_cnn")
        else:
            if pe is None:
                raise ValueError("compact_02345_cnn holds only the shared CNN controls")
            base["sequence_pe"] = pe
    return base


def run_config(cfg: ExperimentConfig, model_id: str, manifest_hash: str) -> dict:
    m = build_model(model_id, model_config_for(cfg, model_id))  # CPU, only for config/parameter accounting
    mcfg, n_params = config_dict(m), count_parameters(m)
    del m
    d = {
        "model_id": model_id,
        "model_config": mcfg,
        "model_params": n_params,
        "experiment": cfg.experiment,
        "train_strings": cfg.train_strings,
        "train": cfg.train.semantic_dict(),
        "data_version": DATA_VERSION,
        "manifest_hash": manifest_hash,
        "n_val_grids": cfg.n_val_grids,
        "val_subset_grids": cfg.val_subset_grids,
        "checkpoint_format": CHECKPOINT_FORMAT,
    }
    if cfg.is_mixed:  # extra keys ONLY for the new experiments, so legacy config hashes are unchanged
        from .mixed import LENGTHS, MIXED_DATA_VERSION
        d["data_version"] = MIXED_DATA_VERSION
        d["n_slots"] = cfg.n_slots
        d["length_schedule"] = list(LENGTHS)
        d["checkpoint_selection"] = "mean final-answer cell CE over the 5 equal-size validation length groups"
    from .idvar import IDVAR_DATA_VERSION, K_VALUES, condition_of, is_idvar
    if is_idvar(cfg.experiment):
        d["data_version"] = IDVAR_DATA_VERSION
        d["n_slots"] = cfg.n_slots
        d["condition"] = condition_of(cfg.experiment)
        d["k_schedule"] = {"k_values": list(K_VALUES), "per_update": "51 per k + 1 rotating extra",
                           "words": "uniform over {A,B,C}^k", "slots": "uniform ordered k-subset of 5"}
        d["positions"] = "learned absolute 1D positions (identity: physical slots; variable: contiguous)"
        d["checkpoint_selection"] = "mean final-answer digit CE over the 5 equal-size validation k groups"
    from .mixed import COMPACT_DATA_VERSION, LAYOUT_VERSION, is_compact
    if is_compact(cfg.experiment):
        d["data_version"] = COMPACT_DATA_VERSION
        d["placement"] = "compact_left"
        d["layout_version"] = LAYOUT_VERSION
        d["sequence_pe"] = cfg.sequence_pe or "n/a (CNN control)"
        d["pe_convention"] = ("rope: rotate_half pairing (i, i+hd/2), theta 10000, full head dim, Q/K only, every attention "
                              "call; sinusoidal: fixed sin/cos base 10000 added once at input, scale 1; shared additive "
                              "row/col (or patch row/col) + segment/type embeddings in both")
    d["config_hash"] = stable_hash(d)
    d["model_hash"] = stable_hash({"model_id": model_id, "model_config": mcfg, "params": n_params})
    return d


def _rng_state() -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(s["cuda"])
        except Exception:
            pass


def _write_logs(rdir: Path, logs: dict) -> None:
    import pandas as pd
    if logs["train"]:
        atomic_write_text(rdir / "train_log.csv", pd.DataFrame(logs["train"]).to_csv(index=False))
    if logs["val"]:
        atomic_write_text(rdir / "val_log.csv", pd.DataFrame(logs["val"]).to_csv(index=False))


def training_complete(cfg: ExperimentConfig, model_id: str, rc: dict) -> bool:
    rdir = run_dir(cfg, model_id)
    f = rdir / "TRAINING_COMPLETE.json"
    if not f.exists() or not (rdir / "best.pt").exists():
        return False
    import json
    info = json.loads(f.read_text())
    if info.get("config_hash") != rc["config_hash"]:
        raise RuntimeError(f"{rdir} contains a completed run with a DIFFERENT configuration "
                           f"({info.get('config_hash')} != {rc['config_hash']}). Use another output directory.")
    return True


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_model(model_id: str, cfg: ExperimentConfig, manifests, val_set, subset_set,
                log: Callable[[str], None] = print, repo_dir: Optional[str] = None,
                stop_after_steps: Optional[int] = None) -> dict:
    """Train one model for the full configured budget (resuming if a checkpoint exists).
    ``stop_after_steps`` optionally ends THIS session early after that many optimizer steps
    (at a checkpoint boundary is best); rerunning resumes from the last saved checkpoint."""
    tc = cfg.train
    rdir = run_dir(cfg, model_id)
    rc = run_config(cfg, model_id, manifests.manifest_hash)
    if training_complete(cfg, model_id, rc):
        log(f"[{model_id}] training already complete with matching config -> skipping")
        import json
        return json.loads((rdir / "train_summary.json").read_text())

    device = resolve_device(tc.device)
    amp_dtype = resolve_amp_dtype(device, tc.dtype)
    if device.type == "cuda" and tc.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    state_path = rdir / "state.pt"
    if state_path.exists() and not cfg.resume:
        raise RuntimeError(f"{state_path} exists but resume=False. Set RESUME=True to continue it, "
                           "or choose a new output directory (existing runs are never silently overwritten).")

    model = build_model(model_id, rc["model_config"], seed=tc.seed).to(device)
    if getattr(model.cfg, "n_slots", 3) != cfg.n_slots:
        raise RuntimeError(f"model has {model.cfg.n_slots} operation slots but the experiment uses {cfg.n_slots}")
    model.activation_checkpointing = tc.activation_checkpointing
    for mod in model.modules():
        if hasattr(mod, "activation_checkpointing"):
            mod.activation_checkpointing = tc.activation_checkpointing
    opt = make_optimizer(model, tc)
    sched = make_scheduler(opt, tc)
    scaler = make_scaler(amp_dtype)

    logs = {"train": [], "val": []}
    global_step, best_ce, best_step, elapsed_prev, oom_total = 0, float("inf"), None, 0.0, 0
    if state_path.exists():
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        if ck.get("sequence_pe", "learned") != rc["model_config"].get("sequence_pe", "learned"):
            raise RuntimeError(f"cannot resume {state_path}: checkpoint PE {ck.get('sequence_pe', 'learned')!r} != "
                               f"{rc['model_config'].get('sequence_pe', 'learned')!r}")
        if ck.get("n_slots", 3) != cfg.n_slots:
            raise RuntimeError(f"cannot resume {state_path}: checkpoint has {ck.get('n_slots', 3)} operation slots, "
                               f"experiment uses {cfg.n_slots}")
        for key in ("config_hash", "model_hash", "data_version", "manifest_hash"):
            want = rc[key] if key in rc else DATA_VERSION
            if ck.get(key) != want:
                raise RuntimeError(f"cannot resume {state_path}: {key} mismatch ({ck.get(key)} != {want})")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        global_step, best_ce, best_step = ck["global_step"], ck["best_val_cell_ce"], ck["best_step"]
        logs, elapsed_prev, oom_total = ck["logs"], ck.get("train_seconds", 0.0), ck.get("oom_retries", 0)
        log(f"[{model_id}] resumed from step {global_step} (best val CE {best_ce:.4f} @ {best_step})")

    rdir.mkdir(parents=True, exist_ok=True)
    env = environment_info(repo_dir)
    atomic_write_json(rdir / "run_config.json", {**rc, "full_experiment_config": cfg.to_dict(), "environment": env})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mb = probe_microbatch(model, tc, device, amp_dtype, log)
    if cfg.is_idvar:
        from .idvar import IdVarStream, condition_of
        stream = IdVarStream(tc.seed, tc.effective_batch_size, condition_of(cfg.experiment), manifests.heldout_bytes)
    elif cfg.is_mixed:
        from .mixed import MixedTrainStream
        stream = MixedTrainStream(tc.seed, cfg.experiment, tc.effective_batch_size, manifests.heldout_bytes)
    else:
        stream = TrainStream(tc.seed, cfg.train_strings, tc.effective_batch_size, manifests.heldout_bytes)
    runner = StepRunner(model, opt, sched, scaler, tc, device, amp_dtype, mb, log)
    n_params = count_parameters(model)
    log(f"[{model_id}] {n_params:,} params | device {hardware_name(device)} | dtype {dtype_name(amp_dtype)} | "
        f"microbatch {mb} x accumulation {math.ceil(tc.effective_batch_size / mb)} = {tc.effective_batch_size}")

    def save_state():
        ck = {
            "format": CHECKPOINT_FORMAT, "model_id": model_id, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "global_step": global_step, "rng": _rng_state(),
            "data_stream_next_step": global_step,  # stateless stream: next logical batch index
            "best_val_cell_ce": best_ce, "best_step": best_step, "n_slots": cfg.n_slots,
            "sequence_pe": rc["model_config"].get("sequence_pe", "learned"),
            "config_hash": rc["config_hash"], "model_hash": rc["model_hash"], "data_version": rc["data_version"],
            "manifest_hash": manifests.manifest_hash, "logs": logs,
            "train_seconds": elapsed_prev + (time.time() - t0), "oom_retries": oom_total + runner.oom_retries,
        }
        atomic_torch_save(ck, state_path)
        _write_logs(rdir, logs)

    def validate():
        nonlocal best_ce, best_step
        tv = time.time()
        r = run_model_on_set(model, val_set, device, amp_dtype, tc.eval_batch_size, want_ce=True, want_pred=False)
        s = run_model_on_set(model, subset_set, device, amp_dtype, tc.eval_batch_size, want_ce=False, want_pred=True)
        per_len = {}
        if "length" in val_set:  # mixed-length: select on the mean over the equal-size length groups
            for L in np.unique(val_set["length"]):
                per_len[f"val_cell_ce_len{int(L)}"] = float(r["cell_ce"][val_set["length"] == L].mean())
            ce = float(np.mean(list(per_len.values())))
            ne = val_set["length"] > 0
            per_len["val_cell_ce_nonempty"] = float(r["cell_ce"][ne].mean())
            for L in np.unique(subset_set["length"]):
                sel = subset_set["length"] == L
                per_len[f"subset_exact_len{int(L)}"] = float(s["exact"][sel].mean())
        else:
            ce = float(r["cell_ce"].mean())
        entry = {"step": global_step, "examples_seen": global_step * tc.effective_batch_size,
                 "val_cell_ce": ce, **per_len,
                 "val_cell_ce_label": (("length-macro " if per_len else "") +
                                       ("teacher-forced answer-DIGIT CE (EOS excluded)" if model.kind == "ar" else
                                       "final-refinement cell CE" if model.kind == "trm" else "cell CE")),
                 "subset_exact_acc": float(s["exact"].mean()), "subset_cell_acc": float(s["cell_correct"].mean() / 36),
                 "subset_valid_rate": float(s["valid"].mean()), "n_val_examples": len(val_set["x"]),
                 "n_subset_examples": len(subset_set["x"]), "val_seconds": time.time() - tv}
        improved = ce < best_ce  # strict: ties keep the EARLIER checkpoint
        if improved:
            best_ce, best_step = ce, global_step
            atomic_torch_save({"model": model.state_dict(), "step": global_step, "val_cell_ce": ce,
                               "config_hash": rc["config_hash"], "model_hash": rc["model_hash"],
                               "model_config": rc["model_config"], "model_id": model_id, "n_slots": cfg.n_slots},
                              rdir / "best.pt")
        entry["is_best"] = improved
        logs["val"].append(entry)
        log(f"  [val @ {global_step}] cell CE {ce:.4f}{' (best)' if improved else ''} | "
            f"subset({entry['n_subset_examples']}) exact {entry['subset_exact_acc']:.3f} "
            f"cell {entry['subset_cell_acc']:.3f}")

    t0 = time.time()
    model.train()
    last_print = time.time()
    steps_this_session = 0
    while global_step < tc.optimizer_steps:
        batch = stream.batch(global_step)
        metrics = runner(batch)
        global_step += 1
        steps_this_session += 1
        if global_step % tc.log_every == 0 or global_step == tc.optimizer_steps or global_step == 1:
            row = {"step": global_step, "examples_seen": global_step * tc.effective_batch_size, **metrics,
                   "loss_label": model.loss_label, "elapsed_s": elapsed_prev + time.time() - t0}
            logs["train"].append(row)
            if time.time() - last_print > 20 or global_step == tc.optimizer_steps or tc.log_every == 1:
                rate = steps_this_session / max(1e-9, time.time() - t0)
                eta = (tc.optimizer_steps - global_step) / max(rate, 1e-9)
                log(f"  step {global_step}/{tc.optimizer_steps} loss {metrics['loss']:.4f} lr {metrics['lr']:.2e} "
                    f"| {rate:.2f} steps/s, {rate * tc.effective_batch_size:.0f} ex/s | ETA {eta / 60:.1f} min")
                last_print = time.time()
        if global_step % tc.val_every == 0 or global_step == tc.optimizer_steps:
            validate()
            save_state()
        if stop_after_steps is not None and steps_this_session >= stop_after_steps and global_step < tc.optimizer_steps:
            save_state()
            log(f"[{model_id}] stopping this session at step {global_step} (stop_after_steps); rerun to resume")
            del model, opt, sched, scaler, runner
            free_memory()
            return {"model_id": model_id, "interrupted_at_step": global_step}

    train_seconds = elapsed_prev + time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    summary = {
        **{k: rc[k] for k in ("model_id", "config_hash", "model_hash", "experiment")},
        "unique_params": n_params, "optimizer_steps": global_step,
        "examples_seen": global_step * tc.effective_batch_size, "effective_batch_size": tc.effective_batch_size,
        "microbatch_size": runner.mb, "accumulation_steps": math.ceil(tc.effective_batch_size / runner.mb),
        "oom_retries": oom_total + runner.oom_retries, "best_step": best_step, "best_val_cell_ce": best_ce,
        "final_val_cell_ce": logs["val"][-1]["val_cell_ce"] if logs["val"] else None,
        "train_seconds": train_seconds, "peak_gpu_memory_gb_this_session": peak,
        "hardware": hardware_name(device), "dtype": dtype_name(amp_dtype), "seed": tc.seed,
        "repo_commit": env["repo_commit"], "torch": env["torch"],
        "last_checkpoint": str(state_path), "best_checkpoint": str(rdir / "best.pt"),
    }
    atomic_write_json(rdir / "train_summary.json", summary)
    atomic_write_json(rdir / "TRAINING_COMPLETE.json", {"config_hash": rc["config_hash"], "step": global_step})
    del model, opt, sched, scaler, runner
    free_memory()
    return summary


def profile_model(model_id: str, tc: TrainConfig, n_steps: int = 3, log=print, n_slots: int = 3,
                  sequence_pe: str = None) -> dict:
    """Short timing run on DUMMY data (does not touch the training stream) to estimate
    the full-run time. Returns seconds per optimizer step at the probed microbatch."""
    device = resolve_device(tc.device)
    amp_dtype = resolve_amp_dtype(device, tc.dtype)
    import dataclasses
    from .models.registry import default_config
    mc = dataclasses.asdict(default_config(model_id))
    if n_slots != 3:
        mc["n_slots"] = n_slots
    if sequence_pe and "sequence_pe" in mc:
        mc["sequence_pe"] = sequence_pe
    model = build_model(model_id, mc, seed=tc.seed).to(device)
    opt = make_optimizer(model, tc)
    mb = probe_microbatch(model, tc, device, amp_dtype, log=lambda *_: None)
    batch = model.dummy_batch(tc.effective_batch_size, device)
    batch_np = {k: v.cpu().numpy() for k, v in batch.items()}
    runner = StepRunner(model, opt, make_scheduler(opt, tc), make_scaler(amp_dtype), tc, device, amp_dtype, mb, log)
    runner(batch_np)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t = time.time()
    for _ in range(n_steps):
        runner(batch_np)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.time() - t) / n_steps
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    del model, opt, runner
    free_memory()
    return {"model_id": model_id, "sec_per_step": sec, "microbatch": mb,
            "est_train_minutes": sec * tc.optimizer_steps / 60, "peak_gpu_gb": peak}
