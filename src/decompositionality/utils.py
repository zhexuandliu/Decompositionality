"""Small utilities: atomic file IO, hashing, device/dtype selection, environment info."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch


def atomic_write_bytes(path: os.PathLike, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path: os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(obj), indent=1, sort_keys=False))


def atomic_torch_save(obj: Any, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(obj), sort_keys=True).encode()).hexdigest()[:16]


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_amp_dtype(device: torch.device, dtype: str = "auto") -> Optional[torch.dtype]:
    """None means full fp32 (no autocast)."""
    if dtype == "fp32" or device.type != "cuda":
        return None  # CPU/MPS smoke paths run in fp32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def dtype_name(dt: Optional[torch.dtype]) -> str:
    return {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[dt]


def git_commit(repo_dir: Optional[os.PathLike] = None) -> str:
    repo_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        commit = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def environment_info(repo_dir: Optional[os.PathLike] = None) -> dict:
    import numpy
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "repo_commit": git_commit(repo_dir),
    }
    for mod in ("pandas", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["gpu_total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    else:
        info["gpu"] = None
    return info


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type} ({platform.processor() or platform.machine()})"


def free_memory() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
