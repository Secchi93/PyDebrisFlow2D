from __future__ import annotations

"""Execution-backend selection and runtime diagnostics."""

from dataclasses import dataclass
from typing import Optional

from .config import SolverConfig


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    selected: str
    cuda_available: bool
    cuda_device_name: Optional[str]
    reason: str


def cuda_runtime_info(device_index: int = 0) -> tuple[bool, Optional[str], str]:
    """Return CUDA availability without requiring a GPU at package import time."""
    try:
        from numba import cuda
    except Exception as exc:  # pragma: no cover - depends on optional CUDA target
        return False, None, f"Numba CUDA import failed: {exc}"
    try:
        if not cuda.is_available():
            return False, None, "No working CUDA driver/device was detected"
        devices = list(cuda.gpus)
        if device_index >= len(devices):
            return False, None, f"CUDA device index {device_index} is out of range ({len(devices)} devices)"
        if bool(getattr(cuda.config, "ENABLE_CUDASIM", False)):
            return True, "CUDA simulator", "CUDA simulator enabled"
        with devices[device_index]:
            dev = cuda.current_context().device
            raw_name = getattr(dev, "name", f"CUDA device {device_index}")
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        return True, name, "CUDA runtime available"
    except Exception as exc:
        return False, None, f"CUDA runtime initialization failed: {exc}"


def resolve_backend(cfg: SolverConfig) -> BackendInfo:
    requested = str(cfg.compute.backend).strip().lower()
    available, name, detail = cuda_runtime_info(int(cfg.compute.cuda_device))
    if requested == "cpu":
        return BackendInfo(requested, "cpu", available, name, "CPU backend explicitly requested")
    if requested == "cuda":
        if available:
            return BackendInfo(requested, "cuda", True, name, detail)
        if cfg.compute.cuda_allow_fallback:
            return BackendInfo(requested, "cpu", False, None, detail + "; falling back to CPU")
        raise RuntimeError(detail + "; set compute.cuda_allow_fallback=true or select backend=cpu")
    if available:
        return BackendInfo(requested, "cuda", True, name, "Auto-selected CUDA")
    return BackendInfo(requested, "cpu", False, None, detail + "; auto-selected CPU")


def configure_cpu_threads(cfg: SolverConfig) -> int:
    """Apply the requested Numba CPU thread count and return the active count."""
    try:
        import numba
    except Exception:  # pragma: no cover
        return 1
    requested = int(cfg.compute.cpu_threads)
    if requested > 0:
        numba.set_num_threads(requested)
    return int(numba.get_num_threads())
