"""Hardware checks shared by commands that load a language model."""

import os
import platform
from typing import Any


def hardware_summary(torch_module: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "cuda_available": torch_module.cuda.is_available(),
        "mps_available": bool(
            getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available()
        ),
        "host_memory_bytes": _host_memory_bytes(),
    }
    if summary["cuda_available"]:
        summary["gpu_name"] = torch_module.cuda.get_device_name(0)
        summary["gpu_memory_bytes"] = torch_module.cuda.get_device_properties(0).total_memory
        summary["cuda_version"] = torch_module.version.cuda
        summary["bf16_supported"] = torch_module.cuda.is_bf16_supported()
    return summary


def _host_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def require_cuda_for_qlora(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "QLoRA training in this project requires a CUDA-enabled NVIDIA GPU. "
            "This machine reports no CUDA device; use a cloud GPU runtime and run `make smoke`."
        )
