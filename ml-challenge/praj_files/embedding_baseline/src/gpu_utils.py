from __future__ import annotations

import gc
import importlib.util
import subprocess
from typing import Any


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def gpu_memory() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(
                available=True,
                name=torch.cuda.get_device_name(0),
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                free_bytes=free,
                total_bytes=total,
                bf16_supported=torch.cuda.is_bf16_supported(),
                capability=list(torch.cuda.get_device_capability()),
            )
    except Exception as exc:
        result["error"] = str(exc)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if proc.returncode == 0:
            result["utilization_percent"] = int(proc.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return result


def environment_report() -> dict[str, Any]:
    import torch
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_memory(),
        "vllm_installed": has_module("vllm"),
        "cuvs_installed": has_module("cuvs"),
        "faiss_installed": has_module("faiss"),
    }
    if report["faiss_installed"]:
        import faiss
        report["faiss_version"] = faiss.__version__
        report["faiss_gpu_count"] = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    return report

