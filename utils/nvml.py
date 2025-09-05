# grpo_single_policy_prm/utils/nvml.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


@dataclass
class GPUSnapshot:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    util_gpu_pct: int
    util_mem_pct: int
    power_watts: float | None = None


def _bytes_to_mb(x: int) -> int:
    return int(x / (1024 * 1024))


def snapshot_all() -> list[GPUSnapshot]:
    if not _NVML_AVAILABLE:
        return []
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out: list[GPUSnapshot] = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            power = None
            try:
                power = float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
            except Exception:
                power = None
            out.append(
                GPUSnapshot(
                    index=i,
                    name=name,
                    memory_total_mb=_bytes_to_mb(mem.total),
                    memory_used_mb=_bytes_to_mb(mem.used),
                    util_gpu_pct=int(util.gpu),
                    util_mem_pct=int(util.memory),
                    power_watts=power,
                )
            )
        pynvml.nvmlShutdown()
        return out
    except Exception:
        # fail silently if NVML misbehaves
        return []
