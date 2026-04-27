"""Hardware detection + unified real-time monitoring.

Single responsibility: know what's available and report utilization.
  - CPU cores (count + per-core usage)
  - RAM (system + process)
  - GPU count + per-GPU utilization
  - GPU memory (used/total per card)
  - Disk (free/total at save_dir)

Exposes:
  detect(cfg)                   -> Resources snapshot
  Monitor(cfg).tick(tag)        -> one-line status print
  Monitor(cfg).guard(tag)       -> auto GC if near limit
"""

import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class Resources:
    num_cpus: int
    num_gpus: int
    gpu_ids: List[int]
    primary_device: torch.device
    multi_gpu: bool

    def __str__(self):
        gstr = ",".join(str(i) for i in self.gpu_ids) if self.gpu_ids else "none"
        return (f"CPUs={self.num_cpus}  GPUs={self.num_gpus}[{gstr}]  "
                f"primary={self.primary_device}  multi_gpu={self.multi_gpu}")


def detect(cfg) -> Resources:
    cpu_count = os.cpu_count() or 1
    if getattr(cfg, "num_workers", -1) > 0:
        num_cpus = min(cfg.num_workers, cpu_count)
    else:
        num_cpus = cpu_count

    cuda_ok = torch.cuda.is_available()
    device_str = getattr(cfg, "device", "auto")

    if device_str == "cpu":
        num_gpus = 0
        gpu_ids: List[int] = []
        primary = torch.device("cpu")
    elif cuda_ok:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        primary = torch.device(device_str if device_str.startswith("cuda:") else "cuda:0")
    elif _HAS_XLA:
        primary = xm.xla_device()
        num_gpus = 0
        gpu_ids = []
        try:
            n_chips = xm.xrt_world_size()
        except Exception:
            n_chips = 1
        print(f"  TPU detected: {primary}  chips={n_chips}")
    else:
        num_gpus = 0
        gpu_ids = []
        primary = torch.device("cpu")

    multi_gpu = cfg.multi_gpu and num_gpus > 1
    return Resources(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        primary_device=primary,
        multi_gpu=multi_gpu,
    )


def setup_cuda_tuning():
    """Enable cudnn.benchmark + TF32."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Snapshots
# ==========================================================================

def ram_snapshot() -> Dict:
    if not _HAS_PSUTIL:
        return {"available": False}
    m = psutil.virtual_memory()
    p = psutil.Process()
    return {
        "available": True,
        "rss_gb": p.memory_info().rss / 1e9,
        "sys_used_gb": (m.total - m.available) / 1e9,
        "sys_available_gb": m.available / 1e9,
        "sys_total_gb": m.total / 1e9,
        "percent": m.percent,
    }


def gpu_snapshot() -> List[Dict]:
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        util = 0
        try:
            util = torch.cuda.utilization(i)  # needs pynvml on some versions
        except Exception:
            util = -1
        out.append({
            "gpu": i,
            "used_gb": (total - free) / 1e9,
            "total_gb": total / 1e9,
            "mem_percent": (1.0 - free / total) * 100.0,
            "util_percent": util,
        })
    return out


def cpu_snapshot() -> Dict:
    if not _HAS_PSUTIL:
        return {"available": False}
    try:
        pct = psutil.cpu_percent(interval=None)
        per = psutil.cpu_percent(interval=None, percpu=True)
    except Exception:
        pct, per = -1, []
    return {"available": True, "percent": pct, "per_core": per}


def disk_snapshot(path: str = ".") -> Dict:
    try:
        du = shutil.disk_usage(path)
        return {
            "available": True,
            "total_gb": du.total / 1e9,
            "used_gb": du.used / 1e9,
            "free_gb": du.free / 1e9,
            "percent": du.used / max(du.total, 1) * 100.0,
        }
    except Exception:
        return {"available": False}


# ==========================================================================
# Monitor — real-time status printer with auto-GC guard
# ==========================================================================

class Monitor:
    def __init__(self, cfg):
        self.log_every = max(1, getattr(cfg, "mem_log_every", 100))
        self.warn_pct = float(getattr(cfg, "mem_warn_percent", 88.0))
        self.save_dir = getattr(cfg, "save_dir", ".")
        self.step = 0
        self.peak_rss_gb = 0.0
        self.peak_gpu_gb = 0.0

    def tick(self, tag: str = "", force: bool = False):
        """Print one-line snapshot if cadence hit or force=True."""
        self.step += 1
        if not force and self.step % self.log_every != 0:
            return
        self._print(tag)

    def snapshot(self, tag: str = ""):
        """Always print; independent of cadence."""
        self._print(tag)

    def guard(self, tag: str = ""):
        """Check memory, GC if near the warn threshold."""
        r = ram_snapshot()
        if not r.get("available"):
            return
        if r["percent"] > self.warn_pct:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [guard{(' '+tag) if tag else ''}] sys mem {r['percent']:.0f}% — GC + empty_cache fired")

    def _print(self, tag: str):
        parts = []
        r = ram_snapshot()
        if r.get("available"):
            self.peak_rss_gb = max(self.peak_rss_gb, r["rss_gb"])
            parts.append(
                f"RAM={r['rss_gb']:.1f}GB sys={r['sys_used_gb']:.1f}/{r['sys_total_gb']:.1f}GB ({r['percent']:.0f}%)"
            )
        c = cpu_snapshot()
        if c.get("available"):
            parts.append(f"CPU={c['percent']:.0f}%")
        for g in gpu_snapshot():
            self.peak_gpu_gb = max(self.peak_gpu_gb, g["used_gb"])
            u = g["util_percent"]
            u_str = f"/{u}%util" if u >= 0 else ""
            parts.append(
                f"GPU{g['gpu']}={g['used_gb']:.1f}/{g['total_gb']:.1f}GB"
                f"({g['mem_percent']:.0f}%{u_str})"
            )
        d = disk_snapshot(self.save_dir)
        if d.get("available"):
            parts.append(f"disk_free={d['free_gb']:.1f}GB")
        if parts:
            print(f"  [mem{(' '+tag) if tag else ''}] " + "  ".join(parts))
