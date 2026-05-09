"""GPU memory footprint tracker for FVM-PINN trainers.

Records ``torch.cuda.memory_allocated`` / ``max_memory_allocated`` /
``memory_reserved`` at training milestones (every ``log_every`` epochs).
Outputs:

* a parseable line in stdout (and therefore stdout.log):
  ``MEM  phase=adam  step=500  wall=42.18s  alloc=412.3 MiB  peak=687.4 MiB  reserved=712.0 MiB``
* a structured JSON dump at the end of training, written next to the run's
  checkpoints, so downstream plotting scripts can read a single file.

The tracker is **shared by reference** across sub-trainers (e.g. the
per-window trainers managed by ``TimeWindowTrainer``) so every recorded
sample is on the same global wall-clock origin. Peak stats are reset at
the start of each phase rather than at every sample, which gives the
"peak since this phase started" semantics that's most useful when
comparing strategies.

CPU-only runs degrade gracefully — ``sample()`` returns ``None`` and the
JSON ends up empty. No CUDA-specific code runs unless ``device.type``
is ``"cuda"``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Lightweight per-step GPU memory recorder.

    Parameters
    ----------
    device : torch.device
        The device whose memory we track. CPU devices are no-ops.
    label : str, optional
        A short tag (e.g. run id) that gets prefixed to every log line.
    """

    def __init__(self, device: torch.device, label: str = "") -> None:
        self.device = device
        self.label = label
        self.is_cuda = device.type == "cuda"
        self.start_wall = time.perf_counter()
        self.records: List[Dict] = []
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

    def reset_peak(self, phase_label: str = "") -> None:
        """Clear ``max_memory_allocated`` so subsequent ``peak_mib`` readings
        reflect this phase only. Call at the start of Adam, L-BFGS, or each
        new window."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            if phase_label:
                logger.debug(f"MEM peak reset at phase={phase_label}")

    def sample(
        self,
        phase: str,
        step: int,
        *,
        log: bool = True,
    ) -> Optional[Dict]:
        """Record a sample. Returns the record dict, or None on CPU."""
        if not self.is_cuda:
            return None
        wall = time.perf_counter() - self.start_wall
        alloc_mib = torch.cuda.memory_allocated(self.device) / 1024 ** 2
        peak_mib = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        reserved_mib = torch.cuda.memory_reserved(self.device) / 1024 ** 2
        rec = {
            "phase":        phase,
            "step":         int(step),
            "wall_s":       round(wall, 3),
            "alloc_mib":    round(alloc_mib, 1),
            "peak_mib":     round(peak_mib, 1),
            "reserved_mib": round(reserved_mib, 1),
        }
        self.records.append(rec)
        if log:
            prefix = f"[{self.label}] " if self.label else ""
            logger.info(
                f"{prefix}MEM  phase={phase}  step={step}  wall={wall:.1f}s  "
                f"alloc={alloc_mib:.1f} MiB  peak={peak_mib:.1f} MiB  "
                f"reserved={reserved_mib:.1f} MiB"
            )
        return rec

    def save_json(self, path: Path) -> None:
        """Persist the recorded samples to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({
                "device":  str(self.device),
                "label":   self.label,
                "records": self.records,
            }, f, indent=2)
        logger.info(f"Memory history JSON: {path}")
