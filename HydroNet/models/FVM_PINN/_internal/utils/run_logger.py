"""
Shared logging setup for experiment run scripts.

Adds a timestamped file handler to the root logger so that all INFO+
messages are duplicated to a log file in the output directory.
Also logs system/GPU info at the start and timing summary at the end.
"""

import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def setup_run_logging(output_dir: str, device: str = "cpu") -> Path:
    """Add a timestamped file handler to the root logger.

    Args:
        output_dir: Directory where the log file will be saved.
        device: Device string ("cpu" or "cuda").

    Returns:
        Path to the log file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_path / f"run_{timestamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

    # Log system info
    logger.info(f"Log file: {log_path}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Device: {device}")

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            idx = int(device.split(":")[1])
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.info("GPU: N/A (running on CPU)")

    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return log_path


def log_run_end(start_time: float) -> None:
    """Log end time and total elapsed time.

    Args:
        start_time: Value from time.time() at the start of the run.
    """
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {int(hours)}h {int(mins)}m {secs:.1f}s ({elapsed:.1f}s)")
