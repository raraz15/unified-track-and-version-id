import os
import random
import subprocess
import time
import sys
import logging
from pathlib import Path
import logging
from multiprocessing import Queue
from logging.handlers import QueueHandler, QueueListener

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 27  # License plate code of Gaziantep, gastronomical capital of Türkiye


def set_seed(seed: int = SEED) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def s_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_git_commit(repo_root: str | Path = REPO_ROOT) -> str:
    """Return current git commit hash, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except:
        return "unknown"


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def sec_to_hms(t0: float, measure: bool, t1: float | None = None) -> tuple[float, str]:
    if measure:
        if t1 is None:
            t1 = time.monotonic()
        t = t1 - t0
    else:
        t = t0
    return t, time.strftime("%H:%M:%S", time.gmtime(t))


def setup_logging(log_path: Path):
    """Configure the root logger to send records into a multiprocessing.Queue."""
    log_queue = Queue(-1)  # infinite size
    # Handlers that will actually write to file and console, run in main process:
    file_h = logging.FileHandler(log_path, mode="a")
    console_h = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | PID:%(process)d | %(levelname)-8s | %(name)-25s | %(message)s"
    )
    file_h.setFormatter(formatter)
    console_h.setFormatter(formatter)

    # QueueListener will pull records off the queue and send them to handlers:
    listener = QueueListener(log_queue, file_h, console_h, respect_handler_level=True)
    listener.start()

    # In *all* processes, the root logger uses a QueueHandler:
    queue_handler = QueueHandler(log_queue)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []  # drop any default handlers
    root.addHandler(queue_handler)

    return listener


def is_global_zero() -> bool:
    """Return True if running on distributed rank 0 or non-distributed."""
    for env_var in ("GLOBAL_RANK", "RANK"):
        rank_value = os.environ.get(env_var)
        if rank_value is not None:
            try:
                return int(rank_value) == 0
            except ValueError:
                break

    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank() == 0
        except RuntimeError:
            return True
    return True
