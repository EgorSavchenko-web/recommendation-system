from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def setup_logging(log_file=None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=handlers, force=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def require_gpu_on_agent(task, device: torch.device, allow_cpu: bool) -> None:
    if task is not None and device.type != "cuda" and not allow_cpu:
        raise RuntimeError("This ClearML run got no GPU; requeue it or pass --allow-cpu.")


def save_json(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """No decay on biases, norms and embedding tables."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name or "wpe" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def cosine_with_warmup(optimizer, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
