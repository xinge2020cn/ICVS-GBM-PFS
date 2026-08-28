"""Determinism controls and environment provenance."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without altering external validation data."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def environment_report() -> dict[str, object]:
    """Return the software and compute environment used for one run."""

    packages = {}
    for name in (
        "icvs-gbm-pfs",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-survival",
        "lifelines",
        "nibabel",
        "SimpleITK",
        "torch",
        "pyradiomics",
        "nnunetv2",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    return report


def write_environment_report(path: str | Path) -> None:
    """Write a machine-readable environment record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(environment_report(), indent=2) + "\n", encoding="utf-8")
