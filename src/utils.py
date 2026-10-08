"""Общие утилиты: метрики, сиды, логирование, поиск корня."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score

import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    """Ищет корень проекта (папку с src/) вверх по дереву."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "src").exists():
            return parent
    return current


def ensure_root_on_path() -> Path:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def rmsle(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def get_scorer(task: str):
    """Возвращает sklearn-совместимый scorer для задачи."""

    if task == "classification":
        return make_scorer(accuracy_score)
    if task == "regression":
        from src.utils import rmse
        return make_scorer(rmse, greater_is_better=False)
    raise ValueError(f"Unknown task: {task}")


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)