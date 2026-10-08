"""Обучение: CV, финальный fit, (опционально) Optuna."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
)

from src.models import build_model, train_dnn
from src.utils import get_scorer, log, rmse


def _cv_splitter(config: dict[str, Any]):
    folds = config["training"]["cv_folds"]
    seed = config["training"]["random_state"]
    if config["task"] == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate(model, X, y, config) -> tuple[float, float]:
    scorer = get_scorer(config["task"])
    scores = cross_val_score(model, X, y, cv=_cv_splitter(config), scoring=scorer)
    if config["task"] == "regression":
        scores = -scores   # sklearn инвертирует RMSE - возвращаем на место (а то там отрицателный скор получается почему-то)
    return float(scores.mean()), float(scores.std())


def train_classic(X_train, y_train, X_val, y_val, config) -> tuple[Any, dict]:
    """Обучает классическую модель (LGBM/CatBoost/...)."""
    model_cfg = config["model"]
    model = build_model(model_cfg["name"], model_cfg["params"], config["task"])

    cv_mean, cv_std = cross_validate(model, X_train, y_train, config)
    log(f"CV {config['training']['scoring']}: {cv_mean:.4f} ± {cv_std:.4f}")

    model.fit(X_train, y_train)
    if X_val is not None and len(X_val) > 0:
        val_pred = model.predict(X_val)
        if config["task"] == "classification":
            from sklearn.metrics import accuracy_score
            log(f"Holdout accuracy: {accuracy_score(y_val, val_pred):.4f}")
        else:
            log(f"Holdout RMSE: {rmse(y_val, val_pred):.4f}")

    return model, {"cv_mean": cv_mean, "cv_std": cv_std}


def train_dnn_model(X_train, y_train, X_val, y_val, config) -> tuple[Any, dict]:
    """Обучает DNN с early stopping."""
    dnn_cfg = config["model"].get("dnn", {})
    model, info = train_dnn(
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train).reshape(-1),
        X_val=np.asarray(X_val),
        y_val=np.asarray(y_val).reshape(-1),
        hidden_sizes=dnn_cfg.get("hidden_sizes", [64, 32]),
        epochs=dnn_cfg.get("epochs", 500),
        batch_size=dnn_cfg.get("batch_size", 32),
        lr=dnn_cfg.get("lr", 1e-3),
        patience=dnn_cfg.get("patience", 20),
        dropout=dnn_cfg.get("dropout", 0.2),
        seed=config["training"]["random_state"],
        task=config["task"]
    )
    log(f"Лучший val loss: {info['best_val_loss']:.5f}")
    return model, info