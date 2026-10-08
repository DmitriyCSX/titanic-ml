"""Фабрика моделей + обучение с early stopping (для DNN)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Классические модели
# ============================================================

def build_model(name: str, params: dict[str, Any], task: str):
    name = name.lower()
    if task == "classification":
        if name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        if name == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**params)
        if name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        if name == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        if name == "logreg":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
    else:
        if name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**params)
        if name == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**params)
        if name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
        if name == "randomforest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        if name == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(**params)
        if name == "lasso":
            from sklearn.linear_model import Lasso
            return Lasso(**params)
        if name == "dnn":
            return None  # обрабатывается отдельно
    raise ValueError(f"Unknown model: {name} for task {task}")


# ============================================================
# DNN (PyTorch) с early stopping и возвратом к лучшему чекпоинту
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: list[int] = (64, 32),
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 20,
    dropout: float = 0.2,
    seed: int = 42,
    task: str = "regression",           
    device: str | None = None,
) -> tuple[MLP, dict]:
    """
    Обучает MLP с early stopping. Возвращает лучшую модель (по val loss)
    и историю обучения.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    model = MLP(X_tr.shape[1], list(hidden_sizes), dropout=dropout).to(device)
    
    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_loss = epoch_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v.to(device)), y_v.to(device)).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[DNN] Early stopping на эпохе {epoch} (best val={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, {"history": history, "best_val_loss": best_val}


def predict_dnn(model: MLP, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy().flatten()