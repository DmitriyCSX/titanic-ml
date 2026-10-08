"""Загрузка данных и разбиение на train/val."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config["data"]["train_path"])
    test = pd.read_csv(config["data"]["test_path"])
    return train, test


def split_train_val(
    df: pd.DataFrame,
    target: str,
    task: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = df[target] if task == "classification" else None
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df, val_df, train_df[target], val_df[target]