"""Загрузка и валидация YAML-конфигов."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Читает YAML-конфиг и превращает относительные пути в абсолютные."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Приводим пути данных/сабмита к абсолютным
    for key in ("train_path", "test_path", "submission_path"):
        if key in cfg.get("data", {}):
            p = Path(cfg["data"][key])
            if not p.is_absolute():
                cfg["data"][key] = str(PROJECT_ROOT / p)

    cfg["_config_path"] = str(config_path)
    return cfg


def get_config(name: str) -> dict[str, Any]:
    """Удобный шорткат: get_config('titanic') => configs/titanic.yaml."""
    return load_config(CONFIGS_DIR / f"{name}.yaml")


# ======================================== #
# ЛУЧШИЕ МОДЕЛИ
# ======================================== #

# TITANIC_BEST_MODEL = {
#     "name": "LightGBM",
#     "params": {
#         "n_estimators": 6,
#         "learning_rate": 0.5,
#         "max_depth": 2,
#         "num_leaves": 10,
#         "reg_alpha": 0.11,
#         "reg_lambda": 0.11,
#         "random_state": 42,
#         "verbose": -1,
#     },
#     "cv_score": 0.8246,
#     "kaggle_score": 0.75119,
# }
# BEST_PARAMS = TITANIC_BEST_MODEL["params"]
# BEST_MODEL_NAME = TITANIC_BEST_MODEL["name"]

# HOUSING_DNN_CONFIG = {
#     "model_type": "create_mlp_v3",
#     "params": {"input_size": None, "hidden_sizes": [64, 32]},
#     "train_params": {"epochs": 450, "batch_size": 16, "lr": 0.000005, "patience": 20},
#     "cv_score": 0.1339,
#     "best_epoch": 430,
#     "kaggle_score": 0.14777,
# }