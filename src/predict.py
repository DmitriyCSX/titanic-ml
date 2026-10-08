"""Инференс + создание submission. Универсально для обеих соревок."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models import predict_dnn
from src.utils import log


def make_submission(
    model,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    is_dnn: bool = False,
    preprocessor=None,
) -> pd.DataFrame:
    """
    Универсальный сабмит:
      - для sklearn-совместимых моделей: model.predict(X_test)
      - для DNN: predict_dnn
      - если таргет лог-трансформировался, возвращаем expm1
    """
    if is_dnn:
        preds = predict_dnn(model, X=np.asarray(a=X_test))
        if config.get("task") == "classification":
            preds = 1.0 / (1.0 + np.exp(-preds))   # засигмоидить в вероятности
            preds = (preds > 0.5).astype(int)       # в 0/1 для Kaggle
    else:
        preds = model.predict(X_test)

    if config["features"].get("log_target"):
        preds = np.expm1(preds)

    submission = pd.DataFrame({
        config["id_column"]: test_df[config["id_column"]].values,
        config["target"]: preds,
    })

    out_path = Path(config["data"]["submission_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    log(f"Submission сохранён: {out_path} ({len(submission)} строк)")
    return submission