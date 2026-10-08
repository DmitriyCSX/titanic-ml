"""
Запуск:
    python main.py
    python main.py --model dnn

"""
from __future__ import annotations

import argparse

from src.config import get_config
from src.data import load_data, split_train_val
from src.features import TabularPreprocessor, build_titanic_features
from src.predict import make_submission
from src.train import train_classic, train_dnn_model
from src.utils import ensure_root_on_path, log, set_seed


def run(model_override: str | None = None) -> None:
    ensure_root_on_path()
    set_seed(seed=42)

    log(msg="=== Проект: Titanic ===")
    config = get_config("titanic")
    if model_override:
        config["model"]["name"] = model_override

    # 1. Данные
    train_df, test_df = load_data(config)
    log(msg=f"Train: {train_df.shape}, Test: {test_df.shape}")

    # 2. Feature engineering
    train_df = build_titanic_features(train_df)
    test_df = build_titanic_features(test_df)
    log(msg=f"После FE: Train={train_df.shape}, Test={test_df.shape}")

    # 3. Сплит train/val ДО препроцессинга
    train_split, val_split, y_train, y_val = split_train_val(
        df=train_df,
        target=config["target"],
        task=config["task"],
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    # 4. Препроцессинг
    prep = TabularPreprocessor(config)
    X_train = prep.fit_transform(train_split)
    X_val = prep.transform(val_split)
    X_test = prep.transform(test_df)

    if config["target"] in X_val.columns:
        X_val = X_val.drop(columns=[config["target"]])
    if config["target"] in X_test.columns:
        X_test = X_test.drop(columns=[config["target"]])

    y_train = prep.transform_target(y_train)
    y_val = prep.transform_target(y_val)

    log(msg=f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # 5. Обучение
    model_name = config["model"]["name"].lower()
    is_dnn = model_name in ("dnn", "mlp")

    if is_dnn:
        model, info = train_dnn_model(X_train, y_train, X_val, y_val, config)
        log(msg=f"DNN обучен. best_val_loss={info['best_val_loss']:.5f}")
    else:
        model, info = train_classic(X_train, y_train, X_val, y_val, config)
        log(msg=f"{model_name} обучен. CV={info['cv_mean']:.4f} ± {info['cv_std']:.4f}")

    # 6. Submission
    submission = make_submission(
        model=model,
        X_test=X_test,
        test_df=test_df,
        config=config,
        is_dnn=is_dnn,
        preprocessor=prep,
    )
    log(msg=f"Готово. Строк в submission: {len(submission)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Titanic: train + predict.")
    parser.add_argument("--model", default=None,
                        help="Переопределить модель из конфига (dnn, lightgbm, catboost).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(model_override=args.model)