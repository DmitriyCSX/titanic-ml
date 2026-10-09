"""Feature engineering + препроцессинг для Titanic. Всё fit на train, transform на test."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Feature engineering (чистые функции, возвращают новый df)
# ============================================================


# извлекает титул (Mr, Mrs, Miss, ...) из колонки Name через regex
def titanic_extract_title(df: pd.DataFrame) -> pd.Series:
    return df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# размер семьи: SibSp + Parch + 1 (сам пассажир)
def titanic_family_size(df: pd.DataFrame) -> pd.Series:
    return df["SibSp"] + df["Parch"] + 1

# бинарный флаг: 1 если пассажир без семьи (SibSp=0 и Parch=0)
def titanic_is_alone(df: pd.DataFrame) -> pd.Series:
    return (df["SibSp"] + df["Parch"] == 0).astype(int)

# категориальная группа возраста (Kid / Teen / Adult / Old) через pd.cut
def titanic_age_group(df: pd.DataFrame) -> pd.Series:
    return pd.cut(
        df["Age"],
        bins=[0, 12, 18, 60, 100],
        labels=["Kid", "Teen", "Adult", "Old"],
    )

# Fare / FamilySize (цена билета на человека)
def titanic_fare_per_person(df: pd.DataFrame) -> pd.Series:
    fam = titanic_family_size(df).replace(0, 1)
    return df["Fare"] / fam

# бинарный флаг: 1 если Cabin заполнен (пассажир знал номер каюты)
def titanic_has_cabin(df: pd.DataFrame) -> pd.Series:
    return df["Cabin"].notna().astype(int)

# полный FE-пайплайн: Title, FamilySize, IsAlone, AgeGroup, FarePerPerson, HasCabin; редкие Title → "Rare"
def build_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = titanic_extract_title(df)
    df["FamilySize"] = titanic_family_size(df)
    df["IsAlone"] = titanic_is_alone(df)
    df["AgeGroup"] = titanic_age_group(df)
    df["FarePerPerson"] = titanic_fare_per_person(df)
    df["HasCabin"] = titanic_has_cabin(df)
    rare = df["Title"].value_counts()[lambda s: s < 10].index
    df["Title"] = df["Title"].replace(rare, "Rare")
    return df


# ============================================================
# Препроцессинг-пайплайн (fit на train, transform на val/test)
# обучает препроцессор на train: drop, fillna, медианы, OHE, scaler; запоминает feature_columns_
# ============================================================

class TabularPreprocessor:
    """
    Универсальный препроцессор:
      - drop_columns
      - заполняет "нет фичи" (None / 0)
      - групповые медианы (fit на train)
      - impute медианой/модой (fit на train)
      - ordinal encoding
      - one-hot (fit на train, transform на val/test)
      - опциональный StandardScaler
      - лог-трансформация таргета
    """

    def __init__(self, config: dict[str, Any]):
        self.cfg = config["features"]
        self.target = config["target"]
        self.task = config["task"]
        self.is_fitted = False

    # ---------- публичные методы ----------

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        df = df.copy()
        self._log_target_fit(df)

        df = self._drop_columns(df)
        df = self._fill_none(df)
        df = self._fit_group_medians(df)
        df = self._impute(df, fit=True)
        df = self._encode_ordinal(df, fit=True)
        df = self._fit_onehot(df)
        df = self._fit_scaler(df)

        df_check = df.copy()
        df_check = self._apply_group_medians(df_check)
        df_check = self._impute(df_check, fit=False)
        df_check = self._encode_ordinal(df_check, fit=False)
        df_check = self._apply_onehot(df_check)
        df_check = self._apply_scaler(df_check)

        self.feature_columns_ = [
            c for c in df_check.columns
            if c not in (self.target, self.cfg.get("id_column"))
        ]
        self.is_fitted = True
        return self

    # применяет обученные преобразования к val/test (без пересчёта статистик)
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Сначала fit()"
        df = df.copy()

        df = self._fill_none(df)
        df = self._apply_group_medians(df)
        df = self._impute(df, fit=False)
        df = self._encode_ordinal(df, fit=False)
        df = self._apply_onehot(df)
        df = self._apply_scaler(df)

        for col in self.feature_columns_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns_]
        return df

    # fit + transform за один вызов
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # обратное преобразование таргета (expm1 если log_target=True)
    def inverse_target(self, y):
        if self.cfg.get("log_target"):
            return np.expm1(y)
        return y

    # прямое преобразование таргета (log1p если log_target=True)
    def transform_target(self, y):
        if self.cfg.get("log_target"):
            return np.log1p(y)
        return y

    # ---------- внутренние шаги ----------

    
    # лог-трансформация применяется отдельно в train/predict
    def _log_target_fit(self, df):
        pass

    # удаляет колонки из config.features.drop_columns
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = self.cfg.get("drop_columns") or []
        existing = [c for c in drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
        return df

    # "нет фичи" → "None" для категориальных, 0 для числовых
    def _fill_none(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cfg.get("none_categorical", []):
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in self.cfg.get("none_numeric", []):
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    # считает медианы по группам (fit на train) и заполняет ими пропуски
    def _fit_group_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        self.group_medians_ = {}
        for col, group in self.cfg.get("group_median", {}).items():
            if col in df.columns and group in df.columns:
                med = df.groupby(group)[col].median()
                self.group_medians_[col] = med
                df[col] = df[col].fillna(df[group].map(med))
        return df

    # применяет сохранённые медианы к val/test
    def _apply_group_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, med in getattr(self, "group_medians_", {}).items():
            group = self.cfg["group_median"][col]
            if col in df.columns and group in df.columns:
                df[col] = df[col].fillna(df[group].map(med))
        return df

    # заполняет пропуски медианой/модой (fit на train) или применяет сохранённые значения
    def _impute(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        fillna_cfg = self.cfg.get("fillna", {})
        if fit:
            self.impute_stats_ = {}
        for col, strategy in fillna_cfg.items():
            if col not in df.columns:
                continue
            if fit:
                if strategy == "median":
                    self.impute_stats_[col] = df[col].median()
                elif strategy == "mode":
                    self.impute_stats_[col] = df[col].mode().iloc[0]
                else:
                    raise ValueError(f"Unknown fillna strategy: {strategy}")
            df[col] = df[col].fillna(self.impute_stats_[col])
        return df

    # ordinal-кодирование по маппингу из конфига (с поддержкой формата {categories: [...]})
    def _encode_ordinal(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        ordinal_cfg = self.cfg.get("ordinal", {})
        if fit:
            self.ordinal_maps_ = {}
        for col, mapping in ordinal_cfg.items():
            if col not in df.columns:
                continue
            if isinstance(mapping, dict) and "categories" in mapping:
                cats = mapping["categories"]
                mapping = {cat: i for i, cat in enumerate(cats)}
            if fit:
                self.ordinal_maps_[col] = mapping
            df[col] = df[col].map(self.ordinal_maps_[col]).fillna(0).astype(int)
        return df

    # обучает OneHotEncoder на train (handle_unknown="ignore", drop="first")
    def _fit_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        onehot_cfg = self.cfg.get("onehot", [])
        if onehot_cfg == "auto":
            onehot_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        else:
            onehot_cols = [c for c in onehot_cfg if c in df.columns]

        self.onehot_cols_ = onehot_cols
        self.onehot_encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="first"
        )
        if onehot_cols:
            self.onehot_encoder_.fit(df[onehot_cols])
        return df

    # применяет OHE к val/test, дропает исходные категориальные колонки
    def _apply_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "onehot_cols_", None):
            return df
        encoded = self.onehot_encoder_.transform(df[self.onehot_cols_])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.onehot_encoder_.get_feature_names_out(self.onehot_cols_),
            index=df.index,
        )
        df = df.drop(columns=self.onehot_cols_)
        return pd.concat([df, encoded_df], axis=1)

    # обучает StandardScaler на scale_columns из конфига (или отключает, если null)
    def _fit_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        scale_cols = self.cfg.get("scale_columns")
        if not scale_cols:
            self.scaler_ = None
            return df
        self.scale_cols_ = [c for c in scale_cols if c in df.columns]
        self.scaler_ = StandardScaler()
        if self.scale_cols_:
            self.scaler_.fit(df[self.scale_cols_])
        return df

    # применяет scaler к val/test
    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        if getattr(self, "scaler_", None) is None:
            return df
        df[self.scale_cols_] = self.scaler_.transform(df[self.scale_cols_])
        return df