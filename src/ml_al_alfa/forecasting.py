from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pandas.tseries.holiday import USFederalHolidayCalendar

from ml_al_alfa.datasets import EXOGENOUS_LABELS, FORECAST_CONFIG

BASE_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
]
US_BIRTH_FEATURES = ["is_us_federal_holiday", "is_friday_13"]
RANDOM_STATE = 42
MAX_FORECAST_DAYS = 14


def _calendar_features(
    target_date: pd.Timestamp,
    dataset_name: str,
) -> dict[str, float]:
    day_of_year = target_date.dayofyear
    angle = 2 * np.pi * day_of_year / 366
    features = {
        "day_of_week": float(target_date.dayofweek),
        "month": float(target_date.month),
        "day_of_year_sin": float(np.sin(angle)),
        "day_of_year_cos": float(np.cos(angle)),
        "is_weekend": float(target_date.dayofweek >= 5),
    }
    if dataset_name == "births":
        holiday = USFederalHolidayCalendar().holidays(
            start=target_date.normalize(),
            end=target_date.normalize(),
        )
        features["is_us_federal_holiday"] = float(target_date.normalize() in holiday)
        features["is_friday_13"] = float(
            target_date.day == 13 and target_date.dayofweek == 4
        )
    return features


def _feature_row(
    history: pd.Series,
    target_date: pd.Timestamp,
    exogenous_values: dict[str, float],
    dataset_name: str,
) -> pd.DataFrame:
    if len(history) < 14:
        raise ValueError("Se necesitan al menos 14 días de historia para pronosticar.")
    expected_previous = target_date - pd.Timedelta(days=1)
    if history.index[-1] != expected_previous:
        raise ValueError("La historia debe terminar justo el día anterior al pronóstico.")
    values = history.to_numpy(dtype=float)
    row: dict[str, float] = {
        "lag_1": float(values[-1]),
        "lag_7": float(values[-7]),
        "lag_14": float(values[-14]),
        "rolling_mean_7": float(values[-7:].mean()),
        **_calendar_features(target_date, dataset_name),
        **exogenous_values,
    }
    return pd.DataFrame([row])


def _make_features(
    frame: pd.DataFrame,
    target: str,
    exogenous_features: list[str],
    dataset_name: str = "births",
) -> tuple[pd.DataFrame, pd.Series]:
    required = {"date", target, *exogenous_features}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas para el forecasting: {sorted(missing_columns)}"
        )

    clean = frame[["date", target, *exogenous_features]].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    clean[target] = pd.to_numeric(clean[target], errors="coerce")
    for column in exogenous_features:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["date", target, *exogenous_features])
    clean = clean.sort_values("date").drop_duplicates("date").set_index("date")
    if clean.empty:
        raise ValueError("No quedan observaciones válidas para entrenar el forecasting.")
    expected_dates = pd.date_range(clean.index.min(), clean.index.max(), freq="D")
    if not clean.index.equals(expected_dates):
        missing_count = len(expected_dates.difference(clean.index))
        raise ValueError(
            f"La serie debe tener una fila por día; faltan {missing_count} fechas. "
            "Completa el CSV diario antes de entrenar."
        )

    target_values = clean[target].astype(float)
    features = pd.DataFrame(index=clean.index)
    features["lag_1"] = target_values.shift(1)
    features["lag_7"] = target_values.shift(7)
    features["lag_14"] = target_values.shift(14)
    features["rolling_mean_7"] = target_values.shift(1).rolling(7).mean()
    calendar_features = [
        _calendar_features(stamp, dataset_name) for stamp in clean.index
    ]
    for column in calendar_features[0]:
        features[column] = [row[column] for row in calendar_features]
    selected_base_features = BASE_FEATURES + (
        US_BIRTH_FEATURES if dataset_name == "births" else []
    )
    features = features[selected_base_features]
    for column in exogenous_features:
        features[column] = clean[column]

    aligned_target = target_values.loc[features.index]
    valid = features.notna().all(axis=1)
    return features.loc[valid], aligned_target.loc[valid]


def _metric(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def train_forecast_and_save(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    output_path: Path,
) -> dict[str, Any]:
    if dataset_name not in FORECAST_CONFIG:
        raise ValueError(f"Conjunto de forecasting desconocido: {dataset_name}")
    config = FORECAST_CONFIG[dataset_name]
    target = str(config["target"])
    exogenous_features = list(config["exogenous"])
    X, y = _make_features(
        frame,
        target,
        exogenous_features,
        dataset_name=dataset_name,
    )
    if len(X) < 120:
        raise ValueError(
            f"Se necesitan al menos 120 días utilizables; solo quedan {len(X)}."
        )

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    if len(X_test) < 2:
        raise ValueError("El conjunto temporal de prueba es demasiado pequeño.")

    variants: dict[str, list[str]] = {
        "Calendario y datos pasados": [],
    }
    if exogenous_features:
        label = "Con " + " y ".join(
            EXOGENOUS_LABELS.get(column, column) for column in exogenous_features
        )
        variants[label] = exogenous_features

    metrics: dict[str, dict[str, float]] = {}
    fitted_variants: dict[str, dict[str, Any]] = {}
    baseline_prediction = (
        frame.assign(date=pd.to_datetime(frame["date"]))
        .set_index("date")[target]
        .astype(float)
        .sort_index()
        .reindex(X_test.index - pd.Timedelta(days=7))
        .to_numpy()
    )
    metrics["Baseline 7 días"] = _metric(y_test, baseline_prediction)

    for variant_label, selected_exogenous in variants.items():
        selected_features = (
            BASE_FEATURES
            + (US_BIRTH_FEATURES if dataset_name == "births" else [])
            + selected_exogenous
        )
        fitted_models: dict[str, Pipeline] = {}
        variant_metrics: dict[str, dict[str, float]] = {}
        for model_name, model in (
            ("LinearRegression", LinearRegression()),
            (
                "RandomForestRegressor",
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=2,
                    max_features=1.0,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ):
            steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="median"))
            ]
            if model_name == "LinearRegression":
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", model))
            pipeline = Pipeline(steps)
            pipeline.fit(X_train[selected_features], y_train)
            predicted = pipeline.predict(X_test[selected_features])
            metric_key = f"{model_name} ({variant_label})"
            metrics[metric_key] = _metric(y_test, predicted)
            variant_metrics[model_name] = metrics[metric_key]
            fitted_models[model_name] = pipeline

        best_model_name = min(
            variant_metrics,
            key=lambda name: (variant_metrics[name]["mae"], name),
        )
        fitted_variants[variant_label] = {
            "features": selected_features,
            "model_name": best_model_name,
            "pipeline": fitted_models[best_model_name],
            "metrics": variant_metrics,
        }

    best_variant = min(
        fitted_variants,
        key=lambda label: (
            fitted_variants[label]["metrics"][
                fitted_variants[label]["model_name"]
            ]["mae"],
            label,
        ),
    )
    best_model = fitted_variants[best_variant]
    best_pipeline = best_model["pipeline"]
    best_metric_name = f"{best_model['model_name']} ({best_variant})"
    best_test_prediction = best_pipeline.predict(X_test[best_model["features"]])

    history = frame[["date", target]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history[target] = pd.to_numeric(history[target], errors="coerce")
    history = (
        history.dropna()
        .sort_values("date")
        .drop_duplicates("date")
        .tail(365)
    )
    if history.empty:
        raise ValueError("No hay historia válida para guardar en el modelo.")

    rows = {
        "rows_before": int(len(frame)),
        "rows_after": int(len(X)),
        "rows_removed": int(len(frame) - len(X)),
        "missing_values_after": 0,
    }
    default_exogenous = {
        column: float(pd.to_numeric(frame[column], errors="coerce").median())
        for column in exogenous_features
    }
    forecast_metadata = {
        "target": target,
        "unit": config["unit"],
        "latest_date": history["date"].iloc[-1].date().isoformat(),
        "history": [
            [row["date"].date().isoformat(), float(row[target])]
            for _, row in history.iterrows()
        ],
        "default_exogenous": default_exogenous,
        "exogenous_labels": {
            column: EXOGENOUS_LABELS.get(column, column)
            for column in exogenous_features
        },
        "variants": fitted_variants,
        "best_variant": best_variant,
        "evaluation": {
            "dates": [stamp.date().isoformat() for stamp in X_test.index[-120:]],
            "actual": y_test.iloc[-120:].astype(float).tolist(),
            "predicted": [float(value) for value in best_test_prediction[-120:]],
        },
        "split_date": X_test.index[0].date().isoformat(),
    }
    bundle: dict[str, Any] = {
        "dataset": dataset_name,
        "task": "forecasting",
        "model_name": best_metric_name,
        "pipeline": best_pipeline,
        "features": best_model["features"],
        "metrics": metrics,
        "baseline_name": "Baseline 7 días",
        "rows": rows,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "split_method": "cronológico (entrenamiento antes del test)",
        "forecast": forecast_metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(bundle, output_path, compress=3)
    summary = {
        "dataset": dataset_name,
        "rows": rows,
        "model_name": best_model["model_name"],
        "best_metric_name": best_metric_name,
        "best_variant": best_variant,
        "metrics": metrics,
        "baseline_name": "Baseline 7 días",
        "split_method": bundle["split_method"],
        "split_date": forecast_metadata["split_date"],
        "latest_date": forecast_metadata["latest_date"],
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle


def forecast_to_date(
    bundle: dict[str, Any],
    *,
    variant_label: str,
    days_ahead: int,
    exogenous_values: dict[str, float] | None = None,
) -> list[tuple[pd.Timestamp, float]]:
    if not 1 <= days_ahead <= MAX_FORECAST_DAYS:
        raise ValueError(
            f"El horizonte debe estar entre 1 y {MAX_FORECAST_DAYS} días."
        )
    forecast = bundle["forecast"]
    variants = forecast["variants"]
    if variant_label not in variants:
        raise ValueError(f"Variante de pronóstico desconocida: {variant_label}")
    variant = variants[variant_label]
    pipeline = variant["pipeline"]
    features = variant["features"]
    history_frame = pd.DataFrame(
        forecast["history"], columns=["date", forecast["target"]]
    )
    history = pd.Series(
        history_frame[forecast["target"]].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(history_frame["date"])),
    ).sort_index()
    selected_exogenous = (
        exogenous_values if exogenous_values is not None else {}
    )
    predictions: list[tuple[pd.Timestamp, float]] = []
    for offset in range(1, days_ahead + 1):
        target_date = history.index[-1] + pd.Timedelta(days=1)
        row = _feature_row(
            history,
            target_date,
            selected_exogenous,
            bundle["dataset"],
        )
        prediction = float(pipeline.predict(row[features])[0])
        predictions.append((target_date, prediction))
        history.loc[target_date] = prediction
    return predictions
