from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_al_alfa.data import (
    clean_ames_data,
    clean_airbnb_data,
    clean_california_data,
    clean_insurance_data,
    feature_defaults,
)
from ml_al_alfa.datasets import FORECAST_CONFIG, load_ames_frame, load_forecast_frame, load_insurance_frame
from ml_al_alfa.forecasting import train_forecast_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42


def load_california() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    dataset = fetch_california_housing(as_frame=True)
    X, y, report = clean_california_data(dataset.frame)
    return X, y, report.as_dict()


def load_airbnb() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    candidates = [
        RAW_DIR / "listings.csv.gz",
        RAW_DIR / "listings.csv",
        PROJECT_ROOT / "listings.csv.gz",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "No se encontro el archivo de Airbnb. Ejecuta primero "
            "'python scripts/download_airbnb_madrid.py'."
        )
    frame = pd.read_csv(source, low_memory=False)
    X, y, report = clean_airbnb_data(frame)
    return X, y, report.as_dict()


def load_insurance() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    frame = load_insurance_frame(RAW_DIR)
    X, y, report = clean_insurance_data(frame)
    return X, y, report.as_dict()


def load_ames() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    frame = load_ames_frame(RAW_DIR)
    X, y, report = clean_ames_data(frame)
    return X, y, report.as_dict()


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    dataset_name: str,
    rows: dict[str, int],
    output_path: Path | None = None,
) -> dict[str, Any]:
    if len(X) < 20:
        raise ValueError("Se necesitan al menos 20 filas limpias para entrenar y evaluar.")
    if len(X) != len(y):
        raise ValueError("Las caracteristicas y el objetivo deben tener el mismo numero de filas.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    candidates: dict[str, Pipeline] = {
        "BaselineMedia": Pipeline(
            [
                ("preprocess", _make_preprocessor(X)),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "LinearRegression": Pipeline(
            [
                ("preprocess", _make_preprocessor(X)),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("preprocess", _make_preprocessor(X)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        min_samples_leaf=2,
                        max_features=1.0,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    scores: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        scores[name] = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }
        fitted[name] = pipeline

    best_name = min(scores, key=lambda name: (scores[name]["mae"], name))
    best_model = fitted[best_name]
    artifact_path = output_path or ARTIFACTS_DIR / f"{dataset_name}_model.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "dataset": dataset_name,
        "model_name": best_name,
        "pipeline": best_model,
        "features": X.columns.tolist(),
        "defaults": feature_defaults(X),
        "categories": {
            column: sorted(X[column].dropna().astype(str).unique().tolist())
            for column in X.select_dtypes(exclude=["number"]).columns
        },
        "numeric_ranges": {
            column: {
                "min": float(X[column].min()),
                "max": float(X[column].max()),
            }
            for column in X.select_dtypes(include=["number"]).columns
        },
        "metrics": scores,
        "rows": rows,
        "target_mean": float(y.mean()),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
    }
    joblib.dump(bundle, artifact_path, compress=3)
    metrics_path = artifact_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in bundle.items()
                if key not in {"pipeline", "defaults"}
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle


def train_dataset(dataset_name: str) -> dict[str, Any]:
    if dataset_name in FORECAST_CONFIG:
        frame = load_forecast_frame(dataset_name, RAW_DIR)
        artifact_path = ARTIFACTS_DIR / f"{dataset_name}_model.joblib"
        bundle = train_forecast_and_save(
            frame,
            dataset_name=dataset_name,
            output_path=artifact_path,
        )
        print(
            f"\n{dataset_name.upper()} — filas: {bundle['rows']['rows_before']} "
            f"recibidas, {bundle['rows']['rows_after']} utilizables."
        )
        print(f"Separacion de datos: {bundle['split_method']}.")
        print(f"Mejor variante: {bundle['forecast']['best_variant']}.")
        print(f"Mejor modelo: {bundle['model_name']}")
        print("MAE en test:")
        for name, score in bundle["metrics"].items():
            print(f"  {name:48s} {score['mae']:.4f} (R2={score['r2']:.4f})")
        best_mae = min(
            score["mae"]
            for name, score in bundle["metrics"].items()
            if name != bundle["baseline_name"]
        )
        baseline_mae = bundle["metrics"][bundle["baseline_name"]]["mae"]
        verdict = (
            "mejora el baseline"
            if best_mae < baseline_mae
            else "no mejora el baseline"
        )
        print(f"Conclusion: el mejor modelo {verdict}.")
        print(f"Modelo guardado en: {artifact_path}")
        return bundle

    if dataset_name == "california":
        X, y, rows = load_california()
    elif dataset_name == "airbnb":
        X, y, rows = load_airbnb()
    elif dataset_name == "insurance":
        X, y, rows = load_insurance()
    elif dataset_name == "ames":
        X, y, rows = load_ames()
    else:
        raise ValueError(f"Conjunto de datos desconocido: {dataset_name}")
    bundle = train_and_save(X, y, dataset_name=dataset_name, rows=rows)
    print(
        f"\n{dataset_name.upper()} — filas: {rows['rows_before']} antes, "
        f"{rows['rows_after']} despues de limpiar "
        f"({rows['rows_removed']} eliminadas)."
    )
    print(f"Mejor modelo: {bundle['model_name']}")
    print("MAE en test:")
    for name, score in bundle["metrics"].items():
        print(f"  {name:24s} {score['mae']:.4f} (R2={score['r2']:.4f})")
    best_mae = bundle["metrics"][bundle["model_name"]]["mae"]
    baseline_mae = bundle["metrics"]["BaselineMedia"]["mae"]
    verdict = "mejora el baseline" if best_mae < baseline_mae else "no mejora el baseline"
    print(f"Conclusion: el mejor modelo {verdict}.")
    print(f"Modelo guardado en: {ARTIFACTS_DIR / f'{dataset_name}_model.joblib'}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar los modelos del proyecto.")
    datasets = ("california", "airbnb", "insurance", "ames", *FORECAST_CONFIG)
    parser.add_argument(
        "--dataset",
        choices=(*datasets, "all"),
        default="california",
        help=(
            "Conjunto a entrenar; los datos se descargan al ejecutarlo. "
            "Airbnb requiere descargar antes listings.csv.gz."
        ),
    )
    args = parser.parse_args()
    selections = datasets if args.dataset == "all" else (args.dataset,)
    for selection in selections:
        train_dataset(selection)


if __name__ == "__main__":
    main()
