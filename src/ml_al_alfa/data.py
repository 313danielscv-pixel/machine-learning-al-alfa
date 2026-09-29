from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


CALIFORNIA_TARGET = "MedHouseVal"
AIRBNB_TARGET = "price"

CALIFORNIA_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

AIRBNB_NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "calculated_host_listings_count",
    "latitude",
    "longitude",
]
AIRBNB_CATEGORICAL_FEATURES = [
    "room_type",
    "neighbourhood_cleansed",
    "host_is_superhost",
]
AIRBNB_CANDIDATE_FEATURES = AIRBNB_NUMERIC_FEATURES + AIRBNB_CATEGORICAL_FEATURES

INSURANCE_TARGET = "charges"
INSURANCE_FEATURES = [
    "age",
    "bmi",
    "children",
    "sex",
    "smoker",
    "region",
]

AMES_TARGET = "SalePrice"
AMES_FEATURES = [
    "Gr Liv Area",
    "Overall Qual",
    "Year Built",
    "Garage Cars",
    "Total Bsmt SF",
    "Lot Area",
    "Full Bath",
    "Neighborhood",
    "Bldg Type",
]


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    rows_removed: int
    missing_values_after: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
            "missing_values_after": self.missing_values_after,
        }


def clean_california_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    """Keep plausible census rows and report the exact number removed."""
    before = len(frame)
    clean = frame.copy().drop_duplicates()
    required = CALIFORNIA_FEATURES + [CALIFORNIA_TARGET]
    missing_columns = sorted(set(required) - set(clean.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas de California: {missing_columns}")

    clean = clean[required].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=required)
    valid = (
        clean[CALIFORNIA_TARGET].between(0, 5.01)
        & clean["MedInc"].gt(0)
        & clean["HouseAge"].gt(0)
        & clean["AveRooms"].gt(0)
        & clean["AveBedrms"].gt(0)
        & clean["Population"].gt(0)
        & clean["AveOccup"].gt(0)
        & clean["Latitude"].between(32, 43)
        & clean["Longitude"].between(-125, -113)
    )
    clean = clean.loc[valid].reset_index(drop=True)
    X = clean[CALIFORNIA_FEATURES].copy()
    y = clean[CALIFORNIA_TARGET].copy()
    report = CleaningReport(
        rows_before=before,
        rows_after=len(clean),
        rows_removed=before - len(clean),
        missing_values_after=int(clean.isna().sum().sum()),
    )
    return X, y, report


def clean_airbnb_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    """Parse nightly prices and remove unusable or implausible public listings."""
    before = len(frame)
    clean = frame.copy().drop_duplicates()
    if AIRBNB_TARGET not in clean.columns:
        raise ValueError("El archivo Airbnb no contiene la columna obligatoria 'price'.")

    clean[AIRBNB_TARGET] = pd.to_numeric(
        clean[AIRBNB_TARGET]
        .astype("string")
        .str.replace(r"[$,]", "", regex=True)
        .str.strip(),
        errors="coerce",
    )

    available_features = [
        column for column in AIRBNB_CANDIDATE_FEATURES if column in clean.columns
    ]
    required_location = {"latitude", "longitude"}
    missing_location = required_location - set(available_features)
    if missing_location:
        raise ValueError(
            "Faltan coordenadas necesarias para analizar anuncios: "
            f"{sorted(missing_location)}"
        )
    if "room_type" not in available_features:
        raise ValueError("El archivo Airbnb no contiene la columna 'room_type'.")

    numeric_features = [
        column for column in AIRBNB_NUMERIC_FEATURES if column in available_features
    ]
    categorical_features = [
        column
        for column in AIRBNB_CATEGORICAL_FEATURES
        if column in available_features
    ]
    for column in numeric_features:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean["latitude"] = pd.to_numeric(clean["latitude"], errors="coerce")
    clean["longitude"] = pd.to_numeric(clean["longitude"], errors="coerce")
    clean = clean.dropna(subset=[AIRBNB_TARGET, "latitude", "longitude"])
    valid = (
        clean[AIRBNB_TARGET].between(10, 500)
        & clean["latitude"].between(40.2, 40.7)
        & clean["longitude"].between(-4.1, -3.4)
    )
    for column in ("accommodates", "bedrooms", "beds", "minimum_nights"):
        if column in clean:
            valid &= clean[column].isna() | clean[column].ge(0)
    clean = clean.loc[valid].reset_index(drop=True)

    features = numeric_features + categorical_features
    X = clean[features].copy()
    for column in categorical_features:
        X[column] = X[column].astype("string").fillna("Desconocido").astype(str)
    y = clean[AIRBNB_TARGET].astype(float).copy()
    report = CleaningReport(
        rows_before=before,
        rows_after=len(clean),
        rows_removed=before - len(clean),
        missing_values_after=int(X.isna().sum().sum() + y.isna().sum()),
    )
    return X, y, report


def _clean_regression_data(
    frame: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    numeric_features: list[str],
    dataset_label: str,
) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    before = len(frame)
    required = features + [target]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Faltan columnas de {dataset_label}: {missing_columns}"
        )

    clean = frame[required].copy().drop_duplicates()
    for column in numeric_features + [target]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=[target])
    clean = clean.loc[clean[target].gt(0)].reset_index(drop=True)

    for column in features:
        if column not in numeric_features:
            clean[column] = (
                clean[column].astype("string").fillna("Desconocido").astype(str)
            )

    X = clean[features].copy()
    y = clean[target].astype(float).copy()
    return X, y, CleaningReport(
        rows_before=before,
        rows_after=len(clean),
        rows_removed=before - len(clean),
        missing_values_after=int(X.isna().sum().sum() + y.isna().sum()),
    )


def clean_insurance_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    X, y, report = _clean_regression_data(
        frame,
        target=INSURANCE_TARGET,
        features=INSURANCE_FEATURES,
        numeric_features=["age", "bmi", "children"],
        dataset_label="seguro médico",
    )
    valid = (
        X["age"].between(18, 100)
        & X["bmi"].gt(0)
        & X["children"].ge(0)
    )
    X["sex"] = X["sex"].str.lower().map({"female": 0, "male": 1})
    X["smoker"] = X["smoker"].str.lower().map({"no": 0, "yes": 1})
    valid &= X["sex"].notna() & X["smoker"].notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    removed_invalid = report.rows_after - len(X)
    report.rows_after = len(X)
    report.rows_removed += removed_invalid
    report.missing_values_after = int(X.isna().sum().sum() + y.isna().sum())
    return X, y, report


def clean_ames_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    X, y, report = _clean_regression_data(
        frame,
        target=AMES_TARGET,
        features=AMES_FEATURES,
        numeric_features=[
            "Gr Liv Area",
            "Overall Qual",
            "Year Built",
            "Garage Cars",
            "Total Bsmt SF",
            "Lot Area",
            "Full Bath",
        ],
        dataset_label="viviendas de Ames",
    )
    valid = (
        X["Gr Liv Area"].isna() | X["Gr Liv Area"].gt(0)
    ) & (X["Overall Qual"].isna() | X["Overall Qual"].between(1, 10)) & (
        X["Year Built"].isna() | X["Year Built"].between(1800, 2030)
    )
    for column in ("Garage Cars", "Total Bsmt SF", "Lot Area", "Full Bath"):
        valid &= X[column].isna() | X[column].ge(0)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    removed_invalid = report.rows_after - len(X)
    report.rows_after = len(X)
    report.rows_removed += removed_invalid
    report.missing_values_after = int(X.isna().sum().sum() + y.isna().sum())
    return X, y, report


def feature_defaults(X: pd.DataFrame) -> dict[str, Any]:
    """Return useful, non-null form defaults for the Streamlit prediction form."""
    defaults: dict[str, Any] = {}
    for column in X.columns:
        values = X[column].dropna()
        if values.empty:
            defaults[column] = 0.0
        elif pd.api.types.is_numeric_dtype(X[column]):
            defaults[column] = float(values.median())
        else:
            defaults[column] = str(values.mode().iloc[0])
    return defaults
