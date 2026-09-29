import pandas as pd
import pytest

from ml_al_alfa.data import (
    clean_ames_data,
    clean_airbnb_data,
    clean_california_data,
    clean_insurance_data,
)


def test_clean_california_removes_impossible_and_duplicate_rows():
    frame = pd.DataFrame(
        {
            "MedInc": [4.0, 4.0, -1.0],
            "HouseAge": [20.0, 20.0, 10.0],
            "AveRooms": [5.0, 5.0, 4.0],
            "AveBedrms": [1.0, 1.0, 1.0],
            "Population": [100.0, 100.0, 50.0],
            "AveOccup": [2.0, 2.0, 2.0],
            "Latitude": [37.0, 37.0, 37.0],
            "Longitude": [-122.0, -122.0, -122.0],
            "MedHouseVal": [2.0, 2.0, 1.0],
        }
    )
    X, y, report = clean_california_data(frame)
    assert len(X) == len(y) == 1
    assert report.rows_before == 3
    assert report.rows_after == 1
    assert report.rows_removed == 2


def test_clean_airbnb_parses_price_filters_outlier_and_drops_identifiers():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "price": ["$125.00", "$9.00", "$9,999.00"],
            "room_type": ["Entire home/apt", "Private room", "Private room"],
            "neighbourhood_cleansed": ["Centro", "Centro", "Centro"],
            "host_is_superhost": ["t", "f", "f"],
            "latitude": [40.42, 40.42, 40.42],
            "longitude": [-3.70, -3.70, -3.70],
            "accommodates": [2, 1, 1],
        }
    )
    X, y, report = clean_airbnb_data(frame)
    assert len(X) == len(y) == 1
    assert y.iloc[0] == pytest.approx(125.0)
    assert "id" not in X.columns
    assert report.rows_removed == 2


def test_clean_airbnb_requires_price_column():
    with pytest.raises(ValueError, match="price"):
        clean_airbnb_data(pd.DataFrame({"room_type": ["Entire home/apt"]}))


def test_clean_insurance_keeps_only_predictors_and_removes_invalid_rows():
    frame = pd.DataFrame(
        {
            "age": [30, 30, 15],
            "bmi": [25.0, 25.0, 22.0],
            "children": [1, 1, 0],
            "sex": ["female", "female", "male"],
            "smoker": ["no", "no", "no"],
            "region": ["southwest", "southwest", "northwest"],
            "charges": [3000, 3000, 1000],
            "customer_id": [1, 2, 3],
        }
    )
    X, y, report = clean_insurance_data(frame)
    assert len(X) == len(y) == 1
    assert list(X.columns) == ["age", "bmi", "children", "sex", "smoker", "region"]
    assert X.iloc[0]["sex"] == 0
    assert X.iloc[0]["smoker"] == 0
    assert y.iloc[0] == 3000
    assert report.rows_removed == 2


def test_clean_ames_uses_selected_features_and_filters_invalid_sale_prices():
    frame = pd.DataFrame(
        {
            "Gr Liv Area": [1500, 1500],
            "Overall Qual": [7, 7],
            "Year Built": [1990, 1990],
            "Garage Cars": [2, 2],
            "Total Bsmt SF": [800, 800],
            "Lot Area": [9000, 9000],
            "Full Bath": [2, 2],
            "Neighborhood": ["CollgCr", "CollgCr"],
            "Bldg Type": ["1Fam", "1Fam"],
            "SalePrice": [200000, 0],
            "PID": [1, 2],
        }
    )
    X, y, report = clean_ames_data(frame)
    assert len(X) == len(y) == 1
    assert "PID" not in X.columns
    assert y.iloc[0] == 200000
    assert report.rows_removed == 1
