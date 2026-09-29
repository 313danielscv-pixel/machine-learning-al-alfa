from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

INSURANCE_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)
AMES_URL = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
BIRTHS_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "births/US_births_2000-2014_SSA.csv"
)
REE_API = "https://apidatos.ree.es"
OPEN_METEO_API = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_CONFIG = {
    "births": {
        "title": "Nacimientos diarios en EE. UU.",
        "start": date(2000, 1, 1),
        "target": "births",
        "unit": "nacimientos por día",
        "exogenous": [],
    },
    "demand": {
        "title": "Demanda eléctrica diaria en España",
        "start": date(2021, 1, 1),
        "target": "demanda_gwh",
        "unit": "GWh por día",
        "exogenous": ["temperature_2m_mean"],
    },
    "solar": {
        "title": "Generación solar fotovoltaica diaria en España",
        "start": date(2023, 1, 1),
        "target": "solar_gwh",
        "unit": "GWh por día",
        "exogenous": ["shortwave_radiation_sum"],
    },
}

EXOGENOUS_LABELS = {
    "temperature_2m_mean": "temperatura media (°C)",
    "shortwave_radiation_sum": "radiación solar (MJ/m²)",
}


def _download_csv(url: str, destination: Path) -> pd.DataFrame:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = Request(url, headers={"User-Agent": "MachineLearningAlAlfa/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if not content:
            raise RuntimeError(f"La fuente no devolvió datos: {url}")
        destination.write_bytes(content)
    return pd.read_csv(destination, low_memory=False)


def load_insurance_frame(raw_dir: Path) -> pd.DataFrame:
    return _download_csv(INSURANCE_URL, raw_dir / "insurance.csv")


def load_ames_frame(raw_dir: Path) -> pd.DataFrame:
    return _download_csv(AMES_URL, raw_dir / "ames_housing.csv")


def _fetch_ree_values(
    *,
    category: str,
    widget: str,
    series_title: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = max(start_date, date(year, 1, 1))
        year_end = min(end_date, date(year, 12, 31))
        params = urlencode(
            {
                "start_date": f"{year_start.isoformat()}T00:00",
                "end_date": f"{year_end.isoformat()}T23:59",
                "time_trunc": "day",
            }
        )
        url = f"{REE_API}/es/datos/{category}/{widget}?{params}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "MachineLearningAlAlfa/1.0 (educational project)",
            },
        )
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        series = next(
            (
                item
                for item in payload.get("included", [])
                if item.get("attributes", {}).get("title") == series_title
            ),
            None,
        )
        if series is None:
            raise RuntimeError(
                f"Red Eléctrica no devolvió la serie '{series_title}' para {year}."
            )
        values.extend(series.get("attributes", {}).get("values", []))

    if not values:
        raise RuntimeError(
            f"Red Eléctrica no devolvió datos de '{series_title}' "
            f"entre {start_date} y {end_date}."
        )
    frame = pd.DataFrame(values)
    frame["date"] = (
        pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Madrid")
        .dt.date
    )
    frame["value_gwh"] = pd.to_numeric(frame["value"], errors="coerce") / 1000
    return frame[["date", "value_gwh"]].dropna().drop_duplicates("date")


def _fetch_weather(start_date: date, end_date: date) -> pd.DataFrame:
    available_end = min(end_date, date.today() - timedelta(days=5))
    if available_end < start_date:
        raise ValueError("No hay fechas históricas disponibles para consultar el clima.")
    params = urlencode(
        {
            "latitude": 40.4168,
            "longitude": -3.7038,
            "start_date": start_date.isoformat(),
            "end_date": available_end.isoformat(),
            "daily": "temperature_2m_mean,shortwave_radiation_sum",
            "timezone": "Europe/Madrid",
        }
    )
    request = Request(
        f"{OPEN_METEO_API}?{params}",
        headers={"User-Agent": "MachineLearningAlAlfa/1.0 (educational project)"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        raise RuntimeError("Open-Meteo no devolvió datos diarios del tiempo.")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]).date,
            "temperature_2m_mean": daily["temperature_2m_mean"],
            "shortwave_radiation_sum": daily["shortwave_radiation_sum"],
        }
    )


def _load_local_energy_csv(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    target = FORECAST_CONFIG[dataset]["target"]
    date_column = next(
        (column for column in ("date", "fecha", "datetime") if column in frame),
        None,
    )
    if date_column is None or target not in frame:
        raise ValueError(
            f"{path.name} debe incluir una columna de fecha y '{target}'."
        )
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    return frame.dropna(subset=["date", target])


def load_forecast_frame(dataset: str, raw_dir: Path) -> pd.DataFrame:
    if dataset not in FORECAST_CONFIG:
        raise ValueError(f"Conjunto de forecasting desconocido: {dataset}")

    config = FORECAST_CONFIG[dataset]
    if dataset == "births":
        path = raw_dir / "US_births_2000-2014_SSA.csv"
        frame = _download_csv(BIRTHS_URL, path)
        required = {"year", "month", "date_of_month", "births"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"El archivo de nacimientos no contiene: {sorted(missing)}")
        frame["date"] = pd.to_datetime(
            {
                "year": frame["year"],
                "month": frame["month"],
                "day": frame["date_of_month"],
            },
            errors="coerce",
        ).dt.date
        frame["births"] = pd.to_numeric(frame["births"], errors="coerce")
        return frame[["date", "births"]].dropna().sort_values("date")

    target = str(config["target"])
    local_path = raw_dir / (
        "demanda_electrica_espana.csv"
        if dataset == "demand"
        else "solar_espana.csv"
    )
    if local_path.exists():
        frame = _load_local_energy_csv(local_path, dataset)
        weather = _fetch_weather(config["start"], max(frame["date"]))
        return frame.merge(weather, on="date", how="inner").sort_values("date")

    today = date.today()
    ree_series = _fetch_ree_values(
        category="demanda" if dataset == "demand" else "generacion",
        widget="evolucion" if dataset == "demand" else "estructura-generacion",
        series_title="Demanda" if dataset == "demand" else "Solar fotovoltaica",
        start_date=config["start"],
        end_date=today,
    ).rename(columns={"value_gwh": target})
    weather = _fetch_weather(config["start"], max(ree_series["date"]))
    frame = ree_series.merge(weather, on="date", how="inner").sort_values("date")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(local_path, index=False)
    return frame
