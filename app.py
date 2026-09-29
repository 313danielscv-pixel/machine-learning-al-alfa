from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from ml_al_alfa.train import ARTIFACTS_DIR
from ml_al_alfa.forecasting import MAX_FORECAST_DAYS, forecast_to_date

st.set_page_config(
    page_title="Machine Learning Al Alfa",
    page_icon="🏠",
    layout="wide",
)

MODEL_INFO = {
    "Viviendas de California": {
        "key": "california",
        "target_unit": "cientos de miles de USD",
        "description": (
            "Estimacion educativa basada en datos censales de California de 1990. "
            "No es una tasacion actual; el objetivo original esta limitado a 500.000 USD."
        ),
    },
    "Reto: Airbnb Madrid": {
        "key": "airbnb",
        "target_unit": "EUR por noche",
        "description": (
            "Estimacion del precio publicado usando atributos del anuncio. "
            "No predice reservas ni el precio final pagado."
        ),
    },
    "Seguro medico": {
        "key": "insurance",
        "target_unit": "USD al ano",
        "description": (
            "Estimacion educativa del coste anual a partir de edad, BMI, hijos "
            "y otras caracteristicas. No es una oferta real de seguro."
        ),
    },
    "Viviendas de Ames, Iowa": {
        "key": "ames",
        "target_unit": "USD",
        "description": (
            "Estimacion educativa del precio de venta usando caracteristicas "
            "de casas vendidas en Ames, Iowa."
        ),
    },
    "Nacimientos diarios en EE. UU.": {
        "key": "births",
        "target_unit": "nacimientos por dia",
        "description": (
            "Pronostico diario basado en nacimientos anteriores y el calendario. "
            "No representa un conteo oficial futuro."
        ),
    },
    "Demanda electrica diaria en Espana": {
        "key": "demand",
        "target_unit": "GWh por dia",
        "description": (
            "Pronostico de demanda peninsular de Red Electrica. La variante "
            "meteorologica usa la temperatura media observada en Madrid."
        ),
    },
    "Generacion solar diaria en Espana": {
        "key": "solar",
        "target_unit": "GWh por dia",
        "description": (
            "Pronostico de generacion solar fotovoltaica de Red Electrica. "
            "La variante meteorologica usa radiacion solar de Madrid."
        ),
    },
}


@st.cache_resource
def load_bundle(path_string: str) -> dict[str, Any]:
    return joblib.load(path_string)


def show_metrics(bundle: dict[str, Any]) -> None:
    metrics = bundle["metrics"]
    baseline_name = bundle.get("baseline_name", "BaselineMedia")
    baseline = metrics[baseline_name]["mae"]
    best = metrics[bundle["model_name"]]["mae"]
    unit = MODEL_INFO[st.session_state["project"]]["target_unit"]
    first, second, third = st.columns(3)
    first.metric("Mejor modelo", bundle["model_name"])
    second.metric("MAE mejor modelo", f"{best:,.3f} {unit}")
    third.metric("MAE baseline media", f"{baseline:,.3f} {unit}")
    if best >= baseline:
        st.warning(
            "El mejor modelo no supera al baseline en este test; no hay evidencia "
            "de mejora predictiva frente a predecir la media."
        )
    st.caption(
        (
            f"Evaluacion cronologica: test posterior al entrenamiento, "
            f"desde {bundle['forecast']['split_date']}."
            if bundle.get("task") == "forecasting"
            else (
                f"Evaluacion reproducible: test aleatorio del 20% "
                f"(random_state={bundle['random_state']})."
            )
        )
        + " MAE calculado en el conjunto de test."
    )


def prediction_form(
    project: str,
    bundle: dict[str, Any],
) -> dict[str, Any] | None:
    info = MODEL_INFO[project]
    defaults = bundle["defaults"]
    ranges = bundle["numeric_ranges"]
    categorical = {
        feature
        for feature in bundle["features"]
        if feature not in ranges
    }
    values: dict[str, Any] = {}
    labels = {
        "age": "Edad",
        "bmi": "Índice de masa corporal (IMC)",
        "children": "Número de hijos",
        "sex": "Sexo",
        "smoker": "Fuma",
        "region": "Región",
        "Gr Liv Area": "Superficie habitable",
        "Overall Qual": "Calidad general",
        "Year Built": "Año de construcción",
        "Garage Cars": "Plazas de garaje",
        "Total Bsmt SF": "Superficie del sótano",
        "Lot Area": "Superficie de la parcela",
        "Full Bath": "Baños completos",
        "Neighborhood": "Barrio",
        "Bldg Type": "Tipo de vivienda",
    }
    with st.form(f"prediction-{info['key']}"):
        columns = st.columns(2)
        for index, feature in enumerate(bundle["features"]):
            column = columns[index % 2]
            label = labels.get(
                feature,
                feature.replace("_", " ").replace("MedInc", "ingreso mediano").title(),
            )
            if feature in categorical:
                choices = bundle.get("categories", {}).get(feature, [])
                if not choices:
                    choices = [str(defaults[feature])]
                default_value = str(defaults[feature])
                selected = default_value if default_value in choices else choices[0]
                values[feature] = column.selectbox(label, choices, index=choices.index(selected))
            else:
                bounds = ranges[feature]
                minimum, maximum = float(bounds["min"]), float(bounds["max"])
                default = float(defaults[feature])
                if feature in {"sex", "smoker"} and minimum == 0 and maximum == 1:
                    options = [0, 1]
                    option_labels = (
                        {0: "Mujer", 1: "Hombre"}
                        if feature == "sex"
                        else {0: "No", 1: "Sí"}
                    )
                    selected = int(round(default))
                    values[feature] = column.selectbox(
                        label,
                        options,
                        index=options.index(selected),
                        format_func=lambda value, labels=option_labels: labels[value],
                    )
                    continue
                if minimum == maximum:
                    values[feature] = column.number_input(
                        label, value=default, key=f"{info['key']}-{feature}"
                    )
                else:
                    step = max((maximum - minimum) / 1000, 0.001)
                    step = round(step, 6)
                    default = minimum + round((default - minimum) / step) * step
                    default = min(max(default, minimum), maximum)
                    values[feature] = column.slider(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=step,
                        key=f"{info['key']}-{feature}",
                    )
        submitted = st.form_submit_button("Estimar precio", type="primary")
    if submitted:
        return values
    return None


def show_feature_chart(bundle: dict[str, Any]) -> None:
    pipeline = bundle["pipeline"]
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
        title = "Importancia relativa estimada por Random Forest"
    elif hasattr(estimator, "coef_"):
        scores = abs(estimator.coef_)
        title = "Magnitud absoluta de los coeficientes lineales"
    else:
        st.info("Este modelo no expone una importancia de variables interpretable.")
        return
    names = preprocessor.get_feature_names_out()
    chart = pd.DataFrame({"variable": names, "importancia": scores})
    chart = chart.nlargest(12, "importancia").sort_values("importancia")
    st.plotly_chart(
        px.bar(chart, x="importancia", y="variable", orientation="h", title=title),
        use_container_width=True,
    )


def show_project(project: str, model_path: Path) -> None:
    bundle = load_bundle(str(model_path))
    st.session_state["project"] = project
    st.subheader(project)
    st.write(MODEL_INFO[project]["description"])
    if bundle.get("task") == "forecasting":
        show_forecast_project(project, bundle)
        return

    st.caption(
        f"Datos limpios: {bundle['rows']['rows_after']:,} filas de "
        f"{bundle['rows']['rows_before']:,}; se retiraron "
        f"{bundle['rows']['rows_removed']:,} filas."
    )
    show_metrics(bundle)
    st.markdown("#### Prueba una prediccion")
    features = prediction_form(project, bundle)
    if features is not None:
        row = pd.DataFrame([features], columns=bundle["features"])
        prediction = float(bundle["pipeline"].predict(row)[0])
        if bundle["dataset"] == "california":
            st.success(
                f"Valor mediano estimado: **${prediction * 100_000:,.0f} USD** "
                f"({prediction:.2f} cientos de miles de USD)."
            )
        elif bundle["dataset"] == "airbnb":
            st.success(f"Precio publicado estimado: **€{prediction:,.2f} por noche**.")
        elif bundle["dataset"] == "insurance":
            st.success(f"Coste anual estimado: **${prediction:,.2f} USD**.")
        elif bundle["dataset"] == "ames":
            st.success(f"Precio de venta estimado: **${prediction:,.0f} USD**.")
    show_feature_chart(bundle)


def show_forecast_project(project: str, bundle: dict[str, Any]) -> None:
    forecast = bundle["forecast"]
    st.caption(
        f"Datos diarios: {bundle['rows']['rows_after']:,} observaciones. "
        f"Último día disponible: {forecast['latest_date']}."
    )
    show_metrics(bundle)

    variants = list(forecast["variants"])
    default_variant = forecast["best_variant"]
    with st.form(f"forecast-form-{bundle['dataset']}"):
        variant = st.selectbox(
            "Información meteorológica",
            variants,
            index=variants.index(default_variant),
            help=(
                "Compara el pronóstico basado solo en calendario e historial "
                "con la variante que añade temperatura o radiación."
            ),
        )
        days_ahead = st.slider(
            "Días hacia delante",
            min_value=1,
            max_value=MAX_FORECAST_DAYS,
            value=1,
        )
        exogenous_values: dict[str, float] = {}
        variant_info = forecast["variants"][variant]
        for feature, label in forecast["exogenous_labels"].items():
            if feature in variant_info["features"]:
                exogenous_values[feature] = st.number_input(
                    f"Valor previsto de {label}",
                    value=float(forecast["default_exogenous"][feature]),
                    help=(
                        "Para varios días, se usa este mismo valor meteorológico "
                        "en cada día del horizonte."
                    ),
                )
        submitted = st.form_submit_button("Pronosticar")

    if submitted:
        predictions = forecast_to_date(
            bundle,
            variant_label=variant,
            days_ahead=days_ahead,
            exogenous_values=exogenous_values,
        )
        final_date, final_value = predictions[-1]
        st.success(
            f"Estimacion para **{final_date:%d/%m/%Y}**: "
            f"**{final_value:,.1f} {forecast['unit']}**."
        )
        if days_ahead > 1:
            st.info(
                "El pronostico es recursivo: para estimar cada día futuro se usan "
                "tambien las predicciones anteriores."
            )
        history = pd.DataFrame(
            forecast["history"], columns=["fecha", "valor"]
        )
        history["fecha"] = pd.to_datetime(history["fecha"])
        historical_window = history.tail(45)
        predicted_frame = pd.DataFrame(
            {
                "fecha": [date for date, _ in predictions],
                "valor": [value for _, value in predictions],
            }
        )
        chart = pd.concat([historical_window, predicted_frame], ignore_index=True)
        chart["tipo"] = [
            "Historial"
        ] * len(historical_window) + ["Pronostico"] * len(predicted_frame)
        st.plotly_chart(
            px.line(
                chart,
                x="fecha",
                y="valor",
                color="tipo",
                markers=True,
                title=f"Historial reciente y pronostico ({forecast['unit']})",
            ),
            use_container_width=True,
        )


st.title("Machine Learning Al Alfa")
st.write(
    "Proyecto de regresion de principio a fin: compara un baseline con modelos "
    "entrenados y explora predicciones de forma interactiva."
)
st.info(
    "Prototipo didactico con estimaciones historicas. No debe usarse para fijar "
    "precios reales sin validacion adicional."
)

available_projects = [
    name
    for name, info in MODEL_INFO.items()
    if (ARTIFACTS_DIR / f"{info['key']}_model.joblib").exists()
]
if not available_projects:
    st.error(
        "No hay modelos entrenados. Desde la carpeta del proyecto ejecuta "
        "`python -m ml_al_alfa.train --dataset california`."
    )
    st.stop()

selected_project = st.sidebar.radio("Elige el proyecto", available_projects)
selected_key = MODEL_INFO[selected_project]["key"]
show_project(selected_project, ARTIFACTS_DIR / f"{selected_key}_model.joblib")

with st.expander("Metricas de todos los modelos"):
    metrics_bundle = load_bundle(
        str(ARTIFACTS_DIR / f"{selected_key}_model.joblib")
    )
    table = pd.DataFrame(metrics_bundle["metrics"]).T
    table.index.name = "Modelo"
    table = table.rename(columns={"mae": "MAE (menor es mejor)", "r2": "R2"})
    st.dataframe(table, use_container_width=True)
