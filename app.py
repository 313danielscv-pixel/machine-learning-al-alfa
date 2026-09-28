from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from ml_al_alfa.train import ARTIFACTS_DIR

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
}


@st.cache_resource
def load_bundle(path_string: str) -> dict[str, Any]:
    return joblib.load(path_string)


def show_metrics(bundle: dict[str, Any]) -> None:
    metrics = bundle["metrics"]
    baseline = metrics["BaselineMedia"]["mae"]
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
        f"Evaluacion reproducible: test aleatorio del 20% (random_state="
        f"{bundle['random_state']}). MAE calculado en el conjunto de test."
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
    with st.form(f"prediction-{info['key']}"):
        columns = st.columns(2)
        for index, feature in enumerate(bundle["features"]):
            column = columns[index % 2]
            label = feature.replace("_", " ").replace("MedInc", "ingreso mediano").title()
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
        else:
            st.success(f"Precio publicado estimado: **€{prediction:,.2f} por noche**.")
    show_feature_chart(bundle)


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
