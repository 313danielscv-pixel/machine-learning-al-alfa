# Machine Learning Al Alfa

Proyecto de regresion de principio a fin: datos, limpieza, exploracion, baseline, modelos, evaluacion, modelo serializado y aplicacion de Streamlit. Incluye el ejercicio principal de viviendas de California y el reto adicional de precios de Airbnb en Madrid.

> Los modelos estiman precios con datos historicos y son material didactico. No constituyen tasaciones ni recomendaciones de precio. No se conectan con Airbnb ni con ningun sistema de reservas.

## Preguntas del proyecto

- **California (principal):** con las caracteristicas censales de una zona, estimar el valor mediano de las viviendas (`MedHouseVal`). El conjunto de scikit-learn procede del censo de 1990; la variable esta expresada en cientos de miles de USD y tiene un tope historico de 5.0 (500.000 USD). No representa el precio actual de una vivienda.
- **Airbnb Madrid (reto):** dados los atributos publicos de un anuncio, estimar su precio nocturno publicado en euros (`price`). Es una fotografia de anuncios, no de reservas completadas o precios realmente pagados.

## Requisitos del encargo cubiertos

- Limpieza: nulos, duplicados, conversion de tipos, rangos imposibles y recuento de filas antes/despues.
- Exploracion: al menos tres visualizaciones por conjunto; mapa de California con latitud y longitud.
- Evaluacion: baseline obligatorio de media, `LinearRegression` y `RandomForestRegressor`; MAE en las unidades del problema. En los dos problemas de regresion se usa una particion aleatoria reproducible train/test.
- Sin leakage: se excluyen identificadores y campos que derivan directamente de la respuesta.
- Preprocesamiento en pipeline: imputacion de nulos, escalado numerico y one-hot para categoricas.
- Serializacion del mejor estimador con joblib y app Streamlit con controles de entrada y graficos.
- Reto Airbnb: descarga desde Inside Airbnb, parseo de precios en texto, limpieza de extremos y una segunda prediccion en la app.

## Inicio rapido en Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Entrenar el proyecto principal (descarga el conjunto oficial de scikit-learn si aun no esta en cache):

```powershell
python -m ml_al_alfa.train --dataset california
```

Descargar y entrenar el reto Airbnb:

```powershell
python scripts\download_airbnb_madrid.py
python -m ml_al_alfa.train --dataset airbnb
```

Entrenar ambos:

```powershell
python -m ml_al_alfa.train --dataset all
```

Abrir la aplicacion:

```powershell
streamlit run app.py
```

La app muestra los formularios de los modelos que encuentre en `artifacts/`. El conjunto Airbnb se mantiene local porque es grande y se actualiza periodicamente. Los modelos generados tambien se guardan localmente y no se incluyen en Git por defecto.

## Notebook

`notebooks/california_housing.ipynb` desarrolla el recorrido principal con limpieza, graficos, comparacion de modelos y guardado del artefacto. `notebooks/airbnb_madrid.ipynb` desarrolla el reto adicional.

## Datos y licencia

- California Housing: cargado con `sklearn.datasets.fetch_california_housing`; ver la [documentacion de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) y la referencia del conjunto en esa documentacion.
- Airbnb Madrid: descarga detallada desde la pagina [Get the Data de Inside Airbnb](https://insideairbnb.com/get-the-data/). El descargador detecta la URL de Madrid publicada actualmente. Los datos de Inside Airbnb se ofrecen bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); consulta y respeta sus condiciones de atribucion antes de redistribuir datos.

## Estructura

```text
Machine Learning Al Alfa/
├── app.py
├── artifacts/                 # modelos y metricas locales generados
├── data/
│   ├── raw/                   # Airbnb descargado (no se sube a Git)
│   └── processed/
├── notebooks/
│   ├── california_housing.ipynb
│   └── airbnb_madrid.ipynb
├── scripts/download_airbnb_madrid.py
├── src/ml_al_alfa/
│   ├── data.py
│   └── train.py
└── tests/
```

## Interpretacion del resultado

El MAE comunica el error medio absoluto en cientos de miles de USD (y su equivalente aproximado en USD) para California y en EUR por noche para Airbnb. Compara cada modelo con el baseline: si no lo mejora en el test, no hay evidencia de valor predictivo frente a predecir la media. La diferencia entre el rendimiento del test y el uso real debe explicarse; una particion aleatoria de anuncios no evalua como cambiarian los precios con el tiempo.
