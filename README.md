# Machine Learning Al Alfa

Proyecto de *machine learning* de principio a fin: desde formular una pregunta y limpiar los datos hasta comparar modelos, guardar el mejor y probar predicciones en una aplicación de Streamlit.

Incluye proyectos de **viviendas de California**, **precios de Airbnb en Madrid**, **seguro médico**, **viviendas de Ames**, **nacimientos en EE. UU.**, **demanda eléctrica en España** y **energía solar en España**. Entrena cada conjunto para generar su modelo y habilitarlo en Streamlit.

> **Alcance:** son estimaciones educativas basadas en datos históricos, no tasaciones ni recomendaciones comerciales. El conjunto de California procede del censo de 1990; los anuncios de Airbnb muestran precios publicados, no reservas ni importes pagados.

## Resultados principales

En regresión usamos una partición aleatoria 80/20 (`random_state=42`); en forecasting reservamos el 20% más reciente, sin mezclar las fechas. Comparamos un baseline con regresión lineal y Random Forest. La tabla resume esta ejecución; los datos energéticos se descargan de fuentes que se actualizan.

| Proyecto | Filas utilizadas | MAE baseline | MAE mejor modelo | Equivalencia sencilla del error medio |
|---|---:|---:|---:|---|
| California Housing | 20.640 | 0,906 | **0,327** | **≈ 33.000 USD** por zona censal |
| Airbnb Madrid | 18.555 de 22.708 | 64,97 € | **35,54 €** | **≈ 36 €** por noche |
| Seguro médico | 1.337 | 9.861,80 USD | **2.475,58 USD** | **≈ 2.500 USD** al año |
| Viviendas de Ames | 2.927 | 59.601,56 USD | **18.499,45 USD** | **≈ 18.500 USD** por casa |
| Nacimientos en EE. UU. | 5.465 | 447,23 | **222,60** | **≈ 223 nacimientos** por día |
| Demanda eléctrica en España | 2.078 | 30,52 GWh | **16,47 GWh** | **≈ 16 GWh** por día |
| Energía solar en España | 1.348 | 27,51 GWh | **17,33 GWh** | **≈ 17 GWh** por día |

En California, `0,327` equivale aproximadamente a `32.700 USD` porque el objetivo original está en cientos de miles de dólares. El seguro se mide en dólares anuales, Ames en dólares por casa, los nacimientos en bebés por día y los dos pronósticos de energía en GWh diarios.

### ¿Qué significa MAE?

MAE (*Mean Absolute Error*, o **error absoluto medio**) indica, en promedio, cuánto se alejan las predicciones de los valores observados:

- **California:** el modelo se equivoca en unos **33.000 USD de media** al estimar el valor mediano histórico de una zona censal. No significa que cada vivienda concreta se desvíe exactamente esa cantidad.
- **Airbnb:** las estimaciones se desvían unos **36 € por noche de media** respecto al precio publicado del anuncio.
- **Seguro y Ames:** el error medio es de unos **2.476 USD por año** y **18.499 USD por casa**, respectivamente.
- **Nacimientos, demanda y solar:** el error medio es de unos **223 nacimientos**, **16 GWh** y **17 GWh** por día, respectivamente.
- **Cuanto menor sea el MAE, mejor.** En regresión el baseline predice la media del entrenamiento; en forecasting predice el valor de hace siete días. En esta ejecución, el mejor modelo de cada proyecto supera su baseline.

| Proyecto | MAE del mejor modelo | MAE del baseline | Reducción del MAE frente al baseline |
|---|---:|---:|---:|
| California Housing | 0,327 | 0,906 | ≈ 64 % |
| Airbnb Madrid | 35,54 € | 64,97 € | ≈ 45 % |
| Seguro médico | 2.475,58 USD | 9.861,80 USD | ≈ 75 % |
| Viviendas de Ames | 18.499,45 USD | 59.601,56 USD | ≈ 69 % |
| Nacimientos en EE. UU. | 222,60 | 447,23 | ≈ 50 % |
| Demanda eléctrica en España | 16,47 GWh | 30,52 GWh | ≈ 46 % |
| Energía solar en España | 17,33 GWh | 27,51 GWh | ≈ 37 % |

Los errores y mejoras corresponden a una partición de prueba reproducible de estos conjuntos concretos. **No garantizan el mismo resultado con otros periodos, otras ciudades ni en uso real.**

**En sencillo:** el MAE es el tamaño típico del error: unos 33.000 USD por zona en California, 36 € por noche en Airbnb y las cantidades de la tabla para los demás proyectos. Son promedios de prueba, no garantías para cada predicción.

## 1. Viviendas de California — proyecto principal

**Pregunta:** dadas las características censales de una zona, ¿cuál era su valor mediano de vivienda?

- **Objetivo (`y`):** `MedHouseVal`, en cientos de miles de USD.
- **Datos:** 20.640 observaciones cargadas mediante `fetch_california_housing` de scikit-learn.
- **Predictores:** ingreso mediano, antigüedad de las viviendas, habitaciones y dormitorios medios, población, ocupación media y coordenadas geográficas.
- **Limpieza:** se revisan duplicados, valores ausentes, tipos y rangos plausibles; se conservaron las 20.640 filas en este conjunto.
- **Análisis:** distribución de precios, relación entre ingresos y valor, y **mapa de precios por latitud y longitud**.
- **Limitación importante:** los datos representan el censo de 1990 y el objetivo está limitado históricamente en 5,0 (500.000 USD). El resultado no representa el mercado inmobiliario actual ni una tasación de una vivienda concreta.

**En sencillo:** usamos datos antiguos de distintas zonas de California para estimar el valor mediano de sus viviendas. No estamos estimando una casa concreta ni su precio actual.

## 2. Airbnb Madrid — reto adicional

**Pregunta:** a partir de las características públicas de un anuncio, ¿qué precio por noche tiene publicado?

- **Objetivo (`y`):** `price`, en euros por noche.
- **Fuente:** captura pública de Inside Airbnb; el descargador busca el enlace de Madrid vigente en la [página oficial de descargas](https://insideairbnb.com/get-the-data/).
- **Datos usados en esta ejecución:** 22.708 anuncios brutos; quedaron 18.555 después de limpiar.
- **Limpieza:** conversión de precios con formato de texto como `"$125.00"`, eliminación de duplicados, precios fuera del intervalo elegido de 10–500 €, coordenadas fuera de Madrid y valores imposibles en variables numéricas.
- **Predictores:** capacidad, dormitorios, camas, noches mínimas, disponibilidad, reseñas, valoración, cantidad de anuncios del anfitrión, tipo de habitación, barrio, condición de superanfitrión y ubicación.
- **Evitar fuga de información:** no se usan el precio como entrada ni identificadores u otros campos que revelen directamente la respuesta.
- **Limitación:** es una fotografía de precios publicados, no de reservas completadas, ocupación futura ni precio finalmente pagado. Al ser una captura temporal, los resultados pueden cambiar al descargar otra.

Inside Airbnb indica que sus datos se ofrecen bajo [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Atribuye la fuente y revisa las condiciones de la licencia antes de redistribuir datos.

**En sencillo:** el modelo aprende de anuncios publicados y estima el precio por noche que podría mostrar un anuncio parecido. No sabe si alguien reservó ni cuánto terminó pagando.

## Otros proyectos de la guía

Estos conjuntos también se pueden entrenar desde la línea de comandos y usar en la app. En seguro médico, `sex` y `smoker` se convierten a 0/1; en Ames se selecciona un grupo pequeño de columnas útiles en vez de alimentar el modelo con las 80 disponibles:

- **Seguro médico (regresión):** estimar el coste anual del seguro de una persona a partir de su edad, índice de masa corporal, número de hijos y otros datos como si fuma.
  **En sencillo:** con datos de clientes anteriores, calcular cuánto podría costar el seguro de una persona con características parecidas.
- **Viviendas de Ames, Iowa (regresión):** estimar el precio de venta de una casa usando características como superficie, calidad, año de construcción y garaje.
  **En sencillo:** predecir el precio de una casa a partir de cómo es, usando ventas pasadas como ejemplos.
- **Nacimientos en EE. UU. (forecasting):** estimar cuántos bebés nacerán cada día, usando el historial diario de 2000 a 2014, el calendario, los festivos federales y los viernes 13.
  **En sencillo:** mirar los nacimientos de días anteriores y los patrones de cada día de la semana para estimar los de días futuros.
- **Demanda eléctrica en España (forecasting):** estimar el consumo eléctrico diario en GWh a partir del historial, el calendario y, como prueba adicional, la temperatura de Madrid.
  **En sencillo:** usar cuánto consumió España antes y si hace más frío o calor para estimar la electricidad que podría necesitar.
- **Energía solar en España (forecasting):** estimar la producción fotovoltaica diaria en GWh usando el historial y, como prueba adicional, la radiación solar de Madrid.
  **En sencillo:** tener en cuenta la energía solar producida antes y cuánta radiación hay para estimar la producción de un día futuro.

En forecasting, el orden de las fechas importa: se entrena con fechas anteriores y se prueba con las más recientes, sin mezclarlas al azar. El baseline es usar como predicción el valor de hace siete días. En la app puedes elegir una variante sin dato meteorológico o con temperatura/radiación y pronosticar hasta 14 días. Para varios días, el valor meteorológico que introduces se mantiene igual durante todo el horizonte.

En las métricas históricas, la variante meteorológica utiliza la temperatura o radiación observada en cada fecha de prueba. Para pronosticar fechas futuras, la app pide introducir un valor meteorológico estimado; no obtiene automáticamente el pronóstico del tiempo.

**En sencillo:** regresión estima una cantidad a partir de las características de un caso. Forecasting estima valores futuros usando el historial; por eso reserva las fechas más recientes para comprobar el resultado en vez de mezclar los días al azar.

## Recorrido de machine learning

1. **Datos y pregunta:** definimos la población, el uso previsto y la variable que queremos estimar.
2. **Limpieza:** convertimos las columnas al tipo adecuado y revisamos duplicados, datos ausentes, valores imposibles y extremos. Registramos filas antes y después.
3. **Análisis visual:** exploramos distribuciones y relaciones entre las variables; California incluye un mapa geográfico.
4. **Separación entrenamiento/prueba:** en regresión se usa una partición aleatoria reproducible de 80/20; en forecasting, el 20% de fechas más recientes se reserva como test y no se baraja.
5. **Baseline:** en regresión, `DummyRegressor` predice la media del entrenamiento; en forecasting, se usa el valor de siete días antes.
6. **Modelos:** comparamos `LinearRegression` y `RandomForestRegressor`.
7. **Preprocesamiento:** los proyectos de regresión imputan valores ausentes, escalan variables numéricas y codifican categorías. Forecasting crea variables de calendario e historial; demanda y energía solar permiten añadir temperatura o radiación.
8. **Evaluación:** MAE y R² se calculan sobre el test reservado. Se selecciona el modelo con menor MAE y se comprueba si mejora al baseline.
9. **Uso:** se guarda el modelo con joblib. La app permite cambiar los atributos de una regresión o generar un pronóstico futuro paso a paso.

**En sencillo:** guardamos ejemplos para probar el modelo. Para precios de casas o seguros los separamos al azar; para pronosticar el futuro conservamos los últimos días en orden. Después comparamos el resultado con una predicción simple.

## Aplicación interactiva

La app muestra los conjuntos cuyo modelo ya está entrenado. En California, Airbnb, seguro médico y Ames permite cambiar los atributos de entrada; en nacimientos, demanda eléctrica y energía solar permite elegir el horizonte y hacer un pronóstico. También muestra el MAE, el baseline, las métricas comparadas y un gráfico apropiado al tipo de proyecto.

Las predicciones son exploratorias. La importancia de variables de Random Forest describe el ajuste del modelo, no demuestra causalidad.

**En sencillo:** elige un proyecto, cambia los valores del formulario y pulsa **Estimar precio**. La app muestra la estimación y algunas medidas para entender qué tan bien funcionó el modelo durante su evaluación.

## Empezar en Windows PowerShell

Desde la carpeta del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

El entorno del proyecto está en `.venv`; actívalo antes de ejecutar los comandos siguientes.

Entrenar California:

```powershell
python -m ml_al_alfa.train --dataset california
```

Descargar y entrenar Airbnb Madrid:

```powershell
python scripts\download_airbnb_madrid.py
python -m ml_al_alfa.train --dataset airbnb
```

Entrenar seguro médico, viviendas de Ames, nacimientos, demanda eléctrica o energía solar:

```powershell
python -m ml_al_alfa.train --dataset insurance
python -m ml_al_alfa.train --dataset ames
python -m ml_al_alfa.train --dataset births
python -m ml_al_alfa.train --dataset demand
python -m ml_al_alfa.train --dataset solar
```

Entrenar los siete conjuntos:

```powershell
python -m ml_al_alfa.train --dataset all
```

El entrenamiento descarga los datos públicos de seguro, Ames y nacimientos cuando hacen falta. Para demanda y energía solar consulta Red Eléctrica y Open-Meteo. Los CSV se guardan en `data/raw/`; los modelos aparecen en `artifacts/`. `all` entrena los siete conjuntos y necesita acceso a Internet; Airbnb requiere primero ejecutar su descargador.

Iniciar la aplicación:

```powershell
streamlit run app.py
```

Ejecutar pruebas:

```powershell
python -m pytest tests -q
```

El descargador de Airbnb guarda `listings.csv.gz` en `data/raw/`. Los datos descargados y los modelos se excluyen de Git para mantener ligero el repositorio. Tras clonar el proyecto, entrena cada conjunto para crear su modelo en `artifacts/` y hacer que aparezca en la app.

**En sencillo:** prepara el entorno una vez, ejecuta el comando de entrenamiento para los proyectos que quieras y después inicia Streamlit. Si ya existe un archivo `*_model.joblib` en `artifacts/`, ese proyecto aparecerá en la app.

## Notebooks y código

- [`notebooks/california_housing.ipynb`](notebooks/california_housing.ipynb): desarrollo guiado del problema principal, con limpieza, gráficos, mapa, modelos y MAE.
- [`notebooks/airbnb_madrid.ipynb`](notebooks/airbnb_madrid.ipynb): descarga, limpieza, exploración, evaluación y reto adicional.
- [`app.py`](app.py): interfaz Streamlit para probar estimaciones.
- [`src/ml_al_alfa/data.py`](src/ml_al_alfa/data.py): limpieza, selección de variables y valores por defecto.
- [`src/ml_al_alfa/datasets.py`](src/ml_al_alfa/datasets.py): descarga y preparación de fuentes públicas.
- [`src/ml_al_alfa/forecasting.py`](src/ml_al_alfa/forecasting.py): variables de calendario e historial, evaluación cronológica y pronósticos futuros.
- [`src/ml_al_alfa/train.py`](src/ml_al_alfa/train.py): carga de datos, pipelines, baseline, entrenamiento, métricas y serialización.
- [`scripts/download_airbnb_madrid.py`](scripts/download_airbnb_madrid.py): encuentra la captura más reciente de Madrid publicada en Inside Airbnb y la descarga.
- [`tests/test_data.py`](tests/test_data.py): pruebas de limpieza y conversión de precios.
- [`tests/test_forecasting.py`](tests/test_forecasting.py): pruebas de separación temporal y pronósticos.

**En sencillo:** `datasets.py` consigue los datos, `train.py` entrena los modelos, `forecasting.py` prepara las series temporales y `app.py` deja probar las predicciones. Las pruebas comprueban que el tratamiento de datos y el pronóstico respeten lo esperado.

## Archivos principales

```text
Machine Learning Al Alfa/
├── app.py
├── notebooks/
│   ├── california_housing.ipynb
│   └── airbnb_madrid.ipynb
├── scripts/download_airbnb_madrid.py
├── src/ml_al_alfa/
│   ├── data.py
│   ├── datasets.py
│   ├── forecasting.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   └── test_forecasting.py
├── data/raw/       # datos descargados, ignorados por Git
└── artifacts/      # modelos y métricas generados, ignorados por Git
```

**En sencillo:** `src/` contiene el código del proyecto, `notebooks/` los cuadernos explicativos, `data/` los datos descargados y `artifacts/` los modelos entrenados que usa la app.

## Recursos de datos

- [California Housing — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Descargas de Inside Airbnb](https://insideairbnb.com/get-the-data/)
- [Seguro médico — CSV público](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv)
- [Viviendas de Ames — CSV público](https://github.com/wblakecannon/ames/blob/master/data/housing.csv)
- [Nacimientos en EE. UU. — FiveThirtyEight](https://github.com/fivethirtyeight/data/blob/master/births/US_births_2000-2014_SSA.csv)
- [API de datos de Red Eléctrica](https://www.ree.es/es/datos/apidatos)
- [Archivo histórico de Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api)
- [Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**En sencillo:** los enlaces indican de dónde vienen los datos públicos; la demanda y la energía solar se consultan a Red Eléctrica y se combinan con datos meteorológicos de Open-Meteo.
