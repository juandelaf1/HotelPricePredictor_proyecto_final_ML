# HotelPricePredictor


Este proyecto desarrolla un modelo de Machine Learning para predecir los precios de las habitaciones de hotel. Esto puede ayudar a los hoteles a optimizar sus precios y maximizar sus ingresos.

## Estructura del Repositorio

* `data/`: Contiene los datos.
    * `raw/`: Datos originales.
    * `processed/`: Datos limpios y preprocesados.
    * `train/`: Datos de entrenamiento.
    * `test/`: Datos de prueba.
* `notebooks/`: Notebooks de Jupyter.
    * `01_Fuentes.ipynb`: Obtención de datos.
    * `02_LimpiezaEDA.ipynb`: Limpieza y EDA.
    * `03_Entrenamiento_Evaluacion.ipynb`: Entrenamiento y evaluación del modelo baseline.
* `src/`: Scripts de Python.
    * `data_processing.py`: Preprocesamiento de datos.
    * `training.py`: Entrenamiento del modelo.
    * `evaluation.py`: Evaluación del modelo.
* `models/`: Modelos entrenados.
* `app_streamlit/`: Aplicación Streamlit.
* `docs/`: Documentación.
* `README.md`: Este archivo.

## Cómo Usar

1.  Asegúrate de tener Python 3.6 o superior instalado.
2.  Instala las dependencias: `pip install -r app_streamlit/requirements.txt`
3.  Ejecuta los scripts en orden:
    * `python src/data_processing.py`
    * `python src/training.py`
    * `python src/evaluation.py`
4.  Ejecuta la aplicación Streamlit: `streamlit run app_streamlit/app.py`

## Próximos Pasos

En la siguiente fase, exploraremos diferentes modelos, realizaremos una optimización de hiperparámetros y mejoraremos la aplicación Streamlit.