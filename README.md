Optimus Price - Recomendador de Precios para Hoteles


🌟 Introducción
Optimus Price es una innovadora herramienta de recomendación de precios basada en Machine Learning, diseñada para ayudar a los administradores de pequeños y medianos hoteles a optimizar sus precios y maximizar sus ingresos.
Los hoteles enfrentan el reto constante de ajustar sus tarifas sin depender de terceros como plataformas de reservas online, que cobran comisiones elevadas. Optimus Price empodera a los hoteleros al proporcionar precios estratégicos y automatizados, permitiendo una mayor rentabilidad sin perder autonomía.


🎯 Objetivo del Proyecto
El propósito de Optimus Price es ofrecer a los hoteles una solución tecnológica capaz de analizar factores clave como demanda, temporada y competencia, proporcionando recomendaciones de precios óptimas en tiempo real.


🏗️ Estructura del Proyecto
El proyecto ha sido desarrollado en varias etapas clave:
- Obtención de Datos
- Descarga del dataset Hotel Reservations.csv desde Kaggle utilizando kagglehub.
- Limpieza y Análisis Exploratorio (EDA)
- Eliminación de valores faltantes.
- Codificación de variables categóricas.
- Análisis de distribuciones y relaciones entre variables.
- Ingeniería de Características
- Creación de atributos como total_guests y total_nights para mejorar el poder predictivo del modelo.
- Entrenamiento del Modelo
- Modelo de Random Forest para la predicción de precios.
- Optimización mediante validación cruzada con Optuna.
- Evaluación del Modelo
- Métricas clave: RMSE, MAE y R².
- Persistencia del Modelo
- Guarda el modelo y el objeto scaler para futuras predicciones.


📂 Estructura del Repositorio
📦 OptimusPrice
├── data/                 # Datos del proyecto
│   ├── raw/              # Datos originales
│   ├── processed/        # Datos preprocesados
│   ├── train/            # Datos de entrenamiento
│   ├── test/             # Datos de prueba
├── notebooks/            # Notebooks de Jupyter
│   ├── 01_Fuentes.ipynb  # Obtención de datos
│   ├── 02_LimpiezaEDA.ipynb # Limpieza y EDA
│   ├── 03_Entrenamiento_Evaluacion.ipynb # Entrenamiento y evaluación
├── src/                  # Scripts de Python
│   ├── data_processing.py # Preprocesamiento de datos
│   ├── training.py       # Entrenamiento del modelo
│   ├── evaluation.py     # Evaluación del modelo
├── models/               # Modelos entrenados
├── app_streamlit/        # Aplicación Streamlit
├── docs/                 # Documentación
└── README.md             # Archivo README


⚙️ Tecnología Utilizada
- Python 3.8+
- Pandas, NumPy, Scikit-learn, Optuna
- Streamlit (Aplicación interactiva)
- Kaggle API para la obtención de datos

🚀 Aplicación Optimus Price
La aplicación Optimus Price es una solución integral para hoteles pequeños y medianos. Con una interfaz intuitiva, los administradores pueden obtener recomendaciones de precios personalizadas con base en análisis de mercado y predicciones de demanda.

🌍 Impacto

Optimus Price ayuda a los hoteles a:

✅ Automatizar el ajuste de precios
✅ Reducir costos por comisiones
✅ Mejorar ingresos y competitividad
📈 Comparación con Plataformas de Reserva

| Escenario                          | Ingreso Bruto (€) | Comisión (%) | Ingreso Neto (€) | 
| Venta directa con Optimus Price    | 100               |      0%      |      100         | 
| Plataforma de reservas (Ej. OTA)   | 100               |     15%      |       85         | 
| Plataforma de reservas (Ej. OTA)   | 100               |     25%      |       75         | 
| Plataforma de reservas (Ej. OTA)   | 100               |     30%      |       70         | 


🎯 ¿Cómo Implementar Optimus Price?

1️⃣ Descargar el código desde GitHub
2️⃣ Ejecutar la aplicación Streamlit (streamlit run app_streamlit/main.py)
3️⃣ Cargar datos y obtener recomendaciones de precios

🔗 Contribución y Contacto
¡Nos encantaría recibir aportes y sugerencias! Si deseas contribuir, por favor, abre un pull request o ponte en contacto con nosotros.

📩 Email: contacto@optimusprice.com
🌎 Sitio Web: Optimus Price

