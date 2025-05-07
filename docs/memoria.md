Memoria del Proceso de Predicción de Precios de Hotel

Este proyecto se ha desarrollado en varias etapas, cada una con un propósito específico:

Obtención de Datos: Se descargó el dataset 'Hotel Reservations.csv' desde Kaggle utilizando la librería kagglehub.

Limpieza y Análisis Exploratorio (EDA): Se realizó una limpieza exhaustiva de los datos, incluyendo el manejo de valores faltantes, la eliminación de columnas irrelevantes y la codificación de variables categóricas. Además, se llevó a cabo un análisis exploratorio para entender las distribuciones de las variables y las relaciones entre ellas.

Ingeniería de Características: Se crearon nuevas características (como 'total_guests' y 'total_nights') para mejorar la capacidad predictiva del modelo.

Entrenamiento del Modelo: Se entrenó un modelo de Random Forest para predecir los precios de las habitaciones de hotel. Se utilizó la técnica de validación cruzada (Optuna) para ajustar los hiperparámetros del modelo y optimizar su rendimiento.

Evaluación del Modelo: Se evaluó el rendimiento del modelo utilizando métricas como el RMSE (Root Mean Squared Error), MAE (Mean Absolute Error) y R² (R-cuadrado).

Persistencia del Modelo: Finalmente, se guardó el modelo entrenado y el objeto scaler para su uso en futuras predicciones.

Este proceso garantiza que los datos estén limpios y preparados para el modelado, y que el modelo de predicción sea robusto y preciso.
