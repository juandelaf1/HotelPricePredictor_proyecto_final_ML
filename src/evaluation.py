import os
import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Obtenemos la ruta base a partir del script evaluation.py
base_path = os.path.dirname(__file__)
model_file = os.path.join(base_path, "..", "models", "pipeline_trained_model.pkl")

# Cargar el modelo entrenado
pipeline = joblib.load(model_file)

# Construir la ruta para los datos de prueba (asegúrate de que el nombre y la ubicación coincidan)
test_file = os.path.join(base_path, "..", "data", "test", "hotel_reservations_test_data.csv")
test_data = pd.read_csv(test_file)

# Separamos las variables y realizamos las predicciones
X_test = test_data.drop(columns=['avg_price_per_room'])
y_test = test_data['avg_price_per_room']
y_pred = pipeline.predict(X_test)

# Calculamos las métricas
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.2f}")