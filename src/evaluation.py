import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np


def load_data(data_path):
 """Carga los datos de evaluación."""
 return pd.read_csv(data_path)


def load_model_and_scaler(model_path, scaler_path):
  """Carga el modelo y el scaler."""
  with open(model_path, 'rb') as f:
   model = pickle.load(f)
  with open(scaler_path, 'rb') as f:
   scaler = pickle.load(f)
  return model, scaler
 

def preprocess_data(df, scaler):
  """Preprocesa los datos para la evaluación."""
  X = df.drop(columns=['avg_price_per_room'], errors='ignore') # Maneja el caso de no tener la columna objetivo
  X_scaled = scaler.transform(X)
  return X_scaled
 

def evaluate_model(model, X, y_true):
  """Evalúa el modelo y devuelve las métricas."""
  y_pred = model.predict(X)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_true)
  return rmse, mae, r2
 

if __name__ == '__main__':
  # Ejemplo de uso
  EVAL_DATA_PATH = 'data/processed/hotel_reservations_clean.csv' # O un nuevo dataset de evaluación
  MODEL_PATH = 'models/hotel_price_model.pkl'
  SCALER_PATH = 'models/hotel_price_scaler.pkl'
 

  df = load_data(EVAL_DATA_PATH)
  model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
  X_scaled = preprocess_data(df, scaler)
  y_true = df['avg_price_per_room']
  rmse, mae, r2 = evaluate_model(model, X_scaled, y_true)
  print(f"Evaluation Results: RMSE: {rmse}, MAE: {mae}, R2: {r2}")