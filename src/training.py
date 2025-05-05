import pandas as pd
 from sklearn.model_selection import train_test_split, GridSearchCV
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.preprocessing import StandardScaler
 from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 import pickle
 

 def load_data(data_path):
  """Carga los datos de entrenamiento."""
  return pd.read_csv(data_path)
 

 def train_model(df):
  """Entrena el modelo de Random Forest."""
  X = df.drop(columns=['avg_price_per_room'])
  y = df['avg_price_per_room']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
 

  param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [10, 20]
  }
  forest_model = RandomForestRegressor(random_state=42)
  grid_search = GridSearchCV(forest_model, param_grid, cv=3, scoring='neg_root_mean_squared_error')
  grid_search.fit(X_train_scaled, y_train)
  best_forest_model = grid_search.best_estimator_
 

  return best_forest_model, scaler, X_test_scaled, y_test
 

 def evaluate_model(model, X_test, y_test):
  """Evalúa el modelo y devuelve las métricas."""
  y_pred = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  return rmse, mae, r2
 

 def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
  """Guarda el modelo y el scaler."""
  with open(model_path, 'wb') as f:
  pickle.dump(model, f)
  with open(scaler_path, 'wb') as f:
  pickle.dump(scaler, f)
 

 if __name__ == '__main__':
  # Ejemplo de uso
  CLEAN_DATA_PATH = 'data/processed/hotel_reservations_clean.csv'
  MODEL_PATH = 'models/hotel_price_model.pkl'
  SCALER_PATH = 'models/hotel_price_scaler.pkl'
 

  df = load_data(CLEAN_DATA_PATH)
  model, scaler, X_test, y_test = train_model(df)
  rmse, mae, r2 = evaluate_model(model, X_test, y_test)
  print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
  save_model(model, scaler, MODEL_PATH, SCALER_PATH)
  print(f"Model and scaler saved to {MODEL_PATH} and {SCALER_PATH}")