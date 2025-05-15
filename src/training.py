import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import os
import numpy as np

# **1. Cargar datos**
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "../data/processed/hotel_reservations_clean.csv")
df = pd.read_csv(data_path)
X = df.drop(columns=['avg_price_per_room'])
y = df['avg_price_per_room']

# **2. Dividir en entrenamiento y prueba**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **3. Guardar train y test en CSV**
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv(r'C:\Users\JUAN\Desktop\BOOTCAMP - DATA SCIENCE\Ejercicios Juan\Optimus_Price_proyecto_final_ML\data\train\hotel_reservations_train_data.csv',
                  index=False)
test_data.to_csv(r'C:\Users\JUAN\Desktop\BOOTCAMP - DATA SCIENCE\Ejercicios Juan\Optimus_Price_proyecto_final_ML\data\test\hotel_reservations_test_data.csv',
                 index=False)

# **4. Definir Pipeline**
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# **5. Definir hiperparámetros reducidos para RandomizedSearchCV**
param_grid = {
    'model__n_estimators': [10, 50],
    'model__max_depth': [10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', None]
}

# Configuración de RandomizedSearchCV para evaluar menos combinaciones
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    cv=3,            # Reducimos el número de folds
    n_iter=10,       # Se evaluarán 10 combinaciones aleatorias
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

# **6. Obtener el mejor modelo**
best_model = random_search.best_estimator_
print("✅ Mejores hiperparámetros encontrados:", random_search.best_params_)

# **7. Guardar el modelo optimizado**
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_file = os.path.join(models_dir, "pipeline_trained_model.pkl")
joblib.dump(best_model, model_file)
print(f"✅ Modelo guardado en: {model_file}")