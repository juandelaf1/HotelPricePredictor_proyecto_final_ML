import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle  # Cambiamos joblib por pickle

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def load_pipeline(pipeline_path):
    """Carga el pipeline guardado usando pickle."""
    with open(pipeline_path, 'rb') as file:  # 'rb' para leer en binario
        pipeline = pickle.load(file)
    return pipeline

def evaluate_model(pipeline, df):
    """Evalúa el modelo usando el pipeline."""

    X = df.drop(columns=['avg_price_per_room'])
    y = df['avg_price_per_room']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = pipeline.predict(X_test)  # El pipeline ya incluye el preprocesamiento

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

if __name__ == '__main__':
    PROCESSED_DATA_PATH = "C:/Users/JUAN/Desktop/BOOTCAMP - DATA SCIENCE/Ejercicios Juan/HotelPricePredictor_proyecto_final_ML/data/processed/hotel_reservations_clean.csv"
    PIPELINE_PATH = "C:/Users/JUAN/Desktop/BOOTCAMP - DATA SCIENCE/Ejercicios Juan/HotelPricePredictor_proyecto_final_ML/models/hotel_price_prediction_pipeline.pkl"

    df = load_data(PROCESSED_DATA_PATH)
    pipeline = load_pipeline(PIPELINE_PATH)
    evaluate_model(pipeline, df)