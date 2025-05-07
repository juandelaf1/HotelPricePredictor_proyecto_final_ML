
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle  # Cambiamos joblib por pickle

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def create_pipeline():
    """Crea el pipeline de preprocesamiento y modelado."""

    # Define las columnas categóricas y numéricas
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
    numerical_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                      'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month',
                      'arrival_date', 'repeated_guest', 'no_of_previous_cancellations',
                      'no_of_previous_bookings_not_canceled', 'no_of_special_requests']

    # Preprocesamiento para columnas numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocesamiento para columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combinar los transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    # Definir el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Crear el pipeline completo
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline

def train_and_evaluate(pipeline, df):
    """Entrena y evalúa el modelo."""

    X = df.drop(columns=['avg_price_per_room'])
    y = df['avg_price_per_room']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return pipeline

def save_pipeline(pipeline, pipeline_path):
    """Guarda el pipeline entrenado usando pickle."""
    with open(pipeline_path, 'wb') as file:  # 'wb' para escribir en binario
        pickle.dump(pipeline, file)
    print(f"Pipeline saved to {pipeline_path}")

if __name__ == '__main__':
    PROCESSED_DATA_PATH = "C:/Users/JUAN/Desktop/BOOTCAMP - DATA SCIENCE/Ejercicios Juan/HotelPricePredictor_proyecto_final_ML/data/processed/hotel_reservations_clean.csv"
    PIPELINE_PATH = "C:/Users/JUAN/Desktop/BOOTCAMP - DATA SCIENCE/Ejercicios Juan/HotelPricePredictor_proyecto_final_ML/models/hotel_price_prediction_pipeline.pkl"

    df = load_data(PROCESSED_DATA_PATH)
    pipeline = create_pipeline()
    trained_pipeline = train_and_evaluate(pipeline, df)
    save_pipeline(trained_pipeline, PIPELINE_PATH)