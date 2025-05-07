import pandas as pd

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Limpia y transforma los datos."""
    # Eliminar columnas
    df.drop(columns=['Booking_ID'], inplace=True)
    df.dropna(inplace=True)
    df = df[(df['avg_price_per_room'] >= 0) & (df['avg_price_per_room'] <= 5000)]

    # Manejo de Outliers (opcional, basado en tu EDA)
    Q1 = df['avg_price_per_room'].quantile(0.25)
    Q3 = df['avg_price_per_room'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df['avg_price_per_room'] <= Q3 + 1.5 * IQR]

    # Codificación One-Hot
    df = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved',
                                     'market_segment_type', 'booking_status'], drop_first=True)

    # Crear nuevas características
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df.drop(columns=['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights'], inplace=True)
    
    # Eliminar filas duplicadas
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, file_path):
    """Guarda el DataFrame en un archivo CSV."""
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    RAW_DATA_PATH = "C:/Users/JUAN/Desktop/BOOTCAMP - DATA SCIENCE/Ejercicios Juan/HotelPricePredictor_proyecto_final_ML/data/raw/Hotel Reservations.csv"
    PROCESSED_DATA_PATH = "C:/Users/JUAN/Desktop/BOOTCAMP - DATA SCIENCE/Ejercicios Juan/HotelPricePredictor_proyecto_final_ML/data/processed/hotel_reservations_clean.csv"
    
    df = load_data(RAW_DATA_PATH)
    df_cleaned = clean_data(df)
    save_data(df_cleaned, PROCESSED_DATA_PATH)
    print(f"Data processed and saved to {PROCESSED_DATA_PATH}")