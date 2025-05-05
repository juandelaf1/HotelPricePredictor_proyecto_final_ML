import pandas as pd
import os


def load_data(data_path):
# Carga el dataset desde la ruta especificada.
 return pd.read_csv(data_path)


def clean_data(df):
# Limpia y preprocesa el DataFrame.
 df = df.copy() # Trabaja sobre una copia para no modificar el original


# Imputar valores faltantes (ejemplo)
 df['no_of_children'] = df['no_of_children'].fillna(0)


# Eliminar columnas (ejemplo)
 df = df.drop(columns=['Booking_ID'])


# Codificar categóricas
 df = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status'], drop_first=True)


# Crear características
 df['total_guests'] = df['no_of_adults'] + df['no_of_children']
 df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
 df = df.drop(columns=['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights'])


 return df


def save_data(df, output_path):
#Guarda el DataFrame procesado en la ruta especificada.
 df.to_csv(output_path, index=False)


if __name__ == '__main__':
# Ejemplo de uso
 RAW_DATA_PATH = 'data/raw/Hotel Reservations.csv'
 CLEAN_DATA_PATH = 'data/processed/hotel_reservations_clean.csv'

df = load_data(RAW_DATA_PATH)
df_clean = clean_data(df)
save_data(df_clean, CLEAN_DATA_PATH)
print(f"Data processed and saved to {CLEAN_DATA_PATH}")