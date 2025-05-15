import pandas as pd
import numpy as np
import os

# **1. Definir ruta relativa**
base_path = os.path.dirname(__file__)  # Ubicación del script actual
data_path = os.path.join(base_path, "../data/raw/Hotel_Reservations.csv")

# **2. Cargar datos usando ruta relativa**
df = pd.read_csv(data_path)

# **3. Manejo de valores faltantes (solo en columnas numéricas)**
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# **4. Llenar valores faltantes en variables categóricas con el valor más frecuente**
for col in df.select_dtypes(include=['object']).columns:
    df.loc[:, col] = df[col].fillna(df[col].mode()[0])  # ✅ Solución `FutureWarning`

# **5. Eliminación de duplicados**
df.drop_duplicates(inplace=True)

# **6. Detección y eliminación de outliers en avg_price_per_room**
Q1 = df['avg_price_per_room'].quantile(0.25)
Q3 = df['avg_price_per_room'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['avg_price_per_room'] < Q1 - 1.5 * IQR) | (df['avg_price_per_room'] > Q3 + 1.5 * IQR))]

# **7. Eliminación de columnas irrelevantes**
df.drop(columns=['Booking_ID'], inplace=True)

# **9. Codificación de otras variables categóricas**
df = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status'])  # ✅ Eliminamos `drop_first`
if 'type_of_meal_plan_Meal Plan 3' not in df.columns:
    df['type_of_meal_plan_Meal Plan 3'] = 0  # Se crea con valor 0 si no existía en los datos

# **10. Creación de nuevas variables**
df['total_guests'] = df['no_of_adults'] + df['no_of_children']
df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

# **11. Eliminación de columnas originales ya procesadas**
df.drop(columns=['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights'], inplace=True)

# **12. Verificación final de columnas**
expected_columns = [
    'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest',
    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests',
    'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected',
    'room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4',
    'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7',
    'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline',
    'market_segment_type_Online', 'booking_status_Canceled', 'booking_status_Not_Canceled', 'total_guests', 'total_nights'
]

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    print(f"⚠️ Advertencia: Estas columnas faltan en el DataFrame después del procesamiento: {missing_cols}")

# **13. Guardar el dataset procesado**
processed_path = os.path.join(base_path, "../data/processed/hotel_reservations_clean.csv")
df.to_csv(processed_path, index=False)

print("✅ Datos preprocesados y guardados correctamente en:", processed_path)


print(df.columns.tolist())
