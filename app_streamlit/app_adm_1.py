import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
import os
import csv

# ----------------- CONFIGURACIÓN GENERAL -----------------
st.set_page_config(page_title="Optimus Price Advisor", layout="centered")
page_bg_gradient = """
<style>
.stApp {
    background: linear-gradient(to bottom, #E0E0E0, #A9A9A9);
}
</style>
"""
st.markdown(page_bg_gradient, unsafe_allow_html=True)

# ----------------- RUTAS DE ARCHIVOS -----------------
DATA_FILE = "hotel_reservations.csv"  # donde se guardarán las reservas
MODEL_FILE = r"C:\Users\JUAN\Desktop\BOOTCAMP - DATA SCIENCE\Ejercicios Juan\Optimus_Price_proyecto_final_ML\models\trained_model_1.pkl"
SCALER_FILE = r"C:\Users\JUAN\Desktop\BOOTCAMP - DATA SCIENCE\Ejercicios Juan\Optimus_Price_proyecto_final_ML\models\scaler_trained_model_1.pkl"
OVERRIDE_FILE = "price_override.txt"  # para guardar el ajuste manual

# ----------------- FUNCIONES -----------------
def save_to_csv(data):
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# La función build_input_data arma el vector de entrada (25 features)
def build_input_data(required_car_parking, lead_time, fecha_llegada, 
                     repeated_guest, special_requests_count, meal_plan, room_type, total_guests, total_nights):
    # Derivar datos de la fecha
    arrival_year = fecha_llegada.year
    arrival_month = fecha_llegada.month
    arrival_date = fecha_llegada.day
    # En esta versión original se usa "repeated_guest" en la posición 6
    no_prev_cancel = 0
    no_prev_bookings = 0
    if meal_plan == "Todo incluido":
        meal_plan_2, meal_plan_3, meal_not_selected = 1, 1, 0
    else:
        meal_plan_2 = int(meal_plan == "Desayuno incluido")
        meal_plan_3 = int(meal_plan == "Cena incluida")
        meal_not_selected = int(meal_plan == "Ninguno")
    # Para el tipo de habitación (se ignora la categoría "Predeterminado")
    rt2 = int(room_type == "Individual")
    rt3 = int(room_type == "Doble")
    rt4 = int(room_type == "Twin")
    rt5 = int(room_type == "Triple")
    rt6 = int(room_type == "Suite")
    rt7 = int(room_type == "Familiar")
    # Variables de mercado: se asumen 0 para las que no sean elegidas (y "Predeterminado" se implementa como 0)
    ms_compl = 0
    ms_corp = 0
    ms_off = 0
    ms_onl = 0
    booking_ok = 1
    data = np.array([[
        int(required_car_parking),             # 1. Estacionamiento
        lead_time,                             # 2. Lead time
        arrival_year,                          # 3. Año de llegada
        arrival_month,                         # 4. Mes de llegada
        arrival_date,                          # 5. Día de llegada
        int(repeated_guest),                   # 6. Cliente recurrente
        no_prev_cancel,                        # 7. Reservas canceladas
        no_prev_bookings,                      # 8. Reservas cumplidas
        special_requests_count,                # 9. Solicitudes especiales
        meal_plan_2,                           # 10. Desayuno incluido
        meal_plan_3,                           # 11. Cena incluida
        meal_not_selected,                     # 12. Sin plan de comidas
        rt2,                                   # 13. Individual
        rt3,                                   # 14. Doble
        rt4,                                   # 15. Twin
        rt5,                                   # 16. Triple
        rt6,                                   # 17. Suite
        rt7,                                   # 18. Familiar
        ms_compl,                              # 19. Complementario
        ms_corp,                               # 20. Corporativo
        ms_off,                                # 21. Offline
        ms_onl,                                # 22. Online
        int(booking_ok),                       # 23. Booking OK
        total_guests,                          # 24. Número de huéspedes
        total_nights                           # 25. Número de noches
    ]])
    return data

# Para la sección de recomendaciones, se permite modificar parámetros de referencia.
def build_input_mod(required_car_parking_mod, lead_time_mod, mes_mod, special_requests_mod, 
                    meal_plan_mod, room_type_mod, total_guests_mod, total_nights_mod, fecha_ref):
    # Utiliza la fecha de referencia y modifica el mes
    try:
        fecha_mod = fecha_ref.replace(month=mes_mod)
    except ValueError:
        fecha_mod = fecha_ref.replace(month=mes_mod, day=28)
    arrival_year_mod = fecha_mod.year
    arrival_date_mod = fecha_mod.day
    arrival_day_of_week_mod = fecha_mod.weekday()
    arrival_week_number_mod = fecha_mod.isocalendar()[1]
    if meal_plan_mod == "Todo incluido":
        m_plan2, m_plan3, m_not_selected = 1, 1, 0
    else:
        m_plan2 = int(meal_plan_mod == "Desayuno incluido")
        m_plan3 = int(meal_plan_mod == "Cena incluida")
        m_not_selected = int(meal_plan_mod == "Ninguno")
    rt2_mod = int(room_type_mod == "Individual")
    rt3_mod = int(room_type_mod == "Doble")
    rt4_mod = int(room_type_mod == "Twin")
    rt5_mod = int(room_type_mod == "Triple")
    rt6_mod = int(room_type_mod == "Suite")
    rt7_mod = int(room_type_mod == "Familiar")
    # Variables de mercado (por defecto)
    ms_compl_mod = 0
    ms_corp_mod = 0
    ms_off_mod = 0
    ms_onl_mod = 0
    booking_ok_mod = 1
    data_mod = np.array([[
        int(required_car_parking_mod),   # index 0
        lead_time_mod,                   # index 1
        arrival_year_mod,                # index 2
        mes_mod,                         # index 3
        arrival_date_mod,                # index 4
        0,                               # repeated_guest (se asume 0 para análisis)
        0,                               # no_prev_cancel
        0,                               # no_prev_bookings
        special_requests_mod,            # index 8
        m_plan2,                         # index 9
        m_plan3,                         # index 10
        m_not_selected,                  # index 11
        rt2_mod,                         # index 12
        rt3_mod,                         # index 13
        rt4_mod,                         # index 14
        rt5_mod,                         # index 15
        rt6_mod,                         # index 16
        rt7_mod,                         # index 17
        ms_compl_mod,                    # index 18
        ms_corp_mod,                     # index 19
        ms_off_mod,                      # index 20
        ms_onl_mod,                      # index 21
        booking_ok_mod,                  # index 22
        total_guests_mod,                # index 23
        total_nights_mod                 # index 24
    ]])
    return data_mod

# ----------------------- MENÚ DE ROLES Y PÁGINAS -----------------------
role = st.sidebar.selectbox("Selecciona el rol", ["Cliente", "Administrador"])
show_admin = False
if role == "Administrador":
    admin_password = st.sidebar.text_input("Contraseña de Administrador", type="password")
    if admin_password:
        if admin_password == "admin123":  # La contraseña es "admin123"
            st.sidebar.success("Acceso de administrador concedido")
            show_admin = True
        else:
            st.sidebar.error("Contraseña incorrecta.")

page = st.sidebar.selectbox("Seleccione la Página", ["Reservas", "Recomendaciones"])

# Si el rol es administrador, se muestra un panel adicional en el sidebar:
if show_admin:
    st.sidebar.markdown("### Configuración de Precio Manual")
    manual_modifier = st.sidebar.slider("Ajuste manual de precio (%)", -20, 30, 0, step=1)
    if st.sidebar.button("Guardar Ajuste Manual"):
        with open(OVERRIDE_FILE, "w") as f:
            f.write(str(manual_modifier))
        st.sidebar.success("Ajuste manual guardado.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Historial de Reservas")
    if os.path.exists(DATA_FILE):
        hist = pd.read_csv(DATA_FILE)
        st.sidebar.dataframe(hist.tail(5))
        csv_hist = hist.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Descargar historial", data=csv_hist, file_name="historial_reservas.csv", mime="text/csv")
    else:
        st.sidebar.info("No hay reservas aún.")

# ----------------------- CARGA DEL MODELO Y ESCALADOR -----------------------
modelo = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

st.image(r"C:\Users\JUAN\Desktop\BOOTCAMP - DATA SCIENCE\Ejercicios Juan\Optimus_Price_proyecto_final_ML\docs\img\optimus_price_logo.jpg", use_container_width=True)
st.title("🏨 Optimus Price Advisor")
st.markdown("##### Complete los datos de la reserva para analizar estrategias de precios:")

# ----------------------- PÁGINA: RESERVAS -----------------------
if page == "Reservas":
    st.header("Formulario de Reserva")
    col_pers1, col_pers2 = st.columns(2)
    with col_pers1:
        nombre = st.text_input("Nombre completo*", help="Nombre y apellidos")
        email = st.text_input("Email*", help="Email de contacto")
        telefono = st.text_input("Teléfono*", help="Número de contacto")
    with col_pers2:
        documento = st.text_input("Documento de identidad*", help="DNI, pasaporte u otra identificación")
        nacionalidad = st.text_input("Nacionalidad", "Española")
        vip_status = st.selectbox("Tipo de cliente", ["Normal", "VIP", "Corporativo"])
    
    st.header("Detalles de la Reserva")
    today = date.today()
    col_fechas1, col_fechas2 = st.columns(2)
    with col_fechas1:
        fecha_llegada = st.date_input("Fecha de llegada*", min_value=datetime(2025,1,1), max_value=datetime(2027,12,31))
    with col_fechas2:
        fecha_salida = st.date_input("Fecha de salida*", min_value=fecha_llegada)
    lead_time = (fecha_llegada - today).days
    total_nights = (fecha_salida - fecha_llegada).days
    st.info(f"""📊 Resumen:
- Anticipación: {lead_time} días
- Duración: {total_nights} noches
- Temporada: {"Alta" if fecha_llegada.month in [6,7,8,12] else "Media" if fecha_llegada.month in [4,5,9,10,11] else "Baja"}""")
    
    st.header("Preferencias de Hospedaje")
    col_pref1, col_pref2 = st.columns(2)
    with col_pref1:
        st.markdown("**Servicios:**")
        meal_2 = st.radio("Plan de comidas*", ['Ninguno', 'Desayuno incluido', 'Cena incluida'])
        special_requests = st.text_area("Solicitudes especiales")
    with col_pref2:
        st.markdown("**Habitación:**")
        room_type = st.selectbox("Tipo de habitación*", ['Predeterminado', 'Individual', 'Doble', 'Twin', 'Triple', 'Suite', 'Familiar'])
        total_guests = st.number_input("Número de huéspedes*", min_value=1, max_value=10, value=2)
        required_car_parking = st.checkbox("Requiere estacionamiento")
    
    st.header("Información de Pago")
    col_pago1, col_pago2 = st.columns(2)
    with col_pago1:
        payment_method = st.selectbox("Método de pago*", ['Tarjeta Crédito', 'Tarjeta Débito', 'Transferencia', 'Efectivo'])
    with col_pago2:
        if payment_method in ['Tarjeta Crédito', 'Tarjeta Débito']:
            card_number = st.text_input("Número de tarjeta*", max_chars=16)
            expiry_date = st.text_input("Fecha expiración (MM/YY)*", max_chars=5)
    
    # Construir vector de entrada (25 features)
    input_data = build_input_data(required_car_parking, lead_time, fecha_llegada,
                                  repeated_guest=0,
                                  special_requests_count=len([r for r in special_requests.split(",") if r.strip()]) if special_requests else 0,
                                  meal_plan=meal_2,
                                  room_type=room_type,
                                  total_guests=total_guests,
                                  total_nights=total_nights)
    
    try:
        input_scaled = scaler.transform(input_data)
        prediccion_base = modelo.predict(input_scaled)[0]
        override_modifier = 0
        if os.path.exists(OVERRIDE_FILE):
            with open(OVERRIDE_FILE, "r") as f:
                try:
                    override_modifier = float(f.read().strip())
                except Exception:
                    override_modifier = 0
        precio_ajustado = prediccion_base * (1 + override_modifier/100)
        st.markdown("### Precio Sugerido")
        st.success(f"Precio por noche sugerido: ${precio_ajustado:.2f} USD")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
    
    if st.button("💳 Confirmar Reserva y Calcular Precio", type="primary"):
        mandatory_fields = {
            "Nombre": nombre,
            "Email": email,
            "Teléfono": telefono,
            "Documento": documento,
            "Fechas": fecha_llegada and fecha_salida,
            "Método de pago": payment_method
        }
        missing_fields = [k for k, v in mandatory_fields.items() if not v]
        if missing_fields:
            st.error(f"❌ Faltan campos obligatorios: {', '.join(missing_fields)}")
        else:
            total_price = precio_ajustado * total_nights
            st.success(f"Total a pagar ({total_nights} noches): ${total