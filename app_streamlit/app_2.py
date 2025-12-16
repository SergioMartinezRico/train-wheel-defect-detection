import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from streamlit_lottie import st_lottie
from fpdf import FPDF
from datetime import datetime

# ===============================
# CONFIGURACIÓN PÁGINA
# ===============================
st.set_page_config(
    page_title="🔧 RUL Bogie Predictor",
    page_icon="🔧",
    layout="wide"
)

# ===============================
# CARGA LOTTIE
# ===============================
def load_lottie(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

LOTTIE_OK = load_lottie("lottie/ok.json")
LOTTIE_WARNING = load_lottie("lottie/Warning.json")
LOTTIE_CRITICAL = load_lottie("lottie/critical.json")

# ===============================
# CARGA MODELOS
# ===============================
@st.cache_resource
def load_models():
    models_path = "../models/"
    with open(f"{models_path}rf_pca_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(f"{models_path}scaler_pca.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return rf_model, pipeline["scaler"], pipeline["pca"]

rf_model, scaler, pca = load_models()

# ===============================
# COLUMNAS DEL MODELO
# ===============================
MODEL_COLUMNS = [
    'vibration_x_rms', 'vibration_y_rms', 'vibration_z_rms',
    'bogie_temp_c', 'wheel_temp_left_c', 'wheel_temp_right_c',
    'speed_kmh', 'load_tons', 'external_temp_c',
    'humidity_pct', 'days_since_inspection', 'bogie_health_score',
    'vib_xy_ratio', 'vib_xz_ratio', 'vib_load_ratio',
    'vib_speed_ratio', 'operation_mode_normal',
    'track_condition_good', 'curve_class_straight'
]

REQUIRED_COLUMNS = [
    'train_id', 'bogie_id',
    'vibration_x_rms','vibration_y_rms','vibration_z_rms',
    'bogie_temp_c','wheel_temp_left_c','wheel_temp_right_c',
    'speed_kmh','load_tons','external_temp_c','humidity_pct',
    'days_since_inspection','bogie_health_score',
    'vib_xy_ratio','vib_xz_ratio','vib_load_ratio','vib_speed_ratio',
    'operation_mode','track_condition','curve_class'
]

# ===============================
# PREPARACIÓN INPUT
# ===============================
def prepare_input(row_dict):
    df = pd.DataFrame([row_dict])
    df = pd.get_dummies(
        df,
        columns=["operation_mode", "track_condition", "curve_class"]
    )
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    X_scaled = scaler.transform(df.values)
    X_pca = pca.transform(X_scaled)
    return X_pca

# ===============================
# FUNCIONES UTILES
# ===============================
def get_status_text(rul):
    if rul < 2:
        return "MUY CRÍTICO", "Mantenimiento inmediato requerido", LOTTIE_CRITICAL
    elif rul < 10:
        return "CRÍTICO", "Planificar mantenimiento urgente", LOTTIE_CRITICAL
    elif rul < 30:
        return "ALTO RIESGO", "Planificar mantenimiento", LOTTIE_WARNING
    else:
        return "SIN RIESGO", "Operación normal", LOTTIE_OK

def generate_pdf(df_filtered, rul_pred, status_text, message, train_id, bogie_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Informe RUL - Tren {train_id} Bogie {bogie_id}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Fecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    # sin emojis para evitar error Unicode
    pdf.cell(0, 8, f"RUL Predicho: {rul_pred:.0f} ciclos", ln=True)
    pdf.cell(0, 8, f"Estado: {status_text}", ln=True)
    pdf.cell(0, 8, f"Recomendación: {message}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 8, "Datos del registro:", ln=True)
    pdf.ln(5)
    for col, val in df_filtered.iloc[0].items():
        pdf.cell(0, 6, f"{col}: {val}", ln=True)
    return pdf.output(dest='S').encode('latin1')

# ===============================
# UI PRINCIPAL
# ===============================
st.title("🔧 RUL Bogie Predictor")
st.markdown("Predicción del Remaining Useful Life (RUL) por bogie")
st.markdown("---")

# ===============================
# SIDEBAR - CSV
# ===============================
st.sidebar.header("📤 Cargar CSV")
uploaded_file = st.sidebar.file_uploader(
    "Sube un CSV (máx 20 registros)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if len(df) > 20:
        st.error("❌ El archivo contiene más de 20 registros. Máximo permitido: 20.")
        st.stop()

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        st.error(f"❌ Faltan columnas obligatorias: {missing_cols}")
        st.stop()

    st.session_state["data"] = df

# ===============================
# SELECTORES
# ===============================
if "data" in st.session_state:
    df = st.session_state["data"]

    st.sidebar.header("🔢 Selección")
    selected_train = st.sidebar.selectbox(
        "Tren",
        options=sorted(df["train_id"].unique())
    )
    filtered_train = df[df["train_id"] == selected_train]
    selected_bogie = st.sidebar.selectbox(
        "Bogie",
        options=sorted(filtered_train["bogie_id"].unique())
    )

    # ===============================
    # BOTÓN CALCULAR
    # ===============================
    if st.sidebar.button("🚀 Calcular RUL"):
        # FILTRAR ÚLTIMO REGISTRO
        df_filtered = filtered_train[filtered_train["bogie_id"] == selected_bogie]
        row = df_filtered.iloc[-1].to_dict()  # último registro

        # PREDICCIÓN
        X_input = prepare_input(row)
        rul_pred = rf_model.predict(X_input)[0]
        status_text, message, lottie_anim = get_status_text(rul_pred)

        # DISPLAY
        col1, col2 = st.columns([1, 2])
        with col1:
            st_lottie(lottie_anim, height=260)
        with col2:
            st.subheader(f"Tren {selected_train} - Bogie {selected_bogie}")
            st.metric("📈 RUL estimado", f"{rul_pred:.0f} ciclos")
            st.markdown(f"### Estado: {status_text}")
            st.info(message)
            with st.expander("🔍 Ver datos del registro completo"):
                st.dataframe(pd.DataFrame([row]))

        # BOTÓN PDF
        pdf_bytes = generate_pdf(df_filtered, rul_pred, status_text, message, selected_train, selected_bogie)
        st.download_button(
            "📄 Descargar informe PDF",
            pdf_bytes,
            file_name=f"informe_rul_{selected_train}_{selected_bogie}.pdf",
            mime="application/pdf"
        )

else:
    st.info("📤 Sube un CSV para comenzar")
