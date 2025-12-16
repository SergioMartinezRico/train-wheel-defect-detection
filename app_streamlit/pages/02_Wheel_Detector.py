import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image

# ===============================
# CONFIGURACIÓN DE RUTAS
# ===============================
# Definimos la base para subir dos niveles y llegar a la carpeta 'models'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8_detector_export.pkl")

# ===============================
# CONFIGURACIÓN PÁGINA
# ===============================
st.set_page_config(page_title="Detector de Ruedas", layout="wide")

@st.cache_resource
def load_yolo_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f) # Basado en la lógica de carga del notebook
        return model
    except Exception as e:
        st.error(f"❌ No se encontró el modelo en {MODEL_PATH}")
        return None

model = load_yolo_model()

# ===============================
# UI PRINCIPAL
# ===============================
st.title("🔍 Detector de Defectos en Ruedas")
st.markdown("Carga una imagen para el análisis automático de fallos.")

uploaded_file = st.file_uploader("Seleccionar imagen...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convertir el archivo subido para OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if model is not None:
        # Predicción (usando la confianza por defecto del notebook)
        results = model.predict(source=image, conf=0.25, save=False)
        
        # Dibujar resultados (Anotar) como en el notebook
        annotated_frame = results[0].plot()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Detección YOLOv8")
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Resumen de resultados
        n_detections = len(results[0].boxes)
        if n_detections > 0:
            st.warning(f"Se detectaron {n_detections} posibles defectos.")
        else:
            st.success("No se detectaron anomalías.")