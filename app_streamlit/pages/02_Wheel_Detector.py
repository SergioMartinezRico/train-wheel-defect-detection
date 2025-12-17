import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
import datetime 

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

# =========================
# 🎨 FUENTES Y ESTILO GLOBAL
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Fuente general */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Métricas, números, outputs técnicos */
.stMetricValue, code, pre {
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)
# =========================
# FIN FUENTES
# =========================
st.logo("../data/img/logo.png", size="large")

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
           
        n_detections = max(0, len(results[0].boxes) - 1)  # Protegido contra negativos
        

        if n_detections > 0:
            st.warning(f"Se detectaron {n_detections} posibles defectos.")
            
            # Botón para guardar imagen SOLO si hay defectos
           

            if st.button("💾 Guardar imagen con defectos", type="primary"):
                # Convertir a PIL para guardar
                input_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                
                # Ruta de guardado en la carpeta actual
                save_dir = "saved_defects"
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"defecto_{n_detections}_detecciones_{timestamp}.png")
                
                input_image.save(filename)
                st.success(f"✅ Imagen guardada: **{filename}**")
                st.image(input_image, caption="Imagen guardada", width=400)
