import streamlit as st
import json

st.set_page_config(page_title="Gestión Ferroviaria IA", page_icon ="../data/img/logo.png",  layout="wide")

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
st.title("🚉 Plataforma de Mantenimiento Inteligente")
st.markdown("""
Bienvenido al sistema centralizado de análisis ferroviario. Selecciona una herramienta en el menú de la izquierda:
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Predicción RUL")
    st.write("Análisis de telemetría y Remaining Useful Life de bogies mediante CSV.")
    if st.button("Ir a Predicción"):
        st.switch_page("pages/01_RUL_Predictor.py")

with col2:
    st.subheader("👁️ Visión Artificial")
    st.write("Detección de defectos en ruedas mediante imágenes y YOLOv8.")
    if st.button("Ir a Detector"):
        st.switch_page("pages/02_Wheel_Detector.py")