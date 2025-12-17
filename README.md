# Predicción de Vida Útil Restante (RUL) en Bogies de Tren 🚆

Proyecto de **mantenimiento predictivo** orientado a la estimación de la **Vida Útil Restante (Remaining Useful Life, RUL)** de bogies ferroviarios mediante *Machine Learning* y un sistema complementario de **visión artificial**.

El objetivo es anticipar fallos con antelación suficiente para **mejorar la seguridad**, **optimizar la planificación del mantenimiento** y **reducir paradas no planificadas**.

---

## 📌 Alcance del Proyecto

Este repositorio recoge un pipeline completo de *data science* aplicado a un entorno industrial realista:

* Procesamiento y limpieza de grandes volúmenes de datos de sensores.
* Análisis exploratorio orientado a fallos raros.
* Definición avanzada de la variable objetivo RUL.
* Entrenamiento y comparación de modelos predictivos.
* Integración de visión artificial para detección visual de defectos.

El enfoque es **técnico**, priorizando interpretabilidad, robustez y aplicabilidad industrial.

---

## 🧠 Enfoque Metodológico

### 1️⃣ Mantenimiento Predictivo

Se abandona el enfoque reactivo (fallo → reparación) para adoptar una estrategia **proactiva basada en datos**, permitiendo:

* Intervenciones antes del fallo.
* Mayor disponibilidad de flota.
* Reducción de costes operativos.

### 2️⃣ Análisis Exploratorio de Datos (EDA)

* Dataset inicial: ~200.000 registros.
* Desbalance severo: ~1.5 % de fallos.
* Limpieza avanzada por bogie y control de rangos físicos.
* Análisis visual para detectar patrones de degradación.

### 3️⃣ Ingeniería de Características

Transformación de señales brutas en indicadores de desgaste:

* Ratios de vibración normalizados por carga y velocidad.
* Diferenciales térmicos bogie–rueda.
* Variables de estrés acumulado por sobretemperatura.

### 4️⃣ Definición del Target (RUL)

Problema formulado como **regresión**:

* `RUL_steps`: número de registros restantes hasta el fallo.
* Rango típico: 0–272.

Clasificación operativa del riesgo:

| Nivel       | Interpretación        |
| ----------- | --------------------- |
| Muy crítico | Fallo inminente       |
| Crítico     | Intervención < 24 h   |
| Alto riesgo | Inspección programada |
| Bajo riesgo | Operación normal      |

### 5️⃣ Pipeline y Validación

* Split por `train_id` para evitar *data leakage* temporal.
* Balanceo de clases mediante *undersampling*.
* Ponderación de muestras para priorizar eventos críticos.

### 6️⃣ Modelado y Evaluación

Modelos evaluados:

* Regresión lineal (baseline).
* Gradient Boosting / XGBoost.
* Random Forest optimizado.
* Random Forest + PCA.

**Modelo final:** Random Forest con PCA

* MAE ≈ **12.1** en el rango crítico.
* Buen equilibrio entre precisión y estabilidad.

Interpretación del error:

* Muestreo cada 10 min → ±2 h.
* Muestreo cada 60 min → ±12 h.

### 7️⃣ Visión Artificial (Complementario)

Sistema adicional para detección de defectos superficiales:

* Modelo YOLOv8 *fine-tuned*.
* Entrenamiento con imágenes 640×640 px.
* Exportación a ONNX para despliegue industrial.

Este módulo cubre defectos que no siempre se reflejan en sensores físicos.

---

## 🧩 Arquitectura General

```
Sensores → Limpieza → Feature Engineering → RUL Model
                               ↘
                                Visión Artificial (YOLO)
```

---

## ⚠️ Limitaciones

* Dependencia directa de la frecuencia de muestreo.
* Incremento del error para horizontes largos.
* No se incluyen históricos de mantenimiento real.

---

## 🚀 Próximos Pasos

* Redefinir RUL en unidades físicas (km, días).
* Incorporar modelos temporales (LSTM, Transformers).
* Integrar meteorología y topografía de la vía.
* Unificación del output ML + visión en un único sistema de decisión.

---



## 📎 Nota Final

Este proyecto está diseñado como **demostrador técnico** de capacidades en *Data Science industrial*, mantenimiento predictivo y visión artificial, con foco en **robustez, trazabilidad y aplicabilidad real**.

Si lo estás revisando desde un punto de vista profesional o industrial, el enfoque y las decisiones metodológicas están pensadas para facilitar un despliegue futuro en entorno productivo.
