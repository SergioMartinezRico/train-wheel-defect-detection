
"""
Entrenamiento del modelo Random Forest + PCA
Estructura MLOps profesional desde src/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def get_project_paths():
    """Rutas relativas desde src/"""
    BASE_DIR = Path(__file__).parent.parent  # Raíz del proyecto
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    return DATA_DIR, MODELS_DIR

def load_training_data(data_dir):
    """Carga datasets de entrenamiento ya preparados"""
    print("📂 Cargando datos de entrenamiento...")
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")["RUL_steps"].values
    
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val = pd.read_csv(data_dir / "y_val.csv")["RUL_steps"].values
    
    train_balanced = pd.read_csv(data_dir / "train_balanced.csv")
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ X_val:   {X_val.shape}")
    print(f"✅ y_train: {y_train.shape}")
    
    return X_train, y_train, X_val, y_val, train_balanced

def create_sample_weights(train_balanced):
    """Crea pesos para balanceo de clases críticas"""
    weights = train_balanced['risk_bin'].map({
        'MUY_CRITICO': 8.0,
        'CRITICO': 5.0, 
        'ALTO_RIESGO': 2.0,
        'BAJO_RIESGO': 1.0
    }).values
    print(f"⚖️  Pesos creados: min={weights.min():.1f}, max={weights.max():.1f}")
    return weights

def apply_pca_pipeline(X_train, X_val):
    """Aplica PCA conservando 95% varianza (fit SOLO en train)"""
    print("\n🔍 PASO 1: PCA EXPLORATORIO")
    
    # Escalado (fit solo en train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # PCA 95% varianza
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    print("✅ REDUCCIÓN:")
    print(f"   Original: {X_train.shape[1]} features")
    print(f"   PCA:      {X_train_pca.shape[1]} componentes")
    print(f"   Varianza: {pca.explained_variance_ratio_.sum():.1%}")
    
    return X_train_pca, X_val_pca, scaler, pca

def train_rf_pca(X_train_pca, y_train, weights, X_val_pca, y_val):
    """Entrena Random Forest optimizado con PCA"""
    print("\n🚀 PASO 2: ENTRENANDO Random Forest PCA...")
    
    rf_pca = RandomForestRegressor(
        n_estimators=550,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features='sqrt',
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    rf_pca.fit(X_train_pca, y_train, sample_weight=weights)
    y_pred = rf_pca.predict(X_val_pca)
    
    return rf_pca, y_pred

def calculate_metrics(y_true, y_pred):
    """Métricas clave: MAE total y MAE RUL<50 (crítico)"""
    mask_critica = y_true < 50
    
    mae_total = mean_absolute_error(y_true, y_pred)
    mae_critica = mean_absolute_error(y_true[mask_critica], y_pred[mask_critica])
    r2_total = r2_score(y_true, y_pred)
    
    return mae_total, mae_critica, r2_total, mask_critica.sum()

def save_model_pipeline(rf_pca, scaler, pca, weights, models_dir, metrics):
    """Guarda modelo completo para producción"""
    print("\n💾 PASO 3: GUARDANDO MODELO DE PRODUCCIÓN")
    
    # 1. Modelo Random Forest
    with open(models_dir / "rf_pca_model.pkl", 'wb') as f:
        pickle.dump(rf_pca, f)
    
    # 2. Pipeline PCA + Scaler 
    with open(models_dir / "scaler_pca_pipeline.pkl", 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'pca': pca
        }, f)
    
    # 3. Pesos para reproducibilidad
    with open(models_dir / "training_weights.pkl", 'wb') as f:
        pickle.dump(weights, f)
    
    # 4. Métricas del entrenamiento (CONVERTIR A TIPOS JSON)
    metrics_dict = {
        'mae_total': float(metrics[0]),
        'mae_rul_critica': float(metrics[1]),
        'r2_total': float(metrics[2]),
        'samples_rul_critica': int(metrics[3]),  # ← Conversión clave
        'pca_components': int(pca.n_components_),  # ← Conversión clave
        'explained_variance': float(pca.explained_variance_ratio_.sum())
    }
    with open(models_dir / "training_metrics.json", 'w') as f:
        import json
        json.dump(metrics_dict, f, indent=2)
    
    print("✅ ARCHIVOS GUARDADOS:")
    print("- rf_pca_model.pkl")
    print("- scaler_pca_pipeline.pkl ⭐ (para predicción)")
    print("- training_weights.pkl")
    print("- training_metrics.json")


def training_pipeline():
    """Pipeline completo de entrenamiento"""
    print("=" * 70)
    print("🤖 ENTRENAMIENTO MODELO RUL PREDICCIÓN (Random Forest + PCA)")
    print("=" * 70)
    
    data_dir, models_dir = get_project_paths()
    
    # Carga datos
    X_train, y_train, X_val, y_val, train_balanced = load_training_data(data_dir)
    
    # Pesos balanceo
    weights = create_sample_weights(train_balanced)
    
    # PCA Pipeline
    X_train_pca, X_val_pca, scaler, pca = apply_pca_pipeline(X_train, X_val)
    
    # Entrenamiento
    rf_pca, y_pred = train_rf_pca(X_train_pca, y_train, weights, X_val_pca, y_val)
    
    # Métricas
    mae_total, mae_crit, r2_total, n_crit = calculate_metrics(y_val, y_pred)
    
    print("\n✅ RESULTADOS Random Forest PCA:")
    print(f"   MAE total:        {mae_total:.2f}")
    print(f"   MAE RUL<50:       {mae_crit:.2f} ⭐")
    print(f"   R² total:         {r2_total:.4f}")
    print(f"   Muestras RUL<50:  {n_crit}")
    
    if mae_crit < 12.23:
        print("🏆 ⭐ NUEVO MEJOR MODELO! ⭐ 🏆")
    else:
        print("⚠️  Modelo competitivo")
    
    # Guardar
    save_model_pipeline(rf_pca, scaler, pca, weights, models_dir, 
                       (mae_total, mae_crit, r2_total, n_crit))
    
    print("\n" + "=" * 70)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("📁 Modelos listos en: models/")
    print("=" * 70)

if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_error, r2_score
    training_pipeline()
