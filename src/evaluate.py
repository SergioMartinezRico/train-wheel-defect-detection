# src/evaluation.py
"""
Evaluación completa del modelo entrenado
Compara métricas, feature importance, PCA loadings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def get_project_paths():
    """Rutas desde src/"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    IMG_DIR = BASE_DIR / "data" / "img"
    IMG_DIR.mkdir(exist_ok=True, parents=True)
    return DATA_DIR, MODELS_DIR, IMG_DIR

def load_model_pipeline(models_dir):
    """Carga modelo completo"""
    print("🔍 Cargando pipeline de modelo...")
    
    with open(models_dir / "rf_pca_model.pkl", 'rb') as f:
        rf_pca = pickle.load(f)
    
    with open(models_dir / "scaler_pca_pipeline.pkl", 'rb') as f:
        pipeline = pickle.load(f)
        scaler = pipeline['scaler']
        pca = pipeline['pca']
    
    try:
        with open(models_dir / "training_metrics.json", 'r') as f:
            training_metrics = json.load(f)
    except:
        training_metrics = {}
    
    print("✅ Pipeline cargado: RF + PCA + Scaler")
    return rf_pca, scaler, pca, training_metrics

def load_test_data(data_dir):
    """Carga datos de test"""
    print("📂 Cargando datos de test...")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")["RUL_steps"].values
    print(f"✅ X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_test, y_test

def evaluate_model(rf_pca, scaler, pca, X_test, y_test, img_dir):
    """Evaluación completa en test"""
    print("\n🎯 EVALUACIÓN EN TEST SET")
    
    # Preprocesamiento test
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Predicciones
    y_pred = rf_pca.predict(X_test_pca)
    
    # Métricas
    mask_critica = y_test < 50
    mae_total = mean_absolute_error(y_test, y_pred)
    mae_crit = mean_absolute_error(y_test[mask_critica], y_pred[mask_critica])
    rmse_total = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_total = r2_score(y_test, y_pred)
    
    print(f"📊 RESULTADOS TEST:")
    print(f"   MAE total:     {mae_total:.2f}")
    print(f"   MAE RUL<50:    {mae_crit:.2f}")
    print(f"   RMSE total:    {rmse_total:.2f}")
    print(f"   R² total:      {r2_total:.4f}")
    print(f"   Muestras <50:  {mask_critica.sum()}")
    
    # Gráficos
    plot_predictions(y_test, y_pred, mask_critica, img_dir)
    plot_feature_importance(rf_pca, img_dir)
    plot_pca_loadings(pca, img_dir)
    
    return {
        'mae_total': mae_total,
        'mae_critica': mae_crit,
        'rmse_total': rmse_total,
        'r2_total': r2_total,
        'samples_critica': int(mask_critica.sum())
    }

def plot_predictions(y_true, y_pred, mask_crit, img_dir):
    """Gráficos de predicciones"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot total
    axes[0,0].scatter(y_true, y_pred, alpha=0.6, s=1)
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0,0].set_xlabel('Real'); axes[0,0].set_ylabel('Predicho')
    axes[0,0].set_title('Predicciones vs Real (Total)')
    
    # Solo RUL<50
    axes[0,1].scatter(y_true[mask_crit], y_pred[mask_crit], alpha=0.7, s=20, c='red')
    axes[0,1].plot([0, 50], [0, 50], 'r--', lw=2)
    axes[0,1].set_xlabel('Real'); axes[0,1].set_ylabel('Predicho')
    axes[0,1].set_title(f'RUL<50 (n={mask_crit.sum()}) ⭐')
    axes[0,1].set_xlim(0, 50); axes[0,1].set_ylim(0, 50)
    
    # Residuales
    residuals = y_true - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.6, s=1)
    axes[1,0].axhline(0, color='r', ls='--')
    axes[1,0].set_xlabel('Predicho'); axes[1,0].set_ylabel('Residuales')
    axes[1,0].set_title('Residuales')
    
    # Histograma errores RUL<50
    if mask_crit.sum() > 0:
        axes[1,1].hist(y_true[mask_crit] - y_pred[mask_crit], bins=30, alpha=0.7, color='orange')
        axes[1,1].axvline(0, color='r', ls='--')
        axes[1,1].set_xlabel('Error'); axes[1,1].set_ylabel('Frecuencia')
        axes[1,1].set_title('Distribución Errores RUL<50')
    
    plt.tight_layout()
    fig.savefig(img_dir / "evaluation_predictions.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráficos predicciones guardados")

def plot_feature_importance(rf_pca, img_dir):
    """Feature importance del Random Forest"""
    importances = rf_pca.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [f'PC{i+1}' for i in indices], rotation=45)
    plt.title('Feature Importance - Componentes PCA')
    plt.tight_layout()
    plt.savefig(img_dir / "feature_importance_pca.jpg", dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_loadings(pca, img_dir):
    """Loadings de PCA"""
    plt.figure(figsize=(12, 8))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i in range(min(10, loadings.shape[1])):
        plt.plot(loadings[:, i], 'o-', label=f'PC{i+1}')
    
    plt.xlabel('Features originales')
    plt.ylabel('Loadings')
    plt.title('PCA Loadings por Componente')
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_dir / "pca_loadings.jpg", dpi=300, bbox_inches='tight')
    plt.close()

def evaluation_pipeline():
    """Pipeline completo de evaluación"""
    print("=" * 70)
    print("📊 EVALUACIÓN COMPLETA MODELO RUL")
    print("=" * 70)
    
    data_dir, models_dir, img_dir = get_project_paths()
    
    # Cargar modelo
    rf_pca, scaler, pca, training_metrics = load_model_pipeline(models_dir)
    
    # Cargar test
    X_test, y_test = load_test_data(data_dir)
    
    # Evaluar
    test_metrics = evaluate_model(rf_pca, scaler, pca, X_test, y_test, img_dir)
    
    # Comparación training vs test
    print("\n🏆 COMPARACIÓN TRAINING vs TEST:")
    print(f"Training MAE RUL<50: {training_metrics.get('mae_rul_critica', 'N/A')}")
    print(f"Test    MAE RUL<50: {test_metrics['mae_critica']:.2f}")
    
    print("\n✅ EVALUACIÓN COMPLETA:")
    print("📁 Gráficos: data/img/")
    print("📊 Métricas guardadas en models/training_metrics.json")
    print("=" * 70)

if __name__ == "__main__":
    evaluation_pipeline()
