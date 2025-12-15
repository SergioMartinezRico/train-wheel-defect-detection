# src/data_processing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def get_project_paths():
    """Define todas las rutas relativas desde src/"""
    BASE_DIR = Path(__file__).parent.parent  # Raíz del proyecto
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    IMG_DIR = DATA_DIR / "img"
    
    # Crear directorios si no existen
    PROCESSED_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)
    
    return RAW_DIR, PROCESSED_DIR, IMG_DIR

def save_plot(fig, filename, img_dir):
    """Guarda gráfico en ruta especificada sin mostrarlo"""
    filepath = img_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Gráfico guardado: {filepath}")

def load_and_initial_clean(raw_dir):
    """Carga y limpieza inicial del dataset"""
    print("📊 Cargando dataset original...")
    file_path = raw_dir / "train_bogie_dataset.csv"
    df_raw = pd.read_csv(file_path)
    print(f"   Shape original: {df_raw.shape}")
    
    # 1. LIMPIAR "ERROR" en timestamp (igual que en Jupyter)
    print("   Limpiando timestamps 'ERROR'...")
    initial_rows = len(df_raw)
    df_raw = df_raw[df_raw['timestamp'] != 'ERROR'].reset_index(drop=True)
    print(f"   Filas con 'ERROR' eliminadas: {initial_rows - len(df_raw)}")
    
    # 2. Conversión timestamp
    print("   Convirtiendo timestamp...")
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    
    # 3. Eliminar duplicados exactos (igual que en Jupyter)
    initial_rows = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset=['timestamp', 'train_id', 'bogie_id'])
    print(f"   Duplicados eliminados: {initial_rows - len(df_raw)}")
    
    print(f"   Shape limpio: {df_raw.shape}")
    return df_raw


def exploratory_plots(df, img_dir):
    """Genera y guarda gráficos exploratorios"""
    print("📈 Generando gráficos exploratorios...")
    
    # 1. Distribuciones numéricas
    num_cols = ['speed_kmh', 'load_tons', 'vibration_x_rms', 'vibration_y_rms', 'vibration_z_rms']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(num_cols[:6]):
        if col in df.columns:
            df[col].hist(bins=50, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribución {col}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, "01_distribucion_numericas.jpg", img_dir)
    
    # 2. Boxplots vibraciones por fault
    if 'target_fault' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        vib_cols = ['vibration_x_rms', 'vibration_y_rms', 'vibration_z_rms']
        
        for i, col in enumerate(vib_cols):
            if col in df.columns:
                sns.boxplot(data=df, x='target_fault', y=col, ax=axes[i], palette='Set2')
                axes[i].set_title(f'{col} por Fault')
        
        plt.tight_layout()
        save_plot(fig, "02_boxplots_vibracion_fault.jpg", img_dir)
    
    # 3. Heatmap correlación
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(14, 12))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Matriz de Correlación', fontsize=16)
        save_plot(plt.gcf(), "03_heatmap_correlacion.jpg", img_dir)

def remove_outliers_iqr(df):
    """Elimina outliers usando método IQR"""
    print("🧹 Eliminando outliers (método IQR)...")
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    initial_rows = len(df_clean)
    outliers_removed = 0
    
    for col in numeric_cols:
        # Excluir IDs y targets
        if col not in ['train_id', 'bogie_id', 'target_fault']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outliers_removed += (before - len(df_clean))
    
    print(f"   Filas eliminadas por outliers: {outliers_removed}")
    print(f"   Shape después outliers: {df_clean.shape}")
    return df_clean

def feature_engineering(df):
    """Crea features ingenieradas"""
    print("🔧 Creando features ingenieradas...")
    df_eng = df.copy()
    
    # Features temporales
    df_eng['hour'] = df_eng['timestamp'].dt.hour
    df_eng['day_of_week'] = df_eng['timestamp'].dt.dayofweek
    df_eng['is_night'] = ((df_eng['hour'] >= 22) | (df_eng['hour'] <= 6)).astype(int)
    
    # Features de vibración
    vib_cols = ['vibration_x_rms', 'vibration_y_rms', 'vibration_z_rms']
    vib_cols_available = [col for col in vib_cols if col in df_eng.columns]
    if vib_cols_available:
        df_eng['vibration_mean'] = df_eng[vib_cols_available].mean(axis=1)
        df_eng['vibration_std'] = df_eng[vib_cols_available].std(axis=1)
    
    # Temperatura ruedas promedio
    wheel_cols = ['wheel_temp_left_c', 'wheel_temp_right_c']
    wheel_cols_available = [col for col in wheel_cols if col in df_eng.columns]
    if len(wheel_cols_available) == 2:
        df_eng['wheel_temp_avg'] = df_eng[wheel_cols_available].mean(axis=1)
    
    # Rolling averages por train_id y bogie_id
    group_cols = ['train_id', 'bogie_id']
    rolling_cols = ['speed_kmh', 'load_tons']
    if 'vibration_mean' in df_eng.columns:
        rolling_cols.append('vibration_mean')
    
    for col in rolling_cols:
        if col in df_eng.columns:
            df_eng[f'{col}_rolling_mean_10'] = df_eng.groupby(group_cols)[col].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
    
    print(f"   Nuevas features creadas: {len(df_eng.columns) - len(df.columns)}")
    return df_eng

def encode_categoricals(df):
    """Encoding de variables categóricas"""
    print("🏷️ Encoding variables categóricas...")
    cat_cols = ['operation_mode', 'track_condition', 'curve_class']
    cat_cols_available = [col for col in cat_cols if col in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=cat_cols_available, prefix=cat_cols_available)
    print(f"   Categorías encodeadas: {len(cat_cols_available)}")
    return df_encoded

def scale_features(df):
    """Escalado de features numéricas principales"""
    print("📏 Escalando features...")
    scale_cols = ['speed_kmh', 'load_tons', 'vibration_mean', 'bogie_temp_c', 'wheel_temp_avg']
    scale_cols_available = [col for col in scale_cols if col in df.columns]
    
    if scale_cols_available:
        scaler = StandardScaler()
        df[scale_cols_available] = scaler.fit_transform(df[scale_cols_available])
        print(f"   Features escaladas: {len(scale_cols_available)}")
    
    return df

def save_processed_datasets(df_final, processed_dir):
    """Guarda todos los datasets procesados"""
    print("💾 Guardando datasets procesados...")
    
    # Dataset completo limpio
    df_final.to_csv(processed_dir / "train_bogie_dataset_clean.csv", index=False)
    
    # Dataset listo para modelado (sin features temporales auxiliares)
    model_cols = [col for col in df_final.columns 
                  if not col.startswith(('hour_', 'day_of_', 'is_night'))]
    df_model = df_final[model_cols].copy()
    df_model.to_csv(processed_dir / "train_bogie_dataset_model_ready.csv", index=False)
    
    # Dataset RUL si existe
    if 'RUL_steps' in df_final.columns:
        df_rul = df_final[['train_id', 'bogie_id', 'timestamp', 'target_fault', 'RUL_steps']].copy()
        df_rul.to_csv(processed_dir / "train_bogie_rul_data.csv", index=False)
        print("   💾 train_bogie_rul_data.csv")
    
    # Gráfico final balance de clases
    if 'target_fault' in df_final.columns:
        plt.figure(figsize=(10, 6))
        df_final['target_fault'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribución Final - Target Fault', fontsize=14)
        plt.xlabel('Target Fault')
        plt.ylabel('Conteo')
        plt.xticks(rotation=0)
        plt.tight_layout()
        save_plot(plt.gcf(), "04_balance_final_target.jpg", processed_dir.parent / "img")
    
    print("✅ Todos los datasets guardados en data/processed/")

def data_processing_pipeline():
    """Pipeline completo de procesamiento de datos"""
    print("=" * 60)
    print("🚂 PIPELINE COMPLETO DE LIMPIEZA Y PROCESAMIENTO")
    print("=" * 60)
    
    # Obtener rutas
    raw_dir, processed_dir, img_dir = get_project_paths()
    
    # 1. Carga y limpieza inicial
    df = load_and_initial_clean(raw_dir)
    
    # 2. Gráficos exploratorios
    exploratory_plots(df, img_dir)
    
    # 3. Limpieza outliers
    df = remove_outliers_iqr(df)
    
    # 4. Feature Engineering
    df = feature_engineering(df)
    
    # 5. Encoding categóricas
    df = encode_categoricals(df)
    
    # 6. Escalado
    df = scale_features(df)
    
    # 7. Guardar resultados
    save_processed_datasets(df, processed_dir)
    
    print("\n" + "=" * 60)
    print("🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"📊 Dataset final: {df.shape}")
    print(f"💾 data/processed/ - Listo para modelado")
    print(f"🖼️  data/img/ - Gráficos generados")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df_final = data_processing_pipeline()
