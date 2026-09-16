import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.ft_engineering import prepare_data, build_feature_pipeline
from src.model_training_evaluation import build_model, summarize_classification
from src.model_monitoring import run_monitoring_pipeline

st.set_page_config(page_title="Sistema de Monitoreo MLOps & Data Drift", layout="wide")

st.title("📊 Panel de Monitoreo MLOps & Detección de Data Drift")
st.markdown("---")

@st.cache_data
def load_and_prep():
    data_path = 'Base_de_datos.csv'
    df_temp = pd.read_csv(data_path, encoding='latin1')
    target_col = [c for c in df_temp.columns if 'target' in c.lower() or 'mora' in c.lower() or 'default' in c.lower()][0]
    
    X_train, X_test, y_train, y_test = prepare_data(data_path, target_col)
    return X_train, X_test, y_train, y_test, target_col

X_train, X_test, y_train, y_test, target_col = load_and_prep()

# Sidebar
st.sidebar.header("⚙️ Configuración del Monitoreo")
noise_level = st.sidebar.slider("Nivel de Perturbación en Datos Actuales (Simulación de Drift)", 0.0, 1.0, 0.25, 0.05)

# Generar Dataset Actual con perturbación simulada
X_current = X_test.copy()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in num_cols:
    X_current[col] = X_current[col] + np.random.normal(loc=noise_level*2, scale=1.0, size=len(X_current))

# Ejecutar Monitoreo
report_df = run_monitoring_pipeline(X_train, X_current)

# 1. Métricas Principales (KPIs)
col1, col2, col3 = st.columns(3)
total_vars = len(report_df)
drift_count = report_df['Drift Detectado'].sum()
drift_pct = (drift_count / total_vars) * 100

col1.metric("Variables Monitoreadas", total_vars)
col2.metric("Variables con Drift", int(drift_count), delta_color="inverse")
col3.metric("Porcentaje de Drift", f"{drift_pct:.1f}%")

st.markdown("---")

# 2. Recomendaciones Automáticas
st.subheader("🚨 Recomendaciones del Sistema")
if drift_pct >= 25.0:
    st.error("⚠️ **ALERTA CRÍTICA DE DRIFT DETECTADA**: Más del 25% de las características presentan variaciones significativas en su distribución. Se recomienda iniciar el proceso de **REENTRENAMIENTO INMEDIATO** del modelo (Retraining Pipeline).")
elif drift_pct > 0:
    st.warning("⚡ **PRECAUCIÓN**: Se observan cambios moderados en algunas variables (PSI / KS Test fuera de rango normal). Mantener monitoreo continuo.")
else:
    st.success("✅ **SISTEMA ESTABLE**: Las distribuciones de los datos actuales coinciden con la población de entrenamiento.")

# 3. Tabla Resumen de Drift
st.subheader("📋 Tabla de Métricas de Data Drift por Variable")

def highlight_status(val):
    if 'ALERTA' in str(val):
        return 'background-color: #ff4b4b; color: white; font-weight: bold;'
    elif 'PRECAUCIÓN' in str(val):
        return 'background-color: #ffa726; color: black;'
    return 'background-color: #66bb6a; color: white;'

st.dataframe(
    report_df.style.applymap(highlight_status, subset=['Estado Drift']),
    use_container_width=True
)

# 4. Visualización Comparativa
st.subheader("📈 Comparativa de Distribuciones (Referencia vs Actual)")
selected_var = st.selectbox("Selecciona una variable para visualizar:", num_cols)

if selected_var:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(X_train[selected_var], label='Población Referencia (Train)', color='blue', fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(X_current[selected_var], label='Población Actual (Inferencia)', color='red', fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"Distribución de: {selected_var}")
    ax.legend()
    st.pyplot(fig)