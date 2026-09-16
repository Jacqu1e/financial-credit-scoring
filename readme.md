# 🚀 Financial Credit Scoring - MLOps Pipeline & Data Drift Monitoring

## 📋 Caso de Negocio
Este proyecto implementa una solución MLOps punta a punta para la evaluación de riesgo crediticio y clasificación scoring financiero. El sistema permite automatizar el procesamiento de datos, entrenar y comparar modelos supervisados, así como monitorear la degradación y desviaciones poblacionales (*Data Drift*) en producción.

---

## 🛠️ Arquitectura del Proyecto y Tech Stack

### Estructura de Módulos
* `src/ft_engineering.py`: Extracción, limpieza e ingeniería de características mediante `ColumnTransformer` y `Pipeline` de Scikit-Learn.
* `src/model_training_evaluation.py`: Entrenamiento, evaluación y generación de tabla comparativa de desempeño (Regresión Logística vs. Random Forest).
* `src/model_monitoring.py`: Cálculo de métricas de Data Drift (KS-Test, PSI, Jensen-Shannon Divergence, Chi-cuadrado).
* `app.py`: Dashboard interactivo en Streamlit para monitoreo visual en tiempo real y alertas automáticas de reentrenamiento.

### Tecnologías Utilizadas
`Python 3.10`, `Pandas`, `NumPy`, `Scikit-Learn`, `SciPy`, `Streamlit`, `Seaborn`, `Matplotlib`, `Git/GitHub`.

---

## 🏷️ Control de Versiones Git & Releases

El proyecto sigue una estrategia de ramificación estructurada (`master`, `certification`, `developer`):

* **`v1.0.1`**: Carga de datos inicial y Análisis Exploratorio de Datos (EDA).
* **`v1.1.0`**: Modularización de Pipelines, Feature Engineering y Evaluación de Modelos.
* **`v1.2.0`**: Motor de Monitoreo de Data Drift, Dashboard en Streamlit y Alertas de Reentrenamiento.

---

## 📊 Evaluación de Modelos (Versión 1.1.0)

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 0.9915 | 0.9863 | 0.9915 | 0.9889 |
| **Random Forest** | 0.9919 | 0.9840 | 0.9919 | 0.9879 |

---

## 🚀 Instrucciones de Ejecución

### 1. Ejecución del Pipeline de Entrenamiento y Evaluación
```powershell
python src/model_training_evaluation.py