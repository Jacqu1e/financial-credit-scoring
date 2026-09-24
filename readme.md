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

## Instalación

### 1. Clonar el repositorio

git clone https://github.com/Jacqu1e/financial-credit-scoring.git
cd financial-credit-scoring

### 2. Crear y activar el entorno virtual

**Entorno virtual estándar (venv):**

python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


### 3. Opción alternativa — Docker

docker build -t financial-credit-scoring .
docker run -p 8501:8501 financial-credit-scoring

---

## Uso

### 1. Ejecutar el Pipeline de Entrenamiento

python src/model_training_evaluation.py

### 2. Ejecutar la aplicación de monitoreo

streamlit run app.py

La aplicación quedará disponible en http://localhost:8501.

### 3. Ejecutar la API REST (FastAPI)

uvicorn src.api:app --reload

La API quedará disponible en http://127.0.0.1:8000 y la documentación interactiva en http://127.0.0.1:8000/docs.

### 4. Configurar rutas (antes de ejecutar)

En los scripts principales, actualiza las rutas al modelo y al dataset según tu entorno local:

model_path = 'model.joblib'
data_path  = 'Base_de_datos.xlsx'

---

## App de Monitoreo

La aplicación está organizada en **cuatro pestañas principales** y un **menú lateral**:

### Predecir
Formulario interactivo para ingresar manualmente las características de un cliente y obtener la **probabilidad de pago** en tiempo real. Las variables de entrada incluyen:

- Salario, edad y plazo del crédito
- Cuota pactada, deuda total e ingreso disponible
- Ratio de endeudamiento y saldo total
- Cantidad de créditos vigentes por sector
 (financiero, cooperativo, real)
- Tipo laboral y tendencia de ingresos

### Cargar Datos
Permite subir un nuevo dataset en formato **CSV** o **Excel (.xlsx)** para evaluar el comportamiento del modelo sobre datos recientes.

### Evaluar Data Drift
Detecta cambios estadísticos entre el dataset de entrenamiento 
original y el nuevo dataset cargado. Ver sección [Data Drift](#data-drift).

### Visualización
Dashboard con las métricas de evaluación del modelo:

- **Curva Precision-Recall** — con Average Precision Score
- **Curva ROC** — con área bajo la curva (AUC)
- **Matriz de Confusión** — con clasificación por umbral de 0.5

---

## Data Drift

El tab **"Evaluar Data Drift"** compara estadísticamente 
el conjunto de entrenamiento original con el nuevo dataset 
cargado, usando dos pruebas según el tipo de variable:

### Variables Numéricas — Test de Kolmogorov-Smirnov (KS)

Compara la distribución de cada variable numérica entre 
ambos datasets. Un **p-value < 0.05** indica que la distribución 
cambió significativamente.

| Variable | KS Stat | P-Value | Drift |
|---|---|---|---|
| salario_cliente | 0.08 | 0.03 | ⚠️ Sí |
| edad_cliente | 0.04 | 0.42 | ✅ No |
| ... | ... | ... | ... |

### Variables Categóricas — Prueba Chi-Cuadrado (χ²)

Compara la frecuencia de cada categoría entre ambos datasets 
mediante una tabla de contingencia. Un **p-value < 0.05** 
indica drift en la distribución categórica.

| Variable | Chi2 Stat | P-Value | Drift |
|---|---|---|---|
| tipo_laboral | 1.23 | 0.27 | ✅ No |
| tendencia_ingresos | 8.45 | 0.01 | ⚠️ Sí |

> **Nota:** Los valores nulos son eliminados antes de aplicar las pruebas. Solo se evalúan las columnas presentes en ambos datasets.
> **Alerta Crítica MLOps:** Si más del **25%** de las variables presentan *drift*, la aplicación dispara automáticamente un aviso de reentrenamiento inmediato.

---

## Variables del Modelo

Recuerda respetar estas variables al ingresar datos manualmente o al cargar un nuevo dataset, ya que el modelo fue entrenado con estas características específicas:

| Variable | Tipo | Descripción |
|---|---|---|
| `salario_cliente` | Numérica | Salario mensual del cliente |
| `edad_cliente` | Numérica | Edad en años |
| `plazo_meses` | Numérica | Plazo del crédito en meses |
| `cuota_pactada` | Numérica | Cuota mensual acordada |
| `deuda_total` | Numérica | Capital prestado + otros préstamos |
| `ingreso_disponible` | Numérica | Ingresos menos cuota pactada |
| `ratio_endeudamiento` | Numérica | Relación deuda / ingreso |
| `saldo_total` | Numérica | Saldo total del cliente |
| `cant_creditosvigentes` | Numérica | Número de créditos activos |
| `creditos_sectorFinanciero` | Numérica | Créditos en sector financiero |
| `creditos_sectorCooperativo` | Numérica | Créditos en sector cooperativo |
| `creditos_sectorReal` | Numérica | Créditos en sector real |
| `tipo_laboral` | Categórica | Empleado / Independiente |
| `tendencia_ingresos` | Categórica | Creciente / Decreciente / Estable |

**Variable objetivo:** `Pago_atiempo` — `1` si el cliente pagó a tiempo, `0` si entró en mora.

---

## Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">Desarrollado por <a href="https://github.com/Jacqu1e">Jacqu1e</a></p>