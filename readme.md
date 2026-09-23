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

## 🛠️ Instalación y Configuración

Sigue estos pasos para clonar el repositorio e instalar el proyecto en tu entorno local.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/Jacqu1e/financial-credit-scoring.git](https://github.com/Jacqu1e/financial-credit-scoring.git)
cd financial-credit-scoring
2. Crear y activar el entorno virtualOpción A — Entorno virtual estándar (venv) [Recomendado]:PowerShell# Crear entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\activate

# Activar en Linux / macOS
# source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
Opción B — Entorno en Anaconda:Bash# Crear el entorno en Conda
conda create -n env_riesgo_crediticio python=3.10 -y
conda activate env_riesgo_crediticio

# Instalar dependencias
pip install -r requirements.txt
3. Opción alternativa — DockerPuedes construir y desplegar la aplicación empaquetada mediante Docker:Bash# Construir la imagen de Docker
docker build -t financial-credit-scoring .

# Ejecutar el contenedor levantando el puerto
docker run -p 8501:8501 financial-credit-scoring
🚀 Uso del Sistema1. Ejecución del Pipeline de Entrenamiento y EvaluaciónPara entrenar el modelo de Machine Learning y generar las métricas junto al artefacto model.joblib:Bashpython src/model_training_evaluation.py
2. Ejecutar la API REST (FastAPI)Para desplegar la API localmente con Uvicorn:Bashuvicorn src.api:app --reload
La API quedará disponible en http://127.0.0.1:8000 y la documentación interactiva Swagger en http://127.0.0.1:8000/docs.3. Ejecutar la Aplicación de Monitoreo (Streamlit)Bashstreamlit run app.py
La aplicación quedará disponible en tu navegador en http://localhost:8501.📊 App de Monitoreo & Detección de Data DriftEl dashboard interactivo de Streamlit permite supervisar la estabilidad del modelo y simular perturbaciones en los datos en tiempo real.Pestañas Principales:Predecir: Formulario interactivo para ingresar las características de un cliente y obtener en tiempo real la clasificación del riesgo y probabilidad de pago.Cargar Datos: Permite cargar un nuevo conjunto de datos en formato .csv o .xlsx para evaluar la performance del modelo sobre datos de producción o recientes.Evaluar Data Drift: Compara estadísticamente el dataset de referencia (entrenamiento) contra el dataset actual para detectar cambios en las distribuciones de las características.Visualización: Muestra métricas de desempeño como la Curva ROC-AUC, Curva Precision-Recall y la Matriz de Confusión.⚠️ Detección y Métricas de Data DriftEl módulo de monitoreo evalúa las 22 variables del modelo utilizando pruebas estadísticas según el tipo de dato:Variables Numéricas — Test Kolmogorov-Smirnov (KS):Evalúa si la distribución empírica de una variable ha variado significativamente ($p\text{-value} < 0.05$).Variables Categóricas — Prueba Chi-Cuadrado ($\chi^2$):Compara las proporciones de frecuencia en variables categóricas ($p\text{-value} < 0.05$).Alerta Crítica de MLOps: Si más del 25% de las variables monitoreadas presentan drift (variación estadística significativa), el sistema emite una Alerta Crítica recomendando activar el pipeline de reentrenamiento inmediato.📋 Variables del ModeloEl modelo evalúa 22 características clave para predecir el comportamiento crediticio:VariableTipoDescripciónsalario_clienteNuméricaSalario mensual reportadoedad_clienteNuméricaEdad del solicitante en añosplazo_mesesNuméricaPlazo solicitado para el créditocuota_pactadaNuméricaValor de la cuota mensualdeuda_totalNuméricaSuma global de deudas vigentesingreso_disponibleNuméricaIngresos libres tras deducción de cuotaratio_endeudamientoNuméricaProporción entre deudas e ingresossaldo_totalNuméricaSaldo en cuentas de ahorro/corrientecant_creditosvigentesNuméricaNúmero total de créditos activoscreditos_sectorFinancieroNuméricaCréditos en bancos e instituciones financierascreditos_sectorCooperativoNuméricaCréditos con cooperativascreditos_sectorRealNuméricaCréditos en comercio/sector realtipo_laboralCategóricaCondición de empleo (Empleado / Independiente)tendencia_ingresosCategóricaComportamiento del ingreso (Creciente / Estable / Decreciente)Variable Objetivo: Pago_atiempo (1 = El cliente cumple el pago a tiempo; 0 = Cliente entra en mora).