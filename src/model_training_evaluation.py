import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def build_model(preprocessor, model_algorithm):
    """
    Ensambla el pipeline completo de procesamiento y modelo.
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model_algorithm)
    ])

def summarize_classification(y_true, y_pred, model_name="Modelo"):
    """
    Calcula las métricas principales de evaluación para clasificación.
    """
    metrics = {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    return


if __name__ == "__main__":
    from ft_engineering import prepare_data, build_feature_pipeline
    
    # 1. Cargar y dividir datos (Ajusta 'Target' al nombre exacto de tu variable objetivo)
    # Si tu columna objetivo se llama distinto, cambia 'Target' por ese nombre
    data_path = 'Base_de_datos.csv'
    
    # Identificar columnas desde el CSV
    df_temp = pd.read_csv(data_path, encoding='latin1')
    target_col = [c for c in df_temp.columns if 'target' in c.lower() or 'mora' in c.lower() or 'default' in c.lower()][0]
    
    X_train, X_test, y_train, y_test = prepare_data(data_path, target_col)
    
    # Clasificar tipos de variables automáticamente
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Crear el preprocesador
    preprocessor = build_feature_pipeline(num_cols, cat_cols)
    
    # 3. Modelos a evaluar
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = []
    
    # 4. Entrenar y evaluar cada modelo
    for name, model in models.items():
        pipeline = build_model(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics = summarize_classification(y_test, y_pred, model_name=name)
        results.append(metrics)
        
    # 5. Mostrar tabla comparativa de resultados
    results_df = pd.DataFrame(results)
    print("\n=== TABLA COMPARATIVA DE EVALUACIÓN DE MODELOS ===")
    print(results_df.to_string(index=False))