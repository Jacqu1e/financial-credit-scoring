import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def build_model(preprocessor, model_algorithm):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model_algorithm)
    ])

def summarize_classification(y_true, y_pred, model_name="Modelo"):
    return {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

if __name__ == "__main__":
    from ft_engineering import prepare_data, build_feature_pipeline
    
    data_path = 'Base_de_datos.csv'
    df_temp = pd.read_csv(data_path, encoding='latin1')
    target_col = [c for c in df_temp.columns if 'target' in c.lower() or 'mora' in c.lower() or 'default' in c.lower()][0]
    
    X_train, X_test, y_train, y_test = prepare_data(data_path, target_col)
    
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = build_feature_pipeline(num_cols, cat_cols)
    
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    best_pipeline = None
    best_f1 = 0
    results = []
    
    for name, model in models.items():
        pipeline = build_model(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics = summarize_classification(y_test, y_pred, model_name=name)
        results.append(metrics)
        
        if metrics['F1-Score'] > best_f1:
            best_f1 = metrics['F1-Score']
            best_pipeline = pipeline

    # Guardar el mejor modelo entrenado
    joblib.dump(best_pipeline, 'model.joblib')
    print("\n✅ Modelo exportado exitosamente como 'model.joblib'")

    results_df = pd.DataFrame(results)
    print("\n=== TABLA COMPARATIVA DE EVALUACIÓN DE MODELOS ===")
    print(results_df.to_string(index=False))