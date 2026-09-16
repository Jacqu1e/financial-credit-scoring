import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def prepare_data(data_path, target_col):
    """
    Carga el dataset, elimina nulos en la columna objetivo,
    separa características y target, y realiza la división train/test sin estratificado.
    """
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Eliminar filas donde el target sea nulo
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Se remueve stratify=y para evitar el error de clases con 1 solo miembro
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_feature_pipeline(num_cols, cat_cols, ordinal_cols=None):
    """
    Crea el ColumnTransformer según el diagrama del Avance #2.
    """
    # Pipeline para variables numéricas
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para variables categóricas nominales
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = [
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
    
    # Pipeline para variables categóricas ordinales (si existen)
    if ordinal_cols:
        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ])
        transformers.append(('ordinal', ordinal_pipeline, ordinal_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

if __name__ == "__main__":
    print("Módulo ft_engineering.py listo para ser importado en el pipeline de entrenamiento.")