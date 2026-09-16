import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

def calculate_psi(reference, current, num_buckets=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones numéricas.
    """
    reference = reference.dropna()
    current = current.dropna()
    
    if len(reference) == 0 or len(current) == 0:
        return 0.0
        
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(reference, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    ref_counts, _ = np.histogram(reference, bins=buckets)
    curr_counts, _ = np.histogram(current, bins=buckets)
    
    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts) / len(reference)
    curr_pct = np.where(curr_counts == 0, 0.0001, curr_counts) / len(current)
    
    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_value)

def calculate_jensen_shannon(reference, current, num_bins=10):
    """
    Calcula la divergencia Jensen-Shannon entre dos distribuciones numéricas.
    """
    reference = reference.dropna()
    current = current.dropna()
    
    if len(reference) == 0 or len(current) == 0:
        return 0.0
        
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    p, _ = np.histogram(reference, bins=bins, density=True)
    q, _ = np.histogram(current, bins=bins, density=True)
    
    p = np.where(p == 0, 1e-6, p)
    q = np.where(q == 0, 1e-6, q)
    
    return float(jensenshannon(p, q))

def detect_numerical_drift(reference, current, feature_name):
    """
    Aplica KS test, PSI y Jensen-Shannon para evaluar Data Drift en variables numéricas.
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()
    
    # 1. Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = stats.ks_2samp(ref_clean, curr_clean)
    
    # 2. Population Stability Index (PSI)
    psi_val = calculate_psi(ref_clean, curr_clean)
    
    # 3. Jensen-Shannon Divergence
    js_div = calculate_jensen_shannon(ref_clean, curr_clean)
    
    # Evaluar alerta de drift (PSI > 0.2 o p-value < 0.05)
    has_drift = (psi_val > 0.2) or (ks_pvalue < 0.05)
    status = "ALERTA - DRIFT CRÍTICO" if psi_val > 0.25 else ("PRECAUCIÓN" if psi_val > 0.1 else "ESTABLE")
    
    return {
        'Variable': feature_name,
        'Tipo': 'Numérica',
        'KS Stat': round(ks_stat, 4),
        'KS p-value': round(ks_pvalue, 4),
        'PSI': round(psi_val, 4),
        'Jensen-Shannon': round(js_div, 4),
        'Estado Drift': status,
        'Drift Detectado': has_drift
    }

def detect_categorical_drift(reference, current, feature_name):
    """
    Aplica Chi-cuadrado para evaluar Data Drift en variables categóricas.
    """
    ref_counts = reference.value_counts()
    curr_counts = current.value_counts()
    
    combined = pd.DataFrame({'Ref': ref_counts, 'Curr': curr_counts}).fillna(0)
    
    chi2, p_val, _, _ = stats.chi2_contingency(combined.T)
    
    has_drift = p_val < 0.05
    status = "ALERTA - DRIFT CRÍTICO" if p_val < 0.01 else ("PRECAUCIÓN" if p_val < 0.05 else "ESTABLE")
    
    return {
        'Variable': feature_name,
        'Tipo': 'Categórica',
        'KS Stat': np.nan,
        'KS p-value': round(p_val, 4),
        'PSI': np.nan,
        'Jensen-Shannon': np.nan,
        'Estado Drift': status,
        'Drift Detectado': has_drift
    }

def run_monitoring_pipeline(df_reference, df_current):
    """
    Ejecuta el analisis de drift en todas las variables.
    """
    results = []
    features = [c for c in df_reference.columns if c in df_current.columns]
    
    for col in features:
        if pd.api.types.is_numeric_dtype(df_reference[col]):
            metrics = detect_numerical_drift(df_reference[col], df_current[col], col)
        else:
            metrics = detect_categorical_drift(df_reference[col], df_current[col], col)
        results.append(metrics)
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Módulo de monitoreo de Data Drift (model_monitoring.py) listo.")