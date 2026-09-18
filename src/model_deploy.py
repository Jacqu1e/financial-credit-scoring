import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(
    title="API de Scoring de Crédito Financiero",
    description="Endpoint MLOps para realizar predicciones individuales y por lotes (batch).",
    version="1.3.0"
)

MODEL_PATH = "model.joblib"

# Carga global del modelo
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "API de Predicción de Riesgo Crediticio activa.",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_batch(payload: PredictionInput):
    """
    Recibe una lista de registros en JSON y retorna las predicciones por lote.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo 'model.joblib' no está cargado o no existe.")
    
    try:
        # Convertir JSON entrante a DataFrame
        df_input = pd.DataFrame(payload.data)
        
        # Generar predicciones
        predictions = model.predict(df_input)
        
        # Generar probabilidades si el modelo las soporta
        probabilities = model.predict_proba(df_input).tolist() if hasattr(model, "predict_proba") else None
        
        return {
            "num_records": len(predictions),
            "predictions": predictions.tolist(),
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la solicitud: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)