"""Веб-сервіс для інференсу класифікатора Wine."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import PredictionResponse, WineFeatures

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
CLASS_NAMES = ["class_0", "class_1", "class_2"]

app = FastAPI(
    title="Wine ML API",
    description="REST API для класифікації зразків вина за 13 фізико-хімічними ознаками",
    version="1.0.0",
)

# Модель тримається у глобальній змінній і завантажується одноразово
# під час старту — інакше кожен /predict витрачав би десятки мілісекунд
# на повторне читання model.joblib з диска.
model = None


@app.on_event("startup")
def loadModel() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "Wine ML API"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    x = np.array([[
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavanoids,
        features.nonflavanoid_phenols,
        features.proanthocyanins,
        features.color_intensity,
        features.hue,
        features.od280_od315_of_diluted_wines,
        features.proline,
    ]])
    classId = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0, classId])
    return PredictionResponse(
        class_id=classId,
        class_name=CLASS_NAMES[classId],
        probability=round(probability, 4),
    )
