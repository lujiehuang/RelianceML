from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'artifacts' / 'overreliance_model.joblib'
app = FastAPI(title='RelianceML API', version='0.1.0')


class Interaction(BaseModel):
    latency_sec: float = Field(ge=0, le=600)
    revision_delta: float = Field(ge=0, le=1)
    verification_clicks: int = Field(ge=0, le=100)
    self_reported_confidence: float = Field(ge=0, le=1)
    task_difficulty: int = Field(ge=1, le=5)
    action_accept: Literal[0, 1]
    ai_correct: Literal[0, 1]


class Prediction(BaseModel):
    overreliance_probability: float
    predicted_label: int
    intervention_recommendation: str


_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail='Model artifact missing. Run: python src/generate_data.py && python src/train.py'
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get('/health')
def health():
    return {'status': 'ok', 'model_ready': MODEL_PATH.exists()}


@app.post('/predict', response_model=Prediction)
def predict(x: Interaction):
    model = get_model()
    row = pd.DataFrame([x.model_dump()])
    p = float(model.predict_proba(row)[:, 1][0])
    label = int(p >= 0.5)
    rec = (
        'trigger_reflection' if p >= 0.65
        else 'consider_light_cue' if p >= 0.40
        else 'continue'
    )
    return Prediction(
        overreliance_probability=round(p, 4),
        predicted_label=label,
        intervention_recommendation=rec,
    )
