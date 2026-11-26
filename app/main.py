import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="MP166 Metro Station Crowd API")
model = joblib.load("models/model.joblib")


class PredictionRequest(BaseModel):
    hour: int
    nearby_offices: int
    temp_c: float
    rain_mm: float = Field(..., ge=0)
    feeder_buses: int = Field(..., le=70)
    ticket_price_rs: float
    day_type: str
    line: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictionRequest):
    df = pd.DataFrame([req.model_dump()])
    pred = float(model.predict(df)[0])
    return {"prediction": round(pred, 2)}