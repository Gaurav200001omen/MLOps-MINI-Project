from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "hour": 17,
    "nearby_offices": 40,
    "temp_c": 28.5,
    "rain_mm": 5.0,
    "feeder_buses": 20,
    "ticket_price_rs": 30,
    "day_type": "weekday",
    "line": "Yellow Line",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert "prediction" in r.json()


def test_reject_feeder_buses_above_70():
    bad = {**VALID_PAYLOAD, "feeder_buses": 75}
    assert client.post("/predict", json=bad).status_code == 422


def test_reject_negative_rain():
    bad = {**VALID_PAYLOAD, "rain_mm": -1.5}
    assert client.post("/predict", json=bad).status_code == 422