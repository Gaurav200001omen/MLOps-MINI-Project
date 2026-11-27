import os
import joblib
import yaml
import pandas as pd


def predict_holdout(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    export_path = params["model"].get("export_path", "models/model.joblib")
    holdout_path = params["data"]["holdout_path"]
    out_path = params["data"].get("predictions_path", "predictions/holdout_predictions.csv")

    model = joblib.load(export_path)
    holdout = pd.read_csv(holdout_path)
    features = params["features"]["numerical"] + params["features"]["categorical"]

    preds = model.predict(holdout[features])
    out = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": preds.round(2),
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    predict_holdout()