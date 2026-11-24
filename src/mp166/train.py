import hashlib
import os
import joblib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean import clean_training_data


def get_file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(model, num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


def train_v1(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    data_sha256 = get_file_sha256(params["data"]["train_path"])
    raw_df = pd.read_csv(params["data"]["train_path"])
    df = clean_training_data(raw_df)

    num_cols = params["features"]["numerical"]
    cat_cols = params["features"]["categorical"]
    target = params["target"]["column"]
    seed = params["model"].get("random_state", 42)
    test_size = params["model"].get("test_size", 0.2)
    n_est = params["model"].get("n_estimators", 200)
    m_depth = params["model"].get("max_depth", 12)

    X = df[num_cols + cat_cols]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    exp_name = params.get("mlflow", {}).get("experiment_name", "mp166-metro-crowd")
    reg_name = params.get("mlflow", {}).get("registered_model_name", "mp166-metro-model")
    export_path = params["model"].get("export_path", "models/model.joblib")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    candidates = [
        ("ridge_alpha_1", Ridge(alpha=1.0)),
        ("rf_params", RandomForestRegressor(n_estimators=n_est, max_depth=m_depth, random_state=seed)),
        ("gbr_150", GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=seed)),
    ]

    best_mae = float("inf")
    best_run_id = None
    best_pipe = None

    for name, est in candidates:
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(est, num_cols, cat_cols)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            mae = mean_absolute_error(y_val, preds)

            mlflow.log_param("model_name", name)
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_metric("mae_val", mae)
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")

            print(f"{name} -> mae_val={mae:.2f} | run_id={run.info.run_id}")
            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id
                best_pipe = pipe

    mv = mlflow.register_model(f"runs:/{best_run_id}/model", reg_name)
    client = MlflowClient()
    client.set_registered_model_alias(reg_name, "champion", mv.version)

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    joblib.dump(best_pipe, export_path)

    print("\n=== UPDATE reports/findings.yaml (tracking) ===")
    print(f"experiment_name: \"{exp_name}\"")
    print(f"best_run_id: \"{best_run_id}\"")
    print(f"registered_model_name: \"{reg_name}\"")
    print(f"champion_version: {mv.version}")


if __name__ == "__main__":
    train_v1()