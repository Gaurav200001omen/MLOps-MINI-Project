import hashlib
import joblib
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.data.clean import clean_training_data
from src.mp166.train import build_pipeline


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected.dropna(), quantiles))
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    exp_counts = np.histogram(expected, bins=breaks)[0] / len(expected)
    act_counts = np.histogram(actual, bins=breaks)[0] / len(actual)
    eps = 1e-6
    exp_counts = np.clip(exp_counts, eps, None)
    act_counts = np.clip(act_counts, eps, None)
    return float(np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts)))


def run_monitoring(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    train_df = clean_training_data(pd.read_csv(params["data"]["train_path"]))
    later_df = pd.read_csv(params["data"]["later_path"]).sort_values("date").reset_index(drop=True)

    num_cols = params["features"]["numerical"]
    cat_cols = params["features"]["categorical"]
    target = params["target"]["column"]
    seed = params["model"].get("random_state", 42)
    export_path = params["model"].get("export_path", "models/model.joblib")
    psi_bins = params["monitoring"].get("psi_bins", 10)
    retrain_split = params["monitoring"].get("retrain_split", 0.7)
    promo_thresh = params["monitoring"].get("promotion_threshold", 0.12)

    # 1. PSI per numeric column
    psi_scores = {
        c: round(compute_psi(train_df[c], later_df[c], psi_bins), 4)
        for c in num_cols
    }
    most_drifted = max(psi_scores, key=psi_scores.get)

    # 2. Load v1 and compute MAE per line on later.csv
    v1_model = joblib.load(export_path)
    mae_by_group = {}
    for line_val, grp in later_df.groupby("line"):
        preds = v1_model.predict(grp[num_cols + cat_cols])
        mae_by_group[line_val] = round(float(mean_absolute_error(grp[target], preds)), 2)
    worst_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Retrain v2 on cleaned train + first 70% of later.csv
    split_idx = int(len(later_df) * retrain_split)
    later_train = later_df.iloc[:split_idx]
    later_check = later_df.iloc[split_idx:]

    combined = pd.concat([train_df, later_train], ignore_index=True)
    v2_pipe = build_pipeline(
        GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=seed),
        num_cols,
        cat_cols,
    )
    v2_pipe.fit(combined[num_cols + cat_cols], combined[target])

    mae_v1 = round(
        float(mean_absolute_error(later_check[target], v1_model.predict(later_check[num_cols + cat_cols]))),
        2,
    )
    mae_v2 = round(
        float(mean_absolute_error(later_check[target], v2_pipe.predict(later_check[num_cols + cat_cols]))),
        2,
    )

    improvement = (mae_v1 - mae_v2) / mae_v1 if mae_v1 else 0.0
    decision = "promote" if improvement >= promo_thresh else "keep"

    # Log & register v2 in MLflow
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    exp_name = params.get("mlflow", {}).get("experiment_name", "mp166-metro-crowd")
    reg_name = params.get("mlflow", {}).get("registered_model_name", "mp166-metro-model")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    with open(params["data"]["train_path"], "rb") as f:
        data_sha256 = hashlib.sha256(f.read()).hexdigest()

    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_metric("mae_later_check", mae_v2)
        mlflow.sklearn.log_model(v2_pipe, artifact_path="model", serialization_format="cloudpickle")
        v2_run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{v2_run_id}/model", reg_name)
    if decision == "promote":
        MlflowClient().set_registered_model_alias(reg_name, "champion", mv.version)
        joblib.dump(v2_pipe, export_path)

    print("\n=== UPDATE reports/findings.yaml (monitoring) ===")
    print("psi:", psi_scores)
    print("most_drifted_feature:", most_drifted)
    print("mae_by_group:", mae_by_group)
    print("worst_served_group:", worst_group)
    print("mae_v1_on_check:", mae_v1)
    print("mae_v2_on_check:", mae_v2)
    print("decision:", decision)
    print("champion_version:", mv.version if decision == "promote" else 1)


if __name__ == "__main__":
    run_monitoring()