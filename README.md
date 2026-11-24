# MLOps Mini Project: MP166

## Group Information

- Group ID: G163
- Problem ID: MP166
- Problem: Metro Station Crowd Prediction
- Faculty: Ravi Pratap
- Batch: 79

## Team Members

| Name            | Enrollment Number |
| --------------- | ----------------- |
| Priyanshu Yadav | S24CSEU1850       |
| Hardik Rakwal   | S24CSEU0753       |
| Gaurav Raj      | S24CSEU0779       |
| Harsh Chouhan   | S24CSEU0744       |

## Problem Statement

Build an end-to-end MLOps pipeline to predict hourly metro station entries using operational and environmental features.

The target variable is `entries_per_hour`.

This project covers data auditing, data cleaning, experiment tracking, model training, API development, containerization, CI, Kubernetes deployment, and data drift monitoring.

## Data Pipeline

- Raw training and later datasets are tracked using DVC.
- Holdout features are tracked directly using Git.
- Data quality issues are investigated and documented.
- Cleaned training data is saved separately from the original data.
- PSI is calculated for six numerical features using 10 bins.

## Data Quality Findings

The investigation covers:

1. Leaking column
2. Unit mix-up
3. Sentinel values
4. Duplicate rows
5. Stuck sensor
6. Target typos
7. Spelling variants

Detailed findings are available in `reports/findings.yaml`.

## Project Structure

```text
MLOps-MINI-Project/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── src/
├── tests/
├── models/
├── params.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

Pull DVC-managed data:

```cmd
dvc pull
```

## Data Drift Monitoring

Population Stability Index (PSI) is used to compare the cleaned training dataset with the later dataset.

The monitoring process uses 10 bins for each numerical feature.
