
from pathlib import Path

import pandas as pd


# Project root directory
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def audit_dataset(file_name):
    file_path = RAW_DIR / file_name

    print("\n" + "=" * 60)
    print(f"DATA AUDIT: {file_name}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    # 1. Dataset dimensions
    print("\n1. DATASET SHAPE")
    print(df.shape)

    # 2. Column information
    print("\n2. COLUMN NAMES AND DATA TYPES")
    print(df.dtypes)

    # 3. Missing values
    print("\n3. MISSING VALUES")
    print(df.isnull().sum())

    # 4. Duplicate records
    print("\n4. DUPLICATE ROWS")
    print("Exact duplicate rows:", df.duplicated().sum())

    # 5. Numeric feature summary
    print("\n5. NUMERIC SUMMARY")
    print(df.describe().T)

    # 6. Unique values in categorical columns
    print("\n6. CATEGORICAL VALUES")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

    # 7. Repeated readings
    print("\n7. POSSIBLE STUCK SENSOR CHECK")

    for column in df.select_dtypes(include="number").columns:
        longest_run = (
            df[column]
            .groupby(df[column].ne(df[column].shift()).cumsum())
            .size()
            .max()
        )

        print(f"{column}: longest consecutive run = {longest_run}")

    # 8. Target analysis
    if "entries_per_hour" in df.columns:
        print("\n8. TARGET SUMMARY BY LINE")
        print(
            df.groupby("line")["entries_per_hour"]
            .agg(["count", "mean", "median", "min", "max"])
        )

    return df


if __name__ == "__main__":
    train = audit_dataset("train.csv")
    later = audit_dataset("later.csv")

    print("\nAUDIT COMPLETED")
