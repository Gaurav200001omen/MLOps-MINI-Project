
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "train_clean.csv"


def clean_training_data(df):
    df = df.copy()

    # 1. Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # 2. Remove the row with an invalid target sentinel.
    df = df[df["entries_per_hour"] != 999].copy()

    # 3. Correct the Red Line rain unit mix-up.
    # Incorrect values = correct values * 0.1.
    # Therefore, divide incorrect values by 0.1.
    red_line = df["line"].str.strip().str.lower() == "red line"
    df.loc[red_line, "rain_mm"] = df.loc[red_line, "rain_mm"] / 0.1

    # 4. Normalize spelling variants in day_type.
    df["day_type"] = df["day_type"].str.strip().str.lower()

    df["day_type"] = df["day_type"].replace({
        "wee.": "weekend",
        "hol.": "holiday",
    })

    # 5. Repair the stuck temperature sensor.
    df["date"] = pd.to_datetime(df["date"])

    stuck_mask = (
        df["date"].between("2026-02-04", "2026-02-25")
        & (df["temp_c"] == 11.5)
    )

    # Calculate one median temperature per line using
    # unaffected observations only.
    valid_temperatures = df.loc[~stuck_mask]

    temperature_medians = (
        valid_temperatures
        .groupby("line")["temp_c"]
        .median()
    )

    for line, median_temperature in temperature_medians.items():
        line_stuck_mask = stuck_mask & (df["line"] == line)
        df.loc[line_stuck_mask, "temp_c"] = median_temperature

    return df


def main():
    print("Loading original training data...")
    df = pd.read_csv(INPUT_FILE)

    print("Original rows:", len(df))

    cleaned_df = clean_training_data(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaning completed!")
    print("Cleaned rows:", len(cleaned_df))
    print("Rows removed:", len(df) - len(cleaned_df))
    print("Remaining duplicates:", cleaned_df.duplicated().sum())
    print("Remaining target sentinels:",
          (cleaned_df["entries_per_hour"] == 999).sum())
    print("Day type categories:", sorted(cleaned_df["day_type"].unique()))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
