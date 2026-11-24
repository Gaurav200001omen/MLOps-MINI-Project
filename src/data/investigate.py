
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

train = pd.read_csv(RAW / "train.csv")
later = pd.read_csv(RAW / "later.csv")

numeric = [
    "hour",
    "nearby_offices",
    "temp_c",
    "rain_mm",
    "feeder_buses",
    "ticket_price_rs",
]

print("\n1. NUMERIC MEDIANS BY LINE: TRAIN VS LATER")

for column in numeric:
    print(f"\n{column}")
    comparison = pd.DataFrame({
        "train": train.groupby("line")[column].median(),
        "later": later.groupby("line")[column].median(),
    })
    comparison["later/train ratio"] = (
        comparison["later"] / comparison["train"]
    )
    print(comparison.round(3))


print("\n2. POSSIBLE SENTINEL VALUES")

sentinels = [-1, -999, 999, 9999]

for column in train.select_dtypes(include="number").columns:
    found = train[column][train[column].isin(sentinels)]
    if not found.empty:
        print(column, found.value_counts().to_dict())


print("\n3. MOST FREQUENT NUMERIC VALUES IN TRAIN")

for column in train.select_dtypes(include="number").columns:
    print(f"\n{column}")
    print(train[column].value_counts().head(12))


print("\n4. POSSIBLE TARGET OUTLIERS")

target = train["entries_per_hour"]

print("Target quantiles:")
print(target.quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]))

print("\nLargest 20 target values:")
print(
    train.nlargest(20, "entries_per_hour")[
        ["record_id", "date", "hour", "line", "entries_per_hour"]
    ].to_string(index=False)
)

print("\nSmallest 20 target values:")
print(
    train.nsmallest(20, "entries_per_hour")[
        ["record_id", "date", "hour", "line", "entries_per_hour"]
    ].to_string(index=False)
)


print("\n5. TEMPERATURE REPEATED-VALUE INVESTIGATION")

# Sort by date and hour before checking consecutive readings.
ordered = train.sort_values(["date", "hour"]).copy()

groups = ordered["temp_c"].ne(
    ordered["temp_c"].shift()
).cumsum()

runs = (
    ordered.groupby(groups)
    .agg(
        temp_c=("temp_c", "first"),
        rows=("temp_c", "size"),
        start_date=("date", "first"),
        end_date=("date", "last"),
        start_hour=("hour", "first"),
        end_hour=("hour", "last"),
    )
    .sort_values("rows", ascending=False)
)

print(runs.head(10).to_string(index=False))


print("\n6. CHECK FOR EXTRA OR MISSING COLUMNS")

print("Train columns:", train.columns.tolist())
print("Later columns:", later.columns.tolist())

print("\nInvestigation completed.")


print("\n7. HOLDOUT COLUMN COMPARISON")

holdout = pd.read_csv(RAW / "holdout_features.csv")

print("Train-only columns:",
      sorted(set(train.columns) - set(holdout.columns)))

print("Holdout-only columns:",
      sorted(set(holdout.columns) - set(train.columns)))

print("\n8. SENTINEL VALUE DETAILS")

for column in train.columns:
    if pd.api.types.is_numeric_dtype(train[column]):
        rows = train[train[column] == 999]
        if not rows.empty:
            print(f"\nColumn: {column}")
            print("Count:", len(rows))
            print(rows.to_string(index=False))

print("\n9. TARGET VALUES ABOVE 10,000")

print(
    train.loc[
        train["entries_per_hour"] > 10000,
        ["record_id", "date", "line", "entries_per_hour"]
    ].to_string(index=False)
)

print("\n10. RED LINE RAIN VALUES")

red_train = train[train["line"] == "Red Line"]
red_later = later[later["line"] == "Red Line"]

print("Train rain values:")
print(red_train["rain_mm"].describe())

print("\nLater rain values:")
print(red_later["rain_mm"].describe())

print("\nFirst 20 Red Line rain values:")
print(red_train[["date", "rain_mm"]].head(20).to_string(index=False))


print("\n11. FINAL DATA ISSUE COUNTS")

# Normalize spelling only for counting true categories.
normalized_day_type = (
    train["day_type"]
    .str.strip()
    .str.lower()
    .replace({
        "wee.": "weekend",
        "hol.": "holiday",
    })
)

print("Normalized day_type categories:")
print(normalized_day_type.value_counts())
print("Number of true categories:", normalized_day_type.nunique())

# Count affected temperature readings.
ordered = train.sort_values(["date", "hour"]).copy()

stuck = ordered[
    (ordered["date"] >= "2026-02-04")
    & (ordered["date"] <= "2026-02-25")
    & (ordered["temp_c"] == 11.5)
]

print("\nStuck temperature rows:", len(stuck))
print("Stuck temperature value:", stuck["temp_c"].unique())

# Check duplicate count.
print("\nExtra duplicate rows:", train.duplicated().sum())

# Check sentinel count.
print("Target sentinel count:", (train["entries_per_hour"] == 999).sum())

# Check target values that could be 10x inflated.
print(
    "Targets above 10000:",
    (train["entries_per_hour"] > 10000).sum()
)
