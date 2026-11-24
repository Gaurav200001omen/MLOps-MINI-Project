
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = PROJECT_ROOT / "data" / "raw" / "later.csv"
FINDINGS_PATH = PROJECT_ROOT / "reports" / "findings.yaml"

# Numeric features required by the assignment
NUMERIC_FEATURES = [
    "hour",
    "nearby_offices",
    "temp_c",
    "rain_mm",
    "feeder_buses",
    "ticket_price_rs",
]


def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI).

    reference: cleaned training feature values
    current: later dataset feature values
    bins: number of bins used for comparison
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    # Create 10 bin boundaries using the reference dataset.
    breakpoints = np.linspace(
        reference.min(),
        reference.max(),
        bins + 1
    )

    # Include values outside the reference range in the edge bins.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Count values in each bin.
    reference_counts = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(sort=False)

    current_counts = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(sort=False)

    # Convert counts into proportions.
    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    # Avoid division by zero or log(0).
    epsilon = 0.0001
    reference_pct = reference_pct.clip(lower=epsilon)
    current_pct = current_pct.clip(lower=epsilon)

    # PSI formula.
    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def main():
    print("Loading datasets...")

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    psi_results = {}

    print("\nCalculating PSI using 10 bins...\n")

    for feature in NUMERIC_FEATURES:
        psi_value = calculate_psi(
            train[feature],
            later[feature],
            bins=10
        )

        psi_results[feature] = psi_value

        print(f"{feature}: {psi_value:.6f}")

    most_drifted = max(psi_results, key=psi_results.get)

    print("\nMost drifted feature:", most_drifted)

    # Update findings.yaml without overwriting other findings.
    with open(FINDINGS_PATH, "r", encoding="utf-8") as file:
        findings = yaml.safe_load(file)

    findings["monitoring"]["psi"] = psi_results
    findings["monitoring"]["most_drifted_feature"] = most_drifted

    with open(FINDINGS_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            findings,
            file,
            sort_keys=False,
            default_flow_style=False
        )

    print("\nPSI results saved to reports/findings.yaml")


if __name__ == "__main__":
    main()
