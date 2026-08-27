"""
Data preprocessing script for the Girl Shoe Size Predictor pipeline.

Combines two raw data sources:
  1. Girls' height-for-age centile table (data/raw/girls_height_centiles_table_DN.csv)
  2. EU shoe size <-> foot length (cm) lookup table (data/raw/eu_shoe_size_measurements.csv)

Produces a synthetic, training-ready dataset with columns:
  Age, Height, Foot Measurement, EU Shoe Size


Output is written to data/clean/shoe_size_training_data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------

RAW_HEIGHT_CENTILES_PATH = Path("data/raw/girls_height_centiles_table_DN.csv")
RAW_SHOE_SIZE_PATH = Path("data/raw/eu_shoe_size_measurements.csv")
OUTPUT_PATH = Path("data/clean/shoe_size_training_data.csv")

# Number of synthetic height samples generated per age group.
# Kept as a constant matching the notebook's original value (1000).
SAMPLES_PER_AGE = 1000

# Fixed random seed for reproducibility. Without this, DVC would see the
# output file "change" on every run even when the raw inputs are identical,
# since the notebook's original code used an unseeded np.random.normal call.
#RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core functions (ported from the notebook)
# ---------------------------------------------------------------------------

def get_foot_measurements_for_gender(heights_in_cm, is_girl: bool = True):
    """Estimate foot length (cm) from height (cm) using a linear approximation.

    Girls: foot_length = |0.14 * height - 1.4|
    Boys:  foot_length = |0.144 * height - 2.0|
    """
    if is_girl:
        return [abs(np.round((0.14 * h - 1.4), 1)) for h in heights_in_cm]
    return [abs(np.round((0.144 * h - 2.0), 1)) for h in heights_in_cm]


def reconstruct_height_data_from_centiles(height_centile_values, centile_values):
    """Reverse-engineer a normal distribution (mu, sigma) from known centile
    values for a single age group, then sample SAMPLES_PER_AGE heights from it.
    """
    z_scores = np.array([st.norm.ppf(c) for c in centile_values])
    z_score_mean = np.mean(z_scores)
    values_mean = np.mean(height_centile_values)

    sigma = np.sum((z_scores - z_score_mean) * (height_centile_values - values_mean)) / \
        np.sum((z_scores - z_score_mean) ** 2)
    mu = values_mean - sigma * z_score_mean

    heights = np.sort(np.round(np.random.normal(mu, sigma, SAMPLES_PER_AGE), 2))
    return heights


def find_closest_eu_size(shoe_size_df: pd.DataFrame, foot_length_cm: float):
    """Find the EU shoe size whose foot length (cm) is closest to the given value."""
    size_index = (shoe_size_df["CM"] - foot_length_cm).abs().argmin()
    return pd.to_numeric(shoe_size_df.loc[size_index, "EU"])


def get_eu_shoe_sizes(foot_measurements, shoe_size_df: pd.DataFrame):
    return [find_closest_eu_size(shoe_size_df, cm) for cm in foot_measurements]


def build_training_dataframe(height_centiles_df: pd.DataFrame, shoe_size_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full synthetic Age / Height / Foot Measurement / EU Shoe Size dataset."""

    # Percentile columns are every column after the first (age) column,
    # named as percentage strings e.g. "0.4", "2", "9", ... "99.6".
    column_names = height_centiles_df.columns.to_numpy()
    centile_values = [float(col) / 100 for col in column_names[1:]]

    # Height values per age (rows) across all percentile columns.
    height_matrix = height_centiles_df.iloc[:, 1:].values

    # Age column (first column).
    age_values = height_centiles_df.iloc[:, 0].to_numpy()

    # Clean the shoe size lookup table.
    shoe_size_df = shoe_size_df.copy()
    shoe_size_df["CM"] = pd.to_numeric(shoe_size_df["CM"], errors="coerce")
    shoe_size_df.dropna(subset=["CM"], inplace=True)
    shoe_size_df.reset_index(drop=True, inplace=True)

    rows = []
    for age_index, height_row in enumerate(height_matrix):
        current_age = age_values[age_index]

        generated_heights = reconstruct_height_data_from_centiles(height_row, centile_values)
        foot_measurements = get_foot_measurements_for_gender(generated_heights, is_girl=True)
        estimated_shoe_sizes = get_eu_shoe_sizes(foot_measurements, shoe_size_df)

        age_df = pd.DataFrame({
            "Age": np.repeat(current_age, len(generated_heights)),
            "Height": generated_heights,
            "Foot Measurement": foot_measurements,
            "EU Shoe Size": estimated_shoe_sizes,
        })
        rows.append(age_df)

    result_df = pd.concat(rows, ignore_index=True)
    result_df["EU Shoe Size"] = pd.to_numeric(result_df["EU Shoe Size"], errors="coerce")

    return result_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw shoe size data for training.")
    parser.add_argument("--height-input", type=Path, default=RAW_HEIGHT_CENTILES_PATH,
                         help="Path to the raw girls height centiles CSV.")
    parser.add_argument("--shoe-size-input", type=Path, default=RAW_SHOE_SIZE_PATH,
                         help="Path to the raw EU shoe size measurements CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                         help="Path to write the processed training-ready CSV.")
    args = parser.parse_args()

    #np.random.seed(RANDOM_SEED)

    if not args.height_input.exists():
        raise FileNotFoundError(f"Raw height centiles file not found: {args.height_input}")
    if not args.shoe_size_input.exists():
        raise FileNotFoundError(f"Raw shoe size measurements file not found: {args.shoe_size_input}")

    # utf-8-sig strips a possible BOM (\ufeff) from the first header cell,
    # e.g. turning "\ufeffGIRL AGE" into "GIRL AGE".
    height_centiles_df = pd.read_csv(args.height_input, encoding="utf-8-sig")
    shoe_size_df = pd.read_csv(args.shoe_size_input, encoding="utf-8-sig")

    print(f"Loaded height centiles: {height_centiles_df.shape[0]} age rows, "
          f"{height_centiles_df.shape[1] - 1} percentile columns")
    print(f"Loaded shoe size lookup: {shoe_size_df.shape[0]} rows")

    training_df = build_training_dataframe(height_centiles_df, shoe_size_df)

    print(f"Generated training dataset: {training_df.shape[0]} rows")
    print(training_df.head())
    print(training_df.tail())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.output, index=False)
    print(f"Saved processed data to {args.output}")


if __name__ == "__main__":
    main()