"""
Model training script for the Girl Shoe Size Predictor pipeline.

Reads the processed training-ready dataset produced by src/preprocess_data.py
(data/clean/shoe_size_training_data.csv), trains a simple linear regression
model predicting EU Shoe Size from Age and Height, and logs the run
(parameters, metrics, and the trained model artifact) to MLflow.

Usage:
    python src/model_training.py
    python src/model_training.py --tracking-uri http://localhost:5555 --experiment-name shoe-size-predictor
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/clean/shoe_size_training_data.csv")

FEATURE_COLUMNS = ["Age", "Height"]
TARGET_COLUMN = "EU Shoe Size"

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
DEFAULT_EXPERIMENT_NAME = "shoe-size-predictor"

# Local output directory used to hand off the trained model + metrics to the
# publish_model.py step, independent of MLflow's own (currently ephemeral)
# artifact storage.
ARTIFACTS_DIR = Path("artifacts")
MODEL_PICKLE_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_JSON_PATH = ARTIFACTS_DIR / "metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed training data not found at {path}. "
            "Run src/preprocess_data.py first (or dvc pull the processed data)."
        )
    df = pd.read_csv(path)

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Training data is missing expected column(s): {missing_cols}")

    # Drop any rows with missing values in the columns we actually use.
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing values in required columns")

    return df


def train_model(df: pd.DataFrame):
    """Split the data, fit a LinearRegression model, and return everything
    needed for evaluation and logging."""

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
    }

    return model, metrics, (X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train the shoe size prediction model and log to MLflow.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the processed training-ready CSV.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (e.g. http://localhost:5555).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name to log this run under.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this specific MLflow run.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"Loading training data from {args.data_path}")
    df = load_training_data(args.data_path)
    print(f"Loaded {len(df)} rows")

    with mlflow.start_run(run_name=args.run_name):
        model, metrics, _ = train_model(df)

        # Log parameters
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_rows", len(df))

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log the fitted model as an MLflow model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")
        print(f"Metrics: {metrics}")
        print(f"Model coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_))}")
        print(f"Model intercept: {model.intercept_}")

        # Also save a plain pickle + metrics.json locally, so the workflow's
        # publish_model.py step can archive/promote this model without
        # depending on MLflow's (currently ephemeral) artifact store.
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PICKLE_PATH, "wb") as f:
            pickle.dump(model, f)

        metrics_output = dict(metrics)
        metrics_output["mlflow_run_id"] = run_id
        with open(METRICS_JSON_PATH, "w") as f:
            json.dump(metrics_output, f, indent=2)

        print(f"Saved model pickle to {MODEL_PICKLE_PATH}")
        print(f"Saved metrics to {METRICS_JSON_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    main()
