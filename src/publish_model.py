import argparse
import json
import os
from pathlib import Path

from google.cloud import storage

METRICS_JSON_PATH = Path("artifacts/metrics.json")
MODEL_PICKLE_PATH = Path("artifacts/model.pkl")

DEFAULT_BUCKET = "shoe-size-predictor-data-2026"
MODELS_PREFIX = "models"
BEST_SCORE_BLOB_NAME = f"{MODELS_PREFIX}/best_score.json"


def get_version_id() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def load_local_metrics() -> dict:
    if not METRICS_JSON_PATH.exists():
        raise FileNotFoundError(f"{METRICS_JSON_PATH} not found. Run model_training.py first.")
    with open(METRICS_JSON_PATH) as f:
        return json.load(f)


def read_best_score(bucket):
    blob = bucket.blob(BEST_SCORE_BLOB_NAME)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def write_best_score(bucket, record: dict) -> None:
    blob = bucket.blob(BEST_SCORE_BLOB_NAME)
    blob.upload_from_string(json.dumps(record, indent=2), content_type="application/json")


def upload_model_pickle(bucket, version_id: str) -> str:
    blob_name = f"{MODELS_PREFIX}/model_{version_id}.pkl"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(MODEL_PICKLE_PATH))
    gcs_path = f"gs://{bucket.name}/{blob_name}"
    print(f"Archived model to {gcs_path}")
    return gcs_path


def write_github_output(is_best: bool, version_id: str, mae: float) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    lines = [
        f"is_best={'true' if is_best else 'false'}",
        f"version_id={version_id}",
        f"mae={mae}",
    ]
    if output_path:
        with open(output_path, "a") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("GITHUB_OUTPUT not set, would have written:")
        print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Archive the trained model and check if it's the new best.")
    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_BUCKET,
        help="GCS bucket to archive models to.",
    )
    args = parser.parse_args()

    metrics = load_local_metrics()
    current_mae = metrics["mean_absolute_error"]
    version_id = get_version_id()

    print(f"Current run: version={version_id}, mean_absolute_error={current_mae}")

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    upload_model_pickle(bucket, version_id)

    best_record = read_best_score(bucket)

    if best_record is None:
        is_best = True
        print("No previous best found — this is the first model, automatically the new best.")
    else:
        is_best = current_mae < best_record["mean_absolute_error"]
        print(f"Previous best: {best_record['mean_absolute_error']} (version {best_record['version_id']})")
        print(f"This run {'IS' if is_best else 'is NOT'} an improvement.")

    if is_best:
        write_best_score(
            bucket,
            {
                "version_id": version_id,
                "mean_absolute_error": current_mae,
                "model_gcs_path": f"gs://{args.bucket}/{MODELS_PREFIX}/model_{version_id}.pkl",
            },
        )
        print("Updated best_score.json in GCS.")

    write_github_output(is_best, version_id, current_mae)


if __name__ == "__main__":
    main()
