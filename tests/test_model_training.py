import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# model-training.py has a hyphen in its filename, so it can't be imported
# with a normal import model_training statement (hyphens aren't valid in
# Python identifiers). Load it directly from its file path instead.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "model-training.py"
_spec = importlib.util.spec_from_file_location("model_training", _MODULE_PATH)
model_training = importlib.util.module_from_spec(_spec)
sys.modules["model_training"] = model_training
_spec.loader.exec_module(model_training)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Height": [75, 85, 92, 98, 104, 110, 115, 120, 125, 130],
        "Foot Measurement": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "EU Shoe Size": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    })


class TestLoadTrainingData:
    def test_raises_on_missing_file(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            model_training.load_training_data(missing_path)

    def test_raises_on_missing_required_columns(self, tmp_path, sample_df):
        bad_df = sample_df.drop(columns=["Height"])
        csv_path = tmp_path / "bad_data.csv"
        bad_df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing expected column"):
            model_training.load_training_data(csv_path)

    def test_drops_rows_with_missing_values(self, tmp_path, sample_df):
        df_with_na = sample_df.copy()
        df_with_na.loc[0, "Height"] = None
        csv_path = tmp_path / "data_with_na.csv"
        df_with_na.to_csv(csv_path, index=False)

        result = model_training.load_training_data(csv_path)
        assert len(result) == len(sample_df) - 1

    def test_loads_clean_data_unchanged_in_length(self, tmp_path, sample_df):
        csv_path = tmp_path / "clean_data.csv"
        sample_df.to_csv(csv_path, index=False)

        result = model_training.load_training_data(csv_path)
        assert len(result) == len(sample_df)


class TestTrainModel:
    def test_returns_fitted_model_and_metrics(self, sample_df):
        model, metrics, split = model_training.train_model(sample_df)

        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert "r2_score" in metrics
        assert "mean_absolute_error" in metrics
        assert metrics["mean_absolute_error"] >= 0

    def test_model_predicts_reasonable_values(self, sample_df):
        model, _, _ = model_training.train_model(sample_df)
        prediction = model.predict([[5, 104]])
        assert 15 <= prediction[0] <= 35
