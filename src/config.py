from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"

X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "X_train.joblib"

X_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "X_val.joblib"

Y_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "y_train.joblib"

Y_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "y_val.joblib"

X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "X_test.joblib"

Y_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "y_test.joblib"

MODEL_PATH = PROJECT_ROOT / "models" / "demand_ann.keras"

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "demand_preprocessor.joblib"

TRAINING_HISTORY_PATH = PROJECT_ROOT / "models" / "Training_history.joblib"

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "test_predictions.joblib"

TEST_METADATA_PATH = PROJECT_ROOT / "data/processed/test_metadata.joblib"

ERROR_PATH = PROJECT_ROOT / "data/predictions/error_analysis.csv"

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions" 