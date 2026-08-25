import joblib
import numpy as np
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import MODEL_PATH, X_TEST_PATH, Y_TEST_PATH, PREDICTIONS_PATH

def load_evaluation_data():

   model = keras.models.load_model(MODEL_PATH)

   X_test = joblib.load(X_TEST_PATH)
   y_test = joblib.load(Y_TEST_PATH)

   print(f"Test features: {X_test.shape}")
   print(f"Test targets:  {y_test.shape}")

   return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Generate predictions and calculate regression metrics."""

    predictions = model.predict(X_test, verbose=0)

    
    predictions = np.ravel(predictions)
    y_test = np.ravel(y_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return predictions


def save_predictions(predictions):

    """Save test predictions for later analysis and Power BI."""

    joblib.dump(predictions, PREDICTIONS_PATH)

    print("\nPredictions saved to:")
    print(Path(PREDICTIONS_PATH).resolve())


if __name__ == "__main__":

    model, X_test, y_test = load_evaluation_data()

    predictions = evaluate_model(model, X_test, y_test)

    save_predictions(predictions)

    print("\nPredictions saved successfully")


