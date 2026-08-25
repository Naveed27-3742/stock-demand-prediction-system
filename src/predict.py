import joblib
import numpy as np
import pandas as pd

from tensorflow import keras

from src.config import MODEL_PATH, PREPROCESSOR_PATH, DATA_PATH, PREDICTIONS_DIR


CATEGORICAL_FEATURES = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]


NUMERICAL_FEATURES = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",

    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",

    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_rolling_mean_7",
    "demand_rolling_mean_14",
    "demand_change_1",
]


FEATURE_COLUMNS = (CATEGORICAL_FEATURES + NUMERICAL_FEATURES)


def load_artifacts():

    model = keras.models.load_model(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = (df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True))


    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_month"] = df["Date"].dt.day
    df["week_of_year"] = (df["Date"].dt.isocalendar().week.astype(int))
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter

    demand_group = (df.groupby(["Store ID", "Product ID"])["Demand"])

    df["demand_lag_1"] = demand_group.shift(1)
    df["demand_lag_7"] = demand_group.shift(7)
    df["demand_lag_14"] = demand_group.shift(14)

  
    df["demand_rolling_mean_7"] = (demand_group.transform(lambda x: x.shift(1).rolling(7).mean()))

    df["demand_rolling_mean_14"] = (demand_group.transform(lambda x: x.shift(1).rolling(14).mean()))

  
    df["demand_change_1"] = (df["demand_lag_1"] - df["demand_lag_7"])

    return df

def predict_current(input_df: pd.DataFrame) -> pd.DataFrame:


    historical_df = pd.read_csv(DATA_PATH)

    input_df = input_df.copy()

    input_df["Demand"] = np.nan

    input_df["_prediction_row"] = True

    historical_df["_prediction_row"] = False

    combined_df = pd.concat([historical_df, input_df], ignore_index=True)

    featured_df = create_features(combined_df)

    prediction_row = featured_df[featured_df["_prediction_row"] == True].copy()

    if prediction_row.empty:
        return pd.DataFrame()

    if prediction_row[NUMERICAL_FEATURES].isna().any().any():
        return pd.DataFrame()

    model, preprocessor = load_artifacts()

    X = prediction_row[FEATURE_COLUMNS]

    X_transformed = preprocessor.transform(X)

    predictions = model.predict(X_transformed, verbose=0).flatten()

    predictions = np.maximum(predictions, 0)

    prediction_row["predicted_demand"] = predictions

    return prediction_row

def predict(df: pd.DataFrame) -> pd.DataFrame:

    model, preprocessor = load_artifacts()

    featured_df = create_features(df)

    valid_df = (featured_df.dropna(subset=NUMERICAL_FEATURES).copy())

    if valid_df.empty:
        return pd.DataFrame()

    X = valid_df[FEATURE_COLUMNS]

    X_transformed = preprocessor.transform(X)

    predictions = (model.predict(X_transformed,  verbose=0).flatten())

    predictions = np.maximum(predictions, 0)

    valid_df["predicted_demand"] = predictions

    return valid_df


if __name__ == "__main__":

    print("Loading historical data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Historical data shape: {df.shape}")

    predictions_df = predict(df)

    print("\nPrediction completed")

    print(f"Rows predicted: {len(predictions_df)}")

    if not predictions_df.empty:

        output_columns = [
            "Date",
            "Store ID",
            "Product ID",
            "Category",
            "Demand",
            "predicted_demand",
        ]

        output_df = predictions_df[output_columns].copy()

        output_path = (PREDICTIONS_DIR / "historical_predictions.csv")

        output_df.to_csv(output_path, index=False)

        print(f"\nPredictions saved to:")
        print(output_path)

        print("\nSample predictions:")

        print(output_df.tail(10).to_string(index=False))