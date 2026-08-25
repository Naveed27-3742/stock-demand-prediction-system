import joblib
import pandas as pd
from src.config import PROCESSED_DATA_PATH
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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


TARGET = ["Demand"]


def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            ("numerical",StandardScaler(), NUMERICAL_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)])


def prepare_data(df):

    df = df.copy()

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]

    return X, y


if __name__ == "__main__":

    df = pd.read_csv(PROCESSED_DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (df.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True))

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    total_days = (max_date - min_date).days

    train_end = min_date + pd.Timedelta(days=int(total_days * 0.70),unit="D")

    val_end = min_date + pd.Timedelta(days=int(total_days * 0.85),unit="D")

    train_df = df[df["Date"] <= train_end].copy()

    val_df = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()

    test_df = df[df["Date"] > val_end].copy()

    test_metadata = test_df.copy()

    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)
    X_test, y_test = prepare_data(test_df)

    preprocessor = create_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)

    X_val_transformed = preprocessor.transform(X_val)

    X_test_transformed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, "models/demand_preprocessor.joblib")

    joblib.dump(X_train_transformed, "data/processed/X_train.joblib")

    joblib.dump(X_val_transformed, "data/processed/X_val.joblib")

    joblib.dump(X_test_transformed, "data/processed/X_test.joblib")

    joblib.dump(y_train.to_numpy(), "data/processed/y_train.joblib")

    joblib.dump(y_val.to_numpy(), "data/processed/y_val.joblib")

    joblib.dump(y_test.to_numpy(), "data/processed/y_test.joblib")

    joblib.dump(test_metadata, "data/processed/test_metadata.joblib")

    print("\nData preparation completed.")
    print("--------------------------------")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Transformed features: {X_train_transformed.shape[1]}")