import pandas as pd
import joblib
from src.config import DATA_PATH,PROCESSED_DATA_PATH
from src.data_loader import load_data


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df['Date'] = pd.to_datetime(df['Date'])

    df = (df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True))


    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_month"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter


    group = df.groupby(["Store ID", "Product ID"])["Demand"]

    df["demand_lag_1"] = group.shift(1)
    df["demand_lag_7"] = group.shift(7)
    df["demand_lag_14"] = group.shift(14)


    df["demand_rolling_mean_7"] = (df.groupby(["Store ID", "Product ID"])["Demand"].transform(lambda x: x.shift(1).rolling(7).mean()))

    df["demand_rolling_mean_14"] = (df.groupby(["Store ID", "Product ID"])["Demand"].transform(lambda x: x.shift(1).rolling(14).mean()))



    df["demand_change_1"] = (df["demand_lag_1"] - df["demand_lag_7"])

    df = df.dropna().reset_index(drop=True)

    return df


df = load_data(DATA_PATH)
featured_df = create_features(df)

FEATURED_DATA_PATH = "data/processed/featured_data.joblib"

joblib.dump(featured_df, FEATURED_DATA_PATH)
