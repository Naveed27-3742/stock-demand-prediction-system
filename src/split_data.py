import pandas as pd
from src.config import PROCESSED_DATA_PATH

def create_time_split(df: pd.DataFrame):

    df = df.copy()

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

    min_date = df['Date'].min()
    max_date = df["Date"].max()

    total_days = (max_date - min_date).days

    train_end = min_date + pd.Timedelta(days=int(total_days * 0.70))

    validation_end = min_date + pd.Timedelta(days=int(total_days * 0.85))

    train_df = df[df["Date"] <= train_end].copy()

    validation_df = df[(df["Date"] > train_end) &(df["Date"] <= validation_end)].copy()

    test_df = df[df["Date"] > validation_end].copy()

    return train_df, validation_df, test_df

if __name__ == "__main__":

    df = pd.read_csv(PROCESSED_DATA_PATH)

    train_df, validation_df, test_df = create_time_split(df)