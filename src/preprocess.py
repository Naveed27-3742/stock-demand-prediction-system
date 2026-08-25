import pandas as pd
from src.config import DATA_PATH
from src.data_loader import load_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    return df


    


   

    
