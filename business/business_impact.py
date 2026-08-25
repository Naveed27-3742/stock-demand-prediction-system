import joblib
import pandas as pd
import numpy as np
from src.config import PREDICTIONS_PATH,TEST_METADATA_PATH



SAFETY_STOCK_RATE = 0.10




def load_data():

    predictions = joblib.load(PREDICTIONS_PATH).flatten()

    metadata = joblib.load(TEST_METADATA_PATH).copy()

    if len(predictions) != len(metadata):
        raise ValueError(
            f"Prediction/metadata mismatch: "
            f"{len(predictions)} predictions vs "
            f"{len(metadata)} metadata rows."
        )

    metadata["predicted_demand"] = predictions

    return metadata



def calculate_forecast_metrics(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["forecast_error"] = (df["predicted_demand"] - df["Demand"])

    df["absolute_forecast_error"] = (df["forecast_error"].abs())

    df["over_forecast_units"] = (df["forecast_error"].clip(lower=0))

    df["under_forecast_units"] = ((-df["forecast_error"]).clip(lower=0))

    df["forecast_accuracy_pct"] = (1 - (df["absolute_forecast_error"] / df["Demand"].replace(0, np.nan))) * 100

    df["forecast_accuracy_pct"] = (df["forecast_accuracy_pct"].clip(lower=0, upper=100))

    return df



def calculate_ordering_scenario(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    df["model_safety_stock"] = (df["predicted_demand"] * SAFETY_STOCK_RATE)

    df["model_target_inventory"] = (df["predicted_demand"] + df["model_safety_stock"])

    df["model_recommended_order"] = (df["model_target_inventory"] - df["Inventory Level"]).clip(lower=0)

    df["model_recommended_order"] = (df["model_recommended_order"].round().astype(int))

    df["order_difference"] = (df["model_recommended_order"] - df["Units Ordered"])

    df["model_order_more_than_actual"] = (df["order_difference"].clip(lower=0))

    df["model_order_less_than_actual"] = ((-df["order_difference"]).clip(lower=0))

    return df


def calculate_operational_metrics(df: pd.DataFrame,) -> pd.DataFrame:
   
    df = df.copy()

    df["model_available_after_order"] = (df["Inventory Level"] + df["model_recommended_order"])

    df["model_scenario_shortage"] = (df["Demand"] - df["model_available_after_order"]).clip(lower=0)

    df["model_scenario_excess"] = (df["model_available_after_order"] - df["Demand"]).clip(lower=0)

    df["model_scenario_fulfilled"] = np.minimum(df["model_available_after_order"], df["Demand"])

    df["model_scenario_fulfillment_rate"] = (df["model_scenario_fulfilled"] / df["Demand"].replace(0, np.nan))

    df["model_scenario_stockout"] = (df["model_scenario_shortage"] > 0).astype(int)

    return df


def print_summary(df: pd.DataFrame) -> None:

    total_demand = df["Demand"].sum()

    total_actual_orders = df["Units Ordered"].sum()
    total_model_orders = df["model_recommended_order"].sum()

    total_under_forecast = df["under_forecast_units"].sum()
    total_over_forecast = df["over_forecast_units"].sum()

    total_absolute_error = (df["absolute_forecast_error"].sum())

    avg_accuracy = (df["forecast_accuracy_pct"].mean())

    model_shortage = (df["model_scenario_shortage"].sum())

    model_excess = (df["model_scenario_excess"].sum())

    stockout_events = (df["model_scenario_stockout"].sum())

    scenario_fulfilled = (df["model_scenario_fulfilled"].sum())

    scenario_fulfillment_rate = (scenario_fulfilled / total_demand)



    print("\nBusiness Analysis")


    print(f"Test rows: {len(df):,}")
    print(f"Actual demand: {total_demand:,.0f}")

    print("\nForecast")
  
    print(f"Total absolute forecast error: {total_absolute_error:,.0f}")

    print(f"Average forecast accuracy: {avg_accuracy:.2f}%")

    print(f"Under-forecast units: {total_under_forecast:,.0f}")

    print(f"Over-forecast units: {total_over_forecast:,.0f}")

    print("\nOrdering Scenario")

    print(f"Historical units ordered: {total_actual_orders:,.0f}")

    print(f"Model recommended units: {total_model_orders:,.0f}")

    print(f"Order difference: {total_model_orders - total_actual_orders:,.0f}")

    print("\nOperational Scenario")
  
    print(f"Scenario shortage units: {model_shortage:,.0f}")

    print(f"Scenario excess units: {model_excess:,.0f}")

    print(f"Scenario stockout events: {stockout_events:,}")

    print(f"Scenario fulfillment rate: {scenario_fulfillment_rate:.2%}")


def run_business_analysis() -> pd.DataFrame:

    df = load_data()

    print(f"Rows loaded: {len(df):,}")

    df = calculate_forecast_metrics(df)

    df = calculate_ordering_scenario(df)

    df = calculate_operational_metrics(df)

    return df

if __name__ == "__main__":

    results = run_business_analysis()

    output_path = ("data/predictions/business_impact.csv")

    results.to_csv(output_path, index=False)

    print_summary(results)

    print("\nBusiness analysis saved to:")

    print(output_path)