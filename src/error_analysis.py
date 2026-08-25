import joblib
import numpy as np
import pandas as pd
from src.config import Y_TEST_PATH,TEST_METADATA_PATH,PREDICTIONS_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUTPUT_PATH = "data/predictions/error_analysis.csv"



print("Loading test metadata...")

test_metadata = joblib.load(TEST_METADATA_PATH)
y_test = joblib.load(Y_TEST_PATH)
predictions = joblib.load(PREDICTIONS_PATH)

print(f"Test metadata: {test_metadata.shape}")
print(f"Test targets:  {np.asarray(y_test).shape}")
print(f"Predictions:   {np.asarray(predictions).shape}")


actual = np.asarray(y_test).flatten()
predicted = np.asarray(predictions).flatten()



if len(test_metadata) != len(predicted):
    raise ValueError(f"Metadata rows ({len(test_metadata)}) do not match predictions ({len(predicted)})")

if len(actual) != len(predicted):
    raise ValueError(f"Actual values ({len(actual)}) do not match predictions ({len(predicted)})")

print("\nAlignment Check")
print("---------------")
print("Metadata rows: ", len(test_metadata))
print("Actual values: ", len(actual))
print("Predictions:   ", len(predicted))
print("Status:         ALIGNED")


analysis_df = test_metadata.copy()

analysis_df["actual_demand"] = actual
analysis_df["predicted_demand"] = predicted

analysis_df["error"] = (analysis_df["actual_demand"] - analysis_df["predicted_demand"])

analysis_df["absolute_error"] = (analysis_df["error"].abs())


mae = mean_absolute_error(actual, predicted)

mse = mean_squared_error(actual, predicted)

rmse = np.sqrt(mse)

r2 = r2_score(actual, predicted)

mean_error = np.mean(analysis_df["error"])

maximum_error = np.max(analysis_df["absolute_error"])


print("\nError Analysis")
print("--------------")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Mean Error: {mean_error:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Maximum Error: {maximum_error:.4f}")


worst_predictions = (analysis_df.sort_values("absolute_error", ascending=False).head(10))


print("\nWorst 10 Predictions")
print("--------------------")

columns_to_show = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "actual_demand",
    "predicted_demand",
    "error",
    "absolute_error",
]

print(worst_predictions[columns_to_show].to_string(index=False))



category_error = (analysis_df.groupby("Category")["absolute_error"].mean().sort_values(ascending=False))


print("\nMAE by Category")
print("---------------")
print(category_error)



store_error = (analysis_df.groupby("Store ID")["absolute_error"].mean().sort_values(ascending=False))


print("\nMAE by Store")
print("------------")
print(store_error)


category_store_error = (analysis_df.pivot_table(values="absolute_error", index="Category", columns="Store ID", aggfunc="mean").round(2))

print("\nMAE by Category and Store")
print("-------------------------")
print(category_store_error)


analysis_df.to_csv(OUTPUT_PATH, index=False)


print("\nError analysis saved to:")
print(OUTPUT_PATH)

print("\nError analysis completed successfully.")