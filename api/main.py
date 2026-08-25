from pathlib import Path
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predict import predict



app = FastAPI(
    title="Stock Demand Prediction API",
    description="API for predicting product demand using the trained ANN model.",
    version="2.0.0",
)

@app.get("/")
def root():
    return {
        "message": "Stock Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"


class PredictionRequest(BaseModel):

    Date: str

    Store_ID: str = Field(alias="Store ID")
    Product_ID: str = Field(alias="Product ID")

    Category: str
    Region: str
    Weather_Condition: str = Field(alias="Weather Condition")
    Seasonality: str

    Inventory_Level: float = Field(alias="Inventory Level")
    Units_Sold: float = Field(alias="Units Sold")
    Units_Ordered: float = Field(alias="Units Ordered")

    Price: float
    Discount: float
    Promotion: float

    Competitor_Pricing: float = Field(alias="Competitor Pricing")
    Epidemic: float

    class Config:
        populate_by_name = True


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "stock-prediction-api",
        "model": "demand_ann.keras",
    }



@app.post("/predict")
def make_prediction(request: PredictionRequest):

    try:

        if not DATA_PATH.exists():

            raise HTTPException(
                status_code=500,
                detail="Historical sales dataset was not found."
            )

        historical_df = pd.read_csv(DATA_PATH)

        historical_df["Date"] = pd.to_datetime(
            historical_df["Date"]
        )


        request_date = pd.to_datetime(request.Date)

        input_row = pd.DataFrame(
            [
                {
                    "Date": request_date,

                    "Store ID": request.Store_ID,
                    "Product ID": request.Product_ID,

                    "Category": request.Category,
                    "Region": request.Region,

                    "Weather Condition":
                        request.Weather_Condition,

                    "Seasonality":
                        request.Seasonality,

                    "Inventory Level":
                        request.Inventory_Level,

                    "Units Sold":
                        request.Units_Sold,

                    "Units Ordered":
                        request.Units_Ordered,

                    "Price":
                        request.Price,

                    "Discount":
                        request.Discount,

                    "Promotion":
                        request.Promotion,

                    "Competitor Pricing":
                        request.Competitor_Pricing,

                    "Epidemic":
                        request.Epidemic,

                  
                    "Demand": 0,
                }
            ]
        )


        historical_before_date = historical_df[
            historical_df["Date"] < request_date
        ].copy()

        if historical_before_date.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Not enough historical data before "
                    "the requested date."
                )
            )


        historical_before_date = historical_before_date[
            (
                historical_before_date["Store ID"]
                == request.Store_ID
            )
            &
            (
                historical_before_date["Product ID"]
                == request.Product_ID
            )
        ].copy()

        if historical_before_date.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No historical demand found for "
                    "the requested Store ID and Product ID."
                )
            )

 

        combined_df = pd.concat(
            [
                historical_before_date,
                input_row,
            ],
            ignore_index=True,
        )

        combined_df = combined_df.sort_values(
            ["Store ID", "Product ID", "Date"]
        ).reset_index(drop=True)

  
        result = predict(combined_df)

        if result.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Prediction could not be generated. "
                    "The available historical data is "
                    "insufficient for the required lag "
                    "and rolling-demand features."
                ),
            )


        requested_result = result[
            result["Date"] == request_date
        ]

        if requested_result.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The requested date could not be "
                    "processed."
                ),
            )

        prediction = float(
            requested_result[
                "predicted_demand"
            ].iloc[-1]
        )


        return {
            "predicted_demand": round(
                prediction,
                2
            ),
            "store_id": request.Store_ID,
            "product_id": request.Product_ID,
            "date": request.Date,
            "status": "success",
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


