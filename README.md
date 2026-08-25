## Live Demo

[Try the live Stock Demand Prediction System](https://stock-demand-prediction-system.streamlit.app/)



# Stock Demand Prediction System

A production-style machine learning project for predicting product demand across multiple stores and product categories.

The project uses historical sales data, engineered time-series features, categorical information, and an Artificial Neural Network (ANN) to forecast demand.

It also includes:

- Data preprocessing and feature engineering
- Time-based train/validation/test splitting
- ANN model training
- Model evaluation
- Error analysis
- Historical demand prediction
- FastAPI inference API
- Streamlit frontend
- Business impact analysis
- Model and preprocessing artifact serialization

---

## 1. Project Objective

The goal of this project is to build a demand forecasting system that can help a business estimate future product demand.

Accurate demand predictions can help reduce:

- Stockouts
- Excess inventory
- Inventory holding costs
- Poor fulfillment
- Manual inventory planning

The model predicts:

> **Demand = expected number of units required for a given store-product combination and set of business conditions.**

---

## 2. Machine Learning Problem

### Problem Type

**Supervised Learning → Regression**

### Target Variable

```text
Demand
```

The model predicts a continuous numerical value representing product demand.

---

## 3. Dataset

The dataset contains historical sales information across:

- 5 stores
- 20 products
- 5 product categories
- Multiple regions
- Weather conditions
- Seasonal information
- Pricing
- Discounts
- Promotions
- Inventory levels
- Historical demand

The processed dataset contains approximately:

```text
74,600 rows
```

---

## 4. Feature Engineering

Several features were created from the original data.

### Calendar Features

```text
day_of_week
day_of_month
week_of_year
month
quarter
```

These features allow the model to learn temporal patterns.

### Demand Lag Features

```text
demand_lag_1
demand_lag_7
demand_lag_14
```

These represent previous demand values for the same store-product combination.

For example:

```text
demand_lag_1
```

represents demand from the previous available observation.

```text
demand_lag_7
```

represents demand approximately one week earlier.

```text
demand_lag_14
```

represents demand approximately two weeks earlier.

### Rolling Demand Features

```text
demand_rolling_mean_7
demand_rolling_mean_14
```

These calculate historical average demand over previous observations.

Importantly, the current target value is excluded from these calculations to avoid target leakage.

### Demand Change

```text
demand_change_1
```

This captures the difference between recent demand observations.

---

## 5. Model Features

The final preprocessing pipeline produces:

```text
59 transformed features
```

These include:

### Numerical Features

```text
Inventory Level
Units Sold
Units Ordered
Price
Discount
Promotion
Competitor Pricing
Epidemic

day_of_week
day_of_month
week_of_year
month
quarter

demand_lag_1
demand_lag_7
demand_lag_14
demand_rolling_mean_7
demand_rolling_mean_14
demand_change_1
```

### Categorical Features

```text
Store ID
Product ID
Category
Region
Weather Condition
Seasonality
```

Categorical variables are encoded using the preprocessing pipeline.

---

## 6. Data Splitting

The dataset is divided chronologically rather than randomly.

This is important for time-series forecasting because future information must not leak into the training data.

Current split:

```text
Training:   51,200 rows
Validation: 11,000 rows
Test:       11,000 rows
```

The model is trained on historical observations and evaluated on later observations.

---

## 7. Model Architecture

The final model is an Artificial Neural Network implemented using TensorFlow/Keras.

Architecture:

```text
Input
  ↓
Dense(128)
  ↓
Dropout
  ↓
Dense(64)
  ↓
Dropout
  ↓
Dense(32)
  ↓
Dense(1)
```

Total trainable parameters:

```text
18,049
```

The model was trained for up to:

```text
100 epochs
```

with validation monitoring.

Best validation epoch:

```text
95
```

Best validation loss:

```text
48.0203
```

---

## 8. Model Performance

Final test-set performance:

| Metric | Result |
|---|---:|
| MAE | 4.8543 |
| MSE | 44.0042 |
| RMSE | 6.6336 |
| R² | 0.9780 |

### Interpretation

The model's:

```text
MAE = 4.85
```

means that, on average, predictions are approximately 4.85 demand units away from the actual demand.

The:

```text
R² = 0.978
```

means the model explains approximately 97.8% of the variance in the test data.

These metrics are strong, but they should not be interpreted as proof that the model will perform equally well in every real-world situation.

---

## 9. Error Analysis

The project includes detailed error analysis rather than relying only on overall metrics.

### Worst Prediction

The largest observed absolute error was approximately:

```text
69.23 units
```

### MAE by Category

| Category | MAE |
|---|---:|
| Toys | 5.88 |
| Groceries | 5.58 |
| Clothing | 4.32 |
| Electronics | 3.86 |
| Furniture | 3.59 |

### MAE by Store

| Store | MAE |
|---|---:|
| S005 | 5.15 |
| S004 | 4.98 |
| S003 | 4.93 |
| S002 | 4.68 |
| S001 | 4.53 |

This analysis shows that model performance is not uniform across all business segments.

For example, Toys and Groceries have higher errors than Furniture and Electronics.

---

## 10. Important Model Insight

The project also examined feature relationships with demand.

The strongest correlations observed were:

```text
Units Sold              0.836
Units Ordered           0.513
Promotion               0.281
Discount                0.224
Inventory Level         0.128
Competitor Pricing     -0.024
Price                  -0.024
Epidemic               -0.366
```

Correlation alone does not determine feature importance in the neural network, but it provides useful exploratory insight.

---

## 11. Project Architecture

```text
stock_prediction/
│
├── api/
│   └── main.py
│
├── business/
│   └── business impact analysis scripts
│
├── data/
│   ├── raw/
│   │   └── sales_data.csv
│   │
│   ├── processed/
│   │
│   └── predictions/
│
├── models/
│   ├── demand_ann.keras
│   └── demand_preprocessor.joblib
│
├── notebooks/
│   └── exploratory analysis
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 12. Pipeline

The overall machine learning pipeline is:

```text
Raw Sales Data
      ↓
Data Loading
      ↓
Feature Engineering
      ↓
Time-Based Data Split
      ↓
Preprocessing
      ↓
ANN Training
      ↓
Model Evaluation
      ↓
Error Analysis
      ↓
Prediction
      ↓
FastAPI
      ↓
Streamlit
```

---

## 13. Running the Project

Create and activate the virtual environment.

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 14. Feature Engineering

Run:

```bash
python -m src.feature_engineering
```

This generates the engineered dataset required by the downstream pipeline.

---

## 15. Prepare Training Data

Run:

```bash
python -m src.prepare_data
```

This performs the chronological train/validation/test split and preprocessing.

Expected output is approximately:

```text
Train rows: 51,200
Validation rows: 11,000
Test rows: 11,000
Transformed features: 59
```

---

## 16. Train the Model

Run:

```bash
python -m src.train
```

The trained model is saved as:

```text
models/demand_ann.keras
```

The preprocessing pipeline is saved as:

```text
models/demand_preprocessor.joblib
```

---

## 17. Evaluate the Model

Run:

```bash
python -m src.evaluate
```

This evaluates the trained model against the held-out test set.

The current model achieved:

```text
MAE:  4.8543
MSE:  44.0042
RMSE: 6.6336
R²:   0.9780
```

---

## 18. Error Analysis

Run:

```bash
python -m src.error_analysis
```

This generates:

- Overall error metrics
- Worst predictions
- MAE by category
- MAE by store
- MAE by store and category
- Prediction error information

The output is stored in:

```text
data/predictions/error_analysis.csv
```

---

## 19. Historical Predictions

The prediction module can generate predictions over historical records for analysis.

Run:

```bash
python -m src.predict
```

The prediction system uses the saved:

```text
demand_ann.keras
demand_preprocessor.joblib
```

and generates historical predictions after calculating the required lag and rolling features.

---

## 20. FastAPI

FastAPI provides a programmatic API interface for the trained model.

Start the API with:

```bash
uvicorn api.main:app --reload
```

The API runs locally at:

```text
http://127.0.0.1:8000
```

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

### Health Check

Endpoint:

```text
GET /health
```

### Prediction

Endpoint:

```text
POST /predict
```

The API validates incoming request data using Pydantic and sends the request through the same prediction pipeline used by the project.

---

## 21. Streamlit Application

The project also contains a Streamlit frontend.

Start it with:

```bash
streamlit run app.py
```

The application provides a simple user interface for interacting with the trained demand prediction model.

The Streamlit application acts as the user-facing layer while the machine learning pipeline and model remain separated from the interface.

---

## 22. Model Artifacts

The trained model and preprocessing pipeline are stored in:

```text
models/
```

Current artifacts:

```text
demand_ann.keras
demand_preprocessor.joblib
```

These files are required for inference.

---

## 23. Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- TensorFlow
- Keras

### Visualization / Analysis

- Matplotlib
- Seaborn
- Plotly

### API

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- Streamlit

### Model Serialization

- Joblib
- Keras model format

### Development

- VS Code
- Git
- GitHub

---

## 24. Key Engineering Decisions

### Chronological Splitting

Random train/test splitting was avoided because this is a forecasting problem.

The model should not learn from future observations when predicting the past.

### Historical Lag Features

Demand history is highly informative for forecasting, so lag and rolling features were incorporated.

### Separate Preprocessing Artifact

The preprocessing pipeline is saved independently from the model.

This ensures that inference uses the same transformations as training.

### Error Analysis

Overall accuracy alone is insufficient for a business forecasting system.

Errors are therefore examined across:

- Stores
- Categories
- Store-category combinations
- Individual predictions

### API Separation

The FastAPI service is kept separate from the machine learning source code.

This makes the project easier to deploy and maintain.

---

## 25. Business Value

The model can potentially support:

```text
Demand forecasting
Inventory planning
Stockout reduction
Excess inventory reduction
Procurement planning
Store-level planning
Product-level planning
```

The business analysis component also evaluates the potential operational impact of using model-generated demand estimates.

---

## 26. Limitations

This is a portfolio and engineering project based on historical data.

Important limitations include:

- The dataset may not represent real-world retail behavior.
- Historical relationships may change over time.
- Unexpected events can cause large forecasting errors.
- Correlation does not imply causation.
- High R² does not guarantee strong performance in production.
- The model depends on the quality and availability of input features.
- Forecasting performance can vary significantly across stores and products.

A production system would require continuous monitoring and periodic retraining.

---

## 27. Future Improvements

Possible next steps include:

- Model comparison against XGBoost and LightGBM
- Hyperparameter optimization
- Better time-series validation
- Prediction intervals
- Automated retraining
- Model monitoring
- Data drift detection
- Feature drift detection
- Docker deployment
- Cloud deployment
- CI/CD
- Model registry
- Experiment tracking
- Production database integration
- Authentication for the API
- Automated testing


## 29. Final Model Summary

The completed system takes historical retail information and produces demand predictions using a trained Artificial Neural Network.

The current test performance is:


MAE  = 4.8543

RMSE = 6.6336

R²   = 0.9780

The project demonstrates an end-to-end machine learning engineering workflow:


Data
 ↓
Feature Engineering
 ↓
Preprocessing
 ↓
Training
 ↓
Evaluation
 ↓
Error Analysis
 ↓
Prediction
 ↓
API
 ↓
Frontend
 ↓
Deployment
