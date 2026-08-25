import streamlit as st
import pandas as pd

from src.predict import predict_current


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Demand Predictor",
    page_icon="📦",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Stock Demand Prediction")

st.write(
    "Enter product, store, pricing, inventory, and environmental "
    "information to predict product demand."
)

st.divider()


# --------------------------------------------------
# Input Form
# --------------------------------------------------

with st.form("prediction_form"):

    st.subheader("Prediction Inputs")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # Product Information
    # --------------------------------------------------

    with col1:

        st.markdown("### Product Information")

        date = st.date_input("Date")

        store_id = st.selectbox(
            "Store",
            ["S001", "S002", "S003", "S004", "S005"],
        )

        product_id = st.selectbox(
            "Product",
            [f"P{i:04d}" for i in range(1, 21)],
        )

        category = st.selectbox(
            "Category",
            [
                "Clothing",
                "Electronics",
                "Furniture",
                "Groceries",
                "Toys",
            ],
        )

        region = st.selectbox(
            "Region",
            [
                "East",
                "North",
                "South",
                "West",
            ],
        )

    # --------------------------------------------------
    # Environmental Information
    # --------------------------------------------------

    with col2:

        st.markdown("### Conditions")

        weather_condition = st.selectbox(
            "Weather Condition",
            [
                "Cloudy",
                "Rainy",
                "Snowy",
                "Sunny",
            ],
        )

        seasonality = st.selectbox(
            "Seasonality",
            [
                "Autumn",
                "Spring",
                "Summer",
                "Winter",
            ],
        )

        epidemic = st.number_input(
            "Epidemic",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=1.0,
        )

        promotion = st.number_input(
            "Promotion",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )

    # --------------------------------------------------
    # Inventory / Sales Information
    # --------------------------------------------------

    with col3:

        st.markdown("### Inventory & Sales")

        inventory_level = st.number_input(
            "Inventory Level",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

        units_sold = st.number_input(
            "Units Sold",
            min_value=0.0,
            value=50.0,
            step=1.0,
        )

        units_ordered = st.number_input(
            "Units Ordered",
            min_value=0.0,
            value=50.0,
            step=1.0,
        )

        price = st.number_input(
            "Price",
            min_value=0.0,
            value=50.0,
            step=0.01,
        )

        competitor_pricing = st.number_input(
            "Competitor Pricing",
            min_value=0.0,
            value=50.0,
            step=0.01,
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Demand",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [
            {
                "Date": str(date),
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": category,
                "Region": region,
                "Weather Condition": weather_condition,
                "Seasonality": seasonality,
                "Inventory Level": inventory_level,
                "Units Sold": units_sold,
                "Units Ordered": units_ordered,
                "Price": price,
                "Discount": discount,
                "Promotion": promotion,
                "Competitor Pricing": competitor_pricing,
                "Epidemic": epidemic,
            }
        ]
    )

    try:

        with st.spinner("Running demand prediction..."):

            result = predict_current(input_data)

        if result.empty:

            st.error(
                "Prediction could not be generated. "
                "The model requires historical demand data to calculate "
                "lag and rolling-demand features."
            )

        else:

            predicted_demand = float(
                result["predicted_demand"].iloc[0]
            )

            st.success("Prediction completed successfully.")

            st.metric(
                label="Predicted Demand",
                value=f"{predicted_demand:.2f} units",
            )

            st.info(
                f"The model predicts approximately "
                f"**{predicted_demand:.0f} units** of demand."
            )

    except Exception as exc:

        st.error("Prediction failed.")

        st.exception(exc)