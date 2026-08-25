import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"


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
    "Enter the product, store, pricing, inventory, and environmental "
    "information to predict product demand."
)


# --------------------------------------------------
# API Health Check
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    if health_response.status_code == 200:
        st.success("Prediction API is online")
    else:
        st.error("Prediction API is not responding correctly.")

except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. "
        "Make sure the FastAPI server is running on port 8000."
    )


st.divider()


# --------------------------------------------------
# Input Form
# --------------------------------------------------

with st.form("prediction_form"):

    st.subheader("Prediction Inputs")

    col1, col2, col3 = st.columns(3)

    # ------------------------------
    # Basic Information
    # ------------------------------

    with col1:

        st.markdown("### Product Information")

        date = st.date_input(
            "Date"
        )

        store_id = st.selectbox(
            "Store",
            [
                "S001",
                "S002",
                "S003",
                "S004",
                "S005",
            ],
        )

        product_id = st.selectbox(
            "Product",
            [
                "P0001",
                "P0002",
                "P0003",
                "P0004",
                "P0005",
                "P0006",
                "P0007",
                "P0008",
                "P0009",
                "P0010",
                "P0011",
                "P0012",
                "P0013",
                "P0014",
                "P0015",
                "P0016",
                "P0017",
                "P0018",
                "P0019",
                "P0020",
            ],
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

    # ------------------------------
    # Environmental Information
    # ------------------------------

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

    # ------------------------------
    # Inventory / Sales Information
    # ------------------------------

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

    payload = {
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

    try:

        with st.spinner("Running demand prediction..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        if response.status_code == 200:

            result = response.json()

            predicted_demand = result["predicted_demand"]

            st.success("Prediction completed successfully")

            st.metric(
                label="Predicted Demand",
                value=f"{predicted_demand:.2f} units",
            )

            st.info(
                f"The model predicts approximately "
                f"**{predicted_demand:.0f} units** of demand."
            )

        else:

            try:
                error_detail = response.json().get(
                    "detail",
                    "Unknown API error.",
                )

            except Exception:
                error_detail = response.text

            st.error(
                f"Prediction failed: {error_detail}"
            )

    except requests.RequestException as exc:

        st.error(
            "Could not connect to the FastAPI server."
        )

        st.code(str(exc))