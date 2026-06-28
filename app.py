import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    package = joblib.load(
        "customer_churn_production.pkl"
    )

    return package

package = load_model()

model = package["model"]
threshold = package["threshold"]
features = package["features"]

# =====================================================
# TITLE
# =====================================================

st.title("📊 Customer Churn Prediction System")

st.markdown(
"""
Predict whether a customer is likely to churn
using an ensemble ML model.
"""
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Model Information")

st.sidebar.metric(
    "ROC-AUC",
    round(package["roc_auc"],4)
)

st.sidebar.metric(
    "F1 Score",
    round(package["f1"],4)
)

st.sidebar.metric(
    "Threshold",
    round(threshold,2)
)

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction"
    ]
)

# =====================================================
# SINGLE PREDICTION
# =====================================================

with tab1:

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider(
            "Tenure",
            0,
            72,
            24
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            0.0,
            200.0,
            70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            1500.0
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0,1]
        )

        partner = st.selectbox(
            "Partner",
            [0,1]
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            [0,1]
        )

        service_count = st.slider(
            "Service Count",
            0,
            8,
            4
        )

        high_value = st.selectbox(
            "High Value Customer",
            [0,1]
        )

        high_risk_contract = st.selectbox(
            "Month-To-Month Contract",
            [0,1]
        )

        senior_alone = st.selectbox(
            "Senior Alone",
            [0,1]
        )

    if st.button("Predict Churn"):

        data = pd.DataFrame(
            columns=features
        )

        data.loc[0] = 0

        if "tenure" in features:
            data["tenure"] = tenure

        if "MonthlyCharges" in features:
            data["MonthlyCharges"] = monthly_charges

        if "TotalCharges" in features:
            data["TotalCharges"] = total_charges

        if "SeniorCitizen" in features:
            data["SeniorCitizen"] = senior

        if "Partner" in features:
            data["Partner"] = partner

        if "Dependents" in features:
            data["Dependents"] = dependents

        if "Service_Count" in features:
            data["Service_Count"] = service_count

        if "High_Value" in features:
            data["High_Value"] = high_value

        if "HighRiskContract" in features:
            data["HighRiskContract"] = high_risk_contract

        if "Senior_Alone" in features:
            data["Senior_Alone"] = senior_alone

        # Engineered Features

        if "Spend_Velocity" in features:
            data["Spend_Velocity"] = (
                total_charges /
                (tenure + 1)
            )

        if "Monthly_to_Total_Ratio" in features:
            data["Monthly_to_Total_Ratio"] = (
                monthly_charges /
                (total_charges + 1)
            )

        if "Charge_Trend" in features:
            data["Charge_Trend"] = (
                monthly_charges /
                (
                    total_charges /
                    (tenure + 1)
                    + 1
                )
            )

        if "Service_Density" in features:
            data["Service_Density"] = (
                service_count /
                (tenure + 1)
            )

        if "Charges_Per_Service" in features:
            data["Charges_Per_Service"] = (
                monthly_charges /
                (service_count + 1)
            )

        if "Tenure_x_Monthly" in features:
            data["Tenure_x_Monthly"] = (
                tenure *
                monthly_charges
            )

        probability = model.predict_proba(
            data
        )[0][1]

        prediction = (
            probability >= threshold
        )

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

        if probability < 0.30:

            st.success(
                "Low Risk Customer"
            )

        elif probability < 0.60:

            st.warning(
                "Medium Risk Customer"
            )

        else:

            st.error(
                "High Risk Customer"
            )

        st.write(
            f"Prediction: {'CHURN' if prediction else 'NO CHURN'}"
        )

# =====================================================
# BATCH PREDICTION
# =====================================================

with tab2:

    st.subheader(
        "Upload CSV File"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            "Preview",
            batch_df.head()
        )

        try:

            probs = model.predict_proba(
                batch_df
            )[:,1]

            preds = (
                probs >= threshold
            ).astype(int)

            batch_df[
                "Churn_Probability"
            ] = probs

            batch_df[
                "Prediction"
            ] = preds

            st.success(
                "Prediction Complete"
            )

            st.dataframe(
                batch_df.head()
            )

            csv = batch_df.to_csv(
                index=False
            )

            st.download_button(
                "Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )

        except Exception as e:

            st.error(
                str(e)
            )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
Built using:

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Extra Trees
- Stacking Ensemble
"""
)