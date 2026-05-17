import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Setup
st.set_page_config(page_title="Employee Attrition App", layout="wide")

# Load trained Model, Scaler, Columns, Dataset
model = pickle.load(open("attrition_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
df = pd.read_csv("final_encoded_df.csv")  # your encoded dataset

st.markdown("<h1 style='text-align:center; color:#F0A500;'>Employee Attrition Analysis Dashboard</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio("📌 Navigation", ["Dashboard", "Predict Attrition"])


# ===================================================
# 🏠 DASHBOARD PAGE
# ===================================================
if menu == "Dashboard":
    st.markdown("<h2 style='color:#F0A500;'>📊 High Risk Employees</h2>", unsafe_allow_html=True)

    df['Attrition_Prob'] = model.predict_proba(scaler.transform(df[model_columns]))[:, 1]
    high_risk = df.sort_values("Attrition_Prob", ascending=False).head(10)

    st.dataframe(high_risk[["Age", "MonthlyIncome", "DistanceFromHome", "Attrition_Prob"]])

    st.markdown("<h2 style='color:#F0A500;'>🏅 Top Job Satisfaction</h2>", unsafe_allow_html=True)
    high_sat = df.sort_values("JobSatisfaction", ascending=False).head(10)
    st.dataframe(high_sat[["Age", "JobSatisfaction", "MonthlyIncome", "Attrition_Prob"]])

    st.markdown("<h2 style='color:#F0A500;'>🔥 Top Important Features</h2>", unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(7)

    st.bar_chart(feat_imp.set_index("Feature"))


# ===================================================
# 🔮 PREDICT ATTRITION PAGE
# ===================================================
if menu == "Predict Attrition":
    st.markdown("<h2 style='color:#F0A500;'>🔮 Predict Employee Attrition</h2>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 25000, 5000)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    distance = st.slider("Distance From Home", 0, 30, 5)
    dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    gender = st.radio("Gender", ["Male", "Female"])

    # Encode input to match training features
    input_data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'DistanceFromHome': distance,
        'JobSatisfaction': job_satisfaction,
        'OverTime': 1 if overtime == "Yes" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        f"Department_{dept}": 1,
    }

    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    for col, val in input_data.items():
        if col in input_df.columns:
            input_df[col] = val

    scaled_data = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100

        if prediction == 1:
            st.error(f"⚠️ High Risk of Attrition — {probability:.2f}%")
        else:
            st.success(f"😊 Low Risk of Attrition — {probability:.2f}%")


