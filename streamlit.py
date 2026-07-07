import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Files
model = joblib.load("attrition_model.pkl")
feature_importance = joblib.load("feature_importance.pkl")

# CSS
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

[data-testid="metric-container"] {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}

h1,h2,h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Attrition",
        "Feature Importance",
        "About"
    ]
)

# Dashboard
if page == "Dashboard":

    st.title("👨‍💼 Employee Attrition Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Employees", "1470")
    c2.metric("Accuracy", "79.93%")
    c3.metric("Attrition Rate", "16.1%")
    c4.metric("Model", "Random Forest")

    st.markdown("---")

    st.subheader("📈 Attrition Risk Overview")

    risk = 80

    st.progress(risk)

    st.warning(
        f"Current Employee Attrition Prediction Accuracy: {risk}%"
    )

# Feature Importance
elif page == "Feature Importance":

    st.title("📊 Feature Importance")

    top10 = feature_importance.head(10)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        top10["Feature"],
        top10["Importance"]
    )

    ax.set_title("Top 10 Features")

    st.pyplot(fig)

    st.dataframe(top10)

elif page == "Predict Attrition":

    st.title("🔮 Employee Attrition Predictor")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 60, 30)

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=50000,
            value=10000
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4]
        )

        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4]
        )

        job_involvement = st.selectbox(
            "Job Involvement",
            [1, 2, 3, 4]
        )

    with col2:

        overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )

        work_life_balance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4]
        )

        years_at_company = st.slider(
            "Years At Company",
            0,
            40,
            5
        )

        stock_option = st.selectbox(
            "Stock Option Level",
            [0, 1, 2, 3]
        )

        distance = st.slider(
            "Distance From Home",
            1,
            30,
            5
        )

    st.markdown("---")

    if st.button("🚀 Predict Attrition"):

        overtime_value = 1 if overtime == "Yes" else 0

        employee_data = [[
            age,                       # Age
            2,                         # BusinessTravel
            800,                       # DailyRate
            2,                         # Department
            distance,                  # DistanceFromHome
            3,                         # Education
            1,                         # EducationField
            environment_satisfaction,
            1,                         # Gender
            70,                        # HourlyRate
            job_involvement,
            2,                         # JobLevel
            7,                         # JobRole
            job_satisfaction,
            1,                         # MaritalStatus
            monthly_income,
            15000,                     # MonthlyRate
            2,                         # NumCompaniesWorked
            overtime_value,
            15,                        # PercentSalaryHike
            3,                         # PerformanceRating
            3,                         # RelationshipSatisfaction
            stock_option,
            10,                        # TotalWorkingYears
            2,                         # TrainingTimesLastYear
            work_life_balance,
            years_at_company,
            3,                         # YearsInCurrentRole
            1,                         # YearsSinceLastPromotion
            3                          # YearsWithCurrManager
        ]]

        prediction = model.predict(employee_data)[0]

        try:
            probability = model.predict_proba(employee_data)[0][1]
        except:
            probability = 0.50

        risk_percent = int(probability * 100)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        st.progress(risk_percent)

        if prediction == 1:

            st.error(
                f"🔴 High Attrition Risk : {risk_percent}%"
            )

            st.markdown("### 💡 HR Recommendations")

            st.write("✅ Improve Job Satisfaction")
            st.write("✅ Reduce Overtime")
            st.write("✅ Increase Work-Life Balance")
            st.write("✅ Review Compensation Package")
            st.write("✅ Conduct Employee Engagement Meetings")

        else:

            st.success(
                f"🟢 Low Attrition Risk : {risk_percent}%"
            )

            st.markdown("### 💡 HR Recommendations")

            st.write("✅ Employee likely to stay")
            st.write("✅ Maintain engagement")
            st.write("✅ Continue performance support")
            st.write("✅ Encourage career growth")

        st.markdown("---")

        st.subheader("📋 Employee Summary")

        st.write(f"Age : {age}")
        st.write(f"Monthly Income : ₹{monthly_income}")
        st.write(f"OverTime : {overtime}")
        st.write(f"Years At Company : {years_at_company}")    

# About
else:

    st.title("ℹ️ About Project")

    st.write("""
    Employee Attrition Prediction System

    • Dataset Size: 1470 Employees
    • Model: Random Forest
    • Accuracy: 79.93%
    • Technique: SMOTE + Random Forest

    Developed using:
    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    """)