import streamlit as st
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)
import pandas as pd
import joblib

# Load trained model
model = joblib.load("employee_attrition_model.pkl")

st.title("Employee Attrition Prediction System")

st.write("This machine learning system predicts whether an employee "
         "is likely to leave the company based on employee-related factors.")

# Employee details
age = st.number_input("Age", 18, 70, 30)

department = st.selectbox(
    "Department",
    ["Sales", "Engineering", "HR", "Marketing", "Finance"]
)

monthly_income = st.number_input(
    "Monthly Income",
    10000,
    200000,
    50000
)

years_at_company = st.number_input(
    "Years at Company",
    0,
    40,
    5
)

job_satisfaction = st.slider(
    "Job Satisfaction",
    1,
    5,
    3
)

work_life_balance = st.slider(
    "Work Life Balance",
    1,
    5,
    3
)

performance_rating = st.slider(
    "Performance Rating",
    1,
    5,
    3
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

distance_from_home = st.number_input(
    "Distance From Home",
    1,
    50,
    10
)

if st.button("Predict Attrition"):

    # Convert department to the same numbers used during training
    department_map = {
        "Engineering": 0,
        "Finance": 1,
        "HR": 2,
        "Marketing": 3,
        "Sales": 4
    }

    overtime_map = {
        "No": 0,
        "Yes": 1
    }

    input_data = pd.DataFrame([[
        1001,
        age,
        department_map[department],
        monthly_income,
        years_at_company,
        job_satisfaction,
        work_life_balance,
        performance_rating,
        overtime_map[overtime],
        distance_from_home
    ]], columns=[
        "EmployeeID",
        "Age",
        "Department",
        "MonthlyIncome",
        "YearsAtCompany",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "OverTime",
        "DistanceFromHome"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("⚠️ Predicted Attrition: YES")
    else:
        st.success("✅ Predicted Attrition: NO")

    st.write(f"Attrition Probability: {probability * 100:.2f}%")