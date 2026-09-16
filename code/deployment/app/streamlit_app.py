import streamlit as st
import requests

API_URL = "http://api:8000/predict"

st.title("Bank Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (years as customer)", min_value=0, max_value=15, value=3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict"):
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    input_data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography_Germany": geography_germany,
        "Geography_Spain": geography_spain,
        "Gender_Male": gender_male
    }

    response = requests.post(API_URL, json=input_data)
    result = response.json()

    if result["prediction"] == 1:
        st.error(f"This customer is likely to churn (probability: {result['probability']:.2%})")
    else:
        st.success(f"This customer is likely to stay (churn probability: {result['probability']:.2%})")