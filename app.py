import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Depression Prediction App", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()

st.title("🧠 Depression Risk Prediction")
st.write("Masukkan data sesuai kondisi responden:")

# Form input
with st.form("prediction_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=10, max_value=100, value=20)
    City = st.text_input("City")
    Profession = st.text_input("Profession")
    AcademicPressure = st.slider("Academic Pressure", 0, 10, 5)
    WorkPressure = st.slider("Work Pressure", 0, 10, 5)
    CGPA = st.number_input("CGPA", min_value=2.0, max_value=10.0, step=0.1, value=7.0)
    StudySatisfaction = st.slider("Study Satisfaction", 0, 10, 5)
    JobSatisfaction = st.slider("Job Satisfaction", 0, 10, 5)
    SleepDuration = st.selectbox("Sleep Duration", 
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    DietaryHabits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
    Degree = st.text_input("Degree")
    SuicidalThoughts = st.selectbox("Suicidal Thoughts?", ["Yes", "No"])
    WorkStudyHours = st.slider("Work / Study Hours per Day", 0, 18, 8)
    FinancialStress = st.slider("Financial Stress", 0, 10, 5)
    FamilyHistory = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    submit = st.form_submit_button("🔮 Predict")

if submit:
    input_data = pd.DataFrame([{
        'Gender': Gender,
        'Age': Age,
        'City': City,
        'Profession': Profession,
        'Academic Pressure': AcademicPressure,
        'Work Pressure': WorkPressure,
        'CGPA': CGPA,
        'Study Satisfaction': StudySatisfaction,
        'Job Satisfaction': JobSatisfaction,
        'Sleep Duration': SleepDuration,
        'Dietary Habits': DietaryHabits,
        'Degree': Degree,
        'Have you ever had suicidal thoughts ?': SuicidalThoughts,
        'Work/Study Hours': WorkStudyHours,
        'Financial Stress': FinancialStress,
        'Family History of Mental Illness': FamilyHistory
    }])

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    prob_yes = probabilities[list(pipeline.classes_).index("Yes")]

    st.success(f"Risk Status : {prediction}")
    st.info(f"Confidence  : {prob_yes*100:.2f}%")
