"""
Diabetes Prediction Web Application
===================================
CAI2C08 - Machine Learning for Developers
Temasek Polytechnic
"""

# ============================================
# IMPORT LIBRARIES
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="plus",
    layout="centered"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load("diabetes_final_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'diabetes_final_model.pkl' is in the same directory.")
        return None

model = load_model()

# ============================================
# APP HEADER
# ============================================
st.title("Diabetes Risk Prediction")
st.write("Enter patient information below to assess diabetes risk.")

st.markdown("---")

# ============================================
# SIDEBAR - INFORMATION
# ============================================
with st.sidebar:
    st.header("About This Application")
    st.write("""
    This application predicts diabetes risk based on:
    - Patient demographics
    - Medical history
    - Health indicators
    """)
    
    st.markdown("---")
    
    st.subheader("How to Use")
    st.write("""
    1. Enter patient details in the form
    2. Click 'Predict Risk'
    3. View the prediction result
    """)
    
    st.markdown("---")
    
    st.write("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")

# ============================================
# INPUT FORM
# ============================================
st.subheader("Patient Information")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.write("**Demographics**")
    
    # Age
    age = st.slider(
        "Age (years)",
        min_value=1,
        max_value=100,
        value=45,
        help="Patient's age in years"
    )
    
    # Gender
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male", "Other"]
    )
    
    # Smoking history
    smoking_history = st.selectbox(
        "Smoking History",
        options=["never", "former", "current", "not current", "ever", "No Info"]
    )

with col2:
    st.write("**Health Indicators**")
    
    # BMI
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        help="Normal range: 18.5-24.9"
    )
    
    # HbA1c level
    hba1c = st.number_input(
        "HbA1c Level (%)",
        min_value=3.0,
        max_value=15.0,
        value=5.5,
        step=0.1,
        help="Normal: below 5.7%"
    )
    
    # Blood glucose level
    blood_glucose = st.number_input(
        "Blood Glucose Level (mg/dL)",
        min_value=50,
        max_value=400,
        value=100,
        step=1,
        help="Normal fasting: 70-100 mg/dL"
    )

st.markdown("---")

# Medical history
st.write("**Medical History**")
col3, col4 = st.columns(2)

with col3:
    hypertension = st.checkbox("Hypertension (High Blood Pressure)")

with col4:
    heart_disease = st.checkbox("Heart Disease")

st.markdown("---")

# ============================================
# PREDICTION
# ============================================
if st.button("Predict Risk", type="primary"):
    if model is not None:
        # Prepare input data
        gender_Male = 1 if gender == "Male" else 0
        gender_Other = 1 if gender == "Other" else 0
        
        smoking_current = 1 if smoking_history == "current" else 0
        smoking_ever = 1 if smoking_history == "ever" else 0
        smoking_former = 1 if smoking_history == "former" else 0
        smoking_never = 1 if smoking_history == "never" else 0
        smoking_not_current = 1 if smoking_history == "not current" else 0
        
        hypertension_val = 1 if hypertension else 0
        heart_disease_val = 1 if heart_disease else 0
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'age': age,
            'hypertension': hypertension_val,
            'heart_disease': heart_disease_val,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': blood_glucose,
            'gender_Male': gender_Male,
            'gender_Other': gender_Other,
            'smoking_history_current': smoking_current,
            'smoking_history_ever': smoking_ever,
            'smoking_history_former': smoking_former,
            'smoking_history_never': smoking_never,
            'smoking_history_not current': smoking_not_current
        }])
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error("HIGH RISK - Diabetes Risk Detected")
                st.write(f"**Probability of Diabetes: {probability[1]*100:.1f}%**")
                
                st.warning("""
                **Recommendations:**
                - Consult a healthcare professional for further testing
                - Monitor blood glucose levels regularly
                - Consider lifestyle modifications (diet, exercise)
                - Schedule regular check-ups
                """)
                
            else:
                st.success("LOW RISK - No Diabetes Risk Detected")
                st.write(f"**Probability of No Diabetes: {probability[0]*100:.1f}%**")
                
                st.info("""
                **Recommendations:**
                - Maintain a healthy lifestyle
                - Continue regular exercise
                - Eat a balanced diet
                - Schedule annual health check-ups
                """)
            
            # Input summary
            with st.expander("View Input Summary"):
                st.write("**Patient Information Entered:**")
                summary_data = {
                    'Parameter': ['Age', 'Gender', 'BMI', 'HbA1c Level', 'Blood Glucose', 
                                  'Hypertension', 'Heart Disease', 'Smoking History'],
                    'Value': [f"{age} years", gender, f"{bmi:.1f}", f"{hba1c:.1f}%", 
                             f"{blood_glucose} mg/dL", "Yes" if hypertension else "No",
                             "Yes" if heart_disease else "No", smoking_history]
                }
                st.table(pd.DataFrame(summary_data))
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Please check if the model file exists.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("Diabetes Prediction Application | CAI2C08 Machine Learning Project | Temasek Polytechnic")
st.caption("This is for educational purposes only. Always consult healthcare professionals for medical advice.")
