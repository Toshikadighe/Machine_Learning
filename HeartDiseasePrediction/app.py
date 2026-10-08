import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'Logistic_Regression.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
expected_columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))

# model = joblib.load('Logistic_Regression.pkl')
# scaler = joblib.load('scaler.pkl')
# expected_columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction')
st.markdown('Provide the following information to predict the likelihood of heart disease:')
age = st.slider('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', options=['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', options=['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
resting_ecg = st.selectbox('Resting ECG Results', options=['Normal', 'ST', 'LVH'])
max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise Induced Angina', options=['Y', 'N'])
oldpeak = st.slider('Oldpeak (ST depression induced by exercise relative to rest)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox('ST Slope', options=['Up','Flat','Down'])

if st.button('Predict'):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 1,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_'+ sex: 1,
        'RestingECG_'+ resting_ecg: 1,
        'ST_Slope_'+ st_slope: 1,
        'ExerciseAngina_'+ exercise_angina: 1,
        'ChestPainType_'+ chest_pain: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = input_df.copy()
    scaled_columns = scaler.feature_names_in_
    scaled_input[scaled_columns] = scaler.transform(input_df[scaled_columns])
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error('High risk of heart disease.')
    else:
        st.success('Low risk of heart disease.')