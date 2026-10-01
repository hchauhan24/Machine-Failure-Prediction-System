import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_artifact():
    return joblib.load('./model/machine_failure_model.joblib')

artifact = load_artifact()
model = artifact['model']
columns = artifact['columns']
threshold = artifact['threshold']

def make_features(air, process, speed, torque, wear, machine_type):
    row = {
        'Air temperature [K]': air,
        'Process temperature [K]': process,
        'Rotational speed [rpm]': speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': wear,
        'Type_L': 1 if machine_type == 'L' else 0,
        'Type_M': 1 if machine_type == 'M' else 0,
        'Power': torque * speed,
        'Temp diff': process - air,
        'Strain': wear * torque,
    }
    return pd.DataFrame([row])[columns]

st.title('Machine Failure Prediction')
st.write('Enter sensor readings to estimate the risk that the machine fails.')

machine_type = st.selectbox('Product quality type', ['L', 'M', 'H'])
air = st.number_input('Air temperature [K]', 295.0, 305.0, 300.0, step=0.1)
process = st.number_input('Process temperature [K]', 305.0, 314.0, 310.0, step=0.1)
speed = st.number_input('Rotational speed [rpm]', 1000, 3000, 1500, step=10)
torque = st.number_input('Torque [Nm]', 0.0, 80.0, 40.0, step=0.5)
wear = st.number_input('Tool wear [min]', 0, 260, 100, step=1)

if st.button('Predict'):
    features = make_features(air, process, speed, torque, wear, machine_type)
    prob = model.predict_proba(features)[0, 1]
    st.metric('Failure probability', f'{prob:.1%}')
    if prob >= threshold:
        st.error('High risk: schedule an inspection.')
    else:
        st.success('Low risk.')
    st.caption(f'Alert threshold: {threshold:.0%}. Trained on a synthetic dataset; this is a demo.')