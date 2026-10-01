import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Machine Failure Predictor', page_icon='⚙️', layout='centered')

# ---------- Styling ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 820px; }

.hero { text-align: center; margin-bottom: 1.5rem; }
.hero h1 { color: #ffffff; font-size: 2.2rem; margin-bottom: 0.2rem; }
.hero p { color: #b8c7d1; font-size: 1rem; }

div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
}
.stApp label p { color: #e6eef3 !important; }

.result { border-radius: 16px; padding: 1.4rem; margin-top: 1.2rem; text-align: center; }
.result.high { background: rgba(220, 53, 69, 0.18); border: 1px solid #dc3545; }
.result.low  { background: rgba(40, 167, 69, 0.18); border: 1px solid #28a745; }
.result .label { color: #b8c7d1; font-size: 0.9rem; }
.result .pct { font-size: 3rem; font-weight: 700; color: #ffffff; }
.result .msg { color: #ffffff; font-size: 1.1rem; }
.bar { height: 10px; border-radius: 6px; background: rgba(255,255,255,0.15);
       margin: 0.8rem 0; overflow: hidden; }
.bar > div { height: 100%; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
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

# Observed ranges in the training data (from df.describe())
TRAIN_RANGE = {
    'Air temperature': (295.3, 304.5),
    'Process temperature': (305.7, 313.8),
    'Rotational speed': (1168, 2886),
    'Torque': (3.8, 76.6),
    'Tool wear': (0, 253),
}

st.markdown("""
<div class="hero">
    <h1>⚙️ Machine Failure Predictor</h1>
    <p>Enter the sensor readings to estimate the risk of failure.</p>
</div>
""", unsafe_allow_html=True)

with st.form('inputs'):
    machine_type = st.selectbox('Product quality type', ['L', 'M', 'H'])
    c1, c2 = st.columns(2)
    with c1:
        air = st.number_input('Air temperature [K]', 295.0, 305.0, 300.0, step=0.1)
        speed = st.number_input('Rotational speed [rpm]', 1000, 3000, 1500, step=10)
        wear = st.number_input('Tool wear [min]', 0, 260, 100, step=1)
    with c2:
        process = st.number_input('Process temperature [K]', 305.0, 314.0, 310.0, step=0.1)
        torque = st.number_input('Torque [Nm]', 0.0, 80.0, 40.0, step=0.5)
    submitted = st.form_submit_button('Predict', use_container_width=True)


if submitted:
    features = make_features(air, process, speed, torque, wear, machine_type)
    prob = model.predict_proba(features)[0, 1]
    high = prob >= threshold
    css_class = 'high' if high else 'low'
    color = '#dc3545' if high else '#28a745'
    message = 'High risk: schedule an inspection' if high else 'Low risk: no action needed'

    st.markdown(f"""
    <div class="result {css_class}">
        <div class="label">Failure probability</div>
        <div class="pct">{prob:.1%}</div>
        <div class="bar"><div style="width:{prob*100:.1f}%; background:{color};"></div></div>
        <div class="msg">{message}</div>
    </div>
    """, unsafe_allow_html=True)

    # Warn if any input is outside what the model saw in training
    inputs = {'Air temperature': air, 'Process temperature': process,
              'Rotational speed': speed, 'Torque': torque, 'Tool wear': wear}
    outside = [name for name, v in inputs.items()
               if not TRAIN_RANGE[name][0] <= v <= TRAIN_RANGE[name][1]]
    if outside:
        st.warning('Outside the training range: ' + ', '.join(outside) +
                   '. The prediction may be unreliable.')

  