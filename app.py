import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="HEA Predictor", layout="wide")
st.title("🧪 High-Entropy Alloy Predictor")

@st.cache_resource
def load_models():
    phase_model = joblib.load('phase_model.pkl')
    mech_model = joblib.load('mech_model.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('features.pkl')
    return phase_model, mech_model, le, features

phase_model, mech_model, le, features = load_models()

st.sidebar.header("1. Design Your Alloy")
selected_elements = st.sidebar.multiselect("Select Elements:", options=features, default=['Al', 'Cr', 'Fe', 'Mn', 'Ni'])

if selected_elements:
    raw_inputs = {}
    for el in selected_elements:
        # You can drag the slider OR click the number on the right to type manually
        raw_inputs[el] = st.sidebar.slider(f"{el} proportion", 0.0, 5.0, 1.0, 0.01)

    total = sum(raw_inputs.values())
    
    if total > 0:
        # Format the name like Al0.5CrFe1.5MnNi0.5
        formula = ""
        for el, val in raw_inputs.items():
            if val > 0:
                # If 1.0, show nothing. If whole number, remove decimal. Else show exact value.
                val_str = "" if val == 1.0 else (str(int(val)) if val.is_integer() else str(round(val, 2)))
                formula += f"{el}{val_str}"
                
        st.markdown(f"### Alloy Formula: **{formula}**")

        # Normalize internally for the ML model
        normalized = {el: val/total for el, val in raw_inputs.items()}
        final_input = {el: normalized.get(el, 0.0) for el in features}
        input_df = pd.DataFrame([final_input])

        phase_pred = phase_model.predict(input_df)
        predicted_phase = le.inverse_transform(phase_pred)[0]
        mech_pred = mech_model.predict(input_df)[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            plot_data = {el: val for el, val in normalized.items() if val > 0}
            fig = px.pie(names=list(plot_data.keys()), values=list(plot_data.values()), hole=0.4, title="Atomic Fraction")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("<br>", unsafe_allow_html=True)
            st.info(f"**Predicted Phase:** {predicted_phase}")
            st.success(f"**Hardness:** {mech_pred[3]:.2f} GPa")
            st.metric("Young's Modulus", f"{mech_pred[0]:.2f} GPa")
            st.metric("Shear Modulus", f"{mech_pred[1]:.2f} GPa")
            st.metric("Bulk Modulus", f"{mech_pred[2]:.2f} GPa")