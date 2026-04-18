import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Dynamic HEA Predictor", layout="wide")
st.title("Dynamic High-Entropy Alloy Predictor")

# 1. Load models and features (cached so it loads fast)
@st.cache_resource
def load_models():
    phase_model = joblib.load('phase_model.pkl')
    mech_model = joblib.load('mech_model.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('features.pkl')
    return phase_model, mech_model, le, features

phase_model, mech_model, le, features = load_models()

# 2. Sidebar setup for Dynamic Inputs
st.sidebar.header("1. Design Your Alloy")

# Default famous Cantor alloy elements
default_elements = ['Al', 'Co', 'Cr', 'Fe', 'Ni']

# Multi-select so user can pick which of the 62 elements they want to use
selected_elements = st.sidebar.multiselect(
    "Select Elements for your HEA:", 
    options=features, 
    default=default_elements
)

if not selected_elements:
    st.warning("👈 Please select at least one element from the sidebar to begin.")
else:
    st.sidebar.subheader("2. Adjust Proportions")
    st.sidebar.caption("Move sliders to see real-time prediction updates!")
    
    # 3. Create sliders dynamically for only the selected elements
    raw_inputs = {}
    for el in selected_elements:
        raw_inputs[el] = st.sidebar.slider(f"{el} proportion", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

    # Calculate total to normalize
    total = sum(raw_inputs.values())
    
    if total == 0:
        st.error("Total composition cannot be zero. Please increase at least one slider.")
    else:
        # Normalize so they act as atomic fractions (sum to 1.0)
        normalized_inputs = {el: val/total for el, val in raw_inputs.items()}
        
        # Create a dictionary with 0.0 for ALL 62 elements, then update the selected ones
        final_input = {el: 0.0 for el in features}
        for el, val in normalized_inputs.items():
            final_input[el] = val
            
        # Convert to DataFrame for the ML Model
        input_df = pd.DataFrame([final_input])

        # 4. Make Real-Time Predictions
        phase_pred = phase_model.predict(input_df)
        predicted_phase = le.inverse_transform(phase_pred)[0]
        
        mech_pred = mech_model.predict(input_df)[0]
        
        # 5. Dynamic Dashboard Display
        st.write("### Normalized Composition")
        # Format a nice formula string (e.g., Al20.0 Co20.0)
        comp_str = " ".join([f"**{el}**_{val*100:.1f}%" for el, val in normalized_inputs.items() if val > 0])
        st.markdown(f"Current Alloy: {comp_str}")

        # Big metric cards for the main predictions
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Predicted Phase Structure:**\n###  {predicted_phase}")
        with col2:
            st.success(f"**Hardness (Vickers):**\n###  {mech_pred[3]:.2f} GPa")
            
        st.write("---")
        st.write("### Additional Mechanical Properties")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Young's Modulus", f"{mech_pred[0]:.2f} GPa")
        mcol2.metric("Shear Modulus", f"{mech_pred[1]:.2f} GPa")
        mcol3.metric("Bulk Modulus", f"{mech_pred[2]:.2f} GPa")