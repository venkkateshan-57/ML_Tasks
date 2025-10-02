import streamlit as st
import pandas as pd
import pickle
from pathlib import Path  # 1. Import the Path library

# --- Build a robust path to the model file ---
# This makes the path relative to the app.py file itself
script_directory = Path(__file__).parent
model_path = script_directory / 'best_model.pkl'

# --- Load the Model and Columns ---
with open(model_path, 'rb') as f: # 2. Use the new model_path
    model_data = pickle.load(f)

model = model_data['model']
model_columns = model_data['columns']

# --- Streamlit App UI (rest of the code is the same) ---
st.set_page_config(page_title="Insurance Charge Predictor", layout="centered")
st.title("💰 Insurance Charge Predictor")
st.write("Enter the client's details to predict their insurance charge.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ("male", "female"))
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.2f")

with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ("yes", "no"))
    region = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))

# --- Prediction Logic ---
if st.button("Predict Charges", use_container_width=True):
    input_data = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'male' else 0],
        'smoker_yes': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    }
    input_df = pd.DataFrame(input_data)
    
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"The predicted insurance charge is: ₹{prediction[0]:.2f}")
