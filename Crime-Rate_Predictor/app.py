import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load all the models and data files
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
city_summary = pd.read_csv('data/city_summary_for_prediction.csv')
regression_cols = rf_model.feature_names_in_

# --- Streamlit App UI ---
st.set_page_config(page_title="Indian Crime Prediction", layout="wide")
st.title("Indian Crime Prediction")
st.write("Use the tools below to classify cities or forecast crime trends.")

# Get the list of cities for the dropdowns
cities = sorted(city_summary['City'].unique().tolist())

# --- Create two columns for the two models ---
col1, col2 = st.columns(2)

# --- Column 1: SVM Classifier ---
with col1:
    st.subheader("Is a City Crime-Prone?")
    svm_city = st.selectbox("Select a City", cities, key='svm_city')

    if st.button("Classify City"):
        city_features = city_summary[city_summary['City'] == svm_city].drop(['City', 'TOTAL_CRIMES', 'IS_CRIME_PRONE'], axis=1)
        city_features_scaled = scaler.transform(city_features)

        prediction = svm_model.predict(city_features_scaled)
        result = 'Crime-Prone' if prediction[0] == 1 else 'Not Crime-Prone'

        if result == 'Crime-Prone':
            st.error(f"Prediction for {svm_city}: **{result}**")
        else:
            st.success(f"Prediction for {svm_city}: **{result}**")

# --- Column 2: Regression Forecaster ---
with col2:
    st.subheader("Forecast Crime Trend")
    reg_city = st.selectbox("Select a City", cities, key='reg_city')
    year = st.number_input("Enter a Future Year", min_value=2024, max_value=2050, value=2025)

    if st.button("Forecast Crimes"):
        input_data = pd.DataFrame(np.zeros((1, len(regression_cols))), columns=regression_cols)
        input_data['YEAR'] = year
        city_col_name = f'City_{reg_city}'
        if city_col_name in input_data.columns:
            input_data[city_col_name] = 1

        prediction = rf_model.predict(input_data)
        st.info(f"Predicted Total Crimes for {reg_city} in {year}: **{int(prediction[0])}**")