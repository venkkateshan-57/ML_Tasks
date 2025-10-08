import streamlit as st
import pandas as pd
import altair as alt

# Load the segmented customer data
try:
    df_segmented = pd.read_csv('data/customers_rfm_segmented.csv')
except FileNotFoundError:
    st.error("The segmented data file was not found. Please run the `train.py` script first.")
    st.stop()

# --- Define Cluster Interpretations (with the typo fixed) ---
interpretations = {
    0: "Champions: Recent, frequent, and high-spending.",
    1: "At-Risk: Not purchased recently, average frequency/spending.",
    2: "Loyal Customers: Average recency, but frequent and high-spending.",
    3: "New/Promising: Very recent, but low frequency/spending." # <-- FIX IS HERE
}

# --- Streamlit App UI ---
st.set_page_config(page_title="RFM Customer Segmentation", layout="wide")
st.title("🛍️ RFM Customer Segmentation")
st.write("Customers are segmented based on Recency, Frequency, and Monetary (RFM) analysis.")

# --- Display the Cluster Visualization (using PCA) ---
st.subheader("Customer Segments Visualization (PCA)")
st.write("Since we have 3 features (R, F, M), we use PCA to visualize the clusters in 2D.")

chart = alt.Chart(df_segmented).mark_circle(size=100).encode(
    x=alt.X('PCA1:Q', scale=alt.Scale(zero=False), title='Principal Component 1'),
    y=alt.Y('PCA2:Q', scale=alt.Scale(zero=False), title='Principal Component 2'),
    color=alt.Color('Cluster:N', scale=alt.Scale(scheme='viridis')),
    tooltip=['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster']
).properties(
    height=500
).interactive()

st.altair_chart(chart, use_container_width=True)

# --- Explore the Segments ---
st.subheader("Explore Customer Segments")
st.write("Select a cluster to see the average characteristics of its customers.")

segment_summary = df_segmented.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
segment_summary['Segment Size'] = df_segmented['Cluster'].value_counts()
segment_summary['Interpretation'] = segment_summary.index.map(lambda x: interpretations.get(x, "General Segment"))

st.dataframe(segment_summary, use_container_width=True)