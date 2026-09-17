import streamlit as st
import pandas as pd

forecast = pd.read_csv(
    "reports/forecast_output.csv"
)

st.title("Demand Forecast")

st.line_chart(
    forecast["Forecast"]
)