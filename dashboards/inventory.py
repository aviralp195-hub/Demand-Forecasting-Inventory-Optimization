import streamlit as st
import pandas as pd

inventory = pd.read_csv(
    "reports/inventory_plan.csv"
)

st.title("Inventory Optimization")

st.dataframe(inventory)