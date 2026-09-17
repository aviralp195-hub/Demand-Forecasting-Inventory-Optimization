import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

forecast = pd.read_csv(
    "reports/forecast_output.csv",
    engine="python"
)

# --------------------------------------------------
# CALCULATIONS
# --------------------------------------------------

forecast["Error"] = (
    forecast["Actual_Sales"]
    - forecast["Predicted_Sales"]
)

forecast["Safety_Stock"] = (
    forecast["Predicted_Sales"] * 0.20
)

forecast["Reorder_Point"] = (
    forecast["Predicted_Sales"]
    + forecast["Safety_Stock"]
)

accuracy = (
    100
    - (
        abs(forecast["Error"]).mean()
        / forecast["Actual_Sales"].mean()
        * 100
    )
)

mape = (
    abs(
        forecast["Error"]
        / forecast["Actual_Sales"]
    ).mean() * 100
)

total_demand = forecast["Predicted_Sales"].sum()

avg_demand = forecast["Predicted_Sales"].mean()

safety_stock = forecast["Safety_Stock"].mean()

stockout_risk = (
    "LOW"
    if mape < 10
    else "MEDIUM"
)

# --------------------------------------------------
# DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
.main{
background-color:#0E1117;
color:white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Demand Forecasting & Inventory Optimization")

st.markdown("---")

# --------------------------------------------------
# KPI ROW
# --------------------------------------------------

c1,c2,c3,c4,c5,c6 = st.columns(6)

c1.metric(
    "Forecast Accuracy",
    f"{accuracy:.2f}%"
)

c2.metric(
    "MAPE",
    f"{mape:.2f}%"
)

c3.metric(
    "Total Forecast",
    f"{total_demand:,.0f}"
)

c4.metric(
    "Average Demand",
    f"{avg_demand:,.0f}"
)

c5.metric(
    "Stockout Risk",
    stockout_risk
)

c6.metric(
    "Safety Stock",
    f"{safety_stock:,.0f}"
)

st.markdown("---")

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.subheader("🔵 Actual vs Predicted Sales")

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        y=forecast["Actual_Sales"],
        name="Actual",
        line=dict(color="orange")
    )
)

fig1.add_trace(
    go.Scatter(
        y=forecast["Predicted_Sales"],
        name="Forecast",
        line=dict(color="blue")
    )
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# --------------------------------------------------
# ERROR HISTOGRAM
# --------------------------------------------------

col1,col2 = st.columns(2)

with col1:

    st.subheader("🔴 Forecast Error Distribution")

    fig2 = px.histogram(
        forecast,
        x="Error",
        nbins=30,
        color_discrete_sequence=["red"]
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# --------------------------------------------------
# MONTHLY DEMAND TREND
# --------------------------------------------------

with col2:

    st.subheader("🟢 Demand Trend")

    trend = (
        forecast["Predicted_Sales"]
        .rolling(10)
        .mean()
    )

    fig3 = px.line(
        y=trend,
        color_discrete_sequence=["green"]
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# --------------------------------------------------
# INVENTORY CHARTS
# --------------------------------------------------

c1,c2 = st.columns(2)

with c1:

    st.subheader("🟣 Safety Stock vs Forecast")

    fig4 = go.Figure()

    fig4.add_trace(
        go.Bar(
            y=forecast["Predicted_Sales"][:50],
            name="Forecast"
        )
    )

    fig4.add_trace(
        go.Bar(
            y=forecast["Safety_Stock"][:50],
            name="Safety Stock"
        )
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

with c2:

    st.subheader("🟡 Reorder Point Analysis")

    fig5 = px.scatter(
        forecast,
        x="Predicted_Sales",
        y="Reorder_Point",
        color="Reorder_Point"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# --------------------------------------------------
# TABLES
# --------------------------------------------------

st.markdown("---")

tab1,tab2,tab3 = st.tabs([
    "Top Demand Days",
    "Inventory Recommendations",
    "Forecast Statistics"
])

with tab1:

    top_days = forecast.sort_values(
        "Predicted_Sales",
        ascending=False
    ).head(20)

    st.dataframe(top_days)

with tab2:

    st.dataframe(
        forecast[
            [
                "Predicted_Sales",
                "Safety_Stock",
                "Reorder_Point"
            ]
        ]
    )

with tab3:

    st.dataframe(
        forecast.describe()
    )

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

st.download_button(
    "Download Forecast CSV",
    forecast.to_csv(index=False),
    file_name="forecast_results.csv",
    mime="text/csv"
)