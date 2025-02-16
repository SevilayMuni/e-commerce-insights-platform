import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet('./data/merged-e-commerce-df.parquet', engine='pyarrow')
    return df

df = load_data()

# Convert Dates
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
selected_segment = st.sidebar.multiselect("Select Customer Segments", df["segment"].unique(), df["segment"].unique())

# Filter Data
filtered_df = df[df["segment"].isin(selected_segment)]

# Dashboard Title
st.title("📊 Personalized Marketing Dashboard")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{df['customer_unique_id'].nunique():,}")
col2.metric("Total Revenue", f"${df['payment_value'].sum():,.2f}")
col3.metric("Average Order Value", f"${df['payment_value'].mean():,.2f}")
col4.metric("Churn Rate", f"{(df[df['recency'] > 180].shape[0] / df['customer_unique_id'].nunique()) * 100:.2f}%")

# Customer Segmentation (RFM)
st.subheader("📌 Customer Segmentation (RFM)")
fig1 = px.scatter(
    df, x="frequency", y="total_spending", color="segment",
    title="Customer Segments Based on Frequency & Spending",
    labels={"frequency": "Total Orders", "total_spending": "Total Spending"},
    size_max=10
)
st.plotly_chart(fig1)

# Spending Analysis
st.subheader("💰 Spending Distribution Across Segments")
fig2 = px.box(filtered_df, x="segment", y="total_spending", color="segment",
              title="Spending Analysis by Customer Segment")
st.plotly_chart(fig2)

# Churn Analysis
st.subheader("⚠️ Churn Risk Analysis")
df["churn_risk"] = df["recency"].apply(lambda x: "High Risk" if x > 180 else "Low Risk")
fig3 = px.pie(df, names="churn_risk", title="Churn Risk Distribution")
st.plotly_chart(fig3)

# Customer Lifetime Value (CLV)
st.subheader("🔍 Customer Lifetime Value (CLV) by Quarter")
df["quarter"] = df["order_purchase_timestamp"].dt.to_period("Q")
clv_by_quarter = df.groupby("quarter")["payment_value"].sum().reset_index()
fig4 = px.line(clv_by_quarter, x="quarter", y="payment_value", markers=True, title="CLV Trends Over Time")
st.plotly_chart(fig4)

# Marketing Recommendations
st.subheader("📢 Personalized Marketing Strategies")
for segment in df["segment"].unique():
    st.markdown(f"### 🏷️ Segment: {segment}")
    if segment == "Lost Customers":
        st.write("🛑 Offer aggressive re-engagement campaigns and discounts.")
    elif segment == "Potential Loyalists":
        st.write("🎁 Encourage repeat purchases with personalized promotions.")
    elif segment == "Loyal Customers":
        st.write("💎 Provide VIP perks and referral programs.")
    else:
        st.write("📈 Optimize new customer experience and conversion rates.")

st.success("🎯 This dashboard provides deep insights for data-driven marketing decisions!")
