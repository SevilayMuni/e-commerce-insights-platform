import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_parquet('./data/e-commerce-dataset.parquet', engine='pyarrow')
customer_df = pd.read_csv('./data/customer-segmentation.csv')
clv_df = pd.read_csv('./data/customer-lifetime-value.csv')

# Convert Dates
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
selected_segment = st.sidebar.multiselect("Select Customer Segments", customer_df["segment"].unique(), customer_df["segment"].unique())

# Filter Data
filtered_df = customer_df[customer_df["segment"].isin(selected_segment)]

# Create 'recency' column
max_purchase_date = df['order_purchase_timestamp'].max()
last_purchase_date = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
df['recency'] = (max_purchase_date - last_purchase_date['order_purchase_timestamp']).dt.days

# Dashboard Title
st.title("📊 Personalized Marketing Dashboard")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{df['customer_unique_id'].nunique():,}")
col2.metric("Total Revenue", f"${df['payment_value'].sum():,.2f}")
col3.metric("Average Order Value", f"${df['payment_value'].mean():,.2f}")
col4.metric("Churn Rate", f"{(df[df['recency'] > 90].shape[0] / df['customer_unique_id'].nunique()) * 100:.2f}%")

# Customer Segmentation (RFM)
st.subheader("📌 Customer Segmentation (RFM)")
fig1 = px.scatter(
    customer_df, x="frequency", y="total_spending", color="segment",
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
st.subheader("🔍 Quarterly CLV vs Weighted CLV")

# Convert quarter to string for proper axis formatting
clv_df["quarter"] = clv_df["quarter"].astype(str)

# Create interactive line chart
fig4 = px.line(clv_df, x="quarter", y=["clv", "weighted_clv"],
               markers=True, title="Quarterly CLV vs Weighted CLV",
               labels={"quarter": "Quarter", "value": "CLV"},
               color_discrete_map={"clv": "teal", "weighted_clv": "firebrick"})

# Rename legend labels
fig4.update_traces(name="Quarterly CLV", selector=dict(name="clv"))
fig4.update_traces(name="Weighted CLV", selector=dict(name="weighted_clv"))

# Display in Streamlit
st.plotly_chart(fig4)

# Marketing Recommendations
st.subheader("📢 Personalized Marketing Strategies")
for segment in customer_df["segment"].unique():
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
