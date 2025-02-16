import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet('./data/e-commerce-dataset.parquet', engine='pyarrow')
    customer_df = pd.read_csv('./data/customer-segmentation.csv')
    clv_df = pd.read_csv('./data/customer-lifetime-value.csv')
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df, customer_df, clv_df

df, customer_df, clv_df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
selected_segment = st.sidebar.multiselect("Select Customer Segments", customer_df["segment"].unique(), customer_df["segment"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["order_purchase_timestamp"].min(), df["order_purchase_timestamp"].max()])
product_category = st.sidebar.multiselect("Select Product Categories", df["product_category_name"].unique(), df["product_category_name"].unique())
churn_threshold = st.sidebar.slider("Define Churn Threshold (Days)", min_value=30, max_value=365, value=180)

# Filter Data
filtered_df = df[(df["order_purchase_timestamp"] >= pd.to_datetime(date_range[0])) & 
                 (df["order_purchase_timestamp"] <= pd.to_datetime(date_range[1]))]
filtered_df = filtered_df[filtered_df["product_category_name"].isin(product_category)]
filtered_customer_df = customer_df[customer_df["segment"].isin(selected_segment)]

# Dynamic Key Metrics
total_customers = filtered_df['customer_unique_id'].nunique()
total_revenue = filtered_df['payment_value'].sum()
avg_order_value = filtered_df['payment_value'].mean()
churn_rate = (filtered_df[filtered_df['recency'] > churn_threshold].shape[0] / total_customers) * 100

# Dashboard Title
st.title("📊 Advanced E-Commerce Analytics Dashboard")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}", help="Total unique customers in the selected segment and date range.")
col2.metric("Total Revenue", f"${total_revenue:,.2f}", help="Total revenue generated in the selected segment and date range.")
col3.metric("Average Order Value", f"${avg_order_value:,.2f}", help="Average value of orders in the selected segment and date range.")
col4.metric("Churn Rate", f"{churn_rate:.2f}%", help=f"Percentage of customers who haven't made a purchase in the last {churn_threshold} days.")

# Customer Segmentation (RFM)
st.subheader("📌 Customer Segmentation (RFM)")
fig1 = px.scatter(
    filtered_customer_df, x="frequency", y="total_spending", color="segment",
    title="Customer Segments Based on Frequency & Spending",
    labels={"frequency": "Total Orders", "total_spending": "Total Spending"},
    size_max=10,
    hover_data=["customer_unique_id"]
)
st.plotly_chart(fig1)

# Advanced Visualizations
st.subheader("🌐 Advanced Visualizations")
st.markdown("### Heatmap: Customer Activity Over Time")
heatmap_data = filtered_df.groupby([filtered_df['order_purchase_timestamp'].dt.date, 'product_category_name']).size().unstack()
fig2 = px.imshow(heatmap_data, labels=dict(x="Product Category", y="Date", color="Activity"), title="Customer Activity Heatmap")
st.plotly_chart(fig2)

st.markdown("### Treemap: Revenue by Product Category")
treemap_data = filtered_df.groupby('product_category_name')['payment_value'].sum().reset_index()
fig3 = px.treemap(treemap_data, path=['product_category_name'], values='payment_value', title="Revenue by Product Category")
st.plotly_chart(fig3)

# Customizable Dashboard
st.subheader("🛠️ Customize Your Dashboard")
selected_metrics = st.multiselect("Select Metrics to Display", ["Total Customers", "Total Revenue", "Average Order Value", "Churn Rate"])
selected_visualizations = st.multiselect("Select Visualizations to Display", ["RFM Scatter Plot", "Activity Heatmap", "Revenue Treemap", "Economic Trends"])

# Actionable Insights
st.subheader("📢 Actionable Insights")
for segment in selected_segment:
    st.markdown(f"### 🏷️ Segment: {segment}")
    if segment == "Lost Customers":
        st.write("🛑 **Insight:** These customers haven't made a purchase in a while. Offer aggressive re-engagement campaigns and discounts.")
    elif segment == "Potential Loyalists":
        st.write("🎁 **Insight:** These customers show potential for loyalty. Encourage repeat purchases with personalized promotions.")
    elif segment == "Loyal Customers":
        st.write("💎 **Insight:** These are your most valuable customers. Provide VIP perks and referral programs to retain them.")
    else:
        st.write("📈 **Insight:** Focus on optimizing the new customer experience and conversion rates.")

st.success("🎯 This dashboard provides deep insights for data-driven marketing decisions!")
