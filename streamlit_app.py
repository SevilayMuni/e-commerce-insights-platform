import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet('./data/e-commerce-dataset.parquet', engine='pyarrow')
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    max_purchase_date = df['order_purchase_timestamp'].max()
    last_purchase_date = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
    df['recency'] = (max_purchase_date - last_purchase_date['order_purchase_timestamp']).dt.days
    customer_df = pd.read_csv('./data/customer-segmentation.csv')
    clv_df = pd.read_csv('./data/customer-lifetime-value.csv')
    return df, customer_df, clv_df

df, customer_df, clv_df = load_data()

# Top Navigation Bar
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Customer Insights", "Product Analysis", "Economic Trends"])


# Collapsible Filters
with st.sidebar.expander("🔍 Filter Data"):
    # Ensure segment names are uniform and match the dataset
    segment_options = customer_df["segment"].unique()
    
    # Clean segment names (remove any unintended characters like "x")
    cleaned_segment_options = [seg.strip().replace(" x", "") for seg in segment_options]
    
    # Checkbox-based segment selection
    st.write("Select Customer Segments:")
    selected_segments = []  # Use 'selected_segments' to store selected segments
    for segment in cleaned_segment_options:
        if st.checkbox(segment, value=(segment in ["promising customer", "at risk customer"])):
            selected_segments.append(segment)
    
    # Date Range Picker
    date_range = st.date_input(
        "Select Date Range", 
        [df["order_purchase_timestamp"].min(), df["order_purchase_timestamp"].max()]
    )
    
    # Product Categories
    product_category = st.multiselect(
        "Select Product Categories", 
        df["product_category"].unique(), default=["electronics", "furniture_decor", "health_beauty"])
    
    # Churn Threshold Slider
    churn_threshold = st.slider("Define Churn Threshold (Days)", min_value=30, max_value=365, value=180)

# Filter Data Dynamically
filtered_df = df[(df["order_purchase_timestamp"] >= pd.to_datetime(date_range[0])) & 
                 (df["order_purchase_timestamp"] <= pd.to_datetime(date_range[1]))]
filtered_df = filtered_df[filtered_df["product_category"].isin(product_category)]
filtered_customer_df = customer_df[customer_df["segment"].isin(selected_segments)]

# Dynamic Key Metrics
total_customers = filtered_df['customer_unique_id'].nunique()
total_revenue = filtered_df['payment_value'].sum()
avg_order_value = filtered_df['payment_value'].mean()
churn_rate = (filtered_df[filtered_df['recency'] > churn_threshold].shape[0] / total_customers) * 100

# Customer Insights Tab
if tab == "Customer Insights":
    st.title("👥 Customer Insights")
    
    # Key Metrics in Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}", help="Total unique customers in the selected segment and date range.")
    col2.metric("Total Revenue", f"${total_revenue:,.2f}", help="Total revenue generated in the selected segment and date range.")
    col3.metric("Average Order Value", f"${avg_order_value:,.2f}", help="Average value of orders in the selected segment and date range.")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%", help=f"Percentage of customers who haven't made a purchase in the last {churn_threshold} days.")

    # RFM Analysis
    st.subheader("📌 Customer Segmentation (RFM)")
    fig1 = px.scatter(
        filtered_customer_df, x="frequency", y="total_spending", color="segment",
        title="Customer Segments Based on Frequency & Spending",
        labels={"frequency": "Total Orders", "total_spending": "Total Spending"},
        size_max=10,
        hover_data=["customer_unique_id"]
    )
    st.plotly_chart(fig1)

    # Churn Risk Analysis
    st.subheader("⚠️ Churn Risk Analysis")
    filtered_df["churn_risk"] = filtered_df["recency"].apply(lambda x: "High Risk" if x > churn_threshold else "Low Risk")
    fig2 = px.pie(filtered_df, names="churn_risk", title="Churn Risk Distribution")
    st.plotly_chart(fig2)

# Product Analysis Tab
elif tab == "Product Analysis":
    st.title("📦 Product Analysis")
    
    # Key Metrics in Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Sold", f"{filtered_df.shape[0]:,}", help="Total products sold in the selected categories and date range.")
    col2.metric("Total Revenue", f"${total_revenue:,.2f}", help="Total revenue generated from the selected categories.")
    col3.metric("Top Category", filtered_df['product_category'].mode()[0], help="Most popular product category.")

    # Heatmap: Customer Activity Over Time
    st.subheader("🌐 Customer Activity Heatmap")
    heatmap_data = filtered_df.groupby([filtered_df['order_purchase_timestamp'].dt.date, 'product_category']).size().unstack()
    fig3 = px.imshow(heatmap_data, labels=dict(x="Product Category", y="Date", color="Activity"), title="Customer Activity Heatmap")
    st.plotly_chart(fig3)

    # Treemap: Revenue by Product Category
    st.subheader("💰 Revenue by Product Category")
    treemap_data = filtered_df.groupby('product_category')['payment_value'].sum().reset_index()
    fig4 = px.treemap(treemap_data, path=['product_category'], values='payment_value', title="Revenue by Product Category")
    st.plotly_chart(fig4)

# Economic Trends Tab
elif tab == "Economic Trends":
    st.title("📈 Economic Trends")
