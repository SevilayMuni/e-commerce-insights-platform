import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import datetime

# FRED API Key
FRED_API_KEY = "fe01e8ff873c535a4652b9f1bc78b788"

# Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet('./data/e-commerce-dataset.parquet', engine='pyarrow')
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    max_purchase_date = df['order_purchase_timestamp'].max()
    last_purchase_date = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
    df['recency'] = (max_purchase_date - last_purchase_date['order_purchase_timestamp']).dt.days
    df['product_category'] = df['product_category'].str.replace("_", " ").str.title()
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
    
    # Checkbox-based segment selection with default selections
    st.write("Select Customer Segments:")
    selected_segments = []  # Use 'selected_segments' to store selected segments
    default_segments = ["Promising Customers", "At Risk Customers"]  # Default segments
    for segment in cleaned_segment_options:
        if st.checkbox(segment, value=(segment in default_segments)):
            selected_segments.append(segment)
    
    # Date Range Picker
    date_range = st.date_input(
        "Select Date Range", 
        [df["order_purchase_timestamp"].min(), df["order_purchase_timestamp"].max()]
    )
    
    # Product Categories
    product_category = st.multiselect(
        "Select Product Categories", 
        df["product_category"].unique(), default=["Electronics", "Furniture Decor", "Health Beauty"])
    
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
formatted_revenue = f"${total_revenue/1e6:.2f}M" if total_revenue > 1e6 else f"${total_revenue:,.2f}"
avg_order_value = filtered_df['payment_value'].mean()
churn_rate = (filtered_df[filtered_df['recency'] > churn_threshold].shape[0] / total_customers) * 100

# Customer Insights Tab
if tab == "Customer Insights":
    st.title("👥 Customer Insights")
    
    # Key Metrics in Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}", help="Total unique customers in the selected segment and date range.")
    col2.metric("Total Revenue $", formatted_revenue, help="Total revenue generated in the selected segment and date range.")
    col3.metric("Average Order Value", f"${avg_order_value:,.2f}", help="Average value of orders in the selected segment and date range.")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%", help=f"Percentage of customers who haven't made a purchase in the last {churn_threshold} days.")

    # RFM Analysis
    st.subheader("📌 Customer Segmentation (RFM)")
    fig1 = px.scatter(
        filtered_customer_df, x="frequency", y="total_spending", color="segment",
        title="Customer Segments Based on Frequency & Spending",
        labels={"frequency": "Total Orders", "total_spending": "Total Spending"},
        size_max=10,
        hover_data=["customer_id"])
    st.plotly_chart(fig1)
    # CLV Graph
    if tab == "Customer Insights":
        st.subheader("📊 Customer Lifetime Value (CLV)")
        clv_df["quarter"] = clv_df["quarter"].astype(str)  # Convert quarter to string for proper axis formatting
        fig_clv = px.line(clv_df, x="quarter", y=["clv", "weighted_clv"], 
                          title="Customer Lifetime Value (CLV) Over Time",
                          labels={"variable": "Variable", "clv": "CLV", "weighted_clv": "Weighted CLV"},
                          color_discrete_map={"clv": "teal", "weighted_clv": "firebrick"})
        st.plotly_chart(fig_clv)
    # Churn Risk Analysis
    st.subheader("⚠️ Churn Risk Analysis")
    filtered_df["churn_risk"] = filtered_df["recency"].apply(lambda x: "High Risk" if x > churn_threshold else "Low Risk")
    fig2 = px.pie(filtered_df, names="churn_risk", title="Churn Risk Distribution")
    st.plotly_chart(fig2)
    
    # Customer Distribution Map
    st.subheader("🌍 Customer Distribution by Region")
    if "customer_state" in df.columns:
        map_data = df.groupby("customer_state")["payment_value"].sum().reset_index()
        fig_map = px.choropleth(map_data, locations="customer_state", locationmode="Brazil-states", 
                                color="payment_value", title="Sales by Region")
        st.plotly_chart(fig_map)

# Product Analysis Tab
elif tab == "Product Analysis":
    st.title("📦 Product Analysis")
    
    # Key Metrics in Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Sold", f"{filtered_df.shape[0]:,}", help="Total products sold in the selected categories and date range.")
    col2.metric("Total Revenue $", formatted_revenue, help="Total revenue generated from the selected categories.")
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
if tab == "Economic Trends":
    st.title("📈 Economic Trends")
    
    # Key Metrics in Cards for Economic Trends
    st.subheader("Key Economic Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inflation (CPI)", "2.5%", help="Latest Consumer Price Index (CPI) data.")
    col2.metric("Interest Rates", "4.25%", help="Federal Funds Rate.")
    col3.metric("Unemployment Rate", "3.8%", help="Latest unemployment rate.")
    col4.metric("Retail Sales Growth", "1.2%", help="Monthly retail sales growth.")

    # Metric Selectbox
    st.subheader("Select Economic Metric")
    metric_options = ["Inflation (CPI)", "Interest Rates (Federal Funds Rate)", "Unemployment Rate", "Retail Sales"]
    selected_metric = st.selectbox("Choose a metric", metric_options, index=3)  # Default to Retail Sales
    # Year Selection
    year_selected = st.slider("Select Year", min_value=2010, max_value=2025, value=2024)
    # Time Granularity Selection
    time_granularity = st.radio("Select Time Period", ["Monthly", "Yearly"], horizontal=True)

    # Fetch FRED Data
    @st.cache_data
    def fetch_fred_data(series_id, start_date, end_date):
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # Ensure 'value' is numeric
            return df
        else:
            st.error("Failed to fetch data from FRED API.")
            return pd.DataFrame()

    # Define FRED Series IDs
    fred_series = {
        "Inflation (CPI)": "CPIAUCSL",
        "Interest Rates (Federal Funds Rate)": "FEDFUNDS",
        "Unemployment Rate": "UNRATE",
        "Retail Sales": "RSXFS"
    }

    # Date Range for FRED Data
    start_date = f"{year_selected}-01-01"
    end_date = f"{year_selected}-12-31"

    # Fetch and Display Economic Data
    series_id = fred_series[selected_metric]
    fred_df = fetch_fred_data(series_id, start_date, end_date)
    if not fred_df.empty:
        st.subheader(f"{selected_metric} Over Time")
        fig = px.line(fred_df, x="date", y="value", title=f"{selected_metric} Over Time")
        st.plotly_chart(fig)

        # Resample data based on selected time period
        fred_df.set_index("date", inplace=True)
        if time_granularity == "Monthly":
            resampled_df = fred_df.resample("M").mean()
        elif time_granularity == "Yearly":
            resampled_df = fred_df.resample("Y").mean()

        # Display resampled data
        st.write(f"**{time_granularity} {selected_metric}:**")
        st.line_chart(resampled_df)
    else:
        st.warning(f"No data available for {selected_metric}.")

    
