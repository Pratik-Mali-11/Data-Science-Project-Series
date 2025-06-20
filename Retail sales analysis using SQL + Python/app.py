import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# PostgreSQL connection
engine = create_engine('postgresql://postgres:pass123@localhost:5432/retail_sales')

st.set_page_config(layout="wide")
st.title("📦 Retail Sales Analysis Dashboard")

# Optional raw data preview
if st.checkbox("Show raw data"):
    df = pd.read_sql("SELECT * FROM orders", engine)
    st.write(df.head())

# KPIs
st.subheader("📈 Key Metrics")
kpi_query = """
SELECT 
    SUM(sales) AS total_sales, 
    SUM(profit) AS total_profit, 
    COUNT(DISTINCT order_id) AS total_orders 
FROM orders;
"""
kpi = pd.read_sql(kpi_query, engine)
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${kpi['total_sales'][0]:,.2f}")
col2.metric("Total Profit", f"${kpi['total_profit'][0]:,.2f}")
col3.metric("Total Orders", kpi['total_orders'][0])

# Region-wise Sales
st.subheader("🌍 Region-wise Sales")
region_df = pd.read_sql("SELECT region, SUM(sales) AS total_sales FROM orders GROUP BY region ORDER BY total_sales DESC", engine)
st.bar_chart(region_df.set_index("region"))

# Category-wise Sales
st.subheader("🗃️ Category-wise Sales")
cat_df = pd.read_sql("SELECT category, SUM(sales) AS total_sales FROM orders GROUP BY category", engine)
st.bar_chart(cat_df.set_index("category"))

# Sub-Category Sales
st.subheader("🛍️ Sub-Category Sales")
subcat_df = pd.read_sql("SELECT sub_category, SUM(sales) AS total_sales FROM orders GROUP BY sub_category ORDER BY total_sales DESC", engine)
st.bar_chart(subcat_df.set_index("sub_category"))

# Year-wise Sales Trend
st.subheader("📅 Year-wise Sales Trend")
year_df = pd.read_sql("""
    SELECT EXTRACT(YEAR FROM order_date::DATE) AS order_year,
           SUM(sales) AS total_sales
    FROM orders
    GROUP BY EXTRACT(YEAR FROM order_date::DATE)
    ORDER BY order_year
""", engine)
st.line_chart(year_df.set_index("order_year"))

# Top Products
st.subheader("🏆 Top 10 Profitable Products")
top_products = pd.read_sql("""
    SELECT product_name, SUM(profit) AS total_profit
    FROM orders
    GROUP BY product_name
    ORDER BY total_profit DESC
    LIMIT 10
""", engine)
st.bar_chart(top_products.set_index("product_name"))

# Top Customers
st.subheader("👤 Top 10 Customers by Sales")
top_customers = pd.read_sql("""
    SELECT customer_name, SUM(sales) AS total_sales
    FROM orders
    GROUP BY customer_name
    ORDER BY total_sales DESC
    LIMIT 10
""", engine)
st.bar_chart(top_customers.set_index("customer_name"))

# Segment-wise Sales
st.subheader("👥 Sales by Segment")
segment_df = pd.read_sql("SELECT segment, SUM(sales) AS total_sales FROM orders GROUP BY segment", engine)
st.bar_chart(segment_df.set_index("segment"))

# Ship Mode Analysis
st.subheader("🚚 Ship Mode - Sales & Profit")
ship_df = pd.read_sql("""
    SELECT ship_mode, SUM(sales) AS total_sales, SUM(profit) AS total_profit
    FROM orders
    GROUP BY ship_mode
""", engine)
st.dataframe(ship_df)

