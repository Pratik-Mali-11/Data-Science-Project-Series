import streamlit as st
import pandas as pd

from src.data_loader import load_merged_data
from src.recommender import build_user_product_matrix, recommend_similar_products
from src.clustering import scale_features, apply_pca, perform_kmeans
from src.preprocessing import simulate_baskets, build_transaction_list, get_top_products
from src.apriori_rules import run_apriori

# 🎯 1. App Title
st.set_page_config(page_title="Customer Segmentation + Product Recommender", layout="wide")
st.title("🧠 Smart Customer Recommender System for Blinkit")

# 📥 2. Load Data
@st.cache_data
def load_all():
    df = load_merged_data(
        "data/blinkit_order_items.csv",
        "data/blinkit_orders.csv",
        "data/blinkit_products.csv"
    )
    return df

df = load_all()

# 🎯 Select Customer
st.sidebar.header("Select Customer")
customer_ids = df["customer_id"].unique()
selected_customer = st.sidebar.selectbox("Choose Customer ID", sorted(customer_ids))

# 📊 3. Customer Summary
st.subheader(f"📇 Customer Profile: {selected_customer}")
cust_data = df[df["customer_id"] == selected_customer]
st.dataframe(cust_data[["order_id", "product_name", "order_date"]].sort_values(by="order_date", ascending=False))

# 🔁 4. Recommend Based on Similar Users (Cosine Similarity)
st.subheader("🧠 Products Other Similar Customers Buy")

user_matrix = build_user_product_matrix(df)
similar_users = recommend_similar_products(selected_customer, user_matrix, top_n=3)

# Get their most common products
top_products = []
for uid in similar_users.index:
    user_top = user_matrix.loc[uid].sort_values(ascending=False).head(3)
    top_products.extend(user_top[user_top > 0].index.tolist())

# Remove duplicates
top_products = list(dict.fromkeys(top_products))
st.write("**Recommended Products:**")
st.write(", ".join(top_products[:10]))

# 📦 5. Association Rule-Based Recommendation
st.subheader("📦 Frequently Bought Together (Apriori Rules)")

# Simulate baskets
df_baskets = simulate_baskets(df, n_orders=2)
df_baskets = get_top_products(df_baskets, top_n=100)
transactions = build_transaction_list(df_baskets)

frequent_itemsets, rules = run_apriori(transactions, min_support=0.001)

# Rules relevant to this customer
purchased = cust_data["product_name"].unique().tolist()
relevant_rules = rules[rules["antecedents"].apply(lambda x: any(item in purchased for item in x))]

if not relevant_rules.empty:
    st.write("**Products bought together with your purchases:**")
    top_apriori = relevant_rules.sort_values(by="lift", ascending=False).head(5)
    for i, row in top_apriori.iterrows():
        st.write(f"🔗 If bought: **{', '.join(row['antecedents'])}** → then also buy: **{', '.join(row['consequents'])}**")
else:
    st.info("No strong basket rules found for this customer's products.")

# 🧪 Optional: Show Cluster (Customer Segmentation)
st.subheader("📊 Customer Segmentation (Cluster)")

df_customer_features = user_matrix.copy()
scaled_df, _ = scale_features(df_customer_features, df_customer_features.columns)
pca_df, _ = apply_pca(scaled_df)
clusters, _ = perform_kmeans(pca_df, n_clusters=4)

pca_df["customer_id"] = user_matrix.index
pca_df["cluster"] = clusters
cust_cluster = pca_df[pca_df["customer_id"] == selected_customer]["cluster"].values[0]

st.markdown(f"📍 This customer belongs to **Cluster {cust_cluster}**")

# 🎨 Optional: Visualize cluster
import seaborn as sns
import matplotlib.pyplot as plt

st.write("📉 PCA Cluster View:")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10")
highlight = pca_df[pca_df["customer_id"] == selected_customer]
plt.scatter(highlight["PC1"], highlight["PC2"], color='red', s=100, edgecolor='black', label='Selected')
plt.legend()
st.pyplot(fig)
