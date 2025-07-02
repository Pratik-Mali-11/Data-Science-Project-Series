import pandas as pd

def simulate_baskets(df, n_orders=2):
    df = df.copy()
    df = df.sort_values(by=['customer_id', 'order_date'])
    df["order_rank"] = df.groupby("customer_id")["order_date"].rank(method="first").astype(int)
    df["basket_id"] = df["customer_id"].astype(str) + "_basket_" + ((df["order_rank"] - 1) // n_orders + 1).astype(str)
    return df

def get_top_products(df, top_n=50):
    return df[df["product_name"].isin(df["product_name"].value_counts().head(top_n).index)]

def build_transaction_list(df):
    basket_groups = df.groupby("basket_id")["product_name"].apply(list)
    return basket_groups.tolist()
