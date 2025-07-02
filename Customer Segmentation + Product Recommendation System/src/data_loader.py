import pandas as pd

def load_merged_data(order_items_path, orders_path, products_path):
    order_items = pd.read_csv(order_items_path)
    orders = pd.read_csv(orders_path)
    products = pd.read_csv(products_path)

    df = pd.merge(order_items, orders, on='order_id', how='left')
    df = pd.merge(df, products, on='product_id', how='left')
    df['order_date'] = pd.to_datetime(df['order_date'])

    return df
