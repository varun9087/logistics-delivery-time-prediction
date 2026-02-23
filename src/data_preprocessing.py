import pandas as pd

def load_and_merge_data():
    orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
    order_items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
    customers = pd.read_csv("data/raw/olist_customers_dataset.csv")

    df = orders.merge(order_items, on="order_id")
    df = df.merge(customers, on="customer_id")

    return df