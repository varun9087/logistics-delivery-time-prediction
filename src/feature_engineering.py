import pandas as pd

def clean_and_engineer(df):
    df = df[[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "price",
        "freight_value",
        "product_id",
        "customer_state"
    ]]

    df = df.dropna()

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])

    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month

    df = df.drop([
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "product_id"
    ], axis=1)

    df = pd.get_dummies(df, drop_first=True)

    return df