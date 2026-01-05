import pandas as pd
import streamlit as st
import os

# Constants for file paths
DATA_PATH = "data"

REQUIRED_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv"
}

@st.cache_data(ttl="2h")
def load_data():
    """
    Loads all Olist datasets, merges key tables for a Customer 360 view,
    and returns a dictionary of DataFrames.
    """
    data = {}
    
    # 1. Load raw datasets
    for key, filename in REQUIRED_FILES.items():
        file_path = os.path.join(DATA_PATH, filename)
        if not os.path.exists(file_path):
            st.error(f"Missing file: {filename}. Please place it in the 'data/' directory.")
            return None
        
        try:
            # Optimize types on load where possible to save memory could be added here
            df = pd.read_csv(file_path)
            data[key] = df
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    # 2. Date conversions
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    
    for col in date_cols:
        if col in data['orders'].columns:
            data['orders'][col] = pd.to_datetime(data['orders'][col], errors='coerce')

    if 'review_creation_date' in data['reviews'].columns:
        data['reviews']['review_creation_date'] = pd.to_datetime(data['reviews']['review_creation_date'])

    # 3. Merging for Customer 360 / Master Table
    # Merge Orders + Items
    orders_items = pd.merge(
        data['orders'], 
        data['items'], 
        on='order_id', 
        how='left'
    )

    # Merge + Products (for category names)
    orders_items_products = pd.merge(
        orders_items,
        data['products'],
        on='product_id',
        how='left'
    )
    
    # Merge + Category Translation (English names)
    if 'product_category_name' in orders_items_products.columns:
        orders_items_products = pd.merge(
            orders_items_products,
            data['category_translation'],
            on='product_category_name',
            how='left'
        )
        # Fill missing English names with original portuguese or 'Unknown'
        orders_items_products['product_category_name_english'].fillna(
            orders_items_products['product_category_name'], 
            inplace=True
        )

    # Merge + Customers (for location/id)
    master_df = pd.merge(
        orders_items_products,
        data['customers'],
        on='customer_id',
        how='left'
    )

    # Merge + Reviews (Note: One order can have multiple reviews, likely 1:1 mostly but can be 1:N)
    # We will aggregate reviews to ensure we don't explode the master_df rows too much if we want one-row-per-item
    # For simplicity, we can do a left join.
    master_df = pd.merge(
        master_df,
        data['reviews'][['order_id', 'review_score', 'review_comment_message', 'review_creation_date']],
        on='order_id',
        how='left'
    )
    
    # Store the master dataframe
    data['all_data'] = master_df
    
    return data

def get_date_range(df):
    min_date = df['order_purchase_timestamp'].min()
    max_date = df['order_purchase_timestamp'].max()
    return min_date, max_date
