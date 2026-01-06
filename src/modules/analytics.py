from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import streamlit as st
from wordcloud import WordCloud

# ... (Previous imports and functions) ...

# --- RFM Analysis ---
def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary scores for customers.
    df: DataFrame containing 'customer_unique_id', 'order_purchase_timestamp', 'price'
    """
    # Ensure datetime
    if 'order_purchase_timestamp' not in df.columns:
        return None
    
    # Snapshot date (latest date in data + 1 day)
    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    # Aggregation
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    })
    
    # Scoring (Quintiles) - simple 1-5 score
    # Note: Olist has many one-time buyers, so Frequency might be skewed.
    labels = [5, 4, 3, 2, 1] 
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=labels).astype(int)
    
    # For Frequency, since many are 1, we might need rank method='first' or just bespoke bins
    # Using rank(percentage) for simplicity or standard qcut with duplicates dropped
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=list(reversed(labels))).astype(int)
    
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=list(reversed(labels))).astype(int)
    
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    # Segment Labels
    def segment_customer(row):
        if row['RFM_Score'] >= 13:
            return 'Champions'
        elif row['RFM_Score'] >= 10:
            return 'Loyal'
        elif row['RFM_Score'] >= 7:
            return 'Potential Loyalists'
        elif row['RFM_Score'] >= 5:
            return 'At Risk'
        else:
            return 'Hibernating'

    rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
    
    return rfm

# --- Forecasting ---
def forecast_sales(df, period='M', horizon=3):
    """
    Simple Exponential Smoothing forecast.
    df: DataFrame with 'order_purchase_timestamp' and 'price'.
    period: 'D', 'W', 'M'
    """
    # Aggregate sales
    ts_data = df.set_index('order_purchase_timestamp').resample(period)['price'].sum()
    
    # Exponential Weighted Moving Average (Simple forecasting)
    # Alpha = smoothing factor
    model = ts_data.ewm(alpha=0.5, adjust=False).mean()
    
    # Future forecast (naive: last known EWMA value projected flat, 
    # OR linear trend if we use Holt linear)
    # Let's do a simple weighted average extension or just return the smoothed line + scalar projection
    last_val = model.iloc[-1]
    
    # Create future dates
    last_date = ts_data.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]
    
    forecast_series = pd.Series([last_val]*horizon, index=future_dates)
    
    return ts_data, model, forecast_series

# --- ML Classification: Late Delivery Prediction ---
def train_late_delivery_model(df):
    """
    Trains a Logistic Regression to predict if an order will be late.
    Returns model, X_test, y_test, y_pred, y_prob for evaluation.
    """
    # Feature Engineering
    # Target: Is Actual Delivery > Estimated Delivery?
    # We need rows where both exist
    data = df.dropna(subset=['order_estimated_delivery_date', 'order_delivered_customer_date']).copy()
    
    data['is_late'] = (data['order_delivered_customer_date'] > data['order_estimated_delivery_date']).astype(int)
    
    # Features:
    # 1. Freight Value
    # 2. Product Weight / Dimensions (if available in master df)
    # 3. Distance (Approximate using seller/customer state? Or text encoded?)
    # For simplicity/speed in demo: Freight Value + Price + Distance Proxy (State match)
    
    # Proxy for distance: Customer State same as Seller State?
    # We need seller info merged. If not in df, we skip.
    # Assuming df is 'all_data' master table from loader.
    
    features = ['freight_value', 'price', 'product_weight_g']
    
    # Drop NAs
    model_data = data[features + ['is_late']].dropna()
    
    if len(model_data) < 100:
        return None # Not enough data
        
    X = model_data[features]
    y = model_data['is_late']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    return clf, X_test, y_test, y_pred, y_prob

def get_analyzer():
    return SentimentIntensityAnalyzer()

def calculate_sentiment_score(text):
    """
    Returns the compound sentiment score for a given text.
    Returns 0.0 if text is not a string.
    """
    if not isinstance(text, str):
        return 0.0
    analyzer = get_analyzer()
    return analyzer.polarity_scores(text)['compound']

def get_sentiment_distribution(df, column='review_comment_message'):
    pass 

def generate_wordcloud(text_series):
    """
    Generates a WordCloud object from a series of text.
    """
    text = " ".join(review for review in text_series.dropna() if isinstance(review, str))
    if not text:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def build_product_network(df, min_cooccurrence=5):
    """
    Builds a NetworkX graph where nodes are products and edges are co-purchases.
    """
    order_groups = df.groupby('order_id')['product_category_name_english'].unique()
    order_groups = order_groups[order_groups.apply(len) > 1]
    
    G = nx.Graph()
    cooccurrences = Counter()
    
    for products in order_groups:
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                source = products[i]
                target = products[j]
                if source and target: 
                    pair = tuple(sorted((source, target)))
                    cooccurrences[pair] += 1
                    
    for (source, target), weight in cooccurrences.items():
        if weight >= min_cooccurrence:
            G.add_edge(source, target, weight=weight)
            
    return G

def get_funnel_stats(all_data):
    """
    Computes counts for funnel steps.
    """
    step1 = len(all_data)
    step2 = all_data['order_approved_at'].notna().sum()
    step3 = all_data['order_delivered_carrier_date'].notna().sum()
    step4 = len(all_data[all_data['order_status'] == 'delivered'])
    
    return {
        "Total Orders": step1,
        "Approved": step2,
        "In Transit": step3,
        "Delivered": step4
    }
