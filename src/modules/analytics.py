import pandas as pd
import numpy as np
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import streamlit as st
from wordcloud import WordCloud

@st.cache_resource
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
    """
    Calculates positive, neutral, negative sentiment counts.
    """
    # Sample or process all? For performance, maybe limit if dataset is huge.
    # Olist is ~100k rows. VADER is fast enough for 100k, but let's be careful.
    # We will compute a column 'sentiment_score' if it doesn't exist, but usually it's better to do this once.
    
    # We'll expect the dataframe to likely have scores already or we compute them here.
    # For a specialized visual, we might just sample 1000 for speed if not pre-computed.
    pass # logic will be in main flow or helper

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
    df: Dataframe containing 'order_id' and 'product_category_name_english' (or product_id).
    Using Category names is better for readability than IDs.
    """
    # Group by order_id to see what was bought together
    # Filter for orders with > 1 item
    order_groups = df.groupby('order_id')['product_category_name_english'].unique()
    order_groups = order_groups[order_groups.apply(len) > 1]
    
    G = nx.Graph()
    
    # Count co-occurrences
    cooccurrences = Counter()
    
    for products in order_groups:
        # Create all pairs
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                source = products[i]
                target = products[j]
                if source and target: # check for nan
                    # Sorting to ensure consistent keys
                    pair = tuple(sorted((source, target)))
                    cooccurrences[pair] += 1
                    
    # Add columns to graph
    for (source, target), weight in cooccurrences.items():
        if weight >= min_cooccurrence:
            G.add_edge(source, target, weight=weight)
            
    return G

def get_funnel_stats(all_data):
    """
    Computes counts for funnel steps:
    1. Total Orders
    2. Approved
    3. Carrier
    4. Delivered
    """
    # Logic:
    # All orders
    step1 = len(all_data)
    
    # Approved (order_approved_at is not null)
    step2 = all_data['order_approved_at'].notna().sum()
    
    # In Transit / Carrier (order_delivered_carrier_date is not null)
    step3 = all_data['order_delivered_carrier_date'].notna().sum()
    
    # Delivered (order_status == 'delivered')
    step4 = len(all_data[all_data['order_status'] == 'delivered'])
    
    return {
        "Total Orders": step1,
        "Approved": step2,
        "In Transit": step3,
        "Delivered": step4
    }
