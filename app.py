import streamlit as st
import pandas as pd
import plotly.express as px
from src.modules.loader import load_data
from src.modules import analytics, visuals

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Olist 360 Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4F46E5;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛍️ E-Commerce 360")
st.sidebar.write("Comprehensive insights into Olist E-Commerce data.")

st.sidebar.markdown("---")
st.sidebar.info("Loading large datasets can take a moment initially. Subsequent loads will be cached.")

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared!")

# --- Data Loading ---
try:
    with st.spinner("Loading and processing data..."):
        data_dict = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if not data_dict:
    st.warning("Data not found. Please upload Olist CSV files to 'data/' folder.")
    st.stop()

df_master = data_dict['all_data']
df_orders = data_dict['orders']
df_items = data_dict['items']
df_reviews = data_dict['reviews']

# --- Main Dashboard ---

# Tabs
tab_overview, tab_network, tab_sentiment, tab_geo = st.tabs([
    "📊 Business Overview", "🕸️ Product Network", "💬 Sentiment Analysis", "🌍 Geographic"
])

# --- Tab 1: Overview ---
with tab_overview:
    st.header("Business Performance Overview")
    
    # metrics
    total_orders = len(df_orders)
    total_revenue = df_items['price'].sum() if 'price' in df_items else 0
    avg_score = df_reviews['review_score'].mean() if 'review_score' in df_reviews else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Total Revenue", f"R$ {total_revenue:,.2f}")
    col3.metric("Avg Review Score", f"{avg_score:.2f} / 5")
    col4.metric("Unique Customers", f"{df_master['customer_unique_id'].nunique():,}")
    
    st.markdown("---")
    
    # Row 1: Sales Trend + Funnel
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Sales Trend Over Time")
        # Daily sales
        if 'order_purchase_timestamp' in df_orders:
            sales_trend = df_orders.set_index('order_purchase_timestamp').resample('M').size()
            fig_trend = px.line(sales_trend, title="Monthly Orders", labels={'value': 'Orders', 'order_purchase_timestamp': 'Date'})
            st.plotly_chart(fig_trend, use_container_width=True)
            
    with c2:
        st.subheader("Order Funnel")
        funnel_stats = analytics.get_funnel_stats(df_orders)
        fig_funnel = visuals.plot_funnel(funnel_stats)
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Row 2: Top Categories
    st.subheader("Top Product Categories by Revenue")
    category_rev = df_master.groupby('product_category_name_english')['price'].sum().nlargest(10).reset_index()
    fig_cat = px.bar(category_rev, x='price', y='product_category_name_english', orientation='h', title="Top 10 Categories")
    fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cat, use_container_width=True)

# --- Tab 2: Product Network ---
with tab_network:
    st.header("Product Co-Purchase Network")
    st.write("Visualize which product categories are frequently bought together.")
    
    with st.expander("Filter & Settings", expanded=True):
        min_occur = st.slider("Minimum Co-occurrences", min_value=1, max_value=50, value=5, help="Filter out weak connections")
        
    if df_master is not None:
        with st.spinner("Building network graph..."):
            # Needs a subset to be fast?
            # We use the master df which has order_id and category
            # Filter rows with categories
            df_network = df_master[['order_id', 'product_category_name_english']].dropna()
            
            G = analytics.build_product_network(df_network, min_cooccurrence=min_occur)
            
            if G.number_of_nodes() > 0:
                col_graph, col_stats = st.columns([3, 1])
                with col_graph:
                    fig_net = visuals.plot_network(G)
                    st.plotly_chart(fig_net, use_container_width=True)
                with col_stats:
                    st.write("### Network Stats")
                    st.write(f"**Nodes (Categories):** {G.number_of_nodes()}")
                    st.write(f"**Edges (Links):** {G.number_of_edges()}")
                    
                    st.write("### Top Connections")
                    edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
                    for u, v, d in edges[:10]:
                        st.write(f"- {u} & {v}: **{d['weight']}**")
            else:
                st.warning("No connections found with current settings. Try lowering the minimum co-occurrences.")

# --- Tab 3: Sentiment ---
with tab_sentiment:
    st.header("Customer Sentiment Analysis")
    
    col_sent1, col_sent2 = st.columns(2)
    
    with col_sent1:
        st.subheader("Review Score Distribution")
        fig_rev = visuals.plot_review_sentiment_distribution(df_reviews)
        if fig_rev:
            st.plotly_chart(fig_rev, use_container_width=True)
            
    with col_sent2:
        st.subheader("Sentiment vs Rating Radar")
        # Logic: Correlation between average Sentiment Score and specific Categories?
        # Or simply Rating (1-5) features? 
        # Let's do a simple Radar of "Average Sentiment" by "Review Score" to see if they align (Validation)
        # Or better: Radar of Top 5 Categories and their characteristics (Price, Sentiment, Days to Deliver)
        
        # We'll calculate sentiment for a sample to avoid slowness if not pre-computed
        if 'review_comment_message' in df_reviews:
            # check if we computed it? No, let's do small sample for demo speed or on demand
            if st.button("Analyze Sentiment on Sample (1000 reviews)"):
                sample_reviews = df_reviews[df_reviews['review_comment_message'].notna()].sample(1000)
                sample_reviews['vader_score'] = sample_reviews['review_comment_message'].apply(analytics.calculate_sentiment_score)
                
                # Radar: Avg VADER Score per Star Rating
                radar_data = sample_reviews.groupby('review_score')['vader_score'].mean().reset_index()
                fig_radar = visuals.plot_radar_chart(
                    categories=radar_data['review_score'].astype(str), 
                    values=radar_data['vader_score'],
                    title="Avg Sentiment Score per Star Rating"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Word Cloud
                st.subheader("Word Cloud of Negative Reviews (Score < 3)")
                neg_reviews = sample_reviews[sample_reviews['review_score'] < 3]['review_comment_message']
                wc = analytics.generate_wordcloud(neg_reviews)
                visuals.display_wordcloud(wc)
            else:
                st.info("Click button to run VADER analysis")

# --- Tab 4: Geographic ---
with tab_geo:
    st.header("Customer Geolocation")
    # Simple map of customers
    # We need lat/lng from geolocation dataset linked to customers
    # The merge logic in loader might not have brought in lat/lng directly to save memory or complexity
    # 'customers' has 'customer_zip_code_prefix'. 'geolocation' has 'geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng'
    
    if 'geolocation' in data_dict:
        geo_df = data_dict['geolocation']
        # Group by zip to reduce points
        geo_agg = geo_df.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
        
        # Merge with customers to get count per zip
        cust_counts = data_dict['customers']['customer_zip_code_prefix'].value_counts().reset_index()
        cust_counts.columns = ['geolocation_zip_code_prefix', 'count']
        
        map_data = pd.merge(cust_counts, geo_agg, on='geolocation_zip_code_prefix')
        
        # Rename for st.map
        map_data.rename(columns={'geolocation_lat': 'lat', 'geolocation_lng': 'lon'}, inplace=True)
        
        st.write(f"Plotting {len(map_data)} locations.")
        st.map(map_data, size='count', color='#0044ff') # color arg available in newer streamlit
    else:
        st.warning("Geolocation data missing.")
