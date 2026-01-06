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
    .insight-box {
        background-color: #e0f2fe;
        border-left: 5px solid #0284c7;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .insight-title {
        font_weight: bold;
        color: #0c4a6e;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛍️ E-Commerce 360")
st.sidebar.write("Comprehensive insights into Olist E-Commerce data.")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Overview", "🕸️ Product Network", "💬 Customer Sentiment", "🌍 Geographic Footprint", "🔮 Predictive Analytics"]
)

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

# --- Helper for Insights ---
def display_insight(text):
    st.markdown(f"""
    <div class="insight-box">
        <span class="insight-title">� Business Insight:</span> {text}
    </div>
    """, unsafe_allow_html=True)

# --- Main Dashboard Logic ---

if page == "📊 Executive Overview":
    st.title("📊 Executive Performance Overview")
    
    # metrics
    total_orders = len(df_orders)
    total_revenue = df_items['price'].sum() if 'price' in df_items else 0
    avg_score = df_reviews['review_score'].mean() if 'review_score' in df_reviews else 0
    unique_cust = df_master['customer_unique_id'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Total Revenue", f"R$ {total_revenue:,.2f}")
    col3.metric("Avg Review Score", f"{avg_score:.2f} / 5")
    col4.metric("Unique Customers", f"{unique_cust:,}")
    
    st.markdown("---")
    
    # Row 1: Sales Trend + Funnel
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Sales Trend Over Time")
        if 'order_purchase_timestamp' in df_orders:
            sales_trend = df_orders.set_index('order_purchase_timestamp').resample('M').size()
            fig_trend = px.line(sales_trend, title="Monthly Orders", labels={'value': 'Orders', 'order_purchase_timestamp': 'Date'})
            st.plotly_chart(fig_trend, use_container_width=True)
            display_insight("Consistent growth observed until late 2017. The sharp drop or spike at the end may indicate incomplete data for the last month or a seasonal anomaly.")
            
    with c2:
        st.subheader("Order Funnel")
        funnel_stats = analytics.get_funnel_stats(df_orders)
        fig_funnel = visuals.plot_funnel(funnel_stats)
        st.plotly_chart(fig_funnel, use_container_width=True)
        display_insight(f"High conversion from Approval to Carrier. Delivery success rate is {(funnel_stats['Delivered']/funnel_stats['Total Orders']):.1%}.")

    # Row 2: Top Categories
    st.subheader("Top Product Categories by Revenue")
    category_rev = df_master.groupby('product_category_name_english')['price'].sum().nlargest(10).reset_index()
    fig_cat = px.bar(category_rev, x='price', y='product_category_name_english', orientation='h', title="Top 10 Categories")
    fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cat, use_container_width=True)
    top_cat = category_rev.iloc[0]['product_category_name_english']
    display_insight(f"**{top_cat}** is the primary revenue driver. Focus inventory optimization and marketing efforts here.")

elif page == "🕸️ Product Network":
    st.title("🕸️ Product Co-Purchase Network")
    st.write("Visualize relationships between product categories bought in the same order.")
    
    with st.expander("Filter & Settings", expanded=True):
        min_occur = st.slider("Minimum Co-occurrences", min_value=1, max_value=50, value=5, help="Filter out weak connections")
        
    if df_master is not None:
        with st.spinner("Building network graph..."):
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
                
                display_insight("Strong connections (clusters) indicate cross-selling opportunities. For example, furniture often sells with decoration items—bundle these for higher AOV.")
            else:
                st.warning("No connections found. Lower the minimum occurrences.")

elif page == "💬 Customer Sentiment":
    st.title("💬 Customer Sentiment Analysis")
    
    col_sent1, col_sent2 = st.columns(2)
    
    with col_sent1:
        st.subheader("Review Score Distribution")
        fig_rev = visuals.plot_review_sentiment_distribution(df_reviews)
        if fig_rev:
            st.plotly_chart(fig_rev, use_container_width=True)
            display_insight("The distribution is heavily skewed towards 5-star ratings, indicating generally high customer satisfaction.")
            
    with col_sent2:
        st.subheader("Sentiment vs Rating Alignment")
        if 'review_comment_message' in df_reviews:
            if st.button("Analyze Sentiment on Sample (1000 reviews)"):
                sample_reviews = df_reviews[df_reviews['review_comment_message'].notna()].sample(1000)
                sample_reviews['vader_score'] = sample_reviews['review_comment_message'].apply(analytics.calculate_sentiment_score)
                
                radar_data = sample_reviews.groupby('review_score')['vader_score'].mean().reset_index()
                fig_radar = visuals.plot_radar_chart(
                    categories=radar_data['review_score'].astype(str), 
                    values=radar_data['vader_score'],
                    title="Avg Sentiment Score per Star Rating"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                display_insight("Sentiment scores correlate with star ratings, validating that low-star reviews genuinely contain negative text.")
                
                st.subheader("Word Cloud of Negative Reviews")
                neg_reviews = sample_reviews[sample_reviews['review_score'] < 3]['review_comment_message']
                wc = analytics.generate_wordcloud(neg_reviews)
                visuals.display_wordcloud(wc)
                display_insight("Common negative terms (e.g., 'entrega', 'atraso') often relate to delivery delays.")
            else:
                st.info("Click button to run VADER analysis")

elif page == "🌍 Geographic Footprint":
    st.title("🌍 Geographic Customer Distribution")
    
    if 'geolocation' in data_dict:
        geo_df = data_dict['geolocation']
        
        # Filter Outliers (Brazil Bounding Box Approx)
        # Lat: -34 to +5, Lon: -74 to -34
        geo_df = geo_df[
            (geo_df['geolocation_lat'] >= -34) & (geo_df['geolocation_lat'] <= 5) &
            (geo_df['geolocation_lng'] >= -74) & (geo_df['geolocation_lng'] <= -34)
        ]
        
        geo_agg = geo_df.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state']].first().reset_index()
        
        cust_counts = data_dict['customers']['customer_zip_code_prefix'].value_counts().reset_index()
        cust_counts.columns = ['geolocation_zip_code_prefix', 'count']
        
        map_data = pd.merge(cust_counts, geo_agg, on='geolocation_zip_code_prefix')
        
        st.write(f"Plotting {len(map_data)} locations across Brazil.")
        
        # Visuals scatter map
        fig_map = visuals.plot_map_scatter(map_data)
        st.plotly_chart(fig_map, use_container_width=True)
        
        display_insight("The Southeast region (São Paulo, Rio) dominates annual orders. Logistics hubs should be centered here, but expansion into the South offers growth potential.")
    else:
        st.warning("Geolocation data missing.")

elif page == "🔮 Predictive Analytics":
    st.title("🔮 Predictive Analytics & Deep Dive")
    
    # Forecasting
    st.subheader("📈 Sales Forecasting (Next 3 Months)")
    if 'order_purchase_timestamp' in df_master and 'price' in df_master:
        ts_data, model, forecast = analytics.forecast_sales(df_master, period='M', horizon=3)
        fig_forecast = visuals.plot_forecast_chart(ts_data, model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
        display_insight("Forecast indicates stable growth. Ensure inventory levels for the predicted 3-month horizon to avoid stockouts.")
    
    st.markdown("---")
    
    col_ml1, col_ml2 = st.columns(2)
    
    with col_ml1:
        st.subheader("📦 Late Delivery Prediction (Logistic Regression)")
        if st.button("Train Model & Show Metrics"):
            with st.spinner("Training Model..."):
                if 'product_weight_g' in df_master.columns:
                    ml_results = analytics.train_late_delivery_model(df_master)
                    if ml_results:
                        clf, X_test, y_test, y_pred, y_prob = ml_results
                        
                        st.write("#### Confusion Matrix")
                        fig_cm = visuals.plot_confusion_matrix(y_test, y_pred)
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        st.write("#### ROC Curve")
                        fig_roc = visuals.plot_roc_curve(y_test, y_prob)
                        st.plotly_chart(fig_roc, use_container_width=True)
                        
                        acc = clf.score(X_test, y_test)
                        st.success(f"Model Accuracy: {acc:.2%}")
                        display_insight(f"Model achieves {acc:.2%} accuracy. High False Positives in the matrix would mean we warn customers unnecessarily; High False Negatives mean we miss actual delays.")
                    else:
                        st.warning("Not enough data.")
                else:
                    st.error("Missing weight data.")
    
    with col_ml2:
        st.subheader("👥 RFM Customer Segmentation")
        if st.button("Run RFM Analysis"):
            with st.spinner("Calculating Segments..."):
                rfm_df = analytics.calculate_rfm(df_master)
                if rfm_df is not None:
                    segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
                    segment_counts.columns = ['Segment', 'Count']
                    
                    fig_rfm = px.treemap(segment_counts, path=['Segment'], values='Count', color='Count',
                                       color_continuous_scale='RdBu', title="Customer Segments Distribution")
                    st.plotly_chart(fig_rfm, use_container_width=True)
                    
                    display_insight("Most customers are likely 'One-time Buyers' (Hibernating) or 'New'. A loyalty program is critical to convert them into 'Champions'.")
                else:
                    st.error("Missing RFM columns.")
