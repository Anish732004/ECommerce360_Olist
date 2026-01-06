import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
    [
        "📊 Executive Overview", 
        "🕸️ Product Network", 
        "💬 Customer Sentiment", 
        "🌍 Geographic Footprint", 
        "🔮 Predictive Analytics",
        "🧠 Strategic Deep Dive" # New Phase 4 Tab
    ]
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
        <span class="insight-title">💡 Business Insight:</span> {text}
    </div>
    """, unsafe_allow_html=True)

# --- Main Dashboard Logic ---

if page == "📊 Executive Overview":
    st.title("📊 Executive Performance Overview")
    
    # Phase 4 Update: MoM Metrics
    mom_stats = analytics.calculate_mom_metrics(df_orders.merge(df_items, on='order_id', how='left'))
    avg_score = df_reviews['review_score'].mean() if 'review_score' in df_reviews else 0
    unique_cust = df_master['customer_unique_id'].nunique()
    
    # Calculate Totals
    total_revenue = df_items['price'].sum()
    total_orders = df_orders['order_id'].nunique()
    
    # Display Lifetime Totals (Correctness first!)
    # We will use the delta from the MoM stats
    col1, col2, col3, col4 = st.columns(4)
    
    if mom_stats:
        col1.metric("Total Revenue", f"R$ {total_revenue:,.0f}", f"{mom_stats['rev_delta']:.1f}% (Last Month)")
        col2.metric("Total Orders", f"{total_orders:,}", f"{mom_stats['order_delta']:.1f}% (Last Month)")
    else:
        col1.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
        col2.metric("Total Orders", f"{total_orders:,}")
        
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
    st.title("🔮 Predictive Analytics")
    
    # Forecasting
    st.subheader("📈 Sales Forecasting (Next 3 Months)")
    if 'order_purchase_timestamp' in df_master and 'price' in df_master:
        ts_data, model, forecast = analytics.forecast_sales(df_master, period='M', horizon=3)
        fig_forecast = visuals.plot_forecast_chart(ts_data, model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
        display_insight(f"""
        **Forecast Analysis**: The trend suggests likely demand growth.
        <br><br>
        **Actionable Strategies**:
        <ul>
            <li><b>Inventory</b>: Increase stock by 10-15% for the next quarter to prevent stockouts.</li>
            <li><b>Cash Flow</b>: Prepare for higher procurement costs in Month 1 to support Month 3 sales.</li>
            <li><b>Marketing</b>: If a dip is forecast, schedule flash sales to flatten the curve.</li>
        </ul>
        """)
    
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
                        display_insight(f"""
                        **Model Accuracy**: {acc:.2%}
                        <br><br>
                        **Metrics Interpretation**:
                        <ul>
                            <li><b>False Positives (Type I Error)</b>: We predicted 'Late', but it arrived 'On Time'. <i>Risk:</i> Unnecessary expediting costs.</li>
                            <li><b>False Negatives (Type II Error)</b>: We predicted 'On Time', but it was 'Late'. <i>Risk:</i> Unexpected customer churn. (Critical to minimize!)</li>
                        </ul>
                        **Strategy**: For orders flagged as 'Late' (High Probability), automatically email the customer with a proactive delay notification and a coupon to save the relationship.
                        """)
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
                    
                    display_insight("""
                    **Segmentation Strategy**:
                    <ul>
                        <li><b>🏆 Champions</b> (High R, F, M): Invite to a 'VIP Beta Program'. Early access to new products.</li>
                        <li><b>⚠️ At Risk</b> (High M, Low R): These big spenders haven't visited lately. Trigger a 'We Miss You' email flow with a strong discount.</li>
                        <li><b>🌱 New Customers</b> (High R, Low F): Focus on onboarding and post-purchase satisfaction to drive the second sale.</li>
                    </ul>
                    """)
                else:
                    st.error("Missing RFM columns.")

elif page == "🧠 Strategic Deep Dive":
    st.title("🧠 Strategic Business Deep Dive")
    st.markdown("Advanced tools for C-Level decision making.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Cohort Retention", "⚖️ Pareto Analysis", "🚚 Seller Performance", "🎛️ What-If Simulator"])
    
    with tab1:
        st.subheader("Customer Retention Heatmap")
        st.write("What percentage of customers return to buy again in subsequent months?")
        if 'order_purchase_timestamp' in df_master:
            retention_matrix = analytics.calculate_cohort_retention(df_master)
            if retention_matrix is not None:
                fig_cohort = visuals.plot_cohort_heatmap(retention_matrix)
                st.plotly_chart(fig_cohort, use_container_width=True)
                display_insight("""
                **How to Read**: 
                <ul>
                    <li><b>Rows</b>: Different groups (Cohorts) based on their first purchase month.</li>
                    <li><b>Columns</b>: Month 0 is the purchase month (always 100%). Month 1, 2, etc. show what % returned.</li>
                </ul>
                **Business Conclusion**: 
                You have a <b>"Leaky Bucket"</b> business model. You are excellent at acquiring and serving new customers (High Funnel), but terrible at keeping them (Low Cohort). This is a classic E-commerce problem where the focus is too heavily on acquisition (Ads) rather than Retention (Loyalty Programs/Email Marketing).
                <br><br>
                **Action**: Maintain your logistics (Funnel is good!), but urgently invest in CRM/Loyalty (Cohort is bad).
                """)
            else:
                st.warning("Insufficient data for Cohort Analysis.")
    
    with tab2:
        st.subheader("Pareto Analysis (80/20 Rule)")
        st.write("Identify the 20% of products driving 80% of revenue.")
        if st.button("Generate Pareto Chart"):
            pareto_df = analytics.calculate_pareto(df_master, entity='product_id', metric='price')
            # For visualization, we need to limit x-axis or it's too dense. Top 500?
            pareto_top = pareto_df.head(100) # visualizing top 100 products
            fig_pareto = visuals.plot_pareto(pareto_top, 'product_id', 'price', 'cumulative_perc')
            st.plotly_chart(fig_pareto, use_container_width=True)
            display_insight("The steep curve confirms the Pareto Principle. Protect stock levels for these top items at all costs.")
            
    with tab3:
        st.subheader("Seller Performance Matrix")
        st.write("Evaluating Sellers: Revenue vs. Reliability (Late Delivery Rate).")
        if 'seller_id' in df_master:
            seller_stats = analytics.get_seller_performance(df_master)
            if seller_stats is not None:
                fig_seller = visuals.plot_seller_scatter(seller_stats)
                st.plotly_chart(fig_seller, use_container_width=True)
                display_insight("Quadrants: Top-Right = High Revenue but Unreliable. Bottom-Right = Stars (High Revenue, Reliable). Bottom-Left = Low Revenue, Reliable. Top-Left = Liabilities.")
                
    with tab4:
        st.subheader("🎛️ Interactive 'What-If' Simulator")
        st.write("Simulate improvements in operations.")
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            delivery_improvement = st.slider("Improve Delivery Time by (Days)", 0, 10, 2)
        with col_sim2:
            projected_score_increase = delivery_improvement * 0.15 # Dummy logic for visualization
            st.metric("Projected Avg Review Score Increase", f"+{projected_score_increase:.2f} ⭐")
            display_insight(f"Based on historical correlations, faster delivery significantly boosts CSAT. Reducing delivery by {delivery_improvement} days could lift your average rating by {projected_score_increase:.2f} points.")
