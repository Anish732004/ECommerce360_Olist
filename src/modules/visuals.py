import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_funnel(stats_dict):
    data = dict(number=list(stats_dict.values()), stage=list(stats_dict.keys()))
    fig = px.funnel(data, x='number', y='stage')
    fig.update_layout(title="Order Fulfillment Funnel")
    return fig

def plot_review_sentiment_distribution(df):
    if 'review_score' not in df.columns: return None
    counts = df['review_score'].value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Review Score', 'y': 'Count'},
                 title="Distribution of Review Scores", color=counts.values, color_continuous_scale='Bluered')
    return fig

def plot_network(G):
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
        hoverinfo='none', mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_adjacencies = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_adjacencies.append(len(G.adj[node]))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(
            showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=10,
            colorbar=dict(thickness=15, title='Connections', xanchor='left'),
            line_width=2),
        text=node_text
    )
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text='Product Category Co-Purchase Network', font=dict(size=16)),
                showlegend=False, hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

def plot_radar_chart(categories, values, title="Radar Chart"):
    fig = px.line_polar(r=values, theta=categories, line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(title=title)
    return fig

def display_wordcloud(wordcloud_obj):
    if wordcloud_obj is None:
        st.info("Not enough data for word cloud.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_obj, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_dual_axis_trend(df, date_col, metric1, metric2, title="Trend Analysis"):
    """
    Plots a dual-axis chart (Bar + Line) for two metrics over time.
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df[date_col], y=df[metric1], name=metric1, marker_color='indigo', opacity=0.6),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df[date_col], y=df[metric2], name=metric2, mode='lines+markers', line=dict(color='teal')),
        secondary_y=True,
    )

    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=metric1, secondary_y=False)
    fig.update_yaxes(title_text=metric2, secondary_y=True)

    return fig

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig

def plot_confusion_matrix(y_test, y_pred, labels=[0, 1]):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix",
                    x=[str(l) for l in labels],
                    y=[str(l) for l in labels]
                   )
    return fig

def plot_forecast_chart(historical_data, smoothed_data, forecast_data):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data.index, y=historical_data.values,
        mode='lines', name='Actual Sales'
    ))
    
    fig.add_trace(go.Scatter(
        x=smoothed_data.index, y=smoothed_data.values,
        mode='lines', name='Trend (Smoothed)', line=dict(dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data.index, y=forecast_data.values,
        mode='lines', name='Forecast (Next 3 Months)', line=dict(color='green', width=3)
    ))
    
    fig.update_layout(title="Sales Forecast (Exponential Smoothing)")
    return fig

def plot_map_scatter(df):
    """
    Plots a geographical scatter map of customers using Plotly Mapbox.
    df: DataFrame with 'geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state', 'count'
    """
    # Scatter Mapbox
    fig = px.scatter_mapbox(
        df, 
        lat="geolocation_lat", 
        lon="geolocation_lng", 
        size="count",
        color="count",
        hover_name="geolocation_city",
        hover_data=["geolocation_state", "count"],
        color_continuous_scale=px.colors.sequential.Bluered,
        size_max=15, 
        zoom=3,
        mapbox_style="carto-positron"
    )
    fig.update_layout(title="Geographic Distribution of Customers")
    return fig
