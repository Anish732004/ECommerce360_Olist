import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported if needed for type hinting or logic

def plot_funnel(stats_dict):
    """
    Plots a funnel chart from a dictionary of steps and counts.
    """
    data = dict(
        number=list(stats_dict.values()),
        stage=list(stats_dict.keys())
    )
    
    fig = px.funnel(data, x='number', y='stage')
    fig.update_layout(title="Order Fulfillment Funnel")
    return fig

def plot_review_sentiment_distribution(df):
    """
    Plots a bar chart of review scores.
    """
    if 'review_score' not in df.columns:
        return None
        
    counts = df['review_score'].value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Review Score', 'y': 'Count'},
                 title="Distribution of Review Scores", color=counts.values, color_continuous_scale='Bluered')
    return fig

def plot_network(G):
    """
    Plots a NetworkX graph using Plotly.
    """
    # 1. Layout
    pos = nx.spring_layout(G, seed=42)
    
    # 2. Edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 3. Nodes
    node_x = []
    node_y = []
    node_text = []
    node_adjacencies = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        # Count connections
        node_adjacencies.append(len(G.adj[node]))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left',
            ),
            line_width=2),
        text=node_text
    )
    
    node_trace.marker.color = node_adjacencies

    # 4. Figure
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    text='Product Category Co-Purchase Network',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Network Graph of Product Categories bought together",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

def plot_radar_chart(categories, values, title="Radar Chart"):
    """
    Plots a Radar chart for comparing metrics across categories.
    """
    fig = px.line_polar(r=values, theta=categories, line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(title=title)
    return fig

def plot_correlation_heatmap(df, cols=None):
    """
    Plots a correlation heatmap for numerical columns.
    """
    if cols:
        curr_df = df[cols]
    else:
        curr_df = df.select_dtypes(include=[np.number])
        
    corr = curr_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    return fig

def display_wordcloud(wordcloud_obj):
    """
    Display WordCloud in Streamlit (requires matplotlib figure or verify image).
    """
    if wordcloud_obj is None:
        st.info("Not enough data for word cloud.")
        return

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_obj, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
