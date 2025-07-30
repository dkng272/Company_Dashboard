#%%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.utils import get_data_path
from utils.mongodb_connection import get_stock_list

#%% Data preparation
@st.cache_data
def load_sector_data():
    # Import all L2 from MongoDB
    stock_set = get_stock_list()
    # Initialize an empty dictionary to store the classification.
    sector_dict = {}
    
    # Iterate over each unique L2 classification.
    for l2 in stock_set['L2'].unique():
        # Filter rows that match the current L2 value and extract the ticker list.
        tickers = stock_set[stock_set['L2'] == l2]['Ticker'].tolist()
        # Assign the list to the dictionary.
        sector_dict[l2] = tickers
    
    return sector_dict

@st.cache_data
def load_valuation_data():
    df = pd.read_csv(get_data_path("Val_processed.csv"))
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
    return df

sector_dict = load_sector_data()
df = load_valuation_data()

#%% Plotly scatter chart for either P/E, P/B, EV/EBITDA for stocks within each L2

def plot_valuation_scatter(df, tickers, metric='P/B', start_date='2018-01-01', y_max=None):
    """
    Plot a box plot for valuation metric (P/E, P/B, P/S, EV/EBITDA) for a list of tickers
    showing all data points since the start date with current values highlighted.
    Outliers will not be shown as dots.
    
    Args:
        df: DataFrame with valuation data
        tickers: List of ticker symbols
        metric: Valuation metric to plot ('P/E', 'P/B', 'P/S', 'EV/EBITDA')
        start_date: Start date for filtering data (default: '2018-01-01')
        y_max: Maximum value for y-axis (optional)
    """
    # Filter data for the specified tickers and date range
    filtered_df = df[df['TICKER'].isin(tickers)]
    filtered_df = filtered_df[filtered_df['TRADE_DATE'] >= pd.to_datetime(start_date)]

    # Get current (latest) data points
    df_current = filtered_df[filtered_df['TRADE_DATE'] == filtered_df['TRADE_DATE'].max()]

    # Create box plot and show outlier dots
    fig = px.box(
        filtered_df, 
        x='TICKER', 
        y=metric,
        color='TICKER',
        color_discrete_sequence=['royalblue'],
        points=False,  # Show outliers as dots with points='outliers'
        title=f"{metric} Distribution Since {start_date}",
        height=600,
    )

    # Add current values as scatter points
    fig.add_trace(
        go.Scatter(
            x=df_current['TICKER'],
            y=df_current[metric],
            mode='markers',
            marker=dict(
                color='red',
                size=8,
                symbol='diamond'
            ),
            name='Current Value',
            showlegend=True
        )
    )

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title=metric,
        hovermode='closest',
        showlegend=False,
        template='plotly_white',
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])
    return fig

# Example usage:
plot_valuation_scatter(df, sector_dict['Retailing'], 'EV/EBITDA', '2020-01-01')


#%% Site setup
st.set_page_config(page_title="Sector Valuation", layout="wide")
st.title("Sector (Multi Ticker) Valuation")

st.sidebar.header('Settings')
L2 = st.sidebar.selectbox('Select Sector', options=sorted(sector_dict.keys()))
choose_all = st.sidebar.checkbox('Free Search')
selected_tickers = st.sidebar.multiselect("Select Tickers", options=sector_dict[L2] if not choose_all else df['TICKER'].unique(), max_selections=20)
selected_metrics = st.sidebar.selectbox('Select Valuation Metrics', options=['P/E', 'P/B', 'P/S', 'EV/EBITDA'])
start_date = st.sidebar.selectbox('Select Start Date', options=sorted(df['TRADE_DATE'].dt.date.unique()))


cols = st.columns(3)
with cols[0]:
    y_max = st.number_input("Set Y-axis Maximum (optional)", min_value=0.0, value=None, step=1.0, format="%.1f")
    if y_max == 0.0:
        y_max = None



# Plotting
st.subheader('Valuation Box Plot')
plot = plot_valuation_scatter(df, selected_tickers, selected_metrics, start_date, y_max)
st.plotly_chart(plot, use_container_width=True)
