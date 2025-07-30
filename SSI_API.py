#%% Data pull from SSI
import requests
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st 


#%% Helper functions
def fetch_historical_price(ticker: str, start_date: str = None) -> pd.DataFrame:
    """Fetch stock historical price and volume data from TCBS API"""
    
    # TCBS API endpoint for historical data
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    
    # Convert start_date string to timestamp if provided
    start_timestamp = str(int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()))
    # Parameters for stock data
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",  # Daily data
        "from": start_timestamp,
        "to": str(int(datetime.now().timestamp()))
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data['data'])
            
            # Convert timestamp to datetime
            if 'tradingDate' in df.columns:
                # Check if tradingDate is already in ISO format
                if df['tradingDate'].dtype == 'object' and df['tradingDate'].str.contains('T').any():
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'])
                else:
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms')
            
            # Select relevant columns
            columns_to_keep = ['tradingDate', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            return df
        else:
            print("No data found in response")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def plot_ohlcv_candlestick(df, symbol, start_date = '2024-12-31'):
    df_temp = df[df['tradingDate'] >= start_date].copy()
    df_temp['tradingDate'] = df_temp['tradingDate'].dt.strftime('%Y-%m-%d')
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{symbol} Price Chart", "Volume"]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_temp['tradingDate'],
            open=df_temp['open'],
            high=df_temp['high'],
            low=df_temp['low'],
            close=df_temp['close'],
            name='OHLC',
            opacity=1.0
        ), row=1, col=1
    )
    # Color volume bars by up/down
    colors = ['green' if c >= o else 'red' for c, o in zip(df_temp['close'], df_temp['open'])]
    fig.add_trace(
        go.Bar(
            x=df_temp['tradingDate'],
            y=df_temp['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.8
        ), row=2, col=1
    )
    # Layout
    fig.update_layout(
        template='plotly_white',
        title=f"{symbol} Price Chart",
        xaxis2_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,  
        xaxis2_rangeslider_visible=False,
        height=600,
        showlegend=False
    )
    fig.update_xaxes(
        showgrid=False,
        type='category',
    )

    return fig 

#%% Putting it together
ytd = datetime(datetime.today().year, 1, 1)

def load_ticker_price(ticker, start_date):
    """
    Load OHLCV data for a specific ticker.
    """
    df = fetch_historical_price(ticker, start_date)
    fig = plot_ohlcv_candlestick(df, ticker, start_date)
    return fig


#%% Streamlit

# st.header("Stock Price Dashboard")
# ticker = st.text_input("Enter ticker symbol (e.g., 'VNINDEX')", value='VNINDEX')
# start_date = st.date_input("Start Date (Default: YTD)", value=ytd)
# if st.button("Load Data"):
#     try:
#         fig = load_ticker_price(ticker, start_date=start_date.strftime('%Y-%m-%d'))
#         st.plotly_chart(fig)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")