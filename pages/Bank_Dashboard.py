#%%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import get_data_path
from utils.mongodb_connection import get_bank_keycodes, get_bank_classification

#%% Load bank data
@st.cache_data
def load_bank_data():
    bank = pd.read_csv(get_data_path("df_q_full.csv"))
    bank_formatted = pd.read_csv(get_data_path("df_q_full_formatted.csv"))
    bank['DATE'] = bank['YEARREPORT'].astype(str) + 'Q' + bank['LENGTHREPORT'].astype(str)
    bank_formatted['DATE'] = bank_formatted['YEARREPORT'].astype(str) + 'Q' + bank_formatted['LENGTHREPORT'].astype(str)
    return bank, bank_formatted

@st.cache_data
def load_mapping_and_classification():
    # Load keycode mapping from MongoDB
    mapping = get_bank_keycodes()
    mapping = mapping[~(mapping['DWHCode'].isna())]
    mapping = mapping[['DWHCode', 'KeyCode','Name','Format']]

    ca_format = mapping[mapping['KeyCode'].str.startswith('CA.')][['KeyCode','Format']]
    ca_pct = ca_format[ca_format['Format'] == 'pct']['KeyCode'].tolist()

    keycode_to_name_dict = mapping.set_index('KeyCode')['Name'].to_dict()
    keycode_to_name_dict.pop("Dividend") # Remove Dividend as it is not used in the dashboard
    name_to_keycode_dict = {v: k for k, v in keycode_to_name_dict.items()}

    # Load ticker classification from MongoDB
    classification = get_bank_classification()
    classification['GROUP'] = classification['GROUP'].astype(str)
    
    return ca_pct, keycode_to_name_dict, name_to_keycode_dict, classification

# Load cached data
bank, bank_formatted = load_bank_data()
ca_pct, keycode_to_name_dict, name_to_keycode_dict, classification = load_mapping_and_classification()

#%% Functions for single bank data table
def single_ticker(df, ticker):
    """
    Extract data for a single ticker in quarterly format
    """
    df_ticker = df[df['TICKER'] == ticker].copy()
    return df_ticker

def single_income_statement(df, startperiod=2022):
    """
    IS.3 - Net Interest Income
    IS.1 - Interest Income
    IS.2 - Interest Expense
    IS.6 - Net Fees Income
    IS.14 - Total Operating Income
    IS.15 - G&A Expense
    IS.16 - PPOP
    IS.17 - Provisions for credit losses
    IS.18 - PBT
    IS.24 - NPATMI
    """
    cols = ['DATE', 'IS.3', 'IS.1', 'IS.2', 'IS.6', 'IS.14', 'IS.15', 'IS.16', 'IS.17', 'IS.18', 'IS.24']
    df_is = df[df['YEARREPORT'] >= startperiod][cols].copy()

    df_melted = df_is.melt(id_vars='DATE', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='DATE', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)
    return df_pivoted

def single_size(df, startperiod = 2022):
    """
    BS.1 - Total Assets
    CA.16 - Total Credit
    BS.13 - Total Loans
    Nt.97 - Total Government Bonds
    BS.56 - Total Deposits
    BS.65 - Total Equity
    """
    cols = ['DATE', 'BS.1','CA.16','BS.13','Nt.97','BS.56','BS.65']
    df_size = df[df['YEARREPORT'] >= startperiod][cols].copy()

    df_melted = df_size.melt(id_vars='DATE', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='DATE', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)    
    return df_pivoted

def single_earnings_quality(df, startperiod = 2022):
    """
    CA.25 - Average Asset Yield
    CA.35 - Average Loan Yield
    CA.38 - Average Bond Yield
    CA.41 - Average Deposit Yield
    CA.26 - Average Funding Cost
    CA.44 - Cost of Funding from Deposit
    CA.47 - Cost of Funding from Loan
    CA.49 - Cost of Funding from Valuable Paper
    CA.27 - NIM
    CA.28 - Provision to PPOP
    CA.14 - CIR    
    """
    cols = ['DATE', 'CA.25', 'CA.35', 'CA.38', 'CA.41', 'CA.26', 'CA.44', 'CA.47', 'CA.49', 'CA.27', 'CA.28', 'CA.14']
    df_eq = df[df['YEARREPORT'] >= startperiod][cols].copy()

    df_melted = df_eq.melt(id_vars='DATE', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='DATE', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)
    return df_pivoted

def single_asset_quality(df, startperiod = 2022):
    """
    CA.5 - NPL %
    CA.13 - NPL Formation %
    CA.6 - Group 5 %
    CA.10 - G2 Formation %
    CA.15 - LLR
    """
    cols = ['DATE', 'CA.5', 'CA.13', 'CA.6', 'CA.10', 'CA.15']
    df_asset_quality = df[df['YEARREPORT'] >= startperiod][cols].copy()

    df_melted = df_asset_quality.melt(id_vars='DATE', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='DATE', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)
    
    return df_pivoted

def plot(df):
    df_temp = df.copy()
    row = df_temp.shape[0] // 2 + 1
    fig = make_subplots(rows=row, cols=2, subplot_titles=df_temp.index.tolist(), vertical_spacing=0.05)

    for i, metric in enumerate(df_temp.index):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Bar(x=df_temp.columns, y=df_temp.loc[metric], name=metric), row=row, col=col)

    fig.update_layout(
        height=400 * row, width=1200, 
        title_text="Asset Quality Metrics", 
        showlegend=False,
        # template='simple_white',
    )
    return fig

#%% Functions for multiple banks data table
def income_statement_multi(df, tickers, period = '2025Q1'):
    """
    IS.3 - Net Interest Income
    IS.1 - Interest Income
    IS.2 - Interest Expense
    IS.6 - Net Fees Income
    IS.14 - Total Operating Income
    IS.15 - G&A Expense
    IS.16 - PPOP
    IS.17 - Provisions for credit losses
    IS.18 - PBT
    IS.24 - NPATMI
    """
    cols = ['TICKER', 'IS.3', 'IS.1', 'IS.2', 'IS.6', 'IS.14', 'IS.15', 'IS.16', 'IS.17', 'IS.18', 'IS.24']
    df_is = df[(df['DATE'] == period) & (df['TICKER'].isin(tickers))][cols].copy()

    df_melted = df_is.melt(id_vars='TICKER', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot_table(index='Metric', columns='TICKER', values='Value', aggfunc='first')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)
    
    return df_pivoted

def size_multi(df, tickers, period = '2025Q1'):
    """
    BS.1 - Total Assets
    CA.16 - Total Credit
    BS.13 - Total Loans
    Nt.97 - Total Government Bonds
    BS.56 - Total Deposits
    BS.65 - Total Equity
    """
    cols = ['TICKER','BS.1','CA.16','BS.13','Nt.97','BS.56','BS.65']
    df_size = df[(df['DATE'] == period) & (df['TICKER'].isin(tickers))][cols].copy()

    df_melted = df_size.melt(id_vars='TICKER', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='TICKER', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)
    return df_pivoted

def earnings_quality_multi(df, tickers , period = '2025Q1'):
    """
    CA.25 - Average Asset Yield
    CA.35 - Average Loan Yield
    CA.38 - Average Bond Yield
    CA.41 - Average Deposit Yield
    CA.26 - Average Funding Cost
    CA.44 - Cost of Funding from Deposit
    CA.47 - Cost of Funding from Loan
    CA.49 - Cost of Funding from Valuable Paper
    CA.27 - NIM
    CA.28 - Provision to PPOP
    CA.14 - CIR    
    """
    cols = ['TICKER', 'CA.25', 'CA.35', 'CA.38', 'CA.41', 'CA.26', 'CA.44', 'CA.47', 'CA.49', 'CA.27', 'CA.28', 'CA.14']
    df_eq = df[(df['DATE'] == period) & (df['TICKER'].isin(tickers))][cols].copy()
        
    df_melted = df_eq.melt(id_vars='TICKER', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='TICKER', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)
    
    return df_pivoted 

def asset_quality_multi(df, tickers, period = '2025Q1'):
    """
    CA.5 - NPL %
    CA.13 - NPL Formation %
    CA.6 - Group 5 %
    CA.10 - G2 Formation %
    CA.15 - LLR
    """
    cols = ['TICKER', 'CA.5', 'CA.13', 'CA.6', 'CA.10', 'CA.15']
    df_asset_quality = df[(df['DATE'] == period) & (df['TICKER'].isin(tickers))][cols].copy()

    df_melted = df_asset_quality.melt(id_vars='TICKER', var_name='Metric', value_name='Value')
    df_pivoted = df_melted.pivot(index='Metric', columns='TICKER', values='Value')
    df_pivoted = df_pivoted.reindex(index=cols[1:])
    df_pivoted = df_pivoted.rename(index=keycode_to_name_dict)

    return df_pivoted

#%% Free plotting function
def visualize_multi_ticker_data(df, tickers, keycodes, startperiod=2021):
    """
    Visualize data for multiple tickers over time on the same chart.
    keycodes: list of keycodes to plot, each in its own subplot.
    If one keycode, use 1 column; else use 2 columns per row.
    """
    if not keycodes or not tickers:
        return go.Figure()  # Return empty figure if nothing selected

    if isinstance(keycodes, str):
        keycodes = [keycodes]
    n = len(keycodes)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols

    # Assign a color to each ticker
    import plotly.colors
    palette = plotly.colors.qualitative.Plotly
    ticker_colors = {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}

    fig = make_subplots(
        rows=nrows, cols=ncols, shared_xaxes=True,
        subplot_titles=[keycode_to_name_dict.get(k, k) for k in keycodes],
        vertical_spacing=0.07
    )
    for idx, keycode in enumerate(keycodes):
        row = idx // ncols + 1
        col = idx % ncols + 1
        for t_idx, ticker in enumerate(tickers):
            df_ticker = df[(df['TICKER'] == ticker) & (df['YEARREPORT'] >= startperiod)].copy()
            df_ticker = df_ticker.sort_values(['YEARREPORT', 'LENGTHREPORT'])
            if keycode in ca_pct:
                df_ticker[keycode] = df_ticker[keycode] * 100
            if keycode in df_ticker.columns and not df_ticker.empty:
                # Only show legend for first subplot
                showlegend = (idx == 0)
                fig.add_trace(
                    go.Scatter(
                        x=df_ticker['DATE'],
                        y=pd.to_numeric(df_ticker[keycode], errors='coerce'),
                        name=ticker,
                        mode='lines+markers',
                        marker=dict(color=ticker_colors[ticker]),
                        line=dict(color=ticker_colors[ticker]),
                        showlegend=showlegend,
                    ),
                    row=row, col=col
                )
    fig.update_layout(
        title="Multi Ticker Data",
        width=1200,
        height=500 * nrows,
        template="plotly_white",
        barmode='group',
        xaxis_showgrid=False,
        showlegend=True
    )
    # Set y-axis titles for each subplot
    for idx, keycode in enumerate(keycodes):
        row = idx // ncols + 1
        col = idx % ncols + 1
        fig.update_yaxes(title_text=keycode_to_name_dict.get(keycode, keycode), row=row, col=col)
        if keycode in ca_pct:
            fig.update_yaxes(ticksuffix="%", tickformat=".2f", row=row, col=col)
        else:
            fig.update_yaxes(tickformat="~s", row=row, col=col)
    return fig

#%% Streamlit App Design
st.set_page_config(layout = 'wide', page_title="Banking Dashboard")

st.title("Banking Dashboard")
latest_date = str(bank['PERIOD_INDEX'].max())
formatted = f"{latest_date[:4]}Q{latest_date[-1]}"
st.write(f"Data last updated: {formatted}")

# Selection for single bank analysis
st.sidebar.header('Ticker Selection')
st.sidebar.write("Type in Industry, SOCB, 1, 2, 3 to view as one FS")
selected_ticker = st.sidebar.selectbox("Select Ticker", bank['TICKER'].unique())
list_periods = pd.Series(bank['YEARREPORT'].unique())
selected_start = st.sidebar.selectbox("Select Start Period", list_periods, index=4)

# Selection for multi-bank analysis
st.sidebar.header('Multi-Bank Selection')
selected_group = st.sidebar.selectbox("Select Group", sorted(classification['GROUP'].unique()))
selected_tickers = classification[classification['GROUP'] == selected_group]['TICKER'].unique()
selected_period = st.sidebar.selectbox("Select Period", sorted(bank['DATE'].unique(), reverse=True), index=0)

# Single-bank tables
IS = single_income_statement(single_ticker(bank_formatted, selected_ticker), startperiod=selected_start)
SIZE = single_size(single_ticker(bank_formatted, selected_ticker), startperiod=selected_start)
EARNINGS_QUALITY = single_earnings_quality(single_ticker(bank_formatted, selected_ticker), startperiod=selected_start)
ASSET_QUALITY = single_asset_quality(single_ticker(bank_formatted, selected_ticker), startperiod=selected_start)

# Multi-bank tables
IS_MULTI = income_statement_multi(bank_formatted, tickers=selected_tickers, period=selected_period)
SIZE_MULTI = size_multi(bank_formatted, tickers=selected_tickers, period=selected_period)
EARNINGS_QUALITY_MULTI = earnings_quality_multi(bank_formatted, tickers=selected_tickers, period=selected_period)
ASSET_QUALITY_MULTI = asset_quality_multi(bank_formatted, tickers=selected_tickers, period=selected_period)

# Plots
IS_PLOT = plot(IS)
ASSET_QUALITY_PLOT = plot(ASSET_QUALITY)
EARNINGS_QUALITY_PLOT = plot(EARNINGS_QUALITY)
SIZE_PLOT = plot(SIZE)

# Multi-bank plot
IS_MULTI_PLOT = plot(IS_MULTI)
EARNINGS_QUALITY_MULTI_PLOT = plot(EARNINGS_QUALITY_MULTI)
ASSET_QUALITY_MULTI_PLOT = plot(ASSET_QUALITY_MULTI)
SIZE_MULTI_PLOT = plot(SIZE_MULTI)

# Display tabs
tab11, tab21, tab31 = st.tabs(["Single Bank", "Multi-Bank",'Charting'])

with tab11:
    st.subheader(f"Single Bank: {selected_ticker}")
    tab1, tab2, tab3, tab4 = st.tabs(["Income Statement", "Sizes", "Earnings Quality", "Asset Quality"])
    with tab1:
        st.dataframe(IS)
        st.plotly_chart(IS_PLOT)
    with tab2:
        st.dataframe(SIZE)
        st.plotly_chart(SIZE_PLOT)
    with tab3:
        st.dataframe(EARNINGS_QUALITY)
        st.plotly_chart(EARNINGS_QUALITY_PLOT)
    with tab4:
        st.dataframe(ASSET_QUALITY)
        st.plotly_chart(ASSET_QUALITY_PLOT)

with tab21:
    st.subheader(f"Multi-Bank for Group {selected_group}")
    tab1, tab2, tab3, tab4 = st.tabs(["Income Statement", "Sizes", "Earnings Quality", "Asset Quality"])
    with tab1:
        if IS_MULTI.empty:
            st.warning("No data available for selected tickers and period.")
        else:
            st.dataframe(IS_MULTI)
            st.plotly_chart(IS_MULTI_PLOT)
    with tab2:
        if SIZE_MULTI.empty:
            st.warning("No data available for selected tickers and period.")
        else:
            st.dataframe(SIZE_MULTI)
            st.plotly_chart(SIZE_MULTI_PLOT)
    with tab3:
        if EARNINGS_QUALITY_MULTI.empty:
            st.warning("No data available for selected tickers and period.")
        else:
            st.dataframe(EARNINGS_QUALITY_MULTI)
            st.plotly_chart(EARNINGS_QUALITY_MULTI_PLOT)
    with tab4:
        if ASSET_QUALITY_MULTI.empty:
            st.warning("No data available for selected tickers and period.")
        else:
            st.dataframe(ASSET_QUALITY_MULTI)
            st.plotly_chart(ASSET_QUALITY_MULTI_PLOT)


list = list(name_to_keycode_dict.keys())

with tab31:
    st.subheader("Charting for multi tickers")
    st.write('You can also select SOCB, Industry, 1, 2, 3 to view')
    chart_tickers = st.multiselect("Select Ticker", bank_formatted['TICKER'].unique(), key='chart_ticker')
    selected_meanings = st.multiselect("Select KeyCode", options=list)
    starting_period = st.selectbox('Select Starting Period', options=(bank_formatted['YEARREPORT'].unique()), index=4)
    selected_keycodes = [name_to_keycode_dict[m] for m in selected_meanings]
    CHART = visualize_multi_ticker_data(
        bank,
        tickers=chart_tickers,
        keycodes=selected_keycodes,
        startperiod=starting_period
    )
    st.plotly_chart(CHART)
