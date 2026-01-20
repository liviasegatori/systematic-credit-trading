import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Credit Trading Desk", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = DATA_PATH / "backtest_results.csv"
MARKET_PATH = DATA_PATH / "corporate_universe.csv"

# --- DATA LOADING ---
@st.cache_data
def load_data():
    if not RESULTS_PATH.exists():
        st.error("Backtest results not found. Run src/backtest/engine.py first!")
        return None, None
    
    # Load Backtest Results
    res_df = pd.read_csv(RESULTS_PATH, parse_dates=['date'])
    res_df = res_df.set_index('date')
    
    # Load Market Data (for context)
    market_df = pd.read_csv(MARKET_PATH, parse_dates=['date'])
    
    return res_df, market_df

def calculate_drawdown(nav_series):
    # Calculate High Water Mark
    hwm = nav_series.cummax()
    # Drawdown = (Current NAV - HWM) / HWM
    dd = (nav_series - hwm) / hwm
    return dd

# --- MAIN DASHBOARD ---
def main():
    st.title("🏦 Systematic Credit Trading Dashboard")
    st.markdown("### Performance & Risk Monitor")
    
    results, market = load_data()
    
    if results is None:
        return

    # 1. KPI SECTION
    # Calculate metrics
    start_nav = results['nav'].iloc[0]
    end_nav = results['nav'].iloc[-1]
    total_return = (end_nav / start_nav) - 1
    
    # Annualized Volatility
    daily_ret = results['nav'].pct_change()
    vol = daily_ret.std() * np.sqrt(252)
    
    # Sharpe (assuming 0% risk free for simplicity in this view, or use FRED data)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    
    # Max Drawdown
    drawdown = calculate_drawdown(results['nav'])
    max_dd = drawdown.min()
    
    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_return:.2%}", delta_color="normal")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Annual Volatility", f"{vol:.2%}")
    col4.metric("Max Drawdown", f"{max_dd:.2%}", delta_color="inverse")

    st.markdown("---")

    # 2. CHARTS SECTION
    
    # Row A: Equity Curve and Drawdown
    st.subheader("Strategy Performance")
    
    # Equity Curve
    fig_nav = px.line(results, y='nav', title='Net Asset Value (NAV)')
    fig_nav.update_layout(height=400)
    st.plotly_chart(fig_nav, use_container_width=True)
    
    # Drawdown Area Chart
    fig_dd = px.area(drawdown, title='Underwater Plot (Drawdown)')
    fig_dd.update_traces(fillcolor='red', line_color='red')
    fig_dd.update_layout(height=300, yaxis_tickformat='.1%')
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Row B: Market Context
    st.subheader("Market Context: Corporate Spreads (bps)")
    
    # Filter by ticker option
    tickers = market['ticker'].unique()
    selected_tickers = st.multiselect("Select Issuers to View", tickers, default=tickers[:3])
    
    subset = market[market['ticker'].isin(selected_tickers)]
    
    # Spreads * 10000 to see bps
    subset['spread_bps'] = subset['spread'] * 10000
    
    fig_spread = px.line(
        subset, 
        x='date', 
        y='spread_bps', 
        color='ticker', 
        title='Credit Spreads Evolution'
    )
    st.plotly_chart(fig_spread, use_container_width=True)
    
    # Row C: Transaction Costs
    st.subheader("Operational Costs")
    fig_cost = px.bar(results, y='transaction_costs', title='Daily Transaction Costs Paid')
    st.plotly_chart(fig_cost, use_container_width=True)

if __name__ == "__main__":
    main()