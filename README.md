# Systematic Credit Trading Framework 🏦

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end quantitative trading framework for **Corporate Bonds**, simulating the full lifecycle of a systematic credit desk: from synthetic market generation and signal construction to realistic backtesting and execution optimization.

## 🚀 Key Features

* **Synthetic Market Generator**: Implements **Ornstein-Uhlenbeck** processes to simulate realistic credit spread dynamics (mean-reversion, volatility clustering) on top of real FRED yield curve data.
* **Multi-Factor Signal Engine**: Calculates proprietary alpha signals based on:
    * *Carry*: Cross-sectional spread capture.
    * *Momentum*: Spread tightening trends.
    * *Value*: Mean-reversion statistical arbitrage (Z-Score).
* **Execution & Cost Modeling**:
    * Realistic Bid-Ask spread simulation (Transaction Cost Analysis).
    * **Smart Order Routing Logic**: Implemented weekly rebalancing and inertia filters to reduce turnover and transaction costs by **~60%** in illiquid credit markets.
* **Risk Dashboard**: Interactive Streamlit app for monitoring NAV, Sharpe Ratio, Drawdowns, and Spread Exposures.

## 📊 Performance & Optimization Case Study

This project demonstrates the critical impact of execution strategy in Fixed Income markets.

| Strategy Metric | Naive Execution (Daily) | Optimized Execution (Weekly + Inertia) | Improvement |
| :--- | :--- | :--- | :--- |
| **Final NAV** | $5,124,192 | **$8,341,702** | **+62%** |
| **Transaction Costs** | $5,718,511 | **$2,409,084** | **-58% (Savings)** |
| **Total Return** | -48.7% | **-16.5%** | **+32 pts** |

*Note: The negative absolute return reflects the structural short-duration bias (Market Beta) during the 2022 global rate hike cycle, which acted as a significant headwind for long-only bond portfolios.*

## 🛠️ Tech Stack

* **Core**: Python 3.10+, Pandas, NumPy
* **Simulation**: SciPy (Stochastic processes), Random
* **Visualization**: Plotly, Streamlit, Matplotlib
* **DevOps**: Git, Virtualenv

## 📂 Project Structure

```text
systematic-credit-trading/
├── data/                  # Market data (FRED rates + Synthetic Bonds)
├── notebooks/             # EDA and Research Sandboxes
├── src/
│   ├── data/              # ETL pipelines (Fred Loader, Market Gen)
│   ├── strategies/        # Alpha Signals (Carry, Momentum, Value)
│   ├── backtest/          # Event-driven engine with Cost Models
│   └── dashboard/         # Streamlit Web App
└── requirements.txt