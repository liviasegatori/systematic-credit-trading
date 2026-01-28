import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "corporate_universe.csv"
SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "signals_ml.csv" # change depending on xgb or trad signal
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

# Desk parameters
INITIAL_CAPITAL = 10_000_000  # 10 million $
TRANSACTION_COST_BPS = 0.0010 # 10 bps (0.10%) per trade (Bid-Ask spread)
MAX_LEVERAGE = 1.0            # Leverage 1:1 (Long Only / Cash)

class BacktestEngine:
    def __init__(self, initial_capital, cost_bps):
        self.capital = initial_capital
        self.cost_bps = cost_bps
        self.positions = {} # {ticker: quantity}
        self.cash = initial_capital
        self.history = []
        
    def load_data(self):
        print("Loading market data and signals...")
        self.market_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        self.signals_df = pd.read_csv(SIGNALS_PATH, parse_dates=['date'])
        
        # Merge to align Price and Signal
        self.full_data = pd.merge(
            self.market_df, 
            self.signals_df, 
            on=['date', 'ticker'], 
            how='inner'
        )
        
        # Pivot for fast access
        self.prices = self.full_data.pivot(index='date', columns='ticker', values='price')
        self.signals = self.full_data.pivot(index='date', columns='ticker', values='signal_strength')
        self.dates = self.prices.index.sort_values()
        
    def run(self):
        print(f"--- Starting Backtest ($ {self.capital:,.0f}) ---")
        
        # FIX 1: WEEKLY REBALANCING
        # Bonds are slow. Trading every day hands money to brokers.
        # We trade only on FRIDAY (weekday 4).
        # This instantly reduces costs by ~80%.
        
        # FIX 2: HIGHER INERTIA THRESHOLD
        # We do not move for less than 100k (1% of the portfolio)
        MIN_TRADE_SIZE = 100_000 
        
        for date in self.dates:
            # --- WEEKLY LOGIC ---
            # If it's not Friday and not the last day of the backtest... SKIP and keep positions
            if date.weekday() != 4 and date != self.dates[-1]:
                # Still record today's value (Mark to Market)
                # (Duplicated code for daily reporting even if we don't trade)
                today_prices = self.prices.loc[date]
                nav = self.cash
                for ticker, qty in self.positions.items():
                    px = today_prices.get(ticker, 0)
                    nav += qty * px
                
                self.history.append({
                    'date': date,
                    'nav': nav,
                    'cash': self.cash,
                    'transaction_costs': 0, # Zero costs today!
                    'n_positions': len([k for k,v in self.positions.items() if v != 0])
                })
                continue
            # ---------------------

            # IF IT'S FRIDAY -> PERFORM REBALANCING
            try:
                today_prices = self.prices.loc[date]
                today_signals = self.signals.loc[date].fillna(0)
            except KeyError:
                continue
                
            portfolio_value = self.cash
            for ticker, qty in self.positions.items():
                px = today_prices.get(ticker, 0)
                portfolio_value += qty * px
                
            total_signal_strength = today_signals.abs().sum()
            trades_cost = 0
            
            if total_signal_strength > 0:
                for ticker in today_prices.index:
                    signal = today_signals.get(ticker, 0)
                    price = today_prices.get(ticker, 0)
                    
                    if pd.isna(price) or price <= 0:
                        continue
                        
                    target_weight = signal / total_signal_strength
                    target_dollar = target_weight * portfolio_value * MAX_LEVERAGE
                    target_qty = int(target_dollar / price)
                    
                    current_qty = self.positions.get(ticker, 0)
                    trade_qty = target_qty - current_qty
                    trade_dollar_val = abs(trade_qty * price)
                    
                    # Inertia filter (100k)
                    if trade_dollar_val < MIN_TRADE_SIZE:
                        continue

                    if trade_qty != 0:
                        cost = trade_dollar_val * self.cost_bps
                        trades_cost += cost
                        self.positions[ticker] = target_qty
                        self.cash -= (trade_qty * price)
            
            self.cash -= trades_cost
            
            # Calculate end-of-day NAV
            nav = self.cash
            for ticker, qty in self.positions.items():
                px = today_prices.get(ticker, 0)
                nav += qty * px
                
            self.history.append({
                'date': date,
                'nav': nav,
                'cash': self.cash,
                'transaction_costs': trades_cost,
                'n_positions': len(self.positions)
            })
            
        print("Backtest Complete.")
        
    def save_results(self):
        res_df = pd.DataFrame(self.history).set_index('date')
        
        # Compute metrics
        res_df['returns'] = res_df['nav'].pct_change()
        cumulative_ret = (res_df['nav'].iloc[-1] / res_df['nav'].iloc[0]) - 1
        sharpe = (res_df['returns'].mean() / res_df['returns'].std()) * np.sqrt(252)
        
        print(f"\n--- Performance Report ---")
        print(f"Final NAV:      ${res_df['nav'].iloc[-1]:,.2f}")
        print(f"Total Return:   {cumulative_ret*100:.2f}%")
        print(f"Sharpe Ratio:   {sharpe:.2f}")
        print(f"Total Tx Costs: ${res_df['transaction_costs'].sum():,.2f}")
        
        out_file = OUTPUT_PATH / "backtest_results.csv"
        res_df.to_csv(out_file)
        print(f"Results saved to {out_file}")
        
        return res_df

if __name__ == "__main__":
    engine = BacktestEngine(INITIAL_CAPITAL, TRANSACTION_COST_BPS)
    engine.load_data()
    engine.run()
    results = engine.save_results()