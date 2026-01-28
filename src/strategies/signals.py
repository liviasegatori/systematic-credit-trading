import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "corporate_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

def load_data():
    """Load the generated market data."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run market_gen.py first! {DATA_PATH} not found.")
    
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    return df

def calculate_carry_signal(df):
    """
    Signal 1: CARRY
    Logic: Higher Spread = Higher Carry.
    We want to be LONG on bonds with high spread.
    """
    # Pivot: Rows=Date, Columns=Ticker, Values=Spread
    spreads = df.pivot(index='date', columns='ticker', values='spread')
    
    # Daily ranking (Rank): Who has the highest spread today?
    # pct=True normalizes between 0 and 1.
    rank_carry = spreads.rank(axis=1, pct=True) 
    
    # Center the signal between -0.5 and +0.5
    # (Values > 0 = Buy, Values < 0 = Sell/Avoid)
    signal_carry = rank_carry - 0.5
    return signal_carry

def calculate_momentum_signal(df, lookback=20):
    """
    Signal 2: SPREAD MOMENTUM
    Logic: If the spread falls, the price rises. Trend following.
    Formula: -(Spread_today - Spread_20days_ago)
    """
    spreads = df.pivot(index='date', columns='ticker', values='spread')
    
    # 20-day spread change
    delta_spread = spreads.diff(lookback)
    
    # If delta is negative (spread decreased), we want a positive signal.
    raw_mom = -delta_spread
    
    # Cross-sectional rank normalization
    signal_mom = raw_mom.rank(axis=1, pct=True) - 0.5
    return signal_mom

def calculate_value_signal(df, lookback=60):
    """
    Signal 3: MEAN REVERSION (VALUE)
    Logic: Z-Score. If the spread is far above the moving average, it's "cheap" (Buy).
    """
    spreads = df.pivot(index='date', columns='ticker', values='spread')
    
    # 60-day rolling mean and standard deviation
    rolling_mean = spreads.rolling(window=lookback).mean()
    rolling_std = spreads.rolling(window=lookback).std()
    
    # Z-Score: How many standard deviations are we from the mean?
    z_score = (spreads - rolling_mean) / rolling_std
    
    # If Z-Score is high (wide spread), we expect mean reversion (price rises).
    signal_value = z_score 
    
    # Winsorize (trim extremes beyond +/- 3 sigma for stability)
    signal_value = signal_value.clip(-3, 3)
    
    # Scale to make it similar to the others (about -0.5 to 0.5)
    signal_value = signal_value / 6.0 
    return signal_value

def generate_signals():
    print("--- Calculating Alpha Signals ---")
    df = load_data()
    
    # 1. Compute individual factors
    print("Computing Carry...")
    s_carry = calculate_carry_signal(df)
    
    print("Computing Momentum...")
    s_mom = calculate_momentum_signal(df)
    
    print("Computing Value (Mean Reversion)...")
    s_value = calculate_value_signal(df)
    
    print("Combining Signals...")
    
    # 2. Combination (Multi-Factor Model)
    # Weights: 40% Carry, 30% Momentum, 30% Value
    combined_signal = (0.4 * s_carry) + (0.3 * s_mom) + (0.3 * s_value)
    
    # 3. Output formatting
    # Transform the matrix into a long list (Date, Ticker, Signal)
    signals_long = combined_signal.reset_index().melt(
        id_vars='date', 
        var_name='ticker', 
        value_name='signal_strength'
    )
    
    # Remove NaNs (early days lack history for momentum/rolling stats)
    signals_long = signals_long.dropna()
    
    # Saving
    save_path = OUTPUT_PATH / "signals.csv"
    signals_long.to_csv(save_path, index=False)
    
    print(f"SUCCESS! Signals saved to {save_path}")
    print(f"Signal Data Shape: {signals_long.shape}")
    print("\nSample Signals (Tail):")
    print(signals_long.tail())

if __name__ == "__main__":
    generate_signals()