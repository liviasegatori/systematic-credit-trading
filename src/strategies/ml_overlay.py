import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "corporate_universe.csv"
SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "signals.csv" # We use raw signals as input
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

def prepare_ml_dataset():
    print("Preparing ML Dataset...")
    
    # 1. Load Prices and Factors (Carry, Mom, Value)
    # Note: We need to recalculate the raw factors here or save them separately beforehand.
    # For simplicity, I recalculate them
    market = pd.read_csv(DATA_PATH, parse_dates=['date'])
    spreads = market.pivot(index='date', columns='ticker', values='spread')
    prices = market.pivot(index='date', columns='ticker', values='price')
    
    # --- FEATURE ENGINEERING (Your factors are the input) ---
    # Carry (Spread level)
    f_carry = spreads
    # Momentum (Change in spread)
    f_mom = -spreads.diff(20)
    # Value (Z-Score)
    rolling_mean = spreads.rolling(60).mean()
    rolling_std = spreads.rolling(60).std()
    f_value = (spreads - rolling_mean) / rolling_std
    
    # --- TARGET (What do we want to predict?) ---
    # We want to predict the 5-day bond return (Next Week Return)
    # Return = (Price_t+5 - Price_t) / Price_t
    future_returns = prices.shift(-5) / prices - 1
    
    # Create a single long DataFrame for scikit-learn
    features = []
    for ticker in spreads.columns:
        df_ticker = pd.DataFrame({
            'ticker': ticker,
            'carry': f_carry[ticker],
            'momentum': f_mom[ticker],
            'value': f_value[ticker],
            'target_return': future_returns[ticker]
        })
        features.append(df_ticker)
        
    full_df = pd.concat(features).dropna()
    return full_df.sort_index() # Time order 

def train_model():
    df = prepare_ml_dataset()
    print(f"Dataset ready: {len(df)} samples.")
    
    # Split Train/Test 
    # We use 2015-2020 for train, 2021+ for test
    train_df = df[df.index < '2021-01-01']
    test_df = df[df.index >= '2021-01-01']
    
    X_train = train_df[['carry', 'momentum', 'value']]
    y_train = train_df['target_return']
    
    X_test = test_df[['carry', 'momentum', 'value']]
    
    print("Training Random Forest (The 'Brain')...")
    # We use RandomForest (similar to XGBoost but easier to install on the fly)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Feature Importance (What is the model looking at?)
    importances = model.feature_importances_
    print("\n--- What did the AI learn? (Feature Importance) ---")
    print(f"Carry:    {importances[0]:.2%}")
    print(f"Momentum: {importances[1]:.2%}")
    print(f"Value:    {importances[2]:.2%}")
    
    # Predictions
    print("Generating AI Signals...")
    pred_test = model.predict(X_test)
    
    # Transform predictions into a signals DataFrame compatible with engine.py
    test_df = test_df.copy()
    test_df['ai_signal'] = pred_test
    
    # Cross-sectional normalization (Ranking)
    # The AI gives us a predicted return. We buy the Top, sell the Bottom.
    test_df = test_df.reset_index()
    ai_signals = test_df.pivot(index='date', columns='ticker', values='ai_signal')
    
    # Rank -0.5 to +0.5
    final_signals = ai_signals.rank(axis=1, pct=True) - 0.5
    
    # Save in 'long' format for the backtester
    signals_long = final_signals.reset_index().melt(id_vars='date', var_name='ticker', value_name='signal_strength').dropna()
    
    out_file = OUTPUT_PATH / "signals_ml.csv"
    signals_long.to_csv(out_file, index=False)
    print(f"AI Signals saved to {out_file}")

if __name__ == "__main__":
    train_model()