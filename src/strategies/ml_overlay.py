import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# --- CONFIGURAZIONE ---
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "corporate_universe.csv"
SIGNALS_PATH = PROJECT_ROOT / "data" / "processed" / "signals.csv" # Usiamo i segnali grezzi come input
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

def prepare_ml_dataset():
    print("Preparing ML Dataset...")
    
    # 1. Carichiamo Prezzi e Fattori (Carry, Mom, Value)
    # Nota: Dobbiamo ricalcolare i fattori grezzi qui o salvarli separati prima.
    # Per semplicità, li ricalcolo 
    market = pd.read_csv(DATA_PATH, parse_dates=['date'])
    spreads = market.pivot(index='date', columns='ticker', values='spread')
    prices = market.pivot(index='date', columns='ticker', values='price')
    
    # --- FEATURE ENGINEERING (I tuoi fattori sono l'Input) ---
    # Carry (Spread level)
    f_carry = spreads
    # Momentum (Change in spread)
    f_mom = -spreads.diff(20)
    # Value (Z-Score)
    rolling_mean = spreads.rolling(60).mean()
    rolling_std = spreads.rolling(60).std()
    f_value = (spreads - rolling_mean) / rolling_std
    
    # --- TARGET (Cosa vogliamo prevedere?) ---
    # Vogliamo prevedere il rendimento del bond a 5 giorni (Next Week Return)
    # Return = (Price_t+5 - Price_t) / Price_t
    future_returns = prices.shift(-5) / prices - 1
    
    # Creiamo un unico DataFrame lungo per scikit-learn
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
    return full_df.sort_index() # Ordine temporale 

def train_model():
    df = prepare_ml_dataset()
    print(f"Dataset ready: {len(df)} samples.")
    
    # Split Train/Test 
    # Usiamo il 2015-2020 per train, 2021+ per test
    train_df = df[df.index < '2021-01-01']
    test_df = df[df.index >= '2021-01-01']
    
    X_train = train_df[['carry', 'momentum', 'value']]
    y_train = train_df['target_return']
    
    X_test = test_df[['carry', 'momentum', 'value']]
    
    print("Training Random Forest (The 'Brain')...")
    # Usiamo RandomForest (simile a XGBoost ma più facile da installare al volo)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Feature Importance (Cosa guarda il modello?)
    importances = model.feature_importances_
    print("\n--- Cosa ha imparato l'AI? (Feature Importance) ---")
    print(f"Carry:    {importances[0]:.2%}")
    print(f"Momentum: {importances[1]:.2%}")
    print(f"Value:    {importances[2]:.2%}")
    
    # Predizioni
    print("Generating AI Signals...")
    pred_test = model.predict(X_test)
    
    # Trasforma le predizioni in un DataFrame segnali compatibile con engine.py
    test_df = test_df.copy()
    test_df['ai_signal'] = pred_test
    
    # Normalizzazione cross-sectional (Ranking)
    # L'AI ci dà un rendimento previsto. Noi compriamo i Top, vendiamo i Bottom.
    test_df = test_df.reset_index()
    ai_signals = test_df.pivot(index='date', columns='ticker', values='ai_signal')
    
    # Rank -0.5 a +0.5
    final_signals = ai_signals.rank(axis=1, pct=True) - 0.5
    
    # Salviamo in formato 'long' per il backtester
    signals_long = final_signals.reset_index().melt(id_vars='date', var_name='ticker', value_name='signal_strength').dropna()
    
    out_file = OUTPUT_PATH / "signals_ml.csv"
    signals_long.to_csv(out_file, index=False)
    print(f"AI Signals saved to {out_file}")

if __name__ == "__main__":
    train_model()