import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURAZIONE ---
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "corporate_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

def load_data():
    """Carica i dati di mercato generati."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run market_gen.py first! {DATA_PATH} not found.")
    
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    return df

def calculate_carry_signal(df):
    """
    Signal 1: CARRY
    Logica: Higher Spread = Higher Carry.
    Vogliamo essere LONG sui bond con spread alto.
    """
    # Pivot: Righe=Date, Colonne=Ticker, Valori=Spread
    spreads = df.pivot(index='date', columns='ticker', values='spread')
    
    # Classifica giornaliera (Rank): Chi ha lo spread più alto oggi?
    # pct=True normalizza tra 0 e 1.
    rank_carry = spreads.rank(axis=1, pct=True) 
    
    # Centriamo il segnale tra -0.5 e +0.5
    # (Valori > 0 = Buy, Valori < 0 = Sell/Avoid)
    signal_carry = rank_carry - 0.5
    return signal_carry

def calculate_momentum_signal(df, lookback=20):
    """
    Signal 2: SPREAD MOMENTUM
    Logica: Se lo spread scende, il prezzo sale. Trend following.
    Formula: -(Spread_oggi - Spread_20gg_fa)
    """
    spreads = df.pivot(index='date', columns='ticker', values='spread')
    
    # Variazione spread a 20 giorni
    delta_spread = spreads.diff(lookback)
    
    # Se delta è negativo (spread sceso), vogliamo segnale positivo.
    raw_mom = -delta_spread
    
    # Normalizzazione rank cross-sectional
    signal_mom = raw_mom.rank(axis=1, pct=True) - 0.5
    return signal_mom

def calculate_value_signal(df, lookback=60):
    """
    Signal 3: MEAN REVERSION (VALUE)
    Logica: Z-Score. Se lo spread è molto sopra la media mobile, è "cheap" (Buy).
    """
    spreads = df.pivot(index='date', columns='ticker', values='spread')
    
    # Media e Deviazione Standard mobile a 60 giorni
    rolling_mean = spreads.rolling(window=lookback).mean()
    rolling_std = spreads.rolling(window=lookback).std()
    
    # Z-Score: Quante deviazioni standard siamo lontani dalla media?
    z_score = (spreads - rolling_mean) / rolling_std
    
    # Se Z-Score è alto (spread largo), ci aspettiamo che torni in media (prezzo sale).
    signal_value = z_score 
    
    # Winsorize (tagliamo gli estremi oltre +/- 3 sigma per stabilità)
    signal_value = signal_value.clip(-3, 3)
    
    # Scaliamo per renderlo simile agli altri (-0.5 a 0.5 circa)
    signal_value = signal_value / 6.0 
    return signal_value

def generate_signals():
    print("--- Calculating Alpha Signals ---")
    df = load_data()
    
    # 1. Calcolo i singoli fattori
    print("Computing Carry...")
    s_carry = calculate_carry_signal(df)
    
    print("Computing Momentum...")
    s_mom = calculate_momentum_signal(df)
    
    print("Computing Value (Mean Reversion)...")
    s_value = calculate_value_signal(df)
    
    print("Combining Signals...")
    
    # 2. Combinazione (Multi-Factor Model)
    # Pesi: 40% Carry, 30% Momentum, 30% Value
    combined_signal = (0.4 * s_carry) + (0.3 * s_mom) + (0.3 * s_value)
    
    # 3. Formattazione Output
    # Trasformiamo la matrice in una lista lunga (Date, Ticker, Signal)
    signals_long = combined_signal.reset_index().melt(
        id_vars='date', 
        var_name='ticker', 
        value_name='signal_strength'
    )
    
    # Rimuoviamo i NaN (i primi giorni non hanno storico per momentum/medie)
    signals_long = signals_long.dropna()
    
    # Salvataggio
    save_path = OUTPUT_PATH / "signals.csv"
    signals_long.to_csv(save_path, index=False)
    
    print(f"SUCCESS! Signals saved to {save_path}")
    print(f"Signal Data Shape: {signals_long.shape}")
    print("\nSample Signals (Tail):")
    print(signals_long.tail())

if __name__ == "__main__":
    generate_signals()