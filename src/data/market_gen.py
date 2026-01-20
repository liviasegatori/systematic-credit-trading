import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# --- PATH CONFIGURATION ---
# Calculate project root relative to this script
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
SEED = 42

# --- ISSUER DEFINITION ---
@dataclass
class Issuer:
    ticker: str
    sector: str
    base_spread_bps: float  # Long-term average spread (e.g., 0.0100 = 100 bps)
    volatility: float       # Spread volatility
    mean_reversion: float   # Speed of mean reversion (Kappa)

# --- SIMULATION FUNCTIONS ---

def simulate_ou_spread_process(
    n_days: int,
    start_spread: float,
    long_term_mean: float,
    kappa: float,
    vol: float,
    dt: float = 1/252
) -> np.ndarray:
    """
    Simulates credit spreads using an Ornstein-Uhlenbeck process.
    Equation: dx = kappa * (theta - x) * dt + sigma * dW
    """
    spreads = np.zeros(n_days)
    spreads[0] = start_spread
    shocks = np.random.normal(0, np.sqrt(dt), n_days)
    
    for t in range(1, n_days):
        # Drift pulls the spread back to the long-term mean
        drift = kappa * (long_term_mean - spreads[t-1]) * dt
        # Diffusion adds market noise
        diffusion = vol * shocks[t]
        
        spreads[t] = spreads[t-1] + drift + diffusion
        
        # Floor spread at 0.5 bps to prevent negative spreads
        spreads[t] = max(spreads[t], 0.00005)
        
    return spreads

def price_bond_approx(yield_decimal: float, coupon: float, maturity_years: float = 5.0) -> float:
    """
    Calculates the price of a generic bond with constant maturity.
    """
    face_value = 100.0
    # Annual coupons for simplicity
    times = np.arange(1, int(maturity_years) + 1)
    
    # Discount factors
    dfs = 1 / ((1 + yield_decimal) ** times)
    
    # PV of Coupons + PV of Principal
    pv_coupons = np.sum(coupon * dfs)
    pv_principal = face_value / ((1 + yield_decimal) ** maturity_years)
    
    return pv_coupons + pv_principal

# --- MAIN EXECUTION ---

def generate_market_data():
    np.random.seed(SEED)
    print("\n--- Starting Synthetic Market Generation ---")
    
    # 1. Load Risk-Free Rates (Cleaned FRED Data)
    rates_path = PROCESSED_PATH / "us_yield_curve.csv"
    if not rates_path.exists():
        print(f"CRITICAL ERROR: {rates_path} not found. Please run fred_loader.py first.")
        return
    
    rates_df = pd.read_csv(rates_path, index_col=0, parse_dates=True)
    print(f"Loaded Risk-Free Rates: {len(rates_df)} days.")
    
    # Use 5Y Rate as benchmark. If missing, fallback to closest available.
    if 'US_Rate_5.0Y' in rates_df.columns:
        base_rates = rates_df['US_Rate_5.0Y'] / 100.0
    elif 'US_Rate_2.0Y' in rates_df.columns:
        print("WARNING: 5Y Rate missing, using 2Y as proxy.")
        base_rates = rates_df['US_Rate_2.0Y'] / 100.0
    else:
        print("CRITICAL: No suitable benchmark rate found (5Y or 2Y).")
        return

    dates = rates_df.index
    n_days = len(dates)

    # 2. Define Universe
    issuers = [
        # Investment Grade - Tech
        Issuer("TMT_CORP_1", "Technology", 0.0080, 0.004, 2.0),
        Issuer("TMT_CORP_2", "Technology", 0.0120, 0.006, 1.5),
        # High Volatility - Energy
        Issuer("IND_ENGY_1", "Energy",     0.0250, 0.015, 0.8),
        # Industrial
        Issuer("IND_AUTO_1", "Industrial", 0.0180, 0.010, 1.0),
        # Financials
        Issuer("FIN_BANK_1", "Financials", 0.0110, 0.008, 3.0),
        Issuer("FIN_INS_1",  "Financials", 0.0130, 0.007, 2.5),
        # High Yield
        Issuer("CNS_RET_1",  "Consumer",   0.0350, 0.020, 0.5),
        # Defensive
        Issuer("CNS_STAP_1", "Consumer",   0.0060, 0.002, 4.0),
    ]

    all_data = []
    print(f"Generating data for {len(issuers)} issuers...")

    # 3. Simulate Data
    for issuer in issuers:
        # A. Simulate Spread
        spreads = simulate_ou_spread_process(
            n_days, 
            start_spread=issuer.base_spread_bps,
            long_term_mean=issuer.base_spread_bps,
            kappa=issuer.mean_reversion,
            vol=issuer.volatility
        )
        
        # B. Calculate Total Yield
        corp_yields = base_rates + spreads
        
        # C. Define Coupon (Par-ish at start)
        avg_yield = np.mean(corp_yields)
        coupon = round(avg_yield * 100 * 2) / 2.0 
        
        # D. Price the Bond
        prices = [price_bond_approx(y, coupon, maturity_years=5.0) for y in corp_yields]
        
        # E. Create DataFrame
        issuer_df = pd.DataFrame({
            'date': dates,
            'ticker': issuer.ticker,
            'sector': issuer.sector,
            'maturity_tenor': '5Y',
            'coupon': coupon,
            'risk_free_rate': base_rates,
            'spread': spreads,
            'yield': corp_yields,
            'price': prices
        })
        
        all_data.append(issuer_df)

    # 4. Save
    market_df = pd.concat(all_data, ignore_index=True)
    market_df = market_df.sort_values(['date', 'ticker'])
    
    # Rounding for cleanliness
    market_df['price'] = market_df['price'].round(4)
    market_df['spread'] = market_df['spread'].round(6)
    
    out_file = OUTPUT_PATH / "corporate_universe.csv"
    market_df.to_csv(out_file, index=False)
    
    print(f"\nSUCCESS! Market data generated at: {out_file}")
    print(f"Total rows: {len(market_df)}")

if __name__ == "__main__":
    generate_market_data()