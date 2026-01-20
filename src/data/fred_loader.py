import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --- DYNAMIC PATH CONFIGURATION ---
# Calculates the project root based on this script's location.
# File location: .../systematic-credit-trading/src/data/fred_loader.py
# Logic: Go up 3 levels (data -> src -> project_root)
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

print(f"DEBUG: Project Root detected at: {PROJECT_ROOT}")
print(f"DEBUG: Looking for raw files in: {RAW_PATH}")
# ----------------------------------

# Mapping filenames to maturity (in years)
# Note: DTB3 is used for 3-month T-Bill (0.25 years)
FILE_MAP = {
    "DTB3.csv": 0.25,
    "DTB6.csv": 0.5,
    "DGS1.csv": 1.0,
    "DGS2.csv": 2.0,
    "DGS5.csv": 5.0,
    "DGS10.csv": 10.0,
    "DGS30.csv": 30.0
}

def clean_fred_series(path, maturity):
    """
    Reads a FRED CSV robustly.
    - Handles comma (,) or semicolon (;) separators automatically.
    - Finds the date column (DATE or OBSERVATION_DATE) regardless of case.
    - Handles '.' as missing data.
    """
    try:
        # engine='python' with sep=None auto-detects the delimiter
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception as e:
        print(f"ERROR: Could not read {path.name}. Reason: {e}")
        return None

    # Normalize column names: strip whitespace and convert to UPPERCASE
    df.columns = [c.upper().strip() for c in df.columns]
    
    # Identify the Date column dynamically
    date_col = None
    for col in df.columns:
        if 'DATE' in col:  # Matches 'DATE', 'OBSERVATION_DATE', etc.
            date_col = col
            break
            
    # Fallback: if no column has "DATE" in the name, assume the first column is the date
    if date_col is None:
        date_col = df.columns[0]
        
    # Parse Dates and Set Index
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    except Exception as e:
        print(f"ERROR: Could not parse dates in {path.name}. Reason: {e}")
        return None
    
    # Clean Values: FRED uses '.' for missing data
    df = df.replace('.', np.nan)
    
    # Identify the value column (the one remaining)
    if len(df.columns) > 0:
        value_col = df.columns[0]
        try:
            # Convert to float
            df[value_col] = df[value_col].astype(float)
            
            # Rename column to standard format (e.g., US_Rate_10.0Y)
            final_col_name = f"US_Rate_{maturity}Y"
            df.columns = [final_col_name]
            return df
        except ValueError:
            print(f"ERROR: Could not convert data to float in {path.name}.")
            return None
    else:
        return None

def process_rates():
    frames = []
    print("\n--- Starting FRED Data Processing ---")
    
    # Ensure processed directory exists
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # Iterate through the file map
    for filename, maturity in FILE_MAP.items():
        file_path = RAW_PATH / filename
        
        if file_path.exists():
            print(f"Processing {filename}...")
            df = clean_fred_series(file_path, maturity)
            if df is not None:
                frames.append(df)
        else:
            print(f"WARNING: File {filename} NOT FOUND in {RAW_PATH}")

    if not frames:
        print("\nCRITICAL ERROR: No files were processed. Please check the 'data/raw' folder.")
        return

    # Merge all series into one DataFrame
    print("\nMerging data...")
    yield_curve = pd.concat(frames, axis=1).sort_index()
    
    # Forward fill to handle weekends/holidays (carry forward last known value)
    yield_curve = yield_curve.ffill()

    # Filter for the requested period
    yield_curve = yield_curve.loc['2015-01-01':'2026-01-20']

    # Save to CSV
    output_file = PROCESSED_PATH / "us_yield_curve.csv"
    yield_curve.to_csv(output_file)
    
    print(f"\nSUCCESS! Yield curve saved to: {output_file}")
    print(f"Dataset Shape: {yield_curve.shape}")
    print("\n--- First 3 Rows ---")
    print(yield_curve.head(3))
    print("\n--- Last 3 Rows ---")
    print(yield_curve.tail(3))

if __name__ == "__main__":
    process_rates()