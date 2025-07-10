import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define the path to your cached data
CACHE_FILE_PATH = Path('cache_v3/master_price_data.feather')
BENCHMARK_COLUMN_NAME = 'شاخص کل' 

print(f"Loading data from {CACHE_FILE_PATH}...")
try:
    # Load the Feather file
    price_df = pd.read_feather(CACHE_FILE_PATH)
    # Set the 'date' column as the index and convert to datetime objects
    price_df.set_index('date', inplace=True)
    price_df.index = pd.to_datetime(price_df.index)
    print("Data loaded successfully.")
    
    # --- DEBUG: Inspect the DataFrame ---
    print("\n--- DataFrame Info ---")
    price_df.info()
    print("\n--- DataFrame Head ---")
    print(price_df.head())
    # --- END DEBUG ---

    # Check if benchmark column exists
    if BENCHMARK_COLUMN_NAME not in price_df.columns:
        print(f"Available columns are: {price_df.columns.tolist()}")
        raise ValueError(f"Benchmark column '{BENCHMARK_COLUMN_NAME}' not found!")

    # Isolate the benchmark prices and drop any missing values
    # Ensure the benchmark column is numeric, coercing errors, then isolate and drop NaNs
    benchmark_prices = pd.to_numeric(price_df[BENCHMARK_COLUMN_NAME], errors='coerce').dropna()
    
    if benchmark_prices.empty:
        print("\nERROR: The benchmark prices series is empty after dropping NaNs.")
        print("This indicates the column in the source file contains no valid data.")
    else:
        # --- Calculations ---
        benchmark_returns = benchmark_prices.pct_change().dropna()
        cumulative_returns = (1 + benchmark_returns).cumprod()
        
        # --- Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 7))
        
        plt.plot(cumulative_returns, label=f'{BENCHMARK_COLUMN_NAME} Cumulative Return', color='cyan')
        
        plt.title('TSE All-Share Index Performance (5 Years)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        plt.show()

except FileNotFoundError:
    print(f"ERROR: Cache file not found at {CACHE_FILE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")