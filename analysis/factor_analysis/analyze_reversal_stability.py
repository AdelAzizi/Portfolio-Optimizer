import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_reversal_stability():
    """
    Performs a rolling window analysis of the long-term reversal factor in the Iranian stock market.
    """
    # --- 1. Data Loading ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'cache_v3', 'master_price_data.feather')
    try:
        df = pd.read_feather(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # --- Data Preparation ---
    if 'date' not in df.columns:
        df = df.reset_index()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Drop benchmark if it exists, as it's not a stock
    if 'شاخص کل' in df.columns:
        df = df.drop(columns=['شاخص کل'])

    # Ensure all price columns are numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that might be entirely empty after coercion
    df = df.dropna(axis=1, how='all')

    prices = df

    # --- 2. Factor Calculation ---
    # Calculate 3-year (756 trading days) historical return
    reversal_factor = prices.pct_change(periods=756).shift(-756)

    # --- 3. Rolling Window Backtesting Engine ---
    window_size = 3 * 252  # 3-year analysis window
    step_size = 126       # 6-month step
    top_quantile = 0.2    # 20% worst-performing stocks

    results = []
    
    start_idx = 0
    end_idx = window_size
    
    while end_idx <= len(prices):
        window_prices = prices.iloc[start_idx:end_idx]
        window_factor = reversal_factor.iloc[start_idx]

        # --- Form Portfolio ---
        eligible_stocks = window_factor.dropna().index
        
        if not eligible_stocks.empty:
            num_stocks_to_select = int(len(eligible_stocks) * top_quantile)
            
            if num_stocks_to_select > 0:
                worst_performers = window_factor.nsmallest(num_stocks_to_select).index
                
                # --- Calculate Performance ---
                portfolio_returns = window_prices[worst_performers].pct_change().mean(axis=1)
                
                # Calculate annualized return for the 3-year window
                cumulative_return = (1 + portfolio_returns).prod()
                annualized_return = (cumulative_return ** (252 / len(window_prices))) - 1
                
                # --- Store Result ---
                window_end_date = window_prices.index[-1]
                results.append({'end_date': window_end_date, 'annualized_return': annualized_return})

        start_idx += step_size
        end_idx += step_size

    if not results:
        print("No results generated. Check data and parameters.")
        return

    results_df = pd.DataFrame(results).set_index('end_date')

    # --- 4. Visualization of Results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(results_df.index, results_df['annualized_return'], marker='o', linestyle='-', color='b')
    
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    
    ax.set_title('3-Year Rolling Annualized Return of Reversal Strategy', fontsize=16)
    ax.set_xlabel('Window End Date', fontsize=12)
    ax.set_ylabel('3-Year Annualized Return', fontsize=12)
    
    # --- Save and Show Plot ---
    plot_path = os.path.join(script_dir, 'reversal_factor_stability.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    plt.show()

if __name__ == '__main__':
    analyze_reversal_stability()