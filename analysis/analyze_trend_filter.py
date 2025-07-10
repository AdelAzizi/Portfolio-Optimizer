# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Comparative Analysis of Multiple Market Trend Filters
# Description: A diagnostic script to visualize and compare the performance of
#              several different market trend filtering strategies.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from typing import Optional

def load_benchmark_data(price_data_path: Path) -> Optional[pd.Series]:
    """
    Loads the benchmark 'شاخص کل' data from the preprocessed feather file.
    """
    print(f"📂 Loading raw price data from '{price_data_path}'...")
    if not price_data_path.exists():
        print(f"❌ ERROR: Data file not found: {price_data_path}.")
        print("   Please run the main preprocessor.py script first to generate the data file.")
        return None
    
    try:
        price_df = pd.read_feather(price_data_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df.set_index('date', inplace=True)
        
        if 'شاخص کل' not in price_df.columns:
            print("❌ ERROR: 'شاخص کل' (benchmark index) not found in the data file.")
            return None
            
        benchmark_prices = price_df['شاخص کل'].dropna()
        benchmark_prices_numeric = pd.to_numeric(benchmark_prices, errors='coerce').dropna()
        
        print(f"✅ Benchmark data loaded successfully. {len(benchmark_prices_numeric)} data points found.")
        return benchmark_prices_numeric
        
    except Exception as e:
        print(f"❌ An error occurred while loading or processing the data file: {e}")
        return None

def analyze_trend_filters():
    """
    Loads preprocessed index data and compares three different trend-following strategies.
    """
    data_path = Path('cache_v3/master_price_data.feather')
    
    # 1. Load Data
    close_price = load_benchmark_data(data_path)
    if close_price is None:
        return

    start_date = pd.Timestamp.now() - pd.DateOffset(years=6)
    close_price = close_price[close_price.index >= start_date]
    print(f"   Filtered data from {close_price.index.min().date()} to {close_price.index.max().date()}.")

    # 2. Calculate All Indicators
    print("🧮 Calculating all moving averages (SMA and EMA)...")
    signals_df = pd.DataFrame(index=close_price.index)
    signals_df['close'] = close_price
    # Strategy 1: SMA Crossover
    signals_df['SMA50'] = close_price.rolling(window=50).mean()
    signals_df['SMA200'] = close_price.rolling(window=200).mean()
    # Strategy 3: EMA Crossover
    signals_df['EMA20'] = close_price.ewm(span=20, adjust=False).mean()
    signals_df['EMA50'] = close_price.ewm(span=50, adjust=False).mean()

    # 3. Identify Signals for Each Strategy
    print("🔍 Identifying crossover signals for all strategies...")
    
    # --- Strategy 1: 50/200 SMA Crossover (Baseline) ---
    signals_df['signal_sma'] = 0.0
    signals_df.loc[signals_df.index[200:], 'signal_sma'] = np.where(signals_df['SMA50'][200:] > signals_df['SMA200'][200:], 1.0, 0.0)
    signals_df['position_sma'] = signals_df['signal_sma'].diff()
    buy_sma = signals_df[signals_df['position_sma'] == 1.0]
    sell_sma = signals_df[signals_df['position_sma'] == -1.0]
    print(f"   - Strategy 1 (SMA 50/200): Found {len(buy_sma)} Buy signals, {len(sell_sma)} Sell signals.")

    # --- Strategy 2: Price / 200 SMA Crossover ---
    signals_df['signal_price_sma'] = 0.0
    signals_df.loc[signals_df.index[200:], 'signal_price_sma'] = np.where(signals_df['close'][200:] > signals_df['SMA200'][200:], 1.0, 0.0)
    signals_df['position_price_sma'] = signals_df['signal_price_sma'].diff()
    buy_price_sma = signals_df[signals_df['position_price_sma'] == 1.0]
    sell_price_sma = signals_df[signals_df['position_price_sma'] == -1.0]
    print(f"   - Strategy 2 (Price/SMA200): Found {len(buy_price_sma)} Buy signals, {len(sell_price_sma)} Sell signals.")

    # --- Strategy 3: 20/50 EMA Crossover ---
    signals_df['signal_ema'] = 0.0
    signals_df.loc[signals_df.index[50:], 'signal_ema'] = np.where(signals_df['EMA20'][50:] > signals_df['EMA50'][50:], 1.0, 0.0)
    signals_df['position_ema'] = signals_df['signal_ema'].diff()
    buy_ema = signals_df[signals_df['position_ema'] == 1.0]
    sell_ema = signals_df[signals_df['position_ema'] == -1.0]
    print(f"   - Strategy 3 (EMA 20/50): Found {len(buy_ema)} Buy signals, {len(sell_ema)} Sell signals.")

    # 4. Plotting
    print("🎨 Generating comparative analysis plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot Price and MAs
    ax.plot(signals_df.index, signals_df['close'], color='black', lw=1.5, label='TSE Index (Close)')
    ax.plot(signals_df.index, signals_df['SMA50'], color='blue', lw=1.0, linestyle='--', alpha=0.7, label='50-Day SMA')
    ax.plot(signals_df.index, signals_df['SMA200'], color='red', lw=1.2, linestyle='--', alpha=0.8, label='200-Day SMA')
    ax.plot(signals_df.index, signals_df['EMA20'], color='purple', lw=1.0, linestyle='-', alpha=0.7, label='20-Day EMA')
    ax.plot(signals_df.index, signals_df['EMA50'], color='orange', lw=1.2, linestyle='-', alpha=0.8, label='50-Day EMA')

    # Plot Signals
    # Strat 1: SMA 50/200
    ax.plot(buy_sma.index, signals_df.loc[buy_sma.index]['SMA50'], '^', markersize=15, color='green', label='Buy (SMA 50/200)', alpha=0.9, markeredgecolor='k')
    ax.plot(sell_sma.index, signals_df.loc[sell_sma.index]['SMA50'], 'v', markersize=15, color='red', label='Sell (SMA 50/200)', alpha=0.9, markeredgecolor='k')
    # Strat 2: Price/SMA200
    ax.plot(buy_price_sma.index, signals_df.loc[buy_price_sma.index]['SMA200'], 'o', markersize=10, color='cyan', label='Buy (Price/SMA200)', alpha=0.9, markeredgecolor='k')
    ax.plot(sell_price_sma.index, signals_df.loc[sell_price_sma.index]['SMA200'], 'o', markersize=10, color='magenta', label='Sell (Price/SMA200)', alpha=0.9, markeredgecolor='k')
    # Strat 3: EMA 20/50
    ax.plot(buy_ema.index, signals_df.loc[buy_ema.index]['EMA50'], 's', markersize=10, color='lime', label='Buy (EMA 20/50)', alpha=0.9, markeredgecolor='k')
    ax.plot(sell_ema.index, signals_df.loc[sell_ema.index]['EMA50'], 's', markersize=10, color='brown', label='Sell (EMA 20/50)', alpha=0.9, markeredgecolor='k')

    ax.set_title('Comparative Analysis of Market Trend Filters', fontsize=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Index Value', fontsize=14)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    fig.autofmt_xdate()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_filename = 'trend_comparison_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved successfully to: {output_filename}")

    print("✅ Displaying plot...")
    plt.show()

if __name__ == "__main__":
    try:
        plt.rcParams['font.family'] = 'Tahoma'
    except:
        print("Warning: 'Tahoma' font not found. Plot labels might not render correctly.")
        pass
    
    analyze_trend_filters()