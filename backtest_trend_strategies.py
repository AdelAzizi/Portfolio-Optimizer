# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Comprehensive Backtesting Engine for Market Trend-Following Strategies
# Description: A robust tool to quantitatively compare various market timing
#              strategies on the TSE All-Share Index, considering transaction costs.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

# --- Configuration ---
RISK_FREE_RATE = 0.35
TRANSACTION_COST = 0.005 # 0.5% per trade

# ==============================================================================
# 1. Data Loading
# ==============================================================================
def load_benchmark_data(price_data_path: Path) -> Optional[pd.Series]:
    """
    Loads the benchmark 'شاخص کل' data from the preprocessed feather file.
    """
    print(f"📂 Loading raw price data from '{price_data_path}'...")
    if not price_data_path.exists():
        print(f"❌ ERROR: Data file not found: {price_data_path}.")
        print("   Please run the main preprocessor.py script first.")
        return None
    
    try:
        price_df = pd.read_feather(price_data_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df.set_index('date', inplace=True)
        
        if 'شاخص کل' not in price_df.columns:
            print("❌ ERROR: 'شاخص کل' (benchmark index) not found.")
            return None
            
        benchmark_prices = pd.to_numeric(price_df['شاخص کل'], errors='coerce').dropna()
        print(f"✅ Benchmark data loaded successfully. {len(benchmark_prices)} data points found.")
        return benchmark_prices
        
    except Exception as e:
        print(f"❌ An error occurred while loading data: {e}")
        return None

# ==============================================================================
# 2. Strategy Logic Functions
# ==============================================================================
def strategy_sma_crossover(prices: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """Implements the 50/200 SMA crossover logic."""
    sma_short = prices.rolling(window=short_window).mean()
    sma_long = prices.rolling(window=long_window).mean()
    signals = pd.Series(0, index=prices.index)
    signals[sma_short > sma_long] = 1
    return signals.fillna(0)

def strategy_price_vs_sma(prices: pd.Series, long_window: int = 200) -> pd.Series:
    """Implements the Price vs. 200-day SMA logic."""
    sma_long = prices.rolling(window=long_window).mean()
    signals = pd.Series(0, index=prices.index)
    signals[prices > sma_long] = 1
    return signals.fillna(0)

def strategy_ema_crossover(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.Series:
    """Implements the 20/50 EMA crossover logic."""
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long = prices.ewm(span=long_window, adjust=False).mean()
    signals = pd.Series(0, index=prices.index)
    signals[ema_short > ema_long] = 1
    return signals.fillna(0)

# ==============================================================================
# 3. Core Backtesting Engine
# ==============================================================================
def run_strategy_backtest(
    price_series: pd.Series, 
    strategy_logic_function: Callable, 
    transaction_cost: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs a backtest for a given strategy and returns performance metrics.
    """
    # a. Get signals from the strategy logic
    signals = strategy_logic_function(price_series)
    
    # b. Calculate daily returns for benchmark and strategy
    benchmark_returns = price_series.pct_change().fillna(0)
    strategy_returns = benchmark_returns * signals.shift(1) # Use yesterday's signal to trade today

    # c. Incorporate Transaction Costs
    trades = signals.diff().abs()
    trade_days = trades[trades > 0].index
    strategy_returns.loc[trade_days] -= transaction_cost
    
    # d. Calculate cumulative returns
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()
    
    # e. Calculate KPIs
    total_days = len(price_series)
    trading_days_per_year = 252

    # Cumulative Annualized Return
    total_return = cumulative_strategy_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (trading_days_per_year / total_days) - 1

    # Annualized Volatility
    annualized_volatility = strategy_returns.std() * np.sqrt(trading_days_per_year)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility if annualized_volatility != 0 else 0

    # Max Drawdown
    running_max = cumulative_strategy_returns.cummax()
    drawdown = (cumulative_strategy_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    kpis = {
        "Cumulative Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Total Number of Trades": trades.sum()
    }
    
    results_df = pd.DataFrame({
        'Strategy': cumulative_strategy_returns,
        'Benchmark': cumulative_benchmark_returns
    })
    
    return results_df, kpis

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
def main():
    """
    Main function to orchestrate the backtesting of all defined strategies.
    """
    print("🚀 Starting Backtesting Engine...")
    
    # Create output directory
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    print(f"   Output will be saved to '{output_dir}' directory.")

    # Load Data
    price_data = load_benchmark_data(Path('cache_v3/master_price_data.feather'))
    if price_data is None:
        return
        
    # Filter data to a reasonable backtest period
    price_data = price_data.loc['2018-01-01':]
    print(f"   Backtesting on data from {price_data.index.min().date()} to {price_data.index.max().date()}.")

    # Define strategies to test
    strategies = {
        "SMA Crossover (50/200)": strategy_sma_crossover,
        "Price vs SMA (200)": strategy_price_vs_sma,
        "EMA Crossover (20/50)": strategy_ema_crossover
    }

    # Open a file to save the results and a dictionary to store plot data
    results_filepath = output_dir / "backtest_summary.txt"
    all_results_for_plotting = {}

    with open(results_filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Backtesting Results for Market Trend-Following Strategies\n")
        f.write("="*80 + "\n\n")

        # Loop through each strategy and run the backtest
        for name, logic_func in strategies.items():
            print("\n" + "="*80)
            print(f"📈 Backtesting Strategy: {name}")
            print("="*80)
            
            # Run backtest
            results, kpis = run_strategy_backtest(price_data, logic_func, TRANSACTION_COST)
            all_results_for_plotting[name] = results # Store for final plot
            
            # --- Generate Output ---
            kpi_string = "\n".join([f"{key:<30}: {value:,.4f}" for key, value in kpis.items()])
            output_block = (
                f"Strategy: {name}\n"
                "-----------------------------------------\n"
                "--- Key Performance Indicators (KPIs) ---\n"
                f"{kpi_string}\n"
                "-----------------------------------------\n"
            )
            
            print(output_block)
            f.write(output_block + "\n\n")

            # Plot and save individual results
            plot_filename = f"backtest_{name.replace(' ', '_').replace('/', '')}.png"
            plot_filepath = output_dir / plot_filename
            
            fig, ax = plt.subplots(figsize=(15, 8))
            results['Strategy'].plot(ax=ax, label=name, color='blue', lw=2)
            results['Benchmark'].plot(ax=ax, label='Benchmark (Buy & Hold)', color='gray', linestyle='--', lw=2)
            
            ax.set_title(f'Backtest Performance: {name} vs. Benchmark', fontsize=16)
            ax.set_ylabel('Cumulative Return')
            ax.set_xlabel('Date')
            ax.legend(loc='upper left')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            plt.savefig(plot_filepath)
            plt.close(fig) # Close the plot to prevent it from showing now
            f.write(f"Plot saved to: {plot_filename}\n")
            f.write("="*80 + "\n\n")
            print(f"💾 Individual plot saved to: {plot_filepath}")

    print(f"\n✅ All backtesting results have been saved to '{results_filepath}'")

    # --- Create the final combined plot ---
    print("\n🎨 Generating final comparison plot...")
    fig_final, ax_final = plt.subplots(figsize=(18, 9))
    
    # Plot benchmark once
    benchmark_data = list(all_results_for_plotting.values())[0]['Benchmark']
    ax_final.plot(benchmark_data.index, benchmark_data, label='Benchmark (Buy & Hold)', color='black', linestyle='--', lw=2, alpha=0.8)

    # Plot each strategy
    colors = ['blue', 'green', 'purple']
    for i, (name, results) in enumerate(all_results_for_plotting.items()):
        ax_final.plot(results.index, results['Strategy'], label=name, color=colors[i % len(colors)], lw=2, alpha=0.9)

    ax_final.set_title('Overall Strategy Comparison vs. Benchmark', fontsize=20)
    ax_final.set_ylabel('Cumulative Return')
    ax_final.set_xlabel('Date')
    ax_final.legend(loc='upper left', fontsize=12)
    ax_final.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    final_plot_path = output_dir / "overall_comparison.png"
    plt.savefig(final_plot_path)
    print(f"💾 Final comparison plot saved to: {final_plot_path}")
    
    print("✅ Displaying final comparison plot...")
    plt.show()

if __name__ == "__main__":
    try:
        plt.rcParams['font.family'] = 'Tahoma'
    except:
        print("Warning: 'Tahoma' font not found. Plot labels might not render correctly.")
    main()