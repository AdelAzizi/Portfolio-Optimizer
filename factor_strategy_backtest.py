import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
# Monkey-patch for empyrical compatibility with NumPy 2.0+
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
import empyrical as ep

# --- Configuration ---
DATA_PATH = 'cache_v3/master_price_data.feather'
BENCHMARK_SYMBOL = 'شاخص کل'
REBALANCE_FREQ = '3M'  # Rebalance every 3 months
TOP_N_STOCKS = 20
TRANSACTION_COST = 0.005  # 0.5%

# Factor Calculation Periods
VOLATILITY_WINDOW = 252  # 1 year
REVERSAL_WINDOW = 756   # 3 years

# Market Trend EMA Periods
EMA_SHORT = 20
EMA_LONG = 50

def load_data(file_path):
    """Loads the master price data from a feather file."""
    try:
        df = pd.read_feather(file_path)
        # The preprocessor saves the date index as a column named 'date'.
        df['Date'] = pd.to_datetime(df['date'])
        df = df.set_index('Date')
        if 'date' in df.columns:
            df = df.drop(columns=['date'])

        # Convert all price columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean up the data after coercion
        df.dropna(axis=1, how='all', inplace=True) # Drop columns that are entirely NaN
        df.ffill(inplace=True) # Forward-fill to handle NaNs within columns
        df.bfill(inplace=True) # Back-fill any remaining NaNs at the start

        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    except KeyError:
        print(f"Error: 'date' column not found in {file_path}. Please check the preprocessor script.")
        return None

def calculate_market_signal(prices, benchmark_symbol, short_window, long_window):
    """Calculates the market trend signal based on EMA crossover."""
    benchmark_prices = prices[benchmark_symbol]
    ema_short = benchmark_prices.ewm(span=short_window, adjust=False).mean()
    ema_long = benchmark_prices.ewm(span=long_window, adjust=False).mean()
    
    market_signal = pd.Series(0, index=prices.index)
    market_signal[ema_short > ema_long] = 1
    return market_signal.ffill()

def calculate_factors(prices, vol_window, rev_window):
    """Calculates volatility and reversal factors for all stocks."""
    daily_returns = prices.pct_change()
    
    # Low Volatility Factor (lower is better)
    volatility = daily_returns.rolling(window=vol_window).std() * np.sqrt(vol_window)
    
    # Long-Term Reversal Factor (higher is better)
    # Total return over the period, then multiplied by -1
    total_return = prices.pct_change(periods=rev_window)
    reversal = -1 * total_return
    
    return volatility, reversal

def run_backtest(prices, market_signal, factors, rebalance_freq, top_n, benchmark_symbol, t_cost):
    """Runs the main backtesting loop."""
    volatility, reversal = factors
    stock_symbols = [col for col in prices.columns if col != benchmark_symbol]
    
    rebalance_dates = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq=rebalance_freq)
    
    portfolio_weights = pd.DataFrame(index=prices.index, columns=stock_symbols).fillna(0)
    
    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i+1]
        
        # Ensure we have data for the rebalance date
        if start_date not in prices.index:
            continue
            
        # 1. Check Market Signal
        signal = market_signal.loc[start_date]
        
        target_weights = pd.Series(0, index=stock_symbols)
        
        if signal == 1: # Uptrend -> Invest
            # 2. Factor Ranking
            vol_rank = volatility.loc[start_date].rank(ascending=True)
            rev_rank = reversal.loc[start_date].rank(ascending=False) # Higher reversal is better
            composite_rank = (vol_rank + rev_rank).dropna()
            
            # 3. Stock Selection
            selected_stocks = composite_rank.nsmallest(top_n).index.tolist()
            
            if not selected_stocks:
                continue

            # 4. Portfolio Optimization (Minimum Volatility)
            hist_prices = prices.loc[:start_date, selected_stocks]
            
            # Use data up to the rebalance date for optimization
            mu = expected_returns.mean_historical_return(hist_prices, frequency=252)
            S = risk_models.sample_cov(hist_prices, frequency=252)
            
            try:
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                weights = ef.clean_weights()
                target_weights.loc[weights.keys()] = list(weights.values())
            except Exception as e:
                # If optimization fails, equal weight the selected stocks
                print(f"Optimization failed on {start_date}. Equal weighting. Error: {e}")
                target_weights.loc[selected_stocks] = 1 / len(selected_stocks)

        # 5. Assign target weights for the period
        portfolio_weights.loc[start_date:end_date] = target_weights.values

    # Calculate portfolio returns
    daily_returns = prices.pct_change()
    strategy_returns = (daily_returns[stock_symbols] * portfolio_weights.shift(1)).sum(axis=1)
    
    # Apply transaction costs
    weight_changes = portfolio_weights.diff().abs().sum(axis=1)
    costs = weight_changes * t_cost
    strategy_returns -= costs
    
    return strategy_returns.fillna(0)

def analyze_performance(returns, benchmark_returns):
    """Calculates and prints performance metrics and plots cumulative returns."""
    # Calculate KPIs
    kpis = {
        'Annualized Return': ep.annual_return(returns),
        'Annualized Volatility': ep.annual_volatility(returns),
        'Sharpe Ratio': ep.sharpe_ratio(returns),
        'Max Drawdown': ep.max_drawdown(returns),
        'Calmar Ratio': ep.calmar_ratio(returns),
        'Sortino Ratio': ep.sortino_ratio(returns)
    }
    
    benchmark_kpis = {
        'Annualized Return': ep.annual_return(benchmark_returns),
        'Annualized Volatility': ep.annual_volatility(benchmark_returns),
        'Sharpe Ratio': ep.sharpe_ratio(benchmark_returns),
        'Max Drawdown': ep.max_drawdown(benchmark_returns),
        'Calmar Ratio': ep.calmar_ratio(benchmark_returns),
        'Sortino Ratio': ep.sortino_ratio(benchmark_returns)
    }
    
    report = pd.DataFrame({'Strategy': kpis, 'Benchmark': benchmark_kpis})
    print("--- Performance Analysis ---")
    print(report)
    
    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ep.cum_returns(returns, starting_value=1).plot(ax=ax, label='Factor Strategy')
    ep.cum_returns(benchmark_returns, starting_value=1).plot(ax=ax, label=f'Benchmark ({BENCHMARK_SYMBOL})')
    
    ax.set_title('Factor Strategy vs. Benchmark Cumulative Returns')
    ax.set_ylabel('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    plt.show()
    
    return report

def main():
    """Main function to run the entire backtest process."""
    print("Loading data...")
    prices = load_data(DATA_PATH)
    if prices is None:
        return
        
    # Ensure benchmark is in the data
    if BENCHMARK_SYMBOL not in prices.columns:
        print(f"Error: Benchmark symbol '{BENCHMARK_SYMBOL}' not found in the data.")
        return

    print("Calculating market timing signal...")
    market_signal = calculate_market_signal(prices, BENCHMARK_SYMBOL, EMA_SHORT, EMA_LONG)
    
    print("Calculating factors...")
    factors = calculate_factors(prices, VOLATILITY_WINDOW, REVERSAL_WINDOW)
    
    print("Running backtest...")
    strategy_returns = run_backtest(
        prices, 
        market_signal, 
        factors, 
        REBALANCE_FREQ, 
        TOP_N_STOCKS, 
        BENCHMARK_SYMBOL,
        TRANSACTION_COST
    )
    
    print("Analyzing performance...")
    benchmark_returns = prices[BENCHMARK_SYMBOL].pct_change().fillna(0)
    analyze_performance(strategy_returns, benchmark_returns)

if __name__ == '__main__':
    main()