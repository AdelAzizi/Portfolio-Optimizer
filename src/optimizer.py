# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Multi-Factor Portfolio Optimizer & Rolling Backtester
# Description: A script that validates a portfolio strategy using a robust,
#              rolling-window backtest over a multi-year period.
# Author: Kilo Code, the AI Software 
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path

# --- Portfolio Optimization & Plotting ---
from pypfopt import expected_returns, risk_models, EfficientFrontier, exceptions
import matplotlib.pyplot as plt

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- Define Project Root and Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'
RESULTS_DIR = PROJECT_ROOT / 'results'

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'optimizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiFactorOptimizer:
    """
    Performs and validates a portfolio strategy using a rolling-window backtest.
    """
    def __init__(self, fundamental_data_path: Path, price_data_dir: Path, 
                 max_position_size: float, top_n_candidates: int = 15):
        self.fundamental_data_path = fundamental_data_path
        self.price_data_dir = price_data_dir
        self.max_position_size = max_position_size
        self.top_n_candidates = top_n_candidates
        
        self.fundamental_df = None
        self.master_price_df = None
        
        logger.info("🚀 Rolling Backtester Initialized")

    def _load_data(self):
        """Loads all necessary fundamental and price data into memory."""
        logger.info("--- Loading All Required Data for Backtest ---")
        if not self.fundamental_data_path.exists():
            raise FileNotFoundError(f"Fundamental data not found at {self.fundamental_data_path}")
        
        self.fundamental_df = pd.read_feather(self.fundamental_data_path).set_index('symbol')
        
        all_symbols = self.fundamental_df.index.tolist() + ['شاخص کل']
        price_data = {}
        for symbol in all_symbols:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                price_data[symbol] = pd.read_csv(file_path, index_col='date', parse_dates=True)['close']
        
        self.master_price_df = pd.DataFrame(price_data).sort_index()
        self.master_price_df.ffill(inplace=True)
        self.master_price_df.bfill(inplace=True)
        
        logger.info(f"✅ Loaded fundamental data for {len(self.fundamental_df)} symbols.")
        logger.info(f"✅ Loaded price data for {self.master_price_df.shape[1]} symbols.")

    def _calculate_metrics_for_period(self, price_df_slice: pd.DataFrame) -> pd.DataFrame:
        """Calculates point-in-time metrics to avoid lookahead bias."""
        returns = price_df_slice.pct_change()
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        
        metrics_df = pd.DataFrame({
            'Return': annualized_return,
            'Volatility': annualized_volatility,
        })
        return metrics_df.dropna()

    def _get_portfolio_for_date(self, historical_prices: pd.DataFrame) -> dict:
        """Runs the screening and optimization for a single point in time."""
        # 1. Calculate point-in-time metrics
        metrics_df = self._calculate_metrics_for_period(historical_prices)
        
        # 2. Merge with fundamental data
        # Drop pre-calculated metrics from the loaded fundamental data to avoid column overlap.
        # We must use the point-in-time metrics calculated for the specific historical period.
        fundamental_data_only = self.fundamental_df.drop(
            columns=['Return', 'Volatility', 'Momentum_6M', 'Momentum_12M'],
            errors='ignore'
        )
        analysis_df = fundamental_data_only.join(metrics_df, how='inner')
        
        # 3. Screen for top candidates (Low Volatility / Quality strategy)
        positive_return_stocks = analysis_df[analysis_df['Return'] > 0]
        if positive_return_stocks.empty:
            return None # No valid stocks for this period
        
        stable_stocks = positive_return_stocks.sort_values(by='Volatility', ascending=True)
        top_candidates = stable_stocks.head(self.top_n_candidates)
        candidate_symbols = top_candidates.index.tolist()
        
        # 4. Optimize for Minimum Volatility portfolio
        try:
            prices_for_opt = historical_prices[candidate_symbols]
            mu = expected_returns.mean_historical_return(prices_for_opt)
            S = risk_models.CovarianceShrinkage(prices_for_opt).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= self.max_position_size)
            weights = ef.min_volatility()
            return ef.clean_weights()
        except Exception:
            return None # Optimization failed for this period

    def run_rolling_backtest(self, years: int = 3, rebalance_freq_days: int = 90):
        """
        Runs a rolling window backtest for the defined strategy.
        """
        self._load_data()
        
        logger.info(f"--- Starting {years}-Year Rolling Backtest (Rebalance every {rebalance_freq_days} days) ---")
        
        end_date = self.master_price_df.index.max()
        start_date = end_date - pd.DateOffset(days=years * 365)
        
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=f'{rebalance_freq_days}D')
        
        all_portfolio_returns = []
        
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_rebalance_date = rebalance_dates[i+1]
            
            logger.info(f"Rebalancing for period starting: {current_date.date()}")
            
            # Data available at the time of rebalancing
            historical_prices = self.master_price_df.loc[:current_date]
            
            # Get the optimal portfolio for this period
            weights = self._get_portfolio_for_date(historical_prices)
            
            if not weights:
                logger.warning("   -> Could not form a portfolio for this period. Skipping.")
                continue
            
            # Simulate holding this portfolio for the next period
            forward_prices = self.master_price_df.loc[current_date:next_rebalance_date]
            forward_returns = forward_prices[list(weights.keys())].pct_change().dropna()
            
            period_returns = (forward_returns * pd.Series(weights)).sum(axis=1)
            all_portfolio_returns.append(period_returns)

        if not all_portfolio_returns:
            logger.error("❌ Backtest failed. No returns were generated.")
            return

        # --- Analyze and Plot Results ---
        logger.info("--- Analyzing and Plotting Backtest Results ---")
        portfolio_returns = pd.concat(all_portfolio_returns)
        
        # Strategy Performance
        strategy_cumulative = 100 * (1 + portfolio_returns).cumprod()
        strategy_total_return = (strategy_cumulative.iloc[-1] / strategy_cumulative.iloc[0] - 1) * 100
        strategy_volatility = portfolio_returns.std() * np.sqrt(252) * 100

        # Benchmark Performance
        benchmark_prices = self.master_price_df['شاخص کل'].loc[strategy_cumulative.index]
        benchmark_cumulative = 100 * (benchmark_prices / benchmark_prices.iloc[0])
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_total_return = (benchmark_cumulative.iloc[-1] / benchmark_cumulative.iloc[0] - 1) * 100
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100

        # Plotting
        plt.figure(figsize=(15, 8))
        plt.plot(strategy_cumulative.index, strategy_cumulative, label="Rolling Low-Volatility Strategy", color='blue', linewidth=2)
        plt.plot(benchmark_cumulative.index, benchmark_cumulative, label="Benchmark (شاخص کل)", color='red', linestyle='--')
        plt.title(f"{years}-Year Rolling Backtest vs. Benchmark")
        plt.ylabel("Portfolio Value (Initial Value = 100)")
        plt.legend()
        plt.grid(True)
        
        save_path = RESULTS_DIR / 'rolling_backtest_result.png'
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"✅ Rolling backtest chart saved to '{save_path}'")
        plt.close()

        # Final Stats
        logger.info("\n--- ROLLING BACKTEST FINAL STATS ---")
        logger.info(f"Strategy Total Return: {strategy_total_return:.2f}%")
        logger.info(f"Strategy Annualized Volatility: {strategy_volatility:.2f}%")
        logger.info("-" * 20)
        logger.info(f"Benchmark Total Return: {benchmark_total_return:.2f}%")
        logger.info(f"Benchmark Annualized Volatility: {benchmark_volatility:.2f}%")
        logger.info("=" * 40)

def main():
    logger.info("="*70)
    logger.info("      Initializing Rolling Window Backtester")
    logger.info("="*70)

    optimizer = MultiFactorOptimizer(
        fundamental_data_path=CACHE_DIR / 'analysis_ready_data.feather',
        price_data_dir=DATA_DIR / 'tickers_data',
        max_position_size=0.30,
        top_n_candidates=15
    )
    optimizer.run_rolling_backtest()

if __name__ == "__main__":
    main()
