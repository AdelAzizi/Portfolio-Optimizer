# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Multi-Factor Portfolio Optimizer & Backtester
# Description: A powerful multi-factor analysis and optimization engine that
#              selects stocks based on a composite score of Value, Momentum,
#              and Low Volatility, then backtests the resulting portfolio.
# Author: Kilo Code, the AI Software Engineer
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
import config

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
    def __init__(self, analysis_data_path: Path, price_data_dir: Path,
                 max_position_size: float, top_n_candidates: int = 15):
        self.analysis_data_path = analysis_data_path
        self.price_data_dir = price_data_dir
        self.max_position_size = max_position_size
        self.top_n_candidates = top_n_candidates
        
        self.analysis_df = None
        self.master_price_df = None
        
        logger.info("🚀 Multi-Factor Optimizer Initialized")

    def _load_data(self):
        """Loads all necessary analysis and price data into memory."""
        logger.info("--- Loading All Required Data for Backtest ---")
        if not self.analysis_data_path.exists():
            raise FileNotFoundError(f"Analysis data not found at {self.analysis_data_path}")
        
        self.analysis_df = pd.read_feather(self.analysis_data_path).set_index('symbol')
        
        all_symbols = self.analysis_df.index.tolist() + ['شاخص کل']
        price_data = {}
        for symbol in all_symbols:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                price_data[symbol] = pd.read_csv(file_path, index_col='date', parse_dates=True)['close']
        
        self.master_price_df = pd.DataFrame(price_data).sort_index()
        self.master_price_df.ffill(inplace=True)
        self.master_price_df.bfill(inplace=True)
        
        logger.info(f"✅ Loaded analysis data for {len(self.analysis_df)} symbols.")
        logger.info(f"✅ Loaded price data for {self.master_price_df.shape[1]} symbols.")

    def screen_stocks(self) -> list:
        """
        Screens and ranks stocks based on a multi-factor model.
    
        Returns:
            list: A list of the top N candidate stock symbols.
        """
        logger.info("--- Screening Stocks with Multi-Factor Model ---")
        logger.info(f"Running experiment with factor weights from config: {config.FACTOR_WEIGHTS}")
        df = self.analysis_df.copy()
    
        # Factor 1: Value (lower is better)
        df['Value_Rank_PE'] = df['P/E'].rank(ascending=True)
        df['Value_Rank_PS'] = df['P/S'].rank(ascending=True)
        df['Value_Score'] = df[['Value_Rank_PE', 'Value_Rank_PS']].mean(axis=1).rank(pct=True)
    
        # Factor 2: Momentum (higher is better)
        df['Momentum_Score'] = df['Momentum_12M'].rank(ascending=True, pct=True)
    
        # Factor 3: Low Volatility (lower is better)
        df['Low_Volatility_Score'] = df['Volatility'].rank(ascending=False, pct=True)
    
        # Composite Score
        df['Composite_Score'] = (
            config.FACTOR_WEIGHTS['Value'] * df['Value_Score'] +
            config.FACTOR_WEIGHTS['Momentum'] * df['Momentum_Score'] +
            config.FACTOR_WEIGHTS['Low_Volatility'] * df['Low_Volatility_Score']
        )
        
        # Select top candidates
        top_candidates = df.sort_values(by='Composite_Score', ascending=False).head(self.top_n_candidates)
        
        logger.info("Top 15 Candidates based on Composite Score:")
        logger.info("\n" + top_candidates[['P/E', 'P/S', 'Momentum_12M', 'Volatility', 'Composite_Score']].to_string())
        
        return top_candidates.index.tolist()

    def _get_portfolio_for_date(self, historical_prices: pd.DataFrame, candidate_symbols: list) -> dict:
        """Runs the optimization for a single point in time for the given candidates."""
        try:
            prices_for_opt = historical_prices[candidate_symbols]
            mu = expected_returns.mean_historical_return(prices_for_opt)
            S = risk_models.CovarianceShrinkage(prices_for_opt).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= self.max_position_size)
            weights = ef.min_volatility()
            return ef.clean_weights()
        except Exception as e:
            logger.error(f"   -> Optimization failed for this period: {e}")
            return None

    def run_backtest(self, years: int = 3, rebalance_freq_days: int = 90):
        """
        Runs a backtest for the defined multi-factor strategy.
        """
        self._load_data()
        
        # 1. Screen stocks based on the full dataset to get candidates
        candidate_symbols = self.screen_stocks()
        
        logger.info(f"\n--- Starting {years}-Year Backtest for Top {len(candidate_symbols)} Candidates ---")
        
        end_date = self.master_price_df.index.max()
        start_date = end_date - pd.DateOffset(days=years * 365)
        
        # 2. Run rolling optimization only on the selected candidates
        historical_prices = self.master_price_df[candidate_symbols + ['شاخص کل']].loc[start_date:end_date]
        
        weights = self._get_portfolio_for_date(historical_prices, candidate_symbols)
        if not weights:
            logger.error("❌ Backtest failed. Could not form a portfolio.")
            return

        # --- Analyze and Plot Results ---
        logger.info("--- Analyzing and Plotting Backtest Results ---")
        
        # Strategy Performance
        portfolio_returns = (historical_prices[list(weights.keys())].pct_change() * pd.Series(weights)).sum(axis=1)
        strategy_cumulative = 100 * (1 + portfolio_returns).cumprod()
        strategy_total_return = (strategy_cumulative.iloc[-1] / strategy_cumulative.iloc[0] - 1) * 100
        strategy_volatility = portfolio_returns.std() * np.sqrt(252) * 100

        # Benchmark Performance
        benchmark_prices = historical_prices['شاخص کل']
        benchmark_cumulative = 100 * (benchmark_prices / benchmark_prices.iloc[0])
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_total_return = (benchmark_cumulative.iloc[-1] / benchmark_cumulative.iloc[0] - 1) * 100
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100

        # Plotting
        plt.figure(figsize=(15, 8))
        plt.plot(strategy_cumulative.index, strategy_cumulative, label="Multi-Factor Strategy", color='blue', linewidth=2)
        plt.plot(benchmark_cumulative.index, benchmark_cumulative, label="Benchmark (شاخص کل)", color='red', linestyle='--')
        plt.title(f"{years}-Year Backtest vs. Benchmark")
        plt.ylabel("Portfolio Value (Initial Value = 100)")
        plt.legend()
        plt.grid(True)
        
        save_path = RESULTS_DIR / 'multifactor_backtest_result.png'
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"✅ Backtest chart saved to '{save_path}'")
        plt.close()

        # Final Stats
        logger.info("\n--- BACKTEST FINAL STATS ---")
        logger.info(f"Strategy Total Return: {strategy_total_return:.2f}%")
        logger.info(f"Strategy Annualized Volatility: {strategy_volatility:.2f}%")
        logger.info("-" * 20)
        logger.info(f"Benchmark Total Return: {benchmark_total_return:.2f}%")
        logger.info(f"Benchmark Annualized Volatility: {benchmark_volatility:.2f}%")
        logger.info("=" * 40)

def main():
    logger.info("="*70)
    logger.info("      Initializing Multi-Factor Optimizer")
    logger.info("="*70)

    optimizer = MultiFactorOptimizer(
        analysis_data_path=CACHE_DIR / 'full_analysis_ready_data.feather',
        price_data_dir=DATA_DIR / 'full_market_data_csvs',
        max_position_size=0.20, # Example: max 20% in any single stock
        top_n_candidates=15
    )
    
    optimizer.run_backtest()

if __name__ == "__main__":
    main()
