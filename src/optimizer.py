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
from .config import TOP_N_CANDIDATES, RISK_FREE_RATE

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
                 max_position_size: float, factor_weights: dict, momentum_period: str,
                 top_n_candidates: int = TOP_N_CANDIDATES):
        self.analysis_data_path = analysis_data_path
        self.price_data_dir = price_data_dir
        self.max_position_size = max_position_size
        self.top_n_candidates = top_n_candidates
        self.factor_weights = factor_weights
        self.momentum_period = momentum_period
        
        self.analysis_df = None
        self.master_price_df = None
        
        logger.info(f"🚀 Initialized Optimizer with Momentum: {self.momentum_period}, Weights: {self.factor_weights}")

    def _load_data(self):
        """Loads all necessary analysis and price data into memory."""
        if self.analysis_df is not None and self.master_price_df is not None:
            return
        logger.info("--- Loading All Required Data for Analysis ---")
        if not self.analysis_data_path.exists():
            raise FileNotFoundError(f"Analysis data not found at {self.analysis_data_path}")
        
        self.analysis_df = pd.read_feather(self.analysis_data_path).set_index('symbol')
        
        all_symbols = self.analysis_df.index.tolist() + ['شاخص کل']
        price_data = {}
        for symbol in all_symbols:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                price_data[symbol] = pd.read_csv(file_path, index_col='date', parse_dates=True)['close']
        
        self.master_price_df = pd.DataFrame(price_data).sort_index().ffill().bfill()
        logger.info(f"✅ Loaded data for {len(self.analysis_df)} symbols.")

    def screen_stocks(self) -> list:
        """
        Screens and ranks stocks based on a multi-factor model.
    
        Returns:
            list: A list of the top N candidate stock symbols.
        """
        logger.info(f"--- Screening with weights: {self.factor_weights} & period: {self.momentum_period} ---")
        df = self.analysis_df.copy()

        momentum_col = f"Momentum_{self.momentum_period}"
        if momentum_col not in df.columns:
            logger.error(f"CRITICAL: Momentum column '{momentum_col}' not found in analysis data.")
            raise ValueError(f"Missing momentum column: {momentum_col}")

        # Factor 1: Value (lower is better)
        df['Value_Rank_PE'] = df['P/E'].rank(ascending=True)
        df['Value_Rank_PS'] = df['P/S'].rank(ascending=True)
        df['Value_Score'] = df[['Value_Rank_PE', 'Value_Rank_PS']].mean(axis=1).rank(pct=True)

        # Factor 2: Momentum (higher is better)
        df['Momentum_Score'] = df[momentum_col].rank(ascending=True, pct=True)

        # Factor 3: Low Volatility (lower is better)
        df['Low_Volatility_Score'] = df['Volatility'].rank(ascending=False, pct=True)

        # Composite Score
        df['Composite_Score'] = (
            self.factor_weights['Value'] * df['Value_Score'] +
            self.factor_weights['Momentum'] * df['Momentum_Score'] +
            self.factor_weights['Low_Volatility'] * df['Low_Volatility_Score']
        )
        
        # Select top candidates
        top_candidates = df.sort_values(by='Composite_Score', ascending=False).head(self.top_n_candidates)
        
        # logger.info(f"Top {self.top_n_candidates} Candidates based on Composite Score:")
        # logger.info("\n" + top_candidates[['P/E', 'P/S', momentum_col, 'Volatility', 'Composite_Score']].to_string())
        
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

    def run_comparative_backtest(self, portfolio_returns: pd.Series, historical_prices: pd.DataFrame) -> dict:
        """
        Calculates time-series data for strategy vs. benchmark performance.
        This method does not plot, it only returns the data.

        Args:
            portfolio_returns (pd.Series): Daily returns of the strategy portfolio.
            historical_prices (pd.DataFrame): DataFrame with prices, including the benchmark.

        Returns:
            dict: A dictionary containing dates and cumulative values for charting.
        """
        logger.info("--- Running Comparative Backtest Data Generation ---")
        
        # Drop the initial NaN from pct_change() and ensure there are returns
        portfolio_returns = portfolio_returns.dropna()
        if portfolio_returns.empty:
            logger.warning("No valid portfolio returns to generate backtest data.")
            return {"dates": [], "strategy_values": [], "benchmark_values": []}

        # Calculate cumulative returns for the strategy
        strategy_cumulative = (1 + portfolio_returns).cumprod()

        # Calculate cumulative returns for the benchmark
        benchmark_returns = historical_prices['شاخص کل'].pct_change()
        aligned_benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
        benchmark_cumulative = (1 + aligned_benchmark_returns).cumprod()

        # Normalize to start at 100 for charting
        strategy_values = 100 * strategy_cumulative
        benchmark_values = 100 * benchmark_cumulative
        
        # Prepend the starting '100' value for both series for a clean chart start
        start_date = portfolio_returns.index[0] - pd.Timedelta(days=1)
        strategy_values = pd.Series([100], index=[start_date])._append(strategy_values)
        benchmark_values = pd.Series([100], index=[start_date])._append(benchmark_values)

        chart_data = {
            "dates": strategy_values.index.strftime('%Y-%m-%d').tolist(),
            "strategy_values": strategy_values.round(2).tolist(),
            "benchmark_values": benchmark_values.round(2).tolist()
        }
        
        logger.info("✅ Generated backtest time-series data.")
        return chart_data

    def run_full_analysis(self, years: int = 3) -> dict:
        """
        Runs the full screening, optimization, and performance analysis.
        Now returns a comprehensive dictionary including backtest time-series data.

        Returns:
            dict: A dictionary containing optimal_weights, performance_summary, and backtest_data.
        """
        self._load_data()

        candidate_symbols = self.screen_stocks()
        if not candidate_symbols:
            logger.warning("No candidate symbols found after screening. Skipping analysis.")
            return None

        end_date = self.master_price_df.index.max()
        start_date = end_date - pd.DateOffset(days=years * 365)
        historical_prices = self.master_price_df.loc[start_date:end_date]

        weights = self._get_portfolio_for_date(historical_prices, candidate_symbols)
        if not weights:
            logger.error("❌ Optimization failed. Could not form a portfolio.")
            return None

        # --- Performance Calculation ---
        portfolio_returns = (historical_prices[list(weights.keys())].pct_change() * pd.Series(weights)).sum(axis=1)
        
        # --- Performance Calculation ---
        valid_returns = portfolio_returns.dropna()
        if valid_returns.empty:
            logger.warning("Portfolio returns are all NaN or empty. Cannot calculate performance.")
            return None

        # Calculate performance metrics
        total_return = (1 + valid_returns).prod() - 1
        annualized_volatility = valid_returns.std() * np.sqrt(252)
        
        # Correctly calculate annualized return for Sharpe ratio
        num_days = len(valid_returns)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0

        if annualized_volatility == 0:
            sharpe_ratio = np.inf if annualized_return > RISK_FREE_RATE else 0 # Handle zero volatility case
        else:
            sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility

        performance_summary = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Volatility': f"{annualized_volatility:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
        }
        
        # --- Generate Backtest Time-Series Data ---
        backtest_data = self.run_comparative_backtest(portfolio_returns, historical_prices)

        logger.info("✅ Full analysis complete. Returning comprehensive results.")
        return {
            'optimal_weights': weights,
            'performance_summary': performance_summary,
            'backtest_data': backtest_data
        }

def main():
    """Example of running a single default analysis."""
    from .config import FACTOR_WEIGHTS, MOMENTUM_PERIOD
    logger.info("="*70)
    logger.info("      Running Single Default Optimizer Test")
    logger.info("="*70)

    optimizer = MultiFactorOptimizer(
        analysis_data_path=CACHE_DIR / 'full_analysis_ready_data.feather',
        price_data_dir=DATA_DIR / 'full_market_data_csvs',
        max_position_size=0.20,
        factor_weights=FACTOR_WEIGHTS,
        momentum_period=MOMENTUM_PERIOD
    )
    
    results = optimizer.run_full_analysis()
    
    if results:
        logger.info("\n--- STANDALONE RUN FINAL RESULTS ---")
        logger.info("Optimal Weights:")
        for symbol, weight in results['optimal_weights'].items():
            logger.info(f"  - {symbol}: {weight:.2%}")
        logger.info("\nPerformance Metrics:")
        for metric, value in results['performance_metrics'].items():
            logger.info(f"  - {metric}: {value:.2f}")
        logger.info("=" * 40)

if __name__ == "__main__":
    main()
