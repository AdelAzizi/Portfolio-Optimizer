# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Multi-Factor Portfolio Optimizer
# Description: A script that uses a pre-built analysis-ready dataset to perform
#              multi-factor screening, portfolio optimization, and backtesting.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path

# --- Portfolio Optimization & Plotting ---
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
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
    Performs multi-factor screening, optimization, and backtesting.
    """
    def __init__(self, analysis_data_path: Path, price_data_dir: Path, risk_free_rate: float, 
                 max_position_size: float, top_n_candidates: int = 15):
        self.analysis_data_path = analysis_data_path
        self.price_data_dir = price_data_dir
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.top_n_candidates = top_n_candidates
        
        logger.info("🚀 Multi-Factor Optimizer Initialized")
        logger.info(f"   Analysis Data: {self.analysis_data_path}")
        logger.info(f"   Risk-Free Rate: {self.risk_free_rate:.2%}")

    def _load_analysis_data(self) -> pd.DataFrame:
        """Loads the analysis-ready data. This is the only data input for screening."""
        logger.info("--- Stage 1: Loading Analysis-Ready Data ---")
        if not self.analysis_data_path.exists():
            logger.error(f"CRITICAL: Analysis data file not found at '{self.analysis_data_path}'.")
            logger.error("Please run the preprocessor.py script first.")
            raise FileNotFoundError("Analysis-ready data file is missing.")
        
        df = pd.read_feather(self.analysis_data_path).set_index('symbol')
        logger.info(f"✅ Loaded analysis data for {len(df)} symbols.")
        return df

    def screen_stocks(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Selects stocks based on a pure momentum (highest 12M return) strategy."""
        logger.info("--- Stage 2: Screening Stocks ---")
        logger.warning("--- EXPERIMENT: Running PURE MOMENTUM strategy ---")

        # 1. Filter for stocks with positive 12-month momentum
        positive_momentum_stocks = analysis_df[analysis_df['Momentum_12M'] > 0].copy()
        if positive_momentum_stocks.empty:
            logger.error("❌ No stocks with positive 12-month momentum found.")
            return pd.DataFrame()
            
        # 2. Sort by highest momentum
        momentum_ranked_stocks = positive_momentum_stocks.sort_values(by='Momentum_12M', ascending=False)
        
        # 3. Select the top N candidates
        top_candidates = momentum_ranked_stocks.head(self.top_n_candidates)
        
        logger.info(f"\n📈 Top {self.top_n_candidates} Candidates based on Pure Momentum (Highest 12M Return):")
        logger.info(top_candidates[['Momentum_12M', 'Return', 'Volatility']].round(4).to_string())
        return top_candidates

    def _get_prices_for_candidates(self, candidates: list) -> pd.DataFrame:
        """Fetches price history for the top candidate stocks."""
        price_data = {}
        for symbol in candidates:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                price_data[symbol] = pd.read_csv(file_path, index_col='date', parse_dates=True)['close']
        
        price_df = pd.DataFrame(price_data)
        price_df.ffill(inplace=True)
        price_df.bfill(inplace=True)
        return price_df

    def optimize_portfolio(self, candidates_df: pd.DataFrame):
        """Runs optimization for Max Sharpe and Min Volatility portfolios."""
        logger.info("--- Stage 3: Portfolio Optimization ---")
        candidate_symbols = candidates_df.index.tolist()
        price_df = self._get_prices_for_candidates(candidate_symbols)

        try:
            mu = expected_returns.mean_historical_return(price_df)
            S = risk_models.CovarianceShrinkage(price_df).ledoit_wolf()
        except Exception as e:
            logger.error(f"❌ Error calculating returns/covariance: {e}")
            return None, None

        # --- Pre-Optimization Check for Max Sharpe ---
        cleaned_weights_ms = None
        if (mu < self.risk_free_rate).all():
            logger.warning("⚠️ All top candidates have expected returns below the risk-free rate.")
            logger.warning("   Optimization for Max Sharpe is not meaningful. It is recommended to invest in the risk-free asset.")
            logger.warning("   Skipping Max Sharpe portfolio and proceeding with Minimum Volatility.")
        else:
            # Max Sharpe Optimization
            ef_ms = EfficientFrontier(mu, S)
            ef_ms.add_constraint(lambda w: w <= self.max_position_size)
            weights_ms = ef_ms.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights_ms = ef_ms.clean_weights()
            
            logger.info("\n--- Optimal Portfolio (Max Sharpe) ---")
            ef_ms.portfolio_performance(verbose=True, risk_free_rate=self.risk_free_rate)

        # Min Volatility Optimization
        ef_mv = EfficientFrontier(mu, S)
        ef_mv.add_constraint(lambda w: w <= self.max_position_size)
        weights_mv = ef_mv.min_volatility()
        cleaned_weights_mv = ef_mv.clean_weights()
        
        logger.info("\n--- Optimal Portfolio (Minimum Volatility) ---")
        ef_mv.portfolio_performance(verbose=True, risk_free_rate=self.risk_free_rate)
        
        return cleaned_weights_ms, cleaned_weights_mv

    def run_comparative_backtest(self, weights_ms: dict, weights_mv: dict):
        """Runs a 1-year backtest comparing the available portfolios against the benchmark."""
        logger.info("--- Stage 4: Comparative Backtesting ---")
        
        all_symbols = ['شاخص کل']
        if weights_ms:
            all_symbols.extend(weights_ms.keys())
        if weights_mv:
            all_symbols.extend(weights_mv.keys())

        price_data = self._get_prices_for_candidates(list(set(all_symbols)))
        price_data_oneyear = price_data.iloc[-252:]

        def calculate_portfolio_value(weights: dict):
            if not weights:
                return None
            returns = price_data_oneyear[list(weights.keys())].pct_change().dropna()
            portfolio_return = (returns * pd.Series(weights)).sum(axis=1)
            return 100 * (1 + portfolio_return).cumprod()

        value_ms = calculate_portfolio_value(weights_ms)
        value_mv = calculate_portfolio_value(weights_mv)
        
        # Benchmark
        benchmark_prices = price_data_oneyear['شاخص کل']
        benchmark_value = 100 * (benchmark_prices / benchmark_prices.iloc[0])
        
        # Plotting
        plt.figure(figsize=(14, 8))
        if value_ms is not None:
            plt.plot(value_ms, label="Max Sharpe Portfolio", color='blue', linewidth=2)
        if value_mv is not None:
            plt.plot(value_mv, label="Min Volatility Portfolio", color='green', linestyle='--')
        
        plt.plot(benchmark_value, label="Benchmark (شاخص کل)", color='red', linestyle=':')
        plt.title("1-Year Comparative Backtest")
        plt.ylabel("Portfolio Value (Initial Value = 100)")
        plt.legend()
        plt.grid(True)
        
        save_path = RESULTS_DIR / 'backtest_comparison.png'
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"✅ Backtest comparison chart saved to '{save_path}'")
        plt.close()

    def run(self):
        """Orchestrates the entire optimization pipeline."""
        try:
            analysis_df = self._load_analysis_data()
            top_candidates = self.screen_stocks(analysis_df)
            
            if top_candidates.empty:
                logger.warning("No stocks passed the screening. Halting.")
                return

            weights_ms, weights_mv = self.optimize_portfolio(top_candidates)
            
            if weights_ms or weights_mv:
                self.run_comparative_backtest(weights_ms, weights_mv)
            else:
                logger.error("Optimization failed completely, cannot run backtest.")

            logger.info("✅ Full optimization and backtesting pipeline completed successfully.")

        except Exception as e:
            logger.error(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)

def main():
    logger.info("="*70)
    logger.info("      Initializing Multi-Factor Optimizer & Backtester")
    logger.info("="*70)

    optimizer = MultiFactorOptimizer(
        analysis_data_path=CACHE_DIR / 'analysis_ready_data.feather',
        price_data_dir=DATA_DIR / 'tickers_data',
        risk_free_rate=0.35,
        max_position_size=0.30,
        top_n_candidates=15
    )
    optimizer.run()

if __name__ == "__main__":
    main()
