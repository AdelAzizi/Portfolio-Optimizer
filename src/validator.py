# -*- coding: utf-8 -*-

# ============================================================================== 
# Title: Walk-Forward Backtester for Strategy Validation
# Description: A robust, out-of-sample validation engine to test portfolio
#              strategies against lookahead bias and overfitting.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================
from pathlib import Path
import logging
import pandas as pd

# --- Import the refactored optimizer ---
from src.optimizer import MultiFactorOptimizer
from src.config import RISK_FREE_RATE, TOP_N_CANDIDATES


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
        logging.FileHandler(LOGS_DIR / 'validator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyValidator:
    """
    Performs a walk-forward analysis to validate a given portfolio strategy.
    """
    def __init__(self, factor_weights: dict, momentum_period: str, top_n: int):
        self.factor_weights = factor_weights
        self.momentum_period = momentum_period
        self.top_n = top_n
        
        self.analysis_df = None
        self.master_price_df = None
        
        self._load_data()

    def _load_data(self):
        """Loads the master analysis and price data files."""
        logger.info("--- Loading Master Data for Validation ---")
        analysis_data_path = CACHE_DIR / 'full_analysis_ready_data.feather'
        price_data_dir = DATA_DIR / 'full_market_data_csvs'
        
        if not analysis_data_path.exists():
            raise FileNotFoundError(f"Analysis data not found at {analysis_data_path}")
        self.analysis_df = pd.read_feather(analysis_data_path).set_index('symbol')
        
        all_symbols = self.analysis_df.index.tolist() + ['شاخص کل']
        price_data = {}
        for symbol in all_symbols:
            file_path = price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                price_data[symbol] = pd.read_csv(file_path, index_col='date', parse_dates=True)['close']
        
        self.master_price_df = pd.DataFrame(price_data).sort_index().ffill().bfill()
        logger.info("✅ Master data loaded successfully.")

    def run_walk_forward_analysis(self):
        """
        Executes the walk-forward backtest.
        """
        # --- Backtest Parameters ---
        backtest_end_date = self.master_price_df.index.max()
        backtest_start_date = backtest_end_date - pd.DateOffset(years=3)
        training_window_size = pd.DateOffset(years=2)
        rebalance_frequency = 'QS' # Quarterly Start

        rebalance_dates = pd.date_range(start=backtest_start_date, end=backtest_end_date, freq=rebalance_frequency)
        
        all_out_of_sample_returns = []
        
        logger.info(f"\n--- Starting Walk-Forward Analysis ---")
        logger.info(f"Strategy: P={self.momentum_period}, W={self.factor_weights}")
        logger.info(f"Period: {backtest_start_date.date()} to {backtest_end_date.date()}")
        logger.info(f"Rebalancing every: {rebalance_frequency}")

        optimizer = MultiFactorOptimizer(
            analysis_data_path=CACHE_DIR / 'full_analysis_ready_data.feather',
            price_data_dir=DATA_DIR / 'full_market_data_csvs',
            max_position_size=0.20,
            factor_weights=self.factor_weights,
            momentum_period=self.momentum_period,
            top_n_candidates=self.top_n
        )

        for i in range(len(rebalance_dates) - 1):
            train_end = rebalance_dates[i]
            test_start = rebalance_dates[i]
            test_end = rebalance_dates[i + 1]

            # Use only data up to train_end for screening
            optimizer.analysis_df = self.analysis_df.copy()
            optimizer.master_price_df = self.master_price_df.loc[:train_end]
            candidate_symbols = optimizer.screen_stocks()
            if not candidate_symbols:
                logger.warning(f"No candidates found for period ending {train_end.date()}")
                continue
            # Optimize on training data
            weights = optimizer._get_portfolio_for_date(optimizer.master_price_df, candidate_symbols)
            if not weights:
                logger.warning(f"Optimization failed for period ending {train_end.date()}")
                continue
            # Test on out-of-sample period
            test_prices = self.master_price_df.loc[test_start:test_end, list(weights.keys())]
            portfolio_returns = (test_prices.pct_change() * pd.Series(weights)).sum(axis=1)
            all_out_of_sample_returns.append(portfolio_returns)

        if not all_out_of_sample_returns:
            logger.error("No out-of-sample returns generated.")
            return

        # --- Analyze and Plot Final Results ---
        strategy_returns = pd.concat(all_out_of_sample_returns)
        self.analyze_and_plot(strategy_returns, backtest_start_date, backtest_end_date)

    def analyze_and_plot(self, strategy_returns, start_date, end_date):
        """Analyzes the combined out-of-sample returns and plots performance."""
        logger.info("\n--- Analyzing Final Walk-Forward Results ---")
        total_return = (1 + strategy_returns).prod() - 1
        annualized_volatility = strategy_returns.std() * (252 ** 0.5)
        sharpe_ratio = (total_return - RISK_FREE_RATE) / annualized_volatility if annualized_volatility != 0 else 0
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Volatility: {annualized_volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")


def main():
    """Example of running the validator with a top strategy."""
    logger.info("="*70)
    logger.info("      Initializing Walk-Forward Validator")
    logger.info("="*70)

    # Example: Use the best strategy found by the grid search
    top_strategy_weights = {'Value': 0.05, 'Momentum': 0.9, 'Low_Volatility': 0.05}
    top_strategy_period = '12M'

    validator = StrategyValidator(
        factor_weights=top_strategy_weights,
        momentum_period=top_strategy_period,
        top_n=TOP_N_CANDIDATES
    )
    validator.run_walk_forward_analysis()

if __name__ == "__main__":
    main()
