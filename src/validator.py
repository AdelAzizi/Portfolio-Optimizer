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

    def run_walk_forward_analysis(self) -> dict:
        """
        Executes the walk-forward backtest and returns a comprehensive results dictionary.
        """
        logger.info(f"\n--- Starting Walk-Forward Validation for Strategy ---")
        logger.info(f"   Momentum Period: {self.momentum_period}")
        logger.info(f"   Factor Weights: {self.factor_weights}")

        optimizer = MultiFactorOptimizer(
            analysis_data_path=CACHE_DIR / 'full_analysis_ready_data.feather',
            price_data_dir=DATA_DIR / 'full_market_data_csvs',
            max_position_size=0.20, # This could be a parameter
            factor_weights=self.factor_weights,
            momentum_period=self.momentum_period,
            top_n_candidates=self.top_n
        )
        
        # The run_full_analysis method now performs the rolling backtest and returns
        # the exact structure needed for the API. We can call it directly.
        # We pass a shorter period for validation to speed it up.
        validation_results = optimizer.run_full_analysis(years=3)

        if not validation_results:
            logger.error("Validation run failed to produce results.")
            return None
            
        logger.info("✅ Walk-forward validation complete.")
        return validation_results



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
