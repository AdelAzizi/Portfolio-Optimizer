# -*- coding: utf-8 -*-

# ============================================================================== 
# Title: Smart Grid Search Tester for Portfolio Strategies
# Description: A script to systematically test and find the best portfolio
#              strategy configurations, including extreme weightings.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

import pandas as pd
import itertools
import logging
from pathlib import Path

# --- Import the refactored optimizer ---
from src.optimizer import MultiFactorOptimizer

# --- Setup Logging ---

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'strategy_tester.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Define the Search Space ---
WEIGHT_POINTS = [0.05, 0.2, 0.4, 0.6, 0.9]
MOMENTUM_PERIODS = ['3M', '6M', '12M']

def generate_valid_weight_combinations():
    """Generates all possible weight combinations that sum to 1.0."""
    all_combinations = list(itertools.product(WEIGHT_POINTS, repeat=3))
    valid_combinations = [
        combo for combo in all_combinations if abs(sum(combo) - 1.0) < 0.01
    ]
    logger.info(f"Generated {len(valid_combinations)} valid weight combinations.")
    return valid_combinations

def run_strategy_grid_search():
    """
    Main function to run the grid search over all defined strategy configurations.
    """
    valid_weight_combinations = generate_valid_weight_combinations()
    all_results = []

    # --- Define paths for the optimizer ---

    cache_dir = PROJECT_ROOT / 'cache'
    data_dir = PROJECT_ROOT / 'data'
    analysis_data_path = cache_dir / 'full_analysis_ready_data.feather'
    price_data_dir = data_dir / 'full_market_data_csvs'

    total_runs = len(valid_weight_combinations) * len(MOMENTUM_PERIODS)
    run_count = 0

    for period in MOMENTUM_PERIODS:
        for weights in valid_weight_combinations:
            run_count += 1
            factor_weights = {'Value': weights[0], 'Momentum': weights[1], 'Low_Volatility': weights[2]}
            
            logger.info(f"\n--- Running Test {run_count}/{total_runs} ---")
            logger.info(f"Period: {period}, Weights: {factor_weights}")

            try:
                optimizer = MultiFactorOptimizer(
                    analysis_data_path=analysis_data_path,
                    price_data_dir=price_data_dir,
                    max_position_size=0.20,
                    factor_weights=factor_weights,
                    momentum_period=period
                )
                
                performance_stats = optimizer.run_analysis()

                if performance_stats:
                    result_entry = {
                        'Momentum Period': period,
                        'Value Weight': factor_weights['Value'],
                        'Momentum Weight': factor_weights['Momentum'],
                        'Low Volatility Weight': factor_weights['Low_Volatility'],
                        **performance_stats
                    }
                    all_results.append(result_entry)
                    logger.info(f"✅ Success! Sharpe Ratio: {performance_stats['Sharpe Ratio']:.2f}")
                else:
                    logger.warning("Analysis returned no stats.")

            except Exception as e:
                logger.error(f"❌ FAILED for Period: {period}, Weights: {factor_weights}. Error: {e}", exc_info=False)

    if not all_results:
        logger.error("No results were generated. Halting analysis.")
        return

    # --- Analyze and Report ---
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)

    logger.info("\n\n" + "="*80)
    logger.info("🎉 STRATEGY GRID SEARCH COMPLETE 🎉")
    logger.info("="*80)
    
    logger.info("\n--- Top 10 Performing Strategies by Sharpe Ratio ---")
    logger.info("\n" + results_df.head(10).to_string())

    # Save full results to CSV

    results_save_path = PROJECT_ROOT / 'results' / 'strategy_test_results.csv'
    results_save_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_save_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n💾 Full strategy results saved to: {results_save_path}")

if __name__ == "__main__":
    run_strategy_grid_search()
