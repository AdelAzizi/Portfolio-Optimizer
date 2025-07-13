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
import hashlib

# --- Import the refactored optimizer ---
from src.optimizer import MultiFactorOptimizer

# --- Setup Logging ---

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CACHE_PATH = PROJECT_ROOT / 'cache' / 'strategy_test_results.csv'

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

def get_file_hash(filepath: Path) -> str:
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def run_strategy_grid_search() -> pd.DataFrame:
    """
    Main function to run the grid search over all defined strategy configurations.
    
    Returns:
        pd.DataFrame: A DataFrame containing the results of all tested strategies.
    """
    # --- Define paths ---
    cache_dir = PROJECT_ROOT / 'cache'
    data_dir = PROJECT_ROOT / 'data'
    analysis_data_path = cache_dir / 'full_analysis_ready_data.feather'
    price_data_dir = data_dir / 'full_market_data_csvs'

    # --- Create Data Hash for Cache Validation ---
    if not analysis_data_path.exists():
        logger.error(f"CRITICAL: Analysis data file not found at {analysis_data_path}. Cannot run tests.")
        return pd.DataFrame()
    current_data_hash = get_file_hash(analysis_data_path)
    logger.info(f"Using data with hash: {current_data_hash}")

    # --- Load or Create Cache ---
    RESULTS_CACHE_PATH.parent.mkdir(exist_ok=True)
    required_columns = [
        'Momentum Period', 'Value Weight', 'Momentum Weight', 'Low Volatility Weight',
        'Total Return', 'Annualized Volatility', 'Annualized Return', 'Sharpe Ratio', 'Data Hash'
    ]
    if RESULTS_CACHE_PATH.exists():
        logger.info(f"Loading existing cache from: {RESULTS_CACHE_PATH}")
        cached_results_df = pd.read_csv(RESULTS_CACHE_PATH)
        # Invalidate cache if the data hash doesn't match
        if 'Data Hash' not in cached_results_df.columns or cached_results_df['Data Hash'].iloc[0] != current_data_hash:
            logger.warning("Data hash mismatch! The underlying data has changed. Invalidating old cache.")
            cached_results_df = pd.DataFrame(columns=required_columns)
    else:
        logger.info("No cache found. Creating a new one.")
        cached_results_df = pd.DataFrame(columns=required_columns)

    valid_weight_combinations = generate_valid_weight_combinations()
    all_results = []
    total_runs = len(valid_weight_combinations) * len(MOMENTUM_PERIODS)
    run_count = 0

    for period in MOMENTUM_PERIODS:
        for weights in valid_weight_combinations:
            run_count += 1
            factor_weights = {'Value': weights[0], 'Momentum': weights[1], 'Low_Volatility': weights[2]}
            logger.info(f"\n--- Evaluating Test {run_count}/{total_runs}: P={period}, W={factor_weights} ---")

            # --- Check Cache First (parameters AND data hash) ---
            cached = cached_results_df[
                (cached_results_df['Momentum Period'] == period) &
                (abs(cached_results_df['Value Weight'] - factor_weights['Value']) < 0.001) &
                (abs(cached_results_df['Momentum Weight'] - factor_weights['Momentum']) < 0.001) &
                (abs(cached_results_df['Low Volatility Weight'] - factor_weights['Low_Volatility']) < 0.001)
            ]

            if not cached.empty:
                logger.info("✅ Found valid cached result. Skipping execution.")
                all_results.append(cached.iloc[0].to_dict())
                continue

            # --- If Not Cached, Run Live ---
            logger.info("No valid cache found. Running live analysis...")
            try:
                optimizer = MultiFactorOptimizer(
                    analysis_data_path=analysis_data_path,
                    price_data_dir=price_data_dir,
                    max_position_size=0.20,
                    factor_weights=factor_weights,
                    momentum_period=period
                )
                
                results = optimizer.run_full_analysis()

                if results and 'performance_summary' in results:
                    performance_stats = results['performance_summary']
                    for key, value in performance_stats.items():
                        if isinstance(value, str) and '%' in value:
                            performance_stats[key] = float(value.strip('%')) / 100
                        else:
                            performance_stats[key] = float(value)

                    result_entry = {
                        'Momentum Period': period,
                        'Value Weight': factor_weights['Value'],
                        'Momentum Weight': factor_weights['Momentum'],
                        'Low Volatility Weight': factor_weights['Low_Volatility'],
                        'Data Hash': current_data_hash,
                        **performance_stats
                    }
                    all_results.append(result_entry)
                    
                    # --- Update and Save Cache Immediately ---
                    new_row_df = pd.DataFrame([result_entry])
                    cached_results_df = pd.concat([cached_results_df, new_row_df], ignore_index=True)
                    cached_results_df.to_csv(RESULTS_CACHE_PATH, index=False, encoding='utf-8-sig')
                    logger.info(f"✅ Success! Sharpe: {performance_stats.get('Sharpe Ratio', 0):.2f}. Cache updated.")
                else:
                    logger.warning("Analysis returned no valid stats.")

            except Exception as e:
                logger.error(f"❌ FAILED for P={period}, W={factor_weights}. Error: {e}", exc_info=False)

    if not all_results:
        logger.error("No results were generated. Halting analysis.")
        return pd.DataFrame()

    # --- Analyze and Report ---
    results_df = pd.DataFrame(all_results)
    if 'Sharpe Ratio' in results_df.columns:
        results_df.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)
    else:
        logger.error("Sharpe Ratio column not found in results. Cannot sort.")
        return pd.DataFrame()

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
    
    return results_df

if __name__ == "__main__":
    run_strategy_grid_search()
