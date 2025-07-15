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
import time

# --- Import the refactored optimizer ---
from src.optimizer import MultiFactorOptimizer

# --- Setup Logging ---

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CACHE_PATH = PROJECT_ROOT / 'cache' / 'strategy_test_results.csv'
REEVALUATION_CACHE_PATH = PROJECT_ROOT / 'cache' / 'top_300_reevaluation_results.feather'

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

def get_file_hash(filepath: Path) -> str:
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def re_evaluate_top_strategies() -> pd.DataFrame:
    """
    Loads existing strategy test results, filters the top 300 strategies by Sharpe Ratio,
    and then re-evaluates only these top strategies using the MultiFactorOptimizer with 1-year backtest.
    Results are cached for 12 hours.

    Returns:
        pd.DataFrame: A DataFrame containing the top 100 strategies from re-evaluated results.
    """
    # --- Define paths ---
    cache_dir = PROJECT_ROOT / 'cache'
    data_dir = PROJECT_ROOT / 'data'
    analysis_data_path = cache_dir / 'full_analysis_ready_data.feather'
    price_data_dir = data_dir / 'full_market_data_csvs'

    # --- Check for a valid cache first ---
    if REEVALUATION_CACHE_PATH.exists():
        last_modified_hours = (time.time() - REEVALUATION_CACHE_PATH.stat().st_mtime) / 3600
        if last_modified_hours < 12:
            logger.info(f"Found valid cache for re-evaluation results (created {last_modified_hours:.2f} hours ago). Loading from cache.")
            try:
                cached_df = pd.read_feather(REEVALUATION_CACHE_PATH)
                if not cached_df.empty and 'Sharpe Ratio' in cached_df.columns:
                    return cached_df
                else:
                    logger.warning("Cache file is empty or invalid. Re-running evaluation.")
            except Exception as e:
                logger.warning(f"Could not read cache file {REEVALUATION_CACHE_PATH}: {e}. Re-running evaluation.")
        else:
            logger.info(f"Re-evaluation cache is outdated ({last_modified_hours:.2f} hours old). Re-running evaluation.")

    # --- Create Data Hash for Cache Validation ---
    if not analysis_data_path.exists():
        logger.error(f"CRITICAL: Analysis data file not found at {analysis_data_path}. Cannot re-evaluate strategies.")
        return pd.DataFrame()
    current_data_hash = get_file_hash(analysis_data_path)
    logger.info(f"Using data with hash: {current_data_hash}")

    # --- Load existing strategy test results from cache ---
    if not RESULTS_CACHE_PATH.exists():
        logger.error(f"CRITICAL: Strategy test results cache not found at {RESULTS_CACHE_PATH}. Cannot filter strategies.")
        return pd.DataFrame()

    logger.info(f"Loading existing strategy test results from: {RESULTS_CACHE_PATH}")
    full_results_df = pd.read_csv(RESULTS_CACHE_PATH)

    if full_results_df.empty or 'Sharpe Ratio' not in full_results_df.columns:
        logger.error("Loaded strategy results are empty or missing 'Sharpe Ratio' column. Cannot filter.")
        return pd.DataFrame()

    # Filter the top 300 strategies by Sharpe Ratio
    top_300_strategies = full_results_df.sort_values(by='Sharpe Ratio', ascending=False).head(300)
    logger.info(f"Selected top 300 strategies by Sharpe Ratio from {len(full_results_df)} total strategies.")

    all_results_from_top_300 = []
    run_count = 0
    total_runs = len(top_300_strategies)

    # Iterate over these top 300 strategies and re-run the optimizer for them
    for index, strategy_row in top_300_strategies.iterrows():
        run_count += 1
        period = strategy_row['Momentum Period']
        factor_weights = {
            'Value': strategy_row['Value Weight'],
            'Momentum': strategy_row['Momentum Weight'],
            'Low_Volatility': strategy_row['Low Volatility Weight']
        }
        top_n = strategy_row['Top N']
        max_weight = strategy_row['Max Weight']

        param_string = f"P={period}, W={factor_weights}, TopN={top_n}, MaxW={max_weight}"
        logger.info(f"\n🔄 STRATEGY BACKTEST {run_count}/{total_runs} (1-YEAR VALIDATION)")
        logger.info(f"📊 Testing: {param_string}")
        logger.info(f"⏳ Progress: {(run_count/total_runs)*100:.1f}% complete")

        try:
            optimizer = MultiFactorOptimizer(
                analysis_data_path=analysis_data_path,
                price_data_dir=price_data_dir,
                max_position_size=max_weight,
                factor_weights=factor_weights,
                momentum_period=period,
                top_n_candidates=top_n
            )
            
            results = optimizer.run_full_analysis()

            if results and 'performance_summary' in results:
                performance_stats = results['performance_summary']
                # Clean percentage strings
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
                    'Top N': top_n,
                    'Max Weight': max_weight,
                    'Data Hash': current_data_hash,
                    **performance_stats
                }
                all_results_from_top_300.append(result_entry)
                logger.info(f"✅ Success! Sharpe: {performance_stats.get('Sharpe Ratio', 0):.2f}.")
                
                # 💾 INCREMENTAL CACHE SAVE - Save progress every 10 strategies
                if run_count % 10 == 0 and all_results_from_top_300:
                    try:
                        temp_df = pd.DataFrame(all_results_from_top_300)
                        temp_df.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)
                        temp_df.reset_index(drop=True).to_feather(REEVALUATION_CACHE_PATH)
                        logger.info(f"💾 Incremental cache saved: {len(all_results_from_top_300)} results")
                    except Exception as cache_error:
                        logger.warning(f"⚠️ Failed to save incremental cache: {cache_error}")
            else:
                logger.warning("Re-evaluation returned no valid stats.")

        except Exception as e:
            logger.error(f"❌ FAILED re-evaluation for {param_string}. Error: {e}", exc_info=False)

    if not all_results_from_top_300:
        logger.error("No results were generated from the top 300 strategies. Halting analysis.")
        return pd.DataFrame()

    results_df_from_top_300 = pd.DataFrame(all_results_from_top_300)
    if 'Sharpe Ratio' in results_df_from_top_300.columns:
        results_df_from_top_300.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)
    else:
        logger.error("Sharpe Ratio column not found in re-evaluated results. Cannot sort.")
        return pd.DataFrame()

    # Select top 100 strategies for strategy selector
    top_100_strategies = results_df_from_top_300.head(100)

    logger.info("\n\n" + "="*80)
    logger.info("🎉 TOP 300 STRATEGIES RE-EVALUATION COMPLETE 🎉")
    logger.info("="*80)
    
    logger.info("\n--- Top 10 Performing Strategies from Re-evaluation by Sharpe Ratio ---")
    logger.info("\n" + top_100_strategies.head(10).to_string())

    # Save the final results to the new cache file
    try:
        results_df_from_top_300.reset_index(drop=True).to_feather(REEVALUATION_CACHE_PATH)
        logger.info(f"💾 Successfully saved re-evaluation results to cache: {REEVALUATION_CACHE_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to save re-evaluation results to cache: {e}")

    # Return top 100 strategies for strategy selector
    return top_100_strategies

if __name__ == "__main__":
    re_evaluate_top_strategies()
