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
from src.config import RISK_FREE_RATE
from src.strategy_tester import re_evaluate_top_strategies
from src.strategy_selector import StrategySelector


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

def validate_and_select_best_strategies(top_100_df: pd.DataFrame):
    """
    Categorizes strategies by risk, selects the best candidates from each category,
    runs walk-forward validation on them, and selects the final best strategy for each approach.

    Args:
        top_100_df (pd.DataFrame): DataFrame of the top 100 strategies to be validated.
    """
    if top_100_df.empty:
        logger.error("Received an empty DataFrame. Cannot proceed with validation.")
        return

    # --- 1. Use Strategy Selector for Risk Categorization and Candidate Selection ---
    logger.info("\n--- Using Strategy Selector for Risk-Based Categorization and Candidate Selection ---")
    
    strategy_selector = StrategySelector()
    candidate_portfolios = strategy_selector.select_final_candidates(top_100_df, candidates_per_category=5)

    final_results = {}

    # --- 3. Combine and Save the 15 Candidates to a CSV File ---
    all_candidates_list = []
    for approach, df in candidate_portfolios.items():
        df_copy = df.copy()
        df_copy['Risk Profile'] = approach
        all_candidates_list.append(df_copy)
    
    all_candidates_df = pd.concat(all_candidates_list, ignore_index=True)
    
    # Ensure the data directory exists
    (DATA_DIR).mkdir(exist_ok=True)
    candidates_output_path = DATA_DIR / 'top_15_validation_candidates.csv'
    
    try:
        all_candidates_df.to_csv(candidates_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Successfully saved the 15 validation candidates to {candidates_output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save candidates CSV: {e}")

    # --- 4. Run Walk-Forward Validation on All 15 Candidates ---
    for approach, candidates_df in candidate_portfolios.items():
        logger.info(f"\n" + "="*70)
        logger.info(f"      VALIDATING CANDIDATES FOR: {approach.upper()} APPROACH")
        logger.info("="*70)
        
        validation_results_list = []

        for index, candidate in candidates_df.iterrows():
            factor_weights = {
                'Value': candidate['Value Weight'],
                'Momentum': candidate['Momentum Weight'],
                'Low_Volatility': candidate['Low Volatility Weight']
            }
            momentum_period = candidate['Momentum Period']
            top_n = int(candidate['Top N'])
            max_weight = candidate['Max Weight']

            logger.info(f"\n--- Running Validation for Candidate: P={momentum_period}, W={factor_weights}, TopN={top_n}, MaxW={max_weight} ---")
            
            try:
                # Use MultiFactorOptimizer directly for validation run
                optimizer = MultiFactorOptimizer(
                    analysis_data_path=CACHE_DIR / 'full_analysis_ready_data.feather',
                    price_data_dir=DATA_DIR / 'full_market_data_csvs',
                    max_position_size=max_weight,
                    factor_weights=factor_weights,
                    momentum_period=momentum_period,
                    top_n_candidates=top_n
                )
                
                # Perform a 5-year walk-forward validation
                validation_run_results = optimizer.run_full_analysis(years=5)

                if validation_run_results and 'performance_summary' in validation_run_results:
                    # Store the original candidate info within the full results object
                    validation_run_results['original_candidate'] = candidate.to_dict()
                    validation_results_list.append(validation_run_results)
                    logger.info(f"✅ Validation Success! Sharpe: {validation_run_results['performance_summary'].get('Sharpe Ratio', 'N/A')}")
                else:
                    logger.warning("Validation run did not produce a performance summary.")

            except Exception as e:
                logger.error(f"❌ FAILED validation for candidate. Error: {e}", exc_info=True)

        # --- 4. Select the Best Strategy for the Approach Based on Validation ---
        if not validation_results_list:
            logger.error(f"No successful validation runs for {approach} approach. Cannot select a final strategy.")
            continue

        # Select the best strategy based on the Sharpe Ratio in the performance summary
        best_strategy_result = max(
            validation_results_list,
            key=lambda x: float(x.get('performance_summary', {}).get('Sharpe Ratio', -100))
        )

        # Extract original candidate info to create a dedicated strategy configuration dict
        original_candidate = best_strategy_result.pop('original_candidate', {})
        strategy_config = {
            'Momentum Period': original_candidate.get('Momentum Period'),
            'Value Weight': original_candidate.get('Value Weight'),
            'Momentum Weight': original_candidate.get('Momentum Weight'),
            'Low Volatility Weight': original_candidate.get('Low Volatility Weight'),
            'Top N': original_candidate.get('Top N'),
            'Max Weight': original_candidate.get('Max Weight')
        }

        # Re-structure the final result for this approach
        final_results[approach] = {
            "strategy_configuration": strategy_config,
            **best_strategy_result
        }
        
        logger.info(f"\n--- 🏆 BEST STRATEGY FOR {approach.upper()} APPROACH (Post-Validation) ---")
        logger.info(f"Sharpe Ratio: {final_results[approach]['performance_summary'].get('Sharpe Ratio')}")
        logger.info(f"Configuration: {strategy_config}")

    # --- 5. Log Final Selections ---
    logger.info("\n\n" + "="*80)
    logger.info("🎉🎉🎉 FINAL STRATEGY SELECTION COMPLETE 🎉🎉🎉")
    logger.info("="*80)
    for approach, result in final_results.items():
        logger.info(f"\n--- FINAL SELECTION for {approach.upper()} ---")
        logger.info(f"  Validation Sharpe Ratio: {result.get('performance_summary', {}).get('Sharpe Ratio')}")
        logger.info(f"  Original Candidate Config:")
        for key, val in result.get('original_candidate', {}).items():
            logger.info(f"    {key}: {val}")
    
    return final_results


def main():
    """
    Main pipeline execution:
    1. Re-evaluates the top 200 strategies.
    2. Validates and selects the best final strategies based on risk profiles.
    """
    logger.info("="*70)
    logger.info("      STARTING FULL STRATEGY VALIDATION PIPELINE")
    logger.info("="*70)

    # 1. Get the top 200 re-evaluated strategies
    top_200_results_df = re_evaluate_top_strategies()

    # 2. Validate and select the best strategies from the top 200
    if top_200_results_df is not None and not top_200_results_df.empty:
        validate_and_select_best_strategies(top_200_results_df)
    else:
        logger.error("Halting pipeline because re-evaluation of top strategies failed or returned no results.")

    logger.info("\n" + "="*70)
    logger.info("      PIPELINE FINISHED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
