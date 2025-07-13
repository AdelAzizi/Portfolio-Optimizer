# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Self-Optimizing Strategy Discovery & Validation Pipeline
# Description: A master orchestrator that runs a fully automated system to
#              discover, validate, and serve the best trading strategies.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

# --- Import pipeline components ---
from src.full_market_downloader import FullMarketDownloader
from src.full_market_preprocessor import FullMarketDataPreprocessor
from src.strategy_tester import run_strategy_grid_search
from src.validator import StrategyValidator
from src.config import TOP_N_CANDIDATES

# --- Define Project Root and Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Executes the entire automated strategy discovery and validation pipeline.
    """
    logger.info("="*80)
    logger.info("🚀 LAUNCHING MASTER STRATEGY DISCOVERY & VALIDATION PIPELINE 🚀")
    logger.info("="*80)

    try:
        # --- STAGE 1: DATA PREPARATION ---
        logger.info("\n--- PIPELINE STAGE 1: RUNNING DATA DOWNLOADER ---")
        downloader = FullMarketDownloader()
        downloader.run_update()
        logger.info("✅ Data Downloader complete.")

        # --- STAGE 2: DATA PREPROCESSING ---
        logger.info("\n--- PIPELINE STAGE 2: RUNNING DATA PREPROCESSOR ---")
        preprocessor = FullMarketDataPreprocessor()
        preprocessor.run()
        logger.info("✅ Data Preprocessor complete.")

        # --- STAGE 3: STRATEGY DISCOVERY (GRID SEARCH) ---
        logger.info("\n--- PIPELINE STAGE 3: RUNNING STRATEGY GRID SEARCH ---")
        run_strategy_grid_search()
        strategy_results_path = RESULTS_DIR / 'strategy_test_results.csv'
        if not strategy_results_path.exists():
            raise FileNotFoundError("Strategy grid search did not produce a results file.")
        
        all_tested_strategies_df = pd.read_csv(strategy_results_path)
        logger.info(f"✅ Strategy Grid Search complete. Found {len(all_tested_strategies_df)} potential strategies.")

        # --- STAGE 4: CANDIDATE SELECTION ---
        logger.info("\n--- PIPELINE STAGE 4: SELECTING TOP 3 CANDIDATE STRATEGIES ---")
        top_10_strategies = all_tested_strategies_df.head(10)
        
        # Ensure the columns are numeric for correct sorting
        top_10_strategies['Total Return'] = pd.to_numeric(top_10_strategies['Total Return'])
        top_10_strategies['Annualized Volatility'] = pd.to_numeric(top_10_strategies['Annualized Volatility'])

        aggressive_strategy = top_10_strategies.loc[top_10_strategies['Total Return'].idxmax()]
        defensive_strategy = top_10_strategies.loc[top_10_strategies['Annualized Volatility'].idxmin()]
        balanced_strategy = top_10_strategies.iloc[0] # Highest Sharpe Ratio

        final_candidates = {
            "aggressive": aggressive_strategy.to_dict(),
            "balanced": balanced_strategy.to_dict(),
            "defensive": defensive_strategy.to_dict()
        }
        logger.info("✅ Selected final candidates:")
        for name, params in final_candidates.items():
            logger.info(f"  - {name.upper()}: P={params['Momentum Period']}, W_V={params['Value Weight']:.2f}, W_M={params['Momentum Weight']:.2f}, W_LV={params['Low Volatility Weight']:.2f}")

        # --- STAGE 5: FINAL VALIDATION (WALK-FORWARD ANALYSIS) ---
        logger.info("\n--- PIPELINE STAGE 5: RUNNING WALK-FORWARD VALIDATION ON CANDIDATES ---")
        final_api_output = {}

        for strategy_name, params in final_candidates.items():
            logger.info(f"\n--- Validating '{strategy_name.upper()}' ---")
            
            factor_weights = {
                'Value': params['Value Weight'],
                'Momentum': params['Momentum Weight'],
                'Low_Volatility': params['Low Volatility Weight']
            }
            momentum_period = params['Momentum Period']

            validator = StrategyValidator(
                factor_weights=factor_weights,
                momentum_period=momentum_period,
                top_n=TOP_N_CANDIDATES
            )
            
            # The validator needs to be modified to return the final JSON object
            # For now, we assume it returns a dict that we can use.
            # This part of the code anticipates a future modification to validator.py
            validation_result = validator.run_walk_forward_analysis() # This needs to be adapted
            
            if validation_result:
                 final_api_output[strategy_name] = validation_result
                 logger.info(f"✅ Validation successful for '{strategy_name}'.")
            else:
                 logger.warning(f"⚠️ Validation failed for '{strategy_name}'.")


        # --- STAGE 6: PRODUCE FINAL OUTPUT ---
        logger.info("\n--- PIPELINE STAGE 6: SAVING FINAL CONSOLIDATED JSON ---")
        if final_api_output:
            output_path = RESULTS_DIR / 'final_results.json'
            logger.info(f"💾 Saving final results to: {output_path}")
            
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_api_output, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            logger.info("✅ Successfully saved the final JSON output.")
        else:
            logger.warning("No strategies passed validation. Nothing to save.")

    except Exception as e:
        logger.critical("💥 MASTER PIPELINE FAILED! 💥", exc_info=True)
        # In a real system, this would trigger an alert (e.g., email, Slack)
        
    finally:
        logger.info("\n🏁 MASTER PIPELINE RUN COMPLETE. 🏁")


if __name__ == "__main__":
    run_pipeline()