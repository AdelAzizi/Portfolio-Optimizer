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
from src.universe_creator import UniverseCreator
from src.full_market_downloader import FullMarketDownloader
from src.full_market_fundamental_collector import FullMarketFundamentalCollector
from src.full_market_preprocessor import FullMarketDataPreprocessor
from src.strategy_tester import re_evaluate_top_strategies
from src.validator import validate_and_select_best_strategies

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
        # --- STAGE 0: UNIVERSE CREATION ---
        logger.info("\n--- PIPELINE STAGE 0: CREATING INVESTMENT UNIVERSE ---")
        universe_creator = UniverseCreator()
        universe_creator.run()
        logger.info("✅ Universe creation complete.")

        # --- STAGE 1: PRICE DATA COLLECTION ---
        logger.info("\n--- PIPELINE STAGE 1: RUNNING PRICE DATA DOWNLOADER ---")
        price_downloader = FullMarketDownloader()
        price_downloader.run_update()
        logger.info("✅ Price Data Downloader complete.")

        # --- STAGE 2: FUNDAMENTAL DATA COLLECTION ---
        logger.info("\n--- PIPELINE STAGE 2: RUNNING FUNDAMENTAL DATA COLLECTOR ---")
        fundamental_collector = FullMarketFundamentalCollector()
        fundamental_collector.run_collection()
        logger.info("✅ Fundamental Data Collector complete.")

        # --- STAGE 3: DATA PREPROCESSING ---
        logger.info("\n--- PIPELINE STAGE 3: RUNNING DATA PREPROCESSOR ---")
        preprocessor = FullMarketDataPreprocessor()
        preprocessor.run()
        logger.info("✅ Data Preprocessor complete.")

        # --- STAGE 4: RE-EVALUATE TOP 200 STRATEGIES ---
        logger.info("\n--- PIPELINE STAGE 4: RE-EVALUATING TOP 200 STRATEGIES ---")
        top_200_results_df = re_evaluate_top_strategies()
        if top_200_results_df is None or top_200_results_df.empty:
            raise RuntimeError("Strategy re-evaluation failed to produce results.")
        logger.info("✅ Top 200 strategies re-evaluated successfully.")

        # --- STAGE 5: VALIDATE AND SELECT FINAL STRATEGIES ---
        logger.info("\n--- PIPELINE STAGE 5: VALIDATING AND SELECTING FINAL STRATEGIES ---")
        final_strategies = validate_and_select_best_strategies(top_200_results_df)
        
        if final_strategies:
            logger.info("✅ Final strategy validation and selection process complete.")
            # --- STAGE 6: SAVE FINAL RESULTS ---
            logger.info("\n--- PIPELINE STAGE 6: SAVING FINAL STRATEGY RESULTS ---")
            RESULTS_DIR.mkdir(exist_ok=True)
            final_results_path = RESULTS_DIR / 'final_results.json'
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj

            try:
                with open(final_results_path, 'w', encoding='utf-8') as f:
                    json.dump(final_strategies, f, indent=4, ensure_ascii=False, default=convert_numpy)
                logger.info(f"💾 Successfully saved final strategies to {final_results_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save final results to JSON: {e}")
        else:
            logger.error("❌ Final strategy selection failed. No results to save.")

    except Exception as e:
        logger.critical("💥 MASTER PIPELINE FAILED! 💥", exc_info=True)
        # In a real system, this would trigger an alert (e.g., email, Slack)
        
    finally:
        logger.info("\n🏁 MASTER PIPELINE RUN COMPLETE. 🏁")


if __name__ == "__main__":
    run_pipeline()