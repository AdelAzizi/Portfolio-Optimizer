# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Full Market Fundamental Data Collector with Smart Caching
# Description: A robust pipeline to collect, validate, and clean fundamental
#              data for the entire market universe. It intelligently reuses
#              data from a smaller, existing cache to minimize redundant fetching.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import pytse_client as tse
import time
import logging
from pathlib import Path
import json
from scipy.stats.mstats import winsorize
from requests.exceptions import RequestException
 

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'full_fundamental_collector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullMarketFundamentalCollector:
    """
    Collects fundamental data for the full market, using an existing smaller
    dataset as a starting cache to avoid re-fetching data.
    """
    def __init__(self, cache_dir: str = 'cache', data_dir: str = 'data'):
        self.cache_dir = PROJECT_ROOT / cache_dir
        self.data_dir = PROJECT_ROOT / data_dir
        self.data_dir.mkdir(exist_ok=True)

        # Define paths for input, initial cache, and final output
        self.universe_file = self.cache_dir / 'full_universe.json'
        self.initial_cache_file = self.data_dir / 'fundamental_data.feather'
        self.output_file = self.data_dir / 'full_fundamental_data.feather'

        logger.info(f"Full universe file path: {self.universe_file}")
        logger.info(f"Initial cache file path: {self.initial_cache_file}")
        logger.info(f"Final output data path: {self.output_file}")

    def _load_full_universe(self) -> list:
        """Loads the full stock universe from the JSON file."""
        logger.info("--- Stage 1: Loading Full Market Universe ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Full universe file not found at '{self.universe_file}'.")
            logger.error("Please run the `full_market_downloader.py` script first.")
            raise FileNotFoundError("Full universe file is missing.")

        with open(self.universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        logger.info(f"✅ Loaded {len(universe)} symbols from full_universe.json.")
        return universe

    def _load_initial_cache(self) -> pd.DataFrame:
        """Loads the initial, smaller fundamental data cache if it exists."""
        if self.initial_cache_file.exists():
            logger.info(f"Found initial cache at {self.initial_cache_file}. Loading to accelerate process.")
            return pd.read_feather(self.initial_cache_file)
        logger.info("No initial cache file found. Will proceed to fetch all data from scratch.")
        return pd.DataFrame()

    def _impute_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values using group means and then overall medians."""
        logger.info("--- Stage 4: Imputing Missing Data on Combined DataFrame ---")
        numeric_cols = ['P/E', 'P/S', 'EPS']
        df_imputed = df.copy()

        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                missing_before = df_imputed[col].isnull().sum()

                # Impute with industry group mean
                group_means = df_imputed.groupby('group_name')[col].transform('mean')
                df_imputed[col] = df_imputed[col].fillna(group_means)
                imputed_by_group = missing_before - df_imputed[col].isnull().sum()
                if imputed_by_group > 0:
                    logger.info(f"   - Imputed {imputed_by_group} missing '{col}' values using industry group averages.")

                # Impute any remaining with overall median
                if df_imputed[col].isnull().any():
                    remaining_missing_before = df_imputed[col].isnull().sum()
                    col_median = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(col_median)
                    logger.info(f"   - Filled {remaining_missing_before} remaining '{col}' NaNs with overall median ({col_median:.2f}).")

        return df_imputed

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Caps outliers at the 1st and 99th percentiles."""
        logger.info("--- Stage 5: Handling Outliers on Combined DataFrame ---")
        numeric_cols = ['P/E', 'P/S']
        df_clean = df.copy()
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]) and not df_clean[col].isnull().any():
                df_clean[col] = winsorize(df_clean[col], limits=[0.01, 0.01])
                logger.info(f"   - Capped outliers in '{col}' at the 1st and 99th percentiles.")
        return df_clean

    def run(self):
        """Executes the full data collection and cleaning pipeline."""
        logger.info("🚀 Starting Full Market Fundamental Data Collector...")

        try:
            full_universe = self._load_full_universe()
        except FileNotFoundError:
            return  # Stop execution if universe file is not found

        # --- Stage 2: Smart Resumable Logic ---
        logger.info("--- Stage 2: Determining Symbols to Process ---")
        initial_cache_df = self._load_initial_cache()
        cached_symbols = set(initial_cache_df['symbol']) if not initial_cache_df.empty else set()
        symbols_to_process = [s for s in full_universe if s not in cached_symbols]

        logger.info(f"Found {len(cached_symbols)} symbols in the initial cache.")
        
        new_data_df = pd.DataFrame()
        if not symbols_to_process:
            logger.info("✅ All symbols from the full universe are already in the initial cache. No new data to fetch.")
        else:
            logger.info(f"Identified {len(symbols_to_process)} new symbols to process.")
            new_data_list = []
            discarded_count = 0
            for i, symbol in enumerate(symbols_to_process):
                try:
                    time.sleep(0.2)  # Respectful delay
                    ticker = tse.Ticker(symbol)

                    # Field-Level Validation
                    pe = float(ticker.p_e_ratio) if ticker.p_e_ratio is not None and ticker.p_e_ratio > 0 else np.nan
                    ps = float(ticker.p_s_ratio) if ticker.p_s_ratio is not None and ticker.p_s_ratio > 0 else np.nan

                    # Record-Level Validation
                    if pd.isna(pe) and pd.isna(ps):
                        logger.warning(f"   - Discarding {symbol}: Does not have at least one valid key factor (P/E or P/S).")
                        discarded_count += 1
                        continue

                    new_data_list.append({
                        'symbol': symbol,
                        'group_name': ticker.group_name,
                        'P/E': pe,
                        'P/S': ps,
                        'EPS': float(ticker.eps) if ticker.eps is not None else np.nan,
                    })
                    logger.info(f"  ({i+1}/{len(symbols_to_process)}) Fetched for {symbol}")
                
                except RequestException as re:
                    logger.warning(f"  - ⚠️ Network error for {symbol}: {re}. Skipping.")
                except Exception as e:
                    logger.error(f"  - ❌ An unexpected error occurred for {symbol}: {e}")
            
            if new_data_list:
                new_data_df = pd.DataFrame(new_data_list)

        # --- Stage 3: Combine, Process, and Save ---
        logger.info("--- Stage 3: Combining and Processing Data ---")
        
        # Combine initial cache with newly fetched data
        combined_df = pd.concat([initial_cache_df, new_data_df], ignore_index=True)
        
        if combined_df.empty:
            logger.error("❌ No data available to process after collection phase. Aborting.")
            return

        logger.info(f"Successfully created combined DataFrame with {len(combined_df)} total records.")

        # Run Post-Processing on the entire combined DataFrame
        df_imputed = self._impute_missing_data(combined_df)
        df_clean = self._handle_outliers(df_imputed)

        # --- Stage 6: Save Final Data ---
        logger.info("--- Stage 6: Saving Full, Analysis-Ready Data ---")
        try:
            df_clean.to_feather(self.output_file)
            logger.info(f"💾 Successfully saved final, cleaned dataset to {self.output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save the final dataset: {e}")

        # --- Stage 7: Final Report ---
        logger.info("--- 📊 FINAL DATA QUALITY REPORT 📊 ---")
        logger.info(f"- Symbols loaded from full universe file: {len(full_universe)}")
        logger.info(f"- Symbols loaded from initial cache: {len(cached_symbols)}")
        logger.info(f"- New symbols processed: {len(symbols_to_process)}")
        if 'discarded_count' in locals():
             logger.info(f"- New symbols discarded (poor quality): {discarded_count}")
        logger.info(f"- Total symbols in final dataset: {len(df_clean)}")
        logger.info(f"- Missing values summary (after imputation):\n{df_clean.isnull().sum()}")
        logger.info("="*50)
        logger.info("✅ Full market fundamental data pipeline finished successfully.")

if __name__ == "__main__":
    collector = FullMarketFundamentalCollector()
    collector.run()