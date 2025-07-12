# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Professional Fundamental Data Collector
# Description: A robust, multi-stage pipeline that collects, validates, imputes,
#              and cleans fundamental data for a pre-defined stock universe.
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


# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'fundamental_collector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FundamentalCollector:
    """
    Collects, cleans, and imputes fundamental data for a pre-filtered stock universe.
    """
    def __init__(self, cache_dir: str = 'cache', data_dir: str = 'data'):
        self.cache_dir = PROJECT_ROOT / cache_dir
        self.data_dir = PROJECT_ROOT / data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        self.universe_file = self.cache_dir / 'universe.json'
        self.output_file = self.data_dir / 'fundamental_data.feather'
        
        logger.info(f"Universe file path: {self.universe_file}")
        logger.info(f"Output data path: {self.output_file}")

    def _load_universe(self) -> list:
        """Loads the pre-filtered stock universe from the JSON file."""
        logger.info("--- Stage 1: Loading Pre-filtered Universe ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Universe file not found at '{self.universe_file}'.")
            logger.error("Please run the main downloader script first to generate the universe.")
            raise FileNotFoundError("Universe file is missing.")
        
        with open(self.universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        logger.info(f"✅ Loaded {len(universe)} symbols from universe.json.")
        return universe

    def _load_existing_data(self) -> pd.DataFrame:
        """Loads already processed fundamental data to enable resumability."""
        if self.output_file.exists():
            logger.info(f"Found existing data at {self.output_file}. Loading to resume.")
            return pd.read_feather(self.output_file)
        return pd.DataFrame()

    def _impute_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values using group means and then overall medians."""
        logger.info("--- Stage 4: Imputing Missing Data ---")
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
        logger.info("--- Stage 5: Handling Outliers ---")
        numeric_cols = ['P/E', 'P/S']
        df_clean = df.copy()
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]) and not df_clean[col].isnull().any():
                df_clean[col] = winsorize(df_clean[col], limits=[0.01, 0.01])
                logger.info(f"   - Capped outliers in '{col}' at the 1st and 99th percentiles.")
        return df_clean

    def run(self):
        """Executes the full data collection and cleaning pipeline."""
        logger.info("🚀 Starting Professional Fundamental Data Collector...")
        
        try:
            universe = self._load_universe()
        except FileNotFoundError:
            return # Stop execution if universe file is not found

        # --- Stage 2: Resumable Data Collection ---
        logger.info("--- Stage 2: Collecting New Symbol Data ---")
        existing_df = self._load_existing_data()
        processed_symbols = set(existing_df['symbol']) if not existing_df.empty else set()
        symbols_to_fetch = [s for s in universe if s not in processed_symbols]
        
        if not symbols_to_fetch:
            logger.info("✅ All symbols from universe are already in the local cache. No new data to fetch.")
            # If no new symbols, we can still re-process the existing data
            full_df = existing_df
        else:
            logger.info(f"{len(symbols_to_fetch)} new symbols to process.")
            new_data_list = []
            discarded_count = 0
            for i, symbol in enumerate(symbols_to_fetch):
                try:
                    time.sleep(0.2) # Respectful delay
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
                    logger.info(f"  ({i+1}/{len(symbols_to_fetch)}) Fetched for {symbol}")

                except Exception as e:
                    logger.error(f"  - ❌ Could not fetch data for {symbol}: {e}")
            
            new_data_df = pd.DataFrame(new_data_list)
            full_df = pd.concat([existing_df, new_data_df], ignore_index=True)

        if full_df.empty:
            logger.error("❌ No data available to process after collection phase. Aborting.")
            return

        # --- Stage 3: Create Master DataFrame ---
        logger.info("--- Stage 3: Creating Master DataFrame ---")
        logger.info(f"Successfully created master DataFrame with {len(full_df)} records.")
        
        # --- Post-Processing ---
        df_imputed = self._impute_missing_data(full_df)
        df_clean = self._handle_outliers(df_imputed)

        # --- Stage 6: Save Final Data ---
        logger.info("--- Stage 6: Saving Analysis-Ready Data ---")
        try:
            df_clean.to_feather(self.output_file)
            logger.info(f"💾 Successfully saved final, cleaned dataset to {self.output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save the final dataset: {e}")

        # --- Stage 7: Final Report ---
        logger.info("--- 📊 FINAL DATA QUALITY REPORT 📊 ---")
        logger.info(f"- Symbols loaded from universe file: {len(universe)}")
        logger.info(f"- Symbols already in cache: {len(processed_symbols)}")
        logger.info(f"- New symbols processed: {len(symbols_to_fetch)}")
        if 'discarded_count' in locals():
             logger.info(f"- Symbols discarded (poor quality): {discarded_count}")
        logger.info(f"- Total symbols in final dataset: {len(df_clean)}")
        logger.info(f"- Missing values summary (after imputation):\n{df_clean.isnull().sum()}")
        logger.info("="*40)
        logger.info("✅ Fundamental data pipeline finished successfully.")

if __name__ == "__main__":
    collector = FundamentalCollector()
    collector.run()