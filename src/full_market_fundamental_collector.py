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
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
  

# --- Define Project Root Path and Import Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import FUNDAMENTAL_COLLECTOR

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
    def __init__(self):
        self.cache_dir = PROJECT_ROOT / FUNDAMENTAL_COLLECTOR["CACHE_DIR"]
        self.data_dir = PROJECT_ROOT / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Define paths for input and output using config
        self.universe_file = self.cache_dir / 'universe.json'
        self.output_file = self.cache_dir / FUNDAMENTAL_COLLECTOR["OUTPUT_FILE"]
        self.blacklist_file = self.cache_dir / FUNDAMENTAL_COLLECTOR["BLACKLIST_FILE"]
        
        # Load configuration
        self.cache_validity_hours = FUNDAMENTAL_COLLECTOR["CACHE_VALIDITY_HOURS"]
        self.request_delay_sec = FUNDAMENTAL_COLLECTOR["REQUEST_DELAY_SEC"]
        self.blacklist_expiry_days = FUNDAMENTAL_COLLECTOR["BLACKLIST_EXPIRY_DAYS"]
        self.fundamental_fields = FUNDAMENTAL_COLLECTOR["FUNDAMENTAL_FIELDS"]
        
        logger.info(f"Universe file path: {self.universe_file}")
        logger.info(f"Resumable cache & output file path: {self.output_file}")
        logger.info(f"Blacklist file path: {self.blacklist_file}")
        logger.info(f"Cache validity: {self.cache_validity_hours} hours")
        logger.info(f"Request delay: {self.request_delay_sec} seconds")
        logger.info(f"Blacklist expiry: {self.blacklist_expiry_days} days")

    def _load_full_universe(self) -> list:
        """Loads the full stock universe from the JSON file."""
        logger.info("--- Stage 1: Loading Full Market Universe ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Universe file not found at '{self.universe_file}'.")
            logger.error("Please run the universe_creator.py script first to generate the universe.")
            raise FileNotFoundError(f"Universe file not found at {self.universe_file}")

        with open(self.universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        logger.info(f"✅ Loaded {len(universe)} symbols from universe.json.")
        return universe

    def _load_resumable_cache(self) -> pd.DataFrame:
        """
        Loads the output data from a previous run to resume gracefully.
        The cache validity is configurable via FUNDAMENTAL_COLLECTOR settings.
        """
        cache_file = self.output_file
        if not cache_file.exists():
            logger.info("No previous data file found. Will fetch all data from scratch.")
            return pd.DataFrame()

        last_modified_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if last_modified_hours > self.cache_validity_hours:
            logger.warning(f"Data file is {last_modified_hours:.1f} hours old (older than {self.cache_validity_hours} hours). Discarding and re-fetching all.")
            return pd.DataFrame()

        logger.info(f"Found recent data file at {cache_file} ({last_modified_hours:.1f} hours old). Loading to resume.")
        try:
            # The output is always feather
            return pd.read_feather(cache_file)
        except Exception as e:
            logger.error(f"Could not read resumable cache file {cache_file}: {e}")
            return pd.DataFrame()

    def _impute_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values using group means and then overall medians."""
        logger.info("--- Stage 4: Imputing Missing Data on Combined DataFrame ---")
        numeric_cols = ['P/E', 'P/S', 'EPS']
        df_imputed = df.copy()
        
        # Handle group_name imputation first
        if df_imputed['group_name'].isnull().any():
            missing_before = df_imputed['group_name'].isnull().sum()
            df_imputed['group_name'] = df_imputed['group_name'].fillna('نامشخص')
            imputed_count = missing_before - df_imputed['group_name'].isnull().sum()
            if imputed_count > 0:
                logger.info(f"   - Filled {imputed_count} missing 'group_name' values with 'نامشخص'.")
        
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                missing_before = df_imputed[col].isnull().sum()
                
                # Impute with industry group mean (only for numeric columns)
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
        numeric_cols = ['P/E', 'P/S', 'EPS']  # Extended to include EPS
        df_clean = df.copy()
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                # Only apply winsorize to non-null values
                mask = df_clean[col].notna()
                if mask.any():
                    df_clean.loc[mask, col] = winsorize(df_clean.loc[mask, col], limits=[0.01, 0.01])
                    logger.info(f"   - Capped outliers in '{col}' at the 1st and 99th percentiles.")
        return df_clean

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=0.2, max=10))
    def _fetch_ticker_with_retry(self, symbol):
        """Fetch ticker data with retry mechanism and exponential backoff."""
        return tse.Ticker(symbol)

    def _validate_final_data(self, df: pd.DataFrame):
        """Perform final validation after imputation to ensure data quality."""
        logger.info("Performing final data validation...")
        
        # Check for remaining NaN values in key fields
        key_fields = ['P/E', 'P/S', 'EPS']
        for field in key_fields:
            if field in df.columns:
                nan_count = df[field].isnull().sum()
                total_count = len(df)
                nan_percentage = (nan_count / total_count) * 100 if total_count > 0 else 0
                
                if nan_percentage > 20:  # More than 20% NaN values
                    logger.warning(f"   ⚠️ High percentage of NaN values in '{field}': {nan_percentage:.1f}% ({nan_count}/{total_count})")
                else:
                    logger.info(f"   - '{field}' NaN percentage: {nan_percentage:.1f}% ({nan_count}/{total_count})")
        
        # Additional validation for group_name
        if 'group_name' in df.columns:
            empty_groups = df['group_name'].isnull().sum() + (df['group_name'] == '').sum()
            total_count = len(df)
            empty_percentage = (empty_groups / total_count) * 100 if total_count > 0 else 0
            
            if empty_percentage > 10:  # More than 10% empty group names
                logger.warning(f"   ⚠️ High percentage of empty group names: {empty_percentage:.1f}% ({empty_groups}/{total_count})")
            else:
                logger.info(f"   - Group name empty percentage: {empty_percentage:.1f}% ({empty_groups}/{total_count})")

    def run_collection(self):
        """Executes the full data collection and cleaning pipeline."""
        logger.info("🚀 Starting Fundamental Data Collector...")

        try:
            full_universe = self._load_full_universe()
        except FileNotFoundError:
            return  # Stop execution if universe file is not found

        # --- Stage 2: Smart Resumable Logic & Blacklisting ---
        logger.info("--- Stage 2: Determining Symbols to Process ---")
        processed_df = self._load_resumable_cache()
        cached_symbols = set(processed_df['symbol']) if not processed_df.empty else set()
        
        # Load existing blacklist with expiry handling
        blacklisted_symbols = {}
        if self.blacklist_file.exists():
            try:
                with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                    blacklist_data = json.load(f)
                    # Convert to dictionary format for easier processing
                    for item in blacklist_data:
                        if isinstance(item, dict) and 'symbol' in item:
                            blacklisted_symbols[item['symbol']] = item
                        else:
                            # Handle old format (backward compatibility)
                            blacklisted_symbols[item] = {
                                'symbol': item,
                                'added_at': None,  # No timestamp in old format
                                'reason': 'unknown'
                            }
                # Remove expired symbols
                expired_symbols = []
                current_time = datetime.now()
                for symbol, info in blacklisted_symbols.copy().items():
                    if info['added_at']:
                        try:
                            added_time = datetime.fromisoformat(info['added_at'])
                            if current_time - added_time > timedelta(days=self.blacklist_expiry_days):
                                expired_symbols.append(symbol)
                                del blacklisted_symbols[symbol]
                        except ValueError:
                            # Handle old format timestamps
                            pass
                
                if expired_symbols:
                    logger.info(f"Removed {len(expired_symbols)} expired symbols from blacklist.")
                
                logger.info(f"Loaded {len(blacklisted_symbols)} active symbols from blacklist (total had {len(blacklist_data)} before expiry check).")
            except Exception as e:
                logger.warning(f"Could not load blacklist file: {e}. Starting with an empty blacklist.")
        
        # Filter out already cached and blacklisted symbols (only get symbol names for filtering)
        blacklisted_symbol_names = set(blacklisted_symbols.keys())
        symbols_to_process = [s for s in full_universe if s not in cached_symbols and s not in blacklisted_symbol_names]

        logger.info(f"Found {len(cached_symbols)} symbols in the resumable cache.")
        logger.info(f"Found {len(blacklisted_symbols)} symbols in the blacklist.")
        
        new_data_df = pd.DataFrame()
        if not symbols_to_process:
            logger.info("✅ All symbols from the universe are already in cache or blacklisted. No new data to fetch.")
        else:
            logger.info(f"Identified {len(symbols_to_process)} new or outdated symbols to process.")
            new_data_list = []
            newly_discarded_symbols = [] # To store symbols to add to blacklist (with metadata)
            
            for i, symbol in enumerate(symbols_to_process):
                try:
                    time.sleep(self.request_delay_sec)  # Respectful delay from config
                    ticker = self._fetch_ticker_with_retry(symbol)
                    
                    # Field-Level Validation
                    pe = float(ticker.p_e_ratio) if ticker.p_e_ratio is not None and ticker.p_e_ratio > 0 else np.nan
                    ps = float(ticker.p_s_ratio) if ticker.p_s_ratio is not None and ticker.p_s_ratio > 0 else np.nan
                    
                    # Record-Level Validation
                    if pd.isna(pe) and pd.isna(ps):
                        logger.warning(f"   - Discarding {symbol}: Does not have at least one valid key factor (P/E or P/S). Adding to blacklist.")
                        newly_discarded_symbols.append({
                            'symbol': symbol,
                            'added_at': datetime.now().isoformat(),
                            'reason': 'no_valid_factors'
                        })
                        continue
                    
                    new_data_list.append({
                        'symbol': symbol,
                        'group_name': ticker.group_name if ticker.group_name else 'نامشخص',  # Handle missing group names
                        'P/E': pe,
                        'P/S': ps,
                        'EPS': float(ticker.eps) if ticker.eps is not None else np.nan,
                    })
                    logger.info(f"  ({i+1}/{len(symbols_to_process)}) Fetched for {symbol}")
                
                except RequestException as re:
                    logger.warning(f"  - ⚠️ Network error for {symbol}: {re}. Skipping.")
                except Exception as e:
                    logger.warning(f"  - ⚠️ An unexpected error occurred for {symbol}: {e}. Adding to blacklist.")
                    newly_discarded_symbols.append({
                        'symbol': symbol,
                        'added_at': datetime.now().isoformat(),
                        'reason': str(type(e).__name__)
                    }) # Add to blacklist for any unexpected error with metadata
            
            if new_data_list:
                new_data_df = pd.DataFrame(new_data_list)

            # Save newly discarded symbols to blacklist (merge with existing)
            if newly_discarded_symbols:
                # Combine existing blacklist with new ones
                all_blacklist_items = list(blacklisted_symbols.values()) + newly_discarded_symbols
                with open(self.blacklist_file, 'w', encoding='utf-8') as f:
                    json.dump(all_blacklist_items, f, indent=4, ensure_ascii=False)
                logger.info(f"Added {len(newly_discarded_symbols)} symbols to blacklist. Total blacklisted: {len(all_blacklist_items)}")

        # --- Stage 3: Combine, Process, and Save ---
        logger.info("--- Stage 3: Combining and Processing Data ---")
        
        # Combine initial cache with newly fetched data
        combined_df = pd.concat([processed_df, new_data_df], ignore_index=True)
        
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

        # --- Stage 7: Final Validation ---
        logger.info("--- Stage 7: Final Data Validation ---")
        self._validate_final_data(df_clean)
        
        # --- Stage 8: Final Report ---
        logger.info("--- 📊 FINAL DATA QUALITY REPORT 📊 ---")
        logger.info(f"- Symbols loaded from full universe file: {len(full_universe)}")
        logger.info(f"- Symbols loaded from initial cache: {len(cached_symbols)}")
        logger.info(f"- New symbols processed: {len(symbols_to_process)}")
        logger.info(f"- New symbols discarded (poor quality/errors): {len(newly_discarded_symbols) if 'newly_discarded_symbols' in locals() else 0}")
        logger.info(f"- Total symbols in blacklist: {len(blacklisted_symbols) + (len(newly_discarded_symbols) if 'newly_discarded_symbols' in locals() else 0)}")
        logger.info(f"- Total symbols in final dataset: {len(df_clean)}")
        logger.info(f"- Missing values summary (after imputation):\n{df_clean.isnull().sum()}")
        logger.info("="*50)
        logger.info("✅ Full market fundamental data pipeline finished successfully.")

if __name__ == "__main__":
    collector = FullMarketFundamentalCollector()
    collector.run_collection()