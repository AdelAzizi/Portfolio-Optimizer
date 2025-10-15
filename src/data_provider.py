# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Centralized Data Provider
# Description: A unified class to handle all data acquisition tasks including
#              universe creation, price data downloading, and fundamental data collection.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import pytse_client as tse
import numpy as np
import time
import logging
import json
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import traceback
from datetime import datetime, timedelta
from scipy.stats.mstats import winsorize
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Project-Specific Imports ---
from .config import PROJECT_ROOT, UNIVERSE_CREATOR, FULL_MARKET_DOWNLOADER, FUNDAMENTAL_COLLECTOR

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'data_provider.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataProvider:
    """
    A centralized data provider that handles all data acquisition tasks:
    - Universe creation and management
    - Price data downloading and incremental updates
    - Fundamental data collection and cleaning
    """
    
    def __init__(self):
        """
        Initializes the DataProvider using settings from the central config file.
        """
        # --- Load paths and settings from config ---
        self.data_dir = PROJECT_ROOT / FULL_MARKET_DOWNLOADER["DATA_DIR"]
        self.cache_dir = PROJECT_ROOT / UNIVERSE_CREATOR["CACHE_DIR"]
        
        # Universe related paths
        self.universe_file = self.cache_dir / UNIVERSE_CREATOR["UNIVERSE_FILENAME"]
        
        # Price data related paths
        self.downloader_blacklist_file = self.cache_dir / FULL_MARKET_DOWNLOADER["BLACKLIST_FILENAME"]
        self.benchmark_symbol = FULL_MARKET_DOWNLOADER["BENCHMARK_SYMBOL"]
        self.api_delay = FULL_MARKET_DOWNLOADER["API_DELAY_SECONDS"]
        
        # Fundamental data related paths
        self.fundamental_output_file = self.cache_dir / FUNDAMENTAL_COLLECTOR["OUTPUT_FILE"]
        self.fundamental_blacklist_file = self.cache_dir / FUNDAMENTAL_COLLECTOR["BLACKLIST_FILE"]
        self.fundamental_fields = FUNDAMENTAL_COLLECTOR["FUNDAMENTAL_FIELDS"]
        
        # Processing logs
        self.processing_log_file = self.cache_dir / "processing_log.json"
        self.fundamental_processing_log_file = self.cache_dir / "fundamental_processing_log.json"
        
        # Setup directories
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.blacklist = self._load_blacklist()
        self.fundamental_blacklist = self._load_fundamental_blacklist()
        self.failure_counts = {}
        self.max_consecutive_failures = 3
        self.processing_log = self._load_processing_log()
        self.fundamental_processing_log = self._load_fundamental_processing_log()
        
        logger.info("DataProvider initialized with the following settings:")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Universe file: {self.universe_file}")
        logger.info(f"Price data blacklist file: {self.downloader_blacklist_file}")
        logger.info(f"Fundamental data blacklist file: {self.fundamental_blacklist_file}")
        logger.info(f"Fundamental output file: {self.fundamental_output_file}")

    def _load_blacklist(self) -> Dict[str, str]:
        """
        Loads the blacklist with timestamps for price data downloader.
        Structure: {"symbol": "YYYY-MM-DD"}
        """
        if self.downloader_blacklist_file.exists():
            try:
                with open(self.downloader_blacklist_file, 'r', encoding='utf-8') as f:
                    blacklist_data = json.load(f)
                
                # Convert old format (list) to new format (dict with timestamps)
                if isinstance(blacklist_data, list):
                    # Old format - convert to new format with current timestamp
                    new_format = {}
                    for symbol in blacklist_data:
                        new_format[symbol] = datetime.now().isoformat()
                    return new_format
                elif isinstance(blacklist_data, dict):
                    # New format - check for expired entries
                    current_time = datetime.now()
                    blacklist_expiry = timedelta(days=30)  # 30 days expiry
                    
                    # Remove expired entries
                    active_blacklist = {}
                    for symbol, timestamp in blacklist_data.items():
                        try:
                            added_time = datetime.fromisoformat(timestamp)
                            if current_time - added_time <= blacklist_expiry:
                                active_blacklist[symbol] = timestamp
                        except ValueError:
                            # Invalid timestamp format, keep the entry
                            active_blacklist[symbol] = timestamp
                    
                    if len(active_blacklist) != len(blacklist_data):
                        logger.info(f"Removed {len(blacklist_data) - len(active_blacklist)} expired entries from blacklist.")
                    
                    return active_blacklist
            except (json.JSONDecodeError, TypeError):
                logger.warning("Blacklist file is corrupted or empty. Starting with an empty list.")
                return {}
        return {}

    def _save_blacklist(self):
        """Saves the current blacklist with timestamps."""
        with open(self.downloader_blacklist_file, 'w', encoding='utf-8') as f:
            json.dump(self.blacklist, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(self.blacklist)} symbols to the price data blacklist.")

    def _load_fundamental_blacklist(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the fundamental data blacklist with metadata.
        Structure: {"symbol": {"added_at": "YYYY-MM-DD", "reason": "reason"}}
        """
        if self.fundamental_blacklist_file.exists():
            try:
                with open(self.fundamental_blacklist_file, 'r', encoding='utf-8') as f:
                    blacklist_data = json.load(f)
                
                # Convert to dictionary format for easier processing
                blacklist_dict = {}
                for item in blacklist_data:
                    if isinstance(item, dict) and 'symbol' in item:
                        blacklist_dict[item['symbol']] = item
                    else:
                        # Handle old format (backward compatibility)
                        blacklist_dict[item] = {
                            'symbol': item,
                            'added_at': datetime.now().isoformat(),
                            'reason': 'unknown'
                        }
                
                # Remove expired symbols
                expired_symbols = []
                current_time = datetime.now()
                blacklist_expiry_days = FUNDAMENTAL_COLLECTOR["BLACKLIST_EXPIRY_DAYS"]
                blacklist_expiry = timedelta(days=blacklist_expiry_days)
                
                for symbol, info in blacklist_dict.copy().items():
                    if info.get('added_at'):
                        try:
                            added_time = datetime.fromisoformat(info['added_at'])
                            if current_time - added_time > blacklist_expiry:
                                expired_symbols.append(symbol)
                                del blacklist_dict[symbol]
                        except ValueError:
                            # Handle old format timestamps
                            pass
                
                if expired_symbols:
                    logger.info(f"Removed {len(expired_symbols)} expired symbols from fundamental blacklist.")
                
                return blacklist_dict
            except Exception as e:
                logger.warning(f"Could not load fundamental blacklist file: {e}. Starting with an empty blacklist.")
                return {}
        return {}

    def _save_fundamental_blacklist(self):
        """Saves the fundamental data blacklist with metadata."""
        with open(self.fundamental_blacklist_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.fundamental_blacklist.values()), f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(self.fundamental_blacklist)} symbols to the fundamental data blacklist.")

    def _load_processing_log(self) -> Dict[str, str]:
        """Loads the processing log that tracks when each symbol was last processed."""
        if self.processing_log_file.exists():
            try:
                with open(self.processing_log_file, 'r', encoding='utf-8') as f:
                    processing_log = json.load(f)
                    logger.info(f"Loaded processing log for {len(processing_log)} symbols.")
                    return processing_log
            except (json.JSONDecodeError, TypeError):
                logger.warning("Processing log file is corrupted or empty. Starting with an empty log.")
                return {}
        return {}

    def _save_processing_log(self):
        """Saves the current processing log to a JSON file."""
        with open(self.processing_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved processing log for {len(self.processing_log)} symbols.")

    def _load_fundamental_processing_log(self) -> Dict[str, str]:
        """Loads the fundamental processing log."""
        if self.fundamental_processing_log_file.exists():
            try:
                with open(self.fundamental_processing_log_file, 'r', encoding='utf-8') as f:
                    processing_log = json.load(f)
                    logger.info(f"Loaded fundamental processing log for {len(processing_log)} symbols.")
                    return processing_log
            except (json.JSONDecodeError, TypeError):
                logger.warning("Fundamental processing log file is corrupted or empty. Starting with an empty log.")
                return {}
        return {}

    def _save_fundamental_processing_log(self):
        """Saves the fundamental processing log to a JSON file."""
        with open(self.fundamental_processing_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.fundamental_processing_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved fundamental processing log for {len(self.fundamental_processing_log)} symbols.")

    def _increment_failure_count(self, symbol: str) -> bool:
        """Increment the failure count for a symbol and check if it should be blacklisted."""
        self.failure_counts[symbol] = self.failure_counts.get(symbol, 0) + 1
        current_failures = self.failure_counts[symbol]
        
        if current_failures >= self.max_consecutive_failures:
            logger.warning(f"🔴 Blacklisting {symbol} after {current_failures} consecutive failures.")
            self.blacklist[symbol] = datetime.now().isoformat()
            # Remove from failure counts since it's now blacklisted
            if symbol in self.failure_counts:
                del self.failure_counts[symbol]
            return True
        else:
            logger.warning(f"⚠️  {symbol} failed, {current_failures}/{self.max_consecutive_failures} failures (will blacklist at {self.max_consecutive_failures}).")
            return False

    def _reset_failure_count(self, symbol: str):
        """Reset the failure count for a symbol after successful download."""
        if symbol in self.failure_counts:
            del self.failure_counts[symbol]

    def _update_processing_log(self, symbol: str):
        """Update the processing log with the current timestamp for a symbol."""
        self.processing_log[symbol] = datetime.now().isoformat()
        # Save periodically to preserve progress
        if len(self.processing_log) % 10 == 0:  # Save every 10 updates
            self._save_processing_log()

    def _update_fundamental_processing_log(self, symbol: str):
        """Update the fundamental processing log with the current timestamp for a symbol."""
        self.fundamental_processing_log[symbol] = datetime.now().isoformat()
        # Save periodically to preserve progress
        if len(self.fundamental_processing_log) % 10 == 0:  # Save every 10 updates
            self._save_fundamental_processing_log()

    def _update_universe(self):
        """
        Fetches market-wide stats, applies quantitative filters from config, 
        and saves the resulting stock universe.
        """
        logger.info("Starting efficient universe creation process using configurations...")
        
        # Check if cache is valid using validity period from config
        if self.universe_file.exists():
            last_modified_hours = (time.time() - self.universe_file.stat().st_mtime) / 3600
            if last_modified_hours < UNIVERSE_CREATOR["CACHE_VALIDITY_HOURS"]:
                logger.info(f"Found valid cache for universe (created {last_modified_hours:.2f} hours ago). Loading from cache.")
                try:
                    with open(self.universe_file, 'r', encoding='utf-8') as f:
                        cached_universe = json.load(f)
                    if cached_universe:
                        logger.info(f"✅ Successfully loaded {len(cached_universe)} symbols from cache.")
                        return
                    else:
                        logger.warning("Cache file is empty. Re-creating universe.")
                except Exception as e:
                    logger.warning(f"Could not read cache file: {e}. Re-creating universe.")
            else:
                logger.info(f"Universe cache is outdated ({last_modified_hours:.2f} hours old). Re-creating universe.")
        
        try:
            # 1. Fetch all market-wide statistics at once with retry mechanism
            logger.info("Fetching market-wide stats using pytse_client.get_stats() with retry mechanism...")
            
            # Retry mechanism for network issues
            for attempt in range(3):
                try:
                    market_stats_df = tse.get_stats(to_csv=False)
                    if market_stats_df is not None and not market_stats_df.empty:
                        break
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Failed to fetch market stats. Retrying in 5 seconds...")
                        time.sleep(5)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Error fetching market stats: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
            else:
                logger.error("Failed to fetch market stats after 3 attempts. The returned DataFrame is empty.")
                return
            
            logger.info(f"Successfully fetched stats for {len(market_stats_df)} symbols from pytse_client.")

            # Convert columns to numeric for filtering, coercing errors to NaN
            numeric_cols = ['volume_of_trans', 'val_company_last_day']
            for col in numeric_cols:
                if col in market_stats_df.columns:
                    market_stats_df[col] = pd.to_numeric(market_stats_df[col], errors='coerce')
            
            # Drop rows where conversion failed for essential columns
            market_stats_df.dropna(subset=numeric_cols, inplace=True)
            logger.info(f"After cleaning non-numeric data, {len(market_stats_df)} symbols remain.")

            # Apply filters step by step with logging
            initial_count = len(market_stats_df)
            
            # a. Market Type Filter
            market_flow_types = UNIVERSE_CREATOR["FILTERS"]["MARKET_FLOW_TYPES"]
            filtered_df = market_stats_df[market_stats_df['flow'].isin(market_flow_types)].copy()
            logger.info(f"After market type filter (Bourse/Fara Bourse): {len(filtered_df)} stocks remaining.")

            # b. Liquidity Filter
            min_liquidity = UNIVERSE_CREATOR["FILTERS"]["MIN_LIQUIDITY"]
            filtered_df = filtered_df[filtered_df['volume_of_trans'] > min_liquidity]
            logger.info(f"After liquidity filter (>{min_liquidity} volume): {len(filtered_df)} stocks remaining.")

            # c. Size Filter
            min_market_cap = UNIVERSE_CREATOR["FILTERS"]["MIN_MARKET_CAP"]
            filtered_df = filtered_df[filtered_df['val_company_last_day'] > min_market_cap]
            logger.info(f"After size filter (>{min_market_cap} market cap): {len(filtered_df)} stocks remaining.")

            # d. State Filter - Only symbols with 'state' of 'مجاز' or 'مجاز-محفوظ'
            if 'state' in filtered_df.columns:
                allowed_states = ['مجاز', 'مجاز-محفوظ']
                filtered_df = filtered_df[filtered_df['state'].isin(allowed_states)]
                logger.info(f"After state filter (only allowed states): {len(filtered_df)} stocks remaining.")

            # e. Priority Rights Filter - Remove symbols ending with 'ح'
            if 'symbol' in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df['symbol'].str.endswith('ح')]
                logger.info(f"After priority rights filter (remove symbols ending with 'ح'): {len(filtered_df)} stocks remaining.")

            # f. Minimum Trading Days Filter - Only if the column exists
            min_trading_days = UNIVERSE_CREATOR.get('MIN_TRADING_DAYS', 30)
            if 'days_of_transaction' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['days_of_transaction'] >= min_trading_days]
                logger.info(f"After minimum trading days filter (>{min_trading_days} days): {len(filtered_df)} stocks remaining.")
            else:
                logger.warning("Column 'days_of_transaction' not found in data. Skipping minimum trading days filter.")

            # 3. Extract and Save the Final Universe
            if filtered_df.empty:
                logger.warning("No stocks passed all filters. The universe will be empty.")
                final_universe = []
            else:
                final_universe = filtered_df['symbol'].tolist()
                final_universe.sort()  # Sort alphabetically as requested
                logger.info(f"Final universe contains {len(final_universe)} stocks.")

            with open(self.universe_file, 'w', encoding='utf-8') as f:
                json.dump(final_universe, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Successfully saved universe of {len(final_universe)} stocks to {self.universe_file}")

        except KeyError as e:
            logger.error(f"A required column is missing from the fetched data: {e}")
            logger.error("Please check the column names provided by the 'pytse_client' library's get_stats() function.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during universe creation: {e}", exc_info=True)
            raise

    def _check_data_integrity(self, symbol: str) -> bool:
        """
        Check data integrity by comparing the last adjusted price with unadjusted price.
        Returns True if data is consistent (adjusted), False if needs full re-download.
        """
        file_path = self.data_dir / f"{symbol}.csv"
        if not file_path.exists():
            return True  # No existing data, can proceed with download
        
        try:
            # Read the last row of existing data
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            if df.empty:
                return True  # Empty file, can proceed
            
            last_date = df.index[-1].strftime('%Y-%m-%d')
            
            # Get unadjusted price for the same date
            ticker = tse.Ticker(symbol, adjust=False)
            hist = ticker.history(start_date=last_date)
            
            if not hist.empty:
                last_existing_price = df.iloc[-1]['close'] if 'close' in df.columns else df.iloc[-1].iloc[0]
                last_fetched_price = hist.iloc[0]['close'] if 'close' in hist.columns else hist.iloc[0].iloc[0]
                
                # Check if prices match (with small tolerance for floating point differences)
                if abs(last_existing_price - last_fetched_price) / last_fetched_price < 0.001:
                    logger.warning(f"⚠️  Data integrity check failed for {symbol}. Existing data appears to be unadjusted. Marking for full re-download.")
                    return False
                else:
                    logger.debug(f"✅ Data integrity check passed for {symbol}.")
                    return True
            else:
                logger.debug(f"✅ No recent price data to check for {symbol}.")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Could not perform data integrity check for {symbol}: {e}")
            return True  # If check fails, proceed with download

    def _update_price_data_for_symbol(self, symbol: str, start_date: Optional[str] = None):
        """
        Update price data for a single symbol with data integrity checking.
        """
        time.sleep(self.api_delay)  # Be respectful to the API
        file_path = self.data_dir / f"{symbol}.csv"
        
        try:
            logger.info(f"⬇️  Processing {symbol} (start_date: {start_date})...")
            
            # Check data integrity before download
            if start_date and not self._check_data_integrity(symbol):
                # If integrity check fails, mark for full re-download
                logger.info(f"🔄 {symbol} marked for full re-download due to integrity issues.")
                start_date = None  # Reset start_date to trigger full download

            # Fetch data (either full or incremental)
            if symbol == self.benchmark_symbol:
                ticker = tse.FinancialIndex(symbol)
            else:
                ticker = tse.Ticker(symbol, adjust=True)
            
            # Fetch with retry mechanism
            hist = self._fetch_with_retry(ticker, start_date)
            
            if hist is not None and not hist.empty:
                if start_date and len(hist) > 1:
                    # Append new data (excluding the first row which is a duplicate of the last known date)
                    hist.iloc[1:].to_csv(file_path, mode='a', header=False, index=True, encoding='utf-8')
                    logger.info(f"💾 Appended {len(hist)-1} new rows for {symbol}.")
                elif start_date and len(hist) <= 1:
                    # This case means no new data was found since the last update
                    logger.info(f"✅ No new data for {symbol} since {start_date}. Already up-to-date.")
                else:  # This is for full download (start_date is None)
                    # Full download
                    hist.to_csv(file_path, index=True, encoding='utf-8')
                    logger.info(f"💾 Performed full download for {symbol} ({len(hist)} rows).")
                logger.debug(f"Successfully updated {symbol}")
            else:
                logger.info(f"✅ No new data for {symbol}. Already up-to-date.")
                
            # After successful processing, touch the file to update its modification time
            file_path.touch()
            
        except IndexError as ie:
            if 'single positional indexer is out-of-bounds' in str(ie):
                should_blacklist = self._increment_failure_count(symbol)
                if should_blacklist:
                    logger.info(f"Symbol {symbol} added to blacklist due to repeated failures.")
            else:
                logger.error(f"❌ An unexpected indexing error occurred for {symbol}: {ie}.")
                logger.debug(f"Full traceback for {symbol}: {traceback.format_exc()}")
                # Increment failure count for other index errors too
                self._increment_failure_count(symbol)
        except Exception as e:
            logger.error(f"❌ A critical error occurred for {symbol}: {e}.")
            logger.debug(f"Full traceback for {symbol}: {traceback.format_exc()}")
            
            # Check if this is a specific error that should trigger blacklisting
            error_msg = str(e).lower()
            if ('dataframe' in error_msg and 'callable' in error_msg) or 'no historical data' in error_msg:
                should_blacklist = self._increment_failure_count(symbol)
                if should_blacklist:
                    logger.info(f"Symbol {symbol} added to blacklist due to critical error.")
            else:
                # For other errors, increment failure count but don't immediately blacklist
                self._increment_failure_count(symbol)
        else:
            # If successful, reset failure count and update processing log
            self._reset_failure_count(symbol)
            self._update_processing_log(symbol)
            logger.debug(f"✅ Successfully processed {symbol}, reset failure count and updated processing log.")

    def _fetch_with_retry(self, ticker, start_date, max_retries=3):
        """Retry function for API calls."""
        for attempt in range(max_retries):
            try:
                # Check if ticker.history is callable (function) or property
                if callable(ticker.history):
                    # If it's callable, call it as a function
                    if start_date:
                        hist = ticker.history(start_date=start_date)
                    else:
                        hist = ticker.history()
                else:
                    # If it's a property, access it directly and then filter by date
                    hist = ticker.history
                    if start_date and hasattr(hist, 'loc'):
                        # Filter the history DataFrame by start_date if provided
                        try:
                            hist = hist[hist.index >= start_date]
                        except Exception as filter_error:
                            logger.warning(f"Could not filter history by start_date {start_date}: {filter_error}, returning full history")
                
                # Check if hist is actually a DataFrame or if there's an issue
                if hist is not None and hasattr(hist, 'empty'):
                    # It's a proper DataFrame
                    if not hist.empty and hasattr(hist, 'index') and not hist.index.empty:
                        logger.debug(f"DataFrame date range: {hist.index.min()} to {hist.index.max()}")
                    return hist
                else:
                    logger.warning(f"⚠️  Received unexpected object type for ticker: {type(hist)}, attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise ValueError(f"Received unexpected object type instead of DataFrame for ticker: {type(hist)}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

    def _get_last_date_batch(self, symbols: List[str]) -> Dict[str, Optional[str]]:
        """
        Batch check of last dates for all symbols to avoid repeated file I/O operations.
        Returns a dictionary mapping symbol to its last date or None if not recent.
        """
        last_dates = {}
        
        for symbol in symbols:
            file_path = self.data_dir / f"{symbol}.csv"
            if file_path.exists():
                try:
                    # Read the CSV file - when saved with index=True, the index contains the dates
                    # Specify the date format explicitly to avoid warnings and improve performance
                    df = pd.read_csv(file_path, parse_dates=[0], index_col=0)  # Parse the first column as date
                    if not df.empty:
                        # Ensure the index is datetime
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        last_date = df.index.max()
                        # Check if data is recent (within last 7 days)
                        current_date = pd.Timestamp.now().normalize()
                        days_diff = (current_date - last_date).days
                        
                        # Enhanced logging for debugging
                        logger.debug(f"Symbol: {symbol}, Last Date: {last_date}, Current Date: {current_date}, Days Diff: {days_diff}")
                        logger.debug(f"Date index range for {symbol}: {df.index.min()} to {df.index.max()}")
                        
                        # Consider data recent if it's from today or within the last 3 days
                        if days_diff <= 3:
                            # Data is recent, so we return the last date but mark that it doesn't need update
                            last_dates[symbol] = last_date.strftime('%Y-%m-%d')
                            logger.debug(f"✅ {symbol} considered recent (last date: {last_dates[symbol]}), skipping download")
                        else:
                            # Data is old, we need the last date for an incremental update
                            last_dates[symbol] = last_date.strftime('%Y-%m-%d')
                            logger.debug(f"🔄 {symbol} needs update (last date: {last_dates[symbol]}, {days_diff} days old)")
                    else:
                        last_dates[symbol] = None
                        logger.debug(f"⚠️ {symbol} file exists but is empty")
                except Exception as e:
                    last_dates[symbol] = None  # Error reading file, treat as needs update
                    logger.debug(f"⚠️ Error reading {symbol} file: {e}")
            else:
                last_dates[symbol] = None  # File doesn't exist, needs full download
                logger.debug(f"📁 {symbol} file doesn't exist, needs full download")
        
        return last_dates

    def _update_price_data(self):
        """
        Runs the main incremental update process for price data with optimized caching.
        """
        logger.info("🚀 Starting Optimized Price Data Downloader...")

        # STAGE 1: LOAD UNIVERSE
        logger.info(f"--- Stage 1: Loading Universe from {self.universe_file} ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Universe file not found at '{self.universe_file}'. Run universe creation first.")
            raise FileNotFoundError(f"Universe file not found at {self.universe_file}")

        with open(self.universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        if not isinstance(universe, list) or not universe:
            logger.error("❌ Universe file is empty or invalid. Aborting.")
            return
        
        logger.info(f"✅ Successfully loaded {len(universe)} symbols from the universe file.")

        # Add benchmark and filter out blacklisted symbols
        if self.benchmark_symbol not in universe:
            universe.append(self.benchmark_symbol)
            logger.info(f"✅ '{self.benchmark_symbol}' added to the download queue.")
        
        symbols_to_download = [s for s in universe if s not in self.blacklist]
        logger.info(f"Skipping {len(self.blacklist)} blacklisted symbols. Processing {len(symbols_to_download)} symbols.")

        # STAGE 1.5: OPTIMIZED CACHING CHECK - BATCH PROCESS ALL SYMBOLS
        logger.info(f"\n--- Stage 1.5: Optimized Caching Check for {len(symbols_to_download)} Symbols ---")
        recent_symbols = set()
        symbols_to_update_incrementally = {}  # Stores symbols that need an incremental update
        symbols_for_full_download = []        # Stores symbols that need a full download

        # Batch check all symbols for recency
        last_dates = self._get_last_date_batch(symbols_to_download)

        for symbol, last_date in last_dates.items():
            if last_date is None:
                # No existing data or file is corrupted/empty, needs full download
                symbols_for_full_download.append(symbol)
                logger.debug(f"📥 {symbol} marked for full download.")
            else:
                # Data exists, check if it's recent
                last_date_ts = pd.to_datetime(last_date)
                current_date = pd.Timestamp.now().normalize()
                days_diff = (current_date - last_date_ts).days

                # Check if the symbol was recently processed (in processing log)
                recently_processed = False
                if symbol in self.processing_log:
                    try:
                        last_processed_time = pd.to_datetime(self.processing_log[symbol])
                        processing_days_diff = (current_date - last_processed_time.normalize()).days
                        if processing_days_diff <= 3:
                            recently_processed = True
                            logger.debug(f"✅ {symbol} was recently processed (last processed: {last_processed_time.date()})")
                    except ValueError:
                        # Invalid timestamp format, treat as not recently processed
                        pass

                if days_diff <= 3 or recently_processed:
                    # Data is recent or was recently processed, skip download
                    recent_symbols.add(symbol)
                    logger.debug(f"✅ {symbol} considered recent (data date: {last_date}, last processed: {self.processing_log.get(symbol, 'N/A')}). Will skip.")
                else:
                    # Data is old, needs incremental update
                    symbols_to_update_incrementally[symbol] = last_date
                    logger.debug(f"🔄 {symbol} marked for incremental update from {last_date}.")

        symbols_needing_update = symbols_for_full_download + list(symbols_to_update_incrementally.keys())
        
        logger.info(f"✅ Found {len(recent_symbols)} symbols with recent data.")
        logger.info(f"🔄 Found {len(symbols_to_update_incrementally)} symbols for incremental update.")
        logger.info(f"📥 Found {len(symbols_for_full_download)} symbols for full download.")
        logger.info(f"Total symbols needing updates: {len(symbols_needing_update)}")

        # STAGE 2: INCREMENTAL DOWNLOAD - ONLY FOR SYMBOLS THAT NEED UPDATES
        logger.info(f"\n--- Stage 2: Downloading Updates for {len(symbols_needing_update)} Symbols ---")
        successful_updates = 0
        failed_updates = 0
        newly_blacklisted = 0

        # Add batch processing with checkpointing
        batch_size = 10  # Process 10 symbols at a time before saving
        current_batch = []
        batch_number = 1
        
        for i, symbol in enumerate(tqdm(symbols_needing_update, desc="Downloading market data", unit="symbol")):
            logger.debug(f"Starting processing for symbol: {symbol}")
            
            # Determine start_date: None for full download, specific date for incremental
            start_date = symbols_to_update_incrementally.get(symbol)
            
            # Process the symbol
            self._update_price_data_for_symbol(symbol, start_date)
            
            # Update counters based on success/failure
            if symbol in self.blacklist:
                newly_blacklisted += 1
                failed_updates += 1
            elif symbol in self.processing_log:
                successful_updates += 1
                failed_updates += 1  # This was successful, so increment failed counter back
            else:
                failed_updates += 1

            # Add symbol to current batch
            current_batch.append(symbol)
            
            # Save progress every batch_size symbols or at the end
            if len(current_batch) >= batch_size or i == len(symbols_needing_update) - 1:
                # Save the blacklist periodically in case of interruption
                self._save_blacklist()
                logger.info(f"💾 Batch {batch_number} completed: Processed {len(current_batch)} symbols. Blacklist saved.")
                current_batch = []
                batch_number += 1

        # STAGE 3: SUMMARIZE - INCLUDE SKIPPED SYMBOLS
        logger.info("\n--- Stage 3: Finalizing Run ---")
        self._save_blacklist()
        self._save_processing_log()  # Save processing log at the end
        
        logger.info("="*50)
        logger.info("📊 OPTIMIZED PRICE DATA DOWNLOADER SUMMARY")
        logger.info("="*50)
        logger.info(f"- Total symbols in universe: {len(symbols_to_download)}")
        logger.info(f"- ✅ Symbols with recent data (skipped): {len(recent_symbols)}")
        logger.info(f"- 🔄 Symbols processed for updates: {len(symbols_needing_update)}")
        logger.info(f"- ✅ Successful updates/downloads: {successful_updates}")
        logger.info(f"- ❌ Failed updates: {failed_updates}")
        logger.info(f"- ⚫️ Newly blacklisted: {newly_blacklisted}")
        logger.info(f"- 💤 Total skipped (cached): {len(recent_symbols)}")
        logger.info("✅ Optimized Price Data Downloader finished its run.")


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=0.2, max=10))
    def _fetch_ticker_with_retry(self, symbol):
        """Fetch ticker data with retry mechanism and exponential backoff."""
        return tse.Ticker(symbol)

    def _update_fundamental_data(self):
        """
        Collects fundamental data for the full market, using an existing smaller
        dataset as a starting cache to avoid re-fetching data.
        """
        logger.info("🚀 Starting Fundamental Data Collector...")

        # Load universe
        logger.info("--- Stage 1: Loading Full Market Universe ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Universe file not found at '{self.universe_file}'.")
            logger.error("Please run the universe creation process first to generate the universe.")
            raise FileNotFoundError(f"Universe file not found at {self.universe_file}")

        with open(self.universe_file, 'r', encoding='utf-8') as f:
            full_universe = json.load(f)
        logger.info(f"✅ Loaded {len(full_universe)} symbols from universe.json.")

        # --- Stage 2: Smart Resumable Logic & Blacklisting ---
        logger.info("--- Stage 2: Determining Symbols to Process ---")
        
        # Load existing processed data
        processed_df = self._load_resumable_fundamental_cache()
        cached_symbols = set(processed_df['symbol']) if not processed_df.empty else set()
        
        # Check if the cache file is still valid based on modification time
        cache_file = self.fundamental_output_file
        cache_validity_hours = FUNDAMENTAL_COLLECTOR["CACHE_VALIDITY_HOURS"]
        cache_is_valid = True
        if cache_file.exists():
            last_modified_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if last_modified_hours > cache_validity_hours:
                logger.warning(f"Data file is {last_modified_hours:.1f} hours old (older than {cache_validity_hours} hours). Will re-fetch all symbols.")
                cache_is_valid = False

        # Filter universe to get symbols that need processing
        # Apply blacklist first
        blacklisted_symbol_names = set(self.fundamental_blacklist.keys())
        symbols_to_process = [s for s in full_universe if s not in blacklisted_symbol_names]
        
        # Then filter based on existing cache and processing log
        if cache_is_valid and not processed_df.empty:
            # Use both file modification time and processing log to determine what needs processing
            final_symbols_to_process = []
            current_time = datetime.now()
            cache_validity_timedelta = timedelta(hours=cache_validity_hours)
            
            for symbol in symbols_to_process:
                # Check if symbol exists in cache
                symbol_in_cache = symbol in cached_symbols
                
                # Check if symbol was recently processed
                recently_processed = False
                if symbol in self.fundamental_processing_log:
                    try:
                        last_processed_time = datetime.fromisoformat(self.fundamental_processing_log[symbol])
                        if current_time - last_processed_time <= cache_validity_timedelta:
                            recently_processed = True
                    except ValueError:
                        # Invalid timestamp format, treat as not recently processed
                        pass
                
                # Add to process list if not in cache or not recently processed
                if not symbol_in_cache or not recently_processed:
                    final_symbols_to_process.append(symbol)
        else:
            # Cache is invalid, process all non-blacklisted symbols
            final_symbols_to_process = symbols_to_process
        
        logger.info(f"Found {len(cached_symbols)} symbols in the resumable cache.")
        logger.info(f"Found {len(blacklisted_symbol_names)} symbols in the blacklist.")
        
        new_data_df = pd.DataFrame()
        if not final_symbols_to_process:
            logger.info("✅ All symbols from the universe are already in cache or blacklisted. No new data to fetch.")
        else:
            logger.info(f"Identified {len(final_symbols_to_process)} new or outdated symbols to process.")
            new_data_list = []
            newly_discarded_symbols = []  # To store symbols to add to blacklist (with metadata)
            
            # Add batch processing with checkpointing
            batch_size = 10  # Process 10 symbols at a time before saving
            current_batch = []
            batch_number = 1
            
            for i, symbol in enumerate(final_symbols_to_process):
                try:
                    time.sleep(FUNDAMENTAL_COLLECTOR["REQUEST_DELAY_SEC"])  # Respectful delay from config
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
                    # Update processing log for successfully fetched symbol
                    self._update_fundamental_processing_log(symbol)
                    logger.info(f"  ({i+1}/{len(final_symbols_to_process)}) Fetched for {symbol}")
                
                except RequestException as re:
                    logger.warning(f"  - ⚠️ Network error for {symbol}: {re}. Skipping.")
                except Exception as e:
                    logger.warning(f"  - ⚠️ An unexpected error occurred for {symbol}: {e}. Adding to blacklist.")
                    newly_discarded_symbols.append({
                        'symbol': symbol,
                        'added_at': datetime.now().isoformat(),
                        'reason': str(type(e).__name__)
                    })  # Add to blacklist for any unexpected error with metadata
                
                # Add symbol to current batch
                current_batch.append(symbol)
                
                # Save progress every batch_size symbols or at the end
                if len(current_batch) >= batch_size or i == len(final_symbols_to_process) - 1:
                    # Save the processing log periodically in case of interruption
                    self._save_fundamental_processing_log()
                    logger.info(f"💾 Batch {batch_number} completed: Processed {len(current_batch)} symbols. Processing log saved.")
                    current_batch = []
                    batch_number += 1
            
            if new_data_list:
                new_data_df = pd.DataFrame(new_data_list)

            # Save newly discarded symbols to blacklist (merge with existing)
            if newly_discarded_symbols:
                # Combine existing blacklist with new ones
                for item in newly_discarded_symbols:
                    self.fundamental_blacklist[item['symbol']] = item
                self._save_fundamental_blacklist()
                logger.info(f"Added {len(newly_discarded_symbols)} symbols to blacklist. Total blacklisted: {len(self.fundamental_blacklist)}")

        # --- Stage 3: Combine, Process, and Save ---
        logger.info("--- Stage 3: Combining and Processing Data ---")
        
        # Combine initial cache with newly fetched data
        combined_df = pd.concat([processed_df, new_data_df], ignore_index=True)
        
        if combined_df.empty:
            logger.error("❌ No data available to process after collection phase. Aborting.")
            return

        logger.info(f"Successfully created combined DataFrame with {len(combined_df)} total records.")


        # --- Stage 6: Save Final Data ---
        logger.info("--- Stage 6: Saving Full, Raw Data ---")
        try:
            combined_df.to_feather(self.fundamental_output_file)
            logger.info(f"💾 Successfully saved final, raw dataset to {self.fundamental_output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save the final dataset: {e}")

        # Save processing log at the end
        self._save_fundamental_processing_log()
        
        # --- Stage 7: Final Validation ---
        logger.info("--- Stage 7: Final Data Validation ---")
        self._validate_final_fundamental_data(combined_df)
        
        # --- Stage 8: Final Report ---
        logger.info("--- 📊 FINAL FUNDAMENTAL DATA QUALITY REPORT 📊 ---")
        logger.info(f"- Symbols loaded from full universe file: {len(full_universe)}")
        logger.info(f"- Symbols loaded from initial cache: {len(cached_symbols)}")
        logger.info(f"- New symbols processed: {len(final_symbols_to_process)}")
        logger.info(f"- New symbols discarded (poor quality/errors): {len(newly_discarded_symbols) if 'newly_discarded_symbols' in locals() else 0}")
        logger.info(f"- Total symbols in blacklist: {len(self.fundamental_blacklist)}")
        logger.info(f"- Total symbols in final dataset: {len(combined_df)}")
        logger.info(f"- Missing values summary (raw data):\n{combined_df.isnull().sum()}")
        logger.info("="*50)
        logger.info("✅ Full market fundamental data pipeline finished successfully. Raw data saved for preprocessing.")

    def _load_resumable_fundamental_cache(self) -> pd.DataFrame:
        """
        Loads the output data from a previous run to resume gracefully.
        The cache validity is configurable via FUNDAMENTAL_COLLECTOR settings.
        """
        cache_file = self.fundamental_output_file
        if not cache_file.exists():
            logger.info("No previous fundamental data file found. Will fetch all data from scratch.")
            return pd.DataFrame()

        cache_validity_hours = FUNDAMENTAL_COLLECTOR["CACHE_VALIDITY_HOURS"]
        last_modified_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if last_modified_hours > cache_validity_hours:
            logger.warning(f"Fundamental data file is {last_modified_hours:.1f} hours old (older than {cache_validity_hours} hours). Discarding and re-fetching all.")
            return pd.DataFrame()

        logger.info(f"Found recent fundamental data file at {cache_file} ({last_modified_hours:.1f} hours old). Loading to resume.")
        try:
            return pd.read_feather(cache_file)
        except Exception as e:
            logger.error(f"Could not read resumable cache file {cache_file}: {e}")
            return pd.DataFrame()

    def _validate_final_fundamental_data(self, df: pd.DataFrame):
        """Perform final validation after imputation to ensure data quality."""
        logger.info("Performing final fundamental data validation...")
        
        # Check for remaining NaN values in key fields
        key_fields = ['P/E', 'P/S', 'EPS']
        for field in key_fields:
            if field in df.columns:
                nan_count = df[field].isnull().sum()
                total_count = len(df)
                nan_percentage = (nan_count / total_count) * 10 if total_count > 0 else 0
                
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

    def update_market_data(self):
        """
        Main method to coordinate all data acquisition tasks in the correct order:
        1. Update universe
        2. Update price data
        3. Update fundamental data
        """
        logger.info("🚀 Starting comprehensive market data update process...")
        
        try:
            # Step 1: Update universe
            logger.info("--- Step 1: Updating Market Universe ---")
            self._update_universe()
            
            # Step 2: Update price data
            logger.info("--- Step 2: Updating Price Data ---")
            self._update_price_data()
            
            # Step 3: Update fundamental data
            logger.info("--- Step 3: Updating Fundamental Data ---")
            self._update_fundamental_data()
            
            logger.info("✅ Comprehensive market data update completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ An error occurred during market data update: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    provider = DataProvider()
    provider.update_market_data()
