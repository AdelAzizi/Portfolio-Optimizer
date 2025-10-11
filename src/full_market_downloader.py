# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Full Market Data Downloader (Refactored)
# Description: An intelligent and efficient downloader that performs true incremental
#              updates for a given stock universe, manages a blacklist for problematic
#              symbols, and is fully configurable via a central config file.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import pytse_client as tse
import time
import logging
import json
from pathlib import Path
from typing import List, Set, Optional, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm  # For progress bar
import traceback  # For detailed error logging

# --- Project-Specific Imports ---
from src.config import PROJECT_ROOT, UNIVERSE_CREATOR, FULL_MARKET_DOWNLOADER


# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'full_downloader.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullMarketDownloader:
    """
    Intelligently downloads and performs incremental updates for stock data.
    It uses a central configuration, loads a universe of symbols, and maintains
    a blacklist of symbols that consistently fail to download.
    """
    def __init__(self):
        """
        Initializes the downloader using settings from the central config file.
        """
        # --- Load paths and settings from config ---
        self.data_dir = PROJECT_ROOT / FULL_MARKET_DOWNLOADER["DATA_DIR"]
        self.cache_dir = PROJECT_ROOT / UNIVERSE_CREATOR["CACHE_DIR"]
        self.universe_file = self.cache_dir / UNIVERSE_CREATOR["UNIVERSE_FILENAME"]
        self.blacklist_file = self.cache_dir / FULL_MARKET_DOWNLOADER["BLACKLIST_FILENAME"]
        self.benchmark_symbol = FULL_MARKET_DOWNLOADER["BENCHMARK_SYMBOL"]
        self.api_delay = FULL_MARKET_DOWNLOADER["API_DELAY_SECONDS"]

        # --- Setup directories and load state ---
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.blacklist = self._load_blacklist()
        self.failure_counts = {}  # Track consecutive failures for each symbol
        self.max_consecutive_failures = 3  # Number of consecutive failures before blacklisting
        self.processing_log_file = self.cache_dir / "processing_log.json"  # Track last processing times
        self.processing_log = self._load_processing_log()
        
        logger.info("Downloader initialized with the following settings:")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Blacklist file: {self.blacklist_file}")
        logger.info(f"Processing log file: {self.processing_log_file}")

    def _load_blacklist(self) -> Set[str]:
        """Loads the set of blacklisted symbols from a JSON file."""
        if self.blacklist_file.exists():
            try:
                with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                    blacklisted_symbols = json.load(f)
                    logger.info(f"Loaded {len(blacklisted_symbols)} blacklisted symbols.")
                    return set(blacklisted_symbols)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Blacklist file is corrupted or empty. Starting with an empty list.")
                return set()
        return set()

    def _save_blacklist(self):
        """Saves the current set of blacklisted symbols to a JSON file."""
        with open(self.blacklist_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.blacklist), f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(self.blacklist)} symbols to the blacklist.")

    def _increment_failure_count(self, symbol: str):
        """Increment the failure count for a symbol and check if it should be blacklisted."""
        self.failure_counts[symbol] = self.failure_counts.get(symbol, 0) + 1
        current_failures = self.failure_counts[symbol]
        
        if current_failures >= self.max_consecutive_failures:
            logger.warning(f"🔴 Blacklisting {symbol} after {current_failures} consecutive failures.")
            self.blacklist.add(symbol)
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

    def _update_processing_log(self, symbol: str):
        """Update the processing log with the current timestamp for a symbol."""
        from datetime import datetime
        self.processing_log[symbol] = datetime.now().isoformat()
        # Save periodically to preserve progress
        if len(self.processing_log) % 10 == 0:  # Save every 10 updates
            self._save_processing_log()

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
                    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
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
                            # We still return the date so the main logic knows the symbol exists with recent data
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
                    last_dates[symbol] = None # Error reading file, treat as needs update
                    logger.debug(f"⚠️ Error reading {symbol} file: {e}")
            else:
                last_dates[symbol] = None  # File doesn't exist, needs full download
                logger.debug(f"📁 {symbol} file doesn't exist, needs full download")
        
        return last_dates

    def run_update(self):
        """
        Runs the main incremental update process with optimized caching.
        """
        logger.info("🚀 Starting Optimized Market Downloader...")

        # STAGE 1: LOAD UNIVERSE
        logger.info(f"--- Stage 1: Loading Universe from {self.universe_file} ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Universe file not found at '{self.universe_file}'. Run universe_creator.py first.")
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
                    last_processed_time = pd.to_datetime(self.processing_log[symbol])
                    processing_days_diff = (current_date - last_processed_time.normalize()).days
                    if processing_days_diff <= 3:
                        recently_processed = True
                        logger.debug(f"✅ {symbol} was recently processed (last processed: {last_processed_time.date()})")

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

        # Add retry function for API calls
        def fetch_with_retry(ticker, start_date, max_retries=3):
            for attempt in range(max_retries):
                try:
                    # Add some logging to debug the 'DataFrame not callable' issue
                    logger.debug(f"Attempt {attempt + 1}: Fetching history for symbol with start_date={start_date}, type={type(start_date)}")
                    
                    # Check if ticker.history is callable (function) or property
                    if callable(ticker.history):
                        # If it's callable, call it as a function
                        if start_date:
                            hist = ticker.history(start_date=start_date)
                            logger.debug(f"Called ticker.history(start_date={start_date}), received type: {type(hist)}")
                        else:
                            hist = ticker.history()
                            logger.debug(f"Called ticker.history(), received type: {type(hist)}")
                    else:
                        # If it's a property, access it directly and then filter by date
                        hist = ticker.history
                        logger.debug(f"Accessed ticker.history as property, type: {type(hist)}")
                        if start_date and hasattr(hist, 'loc'):
                            # Filter the history DataFrame by start_date if provided
                            try:
                                hist = hist[hist.index >= start_date]
                                logger.debug(f"Filtered history by start_date {start_date}, new shape: {hist.shape if hasattr(hist, 'shape') else 'N/A'}")
                            except Exception as filter_error:
                                logger.warning(f"Could not filter history by start_date {start_date}: {filter_error}, returning full history")
                                logger.warning(f"Available index range: {hist.index.min() if hasattr(hist, 'index') and not hist.empty else 'N/A'} to {hist.index.max() if hasattr(hist, 'index') and not hist.empty else 'N/A'}")
                
                    # Check if hist is actually a DataFrame or if there's an issue
                    if hist is not None and hasattr(hist, 'empty'):
                        # It's a proper DataFrame
                        logger.debug(f"Received valid DataFrame for {symbol}, shape: {hist.shape if not hist.empty else 'empty'}")
                        if not hist.empty and hasattr(hist, 'index') and not hist.index.empty:
                            logger.debug(f"DataFrame date range for {symbol}: {hist.index.min()} to {hist.index.max()}")
                        return hist
                    else:
                        logger.warning(f"⚠️  Received unexpected object type for {symbol}: {type(hist)}, attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            raise ValueError(f"Received unexpected object type instead of DataFrame for {symbol}: {type(hist)}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e

        # Add batch processing with checkpointing
        batch_size = 10  # Process 10 symbols at a time before saving
        current_batch = []
        batch_number = 1
        
        for i, symbol in enumerate(tqdm(symbols_needing_update, desc="Downloading market data", unit="symbol")):
            logger.debug(f"Starting processing for symbol: {symbol}")
            time.sleep(self.api_delay)  # Be respectful to the API
            file_path = self.data_dir / f"{symbol}.csv"
            
            # Determine start_date: None for full download, specific date for incremental
            start_date = symbols_to_update_incrementally.get(symbol)

            try:
                logger.info(f"⬇️  Processing {symbol} (start_date: {start_date})...")
                
                # Fetch data (either full or incremental)
                if symbol == self.benchmark_symbol:
                    logger.debug(f"Creating FinancialIndex for benchmark symbol: {symbol}")
                    ticker = tse.FinancialIndex(symbol)
                else:
                    logger.debug(f"Creating Ticker for symbol: {symbol}")
                    ticker = tse.Ticker(symbol, adjust=True)
                
                # Fetch with retry mechanism
                logger.debug(f"Calling fetch_with_retry for {symbol} with start_date: {start_date}")
                hist = fetch_with_retry(ticker, start_date)
                logger.debug(f"Received history for {symbol}, type: {type(hist)}, shape: {hist.shape if hist is not None and not hist.empty else 'N/A'}")

                if hist is not None and not hist.empty:
                    logger.debug(f"History for {symbol} is not empty, rows: {len(hist)}")
                    if start_date and len(hist) > 1:
                        # Append new data (excluding the first row which is a duplicate of the last known date)
                        hist.iloc[1:].to_csv(file_path, mode='a', header=False, index=True, encoding='utf-8')
                        logger.info(f"💾 Appended {len(hist)-1} new rows for {symbol}.")
                        # Log the date range of the appended data for debugging
                        if len(hist) > 1:
                            new_data_dates = hist.iloc[1:].index
                            if len(new_data_dates) > 0:
                                logger.debug(f"📅 Appended data for {symbol} from {new_data_dates.min()} to {new_data_dates.max()}")
                    elif start_date and len(hist) <= 1:
                        # This case means no new data was found since the last update
                        logger.info(f"✅ No new data for {symbol} since {start_date}. Already up-to-date.")
                    else: # This is for full download (start_date is None)
                        # Full download
                        hist.to_csv(file_path, index=True, encoding='utf-8')
                        logger.info(f"💾 Performed full download for {symbol} ({len(hist)} rows).")
                        # Log the date range of the full download for debugging
                        if not hist.empty:
                            logger.debug(f"📅 Full download for {symbol} from {hist.index.min()} to {hist.index.max()}")
                    successful_updates += 1
                    logger.debug(f"Successfully updated {symbol}, total successful: {successful_updates}")
                else:
                    logger.info(f"✅ No new data for {symbol}. Already up-to-date.")
                    
                # After successful processing, touch the file to update its modification time
                # This helps in tracking when the file was last updated
                file_path.touch()
            
            except IndexError as ie:
                if 'single positional indexer is out-of-bounds' in str(ie):
                    should_blacklist = self._increment_failure_count(symbol)
                    if should_blacklist:
                        newly_blacklisted += 1
                else:
                    logger.error(f"❌ An unexpected indexing error occurred for {symbol}: {ie}.")
                    logger.debug(f"Full traceback for {symbol}: {traceback.format_exc()}")
                    # Increment failure count for other index errors too
                    self._increment_failure_count(symbol)
                failed_updates += 1
            except Exception as e:
                logger.error(f"❌ A critical error occurred for {symbol}: {e}.")
                logger.debug(f"Full traceback for {symbol}: {traceback.format_exc()}")
                
                # Check if this is a specific error that should trigger blacklisting
                error_msg = str(e).lower()
                if ('dataframe' in error_msg and 'callable' in error_msg) or 'no historical data' in error_msg:
                    should_blacklist = self._increment_failure_count(symbol)
                    if should_blacklist:
                        newly_blacklisted += 1
                else:
                    # For other errors, increment failure count but don't immediately blacklist
                    self._increment_failure_count(symbol)
                
                failed_updates += 1
            else:
                # If successful, reset failure count and update processing log
                self._reset_failure_count(symbol)
                self._update_processing_log(symbol)
                logger.debug(f"✅ Successfully processed {symbol}, reset failure count and updated processing log.")
            
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
        logger.info("📊 OPTIMIZED DOWNLOADER SUMMARY")
        logger.info("="*50)
        logger.info(f"- Total symbols in universe: {len(symbols_to_download)}")
        logger.info(f"- ✅ Symbols with recent data (skipped): {len(recent_symbols)}")
        logger.info(f"- 🔄 Symbols processed for updates: {len(symbols_needing_update)}")
        logger.info(f"- ✅ Successful updates/downloads: {successful_updates}")
        logger.info(f"- ❌ Failed updates: {failed_updates}")
        logger.info(f"- ⚫️ Newly blacklisted: {newly_blacklisted}")
        logger.info(f"- 💤 Total skipped (cached): {len(recent_symbols)}")
        logger.info("✅ Optimized Downloader finished its run.")

if __name__ == "__main__":
    downloader = FullMarketDownloader()
    downloader.run_update()
