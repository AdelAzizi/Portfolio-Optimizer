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
from typing import List, Set, Optional
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
        
        logger.info("Downloader initialized with the following settings:")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Blacklist file: {self.blacklist_file}")

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

    def _get_last_date(self, file_path: Path) -> Optional[str]:
        """
        Reads the last date from a CSV file to determine the starting point for an incremental update.
        Assumes the date is in the first column.
        """
        try:
            # Read only the last row for efficiency
            last_line = pd.read_csv(file_path, usecols=[0], skip_blank_lines=True).iloc[-1]
            return last_line.iloc[0]  # Fixed: Use .iloc[0] instead of [0] to avoid pandas deprecation warning
        except (pd.errors.EmptyDataError, IndexError):
            logger.warning(f"File {file_path.name} is empty. A full download will be performed.")
            return None
        except Exception as e:
            logger.error(f"Could not read last date from {file_path.name}: {e}. Triggering full download.")
            return None

    def run_update(self):
        """
        Runs the main incremental update process.
        """
        logger.info("🚀 Starting Refactored Market Downloader...")

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

        # STAGE 2: INCREMENTAL DOWNLOAD
        logger.info(f"\n--- Stage 2: Performing Incremental Update for {len(symbols_to_download)} Symbols ---")
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
                        hist = ticker.history(start_date=start_date)
                    else:
                        # If it's a property, access it directly and then filter by date
                        hist = ticker.history
                        if start_date and hasattr(hist, 'loc'):
                            # Filter the history DataFrame by start_date if provided
                            try:
                                hist = hist[hist.index >= start_date]
                            except Exception:
                                logger.warning(f"Could not filter history by start_date {start_date}, returning full history")
                
                    # Check if hist is actually a DataFrame or if there's an issue
                    if hist is not None and hasattr(hist, 'empty'):
                        # It's a proper DataFrame
                        logger.debug(f"Received valid DataFrame for {symbol}, shape: {hist.shape if not hist.empty else 'empty'}")
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

        # Use tqdm for progress bar
        for symbol in tqdm(symbols_to_download, desc="Downloading market data", unit="symbol"):
            logger.debug(f"Starting processing for symbol: {symbol}")
            time.sleep(self.api_delay)  # Be respectful to the API
            file_path = self.data_dir / f"{symbol}.csv"
            start_date = None

            if file_path.exists():
                start_date = self._get_last_date(file_path)
                logger.debug(f"Found existing file for {symbol}, start_date: {start_date}")

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
                    else:
                        # Full download
                        hist.to_csv(file_path, index=True, encoding='utf-8')
                        logger.info(f"💾 Performed full download for {symbol} ({len(hist)} rows).")
                    successful_updates += 1
                    logger.debug(f"Successfully updated {symbol}, total successful: {successful_updates}")
                else:
                    logger.info(f"✅ No new data for {symbol}. Already up-to-date.")
            
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
                # If successful, reset failure count
                self._reset_failure_count(symbol)
                logger.debug(f"✅ Successfully processed {symbol}, reset failure count.")
        
        # STAGE 3: SAVE STATE AND SUMMARIZE
        logger.info("\n--- Stage 3: Finalizing Run ---")
        self._save_blacklist()
        
        logger.info("="*50)
        logger.info("📊 DOWNLOADER SUMMARY")
        logger.info("="*50)
        logger.info(f"- Total symbols processed: {len(symbols_to_download)}")
        logger.info(f"- ✅ Successful updates/downloads: {successful_updates}")
        logger.info(f"- ❌ Failed updates: {failed_updates}")
        logger.info(f"- ⚫️ Newly blacklisted: {newly_blacklisted}")
        logger.info("✅ Downloader finished its run.")

if __name__ == "__main__":
    downloader = FullMarketDownloader()
    downloader.run_update()