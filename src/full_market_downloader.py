
# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Full Market Data Downloader
# Description: A modified version of the smart downloader that fetches data for
#              nearly the entire market, applying only basic filters.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import pytse_client as tse
import time
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional


# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    Intelligently downloads and updates local stock data CSV files for the entire market.
    """
    def __init__(self, data_dir: str = 'data/full_market_data_csvs', cache_dir: str = 'cache'):
        """
        Initialize the downloader.

        Args:
            data_dir (str): Directory for local ticker CSV files.
            cache_dir (str): Directory for caching symbol list.
        """
        self.data_dir = PROJECT_ROOT / data_dir
        self.cache_dir = PROJECT_ROOT / cache_dir
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.universe_file = self.cache_dir / 'universe.json'
        logger.info(f"Full Market Data directory set to: {self.data_dir}")
        logger.info(f"Cache directory set to: {self.cache_dir}")

    def run_update(self):
        """
        Runs the main update process for the full market based on a pre-defined universe.
        """
        logger.info("🚀 Starting Full Market Downloader...")

        # STAGE 1: LOAD THE PRE-DEFINED UNIVERSE
        logger.info(f"--- Stage 1: Loading Pre-defined Universe from {self.universe_file} ---")
        if not self.universe_file.exists():
            logger.error(f"CRITICAL: Universe file not found at '{self.universe_file}'.")
            logger.error("Please run the universe_creator.py script first to generate the universe.")
            raise FileNotFoundError(f"Universe file not found at {self.universe_file}")

        with open(self.universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        if not isinstance(universe, list) or not universe:
            logger.error("❌ Universe file is empty or invalid. Aborting.")
            return
        
        logger.info(f"✅ Successfully loaded {len(universe)} symbols from the universe file.")

        # Add the benchmark index to the list if it's not already there
        if 'شاخص کل' not in universe:
            universe.append('شاخص کل')
            logger.info("✅ 'شاخص کل' added to the download queue as the benchmark.")
        
        # STAGE 2: INCREMENTAL PRICE DATA DOWNLOAD
        logger.info(f"\n--- Stage 2: Updating Price Data for {len(universe)} Selected Symbols ---")
        successful_downloads = 0
        failed_downloads = 0
        skipped_count = 0

        for symbol in universe:
            file_path = self.data_dir / f"{symbol}.csv"
            
            # Check if the local CSV file is up-to-date (cache validity: 12 hours)
            if file_path.exists():
                last_modified_hours = (time.time() - file_path.stat().st_mtime) / 3600
                if last_modified_hours < 12:
                    skipped_count += 1
                    continue  # Skip download if file is recent

            # Download data if the file doesn't exist or is outdated
            try:
                time.sleep(0.5)  # Be respectful to the API
                hist = None
                logger.info(f"⬇️  Downloading data for {symbol}...")
                if symbol == 'شاخص کل':
                    index_ticker = tse.FinancialIndex(symbol)
                    hist = index_ticker.history
                else:
                    ticker = tse.Ticker(symbol, adjust=True)
                    hist = ticker.history

                if hist is not None and not hist.empty:
                    hist.to_csv(file_path, encoding='utf-8')
                    logger.info(f"💾 Successfully downloaded and updated price for {symbol}.")
                    successful_downloads += 1
                else:
                    logger.warning(f"⚠️ No price data returned for {symbol}.")
                    failed_downloads += 1
            
            except IndexError as ie:
                if 'single positional indexer is out-of-bounds' in str(ie):
                    logger.warning(f"🟡 Skipping {symbol}: No historical data available from the source.")
                else:
                    logger.error(f"❌ An unexpected indexing error occurred for {symbol}: {ie}.")
                failed_downloads += 1
            except Exception as e:
                logger.error(f"❌ A critical error occurred for {symbol}: {e}.")
                failed_downloads += 1
        
        logger.info("="*50)
        logger.info("📊 FULL MARKET DOWNLOAD SUMMARY")
        logger.info("="*50)
        logger.info(f"- Total symbols processed: {len(universe)}")
        logger.info(f"- ✅ Successful downloads/updates: {successful_downloads}")
        logger.info(f"- ❌ Failed downloads: {failed_downloads}")
        logger.info(f"- ⏭️ Skipped (up-to-date): {skipped_count}")
        logger.info("✅ Full Market Downloader finished its run.")

if __name__ == "__main__":
    downloader = FullMarketDownloader()
    downloader.run_update()