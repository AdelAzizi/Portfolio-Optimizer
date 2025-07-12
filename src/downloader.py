# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Smart, Cache-Aware Downloader for Iranian Stock Data
# Description: Intelligently updates a local directory of stock data CSV files,
#              fetching only missing or outdated data.
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
# The script is in 'src/', so we go up one level to get the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'downloader.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """
    Intelligently downloads and updates local stock data CSV files.
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
        self.symbols_cache_file = self.cache_dir / 'symbols_data_v3.json'
        self.fundamental_cache_file = self.cache_dir / 'fundamental_data.json'
        logger.info(f"Data directory set to: {self.data_dir}")
        logger.info(f"Cache directory set to: {self.cache_dir}")

    def _is_cache_valid(self, cache_file: Path, max_age_hours: int) -> bool:
        """Check if a cache file is valid."""
        if not cache_file.exists():
            return False
        cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if cache_age_hours > max_age_hours:
            logger.info(f"⏰ Cache {cache_file.name} is {cache_age_hours:.1f} hours old, refreshing...")
            return False
        logger.info(f"✅ Using valid cache {cache_file.name} ({cache_age_hours:.1f} hours old)")
        return True

    def fetch_all_symbols(self) -> Optional[List[Dict]]:
        """Fetch all available symbols using pytse-client, with local caching."""
        try:
            logger.info("📊 Fetching all symbols from TSE...")
            if self._is_cache_valid(self.symbols_cache_file, max_age_hours=168):  # 1 week cache
                with open(self.symbols_cache_file, 'r', encoding='utf-8') as f:
                    symbols_data = json.load(f)
                logger.info(f"   Loaded {len(symbols_data)} symbols from cache")
                return symbols_data

            symbols_data = tse.symbols_data.all_symbols()
            # Convert sets to lists for JSON serialization
            for symbol_info in symbols_data:
                for key, value in symbol_info.items():
                    if isinstance(value, set):
                        symbol_info[key] = list(value)
            
            with open(self.symbols_cache_file, 'w', encoding='utf-8') as f:
                json.dump(symbols_data, f, ensure_ascii=False, indent=2)
            logger.info(f"   Fetched and cached {len(symbols_data)} symbols")
            return symbols_data
        except Exception as e:
            logger.error(f"❌ ERROR: Could not fetch symbols: {e}")
            return None

    def filter_investment_universe(self, symbols_data: List[Dict]) -> List[str]:
        """
        Applies a strict, multi-stage quantitative filter using real-time market stats
        to create a high-quality, tradable stock universe.
        """
        logger.info("🔍 Starting advanced quantitative pre-filtering...")

        # --- Stage 1: Fetch real-time market-wide statistics for screening ---
        try:
            logger.info("   Fetching real-time market-wide stats for screening...")
            market_stats_df = tse.get_stats(to_csv=False)
            if market_stats_df is None or market_stats_df.empty:
                logger.error("❌ Failed to fetch real-time market stats. Aborting filter.")
                return []
            logger.info(f"   Successfully fetched real-time stats for {len(market_stats_df)} symbols.")
        except Exception as e:
            logger.error(f"❌ Critical error fetching real-time market stats: {e}")
            return []

        # --- Stage 2: Apply Sequential Quantitative Filters ---
        filtered_df = market_stats_df.copy()
        initial_count = len(filtered_df)
        logger.info(f"   Initial symbols from real-time stats: {initial_count}")

        # Filter a: Market Type (Bourse and Fara Bourse)
        filtered_df = filtered_df[filtered_df['flow'].isin([1, 2])]
        count_after_market_type = len(filtered_df)
        logger.info(f"   {initial_count} → {count_after_market_type} (After market type filter: Bourse/Fara Bourse)")

        # Filter c: Liquidity (Transaction Volume)
        filtered_df['volume_of_trans'] = pd.to_numeric(filtered_df['volume_of_trans'], errors='coerce').fillna(0)
        min_volume = 100000
        filtered_df = filtered_df[filtered_df['volume_of_trans'] > min_volume]
        count_after_liquidity = len(filtered_df)
        logger.info(f"   {count_after_market_type} → {count_after_liquidity} (After liquidity filter: Volume > {min_volume:,})")

        # Filter d: Size (Market Cap)
        filtered_df['val_company_last_day'] = pd.to_numeric(filtered_df['val_company_last_day'], errors='coerce').fillna(0)
        min_market_cap = 1e12  # 1000 Billion Toman
        filtered_df = filtered_df[filtered_df['val_company_last_day'] > min_market_cap]
        count_after_size = len(filtered_df)
        logger.info(f"   {count_after_liquidity} → {count_after_size} (After size filter: Market Cap > {min_market_cap / 1e12:,.0f}T Toman)")

        # Filter e: Data Quality
        filtered_df = filtered_df[(filtered_df['val_company_last_day'] > 0) & (filtered_df['volume_of_trans'] > 0)]
        count_after_quality = len(filtered_df)
        logger.info(f"   {count_after_size} → {count_after_quality} (After data quality filter)")

        # --- Final Step: Extract and return the universe ---
        if 'symbol' not in filtered_df.columns:
            logger.error(f"❌ Could not find 'symbol' column. Available columns: {filtered_df.columns.tolist()}")
            return []
            
        universe = filtered_df['symbol'].drop_duplicates().tolist()
        final_count = len(universe)
        
        logger.info(f"✅ Quantitative filtering complete. Final universe size: {final_count} stocks.")
# --- Export the final universe to a file for other scripts to use ---
        try:
            universe_file = self.cache_dir / 'universe.json'
            with open(universe_file, 'w', encoding='utf-8') as f:
                json.dump(universe, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Successfully exported the final universe of {final_count} symbols to {universe_file.name}.")
        except Exception as e:
            logger.error(f"❌ Failed to export universe list: {e}")
        return universe

    def fetch_fundamental_data(self, symbols: List[str]) -> Dict:
        """
        Fetches fundamental data (P/E, P/B) for a list of symbols.

        Args:
            symbols (List[str]): A list of stock symbols to fetch data for.

        Returns:
            Dict: A dictionary with fundamental data keyed by symbol.
        """
        logger.info(f"🔬 Fetching fundamental data for {len(symbols)} symbols...")
        
        if self._is_cache_valid(self.fundamental_cache_file, max_age_hours=168): # 1 week cache
            with open(self.fundamental_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        fundamental_data = {}
        for symbol in symbols:
            try:
                time.sleep(0.2) # Respectful delay
                ticker = tse.Ticker(symbol)
                
                pe = ticker.p_e_ratio
                # pb = ticker.group_p_b # This attribute does not exist
                
                # Basic validation
                if pe is not None and pe > 0:
                    fundamental_data[symbol] = {
                        'P/E': pe
                    }
                    logger.info(f"  - Fetched for {symbol}: P/E={pe}")
                else:
                    logger.warning(f"  - Skipping {symbol} due to invalid fundamental data (P/E: {pe})")

            except Exception as e:
                logger.error(f"  - ❌ Could not fetch fundamental data for {symbol}: {e}")
        
        # Save to cache file
        with open(self.fundamental_cache_file, 'w', encoding='utf-8') as f:
            json.dump(fundamental_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Fundamental data for {len(fundamental_data)} symbols saved to cache.")
        return fundamental_data

    def run_update(self):
        """
        Runs the main update process with pre-filtering.
        """
        logger.info("🚀 Starting Smart Downloader v2 with Pre-filtering...")

        # STAGE 1: GET AND FILTER UNIVERSE
        logger.info("--- Stage 1: Defining Investment Universe ---")
        all_symbols_data = self.fetch_all_symbols()
        if not all_symbols_data:
            logger.error("❌ Could not fetch master symbols list. Aborting.")
            return
        
        universe = self.filter_investment_universe(all_symbols_data)
        if not universe:
            logger.error("❌ No symbols passed the pre-filter. Aborting.")
            return

        # Add the benchmark index to the list to be downloaded
        universe.append('شاخص کل')
        logger.info("✅ 'شاخص کل' added to the download queue as the benchmark.")
        
        # STAGE 2: INCREMENTAL PRICE DATA DOWNLOAD
        logger.info(f"\n--- Stage 2: Updating Price Data for {len(universe)} Selected Symbols ---")
        successful_downloads = 0
        failed_downloads = 0
        skipped_count = 0

        # The rest of the download loop remains the same, but now iterates over the clean 'universe'
        for symbol in universe:
            file_path = self.data_dir / f"{symbol}.csv"
            should_download = False

            if file_path.exists():
                last_modified_hours = (time.time() - file_path.stat().st_mtime) / 3600
                if last_modified_hours < 24:
                    # logger.info(f"✅ Skipping {symbol} (already up-to-date, {last_modified_hours:.1f} hours old)")
                    skipped_count += 1
                    continue
                else:
                    # logger.info(f"⏰ File for {symbol} is old ({last_modified_hours:.1f} hours). Attempting to update...")
                    should_download = True
            else:
                # logger.info(f"- File for {symbol} not found. Attempting to download...")
                should_download = True

            if should_download:
                # This part is a LIVE download attempt. The symbol list fetch uses a cache,
                # but this price history download does not, ensuring we get the latest data.
                try:
                    time.sleep(0.5)  # Be respectful to the API
                    hist = None
                    if symbol == 'شاخص کل':
                        # logger.info(f"📈 Downloading financial index: {symbol}")
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
                        logger.warning(f"⚠️ No price data returned for {symbol}. It might be a delisted or untraded symbol.")
                        failed_downloads += 1
                
                except IndexError as ie:
                    # This specific error often occurs when the TSE API returns no data for a symbol
                    # (e.g., rights issues, delisted stocks). We handle it gracefully.
                    if 'single positional indexer is out-of-bounds' in str(ie):
                        logger.warning(f"🟡 Skipping {symbol}: No historical data available from the source.")
                    else:
                        logger.error(f"❌ An unexpected indexing error occurred for {symbol}: {ie}.")
                    failed_downloads += 1
                except Exception as e:
                    logger.error(f"❌ A critical error occurred for {symbol}: {e}.")
                    failed_downloads += 1
        
        logger.info("="*50)
        logger.info("📊 PRICE DOWNLOAD SUMMARY")
        logger.info("="*50)
        logger.info(f"- Total symbols processed: {len(universe)}")
        logger.info(f"- ✅ Successful downloads/updates: {successful_downloads}")
        logger.info(f"- ❌ Failed downloads: {failed_downloads}")
        logger.info(f"- ⏭️ Skipped (up-to-date): {skipped_count}")
        logger.info("✅ Smart Downloader finished its run.")

if __name__ == "__main__":
    # The script assumes it's run from the project root or that the
    # 'data/full_market_data_csvs' and 'cache' directories are relative to the project root.
    downloader = DataDownloader(data_dir='data/full_market_data_csvs', cache_dir='cache')
    downloader.run_update()