# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Offline Data Preprocessor for Financial Analysis
# Description: Fetches, cleans, and preprocesses financial data for the main application.
# Author: Roo, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import pytse_client as tse
import datetime
import os
import logging
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path
import jdatetime
import config

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings('ignore')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/preprocessor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles offline data fetching, cleaning, and metric calculation.
    """
    def __init__(self, time_horizons: Dict[str, float], cache_dir: str = config.CACHE_DIR, tickers_data_dir: str = '../data/tickers_data'):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            time_horizons: Dictionary mapping horizon names to years of data.
            cache_dir: Directory for caching data files.
            tickers_data_dir: Directory for local ticker data files.
        """
        self.time_horizons = time_horizons
        self.cache_dir = Path(cache_dir)
        self.tickers_data_dir = Path(tickers_data_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tickers_data_dir.mkdir(exist_ok=True)
        
        self.symbols_cache_file = self.cache_dir / 'symbols_data_v3.json'
        self.failed_symbols_cache_file = self.cache_dir / 'failed_symbols.json'
        self.master_price_df = None
        self.permanently_failed_symbols = self._load_failed_symbols()
        self.benchmark_cache_file = self.cache_dir / 'benchmark_data_v3.csv'
 
    def is_cache_valid(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file exists and is not older than specified hours."""
        if not cache_file.exists():
            return False
        
        cache_age = time.time() - cache_file.stat().st_mtime
        cache_age_hours = cache_age / 3600
        
        if cache_age_hours > max_age_hours:
            logger.info(f"⏰ Cache {cache_file.name} is {cache_age_hours:.1f} hours old, refreshing...")
            return False
        
        logger.info(f"✅ Using valid cache {cache_file.name} ({cache_age_hours:.1f} hours old)")
        return True

    def fetch_all_symbols(self) -> Optional[List[Dict]]:
        """Fetch all available symbols using pytse-client."""
        try:
            logger.info("📊 Fetching all symbols from TSE...")
            
            if self.is_cache_valid(self.symbols_cache_file, max_age_hours=168):  # Weekly refresh
                with open(self.symbols_cache_file, 'r', encoding='utf-8') as f:
                    symbols_data = json.load(f)
                logger.info(f"   Loaded {len(symbols_data)} symbols from cache")
                return symbols_data
            
            symbols_data = []
            try:
                all_symbols = tse.symbols_data.all_symbols()
                for symbol_info in all_symbols:
                    if isinstance(symbol_info, dict):
                        symbols_data.append(symbol_info)
                    else:
                        symbols_data.append({'symbol': str(symbol_info), 'name': str(symbol_info), 'market': 'بورس', 'type': 'stock'})
            except Exception as e:
                logger.warning(f"⚠️ Method 1 failed: {e}")
                try:
                    tse.download(symbols="all", write_to_csv=True)
                    symbols_df = pd.read_csv(tse.SYMBOLS_DATA_CACHE_PATH)
                    symbols_data = symbols_df.to_dict('records')
                except Exception as e2:
                    logger.error(f"❌ Method 2 also failed: {e2}")
                    return None
            
            with open(self.symbols_cache_file, 'w', encoding='utf-8') as f:
                json.dump(symbols_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"   Fetched and cached {len(symbols_data)} symbols")
            return symbols_data
            
        except Exception as e:
            logger.error(f"❌ ERROR: Could not fetch symbols: {e}")
            return None

    def filter_investment_universe(self, symbols_data: List[Dict]) -> List[str]:
        """Apply robust filters to create investment universe."""
        logger.info(f"🔍 Filtering from {len(symbols_data)} total symbols...")
        if not symbols_data:
            logger.error("❌ No symbols data provided")
            return []
        
        df = pd.DataFrame(symbols_data)
        
        if 'symbol' not in df.columns:
            for alt_col in ['Symbol', 'ticker', 'Ticker', 'code', 'Code']:
                if alt_col in df.columns:
                    df['symbol'] = df[alt_col]
                    break
        
        if 'symbol' not in df.columns:
            logger.error(f"❌ No symbol column found. Available columns: {list(df.columns)}")
            return []
        
        initial_count = len(df)
        df = df[df['symbol'].notna() & (df['symbol'] != '')]
        
        if 'market' in df.columns:
            df = df[df['market'].isin(['بورس', 'فرابورس', 'TSE', 'IFB'])]
        
        if 'type' in df.columns:
            df = df[df['type'].isin(['stock', 'سهم', 'سهام'])]
        
        exclude_patterns = ['تقدم', 'تسهیلات', 'ورشکسته', 'اختیار', 'آتی', 'حق', 'صندوق', 'سرمایه‌گذاری', 'لیزینگ', 'اعتباری', 'تامین', 'ضمانت', 'بیمه', 'ETF', 'FUND', 'RIGHT', 'WARRANT']
        
        for col in ['symbol', 'name', 'Symbol', 'Name']:
            if col in df.columns:
                mask = ~df[col].astype(str).str.contains('|'.join(exclude_patterns), na=False, case=False)
                df = df[mask]
        
        universe = df['symbol'].drop_duplicates().tolist()

        if self.permanently_failed_symbols:
            initial_len = len(universe)
            universe = [s for s in universe if s not in self.permanently_failed_symbols]
            logger.info(f"   🚫 Excluded {initial_len - len(universe)} symbols from the failed symbols snapshot.")

        # universe = universe[:1000]
        # logger.info(f"   Further reduced universe to the top {len(universe)} stocks for performance.")
        logger.info(f"   ✅ Filtered universe: {initial_count} → {len(universe)} stocks")
        return universe

    def download_ticker_data(self, symbol: str, years_of_data: float, max_retries: int = 1, retry_delay: int = 10) -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker with retry logic,
        prioritizing local cache and saving to CSV.
        """
        ticker_file = self.tickers_data_dir / f"{symbol}.csv"

        # 1. Try loading from local CSV cache
        if ticker_file.exists():
            try:
                hist_data = pd.read_csv(ticker_file, index_col='date', parse_dates=True)
                
                # Ensure index is unique after loading from CSV
                if not hist_data.index.is_unique:
                    hist_data = hist_data[~hist_data.index.duplicated(keep='last')]
                    logger.warning(f"   ⚠️ Removed duplicate dates from local CSV for {symbol}.")

                if 'close' in hist_data.columns and len(hist_data) >= 20:
                    # Check if cache is recent enough based on years_of_data
                    end_date_cached = hist_data.index.max()
                    start_date_required = pd.Timestamp.now() - pd.DateOffset(years=int(years_of_data))
                    if end_date_cached >= start_date_required:
                        logger.info(f"✅ Using valid local CSV for {symbol}")
                        return hist_data[['close', 'volume']]
                    else:
                        logger.warning(f"⚠️ Local CSV for {symbol} is too old for required {years_of_data} years. Attempting download from API.")
                else:
                    logger.warning(f"⚠️ Local CSV for {symbol} is empty or missing 'close' column or has insufficient data points. Attempting download from API.")
            except Exception as e:
                logger.warning(f"⚠️ Error reading local CSV for {symbol}: {e}. Attempting download from API.")

        # 2. If local cache is invalid or not found, attempt download from API with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"   🌐 Downloading {symbol} from pytse-client API (Attempt {attempt + 1}/{max_retries})...")
                ticker = tse.Ticker(symbol)
                hist_data = ticker.history
                
                if hist_data is None or hist_data.empty:
                    logger.debug(f"   ❌ No data returned from pytse-client for {symbol}.")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue # Try next attempt
                
                if 'close' not in hist_data.columns:
                    logger.debug(f"   ❌ 'close' column not found in downloaded data for {symbol}.")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue # Try next attempt
                
                if not isinstance(hist_data.index, pd.DatetimeIndex):
                    try:
                        hist_data.index = pd.to_datetime(hist_data.index)
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        continue # Try next attempt
                
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.DateOffset(years=int(years_of_data))
                hist_data = hist_data[hist_data.index >= start_date]
                
                if len(hist_data) < 20: # Basic check for sufficient data
                    logger.debug(f"   ❌ Insufficient data points ({len(hist_data)}) after filtering for {symbol}.")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue # Try next attempt
                
                # Ensure index is unique before saving and returning
                if not hist_data.index.is_unique:
                    hist_data = hist_data[~hist_data.index.duplicated(keep='last')]
                    logger.warning(f"   ⚠️ Removed duplicate dates for {symbol}. Kept last entry.")

                # Save to local CSV after successful download
                hist_data.to_csv(ticker_file, encoding='utf-8')
                logger.info(f"   💾 Successfully downloaded and saved {symbol} to {ticker_file.name}")
                return hist_data[['close', 'volume']]
                
            except Exception as e:
                logger.debug(f"   ❌ Error downloading or processing {symbol} from API: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"   ❌ Failed to download {symbol} after {max_retries} attempts.")
                    return None
        return None # Should not be reached if successful or max retries hit

    def download_all_price_data(self, universe: List[str], years_of_data: float, batch_size: int = 50) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
        """
        Download historical price and volume data for the investment universe in batches.
        Returns price df, volume df, and a list of newly failed symbols.
        """
        price_data = {}
        volume_data = {}
        successful_downloads = 0
        newly_failed_symbols = []
        total_stocks = len(universe)
        logger.info(f"📈 Downloading historical data for {total_stocks} stocks in batches of {batch_size}...")
        
        for i in range(0, total_stocks, batch_size):
            batch_symbols = universe[i:i + batch_size]
            logger.info(f"   Processing batch {int(i/batch_size) + 1}/{int(total_stocks/batch_size) + (1 if total_stocks % batch_size > 0 else 0)}")
            for symbol in batch_symbols:
                # download_ticker_data now needs to return both price and volume
                hist_data = self.download_ticker_data(symbol, years_of_data)
                if hist_data is not None and not hist_data.empty and 'close' in hist_data.columns and 'volume' in hist_data.columns:
                    price_data[symbol] = hist_data['close']
                    volume_data[symbol] = hist_data['volume']
                    successful_downloads += 1
                else:
                    newly_failed_symbols.append(symbol)
        
        failed_downloads = len(newly_failed_symbols)
        logger.info(f"📊 Download complete. Successfully fetched: {successful_downloads}, Failed: {failed_downloads}")

        if not price_data:
            logger.warning("⚠️ No price data was successfully downloaded. Returning empty DataFrames.")
            return None, None, newly_failed_symbols
            
        # Create Price DataFrame
        master_price_df = pd.DataFrame(price_data)
        if not master_price_df.index.is_unique:
            master_price_df = master_price_df[~master_price_df.index.duplicated(keep='last')]
        master_price_df.sort_index(inplace=True)
        master_price_df.ffill(inplace=True)
        master_price_df.bfill(inplace=True)
        master_price_df.dropna(axis=1, how='any', inplace=True)
        logger.info(f"   Master Price DataFrame created with {len(master_price_df.columns)} stocks.")

        # Create Volume DataFrame
        master_volume_df = pd.DataFrame(volume_data)
        if not master_volume_df.index.is_unique:
            master_volume_df = master_volume_df[~master_volume_df.index.duplicated(keep='last')]
        master_volume_df.sort_index(inplace=True)
        master_volume_df.ffill(inplace=True)
        master_volume_df.bfill(inplace=True)
        master_volume_df.dropna(axis=1, how='any', inplace=True)
        logger.info(f"   Master Volume DataFrame created with {len(master_volume_df.columns)} stocks.")

        return master_price_df, master_volume_df, newly_failed_symbols

    def calculate_metrics(self, price_df: pd.DataFrame, risk_free_rate: float = 0.35) -> pd.DataFrame:
        """Calculate screening metrics for a given price DataFrame."""
        daily_returns = price_df.pct_change().dropna()
        
        metrics = pd.DataFrame({
            'Return': daily_returns.mean() * 252,
            'Volatility': daily_returns.std() * (252 ** 0.5),
        })
        metrics['Sharpe'] = (metrics['Return'] - risk_free_rate) / metrics['Volatility']
        
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['Max_Drawdown'] = drawdown.min()
        
        return metrics

    def _convert_jalali_to_gregorian(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly converts a DataFrame index from various Jalali date formats (int or str)
        to a standard Gregorian DatetimeIndex.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info("   Index is already a DatetimeIndex. No conversion needed.")
            return df

        try:
            # Attempt direct conversion first, as it might be a standard format
            temp_index = pd.to_datetime(df.index)
            # Sanity check: if the max year is before 1980, it's probably a misinterpretation of a Jalali date.
            if not temp_index.empty and temp_index.year.max() > 1980:
                df.index = temp_index
                logger.info("   ✅ Successfully converted index using pd.to_datetime.")
                df.sort_index(inplace=True)
                return df
            else:
                # This will trigger the Jalali conversion logic below
                if not temp_index.empty:
                    logger.warning(f"   ⚠️ Direct date conversion resulted in an old year ({temp_index.year.max()}). Forcing Jalali conversion.")
                # If temp_index is empty, we also fall through to Jalali conversion
                raise ValueError("Fallback to Jalali conversion")
        except (ValueError, TypeError):
            logger.info("   Direct conversion with pd.to_datetime failed or was forced to fail. Attempting Jalali conversion...")

        new_index = []
        for date_input in df.index:
            try:
                date_str = str(date_input).replace('-', '').replace('/', '')
                if len(date_str) == 8 and date_str.isdigit():
                    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
                    gregorian_date = jdatetime.date(y, m, d).togregorian()
                    new_index.append(pd.to_datetime(gregorian_date))
                else:
                    new_index.append(pd.NaT)
            except (ValueError, TypeError):
                new_index.append(pd.NaT)

        df.index = new_index
        
        # Drop rows where the date conversion failed
        original_rows = len(df)
        df = df[df.index.notna()]
        new_rows = len(df)
        
        if original_rows > new_rows:
            logger.info(f"   Dropped {original_rows - new_rows} rows with invalid date formats during Jalali conversion.")
        
        if not df.empty:
            df.sort_index(inplace=True)
            logger.info("   ✅ Successfully converted and cleaned Jalali index to Gregorian DatetimeIndex.")
        else:
            logger.warning("   ⚠️ DataFrame is empty after date conversion and cleaning.")
            
        return df

    def _download_benchmark_data(self, years_of_data: float) -> Optional[pd.Series]:
        """
        Downloads and cleans historical data for the TSE All-Share Index ('شاخص کل'),
        with local caching to avoid redundant downloads.
        """
        logger.info("📈 Processing benchmark data (TSE All-Share Index)...")

        # 1. Try loading from local CSV cache first
        if self.benchmark_cache_file.exists():
            try:
                hist_data = pd.read_csv(self.benchmark_cache_file, index_col='date', parse_dates=True)
                
                if not hist_data.index.empty:
                    end_date_cached = hist_data.index.max()
                    # Check if cache is recent (e.g., updated within the last day)
                    if pd.Timestamp.now() - end_date_cached < pd.Timedelta(days=1):
                        logger.info(f"✅ Using valid local cache for benchmark data.")
                        
                        # Filter by the required date range
                        end_date = pd.Timestamp.now()
                        start_date = end_date - pd.DateOffset(years=int(years_of_data))
                        filtered_data = hist_data[hist_data.index >= start_date]
                        
                        if len(filtered_data) > 20:
                            return filtered_data['close'].rename('شاخص کل')
                        else:
                             logger.warning("⚠️ Cached benchmark data is insufficient for the requested time horizon. Re-downloading...")
                    else:
                        logger.info("   ⚠️ Benchmark cache is old. Re-downloading...")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not read benchmark cache file: {e}. Re-downloading...")

        # 2. If cache is invalid or not found, download from API
        logger.info("   🌐 Downloading benchmark data from pytse-client API...")
        try:
            # Use the correct class as per the documentation
            hist_data = tse.FinancialIndex(symbol="شاخص کل").history

            if hist_data is None or hist_data.empty:
                logger.error("❌ No data returned from FinancialIndex for 'شاخص کل'.")
                return None

            if 'date' in hist_data.columns and not isinstance(hist_data.index, pd.DatetimeIndex):
                hist_data.set_index('date', inplace=True)
            
            hist_data = self._convert_jalali_to_gregorian(hist_data)

            if hist_data.empty:
                logger.error("❌ CRITICAL: Benchmark data became empty after date conversion. Aborting.")
                return None

            if 'close' not in hist_data.columns:
                logger.error("❌ CRITICAL: 'close' column not found in benchmark data.")
                return None
            
            # Save the full history to cache before filtering
            hist_data.to_csv(self.benchmark_cache_file, encoding='utf-8')
            logger.info(f"   💾 Successfully downloaded and cached benchmark data to {self.benchmark_cache_file.name}")

            # Filter by date range for the current run
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=int(years_of_data))
            hist_data = hist_data[hist_data.index >= start_date]

            if len(hist_data) < 20:
                logger.error(f"❌ Insufficient benchmark data ({len(hist_data)} points) after date filtering.")
                return None

            logger.info("   ✅ Benchmark data processed and cleaned successfully.")
            return hist_data['close'].rename('شاخص کل')

        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while downloading benchmark data: {e}")
            return None

    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        logger.info("🚀 Starting Offline Data Preprocessing...")
        
        symbols_data = self.fetch_all_symbols()
        if not symbols_data:
            logger.error("❌ Could not fetch symbols data. Aborting.")
            return
        
        universe = self.filter_investment_universe(symbols_data)
        if not universe:
            logger.error("❌ No valid symbols in universe. Aborting.")
            return
            
    def _load_failed_symbols(self) -> List[str]:
        """Loads the list of permanently failed symbols from the snapshot file."""
        if self.failed_symbols_cache_file.exists():
            try:
                with open(self.failed_symbols_cache_file, 'r', encoding='utf-8') as f:
                    failed_list = json.load(f)
                    logger.info(f"✅ Loaded {len(failed_list)} symbols from the failed symbols snapshot.")
                    return failed_list
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️ Could not load failed symbols snapshot: {e}. Starting with an empty list.")
        return []

    def _update_failed_symbols_snapshot(self, newly_failed: List[str]):
        """
        Updates the snapshot of failed symbols with any new failures from the current run.
        """
        if not newly_failed:
            return

        # Combine existing failed symbols with new ones
        current_failed = set(self.permanently_failed_symbols)
        newly_failed_set = set(newly_failed)
        updated_failed_set = current_failed.union(newly_failed_set)

        if updated_failed_set != current_failed:
            logger.info(f"   Updating failed symbols snapshot. Adding {len(newly_failed_set - current_failed)} new symbols.")
            updated_list = sorted(list(updated_failed_set))
            try:
                with open(self.failed_symbols_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_list, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Saved updated snapshot with {len(updated_list)} failed symbols.")
                self.permanently_failed_symbols = updated_list
            except IOError as e:
                logger.error(f"❌ Could not update the failed symbols snapshot: {e}")

    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        logger.info("🚀 Starting Offline Data Preprocessing...")
        
        symbols_data = self.fetch_all_symbols()
        if not symbols_data:
            logger.error("❌ Could not fetch symbols data. Aborting.")
            return
        
        universe = self.filter_investment_universe(symbols_data)
        if not universe:
            logger.error("❌ No valid symbols in universe. Aborting.")
            return
            
        max_years = max(self.time_horizons.values())
        self.master_price_df, self.master_volume_df, newly_failed = self.download_all_price_data(universe, max_years, batch_size=50)
        
        self._update_failed_symbols_snapshot(newly_failed)

        if self.master_price_df is None or self.master_price_df.empty:
            logger.error("❌ Master price DataFrame is empty. Aborting.")
            return
        if self.master_volume_df is None or self.master_volume_df.empty:
            logger.error("❌ Master volume DataFrame is empty. Aborting.")
            return

        benchmark_series = self._download_benchmark_data(max_years)
        if benchmark_series is not None:
            self.master_price_df = pd.merge(self.master_price_df, benchmark_series, how='outer', left_index=True, right_index=True)
            self.master_price_df['شاخص کل'].ffill(inplace=True).bfill(inplace=True)
            logger.info("   ✅ Benchmark data merged into master price DataFrame.")

        # --- Save Master DataFrames ---
        try:
            price_data_path = self.cache_dir / 'master_price_data.feather'
            volume_data_path = self.cache_dir / 'master_volume_data.feather'
            
            self.master_price_df.reset_index().rename(columns={'index': 'date'}).to_feather(price_data_path)
            logger.info(f"💾 Successfully saved price data to: {price_data_path}")
            
            self.master_volume_df.reset_index().rename(columns={'index': 'date'}).to_feather(volume_data_path)
            logger.info(f"💾 Successfully saved volume data to: {volume_data_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save master data: {e}")

        # --- Metric Calculation (Optional, can be removed if not needed) ---
        stock_price_df = self.master_price_df.drop(columns=['شاخص کل'], errors='ignore')
        all_metrics = []
        for horizon_name, years in self.time_horizons.items():
            logger.info(f"Calculating metrics for '{horizon_name}' horizon ({years} years)...")
            end_date = self.master_price_df.index.max()
            start_date = end_date - pd.DateOffset(days=int(years * 365.25))
            sliced_df = stock_price_df[stock_price_df.index >= start_date]
            horizon_metrics = self.calculate_metrics(sliced_df)
            horizon_metrics['Horizon'] = horizon_name
            all_metrics.append(horizon_metrics)
            
        if all_metrics:
            final_df = pd.concat(all_metrics)
            output_file = self.cache_dir / config.PREPROCESSED_DATA_FILE
            final_df.reset_index().rename(columns={'index': 'Symbol'}).to_feather(output_file)
            logger.info(f"✅ Analysis-ready metrics saved to '{output_file}'")
        
        logger.info("✅ Preprocessing complete.")

if __name__ == "__main__":
    time_horizons = {
        'short-term': 1.5,
        'mid-term': 3,
        'long-term': 7
    }
    preprocessor = DataPreprocessor(time_horizons=time_horizons, cache_dir=f'../{config.CACHE_DIR}')
    preprocessor.run_preprocessing()