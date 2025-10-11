# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Full Market Master Analysis Data Builder
# Description: Integrates clean full market fundamental data with raw price history
#              to build a single, final, analysis-ready dataset for optimization.
#              This script operates entirely offline.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

# --- Import Configuration ---
from src.config import FULL_MARKET_PREPROCESSOR as config


# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'full_market_preprocessor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullMarketDataPreprocessor:
    """
    Builds the final analysis-ready dataset by merging full market fundamental and price data.
    """
    def __init__(self, data_dir: str = None, cache_dir: str = None):
        # Use config values if not provided
        self.data_dir = PROJECT_ROOT / (data_dir or config["DATA_DIR"])
        self.cache_dir = PROJECT_ROOT / (cache_dir or config["CACHE_DIR"])
        self.price_data_dir = self.data_dir / 'full_market_data_csvs'
        
        self.fundamental_data_file = self.cache_dir / config["FUNDAMENTAL_FILE"]
        self.output_file = self.cache_dir / config["OUTPUT_FILE"]
        self.processing_log_file = self.cache_dir / "preprocessor_processing_log.json"  # Track when preprocessing was last done
        
        # Load processing log
        self.processing_log = self._load_processing_log()
        
        logger.info(f"Fundamental data source: {self.fundamental_data_file}")
        logger.info(f"Price data source: {self.price_data_dir}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Processing log file: {self.processing_log_file}")

    def _load_fundamental_data(self) -> pd.DataFrame:
        """Loads the clean fundamental data, which defines the stock universe."""
        logger.info("--- Stage 1: Loading Full Market Fundamental Data ---")
        if not self.fundamental_data_file.exists():
            logger.error(f"CRITICAL: Full fundamental data file not found at '{self.fundamental_data_file}'.")
            logger.error("Please run the full_market_fundamental_collector.py script first.")
            raise FileNotFoundError("Clean full fundamental data file is missing.")
        
        fundamental_df = pd.read_feather(self.fundamental_data_file).set_index('symbol')
        logger.info(f"✅ Loaded fundamental data for {len(fundamental_df)} symbols.")
        return fundamental_df

    def _load_and_process_price_data(self, symbols: list) -> pd.DataFrame:
        """Loads raw price CSVs for the given symbols and creates a clean master price DataFrame."""
        logger.info("--- Stage 2: Loading and Processing Full Market Price Data ---")
        price_data = {}
        for symbol in symbols:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, usecols=['date', 'close'], parse_dates=['date'], index_col='date')
                    # Remove duplicate dates by keeping the last occurrence
                    if not df.index.is_unique:
                        df = df[~df.index.duplicated(keep='last')]
                    price_data[symbol] = df['close']
                except Exception as e:
                    logger.warning(f"⚠️ Could not process price file for {symbol}: {e}")
            else:
                logger.warning(f"⚠️ Price CSV for symbol '{symbol}' not found. Skipping.")
        
        if not price_data:
            logger.error("CRITICAL: No price data could be loaded. Aborting.")
            raise ValueError("Failed to load any price data.")

        logger.info(f"🛠️ Creating and cleaning master price DataFrame for {len(price_data)} symbols...")
        master_price_df = pd.DataFrame(price_data)
        master_price_df.sort_index(inplace=True)
        
        # Fill NaNs from non-overlapping trading days
        master_price_df.ffill(inplace=True)
        master_price_df.bfill(inplace=True)
        
        logger.info("✅ Master price DataFrame created and cleaned.")
        return master_price_df

    def _load_processing_log(self) -> dict:
        """Loads the processing log that tracks when preprocessing was last done."""
        if self.processing_log_file.exists():
            try:
                with open(self.processing_log_file, 'r', encoding='utf-8') as f:
                    processing_log = json.load(f)
                    logger.info(f"Loaded processing log.")
                    return processing_log
            except (json.JSONDecodeError, TypeError):
                logger.warning("Processing log file is corrupted or empty. Starting with an empty log.")
                return {}
        return {}

    def _save_processing_log(self):
        """Saves the current processing log to a JSON file."""
        with open(self.processing_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved processing log.")

    def _update_processing_log(self):
        """Update the processing log with the current timestamp."""
        from datetime import datetime
        self.processing_log['last_preprocessing'] = datetime.now().isoformat()
        self._save_processing_log()

    def _calculate_quantitative_metrics(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates price-based metrics like returns, volatility, and momentum."""
        logger.info("--- Stage 3: Calculating Quantitative Metrics for Full Market ---")
        # Calculate daily returns
        returns = price_df.pct_change()
        
        # Annualized Return (compounded)
        # (1 + mean_daily_return)^TRADING_DAYS_PER_YEAR - 1
        annualized_return = (1 + returns.mean())**config["TRADING_DAYS_PER_YEAR"] - 1
        
        # Annualized Volatility
        annualized_volatility = returns.std() * np.sqrt(config["TRADING_DAYS_PER_YEAR"])
        
        # Momentum with fallback for short history symbols
        momentum_metrics = {}
        for period_name, periods in config["MOMENTUM_PERIODS"].items():
            momentum = price_df.pct_change(periods=periods).iloc[-1]
            # Fallback if NaN (for symbols with insufficient history)
            if pd.isna(momentum).any() if hasattr(momentum, 'any') else pd.isna(momentum):
                if len(price_df) > periods // 2:
                    shorter_period = periods // 2
                    momentum = price_df.pct_change(periods=shorter_period).iloc[-1]
            momentum_metrics[f'Momentum_{period_name}'] = momentum

        # Calculate Sharpe Ratio
        risk_free_rate = config["RISK_FREE_RATE"]
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        metrics_df = pd.DataFrame({
            'Return': annualized_return,
            'Volatility': annualized_volatility,
            'Sharpe': sharpe_ratio,
            **momentum_metrics # Unpack momentum metrics
        })
        logger.info("✅ Quantitative metrics calculated.")
        return metrics_df

    def run(self):
        """Executes the full offline data preprocessing and merging pipeline."""
        logger.info("🚀 Starting Full Market Master Analysis Data Builder...")
        
        # Check if output file exists and is up-to-date compared to input files
        if self.output_file.exists():
            try:
                output_modified = self.output_file.stat().st_mtime
                
                # Check fundamental data file
                if self.fundamental_data_file.exists():
                    fundamental_modified = self.fundamental_data_file.stat().st_mtime
                    if fundamental_modified > output_modified:
                        logger.info("Fundamental data file has been updated since last preprocessing. Re-running.")
                    else:
                        # Check if any price data files are newer than output
                        symbols = []
                        if self.price_data_dir.exists():
                            # Get all CSV files in the price data directory
                            csv_files = list(self.price_data_dir.glob("*.csv"))
                            if csv_files:
                                latest_price_modified = max(f.stat().st_mtime for f in csv_files)
                                if latest_price_modified > output_modified:
                                    logger.info("Some price data files have been updated since last preprocessing. Re-running.")
                                else:
                                    logger.info("Output file is up-to-date with all input files. Skipping preprocessing.")
                                    # Check if we should skip based on processing log as well
                                    if 'last_preprocessing' in self.processing_log:
                                        from datetime import datetime
                                        try:
                                            last_run_time = datetime.fromisoformat(self.processing_log['last_preprocessing'])
                                            if output_modified >= last_run_time.timestamp():
                                                logger.info("✅ Output file is already up-to-date. Preprocessing skipped.")
                                                return
                                        except ValueError:
                                            pass
                            else:
                                logger.info("No price data files found in directory. Proceeding with preprocessing.")
                        else:
                            logger.info("Price data directory not found. Proceeding with preprocessing.")
                else:
                    logger.info("Fundamental data file not found. Proceeding with preprocessing.")
            
            except Exception as e:
                logger.warning(f"Could not check file modification times: {e}. Proceeding with preprocessing.")
        
        try:
            # Stage 1
            fundamental_df = self._load_fundamental_data()
            
            # Stage 2
            symbols = fundamental_df.index.tolist()
            master_price_df = self._load_and_process_price_data(symbols)
            
            # Stage 3
            quantitative_df = self._calculate_quantitative_metrics(master_price_df)
            
            # Stage 4: Merge and Finalize
            logger.info("--- Stage 4: Merging Full Market Fundamental and Quantitative Data ---")
            master_analysis_df = fundamental_df.join(quantitative_df)
            
            # Drop any rows that couldn't be joined properly
            master_analysis_df.dropna(inplace=True)
            
            logger.info("✅ Successfully merged dataframes.")
            
            # Stage 5: Save the Final Output
            logger.info("--- Stage 5: Saving Final Full Market Analysis-Ready Data ---")
            self.output_file.parent.mkdir(exist_ok=True)
            master_analysis_df.reset_index().to_feather(self.output_file)
            
            # Update processing log
            self._update_processing_log()
            
            logger.info("="*50)
            logger.info("🎉 FULL MARKET MASTER PREPROCESSING COMPLETE 🎉")
            logger.info("="*50)
            logger.info(f"Final analysis-ready DataFrame shape: {master_analysis_df.shape}")
            logger.info(f"💾 Successfully saved final data to: {self.output_file}")
            logger.info("Final DataFrame preview:")
            logger.info("\n" + master_analysis_df.head().to_string())

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Halting execution due to a critical error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    preprocessor = FullMarketDataPreprocessor()
    preprocessor.run()
