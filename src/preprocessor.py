# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Master Analysis Data Builder
# Description: Integrates clean fundamental data with raw price history to
#              build a single, final, analysis-ready dataset for optimization.
#              This script operates entirely offline.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'preprocessor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Builds the final analysis-ready dataset by merging fundamental and price data.
    """
    def __init__(self, data_dir: str = 'data', cache_dir: str = 'cache'):
        self.data_dir = PROJECT_ROOT / data_dir
        self.cache_dir = PROJECT_ROOT / cache_dir
        self.price_data_dir = self.data_dir / 'tickers_data'
        
        self.fundamental_data_file = self.data_dir / 'fundamental_data.feather'
        self.output_file = self.cache_dir / 'analysis_ready_data.feather'
        
        logger.info(f"Fundamental data source: {self.fundamental_data_file}")
        logger.info(f"Price data source: {self.price_data_dir}")
        logger.info(f"Output file: {self.output_file}")

    def _load_fundamental_data(self) -> pd.DataFrame:
        """Loads the clean fundamental data, which defines the stock universe."""
        logger.info("--- Stage 1: Loading Fundamental Data ---")
        if not self.fundamental_data_file.exists():
            logger.error(f"CRITICAL: Fundamental data file not found at '{self.fundamental_data_file}'.")
            logger.error("Please run the fundamental_collector.py script first.")
            raise FileNotFoundError("Clean fundamental data file is missing.")
        
        fundamental_df = pd.read_feather(self.fundamental_data_file).set_index('symbol')
        logger.info(f"✅ Loaded fundamental data for {len(fundamental_df)} symbols.")
        return fundamental_df

    def _load_and_process_price_data(self, symbols: list) -> pd.DataFrame:
        """Loads raw price CSVs for the given symbols and creates a clean master price DataFrame."""
        logger.info("--- Stage 2: Loading and Processing Price Data ---")
        price_data = {}
        for symbol in symbols:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, usecols=['date', 'close'], parse_dates=['date'], index_col='date')
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

    def _calculate_quantitative_metrics(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates price-based metrics like returns, volatility, and momentum."""
        logger.info("--- Stage 3: Calculating Quantitative Metrics ---")
        # Calculate daily returns
        returns = price_df.pct_change()
        
        # Annualized Return (compounded)
        # (1 + mean_daily_return)^252 - 1
        annualized_return = (1 + returns.mean())**252 - 1
        
        # Annualized Volatility
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Momentum
        momentum_6m = price_df.iloc[-1] / price_df.iloc[-126] - 1
        momentum_12m = price_df.iloc[-1] / price_df.iloc[-252] - 1
        
        metrics_df = pd.DataFrame({
            'Return': annualized_return,
            'Volatility': annualized_volatility,
            'Momentum_6M': momentum_6m,
            'Momentum_12M': momentum_12m
        })
        logger.info("✅ Quantitative metrics calculated.")
        return metrics_df

    def run(self):
        """Executes the full offline data preprocessing and merging pipeline."""
        logger.info("🚀 Starting Master Analysis Data Builder...")
        try:
            # Stage 1
            fundamental_df = self._load_fundamental_data()
            
            # Stage 2
            symbols = fundamental_df.index.tolist()
            master_price_df = self._load_and_process_price_data(symbols)
            
            # Stage 3
            quantitative_df = self._calculate_quantitative_metrics(master_price_df)
            
            # Stage 4: Merge and Finalize
            logger.info("--- Stage 4: Merging Fundamental and Quantitative Data ---")
            master_analysis_df = fundamental_df.join(quantitative_df)
            
            # Drop any rows that couldn't be joined properly
            master_analysis_df.dropna(inplace=True)
            
            logger.info("✅ Successfully merged dataframes.")
            
            # Stage 5: Save the Final Output
            logger.info("--- Stage 5: Saving Final Analysis-Ready Data ---")
            self.output_file.parent.mkdir(exist_ok=True)
            master_analysis_df.reset_index().to_feather(self.output_file)
            
            logger.info("="*50)
            logger.info("🎉 MASTER PREPROCESSING COMPLETE 🎉")
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
    preprocessor = DataPreprocessor()
    preprocessor.run()