import pytse_client as tse
import pandas as pd
import json
import os
import logging
import time
from pathlib import Path

# Import configurations from the central config file
from src.config import UNIVERSE_CREATOR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class UniverseCreator:
    """
    Creates a stock universe by fetching market-wide stats using pytse_client, 
    applying quantitative filters defined in the config, and saving the resulting list of symbols.
    """
    def __init__(self):
        """
        Initializes the UniverseCreator using settings from the config file.
        """
        self.logger = logging.getLogger('UniverseCreator')
        # Use settings from config for path definitions
        self.cache_dir = Path(UNIVERSE_CREATOR["CACHE_DIR"])
        self.universe_path = self.cache_dir / UNIVERSE_CREATOR["UNIVERSE_FILENAME"]
        self._setup_directories()

    def _setup_directories(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created directory: {self.cache_dir}")

    def run(self):
        """
        Fetches market-wide stats, applies quantitative filters from config, 
        and saves the resulting stock universe.
        """
        self.logger.info("Starting efficient universe creation process using configurations...")
        
        # Check if cache is valid using validity period from config
        if self.universe_path.exists():
            last_modified_hours = (time.time() - self.universe_path.stat().st_mtime) / 3600
            if last_modified_hours < UNIVERSE_CREATOR["CACHE_VALIDITY_HOURS"]:
                self.logger.info(f"Found valid cache for universe (created {last_modified_hours:.2f} hours ago). Loading from cache.")
                try:
                    with open(self.universe_path, 'r', encoding='utf-8') as f:
                        cached_universe = json.load(f)
                    if cached_universe:
                        self.logger.info(f"✅ Successfully loaded {len(cached_universe)} symbols from cache.")
                        return
                    else:
                        self.logger.warning("Cache file is empty. Re-creating universe.")
                except Exception as e:
                    self.logger.warning(f"Could not read cache file: {e}. Re-creating universe.")
            else:
                self.logger.info(f"Universe cache is outdated ({last_modified_hours:.2f} hours old). Re-creating universe.")
        
        try:
            # 1. Fetch all market-wide statistics at once
            self.logger.info("Fetching market-wide stats using pytse_client.get_stats()...")
            market_stats_df = tse.get_stats(to_csv=False)
            
            if market_stats_df is None or market_stats_df.empty:
                self.logger.error("Failed to fetch market stats from pytse_client. The returned DataFrame is empty.")
                return
            
            self.logger.info(f"Successfully fetched stats for {len(market_stats_df)} symbols from pytse_client.")

            # Convert columns to numeric for filtering, coercing errors to NaN
            numeric_cols = ['volume_of_trans', 'val_company_last_day']
            for col in numeric_cols:
                market_stats_df[col] = pd.to_numeric(market_stats_df[col], errors='coerce')
            
            # Drop rows where conversion failed for essential columns
            market_stats_df.dropna(subset=numeric_cols, inplace=True)
            self.logger.info(f"After cleaning non-numeric data, {len(market_stats_df)} symbols remain.")

            # 2. Apply a Multi-Layer Quantitative Filter using settings from config
            filters = UNIVERSE_CREATOR["FILTERS"]
            
            # a. Market Type Filter
            market_flow_types = filters["MARKET_FLOW_TYPES"]
            filtered_df = market_stats_df[market_stats_df['flow'].isin(market_flow_types)].copy()
            self.logger.info(f"After market type filter (Bourse/Fara Bourse): {len(filtered_df)} stocks remaining.")

            # b. Liquidity Filter
            min_liquidity = filters["MIN_LIQUIDITY"]
            filtered_df = filtered_df[filtered_df['volume_of_trans'] > min_liquidity]
            self.logger.info(f"After liquidity filter (>{min_liquidity} volume): {len(filtered_df)} stocks remaining.")

            # c. Size Filter
            min_market_cap = filters["MIN_MARKET_CAP"]
            filtered_df = filtered_df[filtered_df['val_company_last_day'] > min_market_cap]
            self.logger.info(f"After size filter (>{min_market_cap} market cap): {len(filtered_df)} stocks remaining.")

            # 3. Extract and Save the Final Universe
            if filtered_df.empty:
                self.logger.warning("No stocks passed all filters. The universe will be empty.")
                final_universe = []
            else:
                final_universe = filtered_df['symbol'].tolist()
                final_universe.sort() # Sort alphabetically as requested
                self.logger.info(f"Final universe contains {len(final_universe)} stocks.")

            with open(self.universe_path, 'w', encoding='utf-8') as f:
                json.dump(final_universe, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Successfully saved universe of {len(final_universe)} stocks to {self.universe_path}")

        except KeyError as e:
            self.logger.error(f"A required column is missing from the fetched data: {e}")
            self.logger.error("Please check the column names provided by the 'pytse_client' library's get_stats() function.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during universe creation: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    # This allows the script to be run directly.
    creator = UniverseCreator()
    creator.run()