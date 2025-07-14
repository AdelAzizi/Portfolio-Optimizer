import pytse_client as tse
import pandas as pd
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class UniverseCreator:
    """
    Creates a stock universe by fetching market-wide stats using pytse_client, 
    applying quantitative filters, and saving the resulting list of symbols.
    """
    def __init__(self):
        """
        Initializes the UniverseCreator.
        """
        self.logger = logging.getLogger('UniverseCreator')
        self.cache_dir = 'cache'
        self.universe_path = os.path.join(self.cache_dir, 'universe.json')
        self._setup_directories()

    def _setup_directories(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created directory: {self.cache_dir}")

    def run(self):
        """
        Fetches market-wide stats using an efficient single API call from pytse_client, 
        applies quantitative filters, and saves the resulting stock universe.
        """
        self.logger.info("Starting efficient universe creation process using pytse_client...")
        try:
            # 1. Fetch all market-wide statistics at once using pytse_client
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

            # 2. Apply a Multi-Layer Quantitative Filter
            # a. Market Type Filter: Keep Bourse (1) and Fara Bourse (2)
            filtered_df = market_stats_df[market_stats_df['flow'].isin([1, 2])].copy()
            self.logger.info(f"After market type filter (Bourse/Fara Bourse): {len(filtered_df)} stocks remaining.")

            # b. Liquidity Filter: Volume greater than 100,000
            filtered_df = filtered_df[filtered_df['volume_of_trans'] > 100000]
            self.logger.info(f"After liquidity filter (>100,000 volume): {len(filtered_df)} stocks remaining.")

            # c. Size Filter: Market cap greater than 1e12 Toman
            filtered_df = filtered_df[filtered_df['val_company_last_day'] > 1e12]
            self.logger.info(f"After size filter (>1e12 market cap): {len(filtered_df)} stocks remaining.")

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