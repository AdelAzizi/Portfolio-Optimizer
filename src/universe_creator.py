import pandas as pd
import pytse_client as tse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UniverseCreator:
    """
    Creates a filtered universe of stocks based on quality and liquidity metrics.
    """
    def __init__(self, output_path: Path = Path("cache/universe.json")):
        """
        Initializes the UniverseCreator.

        Args:
            output_path (Path): The path to save the final universe JSON file.
        """
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_df = pd.DataFrame()
        self.symbols_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()

    def fetch_data(self):
        """
        Fetches real-time market stats and symbol information.
        """
        logging.info("Fetching real-time market-wide statistics...")
        self.stats_df = tse.get_stats()
        logging.info(f"Fetched {len(self.stats_df)} initial symbols from stats.")

        logging.info("Fetching general symbol information...")
        self.symbols_df = tse.symbols_data()
        logging.info(f"Fetched {len(self.symbols_df)} symbol details.")

        # Merge the two data sources
        self.merged_df = pd.merge(self.stats_df, self.symbols_df, on='symbol')
        logging.info(f"Successfully merged data. Total symbols: {len(self.merged_df)}")

    def apply_filters(self):
        """
        Applies a multi-stage filtering process to the merged data.
        """
        if self.merged_df.empty:
            logging.error("No data to filter. Run fetch_data() first.")
            return

        initial_count = len(self.merged_df)
        logging.info(f"Starting filtering process with {initial_count} symbols.")

        # 1. Basic Filter
        df = self.merged_df
        df = df[df['market'].isin(['بورس', 'فرابورس'])]
        post_market_filter_count = len(df)
        logging.info(f"Symbols after market filter ('بورس', 'فرابورس'): {post_market_filter_count}")

        excluded_keywords = ['صندوق', 'حق تقدم']
        for keyword in excluded_keywords:
            df = df[~df['name'].str.contains(keyword)]
        post_keyword_filter_count = len(df)
        logging.info(f"Symbols after excluding keywords {excluded_keywords}: {post_keyword_filter_count}")

        # 2. Quantitative Filter
        # Liquidity
        df = df[df['volume_of_trans'] > 100000]
        post_liquidity_filter_count = len(df)
        logging.info(f"Symbols after liquidity filter (volume > 100,000): {post_liquidity_filter_count}")

        # Size
        df = df[df['val_company_last_day'] > 1e12]
        post_size_filter_count = len(df)
        logging.info(f"Symbols after size filter (market cap > 1e12): {post_size_filter_count}")

        # Data Quality (Fundamental Pre-check)
        # Ensure 'EPS' is numeric and not zero or NaN
        df['EPS'] = pd.to_numeric(df['EPS'], errors='coerce')
        df = df.dropna(subset=['EPS'])
        df = df[df['EPS'] != 0]
        final_count = len(df)
        logging.info(f"Symbols after P/E filter (valid, positive EPS): {final_count}")

        self.filtered_df = df
        logging.info(f"\n--- Filtering Summary ---")
        logging.info(f"Total symbols checked: {initial_count}")
        logging.info(f"Remaining after market filter: {post_market_filter_count}")
        logging.info(f"Remaining after keyword filter: {post_keyword_filter_count}")
        logging.info(f"Remaining after liquidity filter: {post_liquidity_filter_count}")
        logging.info(f"Remaining after size filter: {post_size_filter_count}")
        logging.info(f"Final universe count: {final_count}")
        logging.info(f"-------------------------\n")


    def save_universe(self):
        """
        Saves the final list of symbols to a JSON file.
        """
        if self.filtered_df.empty:
            logging.warning("Filtered DataFrame is empty. No universe file will be saved.")
            return

        final_symbols = self.filtered_df['symbol'].tolist()
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_symbols, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved {len(final_symbols)} symbols to {self.output_path}")

    def run(self):
        """
        Executes the full universe creation pipeline.
        """
        self.fetch_data()
        self.apply_filters()
        self.save_universe()

if __name__ == '__main__':
    creator = UniverseCreator()
    creator.run()