# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Total Index Plotter
# Description: Reads the downloaded Total Index data and plots its closing price.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Define Project Root and Data Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'tickers_data' / 'شاخص کل.csv'
OUTPUT_CHART = PROJECT_ROOT / 'analysis' / 'total_index_chart.png'

def plot_index():
    """
    Reads the Total Index CSV file and generates a plot of the closing price.
    """
    print(f"Attempting to read data from: {DATA_FILE}")
    if not DATA_FILE.exists():
        print(f"❌ ERROR: Data file not found at {DATA_FILE}")
        print("Please run the downloader script first to fetch the data.")
        return

    try:
        # Load the data
        df = pd.read_csv(DATA_FILE)
        print("✅ Data loaded successfully.")

        # --- Data Preparation ---
        # Ensure 'date' column exists and is in datetime format
        if 'date' not in df.columns:
            print("❌ ERROR: 'date' column not found in the CSV file.")
            return
        df['date'] = pd.to_datetime(df['date'])

        # Set date as index for better plotting
        df.set_index('date', inplace=True)

        # Ensure 'close' column exists
        if 'close' not in df.columns:
            print("❌ ERROR: 'close' column not found in the CSV file.")
            return

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        ax.plot(df.index, df['close'], label='Total Index (Close)', color='cyan')

        # --- Formatting ---
        ax.set_title('Historical Performance of Total Index (شاخص کل)', fontsize=18)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Index Value', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        fig.tight_layout()

        # Save the plot
        plt.savefig(OUTPUT_CHART, dpi=300)
        print(f"✅ Chart saved successfully to: {OUTPUT_CHART}")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    plot_index()