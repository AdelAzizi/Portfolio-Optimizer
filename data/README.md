# Data Directory

This directory contains all the raw and processed data used by the Quantitative Investment System.

## Structure

-   `tickers_data/`: This subdirectory holds the raw historical price and volume data for individual stock tickers, with each ticker's data stored in a separate CSV file. This data is downloaded and cached by the `preprocessor.py` script.