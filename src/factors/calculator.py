# -*- coding: utf-8 -*-
"""
This module contains functions for calculating various investment factors.
"""
import pandas as pd
import numpy as np

def calculate_momentum(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Calculates the price momentum (rate of return) for all stocks over a given window.

    Momentum is calculated as: (Price_Today / Price_N_days_ago) - 1.
    This implementation handles potential division by zero or missing data points by
    returning NaN for those cases.

    Args:
        prices (pd.DataFrame): DataFrame of historical prices with dates as index.
        window (int): The lookback window in days for momentum calculation.
                      Defaults to 252 (approx. 1 year).

    Returns:
        pd.Series: A series containing the momentum value for each stock.
    """
    # Ensure prices are numeric
    numeric_prices = prices.apply(pd.to_numeric, errors='coerce')
    
    # Get the price 'window' days ago
    past_prices = numeric_prices.shift(window)
    
    # Calculate momentum, handling potential division by zero
    momentum = numeric_prices.div(past_prices) - 1
    
    # Replace infinite values that can occur if past_price was 0
    momentum.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return momentum

def calculate_volatility(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Calculates the annualized volatility for all stocks.

    Args:
        prices (pd.DataFrame): DataFrame of historical prices with dates as index.
        window (int): The lookback window in days for volatility calculation.

    Returns:
        pd.Series: A series containing the volatility value for each stock.
    """
    daily_returns = prices.pct_change()
    volatility = daily_returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

def calculate_reversal(prices: pd.DataFrame, window: int = 756) -> pd.Series:
    """
    Calculates the long-term reversal for all stocks.
    This is the inverse of the long-term momentum.

    Args:
        prices (pd.DataFrame): DataFrame of historical prices with dates as index.
        window (int): The lookback window in days for reversal calculation.

    Returns:
        pd.Series: A series containing the reversal value for each stock.
    """
    # Using the same robust momentum calculation
    long_term_momentum = calculate_momentum(prices, window=window)
    reversal = -1 * long_term_momentum
    return reversal


def calculate_momentum_6m(prices: pd.DataFrame) -> pd.Series:
    """
    Calculates the 6-month price momentum (approximately 126 trading days).

    Args:
        prices (pd.DataFrame): DataFrame of historical prices with dates as index.

    Returns:
        pd.Series: A series containing the 6-month momentum value for each stock.
    """
    return calculate_momentum(prices, window=126)


def calculate_momentum_12m(prices: pd.DataFrame) -> pd.Series:
    """
    Calculates the 12-month price momentum (approximately 252 trading days).

    Args:
        prices (pd.DataFrame): DataFrame of historical prices with dates as index.

    Returns:
        pd.Series: A series containing the 12-month momentum value for each stock.
    """
    return calculate_momentum(prices, window=252)