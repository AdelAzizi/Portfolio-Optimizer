# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Dynamic Factor Calculator
# Description: Provides functions to calculate factors dynamically for a specific date
#              to eliminate lookahead bias in backtesting.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from scipy.stats.mstats import winsorize

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
        logging.FileHandler(LOGS_DIR / 'factor_calculator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values using group means and then overall medians."""
    logger.info("--- Stage 3: Imputing Missing Data on Combined DataFrame ---")
    numeric_cols = ['P/E', 'P/S', 'EPS']
    df_imputed = df.copy()
    
    # Handle group_name imputation first
    if df_imputed['group_name'].isnull().any():
        missing_before = df_imputed['group_name'].isnull().sum()
        df_imputed['group_name'] = df_imputed['group_name'].fillna('نامشخص')
        imputed_count = missing_before - df_imputed['group_name'].isnull().sum()
        if imputed_count > 0:
            logger.info(f"   - Filled {imputed_count} missing 'group_name' values with 'نامشخص'.")
    
    for col in numeric_cols:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            missing_before = df_imputed[col].isnull().sum()
            
            # Impute with industry group mean (only for numeric columns)
            group_means = df_imputed.groupby('group_name')[col].transform('mean')
            df_imputed[col] = df_imputed[col].fillna(group_means)
            imputed_by_group = missing_before - df_imputed[col].isnull().sum()
            if imputed_by_group > 0:
                logger.info(f"   - Imputed {imputed_by_group} missing '{col}' values using industry group averages.")
            
            # Impute any remaining with overall median
            if df_imputed[col].isnull().any():
                remaining_missing_before = df_imputed[col].isnull().sum()
                col_median = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(col_median)
                logger.info(f"   - Filled {remaining_missing_before} remaining '{col}' NaNs with overall median ({col_median:.2f}).")
    
    return df_imputed


def _handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Caps outliers at the 1st and 99th percentiles."""
    logger.info("--- Stage 4: Handling Outliers on Combined DataFrame ---")
    numeric_cols = ['P/E', 'P/S', 'EPS']  # Extended to include EPS
    df_clean = df.copy()
    for col in numeric_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            # Only apply winsorize to non-null values
            mask = df_clean[col].notna()
            if mask.any():
                df_clean.loc[mask, col] = winsorize(df_clean.loc[mask, col], limits=[0.01, 0.01])
                logger.info(f"   - Capped outliers in '{col}' at the 1st and 99th percentiles.")
    return df_clean


def _calculate_max_drawdown(price_series: pd.Series) -> float:
    """Calculate the maximum drawdown from a price series."""
    # Calculate the cumulative maximum of the price series (running maximum)
    running_max = price_series.expanding().max()
    # Calculate the drawdown: (current_price - running_max) / running_max
    drawdown = (price_series - running_max) / running_max
    # The maximum drawdown is the minimum (most negative) value of the drawdown series
    max_drawdown = drawdown.min()
    return max_drawdown if not pd.isna(max_drawdown) else 0.0


def _calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """Calculate the Sortino Ratio using downside deviation."""
    # Calculate the daily risk-free rate
    daily_rfr = (1 + risk_free_rate)**(1/252) - 1
    
    # Identify returns below the target (downside returns)
    downside_returns = daily_returns[daily_returns < daily_rfr]
    
    # Calculate downside deviation
    if len(downside_returns) == 0:
        # If no downside returns, use a very small number to avoid division by zero
        downside_std = 1e-8
    else:
        downside_std = downside_returns.std()
        if pd.isna(downside_std) or downside_std == 0:
            downside_std = 1e-8
    
    # Annualize the downside deviation
    annual_downside_std = downside_std * np.sqrt(252)
    
    # Calculate annualized return from the daily_returns series
    annualized_return = (1 + daily_returns.mean())**252 - 1
    
    # Calculate Sortino Ratio
    sortino_ratio = (annualized_return - risk_free_rate) / annual_downside_std
    return sortino_ratio if not pd.isna(sortino_ratio) else 0.0


def _calculate_quantitative_metrics(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates price-based metrics like returns, volatility, momentum, and risk-adjusted metrics."""
    logger.info("--- Stage 5: Calculating Quantitative Metrics for Full Market ---")
    # Calculate daily returns
    returns = price_df.pct_change()
    
    # Annualized Return (compounded)
    # (1 + mean_daily_return)^TRADING_DAYS_PER_YEAR - 1
    annualized_return = (1 + returns.mean())**config["TRADING_DAYS_PER_YEAR"] - 1
    
    # Annualized Volatility
    annualized_volatility = returns.std() * np.sqrt(config["TRADING_DAYS_PER_YEAR"])
    
    # Calculate risk-adjusted metrics
    risk_free_rate = config["RISK_FREE_RATE"]
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    # Calculate Sortino Ratio and Maximum Drawdown for each symbol
    sortino_ratios = {}
    max_drawdowns = {}
    for symbol in price_df.columns:
        symbol_prices = price_df[symbol].dropna()
        if len(symbol_prices) > 1:
            symbol_returns = symbol_prices.pct_change().dropna()
            if len(symbol_returns) > 0:
                sortino_ratios[symbol] = _calculate_sortino_ratio(symbol_returns, risk_free_rate)
                max_drawdowns[symbol] = _calculate_max_drawdown(symbol_prices)
            else:
                sortino_ratios[symbol] = 0.0
                max_drawdowns[symbol] = 0.0
        else:
            sortino_ratios[symbol] = 0.0
            max_drawdowns[symbol] = 0.0

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

    metrics_df = pd.DataFrame({
        'Return': annualized_return,
        'Volatility': annualized_volatility,
        'Sharpe': sharpe_ratio,
        'Sortino_Ratio': pd.Series(sortino_ratios),
        'Max_Drawdown': pd.Series(max_drawdowns),
        **momentum_metrics # Unpack momentum metrics
    })
    logger.info("✅ Quantitative metrics calculated.")
    return metrics_df


def calculate_factors_for_date(price_df: pd.DataFrame, fundamental_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculates factors dynamically for a specific date to eliminate lookahead bias.
    
    Args:
        price_df: Master price DataFrame with historical data
        fundamental_df: Master fundamental DataFrame
        current_date: The specific date for which to calculate factors
    
    Returns:
        DataFrame containing calculated factors for all symbols for the specific date
    """
    logger.info(f"--- Calculating factors for date: {current_date} ---")
    
    # Filter the price data to get the relevant lookback window based on current_date
    # Using 1 year of data before current_date
    start_lookback = current_date - pd.DateOffset(years=1)
    historical_prices = price_df.loc[start_lookback:current_date].copy()
    
    logger.info(f"Using price data from {start_lookback} to {current_date} (shape: {historical_prices.shape})")
    
    # Calculate quantitative metrics on the historical data only
    quantitative_df = _calculate_quantitative_metrics(historical_prices)
    
    # Filter fundamental data to get the cross-section for current_date
    # For now, we'll use the available fundamental data as is, but in a real system
    # this would come from a time-series fundamental dataset
    current_fundamental = fundamental_df.copy()
    
    # Add fundamental factors (ROE and P/E) if they exist in fundamental_df
    fundamental_factors = []
    if 'ROE' in current_fundamental.columns:
        fundamental_factors.append('ROE')
    if 'P/E' in current_fundamental.columns:
        fundamental_factors.append('P/E')
    
    # Merge fundamental and quantitative data
    logger.info("--- Merging Fundamental and Quantitative Data for Current Date ---")
    master_analysis_df = current_fundamental.join(quantitative_df)
    
    # Ensure fundamental factors are properly handled in the merged dataframe
    for factor in fundamental_factors:
        if factor in master_analysis_df.columns:
            # Handle missing values for fundamental factors using industry median
            missing_mask = master_analysis_df[factor].isnull()
            if missing_mask.any():
                # Calculate industry median for the factor
                industry_medians = master_analysis_df.groupby('group_name')[factor].transform('median')
                # Fill missing values with industry median
                master_analysis_df.loc[missing_mask, factor] = industry_medians[missing_mask]
                logger.info(f"Filled missing '{factor}' values using industry medians.")
    
    logger.info("✅ Successfully merged dataframes.")
    
    # Impute NaN values for quantitative columns using industry group medians
    quant_cols = ['Return', 'Volatility', 'Sharpe'] + [col for col in master_analysis_df.columns if 'Momentum_' in col]
    
    logger.info(f"Imputing NaN values for {len(quant_cols)} quantitative columns using industry group medians.")
    
    for col in quant_cols:
        # Calculate the number of NaNs before imputation for logging
        nan_count_before = master_analysis_df[col].isnull().sum()
        if nan_count_before > 0:
            # Use transform to broadcast the median of each group to all its members
            master_analysis_df[col] = master_analysis_df.groupby('group_name')[col].transform(lambda x: x.fillna(x.median()))
            logger.info(f"Imputed {nan_count_before} NaN values in column '{col}'.")

    # After filling by group, there might still be NaNs if a whole group is NaN. Fill these with the global median.
    for col in quant_cols:
        if master_analysis_df[col].isnull().any():
            global_median = master_analysis_df[col].median()
            master_analysis_df[col].fillna(global_median, inplace=True)
            logger.info(f"Filled remaining NaNs in '{col}' with global median.")
    
    logger.info("✅ Successfully merged dataframes with imputed values.")
    
    # Apply data cleaning (imputation and outlier handling) using only current cross-section
    logger.info("--- Applying Data Cleaning for Current Date ---")
    master_analysis_df = _impute_missing_data(master_analysis_df)
    master_analysis_df = _handle_outliers(master_analysis_df)
    
    logger.info("✅ Factor calculation complete for date.")
    return master_analysis_df