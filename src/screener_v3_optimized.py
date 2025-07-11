# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Interactive Portfolio Optimizer v3.1
# Description: An interactive tool that uses preprocessed data for fast analysis.
# Author: Roo, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import datetime
import logging
import warnings
import numpy as np
import jdatetime
from typing import Dict, List, Optional
from pathlib import Path
import config
from factors.calculator import calculate_momentum_6m, calculate_volatility

# --- Portfolio Optimization Libraries ---
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt
import empyrical as ep

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings('ignore')

# --- Define Project Root Path ---
# The script is in 'src/', so we go up one level to get the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True) # Ensure the logs directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'portfolio_optimizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Factor Strategy Functions
# These functions encapsulate the logic for calculating a specific factor.
# They take a DataFrame of prices and return a DataFrame of factor scores.
# ==============================================================================

def strategy_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates the 6-month momentum score for each stock."""
    return calculate_momentum_6m(prices)

def strategy_low_volatility(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Calculates the annualized volatility, where lower is better."""
    # We return the negative volatility because the backtester assumes higher scores are better.
    return -calculate_volatility(prices, window)


def robust_jalali_to_gregorian(date_input) -> pd.Timestamp:
    """Converts various Jalali date formats (int or str) to Gregorian datetime."""
    try:
        # Convert integer to string and normalize separators
        date_str = str(date_input).replace('/', '-')
        
        # Check if the date string contains separators
        if '-' in date_str:
            y, m, d = map(int, date_str.split('-'))
        else: # Assumes a dense format like 14020510
            y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
        
        # Convert to Gregorian date and then to pandas Timestamp
        gregorian_date = jdatetime.date(y, m, d).togregorian()
        return pd.to_datetime(gregorian_date)
        
    except (ValueError, TypeError):
        # Return a 'Not a Time' value if conversion fails
        return pd.NaT

class IranianStockOptimizerV3:
    """
    An interactive portfolio optimizer that uses preprocessed data.
    """
    def __init__(self, cache_dir: str, years_of_data: int, risk_free_rate: float = config.RISK_FREE_RATE):
        """
        Initialize the optimizer with configuration parameters.
        
        Args:
            cache_dir: Directory for caching data.
            years_of_data: Number of years of historical data to use.
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
        """
        self.cache_dir = PROJECT_ROOT / cache_dir
        self.years_of_data = years_of_data
        self.risk_free_rate = risk_free_rate
        
        self.master_price_df = None
        self.screener_df = None
        self.volume_df = None
        self.top_candidates = None
        self.weights = {}
        self.fundamental_df = None
        self.backtest_price_df = None # Add a dedicated df for backtesting
        
        # Enhanced filtering criteria
        self.min_return_threshold = config.MIN_RETURN_THRESHOLD
        self.max_volatility_threshold = config.MAX_VOLATILITY_THRESHOLD
        self.max_position_size = config.MAX_POSITION_SIZE
        self.max_drawdown_limit = config.MAX_DRAWDOWN_LIMIT
        self.min_data_points = config.MIN_DATA_POINTS
        self.min_liquidity_threshold = config.MIN_LIQUIDITY_THRESHOLD
        
        logger.info(f"🚀 Iranian Stock Optimizer v3.1 initialized")
        logger.info(f"   Cache Directory: {self.cache_dir}")
        logger.info(f"   Years of Data: {self.years_of_data}")
        logger.info(f"   Risk-free rate: {risk_free_rate:.2%}")

    def run_complete_analysis(self) -> bool:
        """
        Runs the complete analysis pipeline from loading data to optimization.
        """
        logger.info("🚀 Starting Complete Analysis Pipeline...")
        
        # 1. Load data from cache
        try:
            price_data_path = self.cache_dir / 'master_price_data.feather'
            volume_data_path = self.cache_dir / 'master_volume_data.feather'
            fundamental_data_path = self.cache_dir / 'master_fundamental_data.feather'
            
            if not all([price_data_path.exists(), volume_data_path.exists(), fundamental_data_path.exists()]):
                logger.error(f"❌ Data files not found in {self.cache_dir}. Please run the preprocessor first.")
                return False

            logger.info(f"📂 Loading data from {self.cache_dir}...")
            price_df = pd.read_feather(price_data_path)
            volume_df = pd.read_feather(volume_data_path)
            fundamental_df = pd.read_feather(fundamental_data_path)

            # Set date as index
            self.master_price_df = price_df.set_index('date')
            self.volume_df = volume_df.set_index('date')
            self.fundamental_df = fundamental_df.set_index('symbol')
            
            # Preserve a clean copy of the price data for the backtest
            self.backtest_price_df = self.master_price_df.copy()
            
            logger.info("   ✅ Data loaded successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to load preprocessed data: {e}")
            return False

        # 2. Screen stocks using the multi-factor model
        screen_success = self.screen_stocks_by_multifactor(top_n=20)
        if not screen_success:
            logger.error("❌ Stock screening failed. Aborting analysis.")
            return False
            
        # 3. Optimize the portfolio
        optimize_success = self.optimize_portfolio()
        if not optimize_success:
            logger.error("❌ Portfolio optimization failed.")
            return False
            
        logger.info("✅ Complete analysis finished successfully.")
        return True

    def create_master_dataframe(self, price_data: Dict[str, pd.Series]) -> bool:
        """Creates and caches a master DataFrame of all stock prices."""
        logger.info("Creating master price DataFrame...")
        if not price_data:
            logger.warning("⚠️ No price data provided to create master DataFrame.")
            return False
        
        try:
            self.master_price_df = pd.DataFrame(price_data)
            self.master_price_df.ffill(inplace=True)
            self.master_price_df.bfill(inplace=True)
            self.master_price_df.dropna(axis=1, how='all', inplace=True)
            
            logger.info(f"✅ Master DataFrame created with shape: {self.master_price_df.shape}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create master DataFrame: {e}")
            self.master_price_df = None
            return False

    def calculate_max_drawdown(self, prices: pd.DataFrame) -> pd.Series:
        """Calculates the maximum drawdown for each column in a price DataFrame."""
        if not isinstance(prices, pd.DataFrame):
            prices = prices.to_frame() # Convert Series to DataFrame

        def get_drawdown(price_series: pd.Series) -> float:
            """Helper to calculate drawdown for a single series."""
            numeric_series = pd.to_numeric(price_series, errors='coerce').dropna()
            if numeric_series.empty:
                return np.nan
            cumulative_returns = (1 + numeric_series.pct_change()).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            return drawdown.min()

        return prices.apply(get_drawdown)

    def calculate_advanced_metrics(self, price_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates performance metrics for a given historical price dataframe."""
        logger.debug(f"Calculating metrics for period of shape {price_df.shape}")
        numeric_price_df = price_df.apply(pd.to_numeric, errors='coerce')
        returns = numeric_price_df.pct_change().dropna(how='all')
        if returns.empty:
            logger.warning("⚠️ Could not calculate returns, DataFrame is empty after pct_change and dropna.")
            return pd.DataFrame()

        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        sharpe_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

        max_drawdown = price_df.apply(self.calculate_max_drawdown)
        trading_days = price_df.count()
        
        # Liquidity calculation (average daily volume)
        # Align volume data with price data to ensure consistency
        aligned_volume_df = volume_df.reindex(price_df.index).ffill().bfill()
        avg_daily_volume = aligned_volume_df.mean()

        metrics_df = pd.DataFrame({
            'Return': annualized_return, 'Volatility': annualized_volatility,
            'Sharpe': sharpe_ratio, 'Max_Drawdown': max_drawdown,
            'Trading_Days': trading_days,
            'Avg_Volume': avg_daily_volume
        })
        return metrics_df.fillna({'Sharpe': 0}).dropna()

    def screen_stocks(self) -> bool:
        """Screen and filter stocks based on the master price dataframe."""
        logger.info("🎯 STAGE 1: Advanced Stock Screening")
        
        if self.master_price_df is None or self.master_price_df.empty:
            logger.error("❌ Master price DataFrame is not available. Cannot screen stocks.")
            return False

        logger.info(f"📊 Screening stocks based on historical data...")
        if self.volume_df is None:
            logger.error("❌ Volume data is not available. Cannot perform screening.")
            return False
        metrics_df = self.calculate_advanced_metrics(self.master_price_df, self.volume_df)
        
        if metrics_df.empty:
            logger.warning("⚠️ Metrics DataFrame is empty. Cannot screen stocks for this period.")
            self.top_candidates = pd.DataFrame()
            return False

        filtered_stocks = metrics_df[
            (metrics_df['Return'] > self.min_return_threshold) &
            (metrics_df['Volatility'] < self.max_volatility_threshold) &
            (metrics_df['Max_Drawdown'] > self.max_drawdown_limit) &
            (metrics_df['Trading_Days'] >= self.min_data_points) &
            (metrics_df['Avg_Volume'] >= self.min_liquidity_threshold)
        ].copy()

        if filtered_stocks.empty:
            logger.warning("⚠️ No stocks passed the screening criteria for this period.")
            self.top_candidates = pd.DataFrame()
            return False
            
        logger.info(f"   ✅ {len(filtered_stocks)} stocks passed the screening")
        self.screener_df = filtered_stocks
        self.top_candidates = filtered_stocks.nlargest(20, 'Sharpe')
        logger.info("\n📈 Top Candidates for Optimization:")
        logger.info(self.top_candidates[['Return', 'Volatility', 'Sharpe', 'Max_Drawdown']].round(4).to_string())
        return True

    def screen_stocks_by_factor(self, factor: str, ascending: bool, top_n: int = 20) -> bool:
        """
        Screen stocks based on a single factor (e.g., 'Volatility', 'Return').

        Args:
            factor: The column name in the metrics_df to sort by (e.g., 'Volatility').
            ascending: Whether to sort in ascending order (True for low-vol, False for high-return).
            top_n: The number of top candidates to select.
        """
        logger.info(f"🎯 STAGE 1: Screening by Factor: {factor}")
        if self.master_price_df is None or self.master_price_df.empty:
            logger.error("❌ Master price DataFrame is not available.")
            return False

        metrics_df = self.calculate_advanced_metrics(self.master_price_df)
        if metrics_df.empty or factor not in metrics_df.columns:
            logger.warning(f"⚠️ Metrics DataFrame is empty or factor '{factor}' not found.")
            self.top_candidates = pd.DataFrame()
            return False

        # Basic filtering before factor ranking
        filtered_metrics = metrics_df[
            (metrics_df['Trading_Days'] >= self.min_data_points)
        ].copy()

        if filtered_metrics.empty:
            logger.warning("⚠️ No stocks passed the initial filtering.")
            self.top_candidates = pd.DataFrame()
            return False

        self.screener_df = filtered_metrics.sort_values(by=factor, ascending=ascending)
        self.top_candidates = self.screener_df.head(top_n)
        
        logger.info(f"\n📈 Top {top_n} Candidates based on {factor}:")
        logger.info(self.top_candidates[['Return', 'Volatility', 'Sharpe']].round(4).to_string())
        return True

    def screen_stocks_by_multifactor(self, top_n: int = 20) -> bool:
        """
        Screens stocks based on a combination of value, quality, and momentum factors.
        This model ranks stocks based on four key factors:
        - Value Factors: P/E Ratio (lower is better) and P/B Ratio (lower is better).
        - Quality/Profitability Factor: EPS (Earnings Per Share) (higher is better).
        - Momentum Factor: 6-Month Momentum (higher is better).
        A composite score is created by summing the ranks of these factors, and the top N
        stocks with the best (lowest) composite rank are selected.
        If fundamental data is unavailable, it gracefully falls back to a momentum-only screen.
        """
        logger.info("🎯 STAGE 1: Attempting Enhanced Multi-Factor Stock Screening (Value, Quality, Momentum)")

        # Pre-flight check for fundamental data. If it's missing or empty, fall back to momentum.
        if self.fundamental_df is None or self.fundamental_df.empty:
            logger.warning("⚠️ Fundamental data is not available or empty. Falling back to momentum-only screening.")
            return self.screen_stocks_by_momentum(top_n=top_n)

        if self.master_price_df is None or self.master_price_df.empty:
            logger.error("❌ Master price DataFrame is not available.")
            return False

        # 1. Calculate momentum
        momentum_scores = calculate_momentum_6m(self.master_price_df)
        last_momentum = momentum_scores.ffill().iloc[-1].dropna()

        # 2. Combine with fundamental data
        combined_df = self.fundamental_df.copy()
        combined_df['Momentum_6M'] = last_momentum

        # 3. Clean data
        # Replace zeros or negative values in P/E and P/B with NaN to avoid ranking them as good
        combined_df['p_e_ratio'] = combined_df['p_e_ratio'].apply(lambda x: x if x > 0 else np.nan)
        combined_df['p_b_ratio'] = combined_df['p_b_ratio'].apply(lambda x: x if x > 0 else np.nan)
        combined_df.dropna(subset=['p_e_ratio', 'p_b_ratio', 'eps', 'Momentum_6M'], inplace=True)

        if combined_df.empty:
            logger.warning("⚠️ No stocks with complete fundamental and momentum data. Falling back to momentum-only screening.")
            return self.screen_stocks_by_momentum(top_n=top_n)

        # 4. Create ranks for each factor.
        # For P/E and P/B, a lower value is better (ascending rank).
        pe_rank = combined_df['p_e_ratio'].rank(ascending=True)
        pb_rank = combined_df['p_b_ratio'].rank(ascending=True)
        # For EPS and Momentum, a higher value is better (descending rank).
        eps_rank = combined_df['eps'].rank(ascending=False)
        momentum_rank = combined_df['Momentum_6M'].rank(ascending=False)

        # 5. Combine ranks into a composite score (lower is better).
        combined_df['composite_score'] = pe_rank + pb_rank + eps_rank + momentum_rank

        # 6. Select top N stocks with the lowest composite score.
        top_tickers = combined_df['composite_score'].nsmallest(top_n).index
        
        # 7. Get full metrics for the top candidates
        metrics_df = self.calculate_advanced_metrics(self.master_price_df, self.volume_df)
        self.top_candidates = metrics_df.loc[metrics_df.index.isin(top_tickers)].copy()
        
        if self.top_candidates.empty:
            logger.warning("⚠️ Top candidates list is empty after screening.")
            return False

        # Add ranks and scores to the output for clarity
        self.top_candidates['P/E'] = combined_df.loc[top_tickers, 'p_e_ratio']
        self.top_candidates['P/B'] = combined_df.loc[top_tickers, 'p_b_ratio']
        self.top_candidates['EPS'] = combined_df.loc[top_tickers, 'eps']
        self.top_candidates['Composite_Score'] = combined_df.loc[top_tickers, 'composite_score']

        logger.info(f"\n📈 Top {top_n} Candidates based on Enhanced Multi-Factor Model:")
        logger.info(self.top_candidates[['Return', 'Volatility', 'Sharpe', 'P/E', 'P/B', 'EPS', 'Composite_Score']].round(4).to_string())
        
        return True

    def screen_stocks_by_momentum(self, top_n: int = 20) -> bool:
        """Screens stocks based on the 6-month momentum factor."""
        logger.info("🎯 STAGE 1: Screening by Momentum Factor")
        if self.master_price_df is None or self.master_price_df.empty:
            logger.error("❌ Master price DataFrame is not available.")
            return False

        # 1. Calculate 6-month momentum
        momentum_scores = calculate_momentum_6m(self.master_price_df)
        
        # Get the last valid score for each stock
        last_momentum = momentum_scores.ffill().iloc[-1].dropna()

        # 2. Apply basic filters (liquidity, data history)
        metrics_df = self.calculate_advanced_metrics(self.master_price_df, self.volume_df)
        valid_stocks = metrics_df[
            (metrics_df['Trading_Days'] >= self.min_data_points) &
            (metrics_df['Avg_Volume'] >= self.min_liquidity_threshold)
        ].index
        
        final_momentum = last_momentum[last_momentum.index.isin(valid_stocks)]

        if final_momentum.empty:
            logger.warning("⚠️ No stocks passed the momentum screening criteria.")
            self.top_candidates = pd.DataFrame()
            return False

        # 3. Select top N stocks
        top_tickers = final_momentum.nlargest(top_n).index
        self.top_candidates = metrics_df.loc[top_tickers]
        self.screener_df = metrics_df.loc[final_momentum.index] # Store all screened stocks

        logger.info(f"\n📈 Top {top_n} Candidates based on 6-Month Momentum:")
        # Add momentum score to the output for clarity
        display_df = self.top_candidates.copy()
        display_df['Momentum_6M'] = final_momentum.loc[top_tickers]
        logger.info(display_df[['Return', 'Volatility', 'Sharpe', 'Momentum_6M']].round(4).to_string())
        return True

    def optimize_portfolio(self) -> bool:
        """
        Optimize portfolio to find the max Sharpe ratio portfolio.
        Returns True on success, False on failure.
        """
        logger.info("🎯 STAGE 2: Advanced Portfolio Optimization")
        if self.top_candidates is None or self.top_candidates.empty:
            logger.warning("No candidates available for optimization in this period.")
            return False
            
        top_tickers = self.top_candidates.index.tolist()
        
        # To ensure we have a common window, we'll use the number of trading days
        # of the stock with the *least* data in the top candidates as our lookback period.
        min_trading_days = self.top_candidates['Trading_Days'].min()
        
        if pd.isna(min_trading_days) or min_trading_days <= 2: # Need at least 2 data points
            logger.warning(f"Could not determine a valid lookback period from candidates. Min trading days: {min_trading_days}")
            return False
            
        # Take the tail of the master dataframe corresponding to this lookback period
        lookback_period = int(min_trading_days)
        temp_price_df = self.master_price_df[top_tickers].tail(lookback_period)
        
        # Find the common index where all selected tickers have data
        common_index = temp_price_df.dropna().index
        
        # Filter the dataframe to only this common index
        final_price_df = temp_price_df.loc[common_index]
        
        if final_price_df.shape[1] < 2:
            logger.warning(f"Not enough valid assets ({final_price_df.shape[1]}) for optimization.")
            return False
            
        logger.info(f"Optimizing a portfolio of {len(final_price_df.columns)} assets.")
        try:
            mu = expected_returns.mean_historical_return(final_price_df, frequency=252)
            S = risk_models.CovarianceShrinkage(final_price_df, frequency=252).ledoit_wolf()
        except Exception as e:
            logger.error(f"❌ Error calculating mu and S: {e}")
            return False
            
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0) # Long-only constraint
        ef.add_constraint(lambda w: w <= self.max_position_size)
        
        try:
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            self.weights = ef.clean_weights()
            self.display_portfolio_results("Max Sharpe", self.weights, ef)
        except Exception as e:
            # Fallback mechanism: If max_sharpe fails (e.g., because no assets have
            # expected returns exceeding the risk-free rate), we log a warning and
            # switch to a more robust optimization objective that doesn't depend
            # on the risk-free rate, such as minimizing portfolio volatility.
            # This makes the system more resilient to different market conditions.
            logger.warning(f"⚠️ Max Sharpe optimization failed: {e}. Falling back to Minimum Volatility.")
            try:
                ef_min_vol = EfficientFrontier(mu, S)
                ef_min_vol.add_constraint(lambda w: w <= self.max_position_size)
                weights = ef_min_vol.min_volatility()
                self.weights = ef_min_vol.clean_weights()
                self.display_portfolio_results("Min Volatility (Fallback)", self.weights, ef_min_vol)
            except Exception as e_fallback:
                logger.error(f"❌ Fallback Minimum Volatility optimization also failed: {e_fallback}")
                self.weights = {}
                return False
            
        self.run_secondary_strategies_and_plot(mu, S, self.weights)
        return True

    def run_secondary_strategies_and_plot(self, mu, S, cleaned_weights_max_sharpe):
        """Runs min volatility strategy and plots the frontier for a single run."""
        logger.info("\n🎯 Strategy 2: Minimum Volatility Portfolio")
        ef_min_vol = EfficientFrontier(mu, S)
        ef_min_vol.add_constraint(lambda w: w <= self.max_position_size)
        cleaned_weights_min_vol = {}
        try:
            weights_min_vol = ef_min_vol.min_volatility()
            cleaned_weights_min_vol = ef_min_vol.clean_weights()
            self.display_portfolio_results("Min Volatility", cleaned_weights_min_vol, ef_min_vol)
        except Exception as e:
            logger.error(f"❌ Min Volatility optimization failed: {e}")
            cleaned_weights_min_vol = None
        if cleaned_weights_max_sharpe or cleaned_weights_min_vol:
            self.plot_efficient_frontier(mu, S, cleaned_weights_max_sharpe, cleaned_weights_min_vol)

    def display_portfolio_results(self, strategy_name: str, weights: dict, ef: EfficientFrontier):
        """Display portfolio optimization results in a clean, multi-line format."""
        if not weights: return
        weights_str = "\n".join([f"   - {ticker}: {weight:.2%}" for ticker, weight in weights.items() if weight > 0.001])
        try:
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
            output = f"\n--- Portfolio Performance: {strategy_name} ---\n📊 Optimal Weights:\n{weights_str}\n\n📈 Performance Metrics:\n   - Expected Annual Return: {perf[0]:.2%}\n   - Annual Volatility: {perf[1]:.2%}\n   - Sharpe Ratio: {perf[2]:.2f}\n-------------------------------------------------"
            logger.info(output)
        except Exception as e:
            logger.error(f"Could not calculate performance for {strategy_name}: {e}")

    def plot_efficient_frontier(self, mu, S, max_sharpe_weights: dict, min_vol_weights: dict):
        """Plots the efficient frontier and key portfolios."""
        logger.info("📊 Generating Efficient Frontier Plot...")
        ef_for_plotting = EfficientFrontier(mu, S)
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            plotting.plot_efficient_frontier(ef_for_plotting, ax=ax, show_assets=False)
            if max_sharpe_weights:
                ef_max_sharpe = EfficientFrontier(mu, S)
                ef_max_sharpe.set_weights(max_sharpe_weights)
                ret_max, vol_max, _ = ef_max_sharpe.portfolio_performance(risk_free_rate=self.risk_free_rate)
                ax.scatter(vol_max, ret_max, marker='*', s=250, c='r', label='Max Sharpe')
            if min_vol_weights:
                ef_min_vol = EfficientFrontier(mu, S)
                ef_min_vol.set_weights(min_vol_weights)
                ret_min, vol_min, _ = ef_min_vol.portfolio_performance()
                ax.scatter(vol_min, ret_min, marker='*', s=250, c='b', label='Min Volatility')
            ax.set_title(f"Efficient Frontier")
            ax.set_xlabel("Annual Volatility (Risk)"), ax.set_ylabel("Annual Return"), ax.legend(), plt.tight_layout()
            save_path = self.cache_dir / f"efficient_frontier.png"
            plt.savefig(save_path), logger.info(f"✅ Efficient frontier plot saved to '{save_path}'"), plt.close(fig)
        except Exception as e:
            logger.error(f"❌ Failed to generate or save the efficient frontier plot: {e}")

    def run_backtest(self, factor_strategy_function, rebalance_freq='3M', top_n=20, transaction_cost=0.005):
        """
        Runs a vectorized backtest for a given factor strategy.

        Args:
            factor_strategy_function (function): A function that takes a price DataFrame
                                                 and returns a factor score DataFrame.
            rebalance_freq (str): The frequency for rebalancing the portfolio (e.g., 'M', '3M', 'Q').
            top_n (int): The number of top stocks to select based on the factor score.
            transaction_cost (float): The cost per transaction as a percentage.

        Returns:
            pd.Series: A Series containing the net daily returns of the strategy.
        """
        logger.info(f"🚀 STAGE 3: Running Vectorized Backtest for strategy: {factor_strategy_function.__name__}")

        # 1. Prepare Data
        # Use the preserved backtest_price_df to ensure benchmark is present.
        if self.backtest_price_df is None or 'شاخص کل' not in self.backtest_price_df.columns:
            logger.error("❌ Benchmark 'شاخص کل' not found in the data loaded for backtesting.")
            return None
            
        prices = self.backtest_price_df.drop(columns=['شاخص کل'])
        benchmark_returns = self.backtest_price_df['شاخص کل'].pct_change().fillna(0)
        
        # 2. Determine Rebalancing Dates
        rebalance_dates = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq=rebalance_freq)
        
        # 3. Calculate Factor Scores (Vectorized)
        # This is the core of the vectorized approach: calculate the factor for the entire history at once.
        logger.info("   Calculating factor scores for the entire history...")
        factor_scores = factor_strategy_function(prices)
        
        # 4. Generate Signals and Weights (Vectorized)
        logger.info("   Generating trading signals and weights...")
        
        # Create an empty DataFrame to store weights, aligned with the price data index and columns.
        weights = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Loop through rebalancing dates to determine portfolio composition.
        # This loop is small and only runs on rebalance dates, not every day.
        for date in rebalance_dates:
            if date in factor_scores.index:
                # Get the factor scores on the rebalancing date.
                current_scores = factor_scores.loc[date].dropna()
                
                # Select the top N stocks with the highest factor scores.
                top_performers = current_scores.nlargest(top_n).index
                
                # Assign equal weight to the selected stocks on this specific date.
                weights.loc[date, top_performers] = 1 / top_n

        # Propagate weights forward until the next rebalance date.
        # This simulates holding the portfolio constant between rebalances.
        weights = weights.replace(0, np.nan).ffill()
        weights = weights.fillna(0) # Fill any remaining NaNs at the beginning.

        # 5. Calculate Portfolio Returns (Vectorized)
        logger.info("   Calculating portfolio returns...")
        daily_returns = prices.pct_change().fillna(0)
        
        # CRITICAL: Shift weights by 1 day to avoid lookahead bias.
        # We use today's weights to trade on tomorrow's price changes.
        # This is the most common source of errors in naive backtests.
        strategy_gross_returns = (weights.shift(1) * daily_returns).sum(axis=1)
        
        # 6. Calculate Transaction Costs (Vectorized)
        logger.info("   Calculating transaction costs...")
        # Calculate the absolute change in weights only on rebalancing days.
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * transaction_cost
        
        # 7. Calculate Net Returns
        net_returns = strategy_gross_returns - costs
        
        logger.info(f"🔄 Backtest complete. Approximate annual turnover: {turnover.mean() * 252 / 2:.2%}")
        
        # 8. Analyze and Plot Performance
        self.analyze_performance({factor_strategy_function.__name__: net_returns}, benchmark_returns)
        
        return net_returns

    def analyze_performance(self, results, benchmark_returns):
        """Analyzes and plots performance for multiple strategies."""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 9))
        
        all_reports = {}
        
        # Plot benchmark first
        benchmark_cum_returns = ep.cum_returns(benchmark_returns, starting_value=1)
        benchmark_cum_returns.plot(ax=ax, label='Benchmark (شاخص کل)', color='black', linestyle='--')
        
        for name, returns in results.items():
            # Calculate KPIs using empyrical
            kpis = {
                'Annualized Return': ep.annual_return(returns),
                'Annualized Volatility': ep.annual_volatility(returns),
                'Sharpe Ratio': ep.sharpe_ratio(returns, risk_free=self.risk_free_rate),
                'Max Drawdown': ep.max_drawdown(returns),
                'Alpha': ep.alpha(returns, benchmark_returns, risk_free=self.risk_free_rate),
                'Beta': ep.beta(returns, benchmark_returns)
            }
            report = pd.DataFrame({name: kpis})
            all_reports[name] = report
            
            # Plotting cumulative returns
            cum_returns = ep.cum_returns(returns, starting_value=1)
            cum_returns.plot(ax=ax, label=name)

        # Combine reports into a single DataFrame
        final_report = pd.concat(all_reports.values(), axis=1)
        
        # Add benchmark to report
        benchmark_kpis = {
            'Annualized Return': ep.annual_return(benchmark_returns),
            'Annualized Volatility': ep.annual_volatility(benchmark_returns),
            'Sharpe Ratio': ep.sharpe_ratio(benchmark_returns, risk_free=self.risk_free_rate),
            'Max Drawdown': ep.max_drawdown(benchmark_returns),
            'Alpha': 0,
            'Beta': 1
        }
        final_report['Benchmark'] = pd.Series(benchmark_kpis)
        
        print("\n--- Backtest Performance Analysis ---")
        print(final_report.round(3))
        
        ax.set_title('Backtest: Multi-Factor Strategy vs. Benchmark')
        ax.set_ylabel('Cumulative Returns')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        save_path = self.cache_dir / "backtest_performance.png"
        plt.savefig(save_path)
        logger.info(f"\n✅ Backtest plot saved to {save_path}")
        plt.show()
        
        return final_report

    # Methods below this line are from the v4 architecture and may not be compatible with the test suite.
    # They are kept for potential future use or if the interactive runner is used.
    
    def load_preprocessed_data(self) -> bool:
        """Load the preprocessed analysis-ready data and raw price data."""
        try:
            # This method is kept for the interactive runner, but tests will build dataframes manually.
            preprocessed_data_path = self.cache_dir / 'analysis_ready_data.feather'
            price_data_path = self.cache_dir / 'master_price_data.feather'
            logger.info(f"📂 Loading analysis data from '{preprocessed_data_path}'")
            self.all_metrics_df = pd.read_feather(preprocessed_data_path)
            logger.info(f"   Loaded analysis data with shape: {self.all_metrics_df.shape}")
            
            logger.info(f"📂 Loading raw price data from '{price_data_path}'")
            self.price_df = pd.read_feather(price_data_path)
            self.price_df.set_index('date', inplace=True)
            logger.info(f"   Loaded price data with shape: {self.price_df.shape}")
            
            return True
        except FileNotFoundError as e:
            logger.error(f"❌ Data file not found: {e}. Please run preprocessor.py first.")
            return False

    def is_cache_valid(self, file_path: Path, max_age_hours: int) -> bool:
        """Check if a cache file is valid and not too old."""
        if not file_path.exists():
            return False
        file_mod_time = file_path.stat().st_mtime
        age_seconds = datetime.datetime.now().timestamp() - file_mod_time
        return (age_seconds / 3600) < max_age_hours

    def download_ticker_data(self, symbol: str):
        # This is a placeholder, as the test provides data directly.
        # In a real scenario, this would download data.
        pass

def get_user_choice(prompt: str, options: dict) -> str:
    """Generic function to get user input from a list of options."""
    logger.info(prompt)
    for key, value in options.items():
        logger.info(f"[{key}] {value}")
    while True:
        choice = input(f"Enter your choice ({'/'.join(options.keys())}): ").strip()
        if choice in options:
            return choice
        else:
            logger.warning(f"❌ Invalid choice. Please enter one of {list(options.keys())}.")

def main():
    """Main function to run the enhanced portfolio optimizer."""
    print("🇮🇷 Iranian Stock Market Portfolio Optimizer v3.1")
    print("🔧 Optimized for pytse-client API")
    print("=" * 70)
    
    # Initialize the optimizer with desired parameters
    optimizer = IranianStockOptimizerV3(
        cache_dir='cache',
        years_of_data=5,
        risk_free_rate=0.35  # 35% risk-free rate for Iranian market
    )
    
    # --- Run Analysis ---
    # First, ensure data is loaded and available.
    success = optimizer.run_complete_analysis()
    
    if success:
        # Now, run the vectorized backtest with a specific strategy.
        optimizer.run_backtest(factor_strategy_function=strategy_momentum)
        
        print("\n✅ Analysis and backtesting completed successfully!")
        print("📊 Check the 'results' directory for performance plots and logs.")
    else:
        print("\n❌ Initial analysis failed. Could not proceed to backtesting.")
        print("   Check 'portfolio_optimizer.log' for details.")


if __name__ == "__main__":
    main()