# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Interactive Portfolio Optimizer v4.0
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

# --- Portfolio Optimization Libraries ---
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings('ignore')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_optimizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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

class IranianStockOptimizerV4:
    """
    An interactive portfolio optimizer that uses preprocessed data.
    """
    def __init__(self, time_horizon: str, preprocessed_data_path: str, risk_free_rate: float = 0.28):
        """
        Initialize the optimizer with configuration parameters.
        
        Args:
            time_horizon: Investment time horizon ('short-term', 'mid-term', 'long-term').
            preprocessed_data_path: Path to the preprocessed data file.
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
        """
        self.time_horizon = time_horizon
        self.preprocessed_data_path = Path(preprocessed_data_path)
        self.price_data_path = self.preprocessed_data_path.parent / 'master_price_data.feather'
        self.risk_free_rate = risk_free_rate
        self.all_metrics_df = None
        self.price_df = None
        self.screener_df = None
        self.top_candidates = None
        
        # Enhanced filtering criteria
        # Decoupled from risk_free_rate for more realistic backtesting across different market regimes.
        self.min_return_threshold = 0.15
        self.max_volatility_threshold = 0.75
        self.max_position_size = 0.25
        
        logger.info(f"🚀 Iranian Stock Optimizer v4.0 initialized")
        logger.info(f"   Time Horizon: {self.time_horizon}")
        logger.info(f"   Risk-free rate: {risk_free_rate:.2%}")

    def load_preprocessed_data(self) -> bool:
        """Load the preprocessed analysis-ready data and raw price data."""
        try:
            logger.info(f"📂 Loading analysis data from '{self.preprocessed_data_path}'")
            self.all_metrics_df = pd.read_feather(self.preprocessed_data_path)
            logger.info(f"   Loaded analysis data with shape: {self.all_metrics_df.shape}")
            
            logger.info(f"📂 Loading raw price data from '{self.price_data_path}'")
            self.price_df = pd.read_feather(self.price_data_path)
            # The date column was reset during saving, so we set it back as the index
            self.price_df.set_index('date', inplace=True)
            logger.info(f"   Loaded price data with shape: {self.price_df.shape}")
            
            return True
        except FileNotFoundError as e:
            logger.error(f"❌ Data file not found: {e}. Please run preprocessor.py first.")
            return False

    def _calculate_metrics_for_period(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates performance metrics for a given historical price dataframe."""
        logger.debug(f"Calculating metrics for period of shape {price_df.shape}")
        # Ensure all data is numeric before calculations
        numeric_price_df = price_df.apply(pd.to_numeric, errors='coerce')
        returns = numeric_price_df.pct_change().dropna(how='all')
        if returns.empty:
            logger.warning("⚠️ Could not calculate returns, DataFrame is empty after pct_change and dropna.")
            return pd.DataFrame()
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        metrics_df = pd.DataFrame({
            'Return': annualized_return, 'Volatility': annualized_volatility,
            'Sharpe': sharpe_ratio, 'Max_Drawdown': max_drawdown
        })
        return metrics_df.dropna()

    def screen_stocks(self, price_df_period: Optional[pd.DataFrame] = None) -> bool:
        """Screen and filter stocks based on a specific data period or pre-calculated metrics."""
        logger.info("🎯 STAGE 1: Advanced Stock Screening")
        if price_df_period is not None:
            logger.info(f"📊 Screening stocks based on historical data until {price_df_period.index.max().date()}...")
            metrics_df = self._calculate_metrics_for_period(price_df_period)
        else:
            logger.info(f"📊 Screening stocks for '{self.time_horizon}' horizon using pre-calculated data...")
            metrics_df = self.all_metrics_df[self.all_metrics_df['Horizon'] == self.time_horizon].set_index('Symbol')
        
        if metrics_df.empty:
            logger.warning("⚠️ Metrics DataFrame is empty. Cannot screen stocks for this period.")
            self.top_candidates = pd.DataFrame()
            return False

        filtered_stocks = metrics_df[
            (metrics_df['Return'] > self.min_return_threshold) &
            (metrics_df['Volatility'] < self.max_volatility_threshold) &
            (metrics_df['Max_Drawdown'] > -0.8)
        ].copy()

        if filtered_stocks.empty:
            logger.warning("⚠️ No stocks passed the screening criteria for this period.")
            self.top_candidates = pd.DataFrame()
            return False
        logger.info(f"   ✅ {len(filtered_stocks)} stocks passed the screening")
        self.top_candidates = filtered_stocks.nlargest(20, 'Sharpe')
        logger.info("\n📈 Top Candidates for Optimization:")
        logger.info(self.top_candidates[['Return', 'Volatility', 'Sharpe', 'Max_Drawdown']].round(4).to_string())
        return True

    def optimize_portfolio(self, price_df_period: Optional[pd.DataFrame] = None) -> dict:
        """Optimize portfolio, returning the weights for the max sharpe portfolio."""
        logger.info("🎯 STAGE 2: Advanced Portfolio Optimization")
        if self.top_candidates is None or self.top_candidates.empty:
            logger.warning("No candidates available for optimization in this period.")
            return {}
        top_tickers = self.top_candidates.index.tolist()
        if price_df_period is not None:
            final_price_df = price_df_period[top_tickers].dropna(axis=1, how='any')
        else:
            today = self.price_df.index.max()
            start_date = today - pd.DateOffset(years={'long-term': 5, 'mid-term': 3, 'short-term': 1.5}[self.time_horizon])
            final_price_df = self.price_df.loc[start_date:today, self.price_df.columns.isin(top_tickers)].dropna(axis=1, how='any')
        
        if final_price_df.shape[1] < 2:
            logger.warning(f"Not enough valid assets ({final_price_df.shape[1]}) for optimization.")
            return {}
        logger.info(f"Optimizing a portfolio of {len(final_price_df.columns)} assets.")
        try:
            mu = expected_returns.mean_historical_return(final_price_df, frequency=252)
            S = risk_models.CovarianceShrinkage(final_price_df, frequency=252).ledoit_wolf()
        except Exception as e:
            logger.error(f"❌ Error calculating mu and S: {e}")
            return {}
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w <= self.max_position_size)
        cleaned_weights = {}
        try:
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights = ef.clean_weights()
            if price_df_period is None: self.display_portfolio_results("Max Sharpe", cleaned_weights, ef)
        except Exception as e:
            logger.warning(f"Max Sharpe optimization failed for this period: {e}")
            return {}
        if price_df_period is None: self.run_secondary_strategies_and_plot(mu, S, cleaned_weights)
        return cleaned_weights

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
            ax.set_title(f"Efficient Frontier for '{self.time_horizon}' Horizon")
            ax.set_xlabel("Annual Volatility (Risk)"), ax.set_ylabel("Annual Return"), ax.legend(), plt.tight_layout()
            save_path = self.preprocessed_data_path.parent / f"efficient_frontier_{self.time_horizon}.png"
            plt.savefig(save_path), logger.info(f"✅ Efficient frontier plot saved to '{save_path}'"), plt.close(fig)
        except Exception as e:
            logger.error(f"❌ Failed to generate or save the efficient frontier plot: {e}")

    def run_complete_analysis(self) -> bool:
        """Run the complete analysis pipeline for a single optimization."""
        logger.info("🚀 Starting Portfolio Analysis Pipeline v4.0")
        if not self.load_preprocessed_data(): return False
        if not self.screen_stocks():
            logger.error("❌ Stock screening failed")
            return False
        self.optimize_portfolio()
        logger.info("🎉 Analysis completed successfully!")
        return True

    def _get_market_trend_signal(self, benchmark_prices: pd.Series, short_window: int = 50, long_window: int = 200) -> str:
        """Determines market trend using a moving average crossover strategy."""
        if len(benchmark_prices) < long_window:
            logger.warning("Not enough data for trend filter, assuming uptrend.")
            return 'UP'
        
        short_sma = benchmark_prices.rolling(window=short_window).mean()
        long_sma = benchmark_prices.rolling(window=long_window).mean()
        
        last_short_sma = short_sma.iloc[-1]
        last_long_sma = long_sma.iloc[-1]
        
        signal = 'UP' if last_short_sma > last_long_sma else 'DOWN'
        logger.info(f"Market Trend Check: SMA({short_window})={last_short_sma:.0f}, SMA({long_window})={last_long_sma:.0f}. Signal is {signal}.")
        return signal

    def run_backtest(self, backtest_years: int = 3, rebalance_period_days: int = 90):
        """
        Performs a complete, architecturally sound backtest of the portfolio strategy.
        This method avoids look-ahead bias by creating a new, temporary optimizer instance
        for each period, ensuring all decisions are made only with data available at that time.
        """
        logger.info(f"🚀 Starting Final, Architecturally Correct Backtest for {backtest_years} years...")
        self.risk_free_rate = 0.28
        logger.info(f"   Using realistic risk-free rate for backtest: {self.risk_free_rate:.2%}")
        transaction_cost_rate = 0.015  # 1.5% for round-trip trade
        logger.info(f"   Applying transaction cost rate: {transaction_cost_rate:.3%}")

        # 1. Initial Setup
        # =================
        if self.price_df is None:
            if not self.load_preprocessed_data(): return

        end_date = self.price_df.index.max()
        start_date = end_date - pd.DateOffset(years=backtest_years)
        
        # b. Fetch benchmark data for the entire backtest period once
        logger.info(f"Fetching benchmark data for the entire period: {start_date.date()} to {end_date.date()}")
        benchmark_prices = self.price_df.loc[start_date:end_date, 'شاخص کل'].dropna()
        # Ensure benchmark prices are numeric before calculating returns
        benchmark_prices_numeric = pd.to_numeric(benchmark_prices, errors='coerce')
        benchmark_daily_returns = benchmark_prices_numeric.pct_change().fillna(0)
        # The preprocessor already converts dates, so we just slice.

        # 2. The Main Backtesting Loop
        # ============================
        # a. Set up rebalance_dates
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=f'{rebalance_period_days}D')
        
        # b. Create empty lists to store period returns for the portfolio and the benchmark
        portfolio_period_returns = []
        benchmark_period_returns_list = []

        # c. Loop through rebalance_dates
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_rebalance_date = rebalance_dates[i+1]
            
            logger.info("\n" + "="*80)
            logger.info(f"🔄 Processing Period: {current_date.date()} to {next_rebalance_date.date()}")
            logger.info("="*80)

            # 3. Inside Each Loop Iteration
            # =============================
            # a. [Benchmark Calculation] Always calculate and store benchmark returns for the period.
            # This ensures the benchmark line on the plot is continuous.
            mask = (benchmark_daily_returns.index >= current_date) & (benchmark_daily_returns.index < next_rebalance_date)
            benchmark_period_returns = benchmark_daily_returns.loc[mask]
            benchmark_period_returns_list.append(benchmark_period_returns)
            logger.info(f"   ✅ Stored {len(benchmark_period_returns)} daily returns for benchmark for this period.")

            # b. [Trend Filter] Get the market trend signal.
            benchmark_slice = benchmark_prices.loc[:current_date]
            trend_signal = self._get_market_trend_signal(benchmark_slice)

            # c. If the signal is 'DOWN', hold cash for the period.
            if trend_signal == 'DOWN':
                logger.warning(f"   📉 Market signal is DOWN. Holding cash.")
                actual_trading_days = self.price_df.loc[current_date:next_rebalance_date].index
                cash_returns = pd.Series(0, index=actual_trading_days)
                portfolio_period_returns.append(cash_returns)
                continue

            # d. If the signal is 'UP', proceed with screening and optimization.
            logger.info(f"   📈 Market signal is UP. Constructing portfolio for this period.")
            
            # i. [Data Fetching] Create a new, temporary instance of the optimizer.
            logger.info("   Instantiating a temporary optimizer for this period to ensure data isolation.")
            period_optimizer = IranianStockOptimizerV4(
                time_horizon=self.time_horizon,
                preprocessed_data_path=self.preprocessed_data_path,
                risk_free_rate=self.risk_free_rate
            )
            historical_prices = self.price_df.loc[:current_date]

            # ii. [Screen & Optimize]
            screened = period_optimizer.screen_stocks(price_df_period=historical_prices)
            if not screened:
                logger.warning("   Screening failed for this period. Holding cash.")
                actual_trading_days = self.price_df.loc[current_date:next_rebalance_date].index
                cash_returns = pd.Series(0, index=actual_trading_days)
                portfolio_period_returns.append(cash_returns)
                continue

            optimal_weights = period_optimizer.optimize_portfolio(price_df_period=historical_prices)
            
            # iii. [Handle Optimization Failure]
            if not optimal_weights:
                logger.warning("   Optimization failed for this period. Holding cash.")
                actual_trading_days = self.price_df.loc[current_date:next_rebalance_date].index
                cash_returns = pd.Series(0, index=actual_trading_days)
                portfolio_period_returns.append(cash_returns)
                continue
            
            logger.info(f"   ✅ Optimal weights found for {len(optimal_weights)} assets.")

            # iv. [Simulate Holding]
            logger.info(f"   Simulating holding period with actual future prices...")
            holding_period_prices = self.price_df.loc[current_date:next_rebalance_date, list(optimal_weights.keys())]
            
            # v. Calculate daily returns for this period
            daily_returns_of_assets = holding_period_prices.pct_change().dropna(how='all')
            
            # vi. Calculate the daily returns of the optimal portfolio
            weights_series = pd.Series(optimal_weights)
            aligned_returns, aligned_weights = daily_returns_of_assets.align(weights_series, axis=1, join='right')
            period_returns = (aligned_returns.fillna(0) * aligned_weights).sum(axis=1)
            
            # Apply transaction cost to the first day of the holding period
            if not period_returns.empty:
                period_returns.iloc[0] -= transaction_cost_rate
                logger.info(f"   Applied transaction cost of {transaction_cost_rate:.3%} for this period.")

            # vii. Append the resulting Series
            portfolio_period_returns.append(period_returns)
            logger.info(f"   ✅ Calculated and stored {len(period_returns)} daily returns for strategy.")

        # 4. Final Analysis (After the Loop)
        # ===================================
        logger.info("\n" + "="*80)
        logger.info("🏁 Backtest loop complete. Aggregating and analyzing final results...")
        logger.info("="*80)

        if not portfolio_period_returns:
            logger.error("❌ Backtest failed to generate any returns.")
            return

        # a. Concatenate all period returns
        strategy_returns = pd.concat(portfolio_period_returns).sort_index()
        strategy_returns = strategy_returns[~strategy_returns.index.duplicated(keep='first')]
        strategy_returns = strategy_returns.loc[start_date:end_date]

        # b. Aggregate benchmark returns and calculate cumulative returns for both
        benchmark_returns = pd.concat(benchmark_period_returns_list).sort_index()
        benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep='first')]
        benchmark_returns = benchmark_returns.loc[start_date:end_date]

        strategy_cumulative_return = (1 + strategy_returns.fillna(0)).cumprod()
        benchmark_cumulative_return = (1 + benchmark_returns.fillna(0)).cumprod()

        # Plot the results
        logger.info("   Generating final performance plot...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        strategy_cumulative_return.plot(ax=ax, label="Strategy Cumulative Return", color="blue", lw=2)
        benchmark_cumulative_return.plot(ax=ax, label="Benchmark (TSE All-Share)", color="gray", linestyle="--", lw=2)
        
        ax.set_title(f"Architecturally Correct Backtest ({backtest_years} Years, {rebalance_period_days}-Day Rebalancing)", fontsize=16)
        ax.set_ylabel("Cumulative Return"), ax.set_xlabel("Date"), ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        save_path = self.preprocessed_data_path.parent / "backtest_performance_final.png"
        plt.savefig(save_path)
        logger.info(f"✅ Final backtest performance plot saved to '{save_path}'")
        plt.show()

def get_user_choice(prompt: str, options: dict) -> str:
    """Generic function to get user input from a list of options."""
    print(prompt)
    for key, value in options.items():
        print(f"[{key}] {value}")
    while True:
        choice = input(f"Enter your choice ({'/'.join(options.keys())}): ").strip()
        if choice in options:
            return choice
        else:
            print(f"❌ Invalid choice. Please enter one of {list(options.keys())}.")

def main():
    """Main function to run the interactive portfolio optimizer."""
    print("🇮🇷 Iranian Stock Market Portfolio Optimizer v4.0")
    print("=" * 70)
    mode_options = {'1': 'single', '2': 'backtest'}
    mode = get_user_choice("Choose execution mode:", {'1': 'Single Analysis', '2': 'Run Backtest'})
    if mode == 'single':
        horizon_map = {'1': 'short-term', '2': 'mid-term', '3': 'long-term'}
        horizon_options = {'1': "Short-Term (1.5 years)", '2': "Mid-Term (3 years)", '3': "Long-Term (5 years)"}
        chosen_horizon_key = get_user_choice("Please choose your investment time horizon:", horizon_options)
        optimizer = IranianStockOptimizerV4(
            time_horizon=horizon_map[chosen_horizon_key],
            preprocessed_data_path='cache_v3/analysis_ready_data.feather',
            risk_free_rate=0.35
        )
        optimizer.run_complete_analysis()
    else:
        optimizer = IranianStockOptimizerV4(
            time_horizon='backtest',
            preprocessed_data_path='cache_v3/analysis_ready_data.feather',
            risk_free_rate=0.28  # Using the corrected risk-free rate
        )
        if optimizer.load_preprocessed_data():
            optimizer.run_backtest(backtest_years=5, rebalance_period_days=90)

if __name__ == "__main__":
    main()