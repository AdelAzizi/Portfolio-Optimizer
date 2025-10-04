# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Multi-Factor Portfolio Optimizer & Backtester
# Description: A powerful multi-factor analysis and optimization engine that
#              selects stocks based on a composite score of Value, Momentum,
#              and Low Volatility, then backtests the resulting portfolio.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path

# --- Portfolio Optimization & Plotting ---
from pypfopt import expected_returns, risk_models, EfficientFrontier, exceptions
import matplotlib.pyplot as plt
from .config import TOP_N_CANDIDATES, RISK_FREE_RATE, TRADE_COST_PERCENT

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- Define Project Root and Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'
RESULTS_DIR = PROJECT_ROOT / 'results'

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'optimizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiFactorOptimizer:
    """
    Performs and validates a portfolio strategy using a rolling-window backtest.
    """
    def __init__(self, analysis_data_path: Path, price_data_dir: Path,
                 max_position_size: float, factor_weights: dict, momentum_period: str,
                 top_n_candidates: int = TOP_N_CANDIDATES, commission_rate: float = None, slippage_pct: float = None):
        self.analysis_data_path = analysis_data_path
        self.price_data_dir = price_data_dir
        self.max_position_size = max_position_size
        self.top_n_candidates = top_n_candidates
        self.factor_weights = factor_weights
        self.momentum_period = momentum_period
        self.commission_rate = commission_rate if commission_rate is not None else TRADE_COST_PERCENT
        self.slippage_pct = slippage_pct if slippage_pct is not None else 0.001  # Default slippage
        
        self.analysis_df = None
        self.master_price_df = None
        
        logger.info(f"🚀 Initialized Optimizer with Momentum: {self.momentum_period}, Weights: {self.factor_weights}, "
                   f"Commission: {self.commission_rate:.3f}, Slippage: {self.slippage_pct:.3f}")

    def _load_data(self):
        """Loads all necessary analysis and price data into memory."""
        if self.analysis_df is not None and self.master_price_df is not None:
            return
        logger.info("--- Loading All Required Data for Analysis ---")
        if not self.analysis_data_path.exists():
            raise FileNotFoundError(f"Analysis data not found at {self.analysis_data_path}")
        
        self.analysis_df = pd.read_feather(self.analysis_data_path).set_index('symbol')
        
        all_symbols = self.analysis_df.index.tolist() + ['شاخص کل']
        price_data = {}
        for symbol in all_symbols:
            file_path = self.price_data_dir / f"{symbol}.csv"
            if file_path.exists():
                price_data[symbol] = pd.read_csv(file_path, index_col='date', parse_dates=True)['close']
        
        self.master_price_df = pd.DataFrame(price_data).sort_index().ffill().bfill()
        logger.info(f"✅ Loaded data for {len(self.analysis_df)} symbols.")

    def screen_stocks(self) -> list:
        """
        Screens and ranks stocks based on a multi-factor model.
    
        Returns:
            list: A list of the top N candidate stock symbols.
        """
        logger.info(f"--- Screening with weights: {self.factor_weights} & period: {self.momentum_period} ---")
        df = self.analysis_df.copy()

        momentum_col = f"Momentum_{self.momentum_period}"
        if momentum_col not in df.columns:
            logger.error(f"CRITICAL: Momentum column '{momentum_col}' not found in analysis data.")
            raise ValueError(f"Missing momentum column: {momentum_col}")

        # Factor 1: Value (lower is better)
        df['Value_Rank_PE'] = df['P/E'].rank(ascending=True)
        df['Value_Rank_PS'] = df['P/S'].rank(ascending=True)
        df['Value_Score'] = df[['Value_Rank_PE', 'Value_Rank_PS']].mean(axis=1).rank(pct=True)

        # Factor 2: Momentum (higher is better)
        df['Momentum_Score'] = df[momentum_col].rank(ascending=True, pct=True)

        # Factor 3: Low Volatility (lower is better)
        df['Low_Volatility_Score'] = df['Volatility'].rank(ascending=False, pct=True)

        # Composite Score
        df['Composite_Score'] = (
            self.factor_weights['Value'] * df['Value_Score'] +
            self.factor_weights['Momentum'] * df['Momentum_Score'] +
            self.factor_weights['Low_Volatility'] * df['Low_Volatility_Score']
        )
        
        # Select top candidates
        top_candidates = df.sort_values(by='Composite_Score', ascending=False).head(self.top_n_candidates)
        
        # logger.info(f"Top {self.top_n_candidates} Candidates based on Composite Score:")
        # logger.info("\n" + top_candidates[['P/E', 'P/S', momentum_col, 'Volatility', 'Composite_Score']].to_string())
        
        return top_candidates.index.tolist()

    def _get_portfolio_for_date(self, historical_prices: pd.DataFrame, candidate_symbols: list) -> dict:
        """Runs the optimization for a single point in time for the given candidates."""
        try:
            prices_for_opt = historical_prices[candidate_symbols]
            mu = expected_returns.mean_historical_return(prices_for_opt)
            S = risk_models.CovarianceShrinkage(prices_for_opt).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= self.max_position_size)
            weights = ef.min_volatility()
            return ef.clean_weights()
        except Exception as e:
            logger.error(f"   -> Optimization failed for this period: {e}")
            return None

    def run_comparative_backtest(self, portfolio_returns: pd.Series, historical_prices: pd.DataFrame) -> dict:
        """
        Calculates time-series data for strategy vs. benchmark performance.
        This method does not plot, it only returns the data.

        Args:
            portfolio_returns (pd.Series): Daily returns of the strategy portfolio.
            historical_prices (pd.DataFrame): DataFrame with prices, including the benchmark.

        Returns:
            dict: A dictionary containing dates and cumulative values for charting.
        """
        logger.info("--- Running Comparative Backtest Data Generation ---")
        
        # Drop the initial NaN from pct_change() and ensure there are returns
        portfolio_returns = portfolio_returns.dropna()
        if portfolio_returns.empty:
            logger.warning("No valid portfolio returns to generate backtest data.")
            return {"dates": [], "strategy_values": [], "benchmark_values": []}

        # Calculate cumulative returns for the strategy
        strategy_cumulative = (1 + portfolio_returns).cumprod()

        # Calculate cumulative returns for the benchmark
        benchmark_returns = historical_prices['شاخص کل'].pct_change()
        aligned_benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
        benchmark_cumulative = (1 + aligned_benchmark_returns).cumprod()

        # Normalize to start at 100 for charting
        strategy_values = 100 * strategy_cumulative
        benchmark_values = 100 * benchmark_cumulative
        
        # Prepend the starting '100' value for both series for a clean chart start
        start_date = portfolio_returns.index[0] - pd.Timedelta(days=1)
        strategy_values = pd.Series([100], index=[start_date])._append(strategy_values)
        benchmark_values = pd.Series([100], index=[start_date])._append(benchmark_values)

        chart_data = {
            "dates": strategy_values.index.strftime('%Y-%m-%d').tolist(),
            "strategy_values": strategy_values.round(2).tolist(),
            "benchmark_values": benchmark_values.round(2).tolist()
        }
        
        logger.info("✅ Generated backtest time-series data.")
        return chart_data

    def run_rolling_backtest(self, years: int = 3, rebalance_period_days: int = 90) -> dict:
        """
        Performs a rolling backtest with periodic rebalancing, transaction cost
        simulation, and turnover analysis.
        """
        self._load_data()
        logger.info(f"--- Starting Rolling Backtest: {years} years, rebalancing every {rebalance_period_days} days ---")

        # --- Date Range Setup ---
        end_date = self.master_price_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        backtest_range = pd.date_range(start=start_date, end=end_date, freq=f'{rebalance_period_days}D')

        # --- State Tracking ---
        portfolio_values = []
        dates = []
        previous_weights = {}
        all_trades = []
        total_costs = 0
        turnover_history = []
        rebalance_history = []
        final_optimal_weights = {}

        # --- Main Backtest Loop ---
        for i, current_rebalance_date in enumerate(backtest_range):
            # Find the closest available trading day for the rebalance date

            try:
                # Get the index of the last valid date on or before the rebalance date
                # Using get_indexer for broader pandas version compatibility
                loc = self.master_price_df.index.get_indexer([current_rebalance_date], method='ffill')[0]
                actual_rebalance_date = self.master_price_df.index[loc]
            except Exception as e:
                logger.warning(f"Could not find a valid trading day near {current_rebalance_date.date()}. Error: {e}. Skipping rebalance.")
                continue

            logger.info(f"🔄 Processing rebalance for date: {current_rebalance_date.date()} (Actual Trading Day: {actual_rebalance_date.date()})")

            # Define the lookback window for this period's optimization
            lookback_end_date = actual_rebalance_date - pd.Timedelta(days=1)
            lookback_start_date = lookback_end_date - pd.DateOffset(years=1) # 1-year lookback for optimization
            historical_prices = self.master_price_df.loc[lookback_start_date:lookback_end_date]

            if historical_prices.empty:
                logger.warning(f"  -> No historical price data for lookback ending {lookback_end_date.date()}. Skipping.")
                continue

            # --- Get New Portfolio Weights ---
            candidate_symbols = self.screen_stocks()
            new_weights = self._get_portfolio_for_date(historical_prices, candidate_symbols)

            if not new_weights:
                logger.warning("  -> Optimization failed for this period. Holding previous portfolio.")
                new_weights = previous_weights

            # --- Transaction & Turnover Calculation ---
            trades_diff = {sym: new_weights.get(sym, 0) - previous_weights.get(sym, 0) for sym in set(previous_weights) | set(new_weights)}
            
            # Log individual trades
            for symbol, weight_change in trades_diff.items():
                if abs(weight_change) > 1e-6: # If there was a trade
                    trade_value = abs(weight_change) * (portfolio_values[-1] if portfolio_values else 100)
                    all_trades.append({
                        "date": actual_rebalance_date,
                        "symbol": symbol,
                        "amount": trade_value if weight_change > 0 else -trade_value,
                        "price": self.master_price_df.loc[actual_rebalance_date, symbol],
                        "commission": trade_value * self.commission_rate,
                        "slippage": trade_value * self.slippage_pct
                    })

            bought_value = sum(v for v in trades_diff.values() if v > 0)
            sold_value = abs(sum(v for v in trades_diff.values() if v < 0))
            turnover = min(bought_value, sold_value) # Standard turnover calculation
            turnover_history.append(turnover)

            # --- Cost Simulation ---
            transaction_cost = (bought_value + sold_value) * (self.commission_rate + self.slippage_pct)
            total_costs += transaction_cost
            
            # --- Log Rebalancing Actions ---
            rebalance_history.append({
                "date": actual_rebalance_date.strftime('%Y-%m-%d'),
                "sold": {s: f"{abs(w):.2%}" for s, w in trades_diff.items() if w < -0.001},
                "bought": {s: f"{w:.2%}" for s, w in trades_diff.items() if w > 0.001}
            })

            # --- Performance Calculation for the Period ---
            period_end_date = actual_rebalance_date + pd.DateOffset(days=rebalance_period_days)
            prices_for_period = self.master_price_df.loc[actual_rebalance_date:period_end_date]

            if not prices_for_period.empty and new_weights:
                period_returns = prices_for_period[list(new_weights.keys())].pct_change().dropna()
                portfolio_period_returns = (period_returns * pd.Series(new_weights)).sum(axis=1)
                
                # Deduct transaction cost from the first return of the period
                if not portfolio_period_returns.empty:
                    portfolio_period_returns.iloc[0] -= transaction_cost

                # Chain the portfolio value
                start_value = portfolio_values[-1] if portfolio_values else 100
                period_cumulative_returns = (1 + portfolio_period_returns).cumprod()
                portfolio_values.extend((start_value * period_cumulative_returns).tolist())
                dates.extend(portfolio_period_returns.index.tolist())

            previous_weights = new_weights
            if new_weights:
                final_optimal_weights = new_weights

        if not dates:
            logger.error("❌ Rolling backtest generated no data points.")
            return None, None, None, None

        # --- Final Data Assembly ---
        equity_curve = pd.Series(portfolio_values, index=pd.to_datetime(dates)).groupby(level=0).last()
        equity_curve.name = "Equity"
        trades_df = pd.DataFrame(all_trades)

        backtest_data = self.run_comparative_backtest(equity_curve.pct_change(), self.master_price_df)

        # --- Turnover Calculation ---
        avg_period_turnover = np.mean(turnover_history) if turnover_history else 0
        periods_per_year = 365 / rebalance_period_days
        annual_turnover = avg_period_turnover * periods_per_year

        logger.info("✅ Rolling backtest complete.")
        performance_data = {
            "performance_data": backtest_data,
            "transaction_analysis": {
                "annual_turnover": f"{annual_turnover:.2%}",
                "estimated_total_cost": f"{total_costs:.4f}",
                "commission_rate": self.commission_rate,
                "slippage_pct": self.slippage_pct,
                "rebalance_history": rebalance_history
            }
        }
        return performance_data, final_optimal_weights, equity_curve, trades_df

    def run_full_analysis(self, years: int = 3) -> dict:
        """
        Runs the full screening, optimization, and performance analysis using
        the rolling backtest engine.
        """
        self._load_data()
        
        # We now call the rolling backtest, which returns performance and final weights
        backtest_results, final_weights, equity_curve, trades_df = self.run_rolling_backtest(years=years)

        if not backtest_results or not final_weights:
            logger.error("❌ Rolling backtest failed to produce results or final weights.")
            return None

        # --- Calculate Performance Summary from the new backtest data ---
        strategy_values = pd.Series(backtest_results['performance_data']['strategy_values'])
        if len(strategy_values) < 2:
            return {
                'optimal_weights': final_weights,
                'performance_summary': {},
                'backtest_data': backtest_results['performance_data'],
                'transaction_analysis': backtest_results['transaction_analysis'],
                'equity_curve': None,
                'trades': None
            }

        total_return = (strategy_values.iloc[-1] / strategy_values.iloc[0]) - 1
        
        returns = strategy_values.pct_change().dropna()
        annualized_volatility = returns.std() * np.sqrt(252)
        
        num_days = (pd.to_datetime(backtest_results['performance_data']['dates'][-1]) - pd.to_datetime(backtest_results['performance_data']['dates'][0])).days
        annualized_return = (1 + total_return) ** (365.25 / num_days) - 1 if num_days > 0 else 0

        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility
        else:
            sharpe_ratio = np.inf if annualized_return > RISK_FREE_RATE else 0

        performance_summary = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Volatility': f"{annualized_volatility:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
        }

        logger.info("✅ Full analysis complete. Returning comprehensive results.")
        
        # Ensure all required fields are present in the schema
        result_schema = {
            'optimal_weights': final_weights or {},
            'performance_summary': performance_summary or {},
            'backtest_data': backtest_results.get('performance_data', {}) if backtest_results else {},
            'transaction_analysis': backtest_results.get('transaction_analysis', {}) if backtest_results else {},
            'equity_curve': equity_curve,
            'trades': trades_df,
            'success': True # Add explicit success flag
        }
        
        return result_schema

def main():
    """
    Orchestrates the analysis of multiple trading strategies, consolidates
    the results, and saves them to a single JSON file for frontend consumption.
    """
    import json
    from .config import STRATEGY_CONFIGS, MAX_POSITION_SIZE

    logger.info("="*80)
    logger.info("📈 Orchestrating Full Analysis for All Pre-defined Strategies 📈")
    logger.info("="*80)

    all_strategies_results = {}
    
    # --- Define common paths ---
    analysis_data_path = CACHE_DIR / 'full_analysis_ready_data.feather'
    price_data_dir = DATA_DIR / 'full_market_data_csvs'
    
    # --- Loop through each strategy, run analysis, and store results ---
    for strategy_name, strategy_config in STRATEGY_CONFIGS.items():
        logger.info(f"\n{'='*30} Running Analysis for: {strategy_name.upper()} {'='*30}")
        
        # 1. Instantiate the optimizer with the specific strategy's configuration
        optimizer = MultiFactorOptimizer(
            analysis_data_path=analysis_data_path,
            price_data_dir=price_data_dir,
            max_position_size=MAX_POSITION_SIZE,
            factor_weights=strategy_config['factor_weights'],
            momentum_period=strategy_config['momentum_period']
        )
        
        # 2. Run the comprehensive analysis
        results = optimizer.run_full_analysis()
        
        # 3. Store the results in the master dictionary
        if results:
            all_strategies_results[strategy_name] = results
            logger.info(f"✅ Successfully completed analysis for '{strategy_name}'.")
        else:
            logger.warning(f"⚠️ Analysis for '{strategy_name}' failed or produced no results.")

    # --- Save the consolidated results to a JSON file ---
    if all_strategies_results:
        output_path = RESULTS_DIR / 'final_results.json'
        logger.info(f"\n💾 Saving consolidated results for all strategies to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Use a custom converter to handle non-serializable types like numpy int64
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super(NpEncoder, self).default(obj)

                json.dump(all_strategies_results, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            logger.info("✅ Successfully saved the final JSON output.")
        except Exception as e:
            logger.error(f"❌ Failed to save JSON file: {e}")
    else:
        logger.warning("No strategies produced results. Nothing to save.")

    logger.info("\n🏁 Orchestration complete. 🏁")

if __name__ == "__main__":
    main()
