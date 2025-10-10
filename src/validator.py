# -*- coding: utf-8 -*-

# ============================================================================== 
# Title: Walk-Forward Backtester for Strategy Validation
# Description: A robust, out-of-sample validation engine to test portfolio
#              strategies against lookahead bias and overfitting.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================
from pathlib import Path
import logging
import pandas as pd
import json

# --- Import the refactored optimizer ---
from src.optimizer import MultiFactorOptimizer
from src.config import RISK_FREE_RATE, COMMISSION_RATE, SLIPPAGE_PCT
from src.strategy_tester import re_evaluate_top_strategies
from src.strategy_selector import StrategySelector
from src.config import CANDIDATES_PER_CATEGORY
import pandas as pd


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
        logging.FileHandler(LOGS_DIR / 'validator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# -------------------------------------------------
# محور 1 - خروجی کامل‌تر برای هر سه پروفایل ریسک
# -------------------------------------------------

def validate_optimizer_output_schema(result):
    """
    Validates that the optimizer output conforms to the expected schema.
    Returns the validated result or None if validation fails.
    """
    if not isinstance(result, dict):
        logger.error("Optimizer output is not a dictionary")
        return None
    
    required_keys = ['performance_summary', 'equity_curve', 'trades', 'success']
    missing_keys = [key for key in required_keys if key not in result]
    
    if missing_keys:
        logger.error(f"Optimizer output missing required keys: {missing_keys}")
        return None
    
    # Validate performance_summary structure
    perf_summary = result.get('performance_summary', {})
    if not isinstance(perf_summary, dict):
        logger.error("performance_summary is not a dictionary")
        return None
    
    # Validate equity_curve
    equity_curve = result.get('equity_curve')
    if equity_curve is not None and not isinstance(equity_curve, (pd.Series, pd.DataFrame)):
        logger.warning("equity_curve is not a pandas Series or DataFrame")
    
    # Validate trades
    trades = result.get('trades')
    if trades is not None and not isinstance(trades, pd.DataFrame):
        logger.warning("trades is not a pandas DataFrame")
    
    # Check success flag
    if result.get('success') is False:
        logger.error("Optimizer reported failure in result")
        return None
    
    return result

def build_enriched_output(portfolio, profile_name):
    """portfolio = backtest object / df"""
    equity_curve = portfolio.equity_curve
    returns = equity_curve.pct_change().dropna()

    # 1) 5 نقطه بازده سالانه (5 سال آخر)
    annual_returns = equity_curve.groupby(equity_curve.index.year).apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
    ).tail(5).tolist()

    # 2) Sharpe & Volatility rolling سالانه (5 سال آخر)
    sharpe_rolling = (
        returns.rolling(252).mean() / returns.rolling(252).std()
    ).dropna().groupby(lambda x: x.year).mean().tail(5).tolist()

    vol_roll = (
        returns.rolling(252).std() * (252 ** 0.5) * 100
    ).dropna().groupby(lambda x: x.year).mean().tail(5).tolist()

    # 3) 1-year KPI
    one_year_ret = ((equity_curve.iloc[-1] / equity_curve.iloc[-252]) - 1) * 100
    one_year_dd = portfolio.max_drawdown * 100  # می‌توان دقیق‌تر کرد اگر لازم است

    # 4) Full Transaction History
    # Provides a detailed log of all individual buy/sell transactions.
    transaction_history_df = portfolio.trades.copy()
    
    # Convert datetime columns to string for JSON serialization
    for col in transaction_history_df.select_dtypes(include=['datetime64[ns]']).columns:
        transaction_history_df[col] = transaction_history_df[col].dt.strftime('%Y-%m-%d')
    
    # Round float values for cleaner output
    for col in transaction_history_df.select_dtypes(include=['float']).columns:
        transaction_history_df[col] = transaction_history_df[col].round(4)
        
    # Replace potential NaN/inf values with strings to prevent JSON errors
    transaction_history_df.fillna('N/A', inplace=True)
    
    transaction_history = transaction_history_df.to_dict(orient="records")


    return {
        "max_drawdown": round(portfolio.max_drawdown * 100, 2),
        "kpi_sparklines": {
            "Annualized Return": annual_returns,
            "Sharpe Ratio": sharpe_rolling,
            "Annualized Volatility": vol_roll,
        },
        "1y_change": {
            "Total Return": round(one_year_ret, 2),
            "Max Drawdown": round(one_year_dd, 2)
        },
        "transaction_history": transaction_history
    }

def validate_and_select_best_strategies(top_n_df: pd.DataFrame):
    """
    Categorizes strategies by risk, selects the best candidates from each category,
    runs walk-forward validation on them, and selects the final best strategy for each approach.

    Args:
        top_n_df (pd.DataFrame): DataFrame of the top N strategies to be validated (configurable via TOP_REEVALUATION_COUNT).
    """
    if top_n_df.empty:
        logger.error("Received an empty DataFrame. Cannot proceed with validation.")
        return

    # --- 1. Use Strategy Selector for Risk Categorization and Candidate Selection ---
    logger.info("\n--- Using Strategy Selector for Risk-Based Categorization and Candidate Selection ---")
    
    strategy_selector = StrategySelector()
    candidate_portfolios = strategy_selector.select_final_candidates(top_n_df, candidates_per_category=CANDIDATES_PER_CATEGORY)

    final_results = {}

    # --- 3. Combine and Save the 15 Candidates to a CSV File ---
    all_candidates_list = []
    for approach, df in candidate_portfolios.items():
        df_copy = df.copy()
        df_copy['Risk Profile'] = approach
        all_candidates_list.append(df_copy)
    
    all_candidates_df = pd.concat(all_candidates_list, ignore_index=True)
    
    # Ensure the data directory exists
    (DATA_DIR).mkdir(exist_ok=True)
    candidates_output_path = DATA_DIR / 'top_15_validation_candidates.csv'
    
    try:
        all_candidates_df.to_csv(candidates_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Successfully saved the 15 validation candidates to {candidates_output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save candidates CSV: {e}")

    # --- 4. Run Walk-Forward Validation on All 15 Candidates ---
    for approach, candidates_df in candidate_portfolios.items():
        logger.info(f"\n" + "="*70)
        logger.info(f"      VALIDATING CANDIDATES FOR: {approach.upper()} APPROACH")
        logger.info("="*70)
        
        validation_results_list = []

        for index, candidate in candidates_df.iterrows():
            factor_weights = {
                'Value': candidate['Value Weight'],
                'Momentum': candidate['Momentum Weight'],
                'Low_Volatility': candidate['Low Volatility Weight']
            }
            momentum_period = candidate['Momentum Period']
            top_n = int(candidate['Top N'])
            max_weight = candidate['Max Weight']

            logger.info(f"\n--- Running Validation for Candidate: P={momentum_period}, W={factor_weights}, TopN={top_n}, MaxW={max_weight} ---")
            
            try:
                # Use MultiFactorOptimizer directly for validation run
                optimizer = MultiFactorOptimizer(
                    analysis_data_path=CACHE_DIR / 'full_analysis_ready_data.feather',
                    price_data_dir=DATA_DIR / 'full_market_data_csvs',
                    max_position_size=max_weight,
                    factor_weights=factor_weights,
                    momentum_period=momentum_period,
                    top_n_candidates=top_n,
                    commission_rate=COMMISSION_RATE,
                    slippage_pct=SLIPPAGE_PCT
                )
                
                # Perform a 5-year walk-forward validation
                validation_run_results = optimizer.run_full_analysis(years=5)

                # Validate the optimizer output against expected schema
                validated_results = validate_optimizer_output_schema(validation_run_results)
                
                if validated_results:
                    # Store the original candidate info within the full results object
                    validated_results['original_candidate'] = candidate.to_dict()
                    
                    validation_results_list.append(validated_results)
                    logger.info(f"✅ Validation Success! Sharpe: {validated_results['performance_summary'].get('Sharpe Ratio', 'N/A')}")
                else:
                    logger.warning("Validation run did not produce a valid output schema.")

            except Exception as e:
                logger.error(f"❌ FAILED validation for candidate. Error: {e}", exc_info=True)

        # --- 4. Select the Best Strategy for the Approach Based on Validation ---
        if not validation_results_list:
            logger.error(f"No successful validation runs for {approach} approach. Cannot select a final strategy.")
            continue
   
        # Calculate multi-criteria score for each strategy
        def calculate_multi_criteria_score(result):
            perf_summary = result.get('performance_summary', {})
            
            # Extract metrics with default values
            sharpe = float(perf_summary.get('Sharpe Ratio', 0))
            
            # Handle Total Return - convert percentage string to float
            total_return_str = perf_summary.get('Total Return', 0)
            if isinstance(total_return_str, str) and '%' in total_return_str:
                total_return = float(total_return_str.strip('%'))
            else:
                total_return = float(total_return_str)
            
            annual_return = float(perf_summary.get('Annualized Return', 0))
            volatility = float(perf_summary.get('Annualized Volatility', 1.0))
            
            # Handle Max Drawdown - convert percentage string to float
            max_dd_str = perf_summary.get('Max Drawdown [%]', 0)
            if isinstance(max_dd_str, str) and '%' in max_dd_str:
                max_dd = float(max_dd_str.strip('%')) / 100
            else:
                max_dd = float(max_dd_str) / 100 if max_dd_str else 0
            
            # Calculate Sortino ratio (if possible)
            downside_risk = volatility * 0.7  # Simplified approximation if not available
            if volatility > 0 and downside_risk > 0:
                sortino = (annual_return - RISK_FREE_RATE) / downside_risk
            else:
                sortino = sharpe  # Fallback to Sharpe if cannot calculate Sortino
            
            # Calculate stability score (inverse of volatility for lower risk)
            stability_score = 1 / (1 + volatility) if volatility > 0 else 1
            
            # Calculate Calmar ratio (return over max drawdown)
            if max_dd != 0:
                calmar = annual_return / abs(max_dd)
            else:
                calmar = annual_return  # Handle case where max drawdown is 0
            
            # Define weights for different criteria (configurable)
            weights = {
                'sharpe': 0.3,
                'sortino': 0.2,
                'calmar': 0.2,
                'stability': 0.15,
                'return': 0.15
            }
            
            # Calculate weighted score
            weighted_score = (
                weights['sharpe'] * sharpe +
                weights['sortino'] * sortino +
                weights['calmar'] * calmar +
                weights['stability'] * stability_score +
                weights['return'] * annual_return
            )
            
            return weighted_score
   
        # Select the best strategy based on the multi-criteria score
        best_strategy_result = max(
            validation_results_list,
            key=calculate_multi_criteria_score
        )

        # Extract original candidate info to create a dedicated strategy configuration dict
        original_candidate = best_strategy_result.pop('original_candidate', {})
        strategy_config = {
            'Momentum Period': original_candidate.get('Momentum Period'),
            'Value Weight': original_candidate.get('Value Weight'),
            'Momentum Weight': original_candidate.get('Momentum Weight'),
            'Low Volatility Weight': original_candidate.get('Low Volatility Weight'),
            'Top N': original_candidate.get('Top N'),
            'Max Weight': original_candidate.get('Max Weight')
        }

        # Re-structure the final result for this approach
        final_results[approach] = {
            "strategy_configuration": strategy_config,
            **best_strategy_result
        }
        
        logger.info(f"\n--- 🏆 BEST STRATEGY FOR {approach.upper()} APPROACH (Post-Validation) ---")
        logger.info(f"Sharpe Ratio: {final_results[approach]['performance_summary'].get('Sharpe Ratio')}")
        logger.info(f"Configuration: {strategy_config}")

    # --- 5. Log Final Selections ---
    logger.info("\n\n" + "="*80)
    logger.info("🎉🎉🎉 FINAL STRATEGY SELECTION COMPLETE 🎉🎉🎉")
    logger.info("="*80)
    for approach, result in final_results.items():
        logger.info(f"\n--- FINAL SELECTION for {approach.upper()} ---")
        logger.info(f"  Validation Sharpe Ratio: {result.get('performance_summary', {}).get('Sharpe Ratio')}")
        logger.info(f"  Original Candidate Config:")
        for key, val in result.get('original_candidate', {}).items():
            logger.info(f"    {key}: {val}")
    
    return final_results


def main():
    """
    Main pipeline execution:
    1. Re-evaluates the top N strategies (configurable via TOP_REEVALUATION_COUNT).
    2. Validates and selects the best final strategies based on risk profiles.
    3. Enriches the output with detailed metrics.
    4. Saves the final results to a JSON file.
    """
    logger.info("="*70)
    logger.info("      STARTING FULL STRATEGY VALIDATION PIPELINE")
    logger.info("="*70)

    # 1. Get the top N re-evaluated strategies (configurable via TOP_REEVALUATION_COUNT)
    top_n_results_df = re_evaluate_top_strategies()

    # 2. Validate and select the best strategies from the top N (configurable)
    if top_n_results_df is None or top_n_results_df.empty:
        logger.error("Halting pipeline because re-evaluation of top strategies failed or returned no results.")
        return

    raw_results = validate_and_select_best_strategies(top_n_results_df)

    if not raw_results:
        logger.error("Validation step did not return any results. Halting.")
        return

    # -------------------------------------------------
    # Enirch output for each of the three risk profiles
    # -------------------------------------------------
    final_results = {}
    for profile, result_data in raw_results.items():
        # Assumption: The backtest object is passed under the key 'backtest_obj'
        # In the context of this file, the full backtest result is the `result_data` itself
        portfolio_obj = result_data.get('backtest_obj') # A more robust key would be better

        # Let's assume the optimizer returns the backtrader `cerebro` object or similar
        # and that the `run_full_analysis` returns a dictionary where one key holds the backtest object
        # Based on the existing code, `best_strategy_result` is the dict.
        # Let's assume the backtest object is what `build_enriched_output` needs.
        # The `MultiFactorOptimizer` would need to return this object.
        # For now, let's assume `result_data` is a dictionary that contains what we need.
        # A better approach would be to have the optimizer return a class instance.
        
        # The user's code expects `portfolio.equity_curve` and `portfolio.trades`.
        # The `MultiFactorOptimizer` returns a dictionary. Let's check `optimizer.py` to see what it returns.
        # Let's assume for now that the `result_data` is the portfolio object itself.
        # This seems unlikely.
        
        # Let's stick to the user's note: "اگر portfolio.trades یا portfolio.equity_curve نام متفاوتی دارد، همان نام واقعی را جایگزین کنید."
        # This implies the object exists. The most likely candidate is `best_strategy_result` which becomes `result_data`.
        # The `optimizer.run_full_analysis` returns `validation_run_results`.
        # This is appended to `validation_results_list`.
        # Then `best_strategy_result` is selected from this list.
        # So `result_data` is one of the `validation_run_results` dictionaries.
        
        # The `build_enriched_output` expects an object with `.equity_curve` and `.trades` attributes.
        # The `validation_run_results` is a dictionary. This will fail.
        # The user's code is based on an incorrect assumption about the data structure.
        
        # I must adapt. I will assume the dictionary `result_data` contains keys 'equity_curve' and 'trades'.
        # The `build_enriched_output` needs to be adapted to take a dictionary.
        
        # Let's modify `build_enriched_output` to be more robust.
        # No, let's follow the user's instructions as closely as possible.
        # The user said "portfolio = backtest object / df".
        # The `MultiFactorOptimizer` must be returning an object.
        
        # Let's look at the `optimizer.py`... I can't. I'll have to make a smart guess.
        # The `run_full_analysis` in `optimizer.py` returns a dictionary.
        # The dictionary contains `performance_summary`, `equity_curve`, `trades`, etc.
        # So `result_data` is the dictionary.
        
        # Validate the result data before creating the proxy object
        validated_result = validate_optimizer_output_schema(result_data)
        if not validated_result:
            logger.warning(f"Schema validation failed for {profile} profile. Cannot enrich results.")
            final_results[profile] = result_data
            continue

        # I will create a temporary object to pass to the function to satisfy the `.attribute` access.
        class PortfolioProxy:
            def __init__(self, data_dict):
                self.equity_curve = data_dict.get('equity_curve')
                self.trades = data_dict.get('trades')
                # Handle max_drawdown properly by checking if it's a percentage string
                max_dd_value = data_dict.get('performance_summary', {}).get('Max Drawdown [%]', 0)
                if isinstance(max_dd_value, str) and '%' in max_dd_value:
                    self.max_drawdown = float(max_dd_value.strip('%')) / 100
                elif pd.isna(max_dd_value):
                    self.max_drawdown = 0
                else:
                    self.max_drawdown = float(max_dd_value) / 100  # Convert from percentage

        portfolio_proxy = PortfolioProxy(validated_result)

        if portfolio_proxy.equity_curve is not None and portfolio_proxy.trades is not None:
            logger.info(f"Enriching results for {profile} profile...")
            enriched_data = build_enriched_output(portfolio_proxy, profile)

            # Combine original results with the new enriched data
            # We should not save the raw equity curve and trades in the final JSON.
            combined_data = result_data.copy()
            if 'equity_curve' in combined_data: del combined_data['equity_curve']
            if 'trades' in combined_data: del combined_data['trades']
            
            combined_data.update(enriched_data)
            final_results[profile] = combined_data
        else:
            logger.warning(f"Could not find 'equity_curve' or 'trades' for profile '{profile}'. Cannot enrich results.")
            final_results[profile] = result_data


    # -------------------------------------------------
    # Save final results to JSON
    # -------------------------------------------------
    output_path = RESULTS_DIR / 'final_results.json'
    RESULTS_DIR.mkdir(exist_ok=True) # Ensure results directory exists
    logger.info(f"💾 Saving final enriched results to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Custom JSON serializer to handle pandas Timestamps and other non-serializable types
            def json_default(o):
                if isinstance(o, (pd.Timestamp, pd.Period)):
                    return o.isoformat()
                if isinstance(o, float) and (pd.isna(o) or o == float('inf') or o == float('-inf')):
                    return str(o)
                # Add more type checks if necessary
                try:
                    return str(o) # Fallback for other types
                except:
                    return f"NON-SERIALIZABLE: {type(o)}"

            json.dump(final_results, f, ensure_ascii=False, indent=4, default=json_default)
        logger.info("✅ Successfully saved final results.")
    except Exception as e:
        logger.error(f"❌ Failed to save final_results.json: {e}", exc_info=True)


    logger.info("\n" + "="*70)
    logger.info("      PIPELINE FINISHED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
