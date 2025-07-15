# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Strategy Selector - Risk-Based Categorization and Candidate Selection
# Description: Takes the top strategies from strategy_tester and categorizes them
#              by risk level, then selects the best candidates from each category
# Author: Kiro Code, the AI Software Engineer
# ==============================================================================

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'strategy_selector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Categorizes strategies by risk level and selects the best candidates from each category.
    """
    
    def __init__(self):
        """Initialize the Strategy Selector."""
        self.logger = logging.getLogger('StrategySelector')
        
    def categorize_by_risk(self, strategies_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Categorizes strategies by risk level based on Annualized Volatility.
        
        Args:
            strategies_df (pd.DataFrame): DataFrame containing strategy results with 'Annualized Volatility' column
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with risk categories as keys and DataFrames as values
        """
        self.logger.info("--- Categorizing Strategies by Risk Level ---")
        
        # Ensure required column exists
        if 'Annualized Volatility' not in strategies_df.columns:
            raise ValueError("'Annualized Volatility' column not found in strategies DataFrame")
        
        # Sort by volatility (ascending - lowest risk first)
        sorted_strategies = strategies_df.sort_values(by='Annualized Volatility', ascending=True)
        
        n = len(sorted_strategies)
        defensive_count = int(n * 0.30)  # Lowest 30% volatility
        aggressive_start_index = int(n * 0.70)  # Highest 30% volatility
        
        # Split into three risk categories
        defensive_strategies = sorted_strategies.iloc[:defensive_count]
        balanced_strategies = sorted_strategies.iloc[defensive_count:aggressive_start_index]
        aggressive_strategies = sorted_strategies.iloc[aggressive_start_index:]
        
        self.logger.info(f"Defensive Strategies (Lowest 30% Volatility): {len(defensive_strategies)} strategies")
        self.logger.info(f"  - Volatility range: {defensive_strategies['Annualized Volatility'].min():.3f} to {defensive_strategies['Annualized Volatility'].max():.3f}")
        
        self.logger.info(f"Balanced Strategies (Middle 40% Volatility): {len(balanced_strategies)} strategies")
        self.logger.info(f"  - Volatility range: {balanced_strategies['Annualized Volatility'].min():.3f} to {balanced_strategies['Annualized Volatility'].max():.3f}")
        
        self.logger.info(f"Aggressive Strategies (Highest 30% Volatility): {len(aggressive_strategies)} strategies")
        self.logger.info(f"  - Volatility range: {aggressive_strategies['Annualized Volatility'].min():.3f} to {aggressive_strategies['Annualized Volatility'].max():.3f}")
        
        return {
            'Defensive': defensive_strategies,
            'Balanced': balanced_strategies,
            'Aggressive': aggressive_strategies
        }
    
    def select_top_candidates(self, categorized_strategies: Dict[str, pd.DataFrame], 
                            candidates_per_category: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Selects top candidates from each risk category based on Sharpe Ratio.
        
        Args:
            categorized_strategies (Dict[str, pd.DataFrame]): Risk-categorized strategies
            candidates_per_category (int): Number of candidates to select from each category
            
        Returns:
            Dict[str, pd.DataFrame]: Top candidates from each category
        """
        self.logger.info(f"--- Selecting Top {candidates_per_category} Candidates from Each Risk Category ---")
        
        selected_candidates = {}
        
        for risk_level, strategies_df in categorized_strategies.items():
            if 'Sharpe Ratio' not in strategies_df.columns:
                raise ValueError(f"'Sharpe Ratio' column not found in {risk_level} strategies DataFrame")
            
            # Select top candidates by Sharpe Ratio
            top_candidates = strategies_df.nlargest(candidates_per_category, 'Sharpe Ratio')
            selected_candidates[risk_level] = top_candidates
            
            self.logger.info(f"{risk_level} - Selected {len(top_candidates)} candidates:")
            for idx, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
                self.logger.info(f"  {idx}. Sharpe: {candidate['Sharpe Ratio']:.3f}, "
                               f"Volatility: {candidate['Annualized Volatility']:.3f}, "
                               f"Return: {candidate.get('Annualized Return', 'N/A')}")
        
        return selected_candidates
    
    def select_final_candidates(self, strategies_df: pd.DataFrame, 
                              candidates_per_category: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Complete pipeline: categorize by risk and select top candidates.
        
        Args:
            strategies_df (pd.DataFrame): Input strategies DataFrame
            candidates_per_category (int): Number of candidates per category
            
        Returns:
            Dict[str, pd.DataFrame]: Final selected candidates by risk category
        """
        self.logger.info("🚀 Starting Strategy Selection Process...")
        
        # Step 1: Categorize by risk
        categorized_strategies = self.categorize_by_risk(strategies_df)
        
        # Step 2: Select top candidates from each category
        final_candidates = self.select_top_candidates(categorized_strategies, candidates_per_category)
        
        # Summary
        total_candidates = sum(len(candidates) for candidates in final_candidates.values())
        self.logger.info("="*60)
        self.logger.info("🎯 STRATEGY SELECTION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total candidates selected: {total_candidates}")
        for risk_level, candidates in final_candidates.items():
            self.logger.info(f"  - {risk_level}: {len(candidates)} candidates")
        
        return final_candidates

def main():
    """
    Main function for testing the Strategy Selector independently.
    """
    logger.info("="*60)
    logger.info("TESTING STRATEGY SELECTOR")
    logger.info("="*60)
    
    # This would normally be called from the main pipeline
    # For testing, you would need to provide a sample DataFrame
    logger.info("Strategy Selector is ready. Use select_final_candidates() method with strategy data.")

if __name__ == "__main__":
    main()