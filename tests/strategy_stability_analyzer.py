#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول تحلیل ثبات استراتژی‌های سرمایه‌گذاری
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from data_management import DataLoader, AnalysisCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyStabilityAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rolling_window = self.config.get('rolling_window', 252)
        self.stability_threshold = self.config.get('stability_threshold', 0.2)
        self.consistency_threshold = self.config.get('consistency_threshold', 0.7)
        
        self.data_loader = DataLoader()
        self.cache = AnalysisCache(cache_dir="tests/cache")
        
        self.stability_metrics = {}
        self.stability_ranking = {}
        
        logger.info("StrategyStabilityAnalyzer initialized")
    
    def load_strategy_data(self) -> pd.DataFrame:
        try:
            logger.info("📊 Loading strategy data for stability analysis...")
            
            data_1y = self.data_loader.load_backtest_data('1y')
            data_5y = self.data_loader.load_backtest_data('5y')
            
            if not data_1y.empty and not data_5y.empty:
                data_1y['Period'] = '1y'
                data_5y['Period'] = '5y'
                combined_data = pd.concat([data_1y, data_5y], ignore_index=True)
                logger.info(f"✅ Combined data loaded: {len(combined_data)} records")
                return combined_data
            elif not data_1y.empty:
                data_1y['Period'] = '1y'
                return data_1y
            elif not data_5y.empty:
                data_5y['Period'] = '5y'
                return data_5y
            else:
                raise ValueError("No strategy data available")
                
        except Exception as e:
            logger.error(f"❌ Error loading strategy data: {e}")
            raise
    
    def calculate_stability_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("🔄 Calculating stability metrics...")
            
            stability_results = {}
            
            if 'Candidate_Index' in data.columns:
                grouped = data.groupby('Candidate_Index')
                
                for candidate_idx, group in grouped:
                    logger.info(f"📊 Analyzing stability for strategy {candidate_idx}...")
                    
                    metrics = {}
                    
                    # محاسبه Consistency Score
                    if 'Sharpe_Ratio' in group.columns:
                        positive_periods = (group['Sharpe_Ratio'] > 0).sum()
                        total_periods = len(group['Sharpe_Ratio'].dropna())
                        consistency_ratio = positive_periods / total_periods if total_periods > 0 else 0
                        metrics['consistency_ratio'] = consistency_ratio
                    
                    # محاسبه Stability Score
                    stability_score = self._calculate_stability_score(group)
                    metrics['stability_score'] = stability_score
                    
                    # تحلیل الگوهای شکست
                    failure_analysis = self._analyze_failure_patterns(group)
                    metrics.update(failure_analysis)
                    
                    stability_results[f"strategy_{candidate_idx}"] = metrics
                    
                    logger.info(f"  ✅ Strategy {candidate_idx}: Stability={stability_score:.3f}")
            
            self.stability_metrics = stability_results
            logger.info("✅ Stability metrics calculation completed")
            
            return stability_results
            
        except Exception as e:
            logger.error(f"❌ Error calculating stability metrics: {e}")
            raise
    
    def _calculate_stability_score(self, data: pd.DataFrame) -> float:
        try:
            if len(data) < 2:
                return 0.0
            
            scores = []
            
            # امتیاز بر اساس consistency
            if 'Sharpe_Ratio' in data.columns:
                positive_ratio = (data['Sharpe_Ratio'] > 0).mean()
                scores.append(positive_ratio)
            
            # امتیاز بر اساس پایداری بازدهی
            if 'Total_Return' in data.columns:
                return_values = data['Total_Return'].dropna()
                if len(return_values) > 1:
                    return_std = return_values.std()
                    return_mean = return_values.mean()
                    if return_mean > 0:
                        return_stability = max(0, 1 - (return_std / return_mean))
                        scores.append(return_stability)
            
            final_score = np.mean(scores) if scores else 0.0
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.warning(f"Error calculating stability score: {e}")
            return 0.0
    
    def _analyze_failure_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            patterns = {}
            
            if 'Sharpe_Ratio' in data.columns and len(data) > 2:
                sharpe_values = data['Sharpe_Ratio'].dropna()
                
                weak_periods = sharpe_values < 0
                
                if weak_periods.any():
                    patterns['weak_periods_count'] = weak_periods.sum()
                    patterns['weak_periods_ratio'] = weak_periods.mean()
                    
                    negative_sharpe = sharpe_values[weak_periods]
                    if len(negative_sharpe) > 0:
                        patterns['avg_weakness_severity'] = abs(negative_sharpe.mean())
                else:
                    patterns['weak_periods_count'] = 0
                    patterns['weak_periods_ratio'] = 0.0
                    patterns['avg_weakness_severity'] = 0.0
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error analyzing failure patterns: {e}")
            return {}
    
    def rank_strategies_by_stability(self) -> Dict[str, Any]:
        try:
            logger.info("🏆 Ranking strategies by stability...")
            
            if not self.stability_metrics:
                raise ValueError("No stability metrics available for ranking")
            
            stability_scores = {}
            for strategy, metrics in self.stability_metrics.items():
                stability_scores[strategy] = metrics.get('stability_score', 0.0)
            
            sorted_strategies = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            
            ranking = {}
            for rank, (strategy, score) in enumerate(sorted_strategies, 1):
                ranking[strategy] = {
                    'rank': rank,
                    'stability_score': score,
                    'stability_level': self._get_stability_level(score),
                    'full_metrics': self.stability_metrics[strategy]
                }
            
            self.stability_ranking = ranking
            
            logger.info("🏆 Top 5 most stable strategies:")
            for strategy, info in list(ranking.items())[:5]:
                logger.info(f"  {info['rank']}. {strategy}: {info['stability_score']:.3f} ({info['stability_level']})")
            
            return ranking
            
        except Exception as e:
            logger.error(f"❌ Error ranking strategies: {e}")
            raise
    
    def _get_stability_level(self, score: float) -> str:
        if score >= 0.8:
            return "Very Stable"
        elif score >= 0.6:
            return "Stable"
        elif score >= 0.4:
            return "Moderately Stable"
        elif score >= 0.2:
            return "Unstable"
        else:
            return "Very Unstable"
    
    def generate_analysis_report(self, output_file: str = "tests/results/stability_analysis/stability_report.json") -> Dict[str, Any]:
        try:
            logger.info("📝 Generating stability analysis report...")
            
            summary = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'analysis_parameters': {
                    'rolling_window': self.rolling_window,
                    'stability_threshold': self.stability_threshold,
                    'consistency_threshold': self.consistency_threshold
                },
                'stability_metrics': self.stability_metrics,
                'stability_ranking': self.stability_ranking,
                'key_findings': self._extract_key_findings(),
                'recommendations': self._generate_recommendations()
            }
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Stability analysis report saved: {output_file}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating analysis report: {e}")
            raise
    
    def _extract_key_findings(self) -> List[str]:
        findings = []
        
        try:
            if not self.stability_metrics:
                return findings
            
            stability_scores = [metrics['stability_score'] for metrics in self.stability_metrics.values()]
            mean_stability = np.mean(stability_scores)
            
            if mean_stability > 0.7:
                findings.append(f"Overall high stability: average score {mean_stability:.3f}")
            elif mean_stability > 0.5:
                findings.append(f"Moderate stability: average score {mean_stability:.3f}")
            else:
                findings.append(f"Low stability concern: average score {mean_stability:.3f}")
            
            if self.stability_ranking:
                best_strategy = list(self.stability_ranking.keys())[0]
                best_score = self.stability_ranking[best_strategy]['stability_score']
                findings.append(f"Most stable strategy: {best_strategy} (score: {best_score:.3f})")
            
        except Exception as e:
            logger.warning(f"Error extracting key findings: {e}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        recommendations = []
        
        try:
            if not self.stability_ranking:
                return recommendations
            
            stable_strategies = [s for s, info in self.stability_ranking.items() 
                               if info['stability_score'] > 0.7]
            
            if stable_strategies:
                recommendations.append(f"Consider these stable strategies: {', '.join(stable_strategies[:3])}")
            
            mean_stability = np.mean([info['stability_score'] for info in self.stability_ranking.values()])
            if mean_stability < 0.5:
                recommendations.append("Overall low stability - consider diversification")
            else:
                recommendations.append("Good overall stability - focus on top performers")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        try:
            logger.info("🚀 Starting complete stability analysis...")
            
            strategy_data = self.load_strategy_data()
            stability_metrics = self.calculate_stability_metrics(strategy_data)
            ranking = self.rank_strategies_by_stability()
            report = self.generate_analysis_report()
            
            logger.info("✅ Complete stability analysis finished successfully")
            
            return {
                'stability_metrics': stability_metrics,
                'ranking': ranking,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"❌ Error in complete analysis: {e}")
            raise


def test_strategy_stability_analyzer():
    logger.info("🧪 Testing StrategyStabilityAnalyzer...")
    
    try:
        config = {
            'rolling_window': 252,
            'stability_threshold': 0.2,
            'consistency_threshold': 0.7
        }
        
        analyzer = StrategyStabilityAnalyzer(config)
        results = analyzer.run_complete_analysis()
        
        logger.info("✅ StrategyStabilityAnalyzer test completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_strategy_stability_analyzer()