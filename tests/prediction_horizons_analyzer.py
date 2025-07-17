#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول تحلیل قدرت پیش‌بینی افق‌های زمانی

این ماژول شامل کلاس‌های تحلیل همبستگی و دقت پیش‌بینی بین افق‌های مختلف زمانی است.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings

from data_management import DataLoader, AnalysisCache

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionHorizonsAnalyzer:
    """
    کلاس تحلیل قدرت پیش‌بینی افق‌های زمانی
    """
    
    def __init__(self, config: Dict = None):
        """
        مقداردهی اولیه
        
        Args:
            config: تنظیمات تحلیل
        """
        self.config = config or {}
        self.horizons = self.config.get('horizons', ['1y', '3y', '5y'])
        self.confidence_level = self.config.get('confidence_level', 0.05)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.5)
        
        # ابزارهای کمکی
        self.data_loader = DataLoader()
        self.cache = AnalysisCache(cache_dir="tests/cache")
        
        # نتایج تحلیل
        self.correlation_results = {}
        self.prediction_accuracy = {}
        self.analysis_summary = {}
        
        logger.info("PredictionHorizonsAnalyzer initialized")
    
    def load_horizon_data(self) -> Dict[str, pd.DataFrame]:
        """
        بارگذاری داده‌های افق‌های مختلف
        
        Returns:
            دیکشنری حاوی داده‌های هر افق
        """
        try:
            logger.info("📊 Loading horizon data...")
            
            horizon_data = {}
            
            # بارگذاری داده‌های 1 ساله
            if '1y' in self.horizons:
                data_1y = self.data_loader.load_backtest_data('1y')
                if not data_1y.empty:
                    horizon_data['1y'] = data_1y
                    logger.info(f"✅ Loaded 1-year data: {len(data_1y)} records")
            
            # بارگذاری داده‌های 5 ساله
            if '5y' in self.horizons:
                data_5y = self.data_loader.load_backtest_data('5y')
                if not data_5y.empty:
                    horizon_data['5y'] = data_5y
                    logger.info(f"✅ Loaded 5-year data: {len(data_5y)} records")
            
            # برای 3 ساله، از داده‌های موجود استفاده می‌کنیم
            if '3y' in self.horizons:
                # فعلاً از داده‌های 5 ساله استفاده می‌کنیم
                if '5y' in horizon_data:
                    horizon_data['3y'] = horizon_data['5y'].copy()
                    logger.info("📝 Using 5-year data as proxy for 3-year analysis")
            
            if not horizon_data:
                raise ValueError("No horizon data could be loaded")
            
            logger.info(f"✅ Successfully loaded data for {len(horizon_data)} horizons")
            return horizon_data
            
        except Exception as e:
            logger.error(f"❌ Error loading horizon data: {e}")
            raise
    
    def calculate_correlations(self, horizon_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        محاسبه همبستگی بین افق‌های مختلف
        
        Args:
            horizon_data: داده‌های افق‌های مختلف
            
        Returns:
            نتایج همبستگی
        """
        try:
            logger.info("🔗 Calculating correlations between horizons...")
            
            correlations = {}
            
            # شناسایی ستون‌های مشترک برای تحلیل
            common_columns = self._find_common_columns(horizon_data)
            
            if not common_columns:
                raise ValueError("No common performance columns found")
            
            logger.info(f"📊 Analyzing correlations for columns: {common_columns}")
            
            # محاسبه همبستگی بین هر جفت افق
            horizon_pairs = []
            for i, h1 in enumerate(self.horizons):
                for j, h2 in enumerate(self.horizons):
                    if i < j and h1 in horizon_data and h2 in horizon_data:
                        horizon_pairs.append((h1, h2))
            
            for h1, h2 in horizon_pairs:
                logger.info(f"🔍 Calculating correlation between {h1} and {h2}...")
                
                pair_correlations = {}
                
                for column in common_columns:
                    if column in horizon_data[h1].columns and column in horizon_data[h2].columns:
                        # داده‌های مشترک
                        data1 = horizon_data[h1][column].dropna()
                        data2 = horizon_data[h2][column].dropna()
                        
                        if len(data1) > 10 and len(data2) > 10:
                            # همبستگی Pearson
                            pearson_corr, pearson_p = pearsonr(data1[:min(len(data1), len(data2))], 
                                                             data2[:min(len(data1), len(data2))])
                            
                            # همبستگی Spearman
                            spearman_corr, spearman_p = spearmanr(data1[:min(len(data1), len(data2))], 
                                                                data2[:min(len(data1), len(data2))])
                            
                            pair_correlations[column] = {
                                'pearson_correlation': pearson_corr,
                                'pearson_p_value': pearson_p,
                                'pearson_significant': pearson_p < self.confidence_level,
                                'spearman_correlation': spearman_corr,
                                'spearman_p_value': spearman_p,
                                'spearman_significant': spearman_p < self.confidence_level,
                                'sample_size': min(len(data1), len(data2))
                            }
                            
                            logger.info(f"  📈 {column}: Pearson={pearson_corr:.3f} (p={pearson_p:.3f}), "
                                      f"Spearman={spearman_corr:.3f} (p={spearman_p:.3f})")
                
                correlations[f"{h1}_vs_{h2}"] = pair_correlations
            
            self.correlation_results = correlations
            logger.info("✅ Correlation analysis completed")
            
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlations: {e}")
            raise
    
    def _find_common_columns(self, horizon_data: Dict[str, pd.DataFrame]) -> List[str]:
        """یافتن ستون‌های مشترک برای تحلیل"""
        try:
            # ستون‌های احتمالی برای تحلیل عملکرد
            performance_columns = [
                'Sharpe_Ratio', 'Total_Return', 'Volatility', 'Max_Drawdown',
                'Annual_Return', 'Win_Rate', 'Profit_Factor', 'Sortino_Ratio'
            ]
            
            common_columns = []
            
            for col in performance_columns:
                # بررسی وجود ستون در تمام افق‌ها
                exists_in_all = True
                for horizon, data in horizon_data.items():
                    if col not in data.columns:
                        exists_in_all = False
                        break
                
                if exists_in_all:
                    common_columns.append(col)
            
            return common_columns
            
        except Exception as e:
            logger.error(f"Error finding common columns: {e}")
            return []
    
    def calculate_prediction_accuracy(self, horizon_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        محاسبه دقت پیش‌بینی برای هر افق
        
        Args:
            horizon_data: داده‌های افق‌های مختلف
            
        Returns:
            نتایج دقت پیش‌بینی
        """
        try:
            logger.info("🎯 Calculating prediction accuracy...")
            
            accuracy_results = {}
            
            for horizon, data in horizon_data.items():
                logger.info(f"📊 Analyzing prediction accuracy for {horizon}...")
                
                horizon_accuracy = {}
                
                # دقت علامت (Sign Accuracy)
                if 'Sharpe_Ratio' in data.columns:
                    sharpe_values = data['Sharpe_Ratio'].dropna()
                    if len(sharpe_values) > 0:
                        positive_sharpe = (sharpe_values > 0).sum()
                        sign_accuracy = positive_sharpe / len(sharpe_values)
                        horizon_accuracy['sign_accuracy'] = sign_accuracy
                        
                        logger.info(f"  📈 Sign accuracy: {sign_accuracy:.3f} "
                                  f"({positive_sharpe}/{len(sharpe_values)} positive)")
                
                # دقت رتبه‌بندی
                if 'Sharpe_Ratio' in data.columns and len(data) > 10:
                    # رتبه‌بندی بر اساس Sharpe Ratio
                    data_ranked = data.copy()
                    data_ranked['Sharpe_Rank'] = data_ranked['Sharpe_Ratio'].rank(ascending=False)
                    
                    # محاسبه consistency رتبه‌بندی
                    top_quartile = data_ranked['Sharpe_Rank'] <= len(data_ranked) * 0.25
                    ranking_consistency = self._calculate_ranking_consistency(data_ranked)
                    
                    horizon_accuracy['ranking_accuracy'] = ranking_consistency
                    horizon_accuracy['top_quartile_count'] = top_quartile.sum()
                    
                    logger.info(f"  🏆 Ranking consistency: {ranking_consistency:.3f}")
                
                # دقت انتخاب بهترین استراتژی
                if 'Sharpe_Ratio' in data.columns:
                    best_strategy_accuracy = self._calculate_best_strategy_accuracy(data)
                    horizon_accuracy['best_strategy_accuracy'] = best_strategy_accuracy
                    
                    logger.info(f"  🥇 Best strategy accuracy: {best_strategy_accuracy:.3f}")
                
                accuracy_results[horizon] = horizon_accuracy
            
            self.prediction_accuracy = accuracy_results
            logger.info("✅ Prediction accuracy analysis completed")
            
            return accuracy_results
            
        except Exception as e:
            logger.error(f"❌ Error calculating prediction accuracy: {e}")
            raise
    
    def _calculate_ranking_consistency(self, data: pd.DataFrame) -> float:
        """محاسبه consistency رتبه‌بندی"""
        try:
            if 'Sharpe_Ratio' not in data.columns or len(data) < 5:
                return 0.0
            
            # تقسیم داده‌ها به دو نیمه
            mid_point = len(data) // 2
            first_half = data.iloc[:mid_point]
            second_half = data.iloc[mid_point:]
            
            if len(first_half) < 3 or len(second_half) < 3:
                return 0.0
            
            # رتبه‌بندی هر نیمه
            first_ranks = first_half['Sharpe_Ratio'].rank(ascending=False)
            second_ranks = second_half['Sharpe_Ratio'].rank(ascending=False)
            
            # محاسبه همبستگی رتبه‌ها (اگر امکان‌پذیر باشد)
            if len(first_ranks) == len(second_ranks):
                correlation, _ = spearmanr(first_ranks, second_ranks)
                return max(0, correlation)
            else:
                # روش جایگزین: مقایسه top performers
                first_top = set(first_half.nlargest(3, 'Sharpe_Ratio').index)
                second_top = set(second_half.nlargest(3, 'Sharpe_Ratio').index)
                
                if first_top and second_top:
                    overlap = len(first_top.intersection(second_top))
                    return overlap / max(len(first_top), len(second_top))
                
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating ranking consistency: {e}")
            return 0.0
    
    def _calculate_best_strategy_accuracy(self, data: pd.DataFrame) -> float:
        """محاسبه دقت انتخاب بهترین استراتژی"""
        try:
            if 'Sharpe_Ratio' not in data.columns or len(data) < 5:
                return 0.0
            
            # انتخاب top 10% استراتژی‌ها
            top_n = max(1, len(data) // 10)
            top_strategies = data.nlargest(top_n, 'Sharpe_Ratio')
            
            # بررسی consistency عملکرد top strategies
            if len(top_strategies) > 0:
                avg_sharpe = top_strategies['Sharpe_Ratio'].mean()
                overall_avg = data['Sharpe_Ratio'].mean()
                
                if overall_avg != 0:
                    relative_performance = avg_sharpe / overall_avg
                    return min(1.0, max(0.0, (relative_performance - 1.0)))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating best strategy accuracy: {e}")
            return 0.0
    
    def generate_visualizations(self, output_dir: str = "tests/results/prediction_analysis") -> Dict[str, str]:
        """
        تولید نمودارهای تحلیل
        
        Args:
            output_dir: مسیر ذخیره نمودارها
            
        Returns:
            مسیرهای فایل‌های تولید شده
        """
        try:
            logger.info("📊 Generating visualizations...")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            generated_files = {}
            
            # تنظیم matplotlib برای فارسی
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
            
            # نمودار همبستگی‌ها
            if self.correlation_results:
                correlation_plot = self._create_correlation_heatmap(output_path)
                if correlation_plot:
                    generated_files['correlation_heatmap'] = correlation_plot
            
            # نمودار دقت پیش‌بینی
            if self.prediction_accuracy:
                accuracy_plot = self._create_accuracy_chart(output_path)
                if accuracy_plot:
                    generated_files['accuracy_chart'] = accuracy_plot
            
            logger.info(f"✅ Generated {len(generated_files)} visualization files")
            return generated_files
            
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")
            return {}
    
    def _create_correlation_heatmap(self, output_path: Path) -> Optional[str]:
        """ایجاد heatmap همبستگی‌ها"""
        try:
            # آماده‌سازی داده‌ها برای heatmap
            correlation_matrix = {}
            
            for pair, correlations in self.correlation_results.items():
                for metric, values in correlations.items():
                    if metric not in correlation_matrix:
                        correlation_matrix[metric] = {}
                    correlation_matrix[metric][pair] = values.get('pearson_correlation', 0)
            
            if not correlation_matrix:
                return None
            
            # تبدیل به DataFrame
            df_corr = pd.DataFrame(correlation_matrix).T
            
            # ایجاد نمودار
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_corr, annot=True, cmap='RdYlBu_r', center=0, 
                       fmt='.3f', square=True, cbar_kws={'label': 'Correlation'})
            
            plt.title('Correlation Between Different Time Horizons', fontsize=16, pad=20)
            plt.xlabel('Horizon Pairs', fontsize=12)
            plt.ylabel('Performance Metrics', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # ذخیره
            file_path = output_path / "correlation_heatmap.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Correlation heatmap saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return None
    
    def _create_accuracy_chart(self, output_path: Path) -> Optional[str]:
        """ایجاد نمودار دقت پیش‌بینی"""
        try:
            # آماده‌سازی داده‌ها
            horizons = list(self.prediction_accuracy.keys())
            metrics = ['sign_accuracy', 'ranking_accuracy', 'best_strategy_accuracy']
            
            data_for_plot = []
            for horizon in horizons:
                for metric in metrics:
                    if metric in self.prediction_accuracy[horizon]:
                        data_for_plot.append({
                            'Horizon': horizon,
                            'Metric': metric.replace('_', ' ').title(),
                            'Accuracy': self.prediction_accuracy[horizon][metric]
                        })
            
            if not data_for_plot:
                return None
            
            df_accuracy = pd.DataFrame(data_for_plot)
            
            # ایجاد نمودار
            plt.figure(figsize=(12, 8))
            
            # نمودار ستونی گروه‌بندی شده
            sns.barplot(data=df_accuracy, x='Horizon', y='Accuracy', hue='Metric')
            
            plt.title('Prediction Accuracy Across Different Time Horizons', fontsize=16, pad=20)
            plt.xlabel('Time Horizon', fontsize=12)
            plt.ylabel('Accuracy Score', fontsize=12)
            plt.ylim(0, 1)
            plt.legend(title='Accuracy Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # ذخیره
            file_path = output_path / "prediction_accuracy.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Accuracy chart saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error creating accuracy chart: {e}")
            return None
    
    def generate_analysis_report(self, output_file: str = "tests/results/prediction_analysis/horizons_analysis_report.json") -> Dict[str, Any]:
        """
        تولید گزارش کامل تحلیل
        
        Args:
            output_file: مسیر فایل گزارش
            
        Returns:
            گزارش کامل
        """
        try:
            logger.info("📝 Generating analysis report...")
            
            # خلاصه نتایج
            summary = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'horizons_analyzed': self.horizons,
                'correlation_results': self.correlation_results,
                'prediction_accuracy': self.prediction_accuracy,
                'key_findings': self._extract_key_findings(),
                'recommendations': self._generate_recommendations()
            }
            
            # ذخیره گزارش
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            self.analysis_summary = summary
            logger.info(f"📄 Analysis report saved: {output_file}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating analysis report: {e}")
            raise
    
    def _extract_key_findings(self) -> List[str]:
        """استخراج یافته‌های کلیدی"""
        findings = []
        
        try:
            # تحلیل همبستگی‌ها
            if self.correlation_results:
                strong_correlations = []
                for pair, correlations in self.correlation_results.items():
                    for metric, values in correlations.items():
                        if abs(values.get('pearson_correlation', 0)) > self.correlation_threshold:
                            strong_correlations.append(f"{pair} - {metric}: {values['pearson_correlation']:.3f}")
                
                if strong_correlations:
                    findings.append(f"Strong correlations found: {len(strong_correlations)} metric-pair combinations")
                else:
                    findings.append("No strong correlations found between time horizons")
            
            # تحلیل دقت پیش‌بینی
            if self.prediction_accuracy:
                best_horizon = None
                best_accuracy = 0
                
                for horizon, accuracy in self.prediction_accuracy.items():
                    avg_accuracy = np.mean([v for v in accuracy.values() if isinstance(v, (int, float))])
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_horizon = horizon
                
                if best_horizon:
                    findings.append(f"Best performing horizon: {best_horizon} with average accuracy {best_accuracy:.3f}")
            
        except Exception as e:
            logger.warning(f"Error extracting key findings: {e}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """تولید توصیه‌های عملی"""
        recommendations = []
        
        try:
            # بر اساس همبستگی‌ها
            if self.correlation_results:
                avg_correlations = {}
                for pair, correlations in self.correlation_results.items():
                    correlations_values = [v.get('pearson_correlation', 0) for v in correlations.values()]
                    avg_correlations[pair] = np.mean(correlations_values)
                
                if avg_correlations:
                    best_pair = max(avg_correlations, key=avg_correlations.get)
                    recommendations.append(f"Consider using {best_pair.replace('_vs_', ' and ')} for consistent analysis")
            
            # بر اساس دقت پیش‌بینی
            if self.prediction_accuracy:
                for horizon, accuracy in self.prediction_accuracy.items():
                    sign_acc = accuracy.get('sign_accuracy', 0)
                    if sign_acc > 0.6:
                        recommendations.append(f"{horizon} horizon shows good sign prediction accuracy ({sign_acc:.3f})")
                    elif sign_acc < 0.4:
                        recommendations.append(f"{horizon} horizon shows poor sign prediction accuracy ({sign_acc:.3f}) - use with caution")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        اجرای کامل تحلیل افق‌های زمانی
        
        Returns:
            نتایج کامل تحلیل
        """
        try:
            logger.info("🚀 Starting complete prediction horizons analysis...")
            
            # بارگذاری داده‌ها
            horizon_data = self.load_horizon_data()
            
            # محاسبه همبستگی‌ها
            correlations = self.calculate_correlations(horizon_data)
            
            # محاسبه دقت پیش‌بینی
            accuracy = self.calculate_prediction_accuracy(horizon_data)
            
            # تولید نمودارها
            visualizations = self.generate_visualizations()
            
            # تولید گزارش
            report = self.generate_analysis_report()
            
            logger.info("✅ Complete prediction horizons analysis finished successfully")
            
            return {
                'correlations': correlations,
                'accuracy': accuracy,
                'visualizations': visualizations,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"❌ Error in complete analysis: {e}")
            raise


def test_prediction_horizons_analyzer():
    """
    تست کلاس PredictionHorizonsAnalyzer
    """
    logger.info("🧪 Testing PredictionHorizonsAnalyzer...")
    
    try:
        # تنظیمات تست
        config = {
            'horizons': ['1y', '5y'],
            'confidence_level': 0.05,
            'correlation_threshold': 0.3
        }
        
        # ایجاد analyzer
        analyzer = PredictionHorizonsAnalyzer(config)
        
        # اجرای تحلیل کامل
        results = analyzer.run_complete_analysis()
        
        logger.info("✅ PredictionHorizonsAnalyzer test completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_prediction_horizons_analyzer()