#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله 2: تحلیل ثبات استراتژی‌ها
بررسی ثبات عملکرد هر استراتژی در طول زمان و شناسایی الگوهای شکست
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_backtest_data():
    """
    بارگذاری تمام داده‌های بک‌تست موجود
    """
    logger.info("📊 بارگذاری داده‌های بک‌تست...")
    
    data = {}
    
    # بارگذاری داده‌های مختلف
    files_to_load = {
        'walkforward': 'walk_forward_analysis_results.csv',
        'backtest_1y': 'backtest_1year_results.csv',
        'backtest_5y': 'backtest_5year_results.csv',
        'comprehensive': 'comprehensive_backtest_results.csv',
        'test_50': 'test_50_strategies_results.csv'
    }
    
    for key, filename in files_to_load.items():
        try:
            df = pd.read_csv(filename)
            data[key] = df
            logger.info(f"✅ {key}: {len(df)} رکورد")
        except FileNotFoundError:
            logger.warning(f"⚠️ فایل {filename} یافت نشد")
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری {filename}: {e}")
    
    return data

def analyze_temporal_stability(walkforward_df):
    """
    تحلیل ثبات زمانی استراتژی‌ها
    """
    logger.info("⏰ تحلیل ثبات زمانی...")
    
    stability_results = {}
    
    for strategy in walkforward_df['strategy'].unique():
        strategy_data = walkforward_df[walkforward_df['strategy'] == strategy].copy()
        strategy_data['analysis_date'] = pd.to_datetime(strategy_data['analysis_date'])
        strategy_data = strategy_data.sort_values('analysis_date')
        
        if len(strategy_data) < 5:
            continue
        
        # محاسبه معیارهای ثبات
        stability_metrics = {}
        
        # ثبات Sharpe Ratio در افق‌های مختلف
        for period in ['1y', '3y', '5y']:
            sharpe_values = strategy_data[f'sharpe_{period}'].dropna()
            if len(sharpe_values) >= 3:
                stability_metrics[f'{period}_mean'] = sharpe_values.mean()
                stability_metrics[f'{period}_std'] = sharpe_values.std()
                stability_metrics[f'{period}_cv'] = sharpe_values.std() / abs(sharpe_values.mean()) if sharpe_values.mean() != 0 else np.inf
                stability_metrics[f'{period}_min'] = sharpe_values.min()
                stability_metrics[f'{period}_max'] = sharpe_values.max()
                stability_metrics[f'{period}_range'] = sharpe_values.max() - sharpe_values.min()
        
        # ثبات عملکرد آینده
        future_sharpe = strategy_data['future_sharpe'].dropna()
        if len(future_sharpe) >= 3:
            stability_metrics['future_mean'] = future_sharpe.mean()
            stability_metrics['future_std'] = future_sharpe.std()
            stability_metrics['future_cv'] = future_sharpe.std() / abs(future_sharpe.mean()) if future_sharpe.mean() != 0 else np.inf
            stability_metrics['future_positive_rate'] = (future_sharpe > 0).mean()
            stability_metrics['future_above_1_rate'] = (future_sharpe > 1).mean()
            stability_metrics['future_drawdown_rate'] = (future_sharpe < -1).mean()
        
        # تحلیل ترند
        if len(strategy_data) >= 5:
            # ترند عملکرد آینده
            x = np.arange(len(future_sharpe))
            if len(future_sharpe) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, future_sharpe)
                stability_metrics['future_trend_slope'] = slope
                stability_metrics['future_trend_r2'] = r_value ** 2
                stability_metrics['future_trend_significant'] = p_value < 0.05
        
        # امتیاز ثبات کلی (کمتر بهتر)
        cv_scores = [stability_metrics.get(f'{period}_cv', np.inf) for period in ['1y', '3y', '5y']]
        cv_scores = [cv for cv in cv_scores if cv != np.inf]
        stability_metrics['overall_stability_score'] = np.mean(cv_scores) if cv_scores else np.inf
        
        stability_results[strategy] = stability_metrics
        
        logger.info(f"📊 {strategy}: ثبات={stability_metrics['overall_stability_score']:.2f}, "
                   f"عملکرد آینده={stability_metrics.get('future_mean', 0):.2f}")
    
    return stability_results

def analyze_performance_consistency(data):
    """
    تحلیل سازگاری عملکرد در بک‌تست‌های مختلف
    """
    logger.info("🔄 تحلیل سازگاری عملکرد...")
    
    consistency_results = {}
    
    # مقایسه نتایج 1 ساله و 5 ساله
    if 'backtest_1y' in data and 'backtest_5y' in data:
        df_1y = data['backtest_1y']
        df_5y = data['backtest_5y']
        
        # ترکیب داده‌ها بر اساس شاخص کاندیدا
        if 'Candidate_Index' in df_1y.columns and 'Candidate_Index' in df_5y.columns:
            merged = pd.merge(df_1y, df_5y, on='Candidate_Index', suffixes=('_1y', '_5y'))
            
            consistency_results['backtest_comparison'] = {
                'correlation_sharpe': merged['Test_1Y_Sharpe_Ratio'].corr(merged['Test_5Y_Sharpe_Ratio']),
                'correlation_return': merged['Test_1Y_Annualized_Return'].corr(merged['Test_5Y_Annualized_Return']),
                'correlation_volatility': merged['Test_1Y_Annualized_Volatility'].corr(merged['Test_5Y_Annualized_Volatility']),
                'mean_sharpe_decline': (merged['Test_1Y_Sharpe_Ratio'] - merged['Test_5Y_Sharpe_Ratio']).mean(),
                'sharpe_decline_std': (merged['Test_1Y_Sharpe_Ratio'] - merged['Test_5Y_Sharpe_Ratio']).std(),
                'consistent_performers': len(merged[(merged['Test_1Y_Sharpe_Ratio'] > 2) & (merged['Test_5Y_Sharpe_Ratio'] > 1)]),
                'total_strategies': len(merged)
            }
    
    # تحلیل تست 50 استراتژی
    if 'test_50' in data:
        df_50 = data['test_50']
        
        # گروه‌بندی بر اساس پارامترهای استراتژی
        parameter_consistency = {}
        
        # تحلیل بر اساس Momentum Period
        for momentum_period in df_50['Momentum_Period'].unique():
            subset = df_50[df_50['Momentum_Period'] == momentum_period]
            parameter_consistency[f'momentum_{momentum_period}'] = {
                'count': len(subset),
                'mean_test_sharpe': subset['Test_Sharpe_Ratio'].mean(),
                'std_test_sharpe': subset['Test_Sharpe_Ratio'].std(),
                'positive_rate': (subset['Test_Sharpe_Ratio'] > 0).mean(),
                'above_2_rate': (subset['Test_Sharpe_Ratio'] > 2).mean()
            }
        
        consistency_results['parameter_analysis'] = parameter_consistency
    
    return consistency_results

def identify_failure_patterns(data):
    """
    شناسایی الگوهای شکست استراتژی‌ها
    """
    logger.info("🔍 شناسایی الگوهای شکست...")
    
    failure_patterns = {}
    
    # تحلیل شکست در Walk-Forward
    if 'walkforward' in data:
        df = data['walkforward']
        
        # تعریف شکست: عملکرد آینده منفی
        failures = df[df['future_sharpe'] < -1].copy()
        successes = df[df['future_sharpe'] > 1].copy()
        
        failure_patterns['walkforward'] = {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(df),
            'avg_failure_magnitude': failures['future_sharpe'].mean() if len(failures) > 0 else 0,
            'worst_failure': failures['future_sharpe'].min() if len(failures) > 0 else 0
        }
        
        # الگوهای شکست بر اساس استراتژی
        strategy_failure_rates = {}
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            failure_rate = (strategy_data['future_sharpe'] < -1).mean()
            success_rate = (strategy_data['future_sharpe'] > 1).mean()
            
            strategy_failure_rates[strategy] = {
                'failure_rate': failure_rate,
                'success_rate': success_rate,
                'avg_performance': strategy_data['future_sharpe'].mean(),
                'volatility': strategy_data['future_sharpe'].std()
            }
        
        failure_patterns['strategy_failure_rates'] = strategy_failure_rates
        
        # الگوهای شکست بر اساس شرایط گذشته
        if len(failures) > 0:
            failure_conditions = {}
            
            for period in ['1y', '3y', '5y']:
                # آیا عملکرد بالای گذشته منجر به شکست می‌شود؟
                high_past_performance = df[df[f'sharpe_{period}'] > df[f'sharpe_{period}'].quantile(0.8)]
                failure_rate_high_past = (high_past_performance['future_sharpe'] < -1).mean()
                
                # آیا عملکرد پایین گذشته منجر به شکست می‌شود؟
                low_past_performance = df[df[f'sharpe_{period}'] < df[f'sharpe_{period}'].quantile(0.2)]
                failure_rate_low_past = (low_past_performance['future_sharpe'] < -1).mean()
                
                failure_conditions[f'{period}_high_past_failure_rate'] = failure_rate_high_past
                failure_conditions[f'{period}_low_past_failure_rate'] = failure_rate_low_past
            
            failure_patterns['failure_conditions'] = failure_conditions
    
    # تحلیل کاهش عملکرد در بک‌تست طولانی‌تر
    if 'backtest_1y' in data and 'backtest_5y' in data:
        df_1y = data['backtest_1y']
        df_5y = data['backtest_5y']
        
        if 'Candidate_Index' in df_1y.columns and 'Candidate_Index' in df_5y.columns:
            merged = pd.merge(df_1y, df_5y, on='Candidate_Index', suffixes=('_1y', '_5y'))
            
            # شناسایی استراتژی‌هایی که عملکردشان به شدت کاهش یافته
            sharpe_decline = merged['Test_1Y_Sharpe_Ratio'] - merged['Test_5Y_Sharpe_Ratio']
            severe_decline = sharpe_decline > sharpe_decline.quantile(0.8)
            
            failure_patterns['performance_decline'] = {
                'severe_decline_count': severe_decline.sum(),
                'severe_decline_rate': severe_decline.mean(),
                'avg_decline': sharpe_decline.mean(),
                'max_decline': sharpe_decline.max(),
                'decline_threshold': sharpe_decline.quantile(0.8)
            }
    
    return failure_patterns

def calculate_robustness_scores(stability_results, consistency_results, failure_patterns):
    """
    محاسبه امتیاز استحکام برای هر استراتژی
    """
    logger.info("💪 محاسبه امتیازهای استحکام...")
    
    robustness_scores = {}
    
    for strategy, stability in stability_results.items():
        score_components = {}
        
        # امتیاز ثبات (کمتر بهتر، پس معکوس می‌کنیم)
        stability_score = 1 / (1 + stability.get('overall_stability_score', np.inf))
        score_components['stability'] = stability_score
        
        # امتیاز عملکرد آینده
        future_mean = stability.get('future_mean', 0)
        future_positive_rate = stability.get('future_positive_rate', 0)
        performance_score = (future_mean + 5) / 10 * future_positive_rate  # نرمال‌سازی
        score_components['performance'] = performance_score
        
        # امتیاز کاهش نرخ شکست
        if 'walkforward' in failure_patterns and 'strategy_failure_rates' in failure_patterns['walkforward']:
            strategy_failure_data = failure_patterns['walkforward']['strategy_failure_rates'].get(strategy, {})
            failure_rate = strategy_failure_data.get('failure_rate', 1)
            failure_score = 1 - failure_rate
            score_components['failure_resistance'] = failure_score
        else:
            score_components['failure_resistance'] = 0.5  # نمره متوسط
        
        # امتیاز ترند
        trend_slope = stability.get('future_trend_slope', 0)
        trend_score = max(0, min(1, (trend_slope + 1) / 2))  # نرمال‌سازی بین 0 و 1
        score_components['trend'] = trend_score
        
        # امتیاز کلی (میانگین وزنی)
        weights = {
            'stability': 0.3,
            'performance': 0.4,
            'failure_resistance': 0.2,
            'trend': 0.1
        }
        
        overall_score = sum(score_components[key] * weights[key] for key in weights.keys())
        
        robustness_scores[strategy] = {
            'overall_score': overall_score,
            'components': score_components,
            'rank': 0  # خواهد شد پر شود
        }
    
    # رتبه‌بندی
    sorted_strategies = sorted(robustness_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    for rank, (strategy, data) in enumerate(sorted_strategies, 1):
        robustness_scores[strategy]['rank'] = rank
    
    return robustness_scores

def generate_stability_report(stability_results, consistency_results, failure_patterns, robustness_scores):
    """
    تولید گزارش جامع ثبات استراتژی‌ها
    """
    logger.info("📝 تولید گزارش ثبات...")
    
    report = []
    report.append("=" * 80)
    report.append("📊 گزارش تحلیل ثبات استراتژی‌ها")
    report.append("=" * 80)
    
    # 1. رتبه‌بندی استراتژی‌ها بر اساس استحکام
    report.append("\n🏆 1. رتبه‌بندی استراتژی‌ها بر اساس استحکام:")
    report.append("-" * 60)
    
    sorted_strategies = sorted(robustness_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    for strategy, data in sorted_strategies[:10]:  # نمایش 10 تای برتر
        report.append(f"   {data['rank']:2d}. {strategy}")
        report.append(f"       امتیاز کلی: {data['overall_score']:.3f}")
        report.append(f"       ثبات: {data['components']['stability']:.3f}")
        report.append(f"       عملکرد: {data['components']['performance']:.3f}")
        report.append(f"       مقاومت در برابر شکست: {data['components']['failure_resistance']:.3f}")
        report.append(f"       ترند: {data['components']['trend']:.3f}")
    
    # 2. تحلیل ثبات زمانی
    report.append(f"\n⏰ 2. تحلیل ثبات زمانی:")
    report.append("-" * 60)
    
    # آمار کلی ثبات
    stability_scores = [data['overall_stability_score'] for data in stability_results.values() 
                       if data['overall_stability_score'] != np.inf]
    
    if stability_scores:
        report.append(f"   • میانگین ضریب تغییرات: {np.mean(stability_scores):.3f}")
        report.append(f"   • بهترین ثبات: {min(stability_scores):.3f}")
        report.append(f"   • بدترین ثبات: {max(stability_scores):.3f}")
    
    # استراتژی‌های با بیشترین ثبات
    stable_strategies = sorted([(k, v['overall_stability_score']) for k, v in stability_results.items() 
                               if v['overall_stability_score'] != np.inf], key=lambda x: x[1])[:5]
    
    report.append(f"\n   🎯 پایدارترین استراتژی‌ها:")
    for strategy, score in stable_strategies:
        future_mean = stability_results[strategy].get('future_mean', 0)
        report.append(f"     • {strategy}: CV={score:.3f}, میانگین آینده={future_mean:.2f}")
    
    # 3. سازگاری عملکرد
    report.append(f"\n🔄 3. سازگاری عملکرد:")
    report.append("-" * 60)
    
    if 'backtest_comparison' in consistency_results:
        comp = consistency_results['backtest_comparison']
        report.append(f"   • همبستگی Sharpe (1Y vs 5Y): {comp['correlation_sharpe']:.3f}")
        report.append(f"   • میانگین کاهش Sharpe: {comp['mean_sharpe_decline']:.2f}")
        report.append(f"   • استراتژی‌های سازگار: {comp['consistent_performers']}/{comp['total_strategies']}")
    
    if 'parameter_analysis' in consistency_results:
        param_analysis = consistency_results['parameter_analysis']
        report.append(f"\n   📊 تحلیل پارامترها:")
        for param, stats in param_analysis.items():
            report.append(f"     • {param}: میانگین Sharpe={stats['mean_test_sharpe']:.2f}, "
                         f"نرخ مثبت={stats['positive_rate']:.1%}")
    
    # 4. الگوهای شکست
    report.append(f"\n🔍 4. الگوهای شکست:")
    report.append("-" * 60)
    
    if 'walkforward' in failure_patterns:
        wf_failures = failure_patterns['walkforward']
        report.append(f"   • نرخ شکست کلی: {wf_failures['failure_rate']:.1%}")
        report.append(f"   • میانگین شدت شکست: {wf_failures['avg_failure_magnitude']:.2f}")
        report.append(f"   • بدترین شکست: {wf_failures['worst_failure']:.2f}")
        
        # استراتژی‌های پرخطر
        if 'strategy_failure_rates' in wf_failures:
            risky_strategies = sorted(wf_failures['strategy_failure_rates'].items(), 
                                    key=lambda x: x[1]['failure_rate'], reverse=True)[:3]
            
            report.append(f"\n   ⚠️ پرخطرترین استراتژی‌ها:")
            for strategy, data in risky_strategies:
                report.append(f"     • {strategy}: نرخ شکست={data['failure_rate']:.1%}, "
                             f"میانگین عملکرد={data['avg_performance']:.2f}")
    
    if 'performance_decline' in failure_patterns:
        decline = failure_patterns['performance_decline']
        report.append(f"\n   📉 کاهش عملکرد در بک‌تست طولانی:")
        report.append(f"     • نرخ کاهش شدید: {decline['severe_decline_rate']:.1%}")
        report.append(f"     • میانگین کاهش: {decline['avg_decline']:.2f}")
        report.append(f"     • حداکثر کاهش: {decline['max_decline']:.2f}")
    
    # 5. توصیه‌های عملی
    report.append(f"\n💡 5. توصیه‌های عملی:")
    report.append("-" * 60)
    
    # بهترین استراتژی
    best_strategy = sorted_strategies[0][0] if sorted_strategies else "نامشخص"
    best_score = sorted_strategies[0][1]['overall_score'] if sorted_strategies else 0
    
    report.append(f"   🏆 بهترین استراتژی: {best_strategy} (امتیاز: {best_score:.3f})")
    
    # توصیه‌های کلی
    if best_score > 0.7:
        report.append(f"   ✅ کیفیت عالی: استراتژی‌های برتر قابل اعتماد هستند")
    elif best_score > 0.5:
        report.append(f"   ⚠️ کیفیت متوسط: با احتیاط استفاده کنید")
    else:
        report.append(f"   ❌ کیفیت پایین: نیاز به بازنگری اساسی")
    
    # توصیه‌های بهبود
    report.append(f"\n   🔧 راهکارهای بهبود:")
    
    if stability_scores and np.mean(stability_scores) > 1:
        report.append(f"     • کاهش نوسان عملکرد با تنظیم پارامترها")
    
    if 'backtest_comparison' in consistency_results:
        if consistency_results['backtest_comparison']['mean_sharpe_decline'] > 5:
            report.append(f"     • کاهش overfitting با validation بیشتر")
    
    if 'walkforward' in failure_patterns:
        if failure_patterns['walkforward']['failure_rate'] > 0.3:
            report.append(f"     • افزایش مقاومت با diversification")
    
    report.append(f"     • ترکیب چندین استراتژی برای کاهش ریسک")
    report.append(f"     • نظارت مستمر بر عملکرد")
    
    return report

def save_stability_results(stability_results, consistency_results, failure_patterns, robustness_scores, report):
    """
    ذخیره نتایج تحلیل ثبات
    """
    logger.info("💾 ذخیره نتایج تحلیل ثبات...")
    
    results_dir = Path("results/stability_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ذخیره گزارش اصلی
    with open(results_dir / "stability_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # تابع پاکسازی NaN برای JSON
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj
    
    # ذخیره داده‌های خام
    with open(results_dir / "stability_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(clean_for_json(stability_results), f, indent=2, ensure_ascii=False)
    
    with open(results_dir / "consistency_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(clean_for_json(consistency_results), f, indent=2, ensure_ascii=False)
    
    with open(results_dir / "failure_patterns.json", 'w', encoding='utf-8') as f:
        json.dump(clean_for_json(failure_patterns), f, indent=2, ensure_ascii=False)
    
    with open(results_dir / "robustness_scores.json", 'w', encoding='utf-8') as f:
        json.dump(clean_for_json(robustness_scores), f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ نتایج در {results_dir} ذخیره شد")

def main():
    """
    تابع اصلی تحلیل ثبات استراتژی‌ها
    """
    logger.info("🚀 شروع تحلیل ثبات استراتژی‌ها")
    logger.info("=" * 70)
    
    try:
        # بارگذاری داده‌ها
        data = load_all_backtest_data()
        
        if not data:
            logger.error("❌ هیچ داده‌ای بارگذاری نشد")
            return
        
        # تحلیل ثبات زمانی
        stability_results = {}
        if 'walkforward' in data:
            stability_results = analyze_temporal_stability(data['walkforward'])
        
        # تحلیل سازگاری عملکرد
        consistency_results = analyze_performance_consistency(data)
        
        # شناسایی الگوهای شکست
        failure_patterns = identify_failure_patterns(data)
        
        # محاسبه امتیازهای استحکام
        robustness_scores = calculate_robustness_scores(stability_results, consistency_results, failure_patterns)
        
        # تولید گزارش
        report = generate_stability_report(stability_results, consistency_results, failure_patterns, robustness_scores)
        
        # نمایش گزارش
        for line in report:
            logger.info(line)
        
        # ذخیره نتایج
        save_stability_results(stability_results, consistency_results, failure_patterns, robustness_scores, report)
        
        logger.info(f"\n🎉 تحلیل ثبات استراتژی‌ها با موفقیت کامل شد!")
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()