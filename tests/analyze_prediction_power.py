#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تحلیل نهایی: آیا می‌توانیم استراتژی موثری پیدا کنیم؟
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_prediction_correlations():
    """
    تحلیل همبستگی بین عملکرد گذشته و آینده
    """
    logger.info("📊 تحلیل قدرت پیش‌بینی...")
    
    # بارگذاری داده‌ها
    df = pd.read_csv('walk_forward_analysis_results.csv')
    
    # محاسبه همبستگی‌ها
    correlations = {}
    p_values = {}
    
    for period in ['1y', '3y', '5y']:
        # Pearson correlation
        corr, p_val = pearsonr(df[f'sharpe_{period}'], df['future_sharpe'])
        correlations[f'{period}_pearson'] = corr
        p_values[f'{period}_pearson'] = p_val
        
        # Spearman correlation (rank-based)
        corr_spear, p_val_spear = spearmanr(df[f'sharpe_{period}'], df['future_sharpe'])
        correlations[f'{period}_spearman'] = corr_spear
        p_values[f'{period}_spearman'] = p_val_spear
    
    return correlations, p_values, df

def analyze_strategy_consistency():
    """
    تحلیل ثبات عملکرد استراتژی‌ها
    """
    logger.info("🔍 تحلیل ثبات استراتژی‌ها...")
    
    df = pd.read_csv('walk_forward_analysis_results.csv')
    
    # تحلیل ثبات هر استراتژی
    strategy_stats = {}
    
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        
        stats = {
            'mean_sharpe_1y': strategy_data['sharpe_1y'].mean(),
            'std_sharpe_1y': strategy_data['sharpe_1y'].std(),
            'mean_sharpe_5y': strategy_data['sharpe_5y'].mean(),
            'std_sharpe_5y': strategy_data['sharpe_5y'].std(),
            'mean_future_sharpe': strategy_data['future_sharpe'].mean(),
            'std_future_sharpe': strategy_data['future_sharpe'].std(),
            'consistency_1y': strategy_data['sharpe_1y'].std() / abs(strategy_data['sharpe_1y'].mean()) if strategy_data['sharpe_1y'].mean() != 0 else float('inf'),
            'consistency_5y': strategy_data['sharpe_5y'].std() / abs(strategy_data['sharpe_5y'].mean()) if strategy_data['sharpe_5y'].mean() != 0 else float('inf'),
            'future_consistency': strategy_data['future_sharpe'].std() / abs(strategy_data['future_sharpe'].mean()) if strategy_data['future_sharpe'].mean() != 0 else float('inf'),
            'positive_future_rate': (strategy_data['future_sharpe'] > 0).mean()
        }
        
        strategy_stats[strategy] = stats
    
    return strategy_stats

def calculate_realistic_expectations():
    """
    محاسبه انتظارات واقع‌بینانه
    """
    logger.info("💡 محاسبه انتظارات واقع‌بینانه...")
    
    df = pd.read_csv('walk_forward_analysis_results.csv')
    
    # آمار کلی
    overall_stats = {
        'mean_future_sharpe': df['future_sharpe'].mean(),
        'median_future_sharpe': df['future_sharpe'].median(),
        'std_future_sharpe': df['future_sharpe'].std(),
        'positive_future_rate': (df['future_sharpe'] > 0).mean(),
        'sharpe_above_1_rate': (df['future_sharpe'] > 1.0).mean(),
        'sharpe_above_2_rate': (df['future_sharpe'] > 2.0).mean(),
        'max_future_sharpe': df['future_sharpe'].max(),
        'min_future_sharpe': df['future_sharpe'].min(),
        'percentile_75': df['future_sharpe'].quantile(0.75),
        'percentile_25': df['future_sharpe'].quantile(0.25)
    }
    
    return overall_stats

def analyze_market_regime_impact():
    """
    تحلیل تأثیر رژیم بازار
    """
    logger.info("📈 تحلیل تأثیر رژیم بازار...")
    
    df = pd.read_csv('walk_forward_analysis_results.csv')
    df['analysis_date'] = pd.to_datetime(df['analysis_date'])
    
    # تعریف رژیم‌های بازار بر اساس عملکرد آینده
    df['market_regime'] = pd.cut(df['future_sharpe'], 
                                bins=[-np.inf, -1, 1, np.inf], 
                                labels=['Bear', 'Neutral', 'Bull'])
    
    regime_analysis = {}
    for regime in df['market_regime'].unique():
        if pd.isna(regime):
            continue
            
        regime_data = df[df['market_regime'] == regime]
        
        regime_analysis[regime] = {
            'count': len(regime_data),
            'mean_1y_sharpe': regime_data['sharpe_1y'].mean(),
            'mean_5y_sharpe': regime_data['sharpe_5y'].mean(),
            'mean_future_sharpe': regime_data['future_sharpe'].mean(),
            'correlation_1y': regime_data['sharpe_1y'].corr(regime_data['future_sharpe']),
            'correlation_5y': regime_data['sharpe_5y'].corr(regime_data['future_sharpe'])
        }
    
    return regime_analysis

def generate_final_assessment():
    """
    تولید ارزیابی نهایی
    """
    logger.info("🎯 تولید ارزیابی نهایی...")
    
    # اجرای تحلیل‌ها
    correlations, p_values, df = analyze_prediction_correlations()
    strategy_stats = analyze_strategy_consistency()
    realistic_expectations = calculate_realistic_expectations()
    regime_analysis = analyze_market_regime_impact()
    
    # تولید گزارش
    report = []
    report.append("=" * 80)
    report.append("🎯 ارزیابی نهایی: آیا می‌توانیم استراتژی موثری پیدا کنیم؟")
    report.append("=" * 80)
    
    # 1. قدرت پیش‌بینی
    report.append("\n📊 1. قدرت پیش‌بینی افق‌های زمانی:")
    for period in ['1y', '3y', '5y']:
        pearson_corr = correlations[f'{period}_pearson']
        spearman_corr = correlations[f'{period}_spearman']
        p_val = p_values[f'{period}_pearson']
        
        significance = "معنادار" if p_val < 0.05 else "غیرمعنادار"
        
        report.append(f"   - {period}: Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f} ({significance})")
    
    # 2. انتظارات واقع‌بینانه
    report.append(f"\n💡 2. انتظارات واقع‌بینانه:")
    report.append(f"   - میانگین Sharpe آینده: {realistic_expectations['mean_future_sharpe']:.2f}")
    report.append(f"   - میانه Sharpe آینده: {realistic_expectations['median_future_sharpe']:.2f}")
    report.append(f"   - احتمال Sharpe مثبت: {realistic_expectations['positive_future_rate']:.1%}")
    report.append(f"   - احتمال Sharpe > 1: {realistic_expectations['sharpe_above_1_rate']:.1%}")
    report.append(f"   - احتمال Sharpe > 2: {realistic_expectations['sharpe_above_2_rate']:.1%}")
    report.append(f"   - بهترین حالت: {realistic_expectations['max_future_sharpe']:.2f}")
    report.append(f"   - بدترین حالت: {realistic_expectations['min_future_sharpe']:.2f}")
    
    # 3. ثبات استراتژی‌ها
    report.append(f"\n🔍 3. ثبات استراتژی‌ها:")
    best_strategy = None
    best_score = -float('inf')
    
    for strategy, stats in strategy_stats.items():
        # امتیاز ترکیبی (میانگین آینده - عدم ثبات)
        score = stats['mean_future_sharpe'] - stats['future_consistency']
        if score > best_score:
            best_score = score
            best_strategy = strategy
        
        report.append(f"   - {strategy}:")
        report.append(f"     * میانگین Sharpe آینده: {stats['mean_future_sharpe']:.2f}")
        report.append(f"     * نرخ موفقیت: {stats['positive_future_rate']:.1%}")
        report.append(f"     * ثبات: {1/stats['future_consistency']:.2f}" if stats['future_consistency'] != float('inf') else "     * ثبات: نامحدود")
    
    # 4. تأثیر رژیم بازار
    report.append(f"\n📈 4. تأثیر رژیم بازار:")
    for regime, stats in regime_analysis.items():
        report.append(f"   - {regime} Market:")
        report.append(f"     * تعداد مشاهدات: {stats['count']}")
        report.append(f"     * همبستگی 1Y: {stats['correlation_1y']:.3f}")
        report.append(f"     * همبستگی 5Y: {stats['correlation_5y']:.3f}")
    
    # 5. نتیجه‌گیری نهایی
    report.append(f"\n🏆 5. نتیجه‌گیری نهایی:")
    
    max_correlation = max([abs(correlations[f'{period}_pearson']) for period in ['1y', '3y', '5y']])
    
    if max_correlation < 0.3:
        report.append("   ❌ قدرت پیش‌بینی ضعیف:")
        report.append("     - همبستگی‌ها پایین هستند (< 0.3)")
        report.append("     - عملکرد گذشته پیش‌بین ضعیفی از آینده است")
        report.append("     - استراتژی‌های فعلی قابل اعتماد نیستند")
        
        recommendation = "CRITICAL"
    elif max_correlation < 0.5:
        report.append("   ⚠️  قدرت پیش‌بینی متوسط:")
        report.append("     - همبستگی‌ها متوسط هستند (0.3-0.5)")
        report.append("     - عملکرد گذشته تا حدی قابل اعتماد است")
        report.append("     - نیاز به بهبود روش‌ها")
        
        recommendation = "MODERATE"
    else:
        report.append("   ✅ قدرت پیش‌بینی قابل قبول:")
        report.append("     - همبستگی‌ها قوی هستند (> 0.5)")
        report.append("     - عملکرد گذشته پیش‌بین مناسبی از آینده است")
        report.append("     - استراتژی‌های فعلی قابل استفاده هستند")
        
        recommendation = "ACCEPTABLE"
    
    # 6. توصیه‌های عملی
    report.append(f"\n💡 6. توصیه‌های عملی:")
    
    if recommendation == "CRITICAL":
        report.append("   🚨 وضعیت بحرانی - نیاز به تغییر رویکرد:")
        report.append("     - استفاده از Machine Learning پیشرفته")
        report.append("     - تحلیل رژیم بازار (Regime Analysis)")
        report.append("     - ترکیب فاکتورهای جدید (Alternative Data)")
        report.append("     - استفاده از روش‌های Ensemble")
        report.append("     - کاهش انتظارات به Sharpe < 1")
    elif recommendation == "MODERATE":
        report.append("   ⚠️  نیاز به بهبود:")
        report.append("     - بهینه‌سازی پارامترهای موجود")
        report.append("     - اضافه کردن فیلترهای رژیم بازار")
        report.append("     - استفاده از Walk-Forward Optimization")
        report.append("     - انتظار Sharpe 1-2")
    else:
        report.append("   ✅ ادامه با بهبودهای جزئی:")
        report.append("     - بهینه‌سازی دوره‌ای پارامترها")
        report.append("     - نظارت مستمر بر عملکرد")
        report.append("     - انتظار Sharpe 2-3")
    
    report.append(f"\n   🎯 بهترین استراتژی فعلی: {best_strategy}")
    report.append(f"   📊 امتیاز ترکیبی: {best_score:.2f}")
    
    return report, recommendation

def main():
    """
    تابع اصلی
    """
    logger.info("🚀 شروع تحلیل نهایی قدرت پیش‌بینی")
    logger.info("=" * 70)
    
    try:
        report, recommendation = generate_final_assessment()
        
        # نمایش گزارش
        for line in report:
            logger.info(line)
        
        # ذخیره گزارش
        with open('final_prediction_assessment.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"\n✅ گزارش نهایی ذخیره شد: final_prediction_assessment.txt")
        logger.info(f"🎯 توصیه کلی: {recommendation}")
        
        return recommendation
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()