#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله 1: تحلیل قدرت پیش‌بینی افق‌های زمانی مختلف
بررسی اینکه کدام افق زمانی (1، 3، یا 5 ساله) بهترین پیش‌بین عملکرد آینده است
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging
from pathlib import Path

# تنظیم logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_walkforward_data():
    """
    بارگذاری داده‌های Walk-Forward Analysis
    """
    logger.info("📊 بارگذاری داده‌های Walk-Forward...")
    
    try:
        df = pd.read_csv('walk_forward_analysis_results.csv')
        logger.info(f"✅ {len(df)} رکورد بارگذاری شد")
        logger.info(f"📈 دوره زمانی: {df['analysis_date'].min()} تا {df['analysis_date'].max()}")
        logger.info(f"🎯 استراتژی‌ها: {df['strategy'].unique().tolist()}")
        return df
    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری داده‌ها: {e}")
        return None

def calculate_prediction_correlations(df):
    """
    محاسبه همبستگی بین عملکرد گذشته و آینده
    """
    logger.info("🔍 محاسبه همبستگی‌های پیش‌بینی...")
    
    results = {}
    
    for period in ['1y', '3y', '5y']:
        # حذف مقادیر NaN
        valid_data = df.dropna(subset=[f'sharpe_{period}', 'future_sharpe'])
        
        if len(valid_data) < 10:
            logger.warning(f"⚠️ داده کافی برای {period} وجود ندارد")
            continue
        
        # محاسبه همبستگی Pearson
        pearson_corr, pearson_p = pearsonr(valid_data[f'sharpe_{period}'], valid_data['future_sharpe'])
        
        # محاسبه همبستگی Spearman (rank-based)
        spearman_corr, spearman_p = spearmanr(valid_data[f'sharpe_{period}'], valid_data['future_sharpe'])
        
        # محاسبه R-squared
        r_squared = pearson_corr ** 2
        
        results[period] = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'r_squared': r_squared,
            'sample_size': len(valid_data),
            'significant': pearson_p < 0.05
        }
        
        logger.info(f"📊 {period}: Pearson={pearson_corr:.3f} (p={pearson_p:.3f}), "
                   f"Spearman={spearman_corr:.3f}, R²={r_squared:.3f}")
    
    return results

def analyze_prediction_accuracy(df):
    """
    تحلیل دقت پیش‌بینی رتبه‌بندی
    """
    logger.info("🎯 تحلیل دقت پیش‌بینی رتبه‌بندی...")
    
    accuracy_results = {}
    
    # گروه‌بندی بر اساس تاریخ تحلیل
    for period in ['1y', '3y', '5y']:
        correct_predictions = 0
        total_predictions = 0
        top_strategy_matches = 0
        
        for analysis_date in df['analysis_date'].unique():
            date_data = df[df['analysis_date'] == analysis_date].copy()
            
            if len(date_data) < 2:
                continue
            
            # رتبه‌بندی بر اساس عملکرد گذشته
            date_data[f'rank_{period}'] = date_data[f'sharpe_{period}'].rank(ascending=False)
            date_data['rank_future'] = date_data['future_sharpe'].rank(ascending=False)
            
            # بررسی آیا بهترین استراتژی گذشته در آینده هم بهترین است
            best_past = date_data[date_data[f'rank_{period}'] == 1]['strategy'].iloc[0]
            best_future = date_data[date_data['rank_future'] == 1]['strategy'].iloc[0]
            
            if best_past == best_future:
                top_strategy_matches += 1
            
            # محاسبه همبستگی رتبه‌ها
            rank_corr = date_data[f'rank_{period}'].corr(date_data['rank_future'])
            if not np.isnan(rank_corr):
                if rank_corr > 0.5:  # همبستگی قوی
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        top_match_rate = top_strategy_matches / len(df['analysis_date'].unique())
        
        accuracy_results[period] = {
            'ranking_accuracy': accuracy_rate,
            'top_strategy_match_rate': top_match_rate,
            'total_periods': total_predictions
        }
        
        logger.info(f"🎯 {period}: دقت رتبه‌بندی={accuracy_rate:.1%}, "
                   f"تطبیق استراتژی برتر={top_match_rate:.1%}")
    
    return accuracy_results

def analyze_by_strategy(df):
    """
    تحلیل عملکرد هر استراتژی به تفکیک
    """
    logger.info("📈 تحلیل عملکرد به تفکیک استراتژی...")
    
    strategy_results = {}
    
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].copy()
        
        if len(strategy_data) < 5:
            continue
        
        # آمار توصیفی
        stats = {
            'count': len(strategy_data),
            'mean_future_sharpe': strategy_data['future_sharpe'].mean(),
            'std_future_sharpe': strategy_data['future_sharpe'].std(),
            'positive_future_rate': (strategy_data['future_sharpe'] > 0).mean(),
            'sharpe_above_1_rate': (strategy_data['future_sharpe'] > 1).mean(),
            'max_future_sharpe': strategy_data['future_sharpe'].max(),
            'min_future_sharpe': strategy_data['future_sharpe'].min()
        }
        
        # همبستگی برای هر افق زمانی
        for period in ['1y', '3y', '5y']:
            valid_data = strategy_data.dropna(subset=[f'sharpe_{period}', 'future_sharpe'])
            if len(valid_data) >= 3:
                corr, _ = pearsonr(valid_data[f'sharpe_{period}'], valid_data['future_sharpe'])
                stats[f'correlation_{period}'] = corr
            else:
                stats[f'correlation_{period}'] = np.nan
        
        strategy_results[strategy] = stats
        
        logger.info(f"📊 {strategy}: میانگین آینده={stats['mean_future_sharpe']:.2f}, "
                   f"نرخ مثبت={stats['positive_future_rate']:.1%}")
    
    return strategy_results

def analyze_market_conditions(df):
    """
    تحلیل تأثیر شرایط بازار
    """
    logger.info("🌊 تحلیل تأثیر شرایط بازار...")
    
    # تعریف رژیم‌های بازار بر اساس عملکرد آینده
    df['market_regime'] = pd.cut(df['future_sharpe'], 
                                bins=[-np.inf, -1, 1, np.inf], 
                                labels=['Bear', 'Neutral', 'Bull'])
    
    regime_analysis = {}
    
    for regime in ['Bear', 'Neutral', 'Bull']:
        regime_data = df[df['market_regime'] == regime]
        
        if len(regime_data) == 0:
            continue
        
        regime_stats = {
            'count': len(regime_data),
            'percentage': len(regime_data) / len(df) * 100,
            'mean_future_sharpe': regime_data['future_sharpe'].mean()
        }
        
        # همبستگی در هر رژیم
        for period in ['1y', '3y', '5y']:
            valid_data = regime_data.dropna(subset=[f'sharpe_{period}', 'future_sharpe'])
            if len(valid_data) >= 3:
                corr, _ = pearsonr(valid_data[f'sharpe_{period}'], valid_data['future_sharpe'])
                regime_stats[f'correlation_{period}'] = corr
            else:
                regime_stats[f'correlation_{period}'] = np.nan
        
        regime_analysis[regime] = regime_stats
        
        logger.info(f"🌊 {regime}: {regime_stats['count']} مورد ({regime_stats['percentage']:.1f}%), "
                   f"میانگین={regime_stats['mean_future_sharpe']:.2f}")
    
    return regime_analysis

def generate_horizon_report(correlations, accuracy, strategy_analysis, regime_analysis):
    """
    تولید گزارش جامع تحلیل افق‌های زمانی
    """
    logger.info("📝 تولید گزارش جامع...")
    
    report = []
    report.append("=" * 80)
    report.append("📊 گزارش تحلیل قدرت پیش‌بینی افق‌های زمانی")
    report.append("=" * 80)
    
    # 1. خلاصه همبستگی‌ها
    report.append("\n🔍 1. همبستگی عملکرد گذشته با آینده:")
    report.append("-" * 50)
    
    best_correlation = 0
    best_period = None
    
    for period, stats in correlations.items():
        significance = "معنادار ✅" if stats['significant'] else "غیرمعنادار ❌"
        report.append(f"   {period.upper()}:")
        report.append(f"     • Pearson: {stats['pearson_correlation']:.3f} ({significance})")
        report.append(f"     • Spearman: {stats['spearman_correlation']:.3f}")
        report.append(f"     • R²: {stats['r_squared']:.3f}")
        report.append(f"     • نمونه: {stats['sample_size']} مورد")
        
        if abs(stats['pearson_correlation']) > abs(best_correlation):
            best_correlation = stats['pearson_correlation']
            best_period = period
    
    # 2. دقت پیش‌بینی
    report.append(f"\n🎯 2. دقت پیش‌بینی رتبه‌بندی:")
    report.append("-" * 50)
    
    best_accuracy = 0
    best_accuracy_period = None
    
    for period, stats in accuracy.items():
        report.append(f"   {period.upper()}:")
        report.append(f"     • دقت رتبه‌بندی: {stats['ranking_accuracy']:.1%}")
        report.append(f"     • تطبیق استراتژی برتر: {stats['top_strategy_match_rate']:.1%}")
        
        if stats['ranking_accuracy'] > best_accuracy:
            best_accuracy = stats['ranking_accuracy']
            best_accuracy_period = period
    
    # 3. عملکرد استراتژی‌ها
    report.append(f"\n📈 3. عملکرد استراتژی‌ها:")
    report.append("-" * 50)
    
    # مرتب‌سازی بر اساس میانگین عملکرد آینده
    sorted_strategies = sorted(strategy_analysis.items(), 
                              key=lambda x: x[1]['mean_future_sharpe'], 
                              reverse=True)
    
    for strategy, stats in sorted_strategies:
        report.append(f"   {strategy}:")
        report.append(f"     • میانگین Sharpe آینده: {stats['mean_future_sharpe']:.2f}")
        report.append(f"     • نرخ عملکرد مثبت: {stats['positive_future_rate']:.1%}")
        report.append(f"     • نرخ Sharpe > 1: {stats['sharpe_above_1_rate']:.1%}")
        
        # بهترین همبستگی برای این استراتژی
        best_corr_for_strategy = -1
        best_period_for_strategy = None
        for period in ['1y', '3y', '5y']:
            corr = stats.get(f'correlation_{period}', np.nan)
            if not np.isnan(corr) and abs(corr) > abs(best_corr_for_strategy):
                best_corr_for_strategy = corr
                best_period_for_strategy = period
        
        if best_period_for_strategy:
            report.append(f"     • بهترین همبستگی: {best_period_for_strategy} ({best_corr_for_strategy:.3f})")
    
    # 4. تأثیر رژیم بازار
    report.append(f"\n🌊 4. تأثیر رژیم بازار:")
    report.append("-" * 50)
    
    for regime, stats in regime_analysis.items():
        report.append(f"   {regime} Market:")
        report.append(f"     • تعداد: {stats['count']} مورد ({stats['percentage']:.1f}%)")
        report.append(f"     • میانگین عملکرد: {stats['mean_future_sharpe']:.2f}")
        
        # بهترین همبستگی در این رژیم
        best_regime_corr = -1
        best_regime_period = None
        for period in ['1y', '3y', '5y']:
            corr = stats.get(f'correlation_{period}', np.nan)
            if not np.isnan(corr) and abs(corr) > abs(best_regime_corr):
                best_regime_corr = corr
                best_regime_period = period
        
        if best_regime_period:
            report.append(f"     • بهترین پیش‌بینی: {best_regime_period} ({best_regime_corr:.3f})")
    
    # 5. نتیجه‌گیری و توصیه‌ها
    report.append(f"\n🏆 5. نتیجه‌گیری و توصیه‌ها:")
    report.append("-" * 50)
    
    # تعیین کیفیت پیش‌بینی
    if abs(best_correlation) >= 0.5:
        quality = "عالی ✅"
        recommendation = "قابل اعتماد"
    elif abs(best_correlation) >= 0.3:
        quality = "متوسط ⚠️"
        recommendation = "با احتیاط قابل استفاده"
    else:
        quality = "ضعیف ❌"
        recommendation = "غیرقابل اعتماد"
    
    report.append(f"   • بهترین افق زمانی: {best_period.upper()} (همبستگی: {best_correlation:.3f})")
    report.append(f"   • کیفیت پیش‌بینی: {quality}")
    report.append(f"   • توصیه: {recommendation}")
    
    if abs(best_correlation) < 0.3:
        report.append(f"\n   ⚠️ هشدار: قدرت پیش‌بینی ضعیف!")
        report.append(f"   💡 پیشنهادات بهبود:")
        report.append(f"     - استفاده از فاکتورهای اضافی")
        report.append(f"     - تحلیل رژیم بازار")
        report.append(f"     - ترکیب چندین استراتژی")
        report.append(f"     - کاهش انتظارات عملکرد")
    
    # بهترین استراتژی
    best_strategy = sorted_strategies[0][0] if sorted_strategies else "نامشخص"
    report.append(f"\n   🎯 بهترین استراتژی: {best_strategy}")
    
    return report

def save_results(correlations, accuracy, strategy_analysis, regime_analysis, report):
    """
    ذخیره نتایج تحلیل
    """
    logger.info("💾 ذخیره نتایج...")
    
    # ایجاد پوشه نتایج
    results_dir = Path("results/prediction_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ذخیره گزارش اصلی
    with open(results_dir / "horizon_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # ذخیره داده‌های خام
    import json
    
    # همبستگی‌ها
    with open(results_dir / "correlations.json", 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)
    
    # دقت پیش‌بینی
    with open(results_dir / "accuracy.json", 'w', encoding='utf-8') as f:
        json.dump(accuracy, f, indent=2, ensure_ascii=False)
    
    # تحلیل استراتژی‌ها
    with open(results_dir / "strategy_analysis.json", 'w', encoding='utf-8') as f:
        # تبدیل NaN به None برای JSON
        clean_strategy_analysis = {}
        for k, v in strategy_analysis.items():
            clean_v = {}
            for k2, v2 in v.items():
                clean_v[k2] = None if (isinstance(v2, float) and np.isnan(v2)) else v2
            clean_strategy_analysis[k] = clean_v
        json.dump(clean_strategy_analysis, f, indent=2, ensure_ascii=False)
    
    # تحلیل رژیم بازار
    with open(results_dir / "regime_analysis.json", 'w', encoding='utf-8') as f:
        clean_regime_analysis = {}
        for k, v in regime_analysis.items():
            clean_v = {}
            for k2, v2 in v.items():
                clean_v[k2] = None if (isinstance(v2, float) and np.isnan(v2)) else v2
            clean_regime_analysis[k] = clean_v
        json.dump(clean_regime_analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ نتایج در {results_dir} ذخیره شد")

def main():
    """
    تابع اصلی تحلیل افق‌های زمانی
    """
    logger.info("🚀 شروع تحلیل قدرت پیش‌بینی افق‌های زمانی")
    logger.info("=" * 70)
    
    try:
        # بارگذاری داده‌ها
        df = load_walkforward_data()
        if df is None:
            return
        
        # تحلیل همبستگی‌ها
        correlations = calculate_prediction_correlations(df)
        
        # تحلیل دقت پیش‌بینی
        accuracy = analyze_prediction_accuracy(df)
        
        # تحلیل استراتژی‌ها
        strategy_analysis = analyze_by_strategy(df)
        
        # تحلیل رژیم بازار
        regime_analysis = analyze_market_conditions(df)
        
        # تولید گزارش
        report = generate_horizon_report(correlations, accuracy, strategy_analysis, regime_analysis)
        
        # نمایش گزارش
        for line in report:
            logger.info(line)
        
        # ذخیره نتایج
        save_results(correlations, accuracy, strategy_analysis, regime_analysis, report)
        
        logger.info(f"\n🎉 تحلیل افق‌های زمانی با موفقیت کامل شد!")
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()