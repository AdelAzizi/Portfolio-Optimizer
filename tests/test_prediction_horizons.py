#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست علمی: کدام افق زمانی بهتر آینده را پیش‌بینی می‌کند؟
آزمایش Walk-Forward برای مقایسه قدرت پیش‌بینی 1، 3، و 5 ساله
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_market_data(years=10, n_stocks=100):
    """
    تولید داده‌های بازار واقع‌گرایانه برای 10 سال
    """
    logger.info(f"📊 تولید داده‌های بازار برای {years} سال...")
    
    np.random.seed(42)
    
    # تاریخ‌ها
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # فقط روزهای کاری
    
    # شبیه‌سازی سهام با رژیم‌های مختلف بازار
    symbols = [f"STOCK_{i:03d}" for i in range(1, n_stocks + 1)]
    
    price_data = {}
    
    for symbol in symbols:
        # شبیه‌سازی رژیم‌های مختلف بازار
        returns = []
        current_regime = 'normal'
        regime_duration = 0
        
        for i, date in enumerate(dates):
            # تغییر رژیم بازار
            if regime_duration <= 0:
                regime_duration = np.random.randint(60, 300)  # 2-12 ماه
                current_regime = np.random.choice(['bull', 'bear', 'normal'], p=[0.3, 0.2, 0.5])
            
            # پارامترهای بازدهی بر اساس رژیم
            if current_regime == 'bull':
                mean_return = 0.001
                volatility = 0.015
            elif current_regime == 'bear':
                mean_return = -0.0005
                volatility = 0.025
            else:  # normal
                mean_return = 0.0003
                volatility = 0.018
            
            # اضافه کردن فاکتور momentum و mean reversion
            if len(returns) > 20:
                momentum_effect = 0.1 * np.mean(returns[-20:])  # momentum
                mean_reversion = -0.05 * returns[-1] if abs(returns[-1]) > 0.03 else 0  # mean reversion
                mean_return += momentum_effect + mean_reversion
            
            daily_return = np.random.normal(mean_return, volatility)
            returns.append(daily_return)
            regime_duration -= 1
        
        # تبدیل به قیمت‌ها
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[symbol] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    logger.info(f"✅ داده‌های {len(symbols)} سهم برای {len(dates)} روز تولید شد")
    
    return df

def calculate_strategy_performance(price_data, start_date, end_date, strategy_params):
    """
    محاسبه عملکرد یک استراتژی در بازه زمانی مشخص
    """
    period_data = price_data.loc[start_date:end_date]
    if len(period_data) < 50:  # حداقل 50 روز داده
        return None
    
    returns = period_data.pct_change().fillna(0)
    
    # شبیه‌سازی استراتژی momentum ساده
    momentum_period = strategy_params.get('momentum_days', 60)
    rebalance_freq = strategy_params.get('rebalance_freq', 21)
    top_n = strategy_params.get('top_n', 20)
    
    portfolio_value = 100.0
    portfolio_returns = []
    
    for i in range(momentum_period, len(period_data), rebalance_freq):
        if i + rebalance_freq >= len(period_data):
            break
            
        # محاسبه momentum برای انتخاب سهام
        momentum_scores = {}
        for symbol in period_data.columns:
            momentum_return = (period_data[symbol].iloc[i] / period_data[symbol].iloc[i-momentum_period]) - 1
            momentum_scores[symbol] = momentum_return
        
        # انتخاب top N سهم
        top_stocks = sorted(momentum_scores.keys(), key=lambda x: momentum_scores[x], reverse=True)[:top_n]
        
        # محاسبه عملکرد پرتفوی در دوره rebalance
        period_return = 0
        for j in range(rebalance_freq):
            if i + j + 1 < len(period_data):
                daily_return = 0
                for stock in top_stocks:
                    stock_return = (period_data[stock].iloc[i+j+1] / period_data[stock].iloc[i+j]) - 1
                    daily_return += stock_return / len(top_stocks)  # equal weight
                
                portfolio_value *= (1 + daily_return)
                portfolio_returns.append(daily_return)
    
    if not portfolio_returns:
        return None
    
    # محاسبه معیارهای عملکرد
    total_return = (portfolio_value / 100.0) - 1
    returns_array = np.array(portfolio_returns)
    volatility = returns_array.std() * np.sqrt(252)
    mean_return = returns_array.mean() * 252
    sharpe = mean_return / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe
    }

def run_walk_forward_analysis(price_data):
    """
    اجرای آنالیز Walk-Forward برای مقایسه افق‌های زمانی مختلف
    """
    logger.info("🔄 شروع آنالیز Walk-Forward...")
    
    # تعریف استراتژی‌های مختلف برای تست
    strategies = [
        {'name': 'Momentum_60_20', 'momentum_days': 60, 'rebalance_freq': 21, 'top_n': 20},
        {'name': 'Momentum_120_15', 'momentum_days': 120, 'rebalance_freq': 21, 'top_n': 15},
        {'name': 'Momentum_30_25', 'momentum_days': 30, 'rebalance_freq': 21, 'top_n': 25},
        {'name': 'Momentum_90_10', 'momentum_days': 90, 'rebalance_freq': 21, 'top_n': 10},
        {'name': 'Momentum_180_30', 'momentum_days': 180, 'rebalance_freq': 21, 'top_n': 30},
    ]
    
    # تعریف دوره‌های آنالیز
    all_dates = price_data.index
    start_analysis = all_dates[252*2]  # شروع از سال سوم (برای داشتن تاریخچه کافی)
    end_analysis = all_dates[-252]     # پایان یک سال قبل از انتهای داده‌ها
    
    analysis_dates = pd.date_range(start=start_analysis, end=end_analysis, freq='3M')  # هر 3 ماه
    
    results = []
    
    for analysis_date in analysis_dates:
        logger.info(f"📅 تحلیل برای تاریخ: {analysis_date.strftime('%Y-%m-%d')}")
        
        # تعریف دوره‌های ارزیابی
        eval_1y_start = analysis_date - timedelta(days=365)
        eval_3y_start = analysis_date - timedelta(days=3*365)
        eval_5y_start = analysis_date - timedelta(days=5*365)
        
        # دوره آینده برای تست (6 ماه آینده)
        future_start = analysis_date
        future_end = analysis_date + timedelta(days=180)
        
        if future_end > all_dates[-1] or eval_5y_start < all_dates[0]:
            continue
        
        # ارزیابی هر استراتژی در دوره‌های مختلف
        for strategy in strategies:
            # عملکرد در دوره‌های گذشته
            perf_1y = calculate_strategy_performance(price_data, eval_1y_start, analysis_date, strategy)
            perf_3y = calculate_strategy_performance(price_data, eval_3y_start, analysis_date, strategy)
            perf_5y = calculate_strategy_performance(price_data, eval_5y_start, analysis_date, strategy)
            
            # عملکرد در دوره آینده (واقعی)
            future_perf = calculate_strategy_performance(price_data, future_start, future_end, strategy)
            
            if all([perf_1y, perf_3y, perf_5y, future_perf]):
                results.append({
                    'analysis_date': analysis_date,
                    'strategy': strategy['name'],
                    'sharpe_1y': perf_1y['sharpe_ratio'],
                    'sharpe_3y': perf_3y['sharpe_ratio'],
                    'sharpe_5y': perf_5y['sharpe_ratio'],
                    'future_sharpe': future_perf['sharpe_ratio'],
                    'return_1y': perf_1y['annualized_return'],
                    'return_3y': perf_3y['annualized_return'],
                    'return_5y': perf_5y['annualized_return'],
                    'future_return': future_perf['annualized_return']
                })
    
    return pd.DataFrame(results)

def analyze_prediction_power(results_df):
    """
    تحلیل قدرت پیش‌بینی هر افق زمانی
    """
    logger.info("📊 تحلیل قدرت پیش‌بینی...")
    
    # محاسبه همبستگی بین عملکرد گذشته و آینده
    correlations = {
        '1_year': results_df['sharpe_1y'].corr(results_df['future_sharpe']),
        '3_year': results_df['sharpe_3y'].corr(results_df['future_sharpe']),
        '5_year': results_df['sharpe_5y'].corr(results_df['future_sharpe'])
    }
    
    # تحلیل دقت رتبه‌بندی
    ranking_accuracy = {}
    
    for period in ['1y', '3y', '5y']:
        correct_predictions = 0
        total_predictions = 0
        
        # گروه‌بندی بر اساس تاریخ تحلیل
        for analysis_date in results_df['analysis_date'].unique():
            date_data = results_df[results_df['analysis_date'] == analysis_date]
            
            if len(date_data) < 3:
                continue
            
            # رتبه‌بندی بر اساس عملکرد گذشته
            past_ranking = date_data.sort_values(f'sharpe_{period}', ascending=False)['strategy'].tolist()
            
            # رتبه‌بندی بر اساس عملکرد آینده
            future_ranking = date_data.sort_values('future_sharpe', ascending=False)['strategy'].tolist()
            
            # محاسبه دقت (آیا بهترین استراتژی گذشته در آینده هم بهترین است؟)
            if past_ranking[0] == future_ranking[0]:
                correct_predictions += 1
            
            total_predictions += 1
        
        ranking_accuracy[period] = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return correlations, ranking_accuracy

def save_analysis_results(results_df, correlations, ranking_accuracy):
    """
    ذخیره نتایج تحلیل
    """
    logger.info("💾 ذخیره نتایج تحلیل...")
    
    # ذخیره داده‌های خام
    results_df.to_csv('walk_forward_analysis_results.csv', index=False)
    
    # ایجاد گزارش خلاصه
    summary_report = []
    summary_report.append("=== تحلیل قدرت پیش‌بینی افق‌های زمانی مختلف ===\n")
    
    summary_report.append("📊 همبستگی بین عملکرد گذشته و آینده:")
    for period, corr in correlations.items():
        summary_report.append(f"   - {period}: {corr:.3f}")
    
    summary_report.append(f"\n🎯 دقت پیش‌بینی بهترین استراتژی:")
    for period, accuracy in ranking_accuracy.items():
        summary_report.append(f"   - {period}: {accuracy:.1%}")
    
    # تحلیل آماری
    summary_report.append(f"\n📈 آمار توصیفی:")
    summary_report.append(f"   - تعداد کل مشاهدات: {len(results_df)}")
    summary_report.append(f"   - تعداد دوره‌های تحلیل: {results_df['analysis_date'].nunique()}")
    summary_report.append(f"   - تعداد استراتژی‌های تست شده: {results_df['strategy'].nunique()}")
    
    # میانگین عملکرد
    summary_report.append(f"\n📊 میانگین Sharpe Ratio:")
    summary_report.append(f"   - 1 ساله: {results_df['sharpe_1y'].mean():.2f}")
    summary_report.append(f"   - 3 ساله: {results_df['sharpe_3y'].mean():.2f}")
    summary_report.append(f"   - 5 ساله: {results_df['sharpe_5y'].mean():.2f}")
    summary_report.append(f"   - آینده (6 ماه): {results_df['future_sharpe'].mean():.2f}")
    
    # نتیجه‌گیری
    best_correlation = max(correlations, key=correlations.get)
    best_accuracy = max(ranking_accuracy, key=ranking_accuracy.get)
    
    summary_report.append(f"\n🏆 نتیجه‌گیری:")
    summary_report.append(f"   - بهترین همبستگی: {best_correlation} ({correlations[best_correlation]:.3f})")
    summary_report.append(f"   - بهترین دقت پیش‌بینی: {best_accuracy} ({ranking_accuracy[best_accuracy]:.1%})")
    
    if correlations[best_correlation] > 0.3:
        summary_report.append(f"   - توصیه: استفاده از افق {best_correlation} برای انتخاب استراتژی")
    else:
        summary_report.append(f"   - هشدار: همبستگی‌ها پایین هستند - عملکرد گذشته پیش‌بین ضعیفی از آینده است")
    
    # ذخیره گزارش
    with open('prediction_horizon_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_report))
    
    logger.info("✅ گزارش ذخیره شد: prediction_horizon_analysis_report.txt")
    
    return summary_report

def main():
    """
    تابع اصلی
    """
    logger.info("🚀 شروع تحلیل علمی قدرت پیش‌بینی افق‌های زمانی")
    logger.info("=" * 70)
    
    try:
        # تولید داده‌های بازار
        price_data = generate_realistic_market_data(years=10, n_stocks=50)
        
        # اجرای آنالیز Walk-Forward
        results_df = run_walk_forward_analysis(price_data)
        
        if len(results_df) == 0:
            logger.error("❌ هیچ نتیجه‌ای تولید نشد")
            return
        
        # تحلیل قدرت پیش‌بینی
        correlations, ranking_accuracy = analyze_prediction_power(results_df)
        
        # ذخیره و نمایش نتایج
        summary_report = save_analysis_results(results_df, correlations, ranking_accuracy)
        
        # نمایش نتایج
        logger.info("\n" + "\n".join(summary_report))
        
        logger.info(f"\n🎉 تحلیل با موفقیت کامل شد!")
        logger.info(f"📁 فایل‌های خروجی:")
        logger.info(f"   - walk_forward_analysis_results.csv")
        logger.info(f"   - prediction_horizon_analysis_report.txt")
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()