#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست نظریه: آیا انتخاب بر اساس عملکرد 5 ساله بهتر است؟
فرضیه: استراتژی‌هایی که در 5 سال خوب بوده‌اند، در دوره‌های کوتاه‌تر هم خوب هستند
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_multi_period_market_data():
    """
    تولید داده‌های بازار برای دوره‌های مختلف
    """
    logger.info("📊 تولید داده‌های بازار برای تست نظریه...")
    
    np.random.seed(42)
    
    # تاریخ‌ها (7 سال برای داشتن overlap کافی)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # فقط روزهای کاری
    
    # شبیه‌سازی 150 سهم
    symbols = [f"STOCK_{i:03d}" for i in range(1, 151)]
    
    price_data = {}
    
    for symbol in symbols:
        # شبیه‌سازی با رژیم‌های مختلف و persistence
        returns = []
        trend = np.random.normal(0, 0.0002)  # trend طولانی‌مدت برای هر سهم
        volatility_base = np.random.uniform(0.015, 0.025)  # نوسان پایه
        
        for i in range(len(dates)):
            # اضافه کردن persistence و mean reversion
            if len(returns) > 0:
                momentum = 0.05 * returns[-1] if len(returns) > 0 else 0
                mean_reversion = -0.02 * np.mean(returns[-20:]) if len(returns) > 20 else 0
            else:
                momentum = mean_reversion = 0
            
            # تغییرات دوره‌ای در volatility
            cycle_effect = 0.005 * np.sin(2 * np.pi * i / 252)  # چرخه سالانه
            current_vol = volatility_base + cycle_effect
            
            daily_return = np.random.normal(trend + momentum + mean_reversion, current_vol)
            returns.append(daily_return)
        
        # تبدیل به قیمت‌ها
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[symbol] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    logger.info(f"✅ داده‌های {len(symbols)} سهم برای {len(dates)} روز تولید شد")
    
    return df

def calculate_factors_for_period(price_data, end_date, lookback_days=252):
    """
    محاسبه فاکتورها برای یک دوره مشخص
    """
    start_date = end_date - timedelta(days=lookback_days + 100)  # اضافه برای محاسبه momentum
    
    period_data = price_data.loc[start_date:end_date]
    if len(period_data) < lookback_days:
        return {}
    
    returns = period_data.pct_change().fillna(0)
    factors = {}
    
    for symbol in period_data.columns:
        try:
            # Value Factor (شبیه‌سازی)
            recent_price = period_data[symbol].iloc[-1]
            avg_price = period_data[symbol].iloc[-lookback_days:].mean()
            value_factor = avg_price / recent_price if recent_price > 0 else 0.5
            
            # Momentum Factor
            momentum_6m = (period_data[symbol].iloc[-1] / period_data[symbol].iloc[-126]) - 1 if len(period_data) >= 126 else 0
            momentum_12m = (period_data[symbol].iloc[-1] / period_data[symbol].iloc[-252]) - 1 if len(period_data) >= 252 else 0
            
            # Low Volatility Factor
            volatility = returns[symbol].iloc[-lookback_days:].std() * np.sqrt(252)
            low_vol_factor = 1 / (1 + volatility) if volatility > 0 else 0.5
            
            factors[symbol] = {
                'value': value_factor,
                'momentum_6m': momentum_6m,
                'momentum_12m': momentum_12m,
                'low_volatility': low_vol_factor
            }
            
        except Exception:
            factors[symbol] = {
                'value': 0.5,
                'momentum_6m': 0,
                'momentum_12m': 0,
                'low_volatility': 0.5
            }
    
    return factors

def backtest_strategy_for_period(price_data, factors, strategy_config, start_date, end_date):
    """
    بک‌تست استراتژی برای یک دوره مشخص
    """
    period_data = price_data.loc[start_date:end_date]
    if len(period_data) < 50:
        return None
    
    momentum_period = strategy_config['Momentum Period']
    value_weight = strategy_config['Value Weight']
    momentum_weight = strategy_config['Momentum Weight']
    low_vol_weight = strategy_config['Low Volatility Weight']
    top_n = int(strategy_config['Top N'])
    max_weight = strategy_config['Max Weight']
    
    momentum_key = 'momentum_12m' if momentum_period == '12M' else 'momentum_6m'
    
    # انتخاب سهام بر اساس فاکتورها
    scores = {}
    for symbol, factor_values in factors.items():
        if symbol in period_data.columns:
            combined_score = (
                value_weight * factor_values['value'] +
                momentum_weight * factor_values[momentum_key] +
                low_vol_weight * factor_values['low_volatility']
            )
            scores[symbol] = combined_score
    
    if not scores:
        return None
    
    # انتخاب top N سهم
    top_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
    
    if not top_symbols:
        return None
    
    # محاسبه عملکرد پرتفوی (equal weight)
    portfolio_returns = []
    for i in range(1, len(period_data)):
        daily_return = 0
        for symbol in top_symbols:
            if symbol in period_data.columns:
                try:
                    stock_return = (period_data[symbol].iloc[i] / period_data[symbol].iloc[i-1]) - 1
                    daily_return += stock_return / len(top_symbols)
                except:
                    continue
        portfolio_returns.append(daily_return)
    
    if not portfolio_returns:
        return None
    
    # محاسبه معیارهای عملکرد
    returns_array = np.array(portfolio_returns)
    total_return = np.prod(1 + returns_array) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns_array)) - 1
    volatility = returns_array.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'num_days': len(returns_array)
    }

def test_selection_theory():
    """
    تست نظریه انتخاب بر اساس عملکرد 5 ساله
    """
    logger.info("🧪 شروع تست نظریه انتخاب بر اساس عملکرد 5 ساله")
    logger.info("=" * 70)
    
    # تولید داده‌های بازار
    price_data = generate_multi_period_market_data()
    
    # تعریف استراتژی‌های مختلف برای تست
    strategies = []
    
    # ایجاد 50 استراتژی مختلف
    np.random.seed(123)
    for i in range(50):
        # وزن‌های تصادفی که مجموعشان 1 باشد
        weights = np.random.dirichlet([1, 1, 1])
        
        strategy = {
            'name': f'Strategy_{i+1:02d}',
            'Momentum Period': np.random.choice(['6M', '12M']),
            'Value Weight': round(weights[0], 2),
            'Momentum Weight': round(weights[1], 2),
            'Low Volatility Weight': round(weights[2], 2),
            'Top N': np.random.choice([10, 15, 20, 25, 30]),
            'Max Weight': np.random.choice([0.15, 0.2, 0.25, 0.3])
        }
        strategies.append(strategy)
    
    # تعریف دوره‌های تحلیل
    end_date = price_data.index[-1]
    
    # نقاط زمانی برای محاسبه عملکرد
    date_5y_ago = end_date - timedelta(days=5*365)
    date_3y_ago = end_date - timedelta(days=3*365)
    date_1y_ago = end_date - timedelta(days=1*365)
    
    # محاسبه فاکتورها برای هر دوره
    factors_5y = calculate_factors_for_period(price_data, date_5y_ago)
    factors_3y = calculate_factors_for_period(price_data, date_3y_ago)
    factors_1y = calculate_factors_for_period(price_data, date_1y_ago)
    
    logger.info(f"📅 دوره‌های تحلیل:")
    logger.info(f"   - 5 ساله: تا {date_5y_ago.strftime('%Y-%m-%d')}")
    logger.info(f"   - 3 ساله: تا {date_3y_ago.strftime('%Y-%m-%d')}")
    logger.info(f"   - 1 ساله: تا {date_1y_ago.strftime('%Y-%m-%d')}")
    
    # تست هر استراتژی در دوره‌های مختلف
    results = []
    
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"🔄 تست استراتژی {i}/50: {strategy['name']}")
        
        try:
            # عملکرد در دوره 5 ساله (از ابتدای داده‌ها تا 5 سال پیش)
            perf_5y = backtest_strategy_for_period(
                price_data, factors_5y, strategy,
                price_data.index[0], date_5y_ago
            )
            
            # عملکرد در دوره 3 ساله (از 5 سال پیش تا 3 سال پیش)
            perf_3y = backtest_strategy_for_period(
                price_data, factors_3y, strategy,
                date_5y_ago, date_3y_ago
            )
            
            # عملکرد در دوره 1 ساله (از 3 سال پیش تا 1 سال پیش)
            perf_1y = backtest_strategy_for_period(
                price_data, factors_1y, strategy,
                date_3y_ago, date_1y_ago
            )
            
            # عملکرد در دوره آینده (1 سال اخیر)
            factors_future = calculate_factors_for_period(price_data, end_date)
            perf_future = backtest_strategy_for_period(
                price_data, factors_future, strategy,
                date_1y_ago, end_date
            )
            
            if all([perf_5y, perf_3y, perf_1y, perf_future]):
                result = {
                    'strategy_name': strategy['name'],
                    'momentum_period': strategy['Momentum Period'],
                    'value_weight': strategy['Value Weight'],
                    'momentum_weight': strategy['Momentum Weight'],
                    'low_vol_weight': strategy['Low Volatility Weight'],
                    'top_n': strategy['Top N'],
                    'max_weight': strategy['Max Weight'],
                    
                    'sharpe_5y': perf_5y['sharpe_ratio'],
                    'sharpe_3y': perf_3y['sharpe_ratio'],
                    'sharpe_1y': perf_1y['sharpe_ratio'],
                    'sharpe_future': perf_future['sharpe_ratio'],
                    
                    'return_5y': perf_5y['annualized_return'],
                    'return_3y': perf_3y['annualized_return'],
                    'return_1y': perf_1y['annualized_return'],
                    'return_future': perf_future['annualized_return'],
                    
                    'vol_5y': perf_5y['volatility'],
                    'vol_3y': perf_3y['volatility'],
                    'vol_1y': perf_1y['volatility'],
                    'vol_future': perf_future['volatility']
                }
                results.append(result)
                
                logger.info(f"   ✅ Sharpe: 5Y={perf_5y['sharpe_ratio']:.2f}, "
                           f"3Y={perf_3y['sharpe_ratio']:.2f}, "
                           f"1Y={perf_1y['sharpe_ratio']:.2f}, "
                           f"Future={perf_future['sharpe_ratio']:.2f}")
            else:
                logger.warning(f"   ⚠️ داده‌های ناکافی برای استراتژی {strategy['name']}")
                
        except Exception as e:
            logger.error(f"   ❌ خطا در تست استراتژی {strategy['name']}: {e}")
    
    return pd.DataFrame(results)

def analyze_selection_theory(results_df):
    """
    تحلیل نظریه انتخاب بر اساس دوره‌های مختلف
    """
    logger.info("📊 تحلیل نظریه انتخاب...")
    
    if len(results_df) == 0:
        logger.error("❌ هیچ نتیجه‌ای برای تحلیل موجود نیست")
        return
    
    # محاسبه همبستگی‌ها
    correlations = {
        '5Y_vs_Future': results_df['sharpe_5y'].corr(results_df['sharpe_future']),
        '3Y_vs_Future': results_df['sharpe_3y'].corr(results_df['sharpe_future']),
        '1Y_vs_Future': results_df['sharpe_1y'].corr(results_df['sharpe_future']),
        '5Y_vs_3Y': results_df['sharpe_5y'].corr(results_df['sharpe_3y']),
        '5Y_vs_1Y': results_df['sharpe_5y'].corr(results_df['sharpe_1y']),
        '3Y_vs_1Y': results_df['sharpe_3y'].corr(results_df['sharpe_1y'])
    }
    
    # تست نظریه: آیا استراتژی‌های خوب 5 ساله در دوره‌های کوتاه‌تر هم خوب هستند؟
    
    # انتخاب 10 استراتژی برتر بر اساس عملکرد 5 ساله
    top_10_by_5y = results_df.nlargest(10, 'sharpe_5y')
    
    # انتخاب 10 استراتژی برتر بر اساس عملکرد 3 ساله
    top_10_by_3y = results_df.nlargest(10, 'sharpe_3y')
    
    # انتخاب 10 استراتژی برتر بر اساس عملکرد 1 ساله
    top_10_by_1y = results_df.nlargest(10, 'sharpe_1y')
    
    # محاسبه میانگین عملکرد آینده برای هر گروه
    future_perf_5y_selected = top_10_by_5y['sharpe_future'].mean()
    future_perf_3y_selected = top_10_by_3y['sharpe_future'].mean()
    future_perf_1y_selected = top_10_by_1y['sharpe_future'].mean()
    
    # تحلیل overlap بین گروه‌ها
    overlap_5y_3y = len(set(top_10_by_5y['strategy_name']) & set(top_10_by_3y['strategy_name']))
    overlap_5y_1y = len(set(top_10_by_5y['strategy_name']) & set(top_10_by_1y['strategy_name']))
    overlap_3y_1y = len(set(top_10_by_3y['strategy_name']) & set(top_10_by_1y['strategy_name']))
    
    # ذخیره نتایج
    results_df.to_csv('tests/theory_test_results.csv', index=False)
    
    # ایجاد گزارش
    report = []
    report.append("=== تست نظریه انتخاب بر اساس عملکرد 5 ساله ===\n")
    
    report.append("📊 همبستگی‌های عملکرد:")
    for period, corr in correlations.items():
        report.append(f"   - {period}: {corr:.3f}")
    
    report.append(f"\n🎯 عملکرد آینده بر اساس روش انتخاب:")
    report.append(f"   - انتخاب بر اساس 5Y: {future_perf_5y_selected:.3f}")
    report.append(f"   - انتخاب بر اساس 3Y: {future_perf_3y_selected:.3f}")
    report.append(f"   - انتخاب بر اساس 1Y: {future_perf_1y_selected:.3f}")
    
    report.append(f"\n🔄 همپوشانی بین انتخاب‌ها (از 10 استراتژی):")
    report.append(f"   - 5Y و 3Y: {overlap_5y_3y} استراتژی مشترک")
    report.append(f"   - 5Y و 1Y: {overlap_5y_1y} استراتژی مشترک")
    report.append(f"   - 3Y و 1Y: {overlap_3y_1y} استراتژی مشترک")
    
    # نتیجه‌گیری
    best_method = max([
        ('5Y', future_perf_5y_selected),
        ('3Y', future_perf_3y_selected),
        ('1Y', future_perf_1y_selected)
    ], key=lambda x: x[1])
    
    report.append(f"\n🏆 نتیجه‌گیری:")
    report.append(f"   - بهترین روش انتخاب: {best_method[0]} (عملکرد آینده: {best_method[1]:.3f})")
    
    if correlations['5Y_vs_Future'] > max(correlations['3Y_vs_Future'], correlations['1Y_vs_Future']):
        report.append(f"   - نظریه تأیید شد: انتخاب بر اساس 5Y بهترین همبستگی با آینده دارد")
    else:
        report.append(f"   - نظریه رد شد: انتخاب بر اساس 5Y بهترین روش نیست")
    
    if overlap_5y_3y >= 7:  # 70% همپوشانی
        report.append(f"   - فرضیه پایداری تأیید شد: استراتژی‌های خوب 5Y در 3Y هم خوب هستند")
    
    # ذخیره گزارش
    with open('tests/theory_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # نمایش گزارش
    logger.info("\n" + "\n".join(report))
    
    return correlations, {
        'future_perf_5y': future_perf_5y_selected,
        'future_perf_3y': future_perf_3y_selected,
        'future_perf_1y': future_perf_1y_selected,
        'overlap_5y_3y': overlap_5y_3y,
        'overlap_5y_1y': overlap_5y_1y,
        'overlap_3y_1y': overlap_3y_1y
    }

def main():
    """
    تابع اصلی
    """
    try:
        # اجرای تست
        results_df = test_selection_theory()
        
        if len(results_df) > 0:
            # تحلیل نتایج
            correlations, metrics = analyze_selection_theory(results_df)
            
            logger.info(f"\n🎉 تست نظریه با موفقیت کامل شد!")
            logger.info(f"📁 فایل‌های خروجی:")
            logger.info(f"   - tests/theory_test_results.csv")
            logger.info(f"   - tests/theory_analysis_report.txt")
        else:
            logger.error("❌ هیچ نتیجه‌ای تولید نشد")
        
    except Exception as e:
        logger.error(f"❌ خطا در تست نظریه: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()