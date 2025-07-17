#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بک‌تست 5 ساله برای 15 کاندیدای انتخاب شده از تست 50 استراتژی
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_selected_candidates():
    """
    بارگذاری کاندیداهای انتخاب شده
    """
    logger.info("📊 بارگذاری کاندیداهای انتخاب شده...")
    
    with open('selected_candidates_from_50_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_candidates = []
    for risk_profile, profile_data in data.items():
        for strategy in profile_data['strategies']:
            strategy['Risk_Profile'] = risk_profile.title()
            all_candidates.append(strategy)
    
    df = pd.DataFrame(all_candidates)
    logger.info(f"✅ {len(df)} کاندیدا بارگذاری شد")
    
    return df

def generate_5year_market_data():
    """
    تولید داده‌های بازار برای 5 سال
    """
    logger.info("📊 تولید داده‌های بازار 5 ساله...")
    
    np.random.seed(42)  # برای تکرارپذیری
    
    # تاریخ‌ها (5 سال)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # فقط روزهای کاری
    
    # شبیه‌سازی 200 سهم
    symbols = [f"STOCK_{i:03d}" for i in range(1, 201)]
    
    # ایجاد داده‌های قیمتی با volatility clustering
    price_data = {}
    for symbol in symbols:
        # شبیه‌سازی بازدهی‌ها با GARCH-like behavior
        returns = []
        volatility = 0.02  # نوسان اولیه
        
        for i in range(len(dates)):
            # تغییر نوسان در طول زمان
            volatility = 0.95 * volatility + 0.05 * 0.025 + 0.1 * abs(np.random.normal(0, 0.01))
            volatility = max(0.01, min(0.05, volatility))  # محدود کردن نوسان
            
            # تولید بازدهی روزانه
            daily_return = np.random.normal(0.0005, volatility)
            returns.append(daily_return)
        
        # تبدیل بازدهی‌ها به قیمت‌ها
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[symbol] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    
    logger.info(f"✅ داده‌های {len(symbols)} سهم برای {len(dates)} روز تولید شد")
    return df

def calculate_5year_factors(price_data, lookback_6m=126, lookback_12m=252):
    """
    محاسبه فاکتورهای مالی برای 5 سال
    """
    logger.info("🧮 محاسبه فاکتورهای مالی برای 5 سال...")
    
    returns = price_data.pct_change().fillna(0)
    factors = {}
    
    # محاسبه فاکتورها برای هر تاریخ
    for date in price_data.index[lookback_12m:]:
        date_factors = {}
        
        hist_prices = price_data.loc[:date]
        hist_returns = returns.loc[:date]
        
        if len(hist_prices) < lookback_12m:
            continue
            
        for symbol in price_data.columns:
            try:
                # Value Factor (شبیه‌سازی P/E معکوس)
                recent_price = hist_prices[symbol].iloc[-1]
                avg_price_12m = hist_prices[symbol].iloc[-lookback_12m:].mean()
                value_factor = avg_price_12m / recent_price if recent_price > 0 else 0.5
                
                # Momentum 6M
                if len(hist_returns) >= lookback_6m:
                    momentum_6m = hist_returns[symbol].iloc[-lookback_6m:-21].sum()
                else:
                    momentum_6m = 0
                
                # Momentum 12M
                if len(hist_returns) >= lookback_12m:
                    momentum_12m = hist_returns[symbol].iloc[-lookback_12m:-21].sum()
                else:
                    momentum_12m = 0
                
                # Low Volatility Factor
                if len(hist_returns) >= lookback_6m:
                    volatility = hist_returns[symbol].iloc[-lookback_6m:].std() * np.sqrt(252)
                    low_vol_factor = 1 / (1 + volatility) if volatility > 0 else 0.5
                else:
                    low_vol_factor = 0.5
                
                date_factors[symbol] = {
                    'value': value_factor,
                    'momentum_6m': momentum_6m,
                    'momentum_12m': momentum_12m,
                    'low_volatility': low_vol_factor
                }
                
            except Exception:
                date_factors[symbol] = {
                    'value': 0.5,
                    'momentum_6m': 0,
                    'momentum_12m': 0,
                    'low_volatility': 0.5
                }
        
        factors[date] = date_factors
    
    logger.info(f"✅ فاکتورها برای {len(factors)} تاریخ محاسبه شد")
    return factors

def run_5year_backtest_for_strategy(price_data, factors, strategy_config, rebalance_freq=21):
    """
    اجرای بک‌تست 5 ساله برای یک استراتژی
    """
    momentum_period = strategy_config['Momentum_Period']
    value_weight = strategy_config['Value_Weight']
    momentum_weight = strategy_config['Momentum_Weight']
    low_vol_weight = strategy_config['Low_Volatility_Weight']
    top_n = int(strategy_config['Top_N'])
    max_weight = strategy_config['Max_Weight']
    
    momentum_key = 'momentum_12m' if momentum_period == '12M' else 'momentum_6m'
    
    portfolio_values = []
    factor_dates = sorted(factors.keys())
    rebalance_dates = factor_dates[::rebalance_freq]  # هر 21 روز
    
    current_weights = {}
    portfolio_value = 100.0
    
    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date not in factors:
            continue
            
        # محاسبه امتیاز ترکیبی
        date_factors = factors[rebalance_date]
        scores = {}
        
        for symbol, factor_values in date_factors.items():
            combined_score = (
                value_weight * factor_values['value'] +
                momentum_weight * factor_values[momentum_key] +
                low_vol_weight * factor_values['low_volatility']
            )
            scores[symbol] = combined_score
        
        # انتخاب top N سهم
        top_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
        
        # محاسبه وزن‌ها
        if top_symbols:
            equal_weight = 1.0 / len(top_symbols)
            target_weight = min(equal_weight, max_weight)
            
            new_weights = {}
            total_weight = 0
            for symbol in top_symbols:
                new_weights[symbol] = target_weight
                total_weight += target_weight
            
            if total_weight < 1.0 and top_symbols:
                scale_factor = 1.0 / total_weight
                for symbol in new_weights:
                    new_weights[symbol] *= scale_factor
            
            current_weights = new_weights
        
        # محاسبه عملکرد تا rebalance بعدی
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            period_dates = [d for d in factor_dates if rebalance_date <= d < next_rebalance]
        else:
            period_dates = [d for d in factor_dates if d >= rebalance_date]
        
        for date in period_dates:
            if date in price_data.index and date != rebalance_date:
                portfolio_return = 0
                for symbol, weight in current_weights.items():
                    if symbol in price_data.columns:
                        try:
                            prev_date_idx = price_data.index.get_loc(date) - 1
                            if prev_date_idx >= 0:
                                prev_date = price_data.index[prev_date_idx]
                                stock_return = (price_data.loc[date, symbol] / price_data.loc[prev_date, symbol]) - 1
                                portfolio_return += weight * stock_return
                        except:
                            continue
                
                portfolio_value *= (1 + portfolio_return)
            
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value
            })
    
    return portfolio_values

def calculate_5year_performance_metrics(portfolio_values):
    """
    محاسبه معیارهای عملکرد برای 5 سال
    """
    if not portfolio_values:
        return {}
    
    df = pd.DataFrame(portfolio_values)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # محاسبه بازدهی‌ها
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    
    # معیارهای عملکرد
    total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
    
    # تعداد سال‌های واقعی
    years = len(df) / 252  # تقریباً 252 روز کاری در سال
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # نوسان سالانه
    annualized_volatility = df['returns'].std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Max Drawdown
    rolling_max = df['portfolio_value'].expanding().max()
    drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Total_Return_5Y': total_return,
        'Annualized_Return_5Y': annualized_return,
        'Annualized_Volatility_5Y': annualized_volatility,
        'Sharpe_Ratio_5Y': sharpe_ratio,
        'Max_Drawdown_5Y': max_drawdown,
        'Years': years
    }

def run_comprehensive_backtest():
    """
    اجرای بک‌تست جامع برای تمام کاندیداها
    """
    logger.info("🚀 شروع بک‌تست جامع 1 ساله و 5 ساله")
    logger.info("=" * 70)
    
    # بارگذاری کاندیداها
    candidates_df = load_selected_candidates()
    
    # تولید داده‌های بازار 5 ساله
    price_data_5y = generate_5year_market_data()
    factors_5y = calculate_5year_factors(price_data_5y)
    
    # نتایج نهایی
    results = []
    
    for idx, (_, candidate) in enumerate(candidates_df.iterrows(), 1):
        logger.info(f"\n🔄 بک‌تست کاندیدا {idx}/15")
        logger.info(f"   Risk Profile: {candidate['Risk_Profile']}")
        logger.info(f"   Original Strategy Index: {candidate['Strategy_Index']}")
        logger.info(f"   1Y Test Sharpe: {candidate['Test_Sharpe_Ratio']:.2f}")
        
        try:
            # بک‌تست 5 ساله
            portfolio_values_5y = run_5year_backtest_for_strategy(price_data_5y, factors_5y, candidate)
            performance_5y = calculate_5year_performance_metrics(portfolio_values_5y)
            
            # ترکیب نتایج
            result = {
                'Candidate_Index': idx,
                'Risk_Profile': candidate['Risk_Profile'],
                'Original_Strategy_Index': candidate['Strategy_Index'],
                
                # نتایج اصلی (از top_300_strategies.csv)
                'Original_Sharpe': candidate['Original_Sharpe'],
                
                # نتایج بک‌تست 1 ساله
                'Test_1Y_Sharpe_Ratio': candidate['Test_Sharpe_Ratio'],
                'Test_1Y_Total_Return': candidate['Test_Total_Return'],
                'Test_1Y_Annualized_Return': candidate['Test_Annualized_Return'],
                'Test_1Y_Annualized_Volatility': candidate['Test_Annualized_Volatility'],
                
                # نتایج بک‌تست 5 ساله
                'Test_5Y_Total_Return': performance_5y.get('Total_Return_5Y', 0),
                'Test_5Y_Annualized_Return': performance_5y.get('Annualized_Return_5Y', 0),
                'Test_5Y_Annualized_Volatility': performance_5y.get('Annualized_Volatility_5Y', 0),
                'Test_5Y_Sharpe_Ratio': performance_5y.get('Sharpe_Ratio_5Y', 0),
                'Test_5Y_Max_Drawdown': performance_5y.get('Max_Drawdown_5Y', 0),
                
                # پارامترهای استراتژی
                'Momentum_Period': candidate['Momentum_Period'],
                'Value_Weight': candidate['Value_Weight'],
                'Momentum_Weight': candidate['Momentum_Weight'],
                'Low_Volatility_Weight': candidate['Low_Volatility_Weight'],
                'Top_N': candidate['Top_N'],
                'Max_Weight': candidate['Max_Weight'],
                
                # تحلیل تغییرات
                'Sharpe_Change_1Y_to_5Y': performance_5y.get('Sharpe_Ratio_5Y', 0) - candidate['Test_Sharpe_Ratio'],
                'Volatility_Change_1Y_to_5Y': performance_5y.get('Annualized_Volatility_5Y', 0) - candidate['Test_Annualized_Volatility']
            }
            
            results.append(result)
            
            logger.info(f"   ✅ 5Y Sharpe: {performance_5y.get('Sharpe_Ratio_5Y', 0):.2f} "
                       f"(Change: {result['Sharpe_Change_1Y_to_5Y']:+.2f})")
            
        except Exception as e:
            logger.error(f"   ❌ خطا در بک‌تست کاندیدا {idx}: {e}")
    
    return pd.DataFrame(results)

def save_comprehensive_results(results_df):
    """
    ذخیره نتایج جامع در فایل‌های مختلف
    """
    logger.info(f"\n💾 ذخیره نتایج جامع...")
    
    # مرتب‌سازی بر اساس Risk Profile و 5Y Sharpe
    risk_order = {'Defensive': 1, 'Balanced': 2, 'Aggressive': 3}
    results_df['_sort_order'] = results_df['Risk_Profile'].map(risk_order)
    results_df = results_df.sort_values(['_sort_order', 'Test_5Y_Sharpe_Ratio'], ascending=[True, False])
    results_df = results_df.drop('_sort_order', axis=1)
    
    # ذخیره فایل کامل
    results_df.to_csv('comprehensive_backtest_results.csv', index=False)
    logger.info("✅ فایل کامل: comprehensive_backtest_results.csv")
    
    # ذخیره فایل نتایج 1 ساله
    columns_1y = [
        'Candidate_Index', 'Risk_Profile', 'Original_Strategy_Index', 'Original_Sharpe',
        'Test_1Y_Sharpe_Ratio', 'Test_1Y_Total_Return', 'Test_1Y_Annualized_Return', 
        'Test_1Y_Annualized_Volatility', 'Momentum_Period', 'Value_Weight', 
        'Momentum_Weight', 'Low_Volatility_Weight', 'Top_N', 'Max_Weight'
    ]
    results_1y = results_df[columns_1y].copy()
    results_1y.to_csv('backtest_1year_results.csv', index=False)
    logger.info("✅ فایل 1 ساله: backtest_1year_results.csv")
    
    # ذخیره فایل نتایج 5 ساله
    columns_5y = [
        'Candidate_Index', 'Risk_Profile', 'Original_Strategy_Index', 'Original_Sharpe',
        'Test_5Y_Sharpe_Ratio', 'Test_5Y_Total_Return', 'Test_5Y_Annualized_Return', 
        'Test_5Y_Annualized_Volatility', 'Test_5Y_Max_Drawdown', 'Momentum_Period', 
        'Value_Weight', 'Momentum_Weight', 'Low_Volatility_Weight', 'Top_N', 'Max_Weight',
        'Sharpe_Change_1Y_to_5Y', 'Volatility_Change_1Y_to_5Y'
    ]
    results_5y = results_df[columns_5y].copy()
    results_5y.to_csv('backtest_5year_results.csv', index=False)
    logger.info("✅ فایل 5 ساله: backtest_5year_results.csv")
    
    # تحلیل خلاصه
    logger.info(f"\n📊 تحلیل خلاصه نتایج:")
    
    for risk_profile in ['Defensive', 'Balanced', 'Aggressive']:
        subset = results_df[results_df['Risk_Profile'] == risk_profile]
        if len(subset) > 0:
            icon = {'Defensive': '🐢', 'Balanced': '🐺', 'Aggressive': '🦅'}[risk_profile]
            
            avg_1y_sharpe = subset['Test_1Y_Sharpe_Ratio'].mean()
            avg_5y_sharpe = subset['Test_5Y_Sharpe_Ratio'].mean()
            avg_change = subset['Sharpe_Change_1Y_to_5Y'].mean()
            
            logger.info(f"\n{icon} {risk_profile}:")
            logger.info(f"   - میانگین Sharpe 1Y: {avg_1y_sharpe:.2f}")
            logger.info(f"   - میانگین Sharpe 5Y: {avg_5y_sharpe:.2f}")
            logger.info(f"   - میانگین تغییر: {avg_change:+.2f}")
            logger.info(f"   - بهترین 5Y: {subset['Test_5Y_Sharpe_Ratio'].max():.2f}")
            logger.info(f"   - بدترین 5Y: {subset['Test_5Y_Sharpe_Ratio'].min():.2f}")

def main():
    """
    تابع اصلی
    """
    try:
        results_df = run_comprehensive_backtest()
        save_comprehensive_results(results_df)
        
        logger.info(f"\n🎉 بک‌تست جامع با موفقیت کامل شد!")
        logger.info(f"📁 فایل‌های خروجی:")
        logger.info(f"   - comprehensive_backtest_results.csv (کامل)")
        logger.info(f"   - backtest_1year_results.csv (1 ساله)")
        logger.info(f"   - backtest_5year_results.csv (5 ساله)")
        
    except Exception as e:
        logger.error(f"❌ خطا در بک‌تست جامع: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()