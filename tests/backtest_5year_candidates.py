#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بک‌تست 5 ساله برای 15 کاندیدای انتخاب شده
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_market_data():
    """
    بارگذاری داده‌های بازار (شبیه‌سازی شده)
    در پیاده‌سازی واقعی، این داده‌ها از cache/master_price_data.feather می‌آید
    """
    print("📊 بارگذاری داده‌های بازار...")
    
    # شبیه‌سازی داده‌های قیمتی برای 5 سال
    # در پیاده‌سازی واقعی، این از فایل‌های cache بارگذاری می‌شود
    np.random.seed(42)  # برای تکرارپذیری
    
    # تاریخ‌ها (5 سال گذشته)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # فقط روزهای کاری
    
    # شبیه‌سازی 100 سهم
    symbols = [f"SYM{i:03d}" for i in range(1, 101)]
    
    # ایجاد داده‌های قیمتی شبیه‌سازی شده
    price_data = {}
    for symbol in symbols:
        # شبیه‌سازی قیمت‌ها با random walk
        returns = np.random.normal(0.0008, 0.02, len(dates))  # بازدهی روزانه
        prices = 100 * np.exp(np.cumsum(returns))  # قیمت‌ها
        price_data[symbol] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    
    print(f"✅ داده‌های {len(symbols)} سهم برای {len(dates)} روز بارگذاری شد")
    return df

def calculate_factors(price_data, lookback_6m=126, lookback_12m=252):
    """
    محاسبه فاکتورهای مالی
    """
    print("🧮 محاسبه فاکتورهای مالی...")
    
    # محاسبه بازدهی‌ها
    returns = price_data.pct_change().fillna(0)
    
    factors = {}
    
    for date in price_data.index[lookback_12m:]:  # شروع از 12 ماه بعد
        date_factors = {}
        
        # داده‌های تا این تاریخ
        hist_prices = price_data.loc[:date]
        hist_returns = returns.loc[:date]
        
        if len(hist_prices) < lookback_12m:
            continue
            
        # محاسبه فاکتورها برای هر سهم
        for symbol in price_data.columns:
            try:
                # Value Factor (شبیه‌سازی P/E معکوس)
                recent_price = hist_prices[symbol].iloc[-1]
                avg_price = hist_prices[symbol].iloc[-lookback_12m:].mean()
                value_factor = avg_price / recent_price if recent_price > 0 else 0
                
                # Momentum Factor (6M)
                if len(hist_returns) >= lookback_6m:
                    momentum_6m = hist_returns[symbol].iloc[-lookback_6m:-21].sum()  # حذف ماه اخیر
                else:
                    momentum_6m = 0
                
                # Momentum Factor (12M)
                if len(hist_returns) >= lookback_12m:
                    momentum_12m = hist_returns[symbol].iloc[-lookback_12m:-21].sum()  # حذف ماه اخیر
                else:
                    momentum_12m = 0
                
                # Low Volatility Factor (معکوس نوسان)
                if len(hist_returns) >= lookback_6m:
                    volatility = hist_returns[symbol].iloc[-lookback_6m:].std() * np.sqrt(252)
                    low_vol_factor = 1 / (1 + volatility) if volatility > 0 else 0
                else:
                    low_vol_factor = 0
                
                date_factors[symbol] = {
                    'value': value_factor,
                    'momentum_6m': momentum_6m,
                    'momentum_12m': momentum_12m,
                    'low_volatility': low_vol_factor
                }
                
            except Exception as e:
                date_factors[symbol] = {
                    'value': 0,
                    'momentum_6m': 0,
                    'momentum_12m': 0,
                    'low_volatility': 0
                }
        
        factors[date] = date_factors
    
    print(f"✅ فاکتورها برای {len(factors)} تاریخ محاسبه شد")
    return factors

def backtest_strategy(price_data, factors, strategy_config, rebalance_freq=21):
    """
    بک‌تست یک استراتژی
    """
    momentum_period = strategy_config['Momentum Period']
    value_weight = strategy_config['Value Weight']
    momentum_weight = strategy_config['Momentum Weight']
    low_vol_weight = strategy_config['Low Volatility Weight']
    top_n = int(strategy_config['Top N'])
    max_weight = strategy_config['Max Weight']
    
    # انتخاب فاکتور momentum
    momentum_key = 'momentum_12m' if momentum_period == '12M' else 'momentum_6m'
    
    portfolio_values = []
    rebalance_dates = []
    portfolio_weights = {}
    
    # تاریخ‌های rebalance
    factor_dates = sorted(factors.keys())
    rebalance_dates = factor_dates[::rebalance_freq]  # هر 21 روز
    
    current_weights = {}
    portfolio_value = 100.0  # شروع با 100
    
    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date not in factors:
            continue
            
        # محاسبه امتیاز ترکیبی برای هر سهم
        date_factors = factors[rebalance_date]
        scores = {}
        
        for symbol, factor_values in date_factors.items():
            # ترکیب فاکتورها
            combined_score = (
                value_weight * factor_values['value'] +
                momentum_weight * factor_values[momentum_key] +
                low_vol_weight * factor_values['low_volatility']
            )
            scores[symbol] = combined_score
        
        # انتخاب top N سهم
        top_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]
        
        # محاسبه وزن‌ها (equal weight با محدودیت max_weight)
        if top_symbols:
            equal_weight = 1.0 / len(top_symbols)
            target_weight = min(equal_weight, max_weight)
            
            # تنظیم وزن‌ها
            new_weights = {}
            total_weight = 0
            for symbol in top_symbols:
                new_weights[symbol] = target_weight
                total_weight += target_weight
            
            # نرمال‌سازی اگر مجموع وزن‌ها کمتر از 1 باشد
            if total_weight < 1.0 and top_symbols:
                scale_factor = 1.0 / total_weight
                for symbol in new_weights:
                    new_weights[symbol] *= scale_factor
            
            current_weights = new_weights
        
        # محاسبه عملکرد تا تاریخ rebalance بعدی
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            period_dates = [d for d in factor_dates if rebalance_date <= d < next_rebalance]
        else:
            period_dates = [d for d in factor_dates if d >= rebalance_date]
        
        # محاسبه بازدهی پرتفوی
        for date in period_dates:
            if date in price_data.index and date != rebalance_date:
                # محاسبه بازدهی روزانه پرتفوی
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
                'portfolio_value': portfolio_value,
                'weights': current_weights.copy()
            })
    
    return portfolio_values

def calculate_performance_metrics(portfolio_values):
    """
    محاسبه معیارهای عملکرد
    """
    if not portfolio_values:
        return {}
    
    # تبدیل به DataFrame
    df = pd.DataFrame(portfolio_values)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # محاسبه بازدهی‌ها
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    
    # معیارهای عملکرد
    total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    annualized_volatility = df['returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Max Drawdown
    rolling_max = df['portfolio_value'].expanding().max()
    drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Annualized Volatility': f"{annualized_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Total Return (numeric)': total_return,
        'Annualized Return (numeric)': annualized_return,
        'Annualized Volatility (numeric)': annualized_volatility,
        'Sharpe Ratio (numeric)': sharpe_ratio
    }

def run_5year_backtest():
    """
    اجرای بک‌تست 5 ساله برای تمام کاندیداها
    """
    print("🚀 شروع بک‌تست 5 ساله")
    print("=" * 60)
    
    # بارگذاری کاندیداها
    with open('test_selected_candidates.json', 'r', encoding='utf-8') as f:
        candidates_data = json.load(f)
    
    # بارگذاری داده‌های بازار
    price_data = load_market_data()
    
    # محاسبه فاکتورها
    factors = calculate_factors(price_data)
    
    # اجرای بک‌تست برای هر کاندیدا
    results = []
    
    for risk_profile, profile_data in candidates_data.items():
        strategies = profile_data['strategies']
        
        print(f"\n🎯 بک‌تست استراتژی‌های {risk_profile}...")
        
        for i, strategy in enumerate(strategies, 1):
            print(f"   {i}/5 - Sharpe: {strategy['Sharpe Ratio']:.2f}")
            
            # اجرای بک‌تست
            portfolio_values = backtest_strategy(price_data, factors, strategy)
            
            # محاسبه معیارهای عملکرد
            performance = calculate_performance_metrics(portfolio_values)
            
            # ذخیره نتایج
            result = {
                'Risk_Profile': risk_profile.title(),
                'Original_Sharpe_1Y': strategy['Sharpe Ratio'],
                'Original_Return_1Y': f"{strategy['Annualized Return']:.1%}",
                'Original_Volatility_1Y': f"{strategy['Annualized Volatility']:.1%}",
                'Momentum_Period': strategy['Momentum Period'],
                'Value_Weight': strategy['Value Weight'],
                'Momentum_Weight': strategy['Momentum Weight'],
                'Low_Vol_Weight': strategy['Low Volatility Weight'],
                'Top_N': strategy['Top N'],
                'Max_Weight': strategy['Max Weight'],
                **performance
            }
            
            results.append(result)
    
    return results

def save_results_to_csv(results, output_path):
    """
    ذخیره نتایج در فایل CSV
    """
    df = pd.DataFrame(results)
    
    # مرتب‌سازی بر اساس Risk Profile و Sharpe Ratio جدید
    risk_order = {'Defensive': 1, 'Balanced': 2, 'Aggressive': 3}
    df['_sort_order'] = df['Risk_Profile'].map(risk_order)
    df = df.sort_values(['_sort_order', 'Sharpe Ratio (numeric)'], ascending=[True, False])
    df = df.drop(['_sort_order', 'Total Return (numeric)', 'Annualized Return (numeric)', 
                  'Annualized Volatility (numeric)', 'Sharpe Ratio (numeric)'], axis=1)
    
    # ذخیره
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ نتایج در {output_path} ذخیره شد")
    
    # نمایش خلاصه
    print(f"\n📊 خلاصه نتایج بک‌تست 5 ساله:")
    for risk_profile in ['Defensive', 'Balanced', 'Aggressive']:
        subset = df[df['Risk_Profile'] == risk_profile]
        if len(subset) > 0:
            icon = {'Defensive': '🐢', 'Balanced': '🐺', 'Aggressive': '🦅'}[risk_profile]
            avg_sharpe_1y = subset['Original_Sharpe_1Y'].mean()
            avg_sharpe_5y = [float(x) for x in subset['Sharpe Ratio'].str.replace('', '')]
            
            print(f"\n{icon} {risk_profile}:")
            print(f"   - میانگین Sharpe 1Y: {avg_sharpe_1y:.2f}")
            print(f"   - محدوده Sharpe 5Y: {subset['Sharpe Ratio'].min()} - {subset['Sharpe Ratio'].max()}")

def main():
    """
    تابع اصلی
    """
    try:
        results = run_5year_backtest()
        save_results_to_csv(results, "backtest_5year_results.csv")
        
        print(f"\n🎉 بک‌تست 5 ساله با موفقیت کامل شد!")
        print(f"📁 نتایج در backtest_5year_results.csv ذخیره شد")
        
    except Exception as e:
        print(f"❌ خطا در بک‌تست: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()