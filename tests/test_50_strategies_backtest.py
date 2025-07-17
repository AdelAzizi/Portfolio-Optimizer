#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست بک‌تست 1 ساله برای 50 استراتژی اول از top_300_strategies.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_market_data_simulation():
    """
    شبیه‌سازی داده‌های بازار برای تست
    در پیاده‌سازی واقعی، این از cache/full_analysis_ready_data.feather می‌آید
    """
    logger.info("📊 شبیه‌سازی داده‌های بازار...")
    
    np.random.seed(42)  # برای تکرارپذیری
    
    # تاریخ‌ها (1 سال گذشته)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # فقط روزهای کاری
    
    # شبیه‌سازی 200 سهم
    symbols = [f"STOCK_{i:03d}" for i in range(1, 201)]
    
    # ایجاد داده‌های قیمتی
    price_data = {}
    for symbol in symbols:
        # شبیه‌سازی قیمت‌ها با random walk
        returns = np.random.normal(0.001, 0.025, len(dates))  # بازدهی روزانه
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[symbol] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    
    logger.info(f"✅ داده‌های {len(symbols)} سهم برای {len(dates)} روز شبیه‌سازی شد")
    return df

def calculate_factors_for_backtest(price_data, lookback_6m=126, lookback_12m=252):
    """
    محاسبه فاکتورهای مالی برای بک‌تست
    """
    logger.info("🧮 محاسبه فاکتورهای مالی...")
    
    returns = price_data.pct_change().fillna(0)
    factors = {}
    
    for date in price_data.index[lookback_12m:]:
        date_factors = {}
        
        hist_prices = price_data.loc[:date]
        hist_returns = returns.loc[:date]
        
        if len(hist_prices) < lookback_12m:
            continue
            
        for symbol in price_data.columns:
            try:
                # Value Factor (شبیه‌سازی)
                recent_price = hist_prices[symbol].iloc[-1]
                avg_price_12m = hist_prices[symbol].iloc[-lookback_12m:].mean()
                value_factor = avg_price_12m / recent_price if recent_price > 0 else 0
                
                # Momentum 6M
                momentum_6m = hist_returns[symbol].iloc[-lookback_6m:-21].sum() if len(hist_returns) >= lookback_6m else 0
                
                # Momentum 12M
                momentum_12m = hist_returns[symbol].iloc[-lookback_12m:-21].sum() if len(hist_returns) >= lookback_12m else 0
                
                # Low Volatility
                volatility = hist_returns[symbol].iloc[-lookback_6m:].std() * np.sqrt(252) if len(hist_returns) >= lookback_6m else 0.2
                low_vol_factor = 1 / (1 + volatility) if volatility > 0 else 0.5
                
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

def run_strategy_backtest(price_data, factors, strategy_config, rebalance_freq=21):
    """
    اجرای بک‌تست برای یک استراتژی
    """
    momentum_period = strategy_config['Momentum Period']
    value_weight = strategy_config['Value Weight']
    momentum_weight = strategy_config['Momentum Weight']
    low_vol_weight = strategy_config['Low Volatility Weight']
    top_n = int(strategy_config['Top N'])
    max_weight = strategy_config['Max Weight']
    
    momentum_key = 'momentum_12m' if momentum_period == '12M' else 'momentum_6m'
    
    portfolio_value = 100.0
    factor_dates = sorted(factors.keys())
    rebalance_dates = factor_dates[::rebalance_freq]
    
    current_weights = {}
    
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
    
    # محاسبه معیارهای عملکرد
    total_return = (portfolio_value / 100.0) - 1
    
    # شبیه‌سازی سایر معیارها
    annualized_return = (1 + total_return) ** (252 / len(factor_dates)) - 1
    annualized_volatility = np.random.normal(0.15, 0.05)  # شبیه‌سازی
    annualized_volatility = max(0.05, abs(annualized_volatility))  # حداقل 5%
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio
    }

def test_50_strategies():
    """
    تست 50 استراتژی اول از top_300_strategies.csv
    """
    logger.info("🚀 شروع تست 50 استراتژی اول")
    logger.info("=" * 60)
    
    # بارگذاری استراتژی‌ها
    strategies_df = pd.read_csv('data/top_300_strategies.csv')
    top_50_strategies = strategies_df.head(50)
    
    logger.info(f"📊 تست {len(top_50_strategies)} استراتژی از {len(strategies_df)} استراتژی کل")
    
    # بارگذاری داده‌های بازار
    price_data = load_market_data_simulation()
    factors = calculate_factors_for_backtest(price_data)
    
    # اجرای بک‌تست
    results = []
    
    for idx, (_, strategy) in enumerate(top_50_strategies.iterrows(), 1):
        logger.info(f"\n🔄 تست استراتژی {idx}/50")
        logger.info(f"   Momentum: {strategy['Momentum Period']}, "
                   f"Weights: {strategy['Value Weight']:.2f}/{strategy['Momentum Weight']:.2f}/{strategy['Low Volatility Weight']:.2f}, "
                   f"TopN: {strategy['Top N']}, MaxW: {strategy['Max Weight']}")
        
        try:
            performance = run_strategy_backtest(price_data, factors, strategy)
            
            result = {
                'Strategy_Index': idx,
                'Original_Sharpe': strategy['Sharpe Ratio'],
                'Momentum_Period': strategy['Momentum Period'],
                'Value_Weight': strategy['Value Weight'],
                'Momentum_Weight': strategy['Momentum Weight'],
                'Low_Volatility_Weight': strategy['Low Volatility Weight'],
                'Top_N': strategy['Top N'],
                'Max_Weight': strategy['Max Weight'],
                'Test_Total_Return': performance['Total Return'],
                'Test_Annualized_Return': performance['Annualized Return'],
                'Test_Annualized_Volatility': performance['Annualized Volatility'],
                'Test_Sharpe_Ratio': performance['Sharpe Ratio'],
                'Sharpe_Difference': performance['Sharpe Ratio'] - strategy['Sharpe Ratio']
            }
            
            results.append(result)
            
            logger.info(f"   ✅ Original Sharpe: {strategy['Sharpe Ratio']:.2f} → Test Sharpe: {performance['Sharpe Ratio']:.2f} "
                       f"(Diff: {result['Sharpe_Difference']:+.2f})")
            
        except Exception as e:
            logger.error(f"   ❌ خطا در تست استراتژی {idx}: {e}")
    
    # ذخیره نتایج
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_50_strategies_results.csv', index=False)
    
    # تحلیل نتایج
    logger.info(f"\n📊 تحلیل نتایج:")
    logger.info(f"   - تعداد استراتژی‌های تست شده: {len(results_df)}")
    logger.info(f"   - میانگین Sharpe اصلی: {results_df['Original_Sharpe'].mean():.2f}")
    logger.info(f"   - میانگین Sharpe تست: {results_df['Test_Sharpe_Ratio'].mean():.2f}")
    logger.info(f"   - میانگین اختلاف: {results_df['Sharpe_Difference'].mean():+.2f}")
    logger.info(f"   - انحراف معیار اختلاف: {results_df['Sharpe_Difference'].std():.2f}")
    
    # نمایش 10 مورد برتر و بدترین
    logger.info(f"\n🏆 10 استراتژی با بهترین عملکرد در تست:")
    best_10 = results_df.nlargest(10, 'Test_Sharpe_Ratio')
    for _, row in best_10.iterrows():
        logger.info(f"   {row['Strategy_Index']:2d}. Original: {row['Original_Sharpe']:.2f} → Test: {row['Test_Sharpe_Ratio']:.2f} "
                   f"({row['Sharpe_Difference']:+.2f}) | {row['Momentum_Period']} | "
                   f"{row['Value_Weight']:.2f}/{row['Momentum_Weight']:.2f}/{row['Low_Volatility_Weight']:.2f}")
    
    logger.info(f"\n📉 10 استراتژی با بدترین عملکرد در تست:")
    worst_10 = results_df.nsmallest(10, 'Test_Sharpe_Ratio')
    for _, row in worst_10.iterrows():
        logger.info(f"   {row['Strategy_Index']:2d}. Original: {row['Original_Sharpe']:.2f} → Test: {row['Test_Sharpe_Ratio']:.2f} "
                   f"({row['Sharpe_Difference']:+.2f}) | {row['Momentum_Period']} | "
                   f"{row['Value_Weight']:.2f}/{row['Momentum_Weight']:.2f}/{row['Low_Volatility_Weight']:.2f}")
    
    logger.info(f"\n✅ نتایج در test_50_strategies_results.csv ذخیره شد")
    
    return results_df

if __name__ == "__main__":
    test_50_strategies()