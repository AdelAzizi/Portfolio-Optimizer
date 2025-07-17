#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
انتخاب 5 کاندیدا برای هر پروفایل ریسک از نتایج بک‌تست 50 استراتژی
"""

import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_backtest_results():
    """
    بارگذاری نتایج بک‌تست 50 استراتژی
    """
    logger.info("📊 بارگذاری نتایج بک‌تست 50 استراتژی...")
    
    df = pd.read_csv('test_50_strategies_results.csv')
    
    logger.info(f"✅ {len(df)} نتیجه بک‌تست بارگذاری شد")
    logger.info(f"📈 محدوده Test Sharpe Ratio: {df['Test_Sharpe_Ratio'].min():.2f} تا {df['Test_Sharpe_Ratio'].max():.2f}")
    
    return df

def categorize_by_volatility(df):
    """
    دسته‌بندی استراتژی‌ها بر اساس Test_Annualized_Volatility
    """
    logger.info("\n🎯 دسته‌بندی استراتژی‌ها بر اساس نوسان...")
    
    # محاسبه percentile ها بر اساس نوسان تست
    volatility_30th = df['Test_Annualized_Volatility'].quantile(0.30)
    volatility_70th = df['Test_Annualized_Volatility'].quantile(0.70)
    
    logger.info(f"📊 آستانه‌های ریسک (بر اساس نوسان تست):")
    logger.info(f"   - کم‌ریسک (دفاعی): Volatility <= {volatility_30th:.4f}")
    logger.info(f"   - متعادل: {volatility_30th:.4f} < Volatility <= {volatility_70th:.4f}")
    logger.info(f"   - پرریسک (تهاجمی): Volatility > {volatility_70th:.4f}")
    
    # دسته‌بندی
    defensive = df[df['Test_Annualized_Volatility'] <= volatility_30th].copy()
    balanced = df[(df['Test_Annualized_Volatility'] > volatility_30th) & 
                  (df['Test_Annualized_Volatility'] <= volatility_70th)].copy()
    aggressive = df[df['Test_Annualized_Volatility'] > volatility_70th].copy()
    
    logger.info(f"\n📈 نتایج دسته‌بندی:")
    logger.info(f"   - دفاعی (🐢): {len(defensive)} استراتژی")
    logger.info(f"   - متعادل (🐺): {len(balanced)} استراتژی") 
    logger.info(f"   - تهاجمی (🦅): {len(aggressive)} استراتژی")
    
    return {
        'defensive': defensive,
        'balanced': balanced,
        'aggressive': aggressive
    }

def select_top_candidates_by_test_sharpe(categories, candidates_per_category=5):
    """
    انتخاب بهترین کاندیداها از هر دسته بر اساس Test Sharpe Ratio
    """
    logger.info(f"\n🎯 انتخاب {candidates_per_category} کاندیدای برتر از هر دسته (بر اساس Test Sharpe)...")
    
    final_candidates = {}
    
    for category_name, df in categories.items():
        if len(df) >= candidates_per_category:
            # انتخاب top candidates بر اساس Test Sharpe Ratio
            top_candidates = df.nlargest(candidates_per_category, 'Test_Sharpe_Ratio')
            final_candidates[category_name] = top_candidates
            
            logger.info(f"✅ {category_name}: {len(top_candidates)} کاندیدا انتخاب شد")
            logger.info(f"   - Test Sharpe Ratio: {top_candidates['Test_Sharpe_Ratio'].min():.2f} - {top_candidates['Test_Sharpe_Ratio'].max():.2f}")
            logger.info(f"   - Original Sharpe: {top_candidates['Original_Sharpe'].min():.2f} - {top_candidates['Original_Sharpe'].max():.2f}")
            logger.info(f"   - Test Volatility: {top_candidates['Test_Annualized_Volatility'].min():.4f} - {top_candidates['Test_Annualized_Volatility'].max():.4f}")
        else:
            logger.info(f"⚠️  {category_name}: فقط {len(df)} استراتژی موجود است (کمتر از {candidates_per_category})")
            final_candidates[category_name] = df.nlargest(len(df), 'Test_Sharpe_Ratio') if len(df) > 0 else df
    
    return final_candidates

def save_candidates_analysis(candidates, output_csv, output_json):
    """
    ذخیره تحلیل کاندیداها در فایل‌های CSV و JSON
    """
    logger.info(f"\n💾 ذخیره تحلیل کاندیداها...")
    
    # آماده‌سازی داده‌ها برای CSV
    all_candidates = []
    
    for category_name, df in candidates.items():
        df_copy = df.copy()
        df_copy['Risk_Profile'] = category_name.title()
        all_candidates.append(df_copy)
    
    # ترکیب تمام کاندیداها
    combined_df = pd.concat(all_candidates, ignore_index=True)
    
    # مرتب‌سازی بر اساس Risk Profile و Test Sharpe Ratio
    risk_order = {'Defensive': 1, 'Balanced': 2, 'Aggressive': 3}
    combined_df['_sort_order'] = combined_df['Risk_Profile'].map(risk_order)
    combined_df = combined_df.sort_values(['_sort_order', 'Test_Sharpe_Ratio'], ascending=[True, False])
    combined_df = combined_df.drop('_sort_order', axis=1)
    
    # ذخیره CSV
    combined_df.to_csv(output_csv, index=False)
    logger.info(f"✅ فایل CSV ذخیره شد: {output_csv}")
    
    # آماده‌سازی داده‌ها برای JSON
    json_data = {}
    
    for category_name, df in candidates.items():
        strategies_list = []
        
        for _, row in df.iterrows():
            strategy = {
                'Strategy_Index': int(row['Strategy_Index']),
                'Original_Sharpe': float(row['Original_Sharpe']),
                'Test_Sharpe_Ratio': float(row['Test_Sharpe_Ratio']),
                'Sharpe_Difference': float(row['Sharpe_Difference']),
                'Momentum_Period': row['Momentum_Period'],
                'Value_Weight': float(row['Value_Weight']),
                'Momentum_Weight': float(row['Momentum_Weight']),
                'Low_Volatility_Weight': float(row['Low_Volatility_Weight']),
                'Top_N': int(row['Top_N']),
                'Max_Weight': float(row['Max_Weight']),
                'Test_Total_Return': float(row['Test_Total_Return']),
                'Test_Annualized_Return': float(row['Test_Annualized_Return']),
                'Test_Annualized_Volatility': float(row['Test_Annualized_Volatility'])
            }
            strategies_list.append(strategy)
        
        # خلاصه آماری
        summary = {
            'count': len(df),
            'avg_original_sharpe': float(df['Original_Sharpe'].mean()),
            'avg_test_sharpe': float(df['Test_Sharpe_Ratio'].mean()),
            'avg_sharpe_difference': float(df['Sharpe_Difference'].mean()),
            'avg_test_volatility': float(df['Test_Annualized_Volatility'].mean()),
            'avg_test_return': float(df['Test_Annualized_Return'].mean()),
            'strategies': strategies_list
        }
        
        json_data[category_name] = summary
    
    # ذخیره JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ فایل JSON ذخیره شد: {output_json}")
    
    # نمایش خلاصه
    logger.info(f"\n📊 خلاصه کاندیداهای انتخاب شده:")
    total_candidates = sum(len(df) for df in candidates.values())
    logger.info(f"   - مجموع کاندیداها: {total_candidates}")
    
    for category_name, summary in json_data.items():
        icon = {'defensive': '🐢', 'balanced': '🐺', 'aggressive': '🦅'}[category_name]
        logger.info(f"\n   {icon} {category_name.title()}: {summary['count']} کاندیدا")
        logger.info(f"     * میانگین Original Sharpe: {summary['avg_original_sharpe']:.2f}")
        logger.info(f"     * میانگین Test Sharpe: {summary['avg_test_sharpe']:.2f}")
        logger.info(f"     * میانگین اختلاف Sharpe: {summary['avg_sharpe_difference']:+.2f}")
        logger.info(f"     * میانگین Test Volatility: {summary['avg_test_volatility']:.4f}")

def analyze_performance_patterns(candidates):
    """
    تحلیل الگوهای عملکرد
    """
    logger.info(f"\n🔍 تحلیل الگوهای عملکرد:")
    
    for category_name, df in candidates.items():
        icon = {'defensive': '🐢', 'balanced': '🐺', 'aggressive': '🦅'}[category_name]
        logger.info(f"\n{icon} {category_name.title()} Strategies:")
        
        # تحلیل الگوهای Momentum Period
        momentum_counts = df['Momentum_Period'].value_counts()
        logger.info(f"   - Momentum Period: {dict(momentum_counts)}")
        
        # تحلیل میانگین وزن‌ها
        avg_value = df['Value_Weight'].mean()
        avg_momentum = df['Momentum_Weight'].mean()
        avg_low_vol = df['Low_Volatility_Weight'].mean()
        logger.info(f"   - میانگین وزن‌ها: Value={avg_value:.2f}, Momentum={avg_momentum:.2f}, LowVol={avg_low_vol:.2f}")
        
        # بهترین استراتژی
        best_strategy = df.loc[df['Test_Sharpe_Ratio'].idxmax()]
        logger.info(f"   - بهترین استراتژی: Index={best_strategy['Strategy_Index']}, "
                   f"Test Sharpe={best_strategy['Test_Sharpe_Ratio']:.2f}, "
                   f"Original Sharpe={best_strategy['Original_Sharpe']:.2f}")

def main():
    """
    تابع اصلی
    """
    logger.info("🚀 انتخاب کاندیداها از نتایج بک‌تست 50 استراتژی")
    logger.info("=" * 70)
    
    try:
        # بارگذاری نتایج
        results_df = load_backtest_results()
        
        # دسته‌بندی بر اساس ریسک
        risk_categories = categorize_by_volatility(results_df)
        
        # انتخاب کاندیداها
        final_candidates = select_top_candidates_by_test_sharpe(risk_categories, candidates_per_category=5)
        
        # تحلیل الگوهای عملکرد
        analyze_performance_patterns(final_candidates)
        
        # ذخیره نتایج
        save_candidates_analysis(
            final_candidates, 
            'selected_candidates_from_50_test.csv',
            'selected_candidates_from_50_test.json'
        )
        
        logger.info(f"\n🎉 انتخاب کاندیداها با موفقیت کامل شد!")
        
    except Exception as e:
        logger.error(f"❌ خطا در انتخاب کاندیداها: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()