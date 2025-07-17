#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست انتخاب استراتژی‌ها و دسته‌بندی بر اساس ریسک
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_and_select_top_strategies(csv_path: str, top_n: int = 100) -> pd.DataFrame:
    """
    بارگذاری فایل CSV و انتخاب top_n استراتژی برتر بر اساس Sharpe Ratio
    """
    print(f"📊 بارگذاری فایل: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"📈 تعداد کل استراتژی‌ها: {len(df)}")
    print(f"🎯 انتخاب {top_n} استراتژی برتر بر اساس Sharpe Ratio")
    
    # مرتب‌سازی بر اساس Sharpe Ratio (نزولی)
    df_sorted = df.sort_values('Sharpe Ratio', ascending=False)
    
    # انتخاب top_n استراتژی برتر
    top_strategies = df_sorted.head(top_n).copy()
    
    print(f"✅ {len(top_strategies)} استراتژی برتر انتخاب شد")
    print(f"📊 محدوده Sharpe Ratio: {top_strategies['Sharpe Ratio'].min():.2f} - {top_strategies['Sharpe Ratio'].max():.2f}")
    
    return top_strategies

def categorize_by_risk(df: pd.DataFrame) -> dict:
    """
    دسته‌بندی استراتژی‌ها بر اساس Annualized Volatility
    30% کم‌ریسک (دفاعی)، 40% متعادل، 30% پرریسک (تهاجمی)
    """
    print("\n🎯 دسته‌بندی استراتژی‌ها بر اساس ریسک...")
    
    # محاسبه percentile ها
    volatility_30th = df['Annualized Volatility'].quantile(0.30)
    volatility_70th = df['Annualized Volatility'].quantile(0.70)
    
    print(f"📊 آستانه‌های ریسک:")
    print(f"   - کم‌ریسک (دفاعی): Volatility <= {volatility_30th:.4f}")
    print(f"   - متعادل: {volatility_30th:.4f} < Volatility <= {volatility_70th:.4f}")
    print(f"   - پرریسک (تهاجمی): Volatility > {volatility_70th:.4f}")
    
    # دسته‌بندی
    defensive = df[df['Annualized Volatility'] <= volatility_30th].copy()
    balanced = df[(df['Annualized Volatility'] > volatility_30th) & 
                  (df['Annualized Volatility'] <= volatility_70th)].copy()
    aggressive = df[df['Annualized Volatility'] > volatility_70th].copy()
    
    print(f"\n📈 نتایج دسته‌بندی:")
    print(f"   - دفاعی (🐢): {len(defensive)} استراتژی")
    print(f"   - متعادل (🐺): {len(balanced)} استراتژی") 
    print(f"   - تهاجمی (🦅): {len(aggressive)} استراتژی")
    
    return {
        'defensive': defensive,
        'balanced': balanced,
        'aggressive': aggressive
    }

def select_top_candidates(categories: dict, candidates_per_category: int = 5) -> dict:
    """
    انتخاب بهترین کاندیداها از هر دسته بر اساس Sharpe Ratio
    """
    print(f"\n🎯 انتخاب {candidates_per_category} کاندیدای برتر از هر دسته...")
    
    final_candidates = {}
    
    for category_name, df in categories.items():
        if len(df) >= candidates_per_category:
            # انتخاب top candidates بر اساس Sharpe Ratio
            top_candidates = df.nlargest(candidates_per_category, 'Sharpe Ratio')
            final_candidates[category_name] = top_candidates
            
            print(f"✅ {category_name}: {len(top_candidates)} کاندیدا انتخاب شد")
            print(f"   - Sharpe Ratio: {top_candidates['Sharpe Ratio'].min():.2f} - {top_candidates['Sharpe Ratio'].max():.2f}")
            print(f"   - Volatility: {top_candidates['Annualized Volatility'].min():.4f} - {top_candidates['Annualized Volatility'].max():.4f}")
        else:
            print(f"⚠️  {category_name}: فقط {len(df)} استراتژی موجود است (کمتر از {candidates_per_category})")
            final_candidates[category_name] = df
    
    return final_candidates

def save_candidates_to_file(candidates: dict, output_path: str):
    """
    ذخیره کاندیداهای نهایی در فایل
    """
    print(f"\n💾 ذخیره نتایج در: {output_path}")
    
    # آماده‌سازی داده‌ها برای ذخیره
    output_data = {}
    
    for category_name, df in candidates.items():
        # تبدیل DataFrame به لیست dictionary
        strategies_list = df.to_dict('records')
        
        # اضافه کردن اطلاعات خلاصه
        summary = {
            'count': len(df),
            'avg_sharpe_ratio': float(df['Sharpe Ratio'].mean()),
            'avg_volatility': float(df['Annualized Volatility'].mean()),
            'avg_return': float(df['Annualized Return'].mean()),
            'strategies': strategies_list
        }
        
        output_data[category_name] = summary
    
    # ذخیره در فایل JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("✅ فایل با موفقیت ذخیره شد!")
    
    # نمایش خلاصه
    print(f"\n📊 خلاصه نتایج:")
    total_candidates = sum(len(df) for df in candidates.values())
    print(f"   - مجموع کاندیداها: {total_candidates}")
    
    for category_name, summary in output_data.items():
        print(f"   - {category_name}: {summary['count']} کاندیدا")
        print(f"     * میانگین Sharpe: {summary['avg_sharpe_ratio']:.2f}")
        print(f"     * میانگین Volatility: {summary['avg_volatility']:.4f}")

def main():
    """
    تابع اصلی برای اجرای تست
    """
    print("🚀 شروع تست انتخاب و دسته‌بندی استراتژی‌ها")
    print("=" * 60)
    
    # مسیرهای فایل
    input_csv = "data/top_300_strategies.csv"
    output_json = "test_selected_candidates.json"
    
    try:
        # مرحله 1: بارگذاری و انتخاب 100 استراتژی برتر
        top_100 = load_and_select_top_strategies(input_csv, top_n=100)
        
        # مرحله 2: دسته‌بندی بر اساس ریسک
        risk_categories = categorize_by_risk(top_100)
        
        # مرحله 3: انتخاب 5 کاندیدای برتر از هر دسته
        final_candidates = select_top_candidates(risk_categories, candidates_per_category=5)
        
        # مرحله 4: ذخیره نتایج
        save_candidates_to_file(final_candidates, output_json)
        
        print(f"\n🎉 تست با موفقیت کامل شد!")
        print(f"📁 نتایج در فایل {output_json} ذخیره شد")
        
    except Exception as e:
        print(f"❌ خطا در اجرای تست: {str(e)}")
        raise

if __name__ == "__main__":
    main()