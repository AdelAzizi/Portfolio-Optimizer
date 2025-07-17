#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تبدیل کاندیداهای انتخاب شده به فرمت CSV برای نمایش بهتر
"""

import pandas as pd
import json

def load_candidates_from_json(json_path: str) -> pd.DataFrame:
    """
    بارگذاری کاندیداها از فایل JSON و تبدیل به DataFrame
    """
    print(f"📊 بارگذاری کاندیداها از: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_strategies = []
    
    # استخراج استراتژی‌ها از هر دسته
    for category_name, category_data in data.items():
        strategies = category_data['strategies']
        
        for strategy in strategies:
            # اضافه کردن نام دسته ریسک
            strategy['Risk_Profile'] = category_name.title()
            all_strategies.append(strategy)
    
    # تبدیل به DataFrame
    df = pd.DataFrame(all_strategies)
    
    print(f"✅ {len(df)} استراتژی بارگذاری شد")
    return df

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    فرمت‌بندی DataFrame برای نمایش بهتر
    """
    print("🎨 فرمت‌بندی داده‌ها برای نمایش...")
    
    # ترتیب ستون‌ها
    column_order = [
        'Risk_Profile',
        'Sharpe Ratio',
        'Annualized Return', 
        'Annualized Volatility',
        'Total Return',
        'Momentum Period',
        'Value Weight',
        'Momentum Weight',
        'Low Volatility Weight',
        'Top N',
        'Max Weight'
    ]
    
    # انتخاب و مرتب‌سازی ستون‌ها
    df_formatted = df[column_order].copy()
    
    # فرمت‌بندی اعداد
    df_formatted['Sharpe Ratio'] = df_formatted['Sharpe Ratio'].round(2)
    df_formatted['Annualized Return'] = (df_formatted['Annualized Return'] * 100).round(1)
    df_formatted['Annualized Volatility'] = (df_formatted['Annualized Volatility'] * 100).round(1)
    df_formatted['Total Return'] = (df_formatted['Total Return'] * 100).round(1)
    
    # تغییر نام ستون‌ها برای نمایش بهتر
    df_formatted.columns = [
        'Risk Profile',
        'Sharpe Ratio',
        'Annual Return (%)', 
        'Annual Volatility (%)',
        'Total Return (%)',
        'Momentum Period',
        'Value Weight',
        'Momentum Weight',
        'Low Vol Weight',
        'Top N Stocks',
        'Max Weight'
    ]
    
    # مرتب‌سازی بر اساس Risk Profile و Sharpe Ratio
    risk_order = {'Defensive': 1, 'Balanced': 2, 'Aggressive': 3}
    df_formatted['_sort_order'] = df_formatted['Risk Profile'].map(risk_order)
    df_formatted = df_formatted.sort_values(['_sort_order', 'Sharpe Ratio'], ascending=[True, False])
    df_formatted = df_formatted.drop('_sort_order', axis=1)
    
    print("✅ فرمت‌بندی کامل شد")
    return df_formatted

def save_to_csv_with_summary(df: pd.DataFrame, output_path: str):
    """
    ذخیره DataFrame در فایل CSV همراه با خلاصه
    """
    print(f"💾 ذخیره در فایل: {output_path}")
    
    # ذخیره فایل اصلی
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # ایجاد خلاصه
    summary_lines = []
    summary_lines.append("=== خلاصه کاندیداهای انتخاب شده ===")
    summary_lines.append("")
    
    for risk_profile in ['Defensive', 'Balanced', 'Aggressive']:
        subset = df[df['Risk Profile'] == risk_profile]
        if len(subset) > 0:
            icon = {'Defensive': '🐢', 'Balanced': '🐺', 'Aggressive': '🦅'}[risk_profile]
            summary_lines.append(f"{icon} {risk_profile} ({len(subset)} استراتژی):")
            summary_lines.append(f"   - میانگین Sharpe Ratio: {subset['Sharpe Ratio'].mean():.2f}")
            summary_lines.append(f"   - میانگین بازدهی سالانه: {subset['Annual Return (%)'].mean():.1f}%")
            summary_lines.append(f"   - میانگین نوسان سالانه: {subset['Annual Volatility (%)'].mean():.1f}%")
            summary_lines.append(f"   - محدوده Sharpe: {subset['Sharpe Ratio'].min():.2f} - {subset['Sharpe Ratio'].max():.2f}")
            summary_lines.append("")
    
    # ذخیره خلاصه در فایل جداگانه
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✅ فایل CSV ذخیره شد: {output_path}")
    print(f"✅ فایل خلاصه ذخیره شد: {summary_path}")

def display_table_preview(df: pd.DataFrame):
    """
    نمایش پیش‌نمایش جدول
    """
    print("\n📋 پیش‌نمایش جدول:")
    print("=" * 120)
    
    # نمایش 5 ردیف اول از هر دسته
    for risk_profile in ['Defensive', 'Balanced', 'Aggressive']:
        subset = df[df['Risk Profile'] == risk_profile]
        if len(subset) > 0:
            icon = {'Defensive': '🐢', 'Balanced': '🐺', 'Aggressive': '🦅'}[risk_profile]
            print(f"\n{icon} {risk_profile} Strategies:")
            print("-" * 80)
            
            # نمایش ستون‌های کلیدی
            display_cols = ['Sharpe Ratio', 'Annual Return (%)', 'Annual Volatility (%)', 
                          'Momentum Period', 'Value Weight', 'Momentum Weight', 'Low Vol Weight']
            
            for idx, (_, row) in enumerate(subset.iterrows(), 1):
                print(f"{idx}. Sharpe: {row['Sharpe Ratio']:.2f} | "
                      f"Return: {row['Annual Return (%)']:.1f}% | "
                      f"Vol: {row['Annual Volatility (%)']:.1f}% | "
                      f"Period: {row['Momentum Period']} | "
                      f"Weights: {row['Value Weight']:.2f}/{row['Momentum Weight']:.2f}/{row['Low Vol Weight']:.2f}")

def main():
    """
    تابع اصلی
    """
    print("🚀 تبدیل کاندیداهای انتخاب شده به فرمت CSV")
    print("=" * 60)
    
    # مسیرهای فایل
    input_json = "test_selected_candidates.json"
    output_csv = "selected_strategies_candidates.csv"
    
    try:
        # بارگذاری داده‌ها
        df = load_candidates_from_json(input_json)
        
        # فرمت‌بندی
        df_formatted = format_dataframe_for_display(df)
        
        # نمایش پیش‌نمایش
        display_table_preview(df_formatted)
        
        # ذخیره در CSV
        save_to_csv_with_summary(df_formatted, output_csv)
        
        print(f"\n🎉 تبدیل با موفقیت کامل شد!")
        print(f"📁 فایل CSV: {output_csv}")
        print(f"📄 فایل خلاصه: {output_csv.replace('.csv', '_summary.txt')}")
        
    except Exception as e:
        print(f"❌ خطا در تبدیل: {str(e)}")
        raise

if __name__ == "__main__":
    main()