#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from data_management import DataLoader, AnalysisCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🧪 Testing Stability Analysis directly...")

try:
    # بارگذاری داده‌ها
    data_loader = DataLoader()
    data_1y = data_loader.load_backtest_data('1y')
    data_5y = data_loader.load_backtest_data('5y')
    
    print(f"📊 Loaded 1y data: {len(data_1y)} records")
    print(f"📊 Loaded 5y data: {len(data_5y)} records")
    
    # ترکیب داده‌ها
    data_1y['Period'] = '1y'
    data_5y['Period'] = '5y'
    combined_data = pd.concat([data_1y, data_5y], ignore_index=True)
    
    print(f"📊 Combined data: {len(combined_data)} records")
    print(f"📊 Columns: {list(combined_data.columns)}")
    
    # تحلیل ثبات ساده
    stability_results = {}
    
    if 'Candidate_Index' in combined_data.columns:
        grouped = combined_data.groupby('Candidate_Index')
        
        for candidate_idx, group in grouped:
            metrics = {}
            
            # محاسبه Consistency Score
            if 'Sharpe_Ratio' in group.columns:
                positive_periods = (group['Sharpe_Ratio'] > 0).sum()
                total_periods = len(group['Sharpe_Ratio'].dropna())
                consistency_ratio = positive_periods / total_periods if total_periods > 0 else 0
                metrics['consistency_ratio'] = consistency_ratio
                
                # محاسبه Stability Score ساده
                stability_score = consistency_ratio  # ساده‌سازی شده
                metrics['stability_score'] = stability_score
                
                # الگوهای شکست
                weak_periods = group['Sharpe_Ratio'] < 0
                metrics['weak_periods_ratio'] = weak_periods.mean()
                
                stability_results[f"strategy_{candidate_idx}"] = metrics
                
                print(f"  ✅ Strategy {candidate_idx}: Stability={stability_score:.3f}, Consistency={consistency_ratio:.3f}")
    
    # رتبه‌بندی
    sorted_strategies = sorted(stability_results.items(), 
                              key=lambda x: x[1]['stability_score'], reverse=True)
    
    print("\n🏆 Top strategies by stability:")
    for i, (strategy, metrics) in enumerate(sorted_strategies[:5], 1):
        print(f"  {i}. {strategy}: {metrics['stability_score']:.3f}")
    
    # ذخیره گزارش
    report = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'stability_metrics': stability_results,
        'key_findings': [
            f"Total strategies analyzed: {len(stability_results)}",
            f"Average stability score: {np.mean([m['stability_score'] for m in stability_results.values()]):.3f}",
            f"Best strategy: {sorted_strategies[0][0]} (score: {sorted_strategies[0][1]['stability_score']:.3f})"
        ],
        'recommendations': [
            f"Top 3 stable strategies: {', '.join([s[0] for s in sorted_strategies[:3]])}",
            "Focus on strategies with consistency ratio > 0.7"
        ]
    }
    
    # ذخیره فایل
    output_path = Path("tests/results/stability_analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "stability_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Report saved to: {output_path / 'stability_report.json'}")
    print("✅ Stability analysis completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()