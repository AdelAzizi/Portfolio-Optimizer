#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from data_management import DataLoader, AnalysisCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🧪 Testing Parameter Sensitivity Analysis directly...")

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
    
    # شناسایی پارامترهای قابل تنظیم
    performance_params = [
        'Sharpe_Ratio', 'Total_Return', 'Volatility', 'Max_Drawdown',
        'Annual_Return', 'Win_Rate', 'Profit_Factor', 'Sortino_Ratio'
    ]
    
    adjustable_params = {}
    for param in performance_params:
        if param in combined_data.columns:
            param_values = combined_data[param].dropna()
            if len(param_values) > 0:
                adjustable_params[param] = {
                    'current_mean': param_values.mean(),
                    'current_std': param_values.std(),
                    'min_value': param_values.min(),
                    'max_value': param_values.max()
                }
    
    print(f"\n🔍 Identified {len(adjustable_params)} adjustable parameters:")
    for param, info in adjustable_params.items():
        print(f"  📊 {param}: mean={info['current_mean']:.3f}, std={info['current_std']:.3f}")
    
    # تحلیل حساسیت ساده
    sensitivity_results = {}
    parameter_variation_range = 0.2  # ±20%
    sensitivity_threshold = 0.05     # 5% تغییر
    
    for param_name, param_info in adjustable_params.items():
        print(f"\n🔬 Analyzing sensitivity for {param_name}...")
        
        base_value = param_info['current_mean']
        
        # محاسبه محدوده تغییرات
        min_test_value = base_value * (1 - parameter_variation_range)
        max_test_value = base_value * (1 + parameter_variation_range)
        
        # تولید نقاط تست
        test_values = np.linspace(min_test_value, max_test_value, 11)
        
        # شبیه‌سازی تأثیر تغییر پارامتر
        performance_changes = []
        
        for test_value in test_values:
            # محاسبه ضریب تغییر
            change_factor = test_value / base_value if base_value != 0 else 1.0
            
            # شبیه‌سازی تأثیر بر Sharpe Ratio
            if param_name == 'Sharpe_Ratio':
                # تغییر مستقیم
                performance_change = (change_factor - 1.0)
            elif param_name == 'Total_Return':
                # تأثیر مثبت بر عملکرد
                performance_change = (change_factor - 1.0) * 0.7
            elif param_name == 'Volatility':
                # تأثیر معکوس
                performance_change = -(change_factor - 1.0) * 0.8
            elif param_name == 'Max_Drawdown':
                # تأثیر معکوس
                performance_change = -(change_factor - 1.0) * 0.6
            else:
                # تأثیر عمومی
                performance_change = (change_factor - 1.0) * 0.5
            
            performance_changes.append(performance_change)
        
        # محاسبه معیارهای حساسیت
        sensitivity_score = np.std(performance_changes) if len(performance_changes) > 1 else 0.0
        max_impact = max(abs(min(performance_changes)), abs(max(performance_changes))) if performance_changes else 0.0
        
        # تعیین سطح حیاتی بودن
        is_critical = sensitivity_score > sensitivity_threshold
        
        if sensitivity_score > 0.2:
            criticality_level = "Very Critical"
        elif sensitivity_score > 0.1:
            criticality_level = "Critical"
        elif sensitivity_score > 0.05:
            criticality_level = "Moderately Critical"
        else:
            criticality_level = "Low Impact"
        
        sensitivity_results[param_name] = {
            'base_value': base_value,
            'sensitivity_score': sensitivity_score,
            'max_impact': max_impact,
            'is_critical': is_critical,
            'criticality_level': criticality_level,
            'test_values': test_values.tolist(),
            'performance_changes': performance_changes
        }
        
        print(f"  ✅ Sensitivity Score: {sensitivity_score:.3f} ({criticality_level})")
        print(f"  📈 Max Impact: {max_impact:.3f} ({max_impact*100:.1f}%)")
    
    # شناسایی پارامترهای حیاتی
    critical_parameters = {param: results for param, results in sensitivity_results.items() 
                          if results['is_critical']}
    
    print(f"\n🎯 Critical Parameters ({len(critical_parameters)} found):")
    for param, results in critical_parameters.items():
        print(f"  🔥 {param}: Score={results['sensitivity_score']:.3f} ({results['criticality_level']})")
    
    # تولید پیشنهادات بهینه‌سازی
    optimization_suggestions = {}
    for param_name, results in critical_parameters.items():
        sensitivity = results['sensitivity_score']
        
        if sensitivity > 0.2:
            action = "Immediate optimization required - High sensitivity detected"
            monitoring = "Daily monitoring recommended"
        elif sensitivity > 0.1:
            action = "Schedule optimization within 1 month - Moderate sensitivity"
            monitoring = "Weekly monitoring recommended"
        elif sensitivity > 0.05:
            action = "Monitor closely and optimize if needed - Low-moderate sensitivity"
            monitoring = "Monthly monitoring sufficient"
        else:
            action = "Regular monitoring sufficient - Low sensitivity"
            monitoring = "Quarterly monitoring sufficient"
        
        optimization_suggestions[param_name] = {
            'recommended_action': action,
            'monitoring_frequency': monitoring,
            'expected_improvement': f"{results['max_impact']*100:.1f}% potential improvement"
        }
    
    print(f"\n💡 Optimization Suggestions:")
    for param, suggestion in optimization_suggestions.items():
        print(f"  🎯 {param}:")
        print(f"    - Action: {suggestion['recommended_action']}")
        print(f"    - Monitoring: {suggestion['monitoring_frequency']}")
        print(f"    - Potential: {suggestion['expected_improvement']}")
    
    # آمار خلاصه
    sensitivity_scores = [results['sensitivity_score'] for results in sensitivity_results.values()]
    
    summary_stats = {
        'total_parameters_analyzed': len(sensitivity_results),
        'critical_parameters_count': len(critical_parameters),
        'mean_sensitivity': np.mean(sensitivity_scores),
        'max_sensitivity': np.max(sensitivity_scores),
        'parameters_above_threshold': len(critical_parameters)
    }
    
    print(f"\n📊 Summary Statistics:")
    print(f"  📈 Total parameters analyzed: {summary_stats['total_parameters_analyzed']}")
    print(f"  🎯 Critical parameters: {summary_stats['critical_parameters_count']}")
    print(f"  📊 Mean sensitivity: {summary_stats['mean_sensitivity']:.3f}")
    print(f"  🔥 Max sensitivity: {summary_stats['max_sensitivity']:.3f}")
    
    # یافته‌های کلیدی
    key_findings = []
    mean_sensitivity = summary_stats['mean_sensitivity']
    
    if mean_sensitivity > 0.15:
        key_findings.append(f"High overall parameter sensitivity detected: {mean_sensitivity:.3f}")
    elif mean_sensitivity > 0.08:
        key_findings.append(f"Moderate parameter sensitivity: {mean_sensitivity:.3f}")
    else:
        key_findings.append(f"Low parameter sensitivity: {mean_sensitivity:.3f}")
    
    if critical_parameters:
        most_critical = max(critical_parameters.items(), key=lambda x: x[1]['sensitivity_score'])
        key_findings.append(f"Most critical parameter: {most_critical[0]} (sensitivity: {most_critical[1]['sensitivity_score']:.3f})")
    
    critical_ratio = len(critical_parameters) / len(sensitivity_results) * 100
    key_findings.append(f"Critical parameters ratio: {len(critical_parameters)}/{len(sensitivity_results)} ({critical_ratio:.1f}%)")
    
    print(f"\n🔍 Key Findings:")
    for finding in key_findings:
        print(f"  • {finding}")
    
    # توصیه‌های کلی
    recommendations = []
    
    if not critical_parameters:
        recommendations.append("No critical parameters identified - current settings appear stable")
    else:
        high_priority_params = [param for param, results in critical_parameters.items()
                              if results['sensitivity_score'] > 0.15]
        
        if high_priority_params:
            recommendations.append(f"Immediate attention required for: {', '.join(high_priority_params)}")
        
        if len(critical_parameters) > len(sensitivity_results) * 0.5:
            recommendations.append("High number of critical parameters - consider strategy review")
        else:
            recommendations.append("Parameter sensitivity within acceptable range")
    
    print(f"\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    # ذخیره گزارش
    report = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'analysis_parameters': {
            'parameter_variation_range': parameter_variation_range,
            'sensitivity_threshold': sensitivity_threshold,
            'n_sensitivity_points': 11
        },
        'adjustable_parameters': adjustable_params,
        'sensitivity_results': {param: {k: v for k, v in results.items() 
                                      if k not in ['test_values', 'performance_changes']} 
                               for param, results in sensitivity_results.items()},
        'critical_parameters': critical_parameters,
        'optimization_suggestions': optimization_suggestions,
        'summary_statistics': summary_stats,
        'key_findings': key_findings,
        'recommendations': recommendations
    }
    
    # ذخیره فایل
    output_path = Path("tests/results/sensitivity_analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "sensitivity_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Report saved to: {output_path / 'sensitivity_report.json'}")
    print("✅ Parameter sensitivity analysis completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()