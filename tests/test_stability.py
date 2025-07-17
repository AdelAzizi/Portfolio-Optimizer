#!/usr/bin/env python3
import sys
import logging

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from strategy_stability_analyzer import StrategyStabilityAnalyzer
    
    print("🧪 Testing StrategyStabilityAnalyzer...")
    
    # تنظیمات تست
    config = {
        'rolling_window': 252,
        'stability_threshold': 0.2,
        'consistency_threshold': 0.7
    }
    
    # ایجاد analyzer
    analyzer = StrategyStabilityAnalyzer(config)
    
    # اجرای تحلیل کامل
    results = analyzer.run_complete_analysis()
    
    print("✅ StrategyStabilityAnalyzer test completed successfully!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()