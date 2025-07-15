# مستند طراحی سیستم - وب اپلیکیشن هوشمند بهینه‌سازی سبد سهام ایران

## نمای کلی

این سیستم یک پلتفرم جامع برای کشف، اعتبارسنجی و ارائه استراتژی‌های بهینه سرمایه‌گذاری در بازار بورس ایران است. سیستم بر اساس معماری سه لایه طراحی شده که شامل خط لوله داده خودکار، API Contract، و رابط کاربری است.

## معماری کلی

### معماری سطح بالا

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js + TypeScript)              │
│                         🦅 🐺 🐢                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/JSON
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   API Contract Layer                            │
│                 final_results.json                              │
│                   (GitHub Pages)                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │ File I/O
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                Backend Pipeline (Python)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Stage 1   │  │   Stage 2   │  │   Stage 3   │             │
│  │Strategy     │→ │Strategy     │→ │Validator    │             │
│  │Tester       │  │Selector     │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### اجزای اصلی سیستم

1. **خط لوله داده (Data Pipeline)** - موتور پردازش و تحلیل
2. **لایه API (API Contract)** - فایل JSON به عنوان واسط
3. **رابط کاربری (Frontend)** - وب اپلیکیشن تعاملی

## اجزا و رابط‌ها

### 1. خط لوله داده (Backend Pipeline)

#### 1.1 Universe Creator
```python
class UniverseCreator:
    """ایجاد جهان سرمایه‌گذاری از تمام سهام بورس"""
    
    def create_universe(self) -> List[str]:
        # دریافت لیست کامل سهام از tsetmc.com
        # فیلتر بر اساس نقدینگی و کیفیت
        # انتخاب ~700 سهم برتر
        pass
```

#### 1.2 Data Collection Layer
```python
class FullMarketDownloader:
    """دانلود داده‌های قیمتی"""
    cache_duration = 11  # ساعت
    
class FullMarketFundamentalCollector:
    """جمع‌آوری داده‌های بنیادی"""
    cache_duration = 168  # ساعت (هفتگی)
```

#### 1.3 Data Processing Layer
```python
class FullMarketDataPreprocessor:
    """پردازش و محاسبه فاکتورها"""
    
    def calculate_factors(self) -> pd.DataFrame:
        # محاسبه momentum (6M, 12M)
        # محاسبه volatility
        # محاسبه reversal
        # نرمال‌سازی فاکتورها
        pass
```

#### 1.4 Strategy Pipeline (سه مرحله)

##### مرحله 1: Strategy Tester
```python
class StrategyTester:
    """انتخاب و بک‌تست 300 استراتژی برتر"""
    
    def run_backtest_for_selection(self) -> pd.DataFrame:
        # بارگذاری data/strategy_test_results.csv
        # انتخاب 300 استراتژی برتر (Sharpe Ratio)
        # بک‌تست یک ساله
        # انتخاب 100 استراتژی برتر
        # کش 11 ساعته
        pass
```

##### مرحله 2: Strategy Selector
```python
class StrategySelector:
    """دسته‌بندی بر اساس ریسک و انتخاب کاندیداها"""
    
    def select_final_candidates(self, strategies: pd.DataFrame) -> Dict:
        # دسته‌بندی بر اساس Annualized Volatility
        # 30% کم‌ریسک (دفاعی)
        # 40% متعادل
        # 30% پرریسک (تهاجمی)
        # انتخاب 5 کاندیدا از هر دسته
        return {
            'defensive': top_5_defensive,
            'balanced': top_5_balanced,
            'aggressive': top_5_aggressive
        }
```

##### مرحله 3: Validator
```python
class Validator:
    """اعتبارسنجی نهایی و تولید خروجی"""
    
    def validate_selected_strategies(self, candidates: Dict) -> Dict:
        # بک‌تست 5 ساله واقعی‌تر
        # جلوگیری از look-ahead bias
        # لحاظ کردن هزینه‌های معاملاتی
        # انتخاب بهترین از هر دسته
        # تولید final_results.json
        pass
```

### 2. لایه API Contract

#### ساختار فایل final_results.json
```json
{
    "Defensive": {
        "strategy_profile": {
            "name": "لاک‌پشت دانا",
            "icon": "🐢",
            "description": "استراتژی محافظه‌کارانه با ریسک کم"
        },
        "strategy_configuration": {
            "Momentum Period": "12M",
            "Value Weight": 0.1,
            "Momentum Weight": 0.6,
            "Low Volatility Weight": 0.3,
            "Top N": 25,
            "Max Weight": 0.3
        },
        "optimal_weights": {
            "symbol1": 0.15,
            "symbol2": 0.12
        },
        "performance_summary": {
            "Total Return": "205.84%",
            "Annualized Volatility": "11.53%",
            "Annualized Return": "45.34%",
            "Sharpe Ratio": "3.50"
        },
        "backtest_data": {
            "dates": ["2022-07-18", "..."],
            "strategy_values": [100.0, 100.01, "..."],
            "benchmark_values": [100.0, 99.94, "..."]
        },
        "transaction_analysis": {
            "annual_turnover": "51.66%",
            "estimated_total_cost": "0.0216",
            "rebalance_history": [...]
        }
    },
    "Balanced": { /* مشابه ساختار بالا */ },
    "Aggressive": { /* مشابه ساختار بالا */ }
}
```

### 3. رابط کاربری (Frontend)

#### معماری Frontend
```
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js Application                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   🐢        │  │    🐺       │  │    🦅       │             │
│  │ لاک‌پشت دانا │  │ گرگ باتجربه │  │ شاهین تیزبین│             │
│  │ (دفاعی)     │  │ (متعادل)    │  │ (تهاجمی)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Performance Dashboard                      │   │
│  │  • نمودار بازدهی تجمعی                                │   │
│  │  • مقایسه با شاخص کل                                  │   │
│  │  • آنالیز ریسک و بازده                                │   │
│  │  • تاریخچه معاملات                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## مدل‌های داده

### 1. Strategy Configuration
```python
@dataclass
class StrategyConfig:
    momentum_period: str  # "6M" or "12M"
    value_weight: float
    momentum_weight: float
    low_volatility_weight: float
    top_n: int
    max_weight: float
```

### 2. Performance Metrics
```python
@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_volatility: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
```

### 3. Portfolio Weights
```python
@dataclass
class PortfolioWeights:
    weights: Dict[str, float]  # symbol -> weight
    rebalance_date: datetime
    transaction_costs: float
```

## مدیریت خطا

### 1. سطوح خطا
```python
class ErrorLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### 2. استراتژی‌های بازیابی
```python
class RecoveryStrategy:
    def network_error_recovery(self):
        # تلاش مجدد با تأخیر تصاعدی
        # استفاده از cache قدیمی
        pass
    
    def data_missing_recovery(self):
        # استفاده از منابع پشتیبان
        # interpolation داده‌ها
        pass
    
    def critical_error_handling(self):
        # ارسال notification
        # ذخیره state برای بازیابی
        pass
```

### 3. سیستم Logging
```python
class LoggingSystem:
    def __init__(self):
        self.handlers = [
            FileHandler('logs/pipeline.log'),
            StreamHandler(),
            EmailHandler()  # برای خطاهای حیاتی
        ]
```

## استراتژی تست

### 1. Unit Tests
```python
class TestStrategyTester:
    def test_strategy_selection(self):
        # تست انتخاب 300 استراتژی برتر
        pass
    
    def test_backtest_accuracy(self):
        # تست دقت بک‌تست
        pass

class TestStrategySelector:
    def test_risk_categorization(self):
        # تست دسته‌بندی ریسک
        pass

class TestValidator:
    def test_final_validation(self):
        # تست اعتبارسنجی نهایی
        pass
```

### 2. Integration Tests
```python
class TestPipelineIntegration:
    def test_end_to_end_pipeline(self):
        # تست کامل پایپ‌لاین
        pass
    
    def test_cache_system(self):
        # تست سیستم کش
        pass
```

### 3. Performance Tests
```python
class TestPerformance:
    def test_pipeline_execution_time(self):
        # تست زمان اجرای پایپ‌لاین
        pass
    
    def test_memory_usage(self):
        # تست مصرف حافظه
        pass
```

## سیستم کش

### 1. Cache Manager
```python
class CacheManager:
    def __init__(self):
        self.cache_configs = {
            'default': 11 * 3600,  # 11 ساعت
            'fundamental_data': 7 * 24 * 3600,  # هفتگی
        }
    
    def is_cache_valid(self, file_path: str, cache_type: str = 'default') -> bool:
        # بررسی اعتبار کش
        pass
    
    def invalidate_cache(self, pattern: str):
        # باطل کردن کش
        pass
```

### 2. Cache Strategy
- **11 ساعته:** برای اکثر فایل‌های پایپ‌لاین
- **هفتگی:** برای `full_market_fundamental_collector`
- **بر اساس تغییرات:** برای فایل‌های configuration

## امنیت

### 1. Data Security
- رمزگذاری داده‌های حساس
- محدودیت دسترسی به فایل‌ها
- Audit logging برای تمام عملیات

### 2. API Security
- Rate limiting برای درخواست‌ها
- CORS configuration
- Input validation

### 3. Infrastructure Security
- HTTPS برای تمام ارتباطات
- Environment variables برای secrets
- Regular security updates

## نظارت و Monitoring

### 1. System Monitoring
```python
class SystemMonitor:
    def monitor_pipeline_health(self):
        # نظارت بر سلامت پایپ‌لاین
        pass
    
    def track_performance_metrics(self):
        # ردیابی معیارهای عملکرد
        pass
    
    def alert_on_failures(self):
        # هشدار در صورت خرابی
        pass
```

### 2. Business Metrics
- تعداد استراتژی‌های پردازش شده
- زمان اجرای هر مرحله
- نرخ موفقیت پایپ‌لاین
- کیفیت داده‌های تولیدی

## استقرار و DevOps

### 1. Containerization
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

CMD ["python", "run_pipeline.py"]
```

### 2. CI/CD Pipeline
```yaml
# .github/workflows/pipeline.yml
name: Investment Pipeline

on:
  schedule:
    - cron: '0 */12 * * *'  # هر 12 ساعت
  push:
    branches: [main]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Pipeline
        run: python run_pipeline.py
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./results
          publish_branch: gh-pages
```

### 3. Infrastructure as Code
- Terraform برای مدیریت infrastructure
- Docker Compose برای محیط development
- Kubernetes برای production scaling

## مسیرهای توسعه آینده

### 1. تحلیل رژیم بازار
```python
class MarketRegimeAnalyzer:
    def detect_market_regime(self) -> str:
        # تشخیص حالت بازار (صعودی/نزولی/خنثی)
        pass
    
    def adjust_strategy_weights(self, regime: str):
        # تنظیم وزن‌ها بر اساس رژیم بازار
        pass
```

### 2. بهینه‌سازی پویا
```python
class DynamicOptimizer:
    def optimize_with_cvar(self):
        # استفاده از CVaR به جای max_sharpe
        pass
```

### 3. یادگیری ماشین
```python
class MLPredictor:
    def predict_factor_performance(self):
        # پیش‌بینی عملکرد فاکتورها
        pass
```

این طراحی جامع تمام جنبه‌های سیستم شما را پوشش می‌دهد و مسیر واضحی برای پیاده‌سازی فراهم می‌کند.