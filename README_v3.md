# 🇮🇷 Iranian Stock Portfolio Optimizer v3.1

یک سیستم پیشرفته و کاملاً بهینه شده برای تحلیل، غربالگری و بهینه‌سازی پرتفوی سهام ایرانی با استفاده از API جدید pytse-client.

## 🆕 ویژگی‌های جدید v3.1

### 🔧 بهینه‌سازی برای pytse-client
- **سازگاری کامل** با آخرین نسخه pytse-client
- **API های بهینه شده** برای دریافت سریع‌تر داده‌ها
- **مدیریت خطای پیشرفته** برای اتصالات ناپایدار
- **کش چندلایه** برای عملکرد بهتر

### 📊 تحلیل پیشرفته
- **معیارهای ریسک جدید**: Sortino Ratio، VaR 95%، Composite Score
- **تحلیل چندبعدی**: Skewness، Kurtosis، Recent Performance
- **فیلترهای هوشمند**: حذف خودکار داده‌های نامعتبر
- **غربالگری چندمرحله‌ای**: معیارهای سخت‌گیرانه و انعطاف‌پذیر

### 🎯 بهینه‌سازی چندگانه
- **Maximum Sharpe Ratio**: بهترین نسبت ریسک به بازدهی
- **Minimum Volatility**: کمترین ریسک ممکن
- **Maximum Quadratic Utility**: بهینه‌سازی مطلوبیت
- **محدودیت‌های پیشرفته**: کنترل دقیق تخصیص سرمایه

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها
```bash
Python 3.8+
```

### نصب کتابخانه‌ها
```bash
pip install -r requirements_v3.txt
```

### تست سیستم
```bash
python test_v3.py
```

### اجرای تحلیل کامل
```bash
python screener_v3_optimized.py
```

## 📁 ساختار پروژه v3.1

```
Portfolio-Optimizer/
├── screener_v3_optimized.py     # فایل اصلی نسخه 3.1
├── requirements_v3.txt          # کتابخانه‌های مورد نیاز
├── test_v3.py                   # تست جامع سیستم
├── README_v3.md                 # راهنمای نسخه 3.1
├── cache_v3/                    # پوشه کش (خودکار)
│   ├── price_data_v3.feather    # داده‌های قیمت
│   ├── symbols_data_v3.json     # اطلاعات نمادها
│   ├── metadata_v3.json         # متادیتا
│   └── efficient_frontier_v3.png # نمودار
├── portfolio_optimizer.log     # فایل لاگ
└── cache_v3/portfolio_analysis_report_v3_*.txt
```

## ⚙️ تنظیمات پیشرفته

### پارامترهای اصلی:
```python
optimizer = IranianStockOptimizerV3(
    cache_dir='cache_v3',                    # پوشه کش
    years_of_data=5,                         # سال‌های داده
    risk_free_rate=0.35                      # نرخ بدون ریسک
)
```

### معیارهای غربالگری:
```python
optimizer.min_return_threshold = 0.35       # حداقل بازدهی
optimizer.max_volatility_threshold = 0.75   # حداکثر نوسان
optimizer.min_data_points = 200             # حداقل نقاط داده
optimizer.max_position_size = 0.25          # حداکثر سهم هر سهم
optimizer.min_market_cap_percentile = 20    # فیلتر ارزش بازار
```

### تنظیمات pytse-client:
```python
optimizer.adjust_prices = True              # تعدیل قیمت‌ها
optimizer.include_dividends = True          # شامل سود سهام
```

## 📊 خروجی‌های پیشرفته

### 1. نمودار Efficient Frontier v3
- **کیفیت بالا**: 300 DPI، فرمت PNG
- **نمایش چندلایه**: نقاط بهینه، دارایی‌ها، مرز کارآمد
- **طراحی حرفه‌ای**: فونت‌ها و رنگ‌های بهینه

### 2. گزارش جامع
```
IRANIAN STOCK PORTFOLIO OPTIMIZATION REPORT v3.1
================================================
Generated: 2025-01-07 19:35:00
Risk-free rate: 35.00%
Analysis period: 5 years
Optimization engine: pytse-client optimized

Total stocks analyzed: 180
Date range: 2020-01-07 to 2025-01-07
Total trading days: 1,250

TOP CANDIDATES FOR OPTIMIZATION:
Symbol    Return  Volatility  Sharpe  Sortino  Composite_Score
فولاد     0.4250    0.3200    0.2344   0.3120      0.8450
شپنا      0.3890    0.2950    0.1322   0.2890      0.7890
...

DETAILED STOCK ANALYSIS:
فولاد:
  Annual Return: 42.50%
  Volatility: 32.00%
  Sharpe Ratio: 0.23
  Max Drawdown: -25.30%
  Composite Score: 0.845
```

### 3. سه استراتژی بهینه‌سازی

#### 🎯 Maximum Sharpe Ratio
```
📊 Optimal Weights (Max Sharpe):
   فولاد: 18.50%
   شپنا: 15.20%
   خودرو: 12.80%
   ...
📈 Expected Return: 38.20%
📉 Volatility: 28.50%
📊 Sharpe Ratio: 0.11
```

#### 🛡️ Minimum Volatility
```
📊 Optimal Weights (Min Volatility):
   شستا: 22.10%
   فارس: 18.90%
   وبملت: 16.40%
   ...
📈 Expected Return: 32.10%
📉 Volatility: 22.80%
📊 Sharpe Ratio: -0.13
```

#### ⚖️ Maximum Quadratic Utility
```
📊 Optimal Weights (Max Quadratic Utility):
   شپنا: 20.30%
   فولاد: 17.60%
   تاپیکو: 14.20%
   ...
📈 Expected Return: 36.80%
📉 Volatility: 26.20%
📊 Sharpe Ratio: 0.07
```

## 🔧 عیب‌یابی پیشرفته

### مشکلات رایج و راه‌حل‌ها:

#### خطای اتصال pytse-client:
```
❌ Could not fetch symbols: HTTPSConnectionPool...
```
**راه‌حل:**
1. بررسی اتصال اینترنت
2. اجرای مجدد با تاخیر: `time.sleep(2)`
3. استفاده از VPN در صورت نیاز

#### داده ناکافی:
```
❌ No stocks passed strict criteria, relaxing filters...
```
**راه‌حل خودکار:** سیستم به طور خودکار معیارها را آسان‌تر می‌کند:
```python
# معیارهای آسان‌تر
min_return_threshold *= 0.7
max_volatility_threshold *= 1.2
min_data_points *= 0.8
```

#### مشکل کش:
```
⚠️ Corrupted cache file removed
```
**راه‌حل:** سیستم خودکار کش خراب را حذف و داده جدید دانلود می‌کند.

### تنظیمات عیب‌یابی:
```python
# فعال‌سازی لاگ تفصیلی
logging.basicConfig(level=logging.DEBUG)

# تست با داده کمتر
optimizer.years_of_data = 2
optimizer.min_data_points = 100

# معیارهای آسان‌تر
optimizer.min_return_threshold = 0.15
optimizer.max_volatility_threshold = 0.90
```

## 📈 تفسیر نتایج پیشرفته

### معیارهای جدید:

#### Composite Score:
- **> 0.8**: عالی
- **0.6-0.8**: خوب
- **< 0.6**: متوسط

#### Sortino Ratio:
- **> 1.0**: مدیریت ریسک عالی
- **0.5-1.0**: قابل قبول
- **< 0.5**: ضعیف

#### VaR 95%:
- **> -20%**: ریسک کم
- **-20% to -40%**: ریسک متوسط
- **< -40%**: ریسک بالا

### نکات مهم v3.1:
- ⚠️ **کش هوشمند**: داده‌ها هر 24 ساعت بروزرسانی می‌شوند
- ⚠️ **تحلیل چندبعدی**: ترکیب چندین معیار برای انتخاب بهتر
- ⚠️ **بهینه‌سازی تطبیقی**: تنظیم خودکار پارامترها بر اساس داده‌های موجود

## 🔄 مقایسه نسخه‌ها

| ویژگی | v2.0 | v3.1 |
|--------|------|------|
| pytse-client API | پایه | بهینه شده |
| معیارهای ریسک | 4 | 8+ |
| استراتژی بهینه‌سازی | 2 | 3 |
| سیستم کش | ساده | چندلایه |
| مدیریت خطا | پایه | پیشرفته |
| گزارش‌گیری | متنی | جامع + بصری |
| سرعت اجرا | متوسط | سریع |

## 🚀 عملکرد و بهینه‌سازی

### بهبودهای سرعت:
- **دانلود batch**: پردازش گروهی داده‌ها
- **کش چندلایه**: ذخیره هوشمند اطلاعات
- **فیلترهای پیش‌پردازش**: حذف سریع داده‌های نامعتبر

### استفاده از حافظه:
- **پردازش تدریجی**: جلوگیری از اشغال حافظه زیاد
- **پاکسازی خودکار**: حذف داده‌های غیرضروری
- **فرمت بهینه**: استفاده از Feather برای ذخیره سریع

## 📞 پشتیبانی

### فایل‌های لاگ:
1. `portfolio_optimizer.log` - لاگ کامل اجرا
2. `cache_v3/metadata_v3.json` - اطلاعات کش
3. گزارش‌های تفصیلی در پوشه `cache_v3/`

### تست سیستم:
```bash
python test_v3.py
```

### حل مشکل گام به گام:
1. اجرای تست: `python test_v3.py`
2. بررسی لاگ: `tail -f portfolio_optimizer.log`
3. پاک کردن کش: `rm -rf cache_v3/`
4. اجرای مجدد با پارامترهای آسان‌تر

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

---

**⚠️ تذکر مهم**: این ابزار صرفاً برای تحلیل و آموزش است. قبل از هرگونه سرمایه‌گذاری با مشاور مالی مشورت کنید.

**🔧 نسخه 3.1**: کاملاً بهینه شده برای pytse-client API با قابلیت‌های پیشرفته تحلیل و بهینه‌سازی.