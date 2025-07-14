# Data

## 1. Purpose of This Directory
This directory contains all the data used in the portfolio optimization project, including benchmark data, fundamental data, historical price data, and strategy test results.

## 2. File Breakdown

### 📄 benchmark_data_v3.csv
**Purpose:** Contains historical data for the market benchmark.
**Key Components & Logic:** This file stores daily benchmark data, including date, high, low, open, close, volume, and Jalali date.

### 📄 fundamental_data.csv
**Purpose:** Contains fundamental data for various stock symbols.
**Key Components & Logic:** This file includes metrics like P/E ratio, P/S ratio, EPS, and market cap for different stocks.

### 📄 strategy_test_results.csv
**Purpose:** Contains the results of different strategy backtests.
**Key Components & Logic:** This file stores the performance metrics of various investment strategies, including total return, annualized volatility, and Sharpe ratio.

### 📁 full_market_data_csvs/
**Purpose:** This directory contains individual CSV files for each stock symbol, with detailed historical price and volume data.
**How to Use:** The data in this directory is used by the preprocessor to create a master dataset for analysis.

## 3. Dependencies & Interactions
The data in this directory is used by various scripts in the `src/` directory, including the preprocessor, optimizer, and strategy tester. The output of these scripts, such as backtest results, is also stored in this directory.

---

# داده‌ها

## ۱. هدف این پوشه
این پوشه شامل تمام داده‌های مورد استفاده در پروژه بهینه‌سازی پورتفولیو است، از جمله داده‌های شاخص، داده‌های بنیادی، داده‌های تاریخی قیمت و نتایج تست استراتژی‌ها.

## ۲. تفکیک فایل‌ها

### 📄 benchmark_data_v3.csv
**هدف:** شامل داده‌های تاریخی شاخص بازار است.
**اجزای اصلی و منطق:** این فایل داده‌های روزانه شاخص، شامل تاریخ، قیمت بالا، پایین، باز و بسته شدن، حجم معاملات و تاریخ جلالی را ذخیره می‌کند.

### 📄 fundamental_data.csv
**هدف:** شامل داده‌های بنیادی برای نمادهای مختلف سهام است.
**اجزای اصلی و منطق:** این فایل شامل معیارهایی مانند نسبت قیمت به درآمد، نسبت قیمت به فروش، سود هر سهم و ارزش بازار برای سهام‌های مختلف است.

### 📄 strategy_test_results.csv
**هدف:** شامل نتایج بک‌تست استراتژی‌های مختلف است.
**اجزای اصلی و منطق:** این فایل معیارهای عملکرد استراتژی‌های سرمایه‌گذاری مختلف، از جمله بازده کل، نوسان سالانه و نسبت شارپ را ذخیره می‌کند.

### 📁 full_market_data_csvs/
**هدف:** این پوشه شامل فایل‌های CSV جداگانه برای هر نماد سهام است که حاوی داده‌های دقیق تاریخی قیمت و حجم معاملات می‌باشد.
**نحوه استفاده:** داده‌های این پوشه توسط پیش‌پردازنده برای ایجاد مجموعه داده اصلی برای تحلیل استفاده می‌شود.

## ۳. وابستگی‌ها و تعاملات
داده‌های این پوشه توسط اسکریپت‌های مختلف در پوشه `src/` از جمله پیش‌پردازنده، بهینه‌ساز و تستر استراتژی استفاده می‌شود. خروجی این اسکریپت‌ها، مانند نتایج بک‌تست، نیز در همین پوشه ذخیره می‌شود.