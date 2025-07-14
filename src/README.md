# src

## 1. Purpose of This Directory
This directory contains all the core Python source code for the portfolio optimization pipeline, from data collection and preprocessing to optimization, backtesting, and validation.

## 2. File Breakdown

### 📄 config.py
**Purpose:** This file centralizes all global configuration parameters for the project.
**Key Components & Logic:** It defines constants for data paths, API settings, and model parameters such as risk-free rate, factor weights, and strategy configurations. This allows for easy adjustments without hardcoding values in other scripts.
**How to Use:** This file is not run directly. It is imported by other scripts in the `src` directory to access configuration variables.

### 📄 universe_creator.py
**Purpose:** This script is responsible for creating the initial stock universe by fetching all available symbols and applying a series of quantitative filters.
**Key Components & Logic:**
*   **Class: `UniverseCreator`**: Manages the process of fetching and filtering the stock universe.
    *   **Method: `run()`**: Fetches market-wide stats using `pytse_client`, applies filters for market type (Bourse/Fara Bourse), liquidity (minimum volume), and size (minimum market cap), and saves the final list of symbols to `cache/universe.json`.
**How to Use:** Run this script directly from the project root (`python src/universe_creator.py`) to generate or update the list of stocks to be analyzed.

### 📄 full_market_downloader.py
**Purpose:** This script downloads and caches historical price data for all symbols defined in the universe file.
**Key Components & Logic:**
*   **Class: `FullMarketDownloader`**: Manages the download and update process.
    *   **Method: `run_update()`**: Loads the symbol list from `cache/universe.json`, then iterates through each symbol. It checks if a recent CSV file already exists in `data/full_market_data_csvs/`. If not, or if the file is outdated, it fetches the full price history using `pytse_client` and saves it as a new CSV.
**How to Use:** This script is run directly (`python src/full_market_downloader.py`) after `universe_creator.py` to ensure all necessary price data is available locally.

### 📄 full_market_fundamental_collector.py
**Purpose:** This script collects, validates, and cleans fundamental data for the entire market universe, intelligently caching results to minimize redundant API calls.
**Key Components & Logic:**
*   **Class: `FullMarketFundamentalCollector`**: Orchestrates the data collection pipeline.
    *   **Method: `run_collection()`**: Loads the full universe, checks for an existing cache (`cache/master_fundamental_data.feather`), and identifies symbols that need updating. It fetches fundamental data (P/E, P/S, EPS) for new symbols, validates the data, and adds persistently failing symbols to a blacklist.
    *   **Method: `_impute_missing_data()`**: Fills missing numerical data using industry group means and then overall medians.
    *   **Method: `_handle_outliers()`**: Uses winsorization to cap extreme outliers in P/E and P/S ratios at the 1st and 99th percentiles.
**How to Use:** Run this script directly (`python src/full_market_fundamental_collector.py`) after the downloader to gather the fundamental data needed for factor analysis.

### 📄 full_market_preprocessor.py
**Purpose:** This script integrates the clean fundamental data with the raw price data to build a single, final, analysis-ready dataset.
**Key Components & Logic:**
*   **Class: `FullMarketDataPreprocessor`**: Manages the data merging and feature engineering process.
    *   **Method: `run()`**: Loads the fundamental data and all individual price CSVs. It calculates quantitative metrics like annualized return, volatility, and multi-period momentum (3M, 6M, 12M) from the price data. Finally, it joins the fundamental and quantitative data into a single master DataFrame and saves it as `cache/full_analysis_ready_data.feather`.
**How to Use:** This script is run directly (`python src/full_market_preprocessor.py`) after the fundamental collector. It prepares the final dataset used by the optimizer and strategy tester.

### 📄 optimizer.py
**Purpose:** This script acts as the core analysis and optimization engine, using preprocessed data to screen stocks based on a multi-factor model and then running portfolio optimization and backtesting.
**Key Components & Logic:**
*   **Class: `MultiFactorOptimizer`**: Encapsulates the logic for screening, optimization, and backtesting.
    *   **Method: `screen_stocks()`**: Ranks stocks based on a weighted composite score of Value, Momentum, and Low Volatility factors.
    *   **Method: `run_rolling_backtest()`**: Performs a walk-forward backtest by periodically rebalancing the portfolio based on the latest screened stocks. It simulates transaction costs and calculates turnover.
    *   **Method: `run_full_analysis()`**: Orchestrates the entire process for a given strategy configuration, returning final weights, performance summaries, and backtest data.
**How to Use:** This script is not meant to be run directly. It is imported and its main class is instantiated by `strategy_tester.py` and `validator.py` to analyze different strategies.

### 📄 strategy_tester.py
**Purpose:** This script systematically tests a wide range of strategy configurations to find the best-performing ones based on historical data.
**Key Components & Logic:**
*   **Function: `re_evaluate_top_strategies()`**: This is the main function. It loads a pre-existing list of strategy test results, filters for the top 200 performers by Sharpe Ratio, and then re-runs the `MultiFactorOptimizer` on just these top candidates to get a more robust performance measure. The results are cached.
**How to Use:** This script is run directly (`python src/strategy_tester.py`) to identify the most promising strategy configurations for further out-of-sample validation.

### 📄 validator.py
**Purpose:** This script provides a final, robust, out-of-sample validation of the best strategies identified by the `strategy_tester.py`.
**Key Components & Logic:**
*   **Function: `validate_and_select_best_strategies()`**: Takes the top 200 re-evaluated strategies, categorizes them into "Defensive," "Balanced," and "Aggressive" based on their volatility, and selects the top 5 from each category. It then runs the `MultiFactorOptimizer`'s walk-forward backtest on these 15 candidates to make a final selection.
**How to Use:** This is the final validation step. Run it directly (`python src/validator.py`) to get the final, recommended strategies for each risk profile.

## 3. Dependencies & Interactions
- **`config.py`**: Provides configuration to all other scripts in this directory.
- **`universe_creator.py`**: Output (`cache/universe.json`) is the primary input for `full_market_downloader.py`.
- **`full_market_downloader.py`**: Output (CSV files in `data/full_market_data_csvs/`) is a key input for `full_market_preprocessor.py`.
- **`full_market_fundamental_collector.py`**: Output (`cache/master_fundamental_data.feather`) is a key input for `full_market_preprocessor.py`.
- **`full_market_preprocessor.py`**: Output (`cache/full_analysis_ready_data.feather`) is the primary input for `optimizer.py`.
- **`optimizer.py`**: Is a library/module used by `strategy_tester.py` and `validator.py`.
- **`strategy_tester.py`**: Uses `optimizer.py` and its output (`cache/strategy_test_results.csv`) is the input for `validator.py`.
- **`validator.py`**: Uses `optimizer.py` and the results from `strategy_tester.py` to produce the final strategy recommendations.

---

# src

## ۱. هدف این پوشه
این پوشه شامل تمام کدهای اصلی پایتون برای خط لوله بهینه‌سازی پورتفولیو است، از جمع‌آوری و پیش‌پردازش داده‌ها گرفته تا بهینه‌سازی، بک‌تست و اعتبارسنجی.

## ۲. تفکیک فایل‌ها

### 📄 config.py
**هدف:** این فایل تمام پارامترهای پیکربندی سراسری پروژه را متمرکز می‌کند.
**اجزای اصلی و منطق:** این فایل ثابت‌هایی را برای مسیرهای داده، تنظیمات API و پارامترهای مدل مانند نرخ بهره بدون ریسک، وزن فاکتورها و پیکربندی‌های استراتژی تعریف می‌کند. این کار امکان تنظیمات آسان را بدون نیاز به تغییر مقادیر در اسکریپت‌های دیگر فراهم می‌کند.
**نحوه استفاده:** این فایل به طور مستقیم اجرا نمی‌شود. توسط اسکریپت‌های دیگر در پوشه `src` برای دسترسی به متغیرهای پیکربندی ایمپورت می‌شود.

### 📄 universe_creator.py
**هدف:** این اسکریپت مسئول ایجاد جهان سهام اولیه با دریافت تمام نمادهای موجود و اعمال یک سری فیلترهای کمی است.
**اجزای اصلی و منطق:**
*   **کلاس: `UniverseCreator`**: فرآیند دریافت و فیلتر کردن جهان سهام را مدیریت می‌کند.
    *   **متد: `run()`**: آمار کلی بازار را با استفاده از `pytse_client` دریافت می‌کند، فیلترهایی را برای نوع بازار (بورس/فرابورس)، نقدشوندگی (حداقل حجم) و اندازه (حداقل ارزش بازار) اعمال می‌کند و لیست نهایی نمادها را در `cache/universe.json` ذخیره می‌کند.
**نحوه استفاده:** این اسکریپت را مستقیماً از ریشه پروژه اجرا کنید (`python src/universe_creator.py`) تا لیست سهام مورد تحلیل تولید یا به‌روزرسانی شود.

### 📄 full_market_downloader.py
**هدف:** این اسکریپت داده‌های تاریخی قیمت را برای تمام نمادهای تعریف شده در فایل جهان دانلود و کش می‌کند.
**اجزای اصلی و منطق:**
*   **کلاس: `FullMarketDownloader`**: فرآیند دانلود و به‌روزرسانی را مدیریت می‌کند.
    *   **متد: `run_update()`**: لیست نمادها را از `cache/universe.json` بارگیری می‌کند، سپس در هر نماد پیمایش می‌کند. بررسی می‌کند که آیا یک فایل CSV جدید در `data/full_market_data_csvs/` وجود دارد یا خیر. اگر وجود نداشته باشد یا فایل قدیمی باشد، تاریخچه کامل قیمت را با استفاده از `pytse_client` دریافت کرده و آن را به عنوان یک فایل CSV جدید ذخیره می‌کند.
**نحوه استفاده:** این اسکریپت پس از `universe_creator.py` مستقیماً اجرا می‌شود (`python src/full_market_downloader.py`) تا اطمینان حاصل شود که تمام داده‌های قیمت لازم به صورت محلی در دسترس هستند.

### 📄 full_market_fundamental_collector.py
**هدف:** این اسکریپت داده‌های بنیادی را برای کل جهان بازار جمع‌آوری، اعتبارسنجی و پاک‌سازی می‌کند و به طور هوشمند از نتایج کش شده برای به حداقل رساندن فراخوانی‌های API تکراری استفاده می‌کند.
**اجزای اصلی و منطق:**
*   **کلاس: `FullMarketFundamentalCollector`**: خط لوله جمع‌آوری داده را هماهنگ می‌کند.
    *   **متد: `run_collection()`**: جهان کامل را بارگیری می‌کند، کش موجود (`cache/master_fundamental_data.feather`) را بررسی می‌کند و نمادهایی را که نیاز به به‌روزرسانی دارند شناسایی می‌کند. داده‌های بنیادی (P/E، P/S، EPS) را برای نمادهای جدید دریافت می‌کند، داده‌ها را اعتبارسنجی می‌کند و نمادهایی که به طور مداوم با شکست مواجه می‌شوند را به لیست سیاه اضافه می‌کند.
    *   **متد: `_impute_missing_data()`**: داده‌های عددی گمشده را با استفاده از میانگین گروه صنعتی و سپس میانه کلی پر می‌کند.
    *   **متد: `_handle_outliers()`**: از winsorization برای محدود کردن داده‌های پرت شدید در نسبت‌های P/E و P/S در صدک‌های اول و نود و نهم استفاده می‌کند.
**نحوه استفاده:** این اسکریپت پس از جمع‌آوری‌کننده داده‌های بنیادی مستقیماً اجرا می‌شود (`python src/full_market_fundamental_collector.py`) تا داده‌های بنیادی مورد نیاز برای تحلیل فاکتورها را جمع‌آوری کند.

### 📄 full_market_preprocessor.py
**هدف:** این اسکریپت داده‌های بنیادی پاک‌سازی شده را با داده‌های خام قیمت ادغام می‌کند تا یک مجموعه داده واحد و نهایی آماده برای تحلیل ایجاد کند.
**اجزای اصلی و منطق:**
*   **کلاس: `FullMarketDataPreprocessor`**: فرآیند ادغام داده‌ها و مهندسی ویژگی‌ها را مدیریت می‌کند.
    *   **متد: `run()`**: داده‌های بنیادی و تمام فایل‌های CSV قیمت فردی را بارگیری می‌کند. معیارهای کمی مانند بازده سالانه، نوسان و مومنتوم چند دوره‌ای (۳ ماهه، ۶ ماهه، ۱۲ ماهه) را از داده‌های قیمت محاسبه می‌کند. در نهایت، داده‌های بنیادی و کمی را در یک DataFrame اصلی ادغام کرده و آن را به عنوان `cache/full_analysis_ready_data.feather` ذخیره می‌کند.
**نحوه استفاده:** این اسکریپت پس از جمع‌آوری‌کننده بنیادی مستقیماً اجرا می‌شود (`python src/full_market_preprocessor.py`). این اسکریپت مجموعه داده نهایی مورد استفاده توسط بهینه‌ساز و تستر استراتژی را آماده می‌کند.

### 📄 optimizer.py
**هدف:** این اسکریپت به عنوان موتور اصلی تحلیل و بهینه‌سازی عمل می‌کند و با استفاده از داده‌های پیش‌پردازش شده، سهام را بر اساس یک امتیاز ترکیبی از فاکتورهای ارزش، مومنتوم و نوسان پایین انتخاب کرده و سپس پورتفولیو حاصل را بک‌تست می‌کند.
**اجزای اصلی و منطق:**
*   **کلاس: `MultiFactorOptimizer`**: منطق غربالگری، بهینه‌سازی و بک‌تست را در بر می‌گیرد.
    *   **متد: `screen_stocks()`**: سهام را بر اساس یک امتیاز ترکیبی وزنی از فاکتورهای ارزش، مومنتوم و نوسان پایین رتبه‌بندی می‌کند.
    *   **متد: `run_rolling_backtest()`**: یک بک‌تست پیشرو (walk-forward) را با متعادل‌سازی دوره‌ای پورتفولیو بر اساس آخرین سهام غربال شده انجام می‌دهد. این متد هزینه‌های معامله و گردش پرتفوی را شبیه‌سازی می‌کند.
    *   **متد: `run_full_analysis()`**: کل فرآیند را برای یک پیکربندی استراتژی مشخص هماهنگ می‌کند و وزن‌های نهایی، خلاصه‌های عملکرد و داده‌های بک‌تست را برمی‌گرداند.
**نحوه استفاده:** این اسکریپت برای اجرای مستقیم طراحی نشده است. توسط `strategy_tester.py` و `validator.py` برای تحلیل استراتژی‌های مختلف ایمپورت و کلاس اصلی آن نمونه‌سازی می‌شود.

### 📄 strategy_tester.py
**هدف:** این اسکریپت به طور سیستماتیک طیف گسترده‌ای از پیکربندی‌های استراتژی را برای یافتن بهترین عملکرد بر اساس داده‌های تاریخی آزمایش می‌کند.
**اجزای اصلی و منطق:**
*   **تابع: `re_evaluate_top_strategies()`**: این تابع اصلی است. لیستی از نتایج تست استراتژی‌های از پیش موجود را بارگیری می‌کند، ۲۰۰ مورد برتر را بر اساس نسبت شارپ فیلتر می‌کند و سپس `MultiFactorOptimizer` را فقط برای این نامزدهای برتر مجدداً اجرا می‌کند تا یک معیار عملکرد قوی‌تر به دست آورد. نتایج کش می‌شوند.
**نحوه استفاده:** این اسکریپت مستقیماً اجرا می‌شود (`python src/strategy_tester.py`) تا امیدوارکننده‌ترین پیکربندی‌های استراتژی برای اعتبارسنجی بیشتر خارج از نمونه شناسایی شوند.

### 📄 validator.py
**هدف:** این اسکریپت یک اعتبارسنجی نهایی، قوی و خارج از نمونه از بهترین استراتژی‌های شناسایی شده توسط `strategy_tester.py` ارائه می‌دهد.
**اجزای اصلی و منطق:**
*   **تابع: `validate_and_select_best_strategies()`**: ۲۰۰ استراتژی برتر ارزیابی شده را گرفته، آنها را بر اساس نوسانات به دسته‌های "دفاعی"، "متعادل" و "تهاجمی" طبقه‌بندی می‌کند و ۵ مورد برتر از هر دسته را انتخاب می‌کند. سپس بک‌تست پیشرو `MultiFactorOptimizer` را روی این ۱۵ نامزد اجرا می‌کند تا انتخاب نهایی را انجام دهد.
**نحوه استفاده:** این مرحله نهایی اعتبارسنجی است. آن را مستقیماً اجرا کنید (`python src/validator.py`) تا استراتژی‌های نهایی توصیه‌شده برای هر پروفایل ریسک را دریافت کنید.

## ۳. وابستگی‌ها و تعاملات
- **`config.py`**: پیکربندی را برای تمام اسکریپت‌های دیگر در این پوشه فراهم می‌کند.
- **`universe_creator.py`**: خروجی آن (`cache/universe.json`) ورودی اصلی برای `full_market_downloader.py` است.
- **`full_market_downloader.py`**: خروجی آن (فایل‌های CSV در `data/full_market_data_csvs/`) یک ورودی کلیدی برای `full_market_preprocessor.py` است.
- **`full_market_fundamental_collector.py`**: خروجی آن (`cache/master_fundamental_data.feather`) یک ورودی کلیدی برای `full_market_preprocessor.py` است.
- **`full_market_preprocessor.py`**: خروجی آن (`cache/full_analysis_ready_data.feather`) ورودی اصلی برای `optimizer.py` است.
- **`optimizer.py`**: به عنوان یک کتابخانه/ماژول توسط `strategy_tester.py` و `validator.py` استفاده می‌شود.
- **`strategy_tester.py`**: از `optimizer.py` استفاده می‌کند و خروجی آن (`cache/strategy_test_results.csv`) ورودی برای `validator.py` است.
- **`validator.py`**: از `optimizer.py` و نتایج `strategy_tester.py` برای تولید توصیه‌های نهایی استراتژی استفاده می‌کند.