# ==============================================================================
# FINAL, ROBUST, AND CORRECTED SCRIPT (WITH JALALI TO GREGORIAN CONVERSION)
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import finpy_tse as fpy
import jdatetime  # کتابخانه جدید برای تبدیل تاریخ

print("=== Starting Data Analysis and Visualization (Final Corrected Version) ===")

# --- 1. Define Tickers ---
iran_symbols = ['فولاد', 'فارس', 'شپنا', 'خودرو', 'شستا']
print(f"Target symbols: {iran_symbols}")

# --- 2. Fetch and Process Historical Data ---
list_of_dfs = []
for symbol in iran_symbols:
    print(f"Fetching data for: {symbol}...")
    try:
        df = fpy.Get_Price_History(stock=symbol)
        
        if df is not None and not df.empty:
            # >> کلید حل مشکل اینجاست <<
            # ما ایندکس شمسی (که به صورت رشته است) را به تاریخ میلادی تبدیل می‌کنیم
            gregorian_dates = []
            for j_date_str in df.index:
                y, m, d = map(int, j_date_str.split('-'))
                g_date = jdatetime.date(y, m, d).togregorian()
                gregorian_dates.append(g_date)
            
            # ایندکس دیتافریم را با تاریخ‌های میلادی جدید جایگزین می‌کنیم
            df.index = pd.to_datetime(gregorian_dates)
            
            # فقط ستون قیمت پایانی را با نام نماد جدا می‌کنیم
            df_close = df[['Close']].rename(columns={'Close': symbol})
            list_of_dfs.append(df_close)
            print(f"  > Success! Data for {symbol} converted and processed.")
        else:
            print(f"  > Warning: No data returned for {symbol}.")
            
    except Exception as e:
        print(f"  > ERROR fetching data for {symbol}: {e}")

# --- 3. Combine, Clean, and Analyze ---
if not list_of_dfs:
    print("\nFATAL ERROR: No data was fetched for any symbol. Exiting.")
    exit()

print("\nCombining all data into a single DataFrame...")
all_close_prices = pd.concat(list_of_dfs, axis=1)
all_close_prices.sort_index(inplace=True)
all_close_prices.ffill(inplace=True)
all_close_prices.bfill(inplace=True)

print("\n--- Final DataFrame of Close Prices (Head) ---")
print(all_close_prices.head())

# --- 4. Calculate Daily Returns ---
daily_returns = all_close_prices.pct_change().dropna()

# --- 5. Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
print("\nPlotting historical prices...")
all_close_prices.plot(figsize=(15, 7), title="Historical Close Prices (Non-Adjusted)")
plt.xlabel("Date (Gregorian)")
plt.ylabel("Close Price")
plt.show()

print("Plotting daily returns...")
daily_returns.plot(figsize=(15, 7), linestyle='none', marker='.', title="Daily Returns Volatility")
plt.xlabel("Date (Gregorian)")
plt.ylabel("Daily Return")
plt.axhline(0, color='grey', linestyle='--')
plt.show()

# --- 6. Key Statistics ---
risk_return_summary = pd.DataFrame({
    'Mean_Daily_Return': daily_returns.mean(),
    'Daily_Volatility': daily_returns.std()
})
print("\n--- Risk-Return Summary ---")
print(risk_return_summary)

print("\n\n================================================")
print("=== WEEK 1 MISSION ACCOMPLISHED SUCCESSFULLY ===")
print("================================================")
