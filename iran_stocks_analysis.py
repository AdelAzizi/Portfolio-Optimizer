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

# ==============================================================================
# PORTFOLIO PERFORMANCE CALCULATION (MODERN PORTFOLIO THEORY)
# ==============================================================================

# --- 7. Import NumPy for Numerical Operations ---
import numpy as np

# --- 8. Define Portfolio Weights ---
print("\n--- 8. Defining Portfolio Weights ---")
num_stocks = len(iran_symbols)
# For this example, we'll use equal weights
weights = np.array([1/num_stocks] * num_stocks)

print("Stock Symbols:", iran_symbols)
print("Portfolio Weights:", weights)
print("Sum of Weights:", np.sum(weights)) # Should be 1.0

# --- 9. Calculate Expected Annual Portfolio Return ---
print("\n--- 9. Calculating Expected Annual Portfolio Return ---")
mean_daily_returns = daily_returns.mean()
annualized_returns = mean_daily_returns * 252
portfolio_return = np.dot(annualized_returns, weights)

print(f"Expected Annual Portfolio Return: {portfolio_return:.2%}")

# --- 10. Calculate Annual Portfolio Volatility (Risk) ---
print("\n--- 10. Calculating Annual Portfolio Volatility (Risk) ---")
# The covariance matrix is crucial because it measures how stock returns move together.
# We can't just average individual volatilities because that ignores diversification benefits.
# If stocks move in opposite directions (negative covariance), the overall portfolio risk is reduced.
cov_matrix_daily = daily_returns.cov()
cov_matrix_annual = cov_matrix_daily * 252

portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

print(f"Annual Portfolio Volatility (Risk): {portfolio_volatility:.2%}")

# --- 11. Encapsulate into a Reusable Function ---
print("\n--- 11. Encapsulating Logic into a Reusable Function ---")

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculates the annualized return and volatility of a portfolio.
    
    Args:
        weights (np.array): The weights of the assets in the portfolio.
        mean_returns (pd.Series): The mean daily returns of the assets.
        cov_matrix (pd.DataFrame): The daily covariance matrix of returns.
        
    Returns:
        tuple: A tuple containing (annual_portfolio_return, annual_portfolio_volatility).
    """
    # Annualize return and covariance
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

# --- 12. Test the Reusable Function ---
print("\n--- 12. Testing the Reusable Function ---")
# We use the original daily returns and covariance for the function
portfolio_return_func, portfolio_volatility_func = calculate_portfolio_performance(
    weights,
    daily_returns.mean(),
    daily_returns.cov()
)

print(f"Function-Calculated Annual Return: {portfolio_return_func:.2%}")
print(f"Function-Calculated Annual Volatility: {portfolio_volatility_func:.2%}")

print("\n===============================================")
print("=== PORTFOLIO ANALYSIS SUCCESSFULLY ADDED ===")
print("===============================================")

# ==============================================================================
# MONTE CARLO SIMULATION FOR PORTFOLIO OPTIMIZATION
# ==============================================================================

# --- 13. Initialization for Simulation ---
print("\n--- 13. Initializing Monte Carlo Simulation ---")
num_portfolios = 25000
# Risk-free rate for Sharpe Ratio calculation. We assume 0 for simplicity.
# In a real-world scenario, this would be the yield on a government bond (e.g., US Treasury bill).
risk_free_rate = 0.0

# Lists to store the results of each simulated portfolio
results = [] 
weights_list = []

# --- 14. The Simulation Loop ---
print(f"Running {num_portfolios} simulations...")
for i in range(num_portfolios):
    # a. Generate Random Weights
    random_weights = np.random.random(num_stocks)
    # Normalize weights to ensure their sum is 1.0
    random_weights /= np.sum(random_weights)
    
    # b. Store Weights
    weights_list.append(random_weights)
    
    # c. Calculate Performance
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(
        random_weights, daily_returns.mean(), daily_returns.cov()
    )
    
    # d. Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # e. Store Results
    results.append((portfolio_return, portfolio_volatility, sharpe_ratio))

print("Simulation complete.")

# --- 15. Data Post-Processing ---
print("\n--- 15. Processing Simulation Results ---")
# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'SharpeRatio'])
# Add the weights of each portfolio to the DataFrame
results_df['Weights'] = weights_list

# --- 16. Identify Key Portfolios ---
print("\n--- 16. Identifying Key Portfolios ---")
# a. Minimum Volatility Portfolio
min_vol_portfolio = results_df.loc[results_df['Volatility'].idxmin()]

# b. Maximum Sharpe Ratio Portfolio (Optimal Risk-Adjusted Return)
max_sharpe_portfolio = results_df.loc[results_df['SharpeRatio'].idxmax()]

# Print results for the key portfolios
print("\n--- Minimum Volatility Portfolio ---")
print(f"Annual Return: {min_vol_portfolio['Return']:.2%}")
print(f"Annual Volatility: {min_vol_portfolio['Volatility']:.2%}")
print(f"Sharpe Ratio: {min_vol_portfolio['SharpeRatio']:.2f}")
min_vol_weights = pd.DataFrame(min_vol_portfolio['Weights'], index=iran_symbols, columns=['Weight'])
print("Asset Allocation:")
print(min_vol_weights.T.to_string(formatters={col: '{:.2%}'.format for col in iran_symbols}))

print("\n--- Maximum Sharpe Ratio Portfolio (Optimal) ---")
print(f"Annual Return: {max_sharpe_portfolio['Return']:.2%}")
print(f"Annual Volatility: {max_sharpe_portfolio['Volatility']:.2%}")
print(f"Sharpe Ratio: {max_sharpe_portfolio['SharpeRatio']:.2f}")
max_sharpe_weights = pd.DataFrame(max_sharpe_portfolio['Weights'], index=iran_symbols, columns=['Weight'])
print("Asset Allocation:")
print(max_sharpe_weights.T.to_string(formatters={col: '{:.2%}'.format for col in iran_symbols}))

# --- 17. Visualization (The Efficient Frontier) ---
print("\n--- 17. Plotting the Efficient Frontier ---")
plt.figure(figsize=(18, 9))

# Scatter plot of all simulated portfolios, color-coded by Sharpe Ratio
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['SharpeRatio'], cmap='viridis', marker='o', s=20, alpha=0.7)

# Add colorbar to show Sharpe Ratio values
plt.colorbar(label='Sharpe Ratio')

# Highlight the two key portfolios on the plot
# Max Sharpe Ratio Portfolio (Red Star)
plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], marker='*', color='red', s=500, label='Max Sharpe Ratio (Optimal)')
# Minimum Volatility Portfolio (Blue Star)
plt.scatter(min_vol_portfolio['Volatility'], min_vol_portfolio['Return'], marker='*', color='blue', s=500, label='Min Volatility')

# Add plot titles and labels
plt.title('Monte Carlo Simulation - Efficient Frontier', fontsize=20)
plt.xlabel('Annual Volatility (Risk)', fontsize=15)
plt.ylabel('Annual Return', fontsize=15)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
plt.show()

print("\n=====================================================")
print("=== MONTE CARLO OPTIMIZATION SUCCESSFULLY ADDED ===")
print("=====================================================")
