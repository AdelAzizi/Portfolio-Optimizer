# pip install pandas finpy_tse PyPortfolioOpt matplotlib

import pandas as pd
import finpy_tse as fpy
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
from pypfopt import cla
import matplotlib.pyplot as plt

# --- 1. Data Fetching (Corrected Method) ---
print("Fetching adjusted historical data...")

# List of Iranian tickers
tickers = ['خودرو']

# Fetch data with Adjusted=True
all_data = {ticker: fpy.Get_Price_History(stock=ticker, start_date='1399-01-01', end_date='1402-12-29', adjust_price=True) for ticker in tickers}

# Create a DataFrame of adjusted close prices
adj_close_prices = pd.DataFrame({ticker: data['Close'] for ticker, data in all_data.items()})

# Convert index to datetime
from persiantools.jdatetime import JalaliDate

# Convert Jalali dates to Gregorian dates for pandas
adj_close_prices.index = [JalaliDate.strptime(str(d), '%Y-%m-%d').to_gregorian() for d in adj_close_prices.index]
adj_close_prices.index = pd.to_datetime(adj_close_prices.index)

# Handle missing values
adj_close_prices.ffill(inplace=True)
adj_close_prices.bfill(inplace=True)

print("Data fetched and cleaned successfully.")
print(adj_close_prices.head())

# --- 2. Prepare Data for PyPortfolioOpt ---
print("\nCalculating expected returns and covariance...")

# Calculate expected annual returns
mu = expected_returns.mean_historical_return(adj_close_prices)

# Calculate the sample covariance matrix
S = risk_models.sample_cov(adj_close_prices)

# --- 3. Calculate the Efficient Frontier Portfolios ---
print("\n--- Optimization Results ---")

# a) Maximum Sharpe Ratio Portfolio
print("\nCalculating Maximum Sharpe Ratio Portfolio...")
ef_max_sharpe = EfficientFrontier(mu, S)
weights_max_sharpe = ef_max_sharpe.max_sharpe()
cleaned_weights_max_sharpe = ef_max_sharpe.clean_weights()

print("\nOptimal Weights (Max Sharpe):")
print(cleaned_weights_max_sharpe)

print("\nPortfolio Performance (Max Sharpe):")
ef_max_sharpe.portfolio_performance(verbose=True)

# b) Minimum Volatility Portfolio
print("\n\nCalculating Minimum Volatility Portfolio...")
ef_min_vol = EfficientFrontier(mu, S)
weights_min_vol = ef_min_vol.min_volatility()
cleaned_weights_min_vol = ef_min_vol.clean_weights()

print("\nOptimal Weights (Min Volatility):")
print(cleaned_weights_min_vol)

print("\nPortfolio Performance (Min Volatility):")
ef_min_vol.portfolio_performance(verbose=True)

# --- 4. Plot the Efficient Frontier (The Mathematical Way) ---
print("\n\nPlotting the Efficient Frontier...")

# Use the Critical Line Algorithm (CLA) for plotting
cla_instance = cla.CLA(mu, S)

# Plotting the frontier
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(cla_instance, ax=ax, show_assets=True)

# Find the tangency portfolio
# For plotting, we need to get the portfolios from the CLA object
cla_instance.max_sharpe()
cla_instance.min_volatility()

# Get the portfolios for plotting
max_sharpe_ret, max_sharpe_risk, _ = cla_instance.portfolio_performance()
min_vol_ret, min_vol_risk, _ = cla_instance.portfolio_performance()

# Annotate the plot
ax.scatter(max_sharpe_risk, max_sharpe_ret, marker="*", s=100, c="r", label="Max Sharpe")
ax.scatter(min_vol_risk, min_vol_ret, marker="*", s=100, c="g", label="Min Volatility")

# Final plot adjustments
ax.set_title("Efficient Frontier (Mathematical)")
ax.legend()
plt.tight_layout()
plt.show()

print("\nScript finished.")