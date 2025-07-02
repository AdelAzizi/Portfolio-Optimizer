# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Iranian Stock Market Screener & Optimizer v1.0
# Description: An end-to-end Python script for fetching, screening, and
#              optimizing a portfolio of stocks from the Tehran Stock Exchange.
# Author: Quantitative Analyst AI
# ==============================================================================

# --- Core Libraries ---
import pandas as pd
import pytse_client as tse
from jdatetime import datetime as jdatetime
import datetime

# --- Portfolio Optimization Libraries ---
from pypfopt import expected_returns, risk_models, EfficientFrontier

def main():
    """Main function to run the entire screening and optimization pipeline."""

    # ==========================================================================
    # --- STAGE 1: BUILD INVESTMENT UNIVERSE                                 ---
    # ==========================================================================
    print("--- STAGE 1: Building Investment Universe ---")

    # 1.1: Define the list of major, actively traded Iranian stocks
    # This list forms our initial investment universe.
    universe = [
        'فولاد', 'فملی', 'شپنا', 'خودرو', 'وغدیر', 'تاپیکو', 'وبملت', 'شستا',
        'حکشتی', 'کگل', 'فارس', 'رمپنا', 'پارس', 'شبندر', 'شتران', 'وتجارت',
        'خساپا', 'وبصادر', 'کچاد', 'ومعادن', 'نوری', 'زاگرس', 'بوعلی', 'مارون'
    ]
    print(f"Defined universe with {len(universe)} stocks.")

    # 1.1.1: Remove bankrupt stocks from the universe
    bankrupt_stocks = [
        # مثال: 'نماد1', 'نماد2'
        # این لیست را با نمادهای ورشکسته تکمیل کنید
    ]
    universe = [ticker for ticker in universe if ticker not in bankrupt_stocks]

    # 1.2: Fetch, convert, and store historical price data
    price_data = {}
    print("\nFetching historical data for each stock...")

    # دانلود داده‌ها (در اولین اجرا یا برای به‌روزرسانی)
    tse.download(symbols=universe)

    for ticker in universe:
        try:
            # دریافت دیتافریم قیمت‌های تعدیل‌شده برای هر نماد
            ticker_obj = tse.Ticker(ticker, adjust=True)
            hist = ticker_obj.history  # استفاده از ویژگی history به جای df

            if hist is None or hist.empty:
                print(f"- WARNING: No data returned for {ticker}. Skipping.")
                continue

            # فیلتر داده‌ها برای ۵ سال اخیر
            five_years_ago = pd.Timestamp.today() - pd.DateOffset(years=5)
            hist = hist[pd.to_datetime(hist['date']) >= five_years_ago]
            if hist.empty:
                print(f"- WARNING: No recent data (last 5 years) for {ticker}. Skipping.")
                continue

            # تاریخ‌ها به صورت جلالی یا میلادی هستند، تبدیل به میلادی فقط اگر لازم بود
            gregorian_dates = []
            for j_date in hist['date']:
                # اگر مقدار تاریخ از نوع datetime است، مستقیم استفاده کن
                if isinstance(j_date, (datetime.date, datetime.datetime)):
                    gregorian_dates.append(pd.to_datetime(j_date))
                else:
                    # اگر رشته است، سعی کن به صورت میلادی تبدیل کنی
                    try:
                        gregorian_dates.append(pd.to_datetime(j_date))
                    except Exception:
                        # اگر نشد، فرض کن جلالی است و تبدیل کن
                        j_date_obj = jdatetime.strptime(str(j_date), '%Y/%m/%d')
                        gregorian_dates.append(j_date_obj.togregorian())
            hist.index = pd.to_datetime(gregorian_dates)

            # ذخیره قیمت پایانی تعدیل‌شده
            price_data[ticker] = hist['close']
            print(f"- Successfully processed: {ticker}")

        except Exception as e:
            print(f"- ERROR: Could not process {ticker}. Reason: {e}")

    # 1.3: Create and clean the master DataFrame
    if not price_data:
        print("\nFATAL: No data could be fetched. Exiting program.")
        return

    print("\nCreating and cleaning the master price DataFrame...")
    master_price_df = pd.DataFrame(price_data)
    master_price_df.sort_index(inplace=True)

    # Fill missing values to ensure data is perfectly aligned and complete
    master_price_df.ffill(inplace=True)
    master_price_df.bfill(inplace=True)

    print(f"Master DataFrame created with shape: {master_price_df.shape}")

    # ==========================================================================
    # --- STAGE 2: SCREEN & FILTER CANDIDATES                                ---
    # ==========================================================================
    print("\n--- STAGE 2: Screening and Filtering Candidates ---")

    # 2.1: Calculate annualized return and volatility
    daily_returns = master_price_df.pct_change().dropna()
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * (252 ** 0.5)

    # 2.2: Create the Screener DataFrame
    screener_df = pd.DataFrame({
        'Return': annualized_return,
        'Volatility': annualized_volatility
    })

    # 2.3: Filter for quality stocks based on robust criteria
    print(f"\nFiltering {len(screener_df)} stocks based on:")
    print("- Annualized Return > 25%")
    print("- Annualized Volatility < 75%")
    filtered_stocks = screener_df[
        (screener_df['Return'] > 0.25) & (screener_df['Volatility'] < 0.75)
    ]

    if filtered_stocks.empty:
        print("\nNo stocks passed the screening criteria. Exiting.")
        return

    print(f"\n{len(filtered_stocks)} stocks passed the filter.")

    # 2.4: Rank by Sharpe Ratio and select the top 10
    # Assuming a risk-free rate of 0 for simplicity
    filtered_stocks['Sharpe'] = filtered_stocks['Return'] / filtered_stocks['Volatility']
    top_candidates = filtered_stocks.nlargest(10, 'Sharpe')

    print("\n--- Top 10 Candidates for Optimization ---")
    print(top_candidates)

    # ==========================================================================
    # --- STAGE 3: OPTIMIZE PORTFOLIO                                        ---
    # ==========================================================================
    print("\n--- STAGE 3: Optimizing the Final Portfolio ---")

    # 3.1: Prepare the final price data for the optimizer
    top_tickers = top_candidates.index.tolist()
    final_price_df = master_price_df[top_tickers]

    # 3.2: Instantiate the optimizer with expected returns and covariance
    mu = expected_returns.mean_historical_return(final_price_df)
    S = risk_models.sample_cov(final_price_df)
    ef = EfficientFrontier(mu, S)

    # 3.3: Add a critical diversification constraint
    # No single stock can have more than 30% of the portfolio's weight.
    ef.add_constraint(lambda w: w <= 0.30)

    # 3.4: Optimize for the maximum Sharpe Ratio
    weights = ef.max_sharpe(risk_free_rate=0.0)
    cleaned_weights = ef.clean_weights()

    # 3.5: Display the final results in a clear and readable format
    print("\n--- OPTIMAL PORTFOLIO تخصیص --- ")
    print("Optimal Weights (Max Sharpe, max 30% per stock):")
    for ticker, weight in cleaned_weights.items():
        print(f"- {ticker}: {weight:.2%}")

    print("\n--- OPTIMAL PORTFOLIO PERFORMANCE --- ")
    ef.portfolio_performance(verbose=True, risk_free_rate=0.0)


if __name__ == "__main__":
    main()