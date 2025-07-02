# pip install pandas finpy_tse PyPortfolioOpt matplotlib persiantools
#
import pandas as pd
import finpy_tse as fpy
from pypfopt import expected_returns, risk_models, EfficientFrontier
from persiantools.jdatetime import JalaliDate
import datetime

def get_annualized_return_volatility(prices):
    """Calculates annualized return and volatility from a series of prices."""
    log_returns = prices.pct_change().dropna()
    annualized_return = log_returns.mean() * 252
    annualized_volatility = log_returns.std() * (252**0.5)
    return annualized_return, annualized_volatility

def main():
    """Main function: Robust pipeline for Iranian stock screening and portfolio optimization."""
    # --- STAGE 1: BUILD INVESTMENT UNIVERSE ---
    print("Building investment universe from major, actively traded Iranian stocks...")
    universe = [
        'فولاد', 'فملی', 'شپنا', 'خودرو', 'وغدیر', 'تاپیکو', 'وبملت', 'شستا', 'حکشتی', 'کگل',
        'فارس', 'رمپنا', 'پارس', 'شبندر', 'شتران', 'وتجارت', 'خساپا', 'وتوصا', 'خپارس', 'وبصادر',
        'شپدیس', 'شراز', 'شگویا', 'شسپا', 'شکلر', 'شلعاب', 'شپاکسا', 'شپارس', 'شپترو', 'شپلی',
        'شمواد', 'شدوص', 'شدوک', 'شدوص', 'شدوک', 'شدوص', 'شدوک', 'شدوص', 'شدوک', 'شدوص'
    ]
    price_data = {}
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=3*365)
    for ticker in universe:
        try:
            print(f"Fetching data for {ticker}...")
            hist = fpy.Get_Price_History(stock=ticker, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'), adjust_price=True)
            if hist is None or hist.empty or 'Close' not in hist.columns or len(hist) < 2:
                print(f"Insufficient or invalid data for {ticker}")
                continue
            hist.index = pd.to_datetime(hist.index)
            price_data[ticker] = hist['Close']
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    if not price_data:
        print("No valid price data found. Exiting.")
        return
    master_price_df = pd.DataFrame(price_data)
    master_price_df.ffill(inplace=True)
    master_price_df.bfill(inplace=True)
    print(f"\nMaster price DataFrame shape: {master_price_df.shape}")

    # --- STAGE 2: SCREEN & FILTER ---
    print("\nScreening stocks based on annualized return and volatility...")
    daily_returns = master_price_df.pct_change().iloc[1:]
    ann_return = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * (252 ** 0.5)
    screener_df = pd.DataFrame({
        'Return': ann_return,
        'Volatility': ann_vol
    })
    screener_df = screener_df[(screener_df['Return'] > 0.20) & (screener_df['Volatility'] < 0.80)]
    screener_df['Sharpe'] = screener_df['Return'] / screener_df['Volatility']
    top_candidates = screener_df.nlargest(10, 'Sharpe')
    print("\n--- Top Candidates for Optimization ---")
    print(top_candidates)
    if top_candidates.empty:
        print("No stocks passed the screening criteria. Exiting.")
        return

    # --- STAGE 3: OPTIMIZE PORTFOLIO ---
    print("\nOptimizing portfolio with diversification constraint...")
    top_tickers = top_candidates.index.tolist()
    final_price_df = master_price_df[top_tickers]
    mu = expected_returns.mean_historical_return(final_price_df)
    S = risk_models.sample_cov(final_price_df)
    ef = EfficientFrontier(mu, S)
    # Add constraint: No single stock can have more than 25% weight (for diversification)
    ef.add_constraint(lambda w: all(w[i] <= 0.25 for i in range(len(w))))
    weights = ef.max_sharpe(risk_free_rate=0.0)
    cleaned_weights = ef.clean_weights()
    print("\nOptimal Weights (Max Sharpe, max 25% per stock):")
    print(cleaned_weights)
    print("\nPortfolio Performance (Max Sharpe):")
    ef.portfolio_performance(verbose=True, risk_free_rate=0.0)

if __name__ == "__main__":
    main()
