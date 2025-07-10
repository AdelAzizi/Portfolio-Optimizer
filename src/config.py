# config.py

# Global Configuration Parameters
RISK_FREE_RATE = 0.05  # Example: 5% annual risk-free rate
MIN_RETURN_THRESHOLD = 0.15 # Example: 15% minimum annual return for screening
MAX_VOLATILITY_THRESHOLD = 0.75 # Example: 75% maximum annual volatility for screening
MAX_POSITION_SIZE = 0.25 # Example: 25% maximum allocation to a single asset in portfolio optimization
MAX_DRAWDOWN_LIMIT = -0.80 # Example: 80% maximum allowable drawdown
MIN_DATA_POINTS = 252 # Require at least 1 year of trading data
MIN_LIQUIDITY_THRESHOLD = 10000000 # Minimum average daily volume

# Data Paths
PREPROCESSED_DATA_FILE = "analysis_ready_data.feather"
MASTER_PRICE_DATA_FILE = "master_price_data.feather"
CACHE_DIR = "cache_v3"