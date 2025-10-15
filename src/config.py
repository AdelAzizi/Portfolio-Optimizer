# config.py
from pathlib import Path

# This robustly finds the project root     "DATA_DIR": "data/full_market_data_csvs/",C:\Portfolio-Optimizer)
# by going up one level from the script's directory ('src').
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Universe Creator Settings ---
# Parameters for the script that selects the initial pool of stocks.
UNIVERSE_CREATOR = {
    "CACHE_VALIDITY_HOURS": 11,
    "FILTERS": {
        # Minimum trading volume to be considered liquid
        "MIN_LIQUIDITY": 100000,
        # Minimum market capitalization in Toman
        "MIN_MARKET_CAP": 1e12,
        # Market flow types to include: 1 for Bourse, 2 for Fara Bourse
        "MARKET_FLOW_TYPES": [1, 2]
    },
    "MIN_TRADING_DAYS": 30,
    "CACHE_DIR": "cache",
    "UNIVERSE_FILENAME": "universe.json"
}

# --- Full Market Downloader Settings ---
FULL_MARKET_DOWNLOADER = {
    "DATA_DIR": "data/full_market_data_csvs",
    "BLACKLIST_FILENAME": "downloader_blacklist.json",
    "BENCHMARK_SYMBOL": "شاخص کل",
    "API_DELAY_SECONDS": 1.0  # Delay between each API call - increased to prevent rate limiting
}

# --- Full Market Fundamental Collector Settings ---
FUNDAMENTAL_COLLECTOR = {
    "CACHE_DIR": "cache",
    "OUTPUT_FILE": "master_fundamental_data.feather",
    "BLACKLIST_FILE": "failed_symbols.json",
    "CACHE_VALIDITY_HOURS": 168,  # 1 week
    "REQUEST_DELAY_SEC": 0.2,
    "BLACKLIST_EXPIRY_DAYS": 30,
    "FUNDAMENTAL_FIELDS": ['P/E', 'P/S', 'EPS']
}

# Global Configuration Parameters
RISK_FREE_RATE = 0.05  # Example: 5% annual risk-free rate
MIN_RETURN_THRESHOLD = 0.15 # Example: 15% minimum annual return for screening
MAX_VOLATILITY_THRESHOLD = 0.75 # Example: 75% maximum annual volatility for screening
MAX_POSITION_SIZE = 0.15 # Example: 25% maximum allocation to a single asset in portfolio optimization
MAX_DRAWDOWN_LIMIT = -0.80 # Example: 80% maximum allowable drawdown
MIN_DATA_POINTS = 252 # Require at least 1 year of trading data
MIN_LIQUIDITY_THRESHOLD = 10000000 # Minimum average daily volume
TOP_N_CANDIDATES = 20

TRADE_COST_PERCENT = 0.005 # 0.5% cost on each trade (buy/sell)

# Commission and Slippage Parameters
COMMISSION_RATE = 0.002  # 0.2% commission per trade
SLIPPAGE_PCT = 0.001     # 0.1% slippage per trade

# Risk Metrics Weights for Multi-Criteria Scoring
RISK_METRICS_WEIGHTS = {
    'sharpe': 0.3,
    'sortino': 0.2,
    'calmar': 0.2,
    'stability': 0.15,
    'return': 0.15
}

# Re-evaluation count parameter
TOP_REEVALUATION_COUNT = 100

# Candidates per category parameter
CANDIDATES_PER_CATEGORY = 5

# Data Paths
CACHE_DIR = PROJECT_ROOT / "cache"
PREPROCESSED_DATA_FILE = CACHE_DIR / "analysis_ready_data.feather"
MASTER_PRICE_DATA_FILE = CACHE_DIR / "master_price_data.feather"

# --- Full Market Preprocessor Settings ---
FULL_MARKET_PREPROCESSOR = {
    "DATA_DIR": "data",
    "CACHE_DIR": "cache",
    "FUNDAMENTAL_FILE": "master_fundamental_data.feather",
    "OUTPUT_FILE": "full_analysis_ready_data.feather",
    "TRADING_DAYS_PER_YEAR": 220,  # برای بازار ایران
    "MOMENTUM_PERIODS": {
        '3M': 63,
        '6M': 126,
        '12M': 252
    },
    "RISK_FREE_RATE": 0.15,  # ۱۵% نرخ بدون ریسک برای ایران
    "MIN_DATA_POINTS": 30  # حداقل روزهای معاملاتی برای محاسبات
}

# --- Multi-Factor Model Weights ---
# The sum of these weights should ideally be 1.0
# FACTOR_WEIGHTS = {
#     'Value': 0.7,         # STRATEGY 1: "Value is King"
#     'Momentum': 0.2,
#     'Low_Volatility': 0.1
# }

FACTOR_WEIGHTS = {
    'Value': 0.3,
    'Momentum': 0.5,
    'Low_Volatility': 0.2
}

MOMENTUM_PERIOD = '12M'  # Options: '3M', '6M', '12M'

# --- Strategy Configurations ---
STRATEGY_CONFIGS = {
    "aggressive": { # استراتژی ابر مومنتوم
        "factor_weights": {'Value': 0.05, 'Momentum': 0.9, 'Low_Volatility': 0.05},
        "momentum_period": '12M'
    },
    "balanced": { # استراتژی رشد هوشمند
        "factor_weights": {'Value': 0.3, 'Momentum': 0.5, 'Low_Volatility': 0.2},
        "momentum_period": '12M'
    },
    "defensive": { # استراتژی متعادل با تنوع بالا
        "factor_weights": {'Value': 0.33, 'Momentum': 0.33, 'Low_Volatility': 0.34},
        "momentum_period": '12M'
    }
}


# You can add other configurations for different experiments here
# FACTOR_WEIGHTS_MOMENTUM_FOCUS = { ... }
# FACTOR_WEIGHTS_BALANCED = { ... }