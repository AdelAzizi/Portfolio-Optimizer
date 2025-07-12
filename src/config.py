# config.py
from pathlib import Path

# This robustly finds the project root (C:\Portfolio-Optimizer)
# by going up one level from the script's directory ('src').
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Global Configuration Parameters
RISK_FREE_RATE = 0.05  # Example: 5% annual risk-free rate
MIN_RETURN_THRESHOLD = 0.15 # Example: 15% minimum annual return for screening
MAX_VOLATILITY_THRESHOLD = 0.75 # Example: 75% maximum annual volatility for screening
MAX_POSITION_SIZE = 0.15 # Example: 25% maximum allocation to a single asset in portfolio optimization
MAX_DRAWDOWN_LIMIT = -0.80 # Example: 80% maximum allowable drawdown
MIN_DATA_POINTS = 252 # Require at least 1 year of trading data
MIN_LIQUIDITY_THRESHOLD = 10000000 # Minimum average daily volume
TOP_N_CANDIDATES = 20


# Data Paths
CACHE_DIR = PROJECT_ROOT / "cache"
PREPROCESSED_DATA_FILE = CACHE_DIR / "analysis_ready_data.feather"
MASTER_PRICE_DATA_FILE = CACHE_DIR / "master_price_data.feather"

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