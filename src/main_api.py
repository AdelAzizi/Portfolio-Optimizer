# -*- coding: utf-8 -*-

# ==============================================================================
# Title: Portfolio Optimizer FastAPI Endpoint
# Description: Exposes the quantitative portfolio optimization engine as a
#              REST API using FastAPI.
# Author: Kilo Code, the AI Software Engineer
# ==============================================================================

import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import the refactored optimizer and config ---
from src.optimizer import MultiFactorOptimizer
from src import config

# --- Setup Logging ---

# --- Define Project Root Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Setup Logging ---
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Instantiate FastAPI App ---
app = FastAPI(
    title="Portfolio Optimizer API",
    description="An API to run multi-factor portfolio optimization.",
    version="1.0.0"
)

# --- CORS Configuration ---
# The CORSMiddleware must be added BEFORE any routes are defined to handle preflight requests correctly.
origins = [
    "http://localhost",
    "http://localhost:3000", # Common port for React/Next.js development
    # Add the URL of your deployed frontend app here when you have one.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
# -------------------------

# --- Pydantic Models ---
class StrategyRequest(BaseModel):
    """Defines the expected input for selecting a strategy."""
    strategy_name: str = "balanced"

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Portfolio Optimizer API. Use the /docs endpoint for details."}

@app.post("/optimize-strategy")
async def optimize_strategy(request: StrategyRequest):
    """
    Runs the full screening and optimization pipeline based on a dynamically
    selected strategy and returns the resulting portfolio and performance.
    """
    strategy_name = request.strategy_name
    logger.info(f"API endpoint /optimize-strategy called with strategy: '{strategy_name}'")

    try:
        # --- 1. Select Strategy Configuration ---
        strategy_config = config.STRATEGY_CONFIGS.get(strategy_name)
        if not strategy_config:
            logger.warning(f"Invalid strategy name provided: {strategy_name}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy name: '{strategy_name}'. Available strategies are: {list(config.STRATEGY_CONFIGS.keys())}"
            )
        logger.info(f"Selected strategy config: {strategy_config}")

        # --- 2. Define Data Paths ---
        cache_dir = PROJECT_ROOT / 'cache'
        data_dir = PROJECT_ROOT / 'data'
        analysis_data_path = cache_dir / 'full_analysis_ready_data.feather'
        price_data_dir = data_dir / 'full_market_data_csvs'

        # --- 3. Instantiate and Run Optimizer with Dynamic Config ---
        optimizer = MultiFactorOptimizer(
            analysis_data_path=analysis_data_path,
            price_data_dir=price_data_dir,
            max_position_size=config.MAX_POSITION_SIZE, # Using global max position size
            factor_weights=strategy_config['factor_weights'],
            momentum_period=strategy_config['momentum_period'],
            top_n_candidates=config.TOP_N_CANDIDATES
        )
        
        results = optimizer.run_full_analysis()

        if not results:
            logger.warning("Optimization failed to produce a valid portfolio for the selected strategy.")
            raise HTTPException(
                status_code=404,
                detail="Optimization failed to produce a valid portfolio. No assets passed the initial screen for the selected strategy."
            )

        logger.info(f"✅ Successfully generated optimization results for strategy: '{strategy_name}'")
        
        # The structure from run_full_analysis is now nested.
        # The API will return the entire comprehensive dictionary.
        return {
            "strategy_name": strategy_name,
            "results": results
        }

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(status_code=500, detail=f"A required data file was not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")