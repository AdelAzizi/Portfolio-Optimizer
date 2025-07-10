# AI Guidance for Portfolio Optimizer Project Cleanup and Refinement

This document provides a step-by-step guide for an AI agent to perform initial cleanup and structural improvements on the 'Portfolio-Optimizer' project. The primary goal is to enhance the robustness and cleanliness of existing features without altering the core architectural design. This will prepare the project for future development and more advanced functionalities.

## Phase 1: Initial Project Setup and Environment Validation

**Objective:** Ensure the project environment is correctly set up, dependencies are met, and the current state of the project (including tests) is understood.

**Instructions for AI Agent:**

1.  **Verify Repository Access:** Confirm that the 'Portfolio-Optimizer' repository is cloned and accessible in the working directory.
    *   **Prompt Example:** `shell_exec(brief="Verify repository existence", command="ls -d Portfolio-Optimizer/", session_id="setup")`

2.  **Install Dependencies:** Ensure all required Python packages are installed. Refer to `requirements_v3.txt`.
    *   **Prompt Example:** `shell_exec(brief="Install Python dependencies", command="pip install -r Portfolio-Optimizer/requirements_v3.txt", session_id="setup", working_dir="/home/ubuntu")`

3.  **Run Existing Tests (Baseline):** Execute the `test_v3.py` suite to establish a baseline of current functionality and identify any immediate errors. Document the test results.
    *   **Prompt Example:** `shell_exec(brief="Run project tests to establish baseline", command="python -m pytest test_v3.py", session_id="testing", working_dir="/home/ubuntu/Portfolio-Optimizer")`
    *   **Guidance:** Expect some failures or warnings at this stage, as the goal is to identify existing issues before fixing them. Focus on capturing the output for later comparison.



## Phase 2: Configuration Management Refinement

**Objective:** Externalize hardcoded configuration parameters into a dedicated file for easier management and flexibility.

**Instructions for AI Agent:**

1.  **Create Configuration File:** Create a new Python file named `config.py` in the root of the `Portfolio-Optimizer` directory.
    *   **Prompt Example:** `file_write_text(abs_path="/home/ubuntu/Portfolio-Optimizer/config.py", brief="Create config.py file", content="""
# config.py

# Global Configuration Parameters
RISK_FREE_RATE = 0.05  # Example: 5% annual risk-free rate
MIN_RETURN_THRESHOLD = 0.15 # Example: 15% minimum annual return for screening
MAX_VOLATILITY_THRESHOLD = 0.75 # Example: 75% maximum annual volatility for screening
MAX_POSITION_SIZE = 0.25 # Example: 25% maximum allocation to a single asset in portfolio optimization

# Data Paths
PREPROCESSED_DATA_FILE = "analysis_ready_data.feather"
MASTER_PRICE_DATA_FILE = "master_price_data.feather"
CACHE_DIR = "cache_v3"
""")`

2.  **Integrate Configuration into `screener_v3_optimized.py`:**
    *   **Import `config`:** Add `import config` at the top of `screener_v3_optimized.py`.
    *   **Replace Hardcoded Values:** Replace the hardcoded values for `risk_free_rate`, `min_return_threshold`, `max_volatility_threshold`, and `max_position_size` with references to the `config` module.
    *   **Prompt Example (for `risk_free_rate`):** `file_replace_text(abs_path="/home/ubuntu/Portfolio-Optimizer/screener_v3_optimized.py", brief="Replace hardcoded risk_free_rate with config reference", old_str="risk_free_rate: float = 0.28", new_str="risk_free_rate: float = config.RISK_FREE_RATE")`
    *   **Guidance:** Apply similar `file_replace_text` operations for `min_return_threshold`, `max_volatility_threshold`, and `max_position_size`.

3.  **Integrate Configuration into `preprocessor.py` (if applicable):** Review `preprocessor.py` for any hardcoded paths or parameters that could benefit from being in `config.py` and move them accordingly.
    *   **Guidance:** Look for file paths or numerical constants that might change between environments or experiments.

4.  **Update `test_v3.py` to use `config.py`:** Modify the test suite to import and use the new configuration values, ensuring tests reflect the configurable nature of the application.
    *   **Guidance:** This might involve adjusting how `IranianStockOptimizerV4` is initialized in the tests.

5.  **Run Tests (Verification):** Execute the test suite again to ensure that the changes to configuration management have not introduced regressions and that the application still functions as expected.
    *   **Prompt Example:** `shell_exec(brief="Run tests after config integration", command="python -m pytest test_v3.py", session_id="testing", working_dir="/home/ubuntu/Portfolio-Optimizer")`
    *   **Guidance:** All tests should pass. If not, debug the configuration integration.

## Phase 3: Enhanced Error Handling and Logging

**Objective:** Improve the robustness of the application by implementing more informative error handling and consistent logging.

**Instructions for AI Agent:**

1.  **Review `screener_v3_optimized.py` for Error Handling:**
    *   **Focus on `optimize_portfolio`:** Instead of returning an empty dictionary on optimization failure, raise a more specific exception (e.g., `PortfolioOptimizationError`) or log a critical error message and return `None`.
    *   **Prompt Example (Conceptual):** `file_replace_text(abs_path="/home/ubuntu/Portfolio-Optimizer/screener_v3_optimized.py", brief="Improve error handling in optimize_portfolio", old_str="return {}", new_str="logger.error(f\"Optimization failed: {e}\"); return None")` (This is a conceptual example; actual implementation might be more complex).
    *   **Guidance:** Ensure that any function that might fail gracefully logs the reason for failure and provides a clear indication of the outcome.

2.  **Standardize Logging:** Ensure consistent use of the `logger` object throughout `screener_v3_optimized.py` and `preprocessor.py` for all informational, warning, and error messages.
    *   **Guidance:** Replace `print()` statements with `logger.info()`, `logger.warning()`, or `logger.error()` as appropriate.

3.  **Update Tests for Error Handling:** Modify `test_v3.py` to specifically test the improved error handling mechanisms. For example, assert that appropriate exceptions are raised or that `None` is returned on failure.
    *   **Guidance:** Create new test cases that intentionally trigger failure conditions to verify error handling.

4.  **Run Tests (Verification):** Execute the test suite again to confirm that the enhanced error handling works as expected and no new issues have been introduced.
    *   **Prompt Example:** `shell_exec(brief="Run tests after error handling improvements", command="python -m pytest test_v3.py", session_id="testing", working_dir="/home/ubuntu/Portfolio-Optimizer")`

## Phase 4: Refinement of Test Suite and Data

**Objective:** Improve the reliability and realism of the test suite.

**Instructions for AI Agent:**

1.  **Review Test Data Generation in `test_v3.py`:**
    *   **Replace Random Data with Realistic Stubs:** While full real-world data is not available in the sandbox, replace purely random data generation with more structured, representative dummy data that mimics real financial data characteristics (e.g., trends, seasonality, typical value ranges).
    *   **Guidance:** For `analysis_ready_data.feather` and `master_price_data.feather`, ensure the dummy data reflects plausible stock returns, volatilities, and price movements. This will make tests more meaningful.

2.  **Expand Test Coverage:** Add new test cases to cover more edge cases and specific scenarios, especially for data validation and screening logic.
    *   **Guidance:** Consider tests for empty dataframes, dataframes with missing values, and boundary conditions for screening thresholds.

3.  **Run All Tests (Final Verification):** Execute the complete test suite to ensure all changes are stable and the project is robust.
    *   **Prompt Example:** `shell_exec(brief="Run all tests after test suite refinement", command="python -m pytest test_v3.py", session_id="testing", working_dir="/home/ubuntu/Portfolio-Optimizer")`

Upon successful completion of these phases, the project will have a cleaner structure, improved configurability, more robust error handling, and a more reliable test suite, making it ready for the next stage of development (e.g., factor validation and multi-factor modeling).
