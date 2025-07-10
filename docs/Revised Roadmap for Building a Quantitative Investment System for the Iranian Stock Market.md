## Revised Roadmap for Building a Quantitative Investment System for the Iranian Stock Market

Based on the thorough discussion of the critique from "The Critical Coach" AI, and aligning with the "Get to Profit Fast, Then Improve" philosophy, this revised roadmap prioritizes rapid value delivery, mitigates key risks, and focuses on building a robust, interpretable, and profitable quantitative investment system for the Iranian stock market. The initial phases will primarily leverage price-derived factors, with subsequent iterations introducing more complexity and new data sources.

**Overall Goal:** Rapidly develop a Minimum Viable Reliable Product (MVRP) quantitative investment system for the Iranian stock market, focused on the 3-12 month investment horizon, with a strong emphasis on interpretability, robust backtesting, and comprehensive documentation. Subsequent phases will iteratively add complexity and new data sources.

**Revised Phases:**

### Phase 1: Immediate Backtest Issue Resolution & Baseline Establishment (Focus: Price Data & Existing Code)
*   **Objective:** Understand and resolve the underperformance of the existing Low-Vol/Reversal strategy against the market index. Establish a clear baseline for future improvements.
*   **Key Tasks:**
    *   Execute separate backtests for Low Volatility and Reversal factors using the current codebase to isolate their individual performance.
    *   Analyze the results to understand *why* they underperformed. This might involve examining factor definitions, calculation methods, or market conditions during the backtest period.
    *   Document findings and insights from these baseline backtests.
*   **Deliverable:** A clear understanding of the current strategy's performance and identified areas for improvement. Initial test results and analysis.

### Phase 2: Momentum Strategy Implementation & Validation (Focus: Price Data & Proven Factors)
*   **Objective:** Implement and rigorously backtest a momentum strategy, a historically robust factor, as a strong candidate for early profitability.
*   **Key Tasks:**
    *   Refine `factors/calculator.py` to accurately calculate 6-month and 12-month price momentum.
    *   Integrate momentum into `screener_v3_optimized.py` for stock selection.
    *   Backtest a pure momentum strategy against the benchmark.
    *   (Optional, if time permits in this phase) Explore combining momentum with a simple macro filter (e.g., EMA 20/50) and backtest.
    *   Ensure comprehensive code comments and update `factors/README.md`.
*   **Deliverable:** A working momentum strategy integrated into the system, with backtest results demonstrating its performance. Updated code and documentation.

### Phase 3: Refined Multi-Factor Screening & Portfolio Optimization (Focus: Combining Price Factors)
*   **Objective:** Combine the validated price-derived factors (e.g., momentum, low-volatility, reversal if proven effective) into a multi-factor screening model and refine portfolio optimization for the 3-12 month horizon.
*   **Key Tasks:**
    *   Implement a simple multi-factor combination strategy (e.g., equal weighting of normalized factor ranks) in `screener_v3_optimized.py`.
    *   Enhance screening criteria with liquidity filters and minimum data history requirements.
    *   Refine portfolio optimization to apply realistic constraints (e.g., `max_position_size`) and handle edge cases robustly.
    *   Ensure comprehensive code comments and update relevant READMEs.
*   **Deliverable:** A functional multi-factor screening and portfolio optimization pipeline, with initial backtest results for the combined strategy. Updated code and documentation.

### Phase 4: Robust Backtesting Framework & Performance Analysis (Focus: Validation & Reporting)
*   **Objective:** Develop a comprehensive and realistic backtesting framework to accurately evaluate the strategy's performance, incorporating real-world frictions and standard metrics.
*   **Key Tasks:**
    *   Enhance `run_backtest` in `screener_v3_optimized.py` to prevent look-ahead bias, apply realistic transaction costs, and model slippage.
    *   Implement calculation and reporting of key performance metrics (Sharpe, Max Drawdown, Alpha, Beta, etc.).
    *   Generate clear visualizations of cumulative returns against a benchmark.
    *   Ensure comprehensive code comments and create a dedicated README for backtesting methodology.
*   **Deliverable:** A robust backtesting framework, detailed performance reports, and clear visualizations. Updated code and documentation.

### Phase 5: Comprehensive Documentation & Project Overview (Focus: Interpretability & Maintainability)
*   **Objective:** Consolidate all documentation, including inline code comments, module-specific READMEs, and an overarching project README, to ensure interpretability, maintainability, and defensibility.
*   **Key Tasks:**
    *   Review all Python files for comprehensive and up-to-date inline comments.
    *   Create/update `data/README.md` and `factors/README.md`.
    *   Develop a comprehensive `README.md` in the root directory, including project overview, structure, architectural overview (textual or simple ASCII diagram), installation, usage, factors, and backtesting methodology.
*   **Deliverable:** A fully documented project, providing a clear and defensible understanding of the system.

**Post-MVRP Iterations (Future Phases):**
Once the MVRP is established and validated, we can then iteratively introduce more complexity:
*   **Fundamental Data Integration:** Explore reliable methods for acquiring and integrating fundamental data from Codal or other sources.
*   **Advanced Factors:** Introduce more complex factors (e.g., quality, growth) based on fundamental data.
*   **Machine Learning Models:** Experiment with ML models for screening or prediction, with careful attention to overfitting.
*   **Farsi NLP:** Consider this as a long-term research project, separate from the core investment system, once the foundational profitability is established.

This revised roadmap provides a clear, actionable path to achieving a profitable and well-understood quantitative investment system for the Iranian stock market within your desired timeframe, while proactively addressing the valid concerns raised by the critique.


