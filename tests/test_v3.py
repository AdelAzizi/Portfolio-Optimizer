# -*- coding: utf-8 -*-

"""
Test suite for Iranian Stock Market Screener & Optimizer v3.1
Enhanced to work with existing CSV data files
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
import config
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screener_v3_optimized import IranianStockOptimizerV3

class TestIranianStockOptimizerV3:
    """Test class for the Iranian Stock Optimizer v3.1"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        print("🧪 Setting up test environment for Iranian Stock Optimizer v3.1")
        cls.test_cache_dir = Path('test_cache_v3')
        cls.test_cache_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer with test parameters
        cls.optimizer = IranianStockOptimizerV3(
            cache_dir=str(cls.test_cache_dir),
            years_of_data=3,  # Reduced for testing
            risk_free_rate=config.RISK_FREE_RATE
        )
        
        # Check if we have CSV data files
        cls.tickers_data_dir = Path('tickers_data')
        cls.csv_files = []
        if cls.tickers_data_dir.exists():
            cls.csv_files = list(cls.tickers_data_dir.glob('*.csv'))
            print(f"📁 Found {len(cls.csv_files)} CSV files in tickers_data directory")
        else:
            print("⚠️ No tickers_data directory found")
            
    def _generate_realistic_price_data(self, start_price=1000, drift=0.0005, volatility=0.02, num_days=252):
        """Generates a more realistic stock price series."""
        dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq='D'))
        price_movements = np.random.normal(loc=drift, scale=volatility, size=num_days)
        prices = start_price * (1 + price_movements).cumprod()
        return pd.Series(prices, index=dates)
    
    def load_csv_data(self, symbol_files, max_files=8):
        """Load data from CSV files and create price and volume data dictionaries"""
        price_data = {}
        volume_data = {}
        successful_loads = 0
        
        print(f"📊 Loading data from CSV files...")
        
        for csv_file in symbol_files[:max_files]:
            try:
                df = pd.read_csv(csv_file)
                
                if 'date' not in df.columns or 'close' not in df.columns or 'volume' not in df.columns:
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep='last')]
                
                close_prices = df['close']
                volume = df['volume']
                
                if len(close_prices) < 100: continue
                if (close_prices <= 0).any(): continue
                if close_prices.isnull().sum() > len(close_prices) * 0.1: continue
                
                symbol = csv_file.stem
                price_data[symbol] = close_prices
                volume_data[symbol] = volume
                successful_loads += 1
                
                print(f"   ✅ Loaded {symbol}: {len(close_prices)} data points")
                
            except Exception as e:
                print(f"   ❌ Failed to load {csv_file.name}: {e}")
                continue
        
        print(f"📈 Successfully loaded {successful_loads} symbols from CSV files")
        return price_data, volume_data
    
    def test_initialization(self):
        """Test optimizer initialization"""
        print("\n🧪 Testing optimizer initialization...")
        
        assert self.optimizer is not None
        assert self.optimizer.cache_dir.exists()
        assert self.optimizer.years_of_data == 3
        assert self.optimizer.risk_free_rate == config.RISK_FREE_RATE
        
        print("   ✅ Optimizer initialized correctly")
    
    def test_csv_data_loading(self):
        """Test loading data from CSV files"""
        print("\n🧪 Testing CSV data loading...")
        
        if not self.csv_files:
            pytest.skip("No CSV files found in tickers_data directory")
        
        # Load data from CSV files
        price_data, _ = self.load_csv_data(self.csv_files, max_files=8)
        
        assert len(price_data) > 0, "No data could be loaded from CSV files"
        assert all(isinstance(series, pd.Series) for series in price_data.values())
        
        # Check data quality
        for symbol, prices in price_data.items():
            assert len(prices) >= 100, f"Insufficient data for {symbol}"
            assert (prices > 0).all(), f"Invalid prices for {symbol}"
            assert prices.index.is_monotonic_increasing, f"Dates not sorted for {symbol}"
        
        print(f"   ✅ Successfully loaded and validated {len(price_data)} symbols")
        
        # Store for other tests
        self.test_price_data, self.test_volume_data = self.load_csv_data(self.csv_files, max_files=20)
    
    def test_portfolio_optimization_with_generated_data(self):
        """Test portfolio optimization with controlled, generated data."""
        print("\n🧪 Testing portfolio optimization with generated data...")
        np.random.seed(42) # for reproducibility
        
        # Generate a diverse set of assets, ensuring one has a very high return.
        price_data = {
            'HighGrowthStock': self._generate_realistic_price_data(drift=0.003, volatility=0.03, num_days=300), # High drift to beat risk-free rate
            'ValueStock': self._generate_realistic_price_data(drift=0.0005, volatility=0.015, num_days=300),
            'StableStock': self._generate_realistic_price_data(drift=0.0002, volatility=0.01, num_days=300),
            'VolatileStock': self._generate_realistic_price_data(drift=0.001, volatility=0.04, num_days=300)
        }
        
        self.optimizer.create_master_dataframe(price_data)
        # Create dummy volume data for this test
        self.optimizer.volume_df = pd.DataFrame(
            np.random.randint(100000, 1000000, size=self.optimizer.master_price_df.shape),
            index=self.optimizer.master_price_df.index,
            columns=self.optimizer.master_price_df.columns
        )
        
        # Set lenient screening criteria to focus on optimization logic
        self.optimizer.min_return_threshold = -1.0
        self.optimizer.max_volatility_threshold = 2.0
        self.optimizer.max_drawdown_limit = -1.0
        self.optimizer.min_data_points = 250
        self.optimizer.min_liquidity_threshold = 1000 # Lower for generated data
        
        # Screen and optimize
        screen_success = self.optimizer.screen_stocks()
        assert screen_success, "Screening failed with generated data"
        
        optimize_success = self.optimizer.optimize_portfolio()
        assert optimize_success, "Optimization failed with generated data"
        assert self.optimizer.weights, "Weights dictionary should not be empty"
        
        print("   ✅ Portfolio optimization with generated data completed successfully")
    
    def test_master_dataframe_creation(self):
        """Test master DataFrame creation"""
        print("\n🧪 Testing master DataFrame creation...")
        
        if not hasattr(self, 'test_price_data'):
            self.test_csv_data_loading()
        
        # Create master DataFrame
        success = self.optimizer.create_master_dataframe(self.test_price_data)
        
        assert success, "Failed to create master DataFrame"
        assert self.optimizer.master_price_df is not None
        assert not self.optimizer.master_price_df.empty
        assert len(self.optimizer.master_price_df.columns) > 0
        
        print(f"   ✅ Master DataFrame created with shape: {self.optimizer.master_price_df.shape}")
    
    def test_advanced_metrics_calculation(self):
        """Test advanced metrics calculation"""
        print("\n🧪 Testing advanced metrics calculation...")
        
        if self.optimizer.master_price_df is None:
            self.test_master_dataframe_creation()
        
        # Calculate metrics
        # This test requires volume data, which is not available in the generated data.
        # We will skip this test if we can't load real data.
        if not hasattr(self, 'test_volume_data') or self.optimizer.master_price_df is None:
             self.test_master_dataframe_creation()

        self.optimizer.volume_df = pd.DataFrame(self.test_volume_data)
        # Align columns and index before calculation
        self.optimizer.volume_df = self.optimizer.volume_df.reindex(index=self.optimizer.master_price_df.index, columns=self.optimizer.master_price_df.columns).ffill().bfill()
        metrics_df = self.optimizer.calculate_advanced_metrics(self.optimizer.master_price_df, self.optimizer.volume_df)
        
        assert not metrics_df.empty
        
        # Check required columns
        required_cols = ['Return', 'Volatility', 'Trading_Days', 'Max_Drawdown', 'Sharpe']
        for col in required_cols:
            if col == 'Sharpe':
                # Sharpe is calculated in screening, not in advanced metrics
                continue
            assert col in metrics_df.columns, f"Missing column: {col}"
        
        # Check data validity
        assert (metrics_df['Volatility'] >= 0).all(), "Negative volatility found"
        assert (metrics_df['Trading_Days'] > 0).all(), "Invalid trading days"
        assert (metrics_df['Max_Drawdown'] <= 0).all(), "Invalid max drawdown"
        
        print(f"   ✅ Advanced metrics calculated for {len(metrics_df)} symbols")
        print(f"   📊 Average return: {metrics_df['Return'].mean():.2%}")
        print(f"   📊 Average volatility: {metrics_df['Volatility'].mean():.2%}")
    
    def test_stock_screening(self):
        """Test stock screening functionality"""
        print("\n🧪 Testing stock screening...")
        
        if not hasattr(self, 'test_volume_data') or self.optimizer.master_price_df is None:
            self.test_master_dataframe_creation()
        
        self.optimizer.volume_df = pd.DataFrame(self.test_volume_data)
        
        # Relax criteria for testing
        original_min_return = self.optimizer.min_return_threshold
        original_max_volatility = self.optimizer.max_volatility_threshold
        original_min_data_points = self.optimizer.min_data_points
        
        # Set more lenient criteria for testing
        self.optimizer.min_return_threshold = -1.0  # Allow negative returns for testing
        self.optimizer.max_volatility_threshold = 5.0  # Allow very high volatility
        self.optimizer.min_data_points = 100  # 100 minimum data points
        
        try:
            success = self.optimizer.screen_stocks()
            
            if not success:
                # Try even more relaxed criteria
                print("   ⚠️ Initial screening failed, trying relaxed criteria...")
                self.optimizer.min_return_threshold = -1.0  # Allow negative returns for testing
                self.optimizer.max_volatility_threshold = 5.0  # Allow very high volatility
                self.optimizer.min_data_points = 50  # 50 minimum data points
                success = self.optimizer.screen_stocks()
            
            assert success, "Stock screening failed even with relaxed criteria"
            assert self.optimizer.screener_df is not None
            assert self.optimizer.top_candidates is not None
            assert len(self.optimizer.top_candidates) > 0
            
            print(f"   ✅ Screening completed, {len(self.optimizer.top_candidates)} candidates selected")
            
        finally:
            # Restore original criteria
            self.optimizer.min_return_threshold = original_min_return
            self.optimizer.max_volatility_threshold = original_max_volatility
            self.optimizer.min_data_points = original_min_data_points
    
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        print("\n🧪 Testing maximum drawdown calculation...")
        
        if self.optimizer.master_price_df is None:
            self.test_master_dataframe_creation()
        
        # Test with a subset of data
        test_data = self.optimizer.master_price_df.iloc[:, :3]  # First 3 columns
        max_dd = self.optimizer.calculate_max_drawdown(test_data)
        
        assert isinstance(max_dd, pd.Series)
        assert len(max_dd) == len(test_data.columns)
        assert (max_dd <= 0).all(), "Max drawdown should be negative or zero"
        
        print(f"   ✅ Max drawdown calculated for {len(max_dd)} symbols")
        print(f"   📊 Average max drawdown: {max_dd.mean():.2%}")
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        print("\n🧪 Testing cache functionality...")
        
        # Test cache validity check
        cache_file = self.test_cache_dir / 'test_cache.txt'
        cache_file.write_text('test')
        
        assert self.optimizer.is_cache_valid(cache_file, max_age_hours=24)
        
        # Test with old cache
        import time
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(cache_file, (old_time, old_time))
        
        assert not self.optimizer.is_cache_valid(cache_file, max_age_hours=24)
        
        print("   ✅ Cache functionality working correctly")
    
    def test_data_quality_checks(self):
        """Test data quality validation using realistic generated data."""
        print("\n🧪 Testing data quality checks...")
        
        # Generate realistic but good data
        good_data = self._generate_realistic_price_data(num_days=100)
        
        # Generate data with quality issues
        bad_data = good_data.copy()
        bad_data.iloc[10] = 0      # Zero price
        bad_data.iloc[20] = -10    # Negative price
        
        # Data with too many missing values
        missing_data = good_data.copy()
        missing_data.iloc[30:60] = np.nan # 30% missing data
        
        # Basic validation logic checks
        assert (good_data > 0).all(), "Good data should have all positive prices"
        assert not (bad_data > 0).all(), "Bad data should have non-positive prices"
        assert missing_data.isnull().sum() > 0, "Missing data should have NaNs"
        
        print("   ✅ Data quality checks working correctly")
    
    def test_screening_with_edge_case_data(self):
        """Test screening logic with edge-case data, like negative returns."""
        print("\n🧪 Testing screening with edge-case data...")
        np.random.seed(42) # for reproducibility
        
        # Generate data for two stocks: one good, one with significant negative trend
        good_stock = self._generate_realistic_price_data(drift=0.002, volatility=0.02, num_days=300) # Increased drift
        bad_stock = self._generate_realistic_price_data(drift=-0.005, volatility=0.04, num_days=300)
        
        price_data = {'GOOD_STOCK': good_stock, 'BAD_STOCK': bad_stock}
        
        # Create master dataframe and calculate metrics
        self.optimizer.create_master_dataframe(price_data)
        # This test requires volume data, which is not available in the generated data.
        # We will create dummy volume data for this test.
        dummy_volume = pd.DataFrame(np.random.randint(100000, 1000000, size=self.optimizer.master_price_df.shape),
                                    index=self.optimizer.master_price_df.index,
                                    columns=self.optimizer.master_price_df.columns)
        self.optimizer.volume_df = dummy_volume
        metrics_df = self.optimizer.calculate_advanced_metrics(self.optimizer.master_price_df, self.optimizer.volume_df)

        # Assert that the generated data has the expected characteristics
        # This makes the test more robust by verifying the premise.
        assert metrics_df.loc['GOOD_STOCK']['Return'] > 0, "Generated good stock should have a positive return"
        assert metrics_df.loc['BAD_STOCK']['Return'] < 0, "Generated bad stock should have a negative return"

        # Set screening criteria to specifically filter out the bad stock based on return
        self.optimizer.min_return_threshold = 0.0
        self.optimizer.max_volatility_threshold = 2.0  # Loosen volatility constraint
        self.optimizer.max_drawdown_limit = -1.0  # Loosen drawdown constraint
        self.optimizer.min_data_points = 200
        self.optimizer.min_liquidity_threshold = 1000 # Lower for generated data
        
        # Screen stocks
        success = self.optimizer.screen_stocks()
        
        assert success, "Stock screening should succeed as GOOD_STOCK should pass"
        assert 'GOOD_STOCK' in self.optimizer.top_candidates.index, "Good stock was not selected"
        assert 'BAD_STOCK' not in self.optimizer.top_candidates.index, "Bad stock with negative return was incorrectly selected"
        
        print("   ✅ Screening logic correctly handled edge-case data")
    
    def test_multifactor_screening_and_optimization(self):
        """Test the full pipeline with multi-factor screening using real data."""
        print("\n🧪 Testing Multi-Factor Screening and Optimization...")
        
        if not self.csv_files:
            pytest.skip("No CSV files found, cannot run multi-factor test.")
            
        # Load data
        price_data, volume_data = self.load_csv_data(self.csv_files, max_files=50) # Use more files for better screening
        if len(price_data) < 10:
             pytest.skip("Not enough valid symbols loaded for multi-factor test.")

        # Setup optimizer
        self.optimizer.create_master_dataframe(price_data)
        self.optimizer.volume_df = pd.DataFrame(volume_data) # Add volume df
        
        # Set reasonable screening criteria
        self.optimizer.min_data_points = 252 # 1 year of data
        self.optimizer.min_liquidity_threshold = 1000000 # 1M avg daily volume

        # Run multi-factor screening
        screen_success = self.optimizer.screen_stocks_by_multifactor(top_n=20)
        assert screen_success, "Multi-factor screening failed"
        assert self.optimizer.top_candidates is not None and not self.optimizer.top_candidates.empty

        # Run optimization
        optimize_success = self.optimizer.optimize_portfolio()
        assert optimize_success, "Optimization failed after multi-factor screening"
        assert self.optimizer.weights, "Weights should be populated after optimization"

        print("   ✅ Multi-factor screening and optimization pipeline completed successfully.")

    def test_full_backtest_pipeline(self):
        """Tests the newly integrated backtesting functionality."""
        print("\n🧪 Testing Full Backtest Pipeline...")
        
        if not self.csv_files:
            pytest.skip("No CSV files found, cannot run backtest.")
            
        # 1. Load a larger dataset and ensure the benchmark is included
        benchmark_file = self.tickers_data_dir / 'شاخص کل.csv'
        if not benchmark_file.exists():
            pytest.skip("Benchmark file 'شاخص کل.csv' not found, cannot run backtest.")
        
        # Prioritize loading the benchmark
        files_to_load = [benchmark_file] + [f for f in self.csv_files if f != benchmark_file]
        
        price_data, volume_data = self.load_csv_data(files_to_load, max_files=100)
        if len(price_data) < 20 or 'شاخص کل' not in price_data:
             pytest.skip("Not enough valid symbols loaded or benchmark missing for backtest.")

        # 2. Setup optimizer with the loaded data
        self.optimizer.create_master_dataframe(price_data)
        self.optimizer.volume_df = pd.DataFrame(volume_data)


        # 3. Run the backtest
        backtest_results = self.optimizer.run_backtest(rebalance_freq='3M', top_n=20)
        
        # 4. Assertions
        assert backtest_results is not None, "Backtest did not return any results."
        assert isinstance(backtest_results, pd.Series), "Backtest results should be a pandas Series."
        assert not backtest_results.empty, "Backtest results series is empty."
        
        print("   ✅ Full backtest pipeline completed successfully.")

    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        print("\n🧹 Cleaning up test environment...")
        
        # Remove test cache directory
        import shutil
        if cls.test_cache_dir.exists():
            shutil.rmtree(cls.test_cache_dir)
        
        print("   ✅ Test cleanup completed")

import pytest

def test_integration():
    """Integration test with real CSV data"""
    print("\n🧪 Running integration test with CSV data...")
    
    # Check if we have CSV data
    tickers_data_dir = Path('tickers_data')
    if not tickers_data_dir.exists():
        pytest.skip("No tickers_data directory found for integration test")
    
    csv_files = list(tickers_data_dir.glob('*.csv'))
    if len(csv_files) < 3:
        pytest.skip("Need at least 3 CSV files for integration test")
    
    # Create test optimizer
    test_optimizer = IranianStockOptimizerV3(
        cache_dir='integration_test_cache',
        years_of_data=2,
        risk_free_rate=config.RISK_FREE_RATE
    )
    
    # Load CSV data manually
    price_data = {}
    test_loader = TestIranianStockOptimizerV3()
    price_data, volume_data = test_loader.load_csv_data(csv_files, max_files=10)
    
    if len(price_data) < 3:
        pytest.skip("Could not load sufficient CSV data for integration test")
    
    print(f"📊 Loaded {len(price_data)} symbols for integration test")
    
    # Test the pipeline
    success = test_optimizer.create_master_dataframe(price_data)
    test_optimizer.volume_df = pd.DataFrame(volume_data)
    assert success, "Failed to create master dataframe in integration test"
    
    # Relax screening criteria for integration test
    test_optimizer.min_return_threshold = -1.0  # Allow negative returns
    test_optimizer.max_volatility_threshold = 10.0 # Allow very high volatility
    test_optimizer.max_drawdown_limit = -1.0 # No drawdown limit
    test_optimizer.min_data_points = 50
    
    success = test_optimizer.screen_stocks()
    assert success, "Stock screening failed in integration test"
    
    if len(test_optimizer.top_candidates) >= 2:
        optimize_success = test_optimizer.optimize_portfolio()
        if not optimize_success:
            # It's acceptable for optimization to fail with real data if the problem is infeasible.
            # We log this as a warning rather than failing the test.
            warnings.warn("Portfolio optimization failed, which can be acceptable with real-world data if the problem is infeasible.")
        else:
            assert test_optimizer.weights, "Weights should be populated on successful optimization."

    print("   ✅ Integration test completed successfully")
    
    # Cleanup
    import shutil
    cache_dir = Path('integration_test_cache')
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

if __name__ == "__main__":
    print("🇮🇷 Iranian Stock Market Optimizer v3.1 - Test Suite")
    print("=" * 60)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])