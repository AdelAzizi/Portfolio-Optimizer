#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
کلاس‌های پایه برای مدیریت داده‌ها در سیستم تحلیل جامع

این ماژول شامل کلاس‌های اصلی برای بارگذاری، پیش‌پردازش و مدیریت کش داده‌ها است.
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    کلاس بارگذاری و اعتبارسنجی داده‌های ورودی
    """
    
    def __init__(self, config_path: str = "analysis_config.yaml"):
        """
        مقداردهی اولیه
        
        Args:
            config_path: مسیر فایل تنظیمات
        """
        self.config = self._load_config(config_path)
        self.data_sources = self.config.get('data_sources', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """بارگذاری فایل تنظیمات"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def load_strategy_data(self) -> pd.DataFrame:
        """
        بارگذاری داده‌های استراتژی‌ها
        
        Returns:
            DataFrame حاوی داده‌های استراتژی‌ها
        """
        try:
            strategy_path = self.data_sources.get('strategy_data')
            if not strategy_path or not os.path.exists(strategy_path):
                raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
            
            logger.info(f"📊 Loading strategy data from {strategy_path}")
            data = pd.read_csv(strategy_path)
            
            # اعتبارسنجی داده‌ها
            if self.validate_strategy_data(data):
                logger.info(f"✅ Successfully loaded {len(data)} strategies")
                return data
            else:
                raise ValueError("Strategy data validation failed")
                
        except Exception as e:
            logger.error(f"❌ Error loading strategy data: {e}")
            raise
    
    def load_benchmark_data(self) -> pd.DataFrame:
        """
        بارگذاری داده‌های بنچمارک
        
        Returns:
            DataFrame حاوی داده‌های بنچمارک
        """
        try:
            benchmark_path = self.data_sources.get('benchmark_data')
            if not benchmark_path or not os.path.exists(benchmark_path):
                logger.warning(f"Benchmark file not found: {benchmark_path}")
                return pd.DataFrame()
            
            logger.info(f"📊 Loading benchmark data from {benchmark_path}")
            data = pd.read_csv(benchmark_path)
            
            if self.validate_benchmark_data(data):
                logger.info(f"✅ Successfully loaded benchmark data")
                return data
            else:
                logger.warning("Benchmark data validation failed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error loading benchmark data: {e}")
            return pd.DataFrame()
    
    def load_walk_forward_data(self) -> pd.DataFrame:
        """
        بارگذاری داده‌های Walk-Forward
        
        Returns:
            DataFrame حاوی داده‌های Walk-Forward
        """
        try:
            wf_path = self.data_sources.get('walk_forward_data')
            if not wf_path or not os.path.exists(wf_path):
                logger.warning(f"Walk-Forward file not found: {wf_path}")
                return pd.DataFrame()
            
            logger.info(f"📊 Loading Walk-Forward data from {wf_path}")
            data = pd.read_csv(wf_path)
            
            if self.validate_walk_forward_data(data):
                logger.info(f"✅ Successfully loaded Walk-Forward data")
                return data
            else:
                logger.warning("Walk-Forward data validation failed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error loading Walk-Forward data: {e}")
            return pd.DataFrame()
    
    def load_backtest_data(self, period: str = "1y") -> pd.DataFrame:
        """
        بارگذاری داده‌های بک‌تست
        
        Args:
            period: دوره بک‌تست ("1y" یا "5y")
            
        Returns:
            DataFrame حاوی داده‌های بک‌تست
        """
        try:
            if period == "1y":
                backtest_path = self.data_sources.get('backtest_1y')
            elif period == "5y":
                backtest_path = self.data_sources.get('backtest_5y')
            else:
                raise ValueError(f"Invalid period: {period}")
            
            if not backtest_path or not os.path.exists(backtest_path):
                logger.warning(f"Backtest {period} file not found: {backtest_path}")
                return pd.DataFrame()
            
            logger.info(f"📊 Loading backtest {period} data from {backtest_path}")
            data = pd.read_csv(backtest_path)
            
            if self.validate_backtest_data(data):
                logger.info(f"✅ Successfully loaded backtest {period} data")
                return data
            else:
                logger.warning(f"Backtest {period} data validation failed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error loading backtest {period} data: {e}")
            return pd.DataFrame()
    
    def validate_strategy_data(self, data: pd.DataFrame) -> bool:
        """
        اعتبارسنجی داده‌های استراتژی
        
        Args:
            data: DataFrame برای اعتبارسنجی
            
        Returns:
            True اگر داده‌ها معتبر باشند
        """
        try:
            # بررسی وجود ستون‌های ضروری
            required_columns = ['Sharpe_Ratio', 'Total_Return', 'Volatility']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Required columns not found: {missing_columns}")
                return False
            
            # بررسی مقادیر خالی
            if data[required_columns].isnull().any().any():
                logger.warning("Missing values found in required columns")
            
            # بررسی محدوده مقادیر
            if (data['Sharpe_Ratio'] < -10).any() or (data['Sharpe_Ratio'] > 10).any():
                logger.warning("Unusual values found in Sharpe Ratio")
            
            if (data['Volatility'] < 0).any() or (data['Volatility'] > 2).any():
                logger.warning("Unusual values found in Volatility")
            
            logger.info(f"✅ Strategy data validation successful - {len(data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error validating strategy data: {e}")
            return False
    
    def validate_benchmark_data(self, data: pd.DataFrame) -> bool:
        """اعتبارسنجی داده‌های بنچمارک"""
        try:
            if data.empty:
                return False
            
            # بررسی وجود ستون تاریخ و قیمت
            if 'Date' not in data.columns:
                logger.error("Date column not found in benchmark data")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating benchmark data: {e}")
            return False
    
    def validate_walk_forward_data(self, data: pd.DataFrame) -> bool:
        """اعتبارسنجی داده‌های Walk-Forward"""
        try:
            if data.empty:
                return False
            
            required_columns = ['strategy', 'analysis_date', 'future_sharpe']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Required columns not found in Walk-Forward data: {missing_columns}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating Walk-Forward data: {e}")
            return False
    
    def validate_backtest_data(self, data: pd.DataFrame) -> bool:
        """اعتبارسنجی داده‌های بک‌تست"""
        try:
            if data.empty:
                return False
            
            # بررسی وجود ستون‌های ضروری
            required_columns = ['Candidate_Index', 'Risk_Profile']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Required columns not found in backtest data: {missing_columns}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating backtest data: {e}")
            return False


class DataPreprocessor:
    """
    کلاس پیش‌پردازش و آماده‌سازی داده‌ها
    """
    
    def __init__(self):
        """مقداردهی اولیه"""
        pass
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        تمیزکاری داده‌ها
        
        Args:
            data: DataFrame برای تمیزکاری
            
        Returns:
            DataFrame تمیز شده
        """
        try:
            logger.info("🧹 Starting data cleaning...")
            
            # کپی از داده‌ها
            cleaned_data = data.copy()
            
            # حذف سطرهای کاملاً خالی
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(how='all')
            removed_rows = initial_rows - len(cleaned_data)
            
            if removed_rows > 0:
                logger.info(f"🗑️ Removed {removed_rows} empty rows")
            
            # تبدیل ستون‌های عددی
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # حذف مقادیر نامتناهی
                inf_mask = np.isinf(cleaned_data[col])
                if inf_mask.any():
                    logger.warning(f"Infinite values in column {col} replaced with NaN")
                    cleaned_data.loc[inf_mask, col] = np.nan
            
            # گزارش نهایی
            logger.info(f"✅ Data cleaning completed - {len(cleaned_data)} rows remaining")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"❌ Error in data cleaning: {e}")
            return data
    
    def calculate_returns(self, price_data: pd.DataFrame, 
                         price_column: str = 'Close') -> pd.DataFrame:
        """
        محاسبه بازدهی‌ها
        
        Args:
            price_data: DataFrame حاوی قیمت‌ها
            price_column: نام ستون قیمت
            
        Returns:
            DataFrame حاوی بازدهی‌ها
        """
        try:
            logger.info("📈 Calculating returns...")
            
            if price_column not in price_data.columns:
                raise ValueError(f"Column {price_column} not found")
            
            returns_data = price_data.copy()
            
            # محاسبه بازدهی روزانه
            returns_data['Daily_Return'] = price_data[price_column].pct_change()
            
            # محاسبه بازدهی تجمعی
            returns_data['Cumulative_Return'] = (1 + returns_data['Daily_Return']).cumprod() - 1
            
            # محاسبه بازدهی لگاریتمی
            returns_data['Log_Return'] = np.log(price_data[price_column] / price_data[price_column].shift(1))
            
            logger.info("✅ Returns calculation completed")
            return returns_data
            
        except Exception as e:
            logger.error(f"❌ Error calculating returns: {e}")
            return price_data
    
    def align_timeframes(self, *datasets) -> List[pd.DataFrame]:
        """
        هماهنگ‌سازی بازه‌های زمانی چندین dataset
        
        Args:
            *datasets: DataFrameهای برای هماهنگ‌سازی
            
        Returns:
            لیست DataFrameهای هماهنگ شده
        """
        try:
            logger.info("🔄 Aligning timeframes...")
            
            if len(datasets) < 2:
                logger.warning("At least two datasets required for alignment")
                return list(datasets)
            
            # یافتن ستون‌های تاریخ
            date_columns = []
            for i, df in enumerate(datasets):
                date_col = None
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_col = col
                        break
                date_columns.append(date_col)
            
            # اگر ستون تاریخ یافت نشد، بازگشت بدون تغییر
            if not any(date_columns):
                logger.warning("No date columns found - alignment skipped")
                return list(datasets)
            
            # تبدیل ستون‌های تاریخ
            aligned_datasets = []
            for i, (df, date_col) in enumerate(zip(datasets, date_columns)):
                if date_col:
                    df_copy = df.copy()
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                    aligned_datasets.append(df_copy)
                else:
                    aligned_datasets.append(df)
            
            logger.info("✅ Timeframe alignment completed")
            return aligned_datasets
            
        except Exception as e:
            logger.error(f"❌ Error in timeframe alignment: {e}")
            return list(datasets)
    
    def handle_missing_data(self, data: pd.DataFrame, 
                          method: str = 'forward_fill') -> pd.DataFrame:
        """
        مدیریت داده‌های گمشده
        
        Args:
            data: DataFrame حاوی داده‌های گمشده
            method: روش پر کردن ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            
        Returns:
            DataFrame با داده‌های پر شده
        """
        try:
            logger.info(f"🔧 Handling missing data with method: {method}...")
            
            processed_data = data.copy()
            missing_count = processed_data.isnull().sum().sum()
            
            if missing_count == 0:
                logger.info("No missing data found")
                return processed_data
            
            logger.info(f"📊 Found {missing_count} missing values")
            
            if method == 'forward_fill':
                processed_data = processed_data.fillna(method='ffill')
            elif method == 'backward_fill':
                processed_data = processed_data.fillna(method='bfill')
            elif method == 'interpolate':
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                processed_data[numeric_columns] = processed_data[numeric_columns].interpolate()
            elif method == 'drop':
                processed_data = processed_data.dropna()
            else:
                logger.warning(f"Invalid method: {method}")
                return data
            
            final_missing = processed_data.isnull().sum().sum()
            logger.info(f"✅ Filled {missing_count - final_missing} missing values")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error handling missing data: {e}")
            return data


class AnalysisCache:
    """
    کلاس مدیریت کش برای تحلیل‌ها
    """
    
    def __init__(self, cache_dir: str = "cache", 
                 expiry_hours: int = 24):
        """
        مقداردهی اولیه
        
        Args:
            cache_dir: مسیر پوشه کش
            expiry_hours: مدت انقضای کش (ساعت)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.expiry_hours = expiry_hours
        
        logger.info(f"🗄️ Cache system initialized - path: {cache_dir}")
    
    def generate_cache_key(self, module_name: str, data_hash: str, 
                          params: Dict = None) -> str:
        """
        تولید کلید کش
        
        Args:
            module_name: نام ماژول تحلیل
            data_hash: هش داده‌های ورودی
            params: پارامترهای تحلیل
            
        Returns:
            کلید کش
        """
        try:
            params = params or {}
            params_str = json.dumps(params, sort_keys=True)
            combined = f"{module_name}_{data_hash}_{params_str}"
            cache_key = hashlib.md5(combined.encode()).hexdigest()
            
            logger.debug(f"🔑 Cache key generated: {cache_key[:8]}...")
            return cache_key
            
        except Exception as e:
            logger.error(f"❌ Error generating cache key: {e}")
            return f"{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_data_hash(self, data: Union[pd.DataFrame, Dict, List]) -> str:
        """
        محاسبه هش داده‌ها
        
        Args:
            data: داده برای محاسبه هش
            
        Returns:
            هش داده
        """
        try:
            if isinstance(data, pd.DataFrame):
                # برای DataFrame از shape و چند سطر اول استفاده می‌کنیم
                data_str = f"{data.shape}_{data.head().to_string()}"
            elif isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            
            return hashlib.md5(data_str.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"❌ Error calculating data hash: {e}")
            return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def is_cached(self, cache_key: str) -> bool:
        """
        بررسی وجود کش
        
        Args:
            cache_key: کلید کش
            
        Returns:
            True اگر کش موجود و معتبر باشد
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return False
            
            # بررسی انقضای کش
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            expiry_time = file_time + timedelta(hours=self.expiry_hours)
            
            if datetime.now() > expiry_time:
                logger.info(f"🕐 Cache expired: {cache_key[:8]}...")
                cache_file.unlink()  # حذف کش منقضی
                return False
            
            logger.info(f"✅ Valid cache found: {cache_key[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking cache: {e}")
            return False
    
    def save_to_cache(self, cache_key: str, results: Any) -> bool:
        """
        ذخیره در کش
        
        Args:
            cache_key: کلید کش
            results: نتایج برای ذخیره
            
        Returns:
            True اگر ذخیره موفق باشد
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"💾 Results saved to cache: {cache_key[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving to cache: {e}")
            return False
    
    def load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        بارگذاری از کش
        
        Args:
            cache_key: کلید کش
            
        Returns:
            نتایج از کش یا None
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            
            logger.info(f"📂 Results loaded from cache: {cache_key[:8]}...")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error loading from cache: {e}")
            return None
    
    def clear_cache(self, pattern: str = None) -> int:
        """
        پاک‌سازی کش
        
        Args:
            pattern: الگو برای فیلتر فایل‌ها (اختیاری)
            
        Returns:
            تعداد فایل‌های حذف شده
        """
        try:
            deleted_count = 0
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if pattern is None or pattern in cache_file.name:
                    cache_file.unlink()
                    deleted_count += 1
            
            logger.info(f"🗑️ Deleted {deleted_count} cache files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            return 0
    
    def get_cache_info(self) -> Dict:
        """
        اطلاعات کش
        
        Returns:
            دیکشنری حاوی اطلاعات کش
        """
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            info = {
                'cache_directory': str(self.cache_dir),
                'total_files': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'expiry_hours': self.expiry_hours,
                'oldest_file': None,
                'newest_file': None
            }
            
            if cache_files:
                oldest = min(cache_files, key=lambda f: f.stat().st_mtime)
                newest = max(cache_files, key=lambda f: f.stat().st_mtime)
                
                info['oldest_file'] = {
                    'name': oldest.name,
                    'age_hours': round((datetime.now().timestamp() - oldest.stat().st_mtime) / 3600, 1)
                }
                info['newest_file'] = {
                    'name': newest.name,
                    'age_hours': round((datetime.now().timestamp() - newest.stat().st_mtime) / 3600, 1)
                }
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error getting cache info: {e}")
            return {}


def test_data_management():
    """
    تست کلاس‌های مدیریت داده
    """
    logger.info("🧪 Starting data management classes test...")
    
    try:
        # تست DataLoader
        logger.info("📊 Testing DataLoader...")
        loader = DataLoader()
        
        # تست بارگذاری داده‌های استراتژی
        try:
            strategy_data = loader.load_strategy_data()
            logger.info(f"✅ Strategy loading successful - {len(strategy_data)} records")
        except Exception as e:
            logger.warning(f"⚠️ Strategy loading failed: {e}")
        
        # تست DataPreprocessor
        logger.info("🧹 Testing DataPreprocessor...")
        preprocessor = DataPreprocessor()
        
        # ایجاد داده نمونه برای تست
        sample_data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1.1, 2.2, 3.3, np.inf, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        cleaned_data = preprocessor.clean_data(sample_data)
        logger.info(f"✅ Data cleaning successful - {len(cleaned_data)} rows")
        
        # تست AnalysisCache
        logger.info("🗄️ Testing AnalysisCache...")
        cache = AnalysisCache()
        
        # تست ذخیره و بازیابی
        test_data = {'test': 'data', 'number': 123}
        data_hash = cache.get_data_hash(test_data)
        cache_key = cache.generate_cache_key('test_module', data_hash)
        
        # ذخیره در کش
        if cache.save_to_cache(cache_key, test_data):
            logger.info("✅ Cache save successful")
            
            # بازیابی از کش
            cached_data = cache.load_from_cache(cache_key)
            if cached_data == test_data:
                logger.info("✅ Cache load successful")
            else:
                logger.error("❌ Cache load failed")
        
        # اطلاعات کش
        cache_info = cache.get_cache_info()
        logger.info(f"📊 Cache info: {cache_info}")
        
        logger.info("🎉 Data management classes test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test error: {e}")


if __name__ == "__main__":
    test_data_management()