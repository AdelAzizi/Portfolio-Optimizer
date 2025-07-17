# مستند طراحی سیستم - تحلیل جامع استراتژی‌های سرمایه‌گذاری

## نمای کلی

این سیستم یک پلتفرم تحلیلی جامع برای ارزیابی و بهینه‌سازی استراتژی‌های سرمایه‌گذاری است. سیستم از معماری modular استفاده می‌کند که هر ماژول مسئول یک جنبه خاص از تحلیل است و در نهایت تمام نتایج در یک گزارش جامع ترکیب می‌شوند.

## معماری کلی

### معماری سطح بالا

```
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Orchestrator                        │
│                  (run_comprehensive_analysis.py)               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Analysis Modules                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Prediction  │  │ Stability   │  │ Market      │             │
│  │ Horizons    │  │ Analysis    │  │ Regime      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Strategy    │  │ Parameter   │  │ Benchmark   │             │
│  │ Combination │  │ Sensitivity │  │ Comparison  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Advanced    │  │ Walk-Forward│  │ Economic    │             │
│  │ Risk        │  │ Analysis    │  │ Factors     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Report Generator                                │
│              (generate_final_recommendations.py)               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Output Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    PDF      │  │    HTML     │  │    JSON     │             │
│  │   Report    │  │  Dashboard  │  │    Data     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### جریان داده

```
Input Data → Data Validation → Analysis Modules → Results Aggregation → Report Generation → Output Files
```

## اجزا و رابط‌ها

### 1. Data Management Layer

#### 1.1 Data Loader
```python
class DataLoader:
    """بارگذاری و اعتبارسنجی داده‌های ورودی"""
    
    def load_strategy_data(self) -> pd.DataFrame:
        # بارگذاری داده‌های استراتژی از فایل‌های CSV
        pass
    
    def load_benchmark_data(self) -> pd.DataFrame:
        # بارگذاری داده‌های بنچمارک
        pass
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        # اعتبارسنجی کیفیت داده‌ها
        pass
```

#### 1.2 Data Preprocessor
```python
class DataPreprocessor:
    """پیش‌پردازش و آماده‌سازی داده‌ها"""
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # تمیزکاری داده‌ها
        pass
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        # محاسبه بازدهی‌ها
        pass
    
    def align_timeframes(self, *datasets) -> List[pd.DataFrame]:
        # هماهنگ‌سازی بازه‌های زمانی
        pass
```

### 2. Analysis Modules

#### 2.1 Prediction Horizons Analyzer
```python
class PredictionHorizonsAnalyzer:
    """تحلیل قدرت پیش‌بینی افق‌های زمانی"""
    
    def analyze_correlations(self, data: pd.DataFrame) -> Dict:
        # محاسبه همبستگی بین افق‌های مختلف
        correlations = {}
        for horizon in ['1y', '3y', '5y']:
            corr, p_value = self.calculate_correlation(data, horizon)
            correlations[horizon] = {
                'correlation': corr,
                'p_value': p_value,
                'significance': 'significant' if p_value < 0.05 else 'not_significant'
            }
        return correlations
    
    def calculate_prediction_accuracy(self, data: pd.DataFrame) -> Dict:
        # محاسبه دقت پیش‌بینی
        pass
    
    def generate_visualizations(self, results: Dict) -> None:
        # تولید نمودارهای تحلیل
        pass
```

#### 2.2 Strategy Stability Analyzer
```python
class StrategyStabilityAnalyzer:
    """تحلیل ثبات استراتژی‌ها"""
    
    def calculate_stability_metrics(self, data: pd.DataFrame) -> Dict:
        # محاسبه معیارهای ثبات
        stability_metrics = {}
        for strategy in data.columns:
            rolling_sharpe = self.calculate_rolling_sharpe(data[strategy])
            stability_metrics[strategy] = {
                'sharpe_volatility': rolling_sharpe.std(),
                'consistency_score': self.calculate_consistency_score(rolling_sharpe),
                'failure_periods': self.identify_failure_periods(data[strategy])
            }
        return stability_metrics
    
    def identify_failure_patterns(self, data: pd.DataFrame) -> Dict:
        # شناسایی الگوهای شکست
        pass
    
    def rank_strategies_by_stability(self, metrics: Dict) -> List:
        # رتبه‌بندی بر اساس ثبات
        pass
```

#### 2.3 Market Regime Analyzer
```python
class MarketRegimeAnalyzer:
    """تحلیل رژیم بازار"""
    
    def detect_market_regimes(self, benchmark_data: pd.DataFrame) -> pd.Series:
        # تشخیص رژیم‌های بازار
        returns = benchmark_data.pct_change()
        
        # استفاده از Hidden Markov Model یا روش‌های آماری
        regimes = self.hmm_regime_detection(returns)
        return regimes
    
    def analyze_performance_by_regime(self, strategy_data: pd.DataFrame, 
                                    regimes: pd.Series) -> Dict:
        # تحلیل عملکرد در هر رژیم
        performance_by_regime = {}
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_performance = {}
            
            for strategy in strategy_data.columns:
                regime_returns = strategy_data[strategy][regime_mask]
                regime_performance[strategy] = {
                    'sharpe_ratio': self.calculate_sharpe(regime_returns),
                    'total_return': regime_returns.sum(),
                    'volatility': regime_returns.std() * np.sqrt(252)
                }
            
            performance_by_regime[regime] = regime_performance
        
        return performance_by_regime
    
    def identify_regime_robust_strategies(self, performance: Dict) -> List:
        # شناسایی استراتژی‌های مقاوم
        pass
```

#### 2.4 Strategy Combination Analyzer
```python
class StrategyCombinationAnalyzer:
    """تحلیل ترکیب استراتژی‌ها"""
    
    def optimize_portfolio_weights(self, returns: pd.DataFrame) -> Dict:
        # بهینه‌سازی وزن‌های پرتفوی
        from scipy.optimize import minimize
        
        def objective(weights):
            portfolio_return = (returns * weights).sum(axis=1)
            return -self.calculate_sharpe(portfolio_return)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(returns.columns))]
        
        result = minimize(objective, 
                         x0=np.ones(len(returns.columns)) / len(returns.columns),
                         bounds=bounds, 
                         constraints=constraints)
        
        return {
            'optimal_weights': dict(zip(returns.columns, result.x)),
            'expected_sharpe': -result.fun,
            'diversification_benefit': self.calculate_diversification_benefit(returns, result.x)
        }
    
    def analyze_correlation_matrix(self, returns: pd.DataFrame) -> Dict:
        # تحلیل ماتریس همبستگی
        pass
    
    def monte_carlo_simulation(self, returns: pd.DataFrame, n_simulations: int = 10000) -> Dict:
        # شبیه‌سازی مونت کارلو
        pass
```

#### 2.5 Parameter Sensitivity Analyzer
```python
class ParameterSensitivityAnalyzer:
    """تحلیل حساسیت پارامترها"""
    
    def analyze_parameter_sensitivity(self, strategy_configs: Dict) -> Dict:
        # تحلیل حساسیت پارامترها
        sensitivity_results = {}
        
        for strategy_name, config in strategy_configs.items():
            strategy_sensitivity = {}
            
            for param_name, param_value in config.items():
                if isinstance(param_value, (int, float)):
                    sensitivity = self.calculate_parameter_sensitivity(
                        strategy_name, param_name, param_value
                    )
                    strategy_sensitivity[param_name] = sensitivity
            
            sensitivity_results[strategy_name] = strategy_sensitivity
        
        return sensitivity_results
    
    def identify_critical_parameters(self, sensitivity_results: Dict) -> Dict:
        # شناسایی پارامترهای حیاتی
        pass
    
    def optimize_parameter_ranges(self, sensitivity_results: Dict) -> Dict:
        # بهینه‌سازی محدوده پارامترها
        pass
```

### 3. Advanced Analysis Modules

#### 3.1 Advanced Risk Analyzer
```python
class AdvancedRiskAnalyzer:
    """تحلیل ریسک پیشرفته"""
    
    def calculate_var_cvar(self, returns: pd.DataFrame, confidence_level: float = 0.05) -> Dict:
        # محاسبه VaR و CVaR
        risk_metrics = {}
        
        for strategy in returns.columns:
            strategy_returns = returns[strategy].dropna()
            
            # Value at Risk
            var = np.percentile(strategy_returns, confidence_level * 100)
            
            # Conditional Value at Risk
            cvar = strategy_returns[strategy_returns <= var].mean()
            
            risk_metrics[strategy] = {
                'VaR_5%': var,
                'CVaR_5%': cvar,
                'max_drawdown': self.calculate_max_drawdown(strategy_returns),
                'downside_deviation': self.calculate_downside_deviation(strategy_returns)
            }
        
        return risk_metrics
    
    def analyze_tail_risk(self, returns: pd.DataFrame) -> Dict:
        # تحلیل ریسک دم توزیع
        pass
    
    def stress_testing(self, returns: pd.DataFrame, scenarios: List[Dict]) -> Dict:
        # تست استرس
        pass
```

#### 3.2 Walk Forward Analyzer
```python
class WalkForwardAnalyzer:
    """تحلیل Walk-Forward بهبود یافته"""
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, 
                                 window_size: int = 252,
                                 step_size: int = 63) -> Dict:
        # اجرای تحلیل Walk-Forward
        results = {
            'in_sample_performance': [],
            'out_of_sample_performance': [],
            'overfitting_metrics': {}
        }
        
        for start_idx in range(0, len(data) - window_size, step_size):
            end_idx = start_idx + window_size
            
            # داده‌های In-Sample
            in_sample_data = data.iloc[start_idx:end_idx]
            
            # داده‌های Out-of-Sample
            out_sample_data = data.iloc[end_idx:end_idx + step_size]
            
            # محاسبه عملکرد
            in_sample_perf = self.calculate_performance_metrics(in_sample_data)
            out_sample_perf = self.calculate_performance_metrics(out_sample_data)
            
            results['in_sample_performance'].append(in_sample_perf)
            results['out_of_sample_performance'].append(out_sample_perf)
        
        # تحلیل Overfitting
        results['overfitting_metrics'] = self.analyze_overfitting(results)
        
        return results
    
    def detect_overfitting(self, in_sample_results: List, out_sample_results: List) -> Dict:
        # تشخیص Overfitting
        pass
```

### 4. Report Generation Layer

#### 4.1 Report Generator
```python
class ReportGenerator:
    """تولید گزارش نهایی"""
    
    def __init__(self):
        self.analysis_results = {}
        self.recommendations = []
    
    def aggregate_results(self, module_results: Dict) -> Dict:
        # تجمیع نتایج تمام ماژول‌ها
        aggregated = {
            'executive_summary': self.create_executive_summary(module_results),
            'key_findings': self.extract_key_findings(module_results),
            'strategy_rankings': self.create_strategy_rankings(module_results),
            'risk_assessment': self.create_risk_assessment(module_results),
            'recommendations': self.generate_recommendations(module_results)
        }
        return aggregated
    
    def generate_recommendations(self, results: Dict) -> List[Dict]:
        # تولید توصیه‌های عملی
        recommendations = []
        
        # توصیه بر اساس بهترین افق زمانی
        best_horizon = self.find_best_prediction_horizon(results['prediction_horizons'])
        recommendations.append({
            'type': 'prediction_horizon',
            'recommendation': f'استفاده از افق {best_horizon} برای انتخاب استراتژی',
            'confidence': 'high' if results['prediction_horizons'][best_horizon]['correlation'] > 0.5 else 'medium'
        })
        
        # توصیه بر اساس ثبات
        stable_strategies = self.find_stable_strategies(results['stability_analysis'])
        recommendations.append({
            'type': 'strategy_selection',
            'recommendation': f'اولویت با استراتژی‌های پایدار: {stable_strategies[:3]}',
            'confidence': 'high'
        })
        
        return recommendations
    
    def create_visualizations(self, results: Dict) -> Dict:
        # تولید نمودارها
        pass
    
    def export_to_formats(self, report: Dict, output_dir: str) -> Dict:
        # صادرات به فرمت‌های مختلف
        output_files = {}
        
        # JSON
        json_file = os.path.join(output_dir, 'comprehensive_analysis_report.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        output_files['json'] = json_file
        
        # HTML
        html_file = self.generate_html_report(report, output_dir)
        output_files['html'] = html_file
        
        # PDF
        pdf_file = self.generate_pdf_report(report, output_dir)
        output_files['pdf'] = pdf_file
        
        return output_files
```

## مدل‌های داده

### 1. Analysis Result Models
```python
@dataclass
class AnalysisResult:
    module_name: str
    execution_time: float
    status: str  # 'success', 'failed', 'warning'
    results: Dict
    visualizations: List[str]
    recommendations: List[str]

@dataclass
class StrategyMetrics:
    name: str
    sharpe_ratio: float
    total_return: float
    volatility: float
    max_drawdown: float
    var_5: float
    cvar_5: float
    stability_score: float

@dataclass
class MarketRegime:
    regime_type: str  # 'bull', 'bear', 'neutral'
    start_date: datetime
    end_date: datetime
    market_return: float
    volatility: float
```

### 2. Configuration Models
```python
@dataclass
class AnalysisConfig:
    input_data_path: str
    output_directory: str
    analysis_modules: List[str]
    confidence_level: float = 0.05
    walk_forward_window: int = 252
    monte_carlo_simulations: int = 10000
    
@dataclass
class ReportConfig:
    include_visualizations: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['json', 'html', 'pdf'])
    language: str = 'fa'  # Persian
    template_style: str = 'professional'
```

## مدیریت خطا و Logging

### 1. Error Handling Strategy
```python
class AnalysisError(Exception):
    """خطاهای مربوط به تحلیل"""
    pass

class DataValidationError(AnalysisError):
    """خطاهای اعتبارسنجی داده"""
    pass

class InsufficientDataError(AnalysisError):
    """خطای کمبود داده"""
    pass

class ErrorHandler:
    def __init__(self):
        self.error_log = []
    
    def handle_module_error(self, module_name: str, error: Exception) -> bool:
        """مدیریت خطاهای ماژول‌ها"""
        error_info = {
            'module': module_name,
            'error_type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now(),
            'recoverable': self.is_recoverable_error(error)
        }
        
        self.error_log.append(error_info)
        
        if error_info['recoverable']:
            logger.warning(f"خطای قابل بازیابی در {module_name}: {error}")
            return True
        else:
            logger.error(f"خطای حیاتی در {module_name}: {error}")
            return False
```

### 2. Progress Tracking
```python
class ProgressTracker:
    """ردیابی پیشرفت تحلیل"""
    
    def __init__(self, total_modules: int):
        self.total_modules = total_modules
        self.completed_modules = 0
        self.current_module = None
        self.start_time = datetime.now()
    
    def start_module(self, module_name: str):
        self.current_module = module_name
        logger.info(f"🔄 شروع {module_name}...")
    
    def complete_module(self, module_name: str, success: bool = True):
        self.completed_modules += 1
        status = "✅" if success else "❌"
        progress = (self.completed_modules / self.total_modules) * 100
        
        logger.info(f"{status} {module_name} کامل شد - پیشرفت: {progress:.1f}%")
    
    def get_eta(self) -> str:
        """تخمین زمان باقی‌مانده"""
        if self.completed_modules == 0:
            return "نامشخص"
        
        elapsed = datetime.now() - self.start_time
        avg_time_per_module = elapsed / self.completed_modules
        remaining_modules = self.total_modules - self.completed_modules
        eta = avg_time_per_module * remaining_modules
        
        return str(eta).split('.')[0]  # حذف میکروثانیه‌ها
```

## Performance Optimization

### 1. Caching Strategy
```python
class AnalysisCache:
    """سیستم کش برای تحلیل‌ها"""
    
    def __init__(self, cache_dir: str = "tests/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, module_name: str, data_hash: str, params: Dict) -> str:
        """تولید کلید کش"""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{module_name}_{data_hash}_{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def is_cached(self, cache_key: str) -> bool:
        """بررسی وجود کش"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def save_to_cache(self, cache_key: str, results: Dict):
        """ذخیره در کش"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
    
    def load_from_cache(self, cache_key: str) -> Dict:
        """بارگذاری از کش"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
```

### 2. Parallel Processing
```python
class ParallelAnalyzer:
    """اجرای موازی تحلیل‌ها"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or os.cpu_count()
    
    def run_parallel_analysis(self, analysis_modules: List, data: pd.DataFrame) -> Dict:
        """اجرای موازی ماژول‌های تحلیل"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for module in analysis_modules:
                if module.can_run_parallel:
                    future = executor.submit(module.analyze, data)
                    futures[module.name] = future
            
            results = {}
            for module_name, future in futures.items():
                try:
                    results[module_name] = future.result(timeout=300)  # 5 دقیقه timeout
                except Exception as e:
                    logger.error(f"خطا در اجرای موازی {module_name}: {e}")
                    results[module_name] = None
            
            return results
```

## Testing Strategy

### 1. Unit Tests
```python
class TestAnalysisModules:
    def test_prediction_horizons_analyzer(self):
        # تست تحلیل افق‌های زمانی
        pass
    
    def test_stability_analyzer(self):
        # تست تحلیل ثبات
        pass
    
    def test_regime_analyzer(self):
        # تست تحلیل رژیم بازار
        pass

class TestDataIntegrity:
    def test_data_validation(self):
        # تست اعتبارسنجی داده‌ها
        pass
    
    def test_missing_data_handling(self):
        # تست مدیریت داده‌های گمشده
        pass
```

### 2. Integration Tests
```python
class TestEndToEndAnalysis:
    def test_complete_analysis_pipeline(self):
        # تست کامل پایپ‌لاین
        pass
    
    def test_report_generation(self):
        # تست تولید گزارش
        pass
```

## Deployment and Monitoring

### 1. Configuration Management
```python
class ConfigManager:
    """مدیریت تنظیمات"""
    
    def __init__(self, config_file: str = "analysis_config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """بارگذاری تنظیمات"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """تنظیمات پیش‌فرض"""
        return {
            'data_sources': {
                'strategy_data': 'tests/data/strategy_test_results.csv',
                'benchmark_data': 'tests/data/benchmark_data.csv'
            },
            'analysis_parameters': {
                'confidence_level': 0.05,
                'walk_forward_window': 252,
                'monte_carlo_simulations': 10000
            },
            'output_settings': {
                'base_directory': 'tests/results',
                'export_formats': ['json', 'html', 'pdf'],
                'include_visualizations': True
            }
        }
```

این طراحی جامع تمام جنبه‌های سیستم تحلیل را پوشش می‌دهد و مسیر واضحی برای پیاده‌سازی فراهم می‌کند.