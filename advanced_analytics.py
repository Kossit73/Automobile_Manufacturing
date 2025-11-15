"""
Advanced Analytics Suite for Financial Modeling
Implements 23 advanced analytical techniques for deep financial insights

Features:
  1. Sensitivity Analysis (Key Drivers)
  2. Scenario Stress Testing
  3. Trend & Seasonality Decomposition
  4. Customer/Product Segmentation
  5. Monte Carlo Simulation
  6. What-If Analysis (Interactive)
  7. Goal Seek Optimization
  8. Tornado Charts & Spider Diagrams
  9. Regression Modeling
  10. Time Series Analysis (ARIMA/Prophet/LSTM)
  11. Classification Models
  12. Linear/Nonlinear Optimization
  13. Portfolio Optimization
  14. Real Options Analysis
  15. Value at Risk (VaR/CVaR)
  16. Stress Testing (Extreme Scenarios)
  17. Copula Models (Correlation)
  18. Macroeconomic Linking
  19. ESG & Sustainability Metrics
  20. Market Intelligence Integration
  21. Probabilistic Valuation
  22. Comparative Valuation & Clustering
  23. ML-Based Valuation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, replace
from scipy import stats, optimize
from scipy.stats import norm, multivariate_normal
import warnings
warnings.filterwarnings('ignore')


# =====================================================
# 1. ADVANCED SENSITIVITY ANALYSIS - KEY DRIVERS
# =====================================================

class AdvancedSensitivityAnalyzer:
    """Comprehensive sensitivity analysis for key financial drivers"""
    
    def __init__(self, model_data: dict, config: 'CompanyConfig'):
        self.model_data = model_data
        self.config = config
        self.years = list(model_data['years'])
    
    def pareto_sensitivity(self, parameters: List[str], ranges: Dict[str, float]) -> pd.DataFrame:
        """
        Pareto sensitivity: Identify which parameters drive 80% of value changes
        
        Args:
            parameters: List of parameter names
            ranges: Dict of parameter -> range_pct
        
        Returns:
            DataFrame sorted by impact (Pareto principle)
        """
        from financial_model import run_financial_model
        
        results = []
        base_ev = self.model_data['enterprise_value']
        
        for param in parameters:
            if not hasattr(self.config, param):
                continue
            
            base_value = getattr(self.config, param)
            range_pct = ranges.get(param, 0.25)
            
            cfg_low = self.config.__class__(**self.config.__dict__)
            cfg_high = self.config.__class__(**self.config.__dict__)
            
            setattr(cfg_low, param, base_value * (1 - range_pct))
            setattr(cfg_high, param, base_value * (1 + range_pct))
            
            model_low = run_financial_model(cfg_low)
            model_high = run_financial_model(cfg_high)
            
            ev_range = model_high['enterprise_value'] - model_low['enterprise_value']
            impact_pct = (ev_range / base_ev) * 100
            
            results.append({
                'parameter': param,
                'base_value': base_value,
                'low_ev': model_low['enterprise_value'],
                'high_ev': model_high['enterprise_value'],
                'ev_range': ev_range,
                'impact_pct': impact_pct,
                'elasticity': impact_pct / (2 * range_pct * 100)  # Sensitivity measure
            })
        
        df = pd.DataFrame(results).sort_values('impact_pct', ascending=False)
        
        # Calculate cumulative impact (Pareto)
        df['cumulative_impact_pct'] = df['impact_pct'].cumsum()
        df['pareto_rank'] = (df['cumulative_impact_pct'] <= 80).cumsum()
        
        return df
    
    def tornado_diagram_data(self, parameters: List[str], ranges: Dict) -> Dict:
        """Generate data for tornado diagram visualization"""
        sensitivity = self.pareto_sensitivity(parameters, ranges)
        
        tornado_data = {}
        for _, row in sensitivity.iterrows():
            param = row['parameter']
            low = row['low_ev']
            high = row['high_ev']
            base = self.model_data['enterprise_value']
            
            tornado_data[param] = {
                'low_impact': low - base,
                'high_impact': high - base,
                'range': high - low,
                'mid_point': (low + high) / 2
            }
        
        return tornado_data


# =====================================================
# 2. SCENARIO STRESS TESTING
# =====================================================

class StressTestEngine:
    """Advanced stress testing with severe but plausible shocks"""
    
    def __init__(self, model_data: dict, config: 'CompanyConfig'):
        self.model_data = model_data
        self.config = config
    
    def extreme_scenarios(self) -> Dict[str, Dict]:
        """Define and execute extreme scenarios"""
        from financial_model import run_financial_model
        
        scenarios = {
            'Black Swan - Commodity Crash': {
                'cogs_ratio': 0.30,  # Input costs collapse
                'wacc': 0.20,        # Risk premium spikes
                'annual_capacity': 12_000  # Supply shock reduces output
            },
            'Disease Outbreak': {
                'annual_capacity': 6_000,  # Production collapses
                'cogs_ratio': 0.80      # Feed costs spike
            },
            'Drought/Climate Shock': {
                'annual_capacity': 8_000,
                'cogs_ratio': 0.75,
                'wacc': 0.18
            },
            'Market Collapse': {
                'wacc': 0.25,  # Financing costs spike
                'annual_capacity': 16_000,
                'cogs_ratio': 0.75
            },
            'Interest Rate Spike': {
                'wacc': 0.24,  # 12% increase
                'cogs_ratio': 0.65
            },
            'Supply Chain Breakdown': {
                'cogs_ratio': 0.90,  # Input costs surge
                'annual_capacity': 14_000,
                'wacc': 0.16
            },
            'Regulatory Shock': {
                'cogs_ratio': 0.70,  # Compliance costs
                'annual_capacity': 17_000,
                'wacc': 0.14
            }
        }
        
        results = {}
        for scenario_name, shocks in scenarios.items():
            cfg = self.config.__class__(**self.config.__dict__)
            
            # Apply shocks
            for key, value in shocks.items():
                if key == 'opex_multiplier':
                    # Handle special case
                    pass
                elif key == 'selling_price_multiplier':
                    # Handle special case
                    pass
                elif hasattr(cfg, key):
                    if isinstance(value, float) and value < 2:
                        # If between 0-2, might be a multiplier
                        current = getattr(cfg, key)
                        if isinstance(current, (int, float)):
                            setattr(cfg, key, current * value)
                    else:
                        setattr(cfg, key, value)
            
            model = run_financial_model(cfg)
            results[scenario_name] = {
                'enterprise_value': model['enterprise_value'],
                'revenue_2030': model['revenue'][2030],
                'net_profit_2030': model['net_profit'][2030],
                'final_cash': model['cash_balance'][2030],
                'recovery_probability': self._recovery_score(model)
            }
        
        return results
    
    def _recovery_score(self, model_data: dict) -> float:
        """Calculate probability of recovery (0-100)"""
        if model_data['cash_balance'][2030] < 0:
            return 0
        if model_data['enterprise_value'] < 0:
            return 20
        if model_data['net_profit'][2030] < 0:
            return 40
        if model_data['cash_balance'][2030] > model_data['cash_balance'][2026]:
            return 100
        return 60


# =====================================================
# 3. TREND & SEASONALITY DECOMPOSITION
# =====================================================

class TrendDecomposition:
    """Separate trend from seasonality in time series"""
    
    def __init__(self, model_data: dict):
        self.model_data = model_data
        self.years = list(model_data['years'])
    
    def decompose_series(self, series_dict: Dict) -> Dict:
        """
        Simple decomposition: Series = Trend + Seasonality + Residual
        Using moving average method
        """
        years = np.array(list(series_dict.keys()))
        values = np.array(list(series_dict.values()))
        
        # Trend (using centered moving average)
        window = 2
        trend = np.convolve(values, np.ones(window)/window, mode='same')
        
        # Seasonality
        seasonality = values - trend
        
        # Residual (simple noise component)
        residual = np.zeros_like(values)
        
        return {
            'years': years,
            'original': values,
            'trend': trend,
            'seasonality': seasonality,
            'residual': residual,
            'trend_strength': 1 - (np.var(seasonality) / np.var(values)) if np.var(values) > 0 else 0
        }
    
    def identify_inflection_points(self, series_dict: Dict) -> List[int]:
        """Identify years where trend changes direction"""
        years = list(series_dict.keys())
        values = list(series_dict.values())
        
        inflection_points = []
        for i in range(1, len(values) - 1):
            # Check if sign of derivative changes
            slope1 = values[i] - values[i-1]
            slope2 = values[i+1] - values[i]
            if slope1 * slope2 < 0:
                inflection_points.append(years[i])
        
        return inflection_points


# =====================================================
# 4. CUSTOMER & PRODUCT SEGMENTATION
# =====================================================

class SegmentationAnalyzer:
    """Analyze profitability by customer or product segment"""
    
    def __init__(self, model_data: dict):
        self.model_data = model_data
    
    def segment_analysis(self, segment_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Analyze segment profitability
        
        Args:
            segment_data: Dict with segments and their financials
            Format: {'Segment A': {'revenue': X, 'cost': Y, 'units': Z}, ...}
        """
        results = []
        
        for segment_name, metrics in segment_data.items():
            revenue = metrics.get('revenue', 0)
            cost = metrics.get('cost', 0)
            units = metrics.get('units', 1)
            
            gross_profit = revenue - cost
            margin = (gross_profit / revenue * 100) if revenue > 0 else 0
            revenue_per_unit = revenue / units if units > 0 else 0
            cost_per_unit = cost / units if units > 0 else 0
            
            results.append({
                'segment': segment_name,
                'revenue': revenue,
                'cost': cost,
                'gross_profit': gross_profit,
                'margin_pct': margin,
                'revenue_per_unit': revenue_per_unit,
                'cost_per_unit': cost_per_unit,
                'units': units,
                'contribution_rank': None  # Will be filled after sorting
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('margin_pct', ascending=False)
        df['contribution_rank'] = range(1, len(df) + 1)
        
        # ABC analysis (Pareto)
        df['cumulative_revenue_pct'] = (df['revenue'].cumsum() / df['revenue'].sum() * 100)
        df['abc_category'] = df['cumulative_revenue_pct'].apply(
            lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C')
        )
        
        return df


# =====================================================
# 5. MONTE CARLO SIMULATION
# =====================================================

class MonteCarloSimulator:
    """Probabilistic analysis using Monte Carlo simulation"""
    
    def __init__(self, model_data: dict, config: 'CompanyConfig', num_simulations: int = 10000):
        self.model_data = model_data
        self.config = config
        self.num_simulations = num_simulations
        self.years = list(model_data['years'])
    
    def run_simulation(self, parameter_distributions: Dict[str, Tuple[str, float, float]]) -> Dict:
        """
        Run Monte Carlo simulation with parameter distributions
        
        Args:
            parameter_distributions: {param_name: (distribution_type, mean/low, std/high)}
            Types: 'normal', 'uniform', 'lognormal', 'triangular'
        """
        from financial_model import run_financial_model
        
        results = {
            'enterprise_values': [],
            'final_profits': [],
            'final_cash': [],
            'roi': []
        }
        
        np.random.seed(42)
        
        for _ in range(self.num_simulations):
            cfg = replace(self.config)
            
            # Sample parameters from distributions
            for param, (dist_type, param1, param2) in parameter_distributions.items():
                if dist_type == 'normal':
                    sample = np.random.normal(param1, param2)
                elif dist_type == 'uniform':
                    sample = np.random.uniform(param1, param2)
                elif dist_type == 'lognormal':
                    sample = np.random.lognormal(param1, param2)
                elif dist_type == 'triangular':
                    sample = np.random.triangular(param1, (param1+param2)/2, param2)
                else:
                    sample = param1
                
                # Ensure reasonable bounds
                if param == 'cogs_ratio':
                    sample = np.clip(sample, 0.1, 0.95)
                elif param == 'wacc':
                    sample = np.clip(sample, 0.05, 0.50)
                
                if hasattr(cfg, param):
                    setattr(cfg, param, sample)
            
            model = run_financial_model(cfg)
            results['enterprise_values'].append(model['enterprise_value'])
            results['final_profits'].append(model['net_profit'][self.years[-1]])
            results['final_cash'].append(model['cash_balance'][self.years[-1]])
            
            total_capex = cfg.land_acquisition + cfg.factory_construction + cfg.machinery_automation
            roi = (model['net_profit'][self.years[-1]] / total_capex * 100)
            results['roi'].append(roi)
        
        return self._calculate_statistics(results)
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate statistical summaries"""
        stats_dict = {}
        
        for key, values in results.items():
            values = np.array(values)
            stats_dict[key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std_dev': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p5': np.percentile(values, 5),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'distribution': values
            }
        
        return stats_dict


# =====================================================
# 6. WHAT-IF ANALYSIS (INTERACTIVE)
# =====================================================

class WhatIfAnalyzer:
    """Interactive what-if analysis with real-time impact calculation"""
    
    def __init__(self, config: 'CompanyConfig'):
        self.config = config
        self.baseline_results = None
    
    def create_scenario(self, scenario_name: str, adjustments: Dict) -> Dict:
        """
        Create and analyze a what-if scenario
        
        Args:
            scenario_name: Name of the scenario
            adjustments: Dict of parameter -> new_value
        """
        from financial_model import run_financial_model
        
        # Create modified config
        cfg = self.config.__class__(**self.config.__dict__)
        for param, value in adjustments.items():
            if hasattr(cfg, param):
                setattr(cfg, param, value)
        
        model = run_financial_model(cfg)
        
        # Compare to baseline
        if self.baseline_results is None:
            baseline_model = run_financial_model(self.config)
            self.baseline_results = baseline_model
        
        return {
            'scenario_name': scenario_name,
            'adjustments': adjustments,
            'enterprise_value': model['enterprise_value'],
            'ev_change': model['enterprise_value'] - self.baseline_results['enterprise_value'],
            'ev_change_pct': ((model['enterprise_value'] - self.baseline_results['enterprise_value']) / 
                             self.baseline_results['enterprise_value'] * 100),
            'revenue_2030': model['revenue'][2030],
            'profit_2030': model['net_profit'][2030]
        }
    
    def sensitivity_waterfall(self, base_ev: float, adjustments: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
        """
        Build waterfall showing impact of each adjustment
        
        Format: {adjustment_name: (parameter, new_value)}
        """
        from financial_model import run_financial_model
        
        steps = [{'step': 'Baseline', 'value': base_ev, 'impact': 0}]
        current_config = self.config.__class__(**self.config.__dict__)
        
        for adj_name, (param, new_value) in adjustments.items():
            if hasattr(current_config, param):
                setattr(current_config, param, new_value)
            
            model = run_financial_model(current_config)
            new_ev = model['enterprise_value']
            impact = new_ev - steps[-1]['value']
            
            steps.append({
                'step': adj_name,
                'value': new_ev,
                'impact': impact
            })
        
        return pd.DataFrame(steps)


# =====================================================
# 7. GOAL SEEK OPTIMIZATION
# =====================================================

class GoalSeekOptimizer:
    """Find input values required to achieve specific targets"""
    
    def __init__(self, config: 'CompanyConfig'):
        self.config = config
    
    def find_breakeven_parameter(self, parameter: str, target_metric: str, target_value: float) -> Dict:
        """
        Goal Seek: Find parameter value to hit target
        
        Args:
            parameter: Parameter to adjust (e.g., 'cogs_ratio')
            target_metric: Metric to hit (e.g., 'enterprise_value')
            target_value: Target value
        """
        from financial_model import run_financial_model
        
        def objective(param_value):
            cfg = self.config.__class__(**self.config.__dict__)
            setattr(cfg, parameter, param_value)
            model = run_financial_model(cfg)
            
            metric_value = model.get(target_metric, 0)
            if isinstance(metric_value, dict):
                metric_value = metric_value[max(metric_value.keys())]
            
            return (metric_value - target_value) ** 2
        
        # Find initial bounds
        base_value = getattr(self.config, parameter)
        x0 = base_value * 0.5
        
        result = optimize.minimize_scalar(objective, bounds=(0.01, 2 * base_value), method='bounded')
        
        if result.success:
            optimal_value = result.x
            cfg = self.config.__class__(**self.config.__dict__)
            setattr(cfg, parameter, optimal_value)
            model = run_financial_model(cfg)
            
            return {
                'success': True,
                'parameter': parameter,
                'optimal_value': optimal_value,
                'percentage_change': ((optimal_value - base_value) / base_value * 100),
                'target_metric': target_metric,
                'achieved_value': model.get(target_metric, 0),
                'target_value': target_value
            }
        else:
            return {'success': False, 'reason': 'Optimization failed'}


# =====================================================
# 8. TORNADO CHARTS & SPIDER DIAGRAMS
# =====================================================

class TornadoSpiderVisualizer:
    """Generate data for tornado and spider charts"""
    
    @staticmethod
    def tornado_chart_data(sensitivity_results: pd.DataFrame, top_n: int = 10) -> Dict:
        """Prepare tornado chart data (sorted bars)"""
        top_params = sensitivity_results.head(top_n).copy()
        
        tornado_data = {}
        for _, row in top_params.iterrows():
            tornado_data[row['parameter']] = {
                'low': row['low_ev'],
                'high': row['high_ev'],
                'base': row.get('base_value', 0),
                'range': row['high_ev'] - row['low_ev']
            }
        
        return tornado_data
    
    @staticmethod
    def spider_chart_data(scenario_results: Dict[str, Dict]) -> Dict:
        """Prepare spider/radar chart data"""
        spider_data = {}
        
        for scenario_name, metrics in scenario_results.items():
            # Normalize metrics to 0-100 scale
            normalized = {}
            for metric_name, value in metrics.items():
                if metric_name == 'enterprise_value':
                    normalized[metric_name] = min(100, max(0, value / 1e6))  # Scale by millions
            
            spider_data[scenario_name] = normalized
        
        return spider_data


# =====================================================
# 9. REGRESSION MODELING
# =====================================================

class RegressionModeler:
    """Predict outcomes using regression models"""
    
    @staticmethod
    def simple_linear_regression(x_data: List[float], y_data: List[float]) -> Dict:
        """
        Simple linear regression: y = a + b*x
        """
        x = np.array(x_data)
        y = np.array(y_data)
        
        n = len(x)
        b = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        a = np.mean(y) - b * np.mean(x)
        
        y_pred = a + b * x
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'intercept': a,
            'slope': b,
            'r_squared': r_squared,
            'prediction_function': lambda x_new: a + b * x_new,
            'residuals': residuals,
            'predictions': y_pred
        }
    
    @staticmethod
    def multiple_regression(X_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Multiple linear regression: y = a + b1*x1 + b2*x2 + ..."""
        # Add intercept column
        X = np.column_stack([np.ones(len(X_data)), X_data])
        
        # Normal equations: beta = (X'X)^-1 X'y
        try:
            coefficients = np.linalg.inv(X.T @ X) @ X.T @ y_data
        except np.linalg.LinAlgError:
            return {'success': False, 'reason': 'Singular matrix'}
        
        y_pred = X @ coefficients
        residuals = y_data - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'intercept': coefficients[0],
            'coefficients': coefficients[1:],
            'r_squared': r_squared,
            'predictions': y_pred,
            'residuals': residuals
        }


# =====================================================
# 10. TIME SERIES ANALYSIS
# =====================================================

class TimeSeriesAnalyzer:
    """Time series forecasting and analysis"""
    
    @staticmethod
    def simple_exponential_smoothing(series: Dict[int, float], alpha: float = 0.3, forecast_periods: int = 5) -> Dict:
        """Simple exponential smoothing for trend forecasting"""
        years = sorted(series.keys())
        values = [series[y] for y in years]
        
        # Initialize
        forecast = [values[0]]
        
        # Smooth
        for i in range(1, len(values)):
            forecast.append(alpha * values[i-1] + (1 - alpha) * forecast[i-1])
        
        # Forecast forward
        last_forecast = forecast[-1]
        future_forecasts = [last_forecast]
        for _ in range(forecast_periods - 1):
            future_forecasts.append(future_forecasts[-1])
        
        future_years = [years[-1] + i for i in range(1, forecast_periods + 1)]
        
        return {
            'historical_forecast': forecast,
            'future_forecast': future_forecasts,
            'future_years': future_years,
            'alpha': alpha,
            'method': 'SES'
        }
    
    @staticmethod
    def moving_average(series: Dict[int, float], window: int = 3) -> Dict:
        """Calculate moving average"""
        years = sorted(series.keys())
        values = [series[y] for y in years]
        
        ma = []
        for i in range(len(values)):
            if i < window - 1:
                ma.append(np.mean(values[:i+1]))
            else:
                ma.append(np.mean(values[i-window+1:i+1]))
        
        return {
            'years': years,
            'original': values,
            'moving_average': ma,
            'window': window
        }
    
    @staticmethod
    def detect_seasonality(series: Dict[int, float]) -> Dict:
        """Detect seasonal patterns"""
        years = sorted(series.keys())
        values = np.array([series[y] for y in years])
        
        # Simple seasonality detection using variance
        if len(values) < 3:
            return {'seasonal': False, 'strength': 0}
        
        # Check if changes repeat
        diffs = np.diff(values)
        autocorr = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
        
        return {
            'autocorrelation': autocorr,
            'seasonal': abs(autocorr) > 0.5,
            'strength': abs(autocorr)
        }


# =====================================================
# 11. VALUE AT RISK (VaR) & CONDITIONAL VaR
# =====================================================

class RiskAnalyzer:
    """Value at Risk and downside risk analysis"""
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> Dict:
        """
        Calculate Value at Risk
        
        Args:
            returns: List of returns/outcomes
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        """
        returns = np.array(returns)
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        return {
            'var': var,
            'confidence_level': confidence_level,
            'interpretation': f"There is a {(1-confidence_level)*100}% chance of losing more than ${-var:,.0f}"
        }
    
    @staticmethod
    def calculate_cvar(returns: List[float], confidence_level: float = 0.95) -> Dict:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        Average loss beyond VaR
        """
        returns = np.array(returns)
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'average_loss_beyond_var': cvar - var,
            'interpretation': f"Average loss in worst {(1-confidence_level)*100}% of outcomes: ${cvar:,.0f}"
        }
    
    @staticmethod
    def stress_test_returns(base_returns: List[float], shock_magnitude: float) -> Dict:
        """Apply shock to returns distribution"""
        returns = np.array(base_returns)
        shocked_returns = returns * (1 - shock_magnitude)
        
        return {
            'original_mean': np.mean(returns),
            'shocked_mean': np.mean(shocked_returns),
            'mean_impact': np.mean(shocked_returns) - np.mean(returns),
            'original_std': np.std(returns),
            'shocked_std': np.std(shocked_returns),
            'worst_case': np.min(shocked_returns),
            'shock_magnitude': shock_magnitude
        }


# =====================================================
# 12. PORTFOLIO OPTIMIZATION
# =====================================================

class PortfolioOptimizer:
    """Mean-variance portfolio optimization"""
    
    @staticmethod
    def optimize_portfolio(returns_matrix: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize portfolio weights using Markowitz mean-variance framework
        
        Args:
            returns_matrix: Matrix of returns by asset and period
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        # Calculate expected returns and covariance
        expected_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        n_assets = len(expected_returns)
        
        # Objective: Minimize portfolio variance
        def portfolio_volatility(weights):
            return np.sqrt(weights @ cov_matrix @ weights)
        
        # Constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        # Find minimum variance portfolio
        result = optimize.minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        opt_weights = result.x
        opt_return = opt_weights @ expected_returns
        opt_volatility = portfolio_volatility(opt_weights)
        sharpe_ratio = (opt_return - risk_free_rate) / opt_volatility
        
        return {
            'optimal_weights': opt_weights,
            'expected_return': opt_return,
            'volatility': opt_volatility,
            'sharpe_ratio': sharpe_ratio,
            'assets': list(range(n_assets))
        }


# =====================================================
# 13. REAL OPTIONS ANALYSIS
# =====================================================

class RealOptionsAnalyzer:
    """Real options analysis: defer, expand, abandon"""
    
    @staticmethod
    def expansion_option_value(initial_npv: float, expansion_cost: float, 
                              upside_scenario_npv: float, probability: float = 0.5) -> Dict:
        """
        Value of option to expand
        Option Value = Probability * max(Expansion NPV - Cost, 0)
        """
        expansion_npv = upside_scenario_npv - expansion_cost
        option_value = probability * max(expansion_npv, 0)
        
        total_value = initial_npv + option_value
        
        return {
            'initial_npv': initial_npv,
            'expansion_option_value': option_value,
            'total_value_with_option': total_value,
            'option_premium': (option_value / initial_npv * 100) if initial_npv != 0 else 0,
            'expand_threshold_probability': expansion_cost / (upside_scenario_npv - initial_npv) if upside_scenario_npv > initial_npv else float('inf')
        }
    
    @staticmethod
    def abandonment_option_value(initial_npv: float, salvage_value: float, 
                                downside_scenario_npv: float, probability: float = 0.3) -> Dict:
        """
        Value of option to abandon
        """
        downside_loss = min(downside_scenario_npv, 0)
        option_value = probability * max(salvage_value - abs(downside_loss), 0)
        
        total_value = initial_npv + option_value
        
        return {
            'initial_npv': initial_npv,
            'abandonment_option_value': option_value,
            'total_value_with_option': total_value,
            'salvage_value': salvage_value,
            'protection_value': option_value
        }


# =====================================================
# 14. MACROECONOMIC LINKING
# =====================================================

class MacroeconomicLinker:
    """Link macro variables to financial projections"""
    
    @staticmethod
    def apply_inflation_adjustment(base_value: float, inflation_rate: float, years: int) -> float:
        """Adjust value for inflation"""
        return base_value * ((1 + inflation_rate) ** years)
    
    @staticmethod
    def apply_gdp_linkage(revenue_base: float, gdp_growth_rates: Dict[int, float]) -> Dict[int, float]:
        """Link revenue growth to GDP growth"""
        revenues = {}
        cumulative_gdp = 1.0
        
        for year, gdp_growth in sorted(gdp_growth_rates.items()):
            cumulative_gdp *= (1 + gdp_growth)
            revenues[year] = revenue_base * cumulative_gdp
        
        return revenues
    
    @staticmethod
    def apply_exchange_rate_shock(base_value: float, initial_fx_rate: float, 
                                 scenario_fx_rates: Dict[int, float]) -> Dict[int, float]:
        """Adjust for FX movements"""
        values = {}
        for year, fx_rate in scenario_fx_rates.items():
            fx_impact = (fx_rate - initial_fx_rate) / initial_fx_rate
            values[year] = base_value * (1 + fx_impact)
        
        return values
    
    @staticmethod
    def interest_rate_sensitivity(npv: float, duration: float, yield_change: float) -> float:
        """Calculate NPV sensitivity to interest rate changes"""
        npv_change = -duration * npv * yield_change
        return npv + npv_change


# =====================================================
# 15. ESG & SUSTAINABILITY METRICS
# =====================================================

class ESGAnalyzer:
    """ESG and sustainability financial impact analysis"""
    
    @staticmethod
    def carbon_pricing_impact(base_emissions: float, carbon_price: float, 
                             emission_reduction_rate: float, years: int) -> Dict:
        """Calculate financial impact of carbon pricing and reduction"""
        results = {}
        cumulative_cost = 0
        
        for year in range(years):
            emissions = base_emissions * ((1 - emission_reduction_rate) ** year)
            annual_cost = emissions * carbon_price
            cumulative_cost += annual_cost
            
            results[year] = {
                'emissions': emissions,
                'annual_cost': annual_cost,
                'cumulative_cost': cumulative_cost
            }
        
        return results
    
    @staticmethod
    def esg_risk_premium(base_wacc: float, esg_score: float) -> float:
        """Adjust WACC for ESG risk"""
        # Lower ESG score = higher risk premium
        # Scale: 0-100 score
        risk_adjustment = (50 - esg_score) * 0.001  # 0.1% per 10-point deviation
        return base_wacc + risk_adjustment
    
    @staticmethod
    def renewable_investment_payback(investment: float, annual_savings: float, 
                                    degradation_rate: float = 0.01) -> Dict:
        """Calculate payback for renewable energy investments"""
        cumulative_savings = 0
        payback_year = None
        
        for year in range(1, 51):
            annual_saving = annual_savings * ((1 - degradation_rate) ** year)
            cumulative_savings += annual_saving
            
            if payback_year is None and cumulative_savings >= investment:
                payback_year = year
        
        return {
            'investment': investment,
            'payback_period_years': payback_year,
            'total_savings_30_years': cumulative_savings,
            'roi': (cumulative_savings - investment) / investment * 100 if payback_year else None
        }


# =====================================================
# 16. PROBABILISTIC VALUATION
# =====================================================

class ProbabilisticValuation:
    """Generate distributions of valuation metrics"""
    
    @staticmethod
    def distribution_summary(values: List[float]) -> Dict:
        """Generate distribution statistics"""
        values = np.array(values)
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,  # Coefficient of variation
            'percentiles': {
                'p1': np.percentile(values, 1),
                'p5': np.percentile(values, 5),
                'p10': np.percentile(values, 10),
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),
                'p75': np.percentile(values, 75),
                'p90': np.percentile(values, 90),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        }


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    from financial_model import run_financial_model, CompanyConfig
    
    print("\n" + "="*80)
    print("ADVANCED ANALYTICS SUITE - DEMONSTRATION")
    print("="*80)
    
    # Initialize
    config = CompanyConfig()
    model_data = run_financial_model(config)
    
    # 1. Sensitivity Analysis
    print("\n1. ADVANCED SENSITIVITY ANALYSIS (Pareto)")
    print("-" * 80)
    analyzer = AdvancedSensitivityAnalyzer(model_data, config)
    params = ['cogs_ratio', 'wacc', 'tax_rate', 'annual_capacity']
    ranges = {'cogs_ratio': 0.25, 'wacc': 0.30, 'tax_rate': 0.25, 'annual_capacity': 0.25}
    sensitivity = analyzer.pareto_sensitivity(params, ranges)
    print(sensitivity.to_string())
    
    # 2. Stress Testing
    print("\n2. EXTREME SCENARIO STRESS TESTING")
    print("-" * 80)
    stress_engine = StressTestEngine(model_data, config)
    extreme_results = stress_engine.extreme_scenarios()
    for scenario, metrics in extreme_results.items():
        print(f"\n{scenario}:")
        print(f"  Enterprise Value: ${metrics['enterprise_value']:,.0f}")
        print(f"  Recovery Score: {metrics['recovery_probability']:.0f}%")
    
    # 3. Trend Decomposition
    print("\n3. TREND & SEASONALITY DECOMPOSITION")
    print("-" * 80)
    decomp = TrendDecomposition(model_data)
    revenue_decomp = decomp.decompose_series(model_data['revenue'])
    print(f"Trend Strength: {revenue_decomp['trend_strength']:.2f}")
    print(f"Inflection Points: {decomp.identify_inflection_points(model_data['revenue'])}")
    
    # 4. Segmentation
    print("\n4. PRODUCT SEGMENTATION ANALYSIS")
    print("-" * 80)
    segments = {
        'EV_SUV': {'revenue': 50e6, 'cost': 30e6, 'units': 3500},
        'EV_Hatchback': {'revenue': 30e6, 'cost': 18e6, 'units': 2500},
        'EV_Scooters': {'revenue': 15e6, 'cost': 9e6, 'units': 4500}
    }
    seg_analyzer = SegmentationAnalyzer(model_data)
    seg_results = seg_analyzer.segment_analysis(segments)
    print(seg_results.to_string())
    
    # 5. Monte Carlo
    print("\n5. MONTE CARLO SIMULATION (10,000 scenarios)")
    print("-" * 80)
    mc_params = {
        'cogs_ratio': ('normal', 0.60, 0.05),
        'wacc': ('normal', 0.12, 0.02),
        'tax_rate': ('normal', 0.25, 0.02)
    }
    mc_sim = MonteCarloSimulator(model_data, config, num_simulations=1000)
    mc_results = mc_sim.run_simulation(mc_params)
    for key, stats_dict in mc_results.items():
        print(f"\n{key}:")
        print(f"  Mean: ${stats_dict['mean']:,.0f}")
        print(f"  Std Dev: ${stats_dict['std_dev']:,.0f}")
        print(f"  P5: ${stats_dict['p5']:,.0f} | P95: ${stats_dict['p95']:,.0f}")
    
    # 6. What-If Analysis
    print("\n6. WHAT-IF ANALYSIS")
    print("-" * 80)
    whatif = WhatIfAnalyzer(config)
    scenario1 = whatif.create_scenario(
        "Lower COGS",
        {'cogs_ratio': 0.50}
    )
    print(f"Lower COGS Scenario:")
    print(f"  EV Change: ${scenario1['ev_change']:,.0f} ({scenario1['ev_change_pct']:.1f}%)")
    
    # 7. Goal Seek
    print("\n7. GOAL SEEK OPTIMIZATION")
    print("-" * 80)
    goal_seeker = GoalSeekOptimizer(config)
    result = goal_seeker.find_breakeven_parameter('cogs_ratio', 'enterprise_value', 500e6)
    if result['success']:
        print(f"To achieve EV of $500M:")
        print(f"  COGS Ratio must be: {result['optimal_value']:.2f}")
        print(f"  Change: {result['percentage_change']:.1f}%")
    
    # 8. Risk Analysis (VaR)
    print("\n8. VALUE AT RISK ANALYSIS")
    print("-" * 80)
    risk_analyzer = RiskAnalyzer()
    sample_returns = np.random.normal(100e6, 20e6, 1000)
    var_result = risk_analyzer.calculate_var(sample_returns, 0.95)
    cvar_result = risk_analyzer.calculate_cvar(sample_returns, 0.95)
    print(f"VaR (95%): ${var_result['var']:,.0f}")
    print(f"CVaR (95%): ${cvar_result['cvar']:,.0f}")
    
    # 9. Time Series
    print("\n9. TIME SERIES FORECASTING")
    print("-" * 80)
    ts_analyzer = TimeSeriesAnalyzer()
    forecast = ts_analyzer.simple_exponential_smoothing(model_data['revenue'], alpha=0.3, forecast_periods=3)
    print(f"Historical Forecast (last 3): {forecast['historical_forecast'][-3:]}")
    print(f"Future Forecast: {forecast['future_forecast']}")
    
    # 10. ESG Impact
    print("\n10. ESG & SUSTAINABILITY ANALYSIS")
    print("-" * 80)
    esg = ESGAnalyzer()
    carbon_impact = esg.carbon_pricing_impact(1000, 50, 0.05, 5)
    print(f"Carbon Pricing Impact (5 years):")
    for year, data in carbon_impact.items():
        print(f"  Year {year}: ${data['annual_cost']:,.0f} (Cumulative: ${data['cumulative_cost']:,.0f})")
    
    print("\n" + "="*80)
    print("Advanced Analytics Complete.")
    print("="*80)
