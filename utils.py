"""
Utility Functions and Validators for Financial Model
Provides data validation, formatting, and helper functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings


# =====================================================
# 1. VALIDATORS
# =====================================================

class FinancialValidator:
    """Validate financial model parameters and outputs"""
    
    @staticmethod
    def validate_config(config: 'CompanyConfig') -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings_list = []
        
        # Check positive values
        if config.land_acquisition <= 0:
            warnings_list.append("Land acquisition cost should be positive")
        
        if config.factory_construction <= 0:
            warnings_list.append("Factory construction cost should be positive")
        
        if config.annual_capacity <= 0:
            warnings_list.append("Annual capacity should be positive")
        
        if config.headcount <= 0:
            warnings_list.append("Headcount should be positive")
        
        # Check reasonable ranges
        if config.cogs_ratio < 0 or config.cogs_ratio > 1:
            warnings_list.append("COGS ratio should be between 0 and 1")
        
        if config.tax_rate < 0 or config.tax_rate > 1:
            warnings_list.append("Tax rate should be between 0 and 1")
        
        if config.wacc <= 0 or config.wacc >= 1:
            warnings_list.append("WACC should be between 0 and 1")
        
        if config.annual_salary_growth < -0.5 or config.annual_salary_growth > 0.5:
            warnings_list.append("Annual salary growth should be between -50% and 50%")
        
        if config.loan_term <= 0:
            warnings_list.append("Loan term should be positive")
        
        if config.useful_life <= 0:
            warnings_list.append("Useful life should be positive")
        
        # Check capacity utilization
        for year, util in config.capacity_utilization.items():
            if util < 0 or util > 1:
                warnings_list.append(f"Capacity utilization for {year} should be between 0 and 1")
        
        return len(warnings_list) == 0, warnings_list
    
    @staticmethod
    def validate_financial_data(model_data: dict) -> Tuple[bool, List[str]]:
        """Validate financial model output data"""
        warnings_list = []
        years = model_data['years']
        
        # Check balance sheet balance
        for y in years:
            if not model_data['balance_check'][y]:
                warnings_list.append(f"Balance sheet doesn't balance for year {y}")
        
        # Check positive revenue
        for y in years:
            if model_data['revenue'][y] <= 0:
                warnings_list.append(f"Negative or zero revenue in {y}")
        
        # Check negative debt
        for y in years:
            if model_data['long_term_debt'][y] < 0:
                warnings_list.append(f"Negative debt in {y} (should not happen)")
        
        # Check cash flow validity
        if model_data['cash_balance'][years[0]] < 0:
            warnings_list.append("Negative cash in first year (company may be insolvent)")
        
        # Check for decreasing cash in critical periods
        for i in range(1, len(years)):
            if model_data['cash_balance'][years[i]] < 0:
                warnings_list.append(f"Negative cash balance in {years[i]}")
        
        return len(warnings_list) == 0, warnings_list


# =====================================================
# 2. FORMATTERS
# =====================================================

class FinancialFormatter:
    """Format financial data for display"""
    
    @staticmethod
    def format_currency(value: float, decimals: int = 2) -> str:
        """Format number as currency"""
        if value >= 1_000_000:
            return f"${value/1_000_000:.{decimals}f}M"
        elif value >= 1_000:
            return f"${value/1_000:.{decimals}f}K"
        else:
            return f"${value:.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format number as percentage"""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_dataframe(df: pd.DataFrame, float_format: str = None) -> pd.DataFrame:
        """Format DataFrame for better display"""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].dtype in ['float64', 'float32', 'int64']:
                if 'margin' in col.lower() or '%' in col:
                    df_copy[col] = df_copy[col].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
                elif any(keyword in col.lower() for keyword in ['revenue', 'profit', 'cash', 'asset', 'cost', 'debt', 'equity']):
                    df_copy[col] = df_copy[col].apply(lambda x: FinancialFormatter.format_currency(x) if isinstance(x, (int, float)) else x)
        
        return df_copy
    
    @staticmethod
    def create_summary_table(data: Dict[str, Any], title: str = None) -> str:
        """Create a formatted summary table"""
        table = ""
        
        if title:
            table += f"\n{title}\n"
            table += "=" * 70 + "\n"
        
        for key, value in data.items():
            if isinstance(value, dict):
                table += f"\n{key}:\n"
                for sub_key, sub_value in value.items():
                    formatted_key = sub_key.replace('_', ' ').title()
                    if isinstance(sub_value, (int, float)):
                        if sub_key.endswith('_pct') or '%' in formatted_key:
                            formatted_val = f"{sub_value:.2f}%"
                        else:
                            formatted_val = FinancialFormatter.format_currency(sub_value)
                    else:
                        formatted_val = str(sub_value)
                    table += f"  {formatted_key:<40} {formatted_val:>15}\n"
            else:
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    if key.endswith('_pct') or '%' in formatted_key:
                        formatted_val = f"{value:.2f}%"
                    else:
                        formatted_val = FinancialFormatter.format_currency(value)
                else:
                    formatted_val = str(value)
                table += f"{formatted_key:<40} {formatted_val:>15}\n"
        
        return table


# =====================================================
# 3. CONVERTERS
# =====================================================

class FinancialConverters:
    """Convert between financial units and formats"""
    
    @staticmethod
    def annualize(monthly_value: float) -> float:
        """Convert monthly value to annual"""
        return monthly_value * 12
    
    @staticmethod
    def monthly(annual_value: float) -> float:
        """Convert annual value to monthly"""
        return annual_value / 12
    
    @staticmethod
    def quarterly(annual_value: float) -> float:
        """Convert annual value to quarterly"""
        return annual_value / 4
    
    @staticmethod
    def daily(annual_value: float, working_days: int = 250) -> float:
        """Convert annual value to daily"""
        return annual_value / working_days
    
    @staticmethod
    def present_value(future_value: float, rate: float, periods: int) -> float:
        """Calculate present value of future cash flow"""
        return future_value / ((1 + rate) ** periods)
    
    @staticmethod
    def future_value(present_value: float, rate: float, periods: int) -> float:
        """Calculate future value of present cash flow"""
        return present_value * ((1 + rate) ** periods)
    
    @staticmethod
    def cagr(beginning_value: float, ending_value: float, periods: int) -> float:
        """Calculate Compound Annual Growth Rate"""
        if beginning_value <= 0:
            return None
        return (((ending_value / beginning_value) ** (1 / periods)) - 1) * 100


# =====================================================
# 4. CALCULATORS
# =====================================================

class FinancialCalculators:
    """Helper calculations for financial analysis"""
    
    @staticmethod
    def calculate_irr_simple(cash_flows: List[float], guess: float = 0.1, max_iterations: int = 100) -> float:
        """Calculate Internal Rate of Return using Newton's method"""
        rate = guess
        
        for _ in range(max_iterations):
            # Calculate NPV and its derivative
            npv = sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
            dnpv = sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
            
            if abs(npv) < 1e-6:  # Close enough
                return rate * 100
            
            if dnpv == 0:
                break
            
            rate = rate - npv / dnpv
        
        return rate * 100
    
    @staticmethod
    def calculate_profitability_index(initial_investment: float, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate profitability index (PI)"""
        pv_inflows = sum(cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
        return pv_inflows / initial_investment
    
    @staticmethod
    def calculate_loan_amortization(principal: float, annual_rate: float, years: int) -> List[Dict]:
        """Generate loan amortization schedule"""
        monthly_rate = annual_rate / 12
        num_payments = years * 12
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
                         ((1 + monthly_rate) ** num_payments - 1)
        
        amortization = []
        remaining_balance = principal
        
        for month in range(1, int(num_payments) + 1):
            interest_payment = remaining_balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
            remaining_balance -= principal_payment
            
            if month % 12 == 0 or month == 1:  # Show annual summaries
                amortization.append({
                    'month': month,
                    'payment': monthly_payment,
                    'principal': principal_payment,
                    'interest': interest_payment,
                    'balance': max(0, remaining_balance)
                })
        
        return amortization
    
    @staticmethod
    def calculate_wacc(equity: float, debt: float, cost_of_equity: float, 
                      cost_of_debt: float, tax_rate: float) -> float:
        """Calculate Weighted Average Cost of Capital"""
        total_value = equity + debt
        e_weight = equity / total_value
        d_weight = debt / total_value
        
        wacc = (e_weight * cost_of_equity) + (d_weight * cost_of_debt * (1 - tax_rate))
        return wacc * 100


# =====================================================
# 5. STATISTICAL ANALYSIS
# =====================================================

class FinancialStatistics:
    """Statistical analysis tools"""
    
    @staticmethod
    def calculate_variance(values: List[float]) -> float:
        """Calculate variance of values"""
        return np.var(values)
    
    @staticmethod
    def calculate_std_dev(values: List[float]) -> float:
        """Calculate standard deviation"""
        return np.std(values)
    
    @staticmethod
    def calculate_correlation(values1: List[float], values2: List[float]) -> float:
        """Calculate correlation between two series"""
        return np.corrcoef(values1, values2)[0, 1]
    
    @staticmethod
    def linear_regression(x_values: List[float], y_values: List[float]) -> Tuple[float, float, float]:
        """
        Perform linear regression
        
        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        coeffs = np.polyfit(x_values, y_values, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate R-squared
        y_pred = slope * np.array(x_values) + intercept
        ss_res = np.sum((np.array(y_values) - y_pred) ** 2)
        ss_tot = np.sum((np.array(y_values) - np.mean(y_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return slope, intercept, r_squared


# =====================================================
# 6. DATA QUALITY CHECKS
# =====================================================

class DataQualityChecker:
    """Check data quality and consistency"""
    
    @staticmethod
    def check_for_nulls(df: pd.DataFrame) -> Dict[str, int]:
        """Check for null values in DataFrame"""
        return df.isnull().sum().to_dict()
    
    @staticmethod
    def check_for_duplicates(df: pd.DataFrame) -> int:
        """Check for duplicate rows"""
        return df.duplicated().sum()
    
    @staticmethod
    def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Check data types in DataFrame"""
        return df.dtypes.to_dict()
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> Tuple[bool, List[int]]:
        """Check if values are within range"""
        mask = (df[column] < min_val) | (df[column] > max_val)
        outlier_indices = df[mask].index.tolist()
        return len(outlier_indices) == 0, outlier_indices


# =====================================================
# 7. REPORT UTILITIES
# =====================================================

class ReportUtilities:
    """Utilities for report generation"""
    
    @staticmethod
    def create_header(title: str, width: int = 80, char: str = "=") -> str:
        """Create formatted header"""
        return f"\n{char * width}\n{title.center(width)}\n{char * width}\n"
    
    @staticmethod
    def create_section(title: str, width: int = 80) -> str:
        """Create formatted section header"""
        return f"\n{title}\n{'-' * len(title)}\n"
    
    @staticmethod
    def create_footer(width: int = 80) -> str:
        """Create report footer"""
        return f"\n{('='*width)}\nReport generated by Financial Analysis System\n{'='*width}\n"
    
    @staticmethod
    def highlight_value(value: float, threshold: float, good_if_above: bool = True) -> str:
        """Highlight value if it meets condition"""
        symbol = "✓" if (value >= threshold) == good_if_above else "✗"
        return f"{symbol} {value:.2f}"


# =====================================================
# MAIN EXECUTION - DEMONSTRATION
# =====================================================

if __name__ == "__main__":
    print("Financial Utilities Module")
    print("="*80)
    
    # Test Validators
    print("\n1. VALIDATOR TESTS:")
    from financial_model import CompanyConfig
    
    cfg = CompanyConfig()
    is_valid, warnings = FinancialValidator.validate_config(cfg)
    print(f"Config valid: {is_valid}")
    if warnings:
        for w in warnings:
            print(f"  Warning: {w}")
    
    # Test Formatters
    print("\n2. FORMATTER TESTS:")
    print(f"Currency: {FinancialFormatter.format_currency(1500000)}")
    print(f"Percentage: {FinancialFormatter.format_percentage(25.5)}")
    
    # Test Converters
    print("\n3. CONVERTER TESTS:")
    print(f"Monthly to Annual (10K): ${FinancialConverters.annualize(10000):,.2f}")
    print(f"Annual to Monthly (120K): ${FinancialConverters.monthly(120000):,.2f}")
    print(f"CAGR (100->200 over 5 years): {FinancialConverters.cagr(100, 200, 5):.2f}%")
    
    # Test Calculators
    print("\n4. CALCULATOR TESTS:")
    cash_flows = [-1000, 300, 300, 300, 300, 300]
    irr = FinancialCalculators.calculate_irr_simple(cash_flows)
    print(f"IRR for cash flows {cash_flows}: {irr:.2f}%")
    
    # Test Statistics
    print("\n5. STATISTICS TESTS:")
    data1 = [10, 20, 30, 40, 50]
    data2 = [15, 25, 35, 45, 55]
    print(f"Std Dev of {data1}: {FinancialStatistics.calculate_std_dev(data1):.2f}")
    print(f"Correlation between series: {FinancialStatistics.calculate_correlation(data1, data2):.2f}")
    slope, intercept, r_sq = FinancialStatistics.linear_regression([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
    print(f"Linear Regression - Slope: {slope:.2f}, R²: {r_sq:.4f}")
    
    # Test Report Utilities
    print("\n6. REPORT UTILITIES:")
    print(ReportUtilities.create_header("SAMPLE REPORT"))
    print(ReportUtilities.create_section("Introduction"))
    print("Report content would go here...")
    print(ReportUtilities.create_footer())
