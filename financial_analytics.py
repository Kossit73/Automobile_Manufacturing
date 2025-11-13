"""
Advanced Financial Analytics Tools
Provides sensitivity analysis, scenario planning, ratio analysis, and trend analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from financial_model import run_financial_model, CompanyConfig, generate_financial_statements


class FinancialAnalyzer:
    """Comprehensive financial analysis toolkit"""
    
    def __init__(self, model_data: dict = None):
        """Initialize with model data or generate new"""
        if model_data is None:
            self.model_data = run_financial_model()
        else:
            self.model_data = model_data
        
        self.years = list(self.model_data['years'])
        self.cfg = self.model_data['config']
    
    # =====================================================
    # 1. SENSITIVITY ANALYSIS
    # =====================================================
    
    def sensitivity_analysis(self, parameter: str, range_pct: float = 0.5, steps: int = 11) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a parameter
        
        Args:
            parameter: Parameter name to vary (e.g., 'cogs_ratio', 'wacc', 'tax_rate')
            range_pct: Range to vary (0.5 = +/-50%)
            steps: Number of steps in analysis
        
        Returns:
            DataFrame with enterprise value at different parameter values
        """
        base_value = getattr(self.cfg, parameter)
        variations = np.linspace(1 - range_pct, 1 + range_pct, steps)
        
        results = []
        for var in variations:
            cfg_copy = CompanyConfig(**self.cfg.__dict__)
            setattr(cfg_copy, parameter, base_value * var)
            
            model = run_financial_model(cfg_copy)
            ev = model['enterprise_value']
            
            results.append({
                'parameter_value': base_value * var,
                'pct_change': (var - 1) * 100,
                'enterprise_value': ev,
                'ev_change_pct': ((ev - self.model_data['enterprise_value']) / self.model_data['enterprise_value']) * 100
            })
        
        return pd.DataFrame(results)
    
    def multi_parameter_sensitivity(self, parameters: List[Tuple[str, float]]) -> Dict:
        """
        Perform multi-parameter sensitivity analysis
        
        Args:
            parameters: List of tuples (parameter_name, range_pct)
        
        Returns:
            Dictionary with sensitivity results for each parameter
        """
        results = {}
        for param, range_pct in parameters:
            results[param] = self.sensitivity_analysis(param, range_pct)
        return results
    
    # =====================================================
    # 2. SCENARIO ANALYSIS
    # =====================================================
    
    def scenario_analysis(self, scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """
        Run scenario analysis with predefined parameter sets
        
        Args:
            scenarios: Dict with scenario names and parameter overrides
                e.g., {'Conservative': {'cogs_ratio': 0.7}, 'Optimistic': {'cogs_ratio': 0.5}}
        
        Returns:
            DataFrame comparing all scenarios
        """
        results = []
        
        for scenario_name, params in scenarios.items():
            cfg_copy = CompanyConfig(**self.cfg.__dict__)
            for param, value in params.items():
                setattr(cfg_copy, param, value)
            
            model = run_financial_model(cfg_copy)
            income_df, _, _ = generate_financial_statements(model)
            
            results.append({
                'Scenario': scenario_name,
                'Enterprise Value': model['enterprise_value'],
                '2030 Revenue': model['revenue'][2030],
                '2030 Net Profit': model['net_profit'][2030],
                'Avg Profit Margin': income_df['Net Profit'].sum() / income_df['Revenue'].sum() * 100,
                'Total FCF': sum(model['fcf'].values()),
                'Final Cash Balance': model['cash_balance'][2030]
            })
        
        return pd.DataFrame(results)
    
    def create_standard_scenarios(self) -> pd.DataFrame:
        """Create and analyze pessimistic, base, and optimistic scenarios"""
        scenarios = {
            'Pessimistic': {
                'cogs_ratio': 0.75,
                'wacc': 0.15,
                'annual_capacity': 15_000
            },
            'Base Case': {},  # Uses current config
            'Optimistic': {
                'cogs_ratio': 0.50,
                'wacc': 0.10,
                'annual_capacity': 25_000
            }
        }
        
        return self.scenario_analysis(scenarios)
    
    # =====================================================
    # 3. FINANCIAL RATIO ANALYSIS
    # =====================================================
    
    def calculate_ratios(self) -> pd.DataFrame:
        """Calculate comprehensive financial ratios for all years"""
        years = self.years
        ratios = []
        
        for y in years:
            # Profitability Ratios
            gross_profit = self.model_data['revenue'][y] - self.model_data['cogs'][y]
            gross_margin = (gross_profit / self.model_data['revenue'][y]) * 100
            net_margin = (self.model_data['net_profit'][y] / self.model_data['revenue'][y]) * 100
            roa = (self.model_data['net_profit'][y] / self.model_data['total_assets'][y]) * 100
            roe = (self.model_data['net_profit'][y] / self.model_data['total_equity'][y]) * 100
            
            # Liquidity Ratios
            current_ratio = self.model_data['current_assets'][y] / self.model_data['current_liabilities'][y]
            quick_ratio = (self.model_data['current_assets'][y] - 0) / self.model_data['current_liabilities'][y]  # No inventory
            
            # Leverage Ratios
            debt_to_equity = self.model_data['long_term_debt'][y] / self.model_data['total_equity'][y]
            debt_to_assets = (self.model_data['current_liabilities'][y] + self.model_data['long_term_debt'][y]) / self.model_data['total_assets'][y]
            
            # Efficiency Ratios
            asset_turnover = self.model_data['revenue'][y] / self.model_data['total_assets'][y]
            
            # Cash Flow Ratios
            operating_cf_to_net_income = self.model_data['cfo'][y] / self.model_data['net_profit'][y] if self.model_data['net_profit'][y] != 0 else 0
            fcf_to_revenue = self.model_data['fcf'][y] / self.model_data['revenue'][y]
            
            ratios.append({
                'Year': y,
                'Gross Margin (%)': gross_margin,
                'Net Margin (%)': net_margin,
                'ROA (%)': roa,
                'ROE (%)': roe,
                'Current Ratio': current_ratio,
                'Quick Ratio': quick_ratio,
                'Debt-to-Equity': debt_to_equity,
                'Debt-to-Assets': debt_to_assets,
                'Asset Turnover': asset_turnover,
                'Operating CF/NI': operating_cf_to_net_income,
                'FCF/Revenue': fcf_to_revenue
            })
        
        return pd.DataFrame(ratios)
    
    def ratio_summary(self) -> Dict:
        """Get summary statistics of key ratios"""
        ratios_df = self.calculate_ratios()
        
        summary = {
            'Profitability': {
                'Avg Net Margin (%)': ratios_df['Net Margin (%)'].mean(),
                'Avg ROE (%)': ratios_df['ROE (%)'].mean(),
                'Avg ROA (%)': ratios_df['ROA (%)'].mean()
            },
            'Liquidity': {
                'Avg Current Ratio': ratios_df['Current Ratio'].mean(),
                'Min Current Ratio': ratios_df['Current Ratio'].min()
            },
            'Leverage': {
                'Avg Debt-to-Equity': ratios_df['Debt-to-Equity'].mean(),
                'Final Debt-to-Equity': ratios_df['Debt-to-Equity'].iloc[-1]
            },
            'Efficiency': {
                'Avg Asset Turnover': ratios_df['Asset Turnover'].mean()
            }
        }
        
        return summary
    
    # =====================================================
    # 4. TREND ANALYSIS
    # =====================================================
    
    def calculate_growth_rates(self) -> pd.DataFrame:
        """Calculate year-over-year growth rates"""
        years = self.years
        growth = []
        
        for i, y in enumerate(years):
            if i == 0:
                growth.append({
                    'Year': y,
                    'Revenue Growth (%)': None,
                    'Net Profit Growth (%)': None,
                    'EBITDA Growth (%)': None,
                    'FCF Growth (%)': None,
                    'Cash Balance Growth (%)': None
                })
            else:
                prev_year = years[i - 1]
                
                rev_growth = ((self.model_data['revenue'][y] - self.model_data['revenue'][prev_year]) / 
                             self.model_data['revenue'][prev_year]) * 100
                profit_growth = ((self.model_data['net_profit'][y] - self.model_data['net_profit'][prev_year]) / 
                                abs(self.model_data['net_profit'][prev_year])) * 100 if self.model_data['net_profit'][prev_year] != 0 else None
                ebitda_growth = ((self.model_data['ebitda'][y] - self.model_data['ebitda'][prev_year]) / 
                                self.model_data['ebitda'][prev_year]) * 100
                fcf_growth = ((self.model_data['fcf'][y] - self.model_data['fcf'][prev_year]) / 
                             self.model_data['fcf'][prev_year]) * 100
                cash_growth = ((self.model_data['cash_balance'][y] - self.model_data['cash_balance'][prev_year]) / 
                              self.model_data['cash_balance'][prev_year]) * 100
                
                growth.append({
                    'Year': y,
                    'Revenue Growth (%)': rev_growth,
                    'Net Profit Growth (%)': profit_growth,
                    'EBITDA Growth (%)': ebitda_growth,
                    'FCF Growth (%)': fcf_growth,
                    'Cash Balance Growth (%)': cash_growth
                })
        
        return pd.DataFrame(growth)
    
    def trend_analysis(self) -> Dict:
        """Analyze trends in key metrics"""
        growth_df = self.calculate_growth_rates()
        
        # Calculate trend statistics (excluding first year)
        stats = growth_df.iloc[1:]
        
        trends = {
            'Revenue': {
                'avg_growth': stats['Revenue Growth (%)'].mean(),
                'trend': 'increasing' if stats['Revenue Growth (%)'].mean() > 0 else 'decreasing'
            },
            'Net Profit': {
                'avg_growth': stats['Net Profit Growth (%)'].mean(),
                'trend': 'increasing' if stats['Net Profit Growth (%)'].mean() > 0 else 'decreasing'
            },
            'Cash Balance': {
                'avg_growth': stats['Cash Balance Growth (%)'].mean(),
                'trend': 'increasing' if stats['Cash Balance Growth (%)'].mean() > 0 else 'decreasing'
            }
        }
        
        return trends
    
    # =====================================================
    # 5. BREAK-EVEN AND MARGIN ANALYSIS
    # =====================================================
    
    def break_even_analysis(self) -> Dict:
        """Calculate break-even metrics"""
        # Break-even volume: Fixed Costs / (Price - Variable Cost per Unit)
        first_year = self.years[0]
        production_volume = self.model_data['production_volume'][first_year]
        avg_selling_price = self.model_data['revenue'][first_year] / production_volume if production_volume else 0.0

        variable_total = self.model_data.get('variable_cogs', {}).get(first_year, 0.0)
        variable_cost_per_unit = variable_total / production_volume if production_volume else 0.0

        fixed_production = self.model_data.get('fixed_cogs', {}).get(first_year, 0.0)
        total_fixed_costs = self.model_data['opex'][first_year] + fixed_production

        contribution_margin_per_unit = avg_selling_price - variable_cost_per_unit
        break_even_volume = total_fixed_costs / contribution_margin_per_unit if contribution_margin_per_unit != 0 else 0

        break_even_revenue = break_even_volume * avg_selling_price
        
        return {
            'break_even_volume': break_even_volume,
            'break_even_revenue': break_even_revenue,
            'avg_selling_price': avg_selling_price,
            'variable_cost_per_unit': variable_cost_per_unit,
            'contribution_margin_per_unit': contribution_margin_per_unit,
            'current_margin_of_safety_pct': ((self.model_data['production_volume'][self.years[0]] - break_even_volume) / 
                                             self.model_data['production_volume'][self.years[0]]) * 100
        }
    
    # =====================================================
    # 6. CASH FLOW ANALYSIS
    # =====================================================
    
    def cash_flow_analysis(self) -> Dict:
        """Comprehensive cash flow analysis"""
        years = self.years
        
        analysis = {
            'total_cfo': sum(self.model_data['cfo'][y] for y in years),
            'total_cfi': sum(self.model_data['cfi'][y] for y in years),
            'total_cff': sum(self.model_data['cff'][y] for y in years),
            'total_fcf': sum(self.model_data['fcf'][y] for y in years),
            'operating_cash_flow_avg': sum(self.model_data['cfo'][y] for y in years) / len(years),
            'fcf_avg': sum(self.model_data['fcf'][y] for y in years) / len(years),
            'final_cash_balance': self.model_data['cash_balance'][years[-1]],
            'cash_conversion_cycle': None  # Would need more detail data
        }
        
        return analysis
    
    # =====================================================
    # 7. VALUATION ANALYSIS
    # =====================================================
    
    def valuation_summary(self) -> Dict:
        """Summary of valuation metrics"""
        years = self.years
        total_capex = (self.cfg.land_acquisition + self.cfg.factory_construction + 
                      self.cfg.machinery_automation)
        
        roi = (self.model_data['net_profit'][years[-1]] / total_capex) * 100
        payback_period = self._calculate_payback_period()
        
        return {
            'enterprise_value': self.model_data['enterprise_value'],
            'equity_value': self.model_data['total_equity'][years[-1]],
            'roi_2030': roi,
            'payback_period_years': payback_period,
            'wacc': self.cfg.wacc,
            'terminal_growth': self.cfg.terminal_growth,
            'cumulative_fcf': sum(self.model_data['fcf'][y] for y in years)
        }
    
    def _calculate_payback_period(self) -> float:
        """Calculate project payback period"""
        years = self.years
        total_capex = (self.cfg.land_acquisition + self.cfg.factory_construction + 
                      self.cfg.machinery_automation)
        cumulative_cf = 0
        
        for y in years:
            cumulative_cf += self.model_data['fcf'][y]
            if cumulative_cf >= total_capex:
                return y - years[0] + 1 - (cumulative_cf - total_capex) / self.model_data['fcf'][y]
        
        return None  # Not paid back within period


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    analyzer = FinancialAnalyzer()
    
    print("\n" + "="*80)
    print("ADVANCED FINANCIAL ANALYTICS")
    print("="*80)
    
    # 1. Sensitivity Analysis
    print("\n--- SENSITIVITY ANALYSIS: COGS Ratio (+/-50%) ---")
    cogs_sensitivity = analyzer.sensitivity_analysis('cogs_ratio', range_pct=0.5, steps=11)
    print(cogs_sensitivity.to_string(index=False))
    
    # 2. Scenario Analysis
    print("\n--- SCENARIO ANALYSIS ---")
    scenarios = analyzer.create_standard_scenarios()
    print(scenarios.to_string(index=False))
    
    # 3. Financial Ratios
    print("\n--- KEY FINANCIAL RATIOS ---")
    ratios = analyzer.calculate_ratios()
    print(ratios.to_string(index=False))
    
    print("\n--- RATIO SUMMARY ---")
    ratio_summary = analyzer.ratio_summary()
    for category, values in ratio_summary.items():
        print(f"\n{category}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.2f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    # 4. Growth Rates
    print("\n--- YEAR-OVER-YEAR GROWTH RATES ---")
    growth = analyzer.calculate_growth_rates()
    print(growth.to_string(index=False))
    
    # 5. Trend Analysis
    print("\n--- TREND ANALYSIS ---")
    trends = analyzer.trend_analysis()
    for metric, trend_data in trends.items():
        print(f"\n{metric}:")
        print(f"  Average Growth: {trend_data['avg_growth']:.2f}%")
        print(f"  Trend: {trend_data['trend'].upper()}")
    
    # 6. Break-even Analysis
    print("\n--- BREAK-EVEN ANALYSIS ---")
    be = analyzer.break_even_analysis()
    for metric, value in be.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:,.2f}" if value > 100 else f"  {metric}: {value:.2f}")
    
    # 7. Cash Flow Analysis
    print("\n--- CASH FLOW ANALYSIS ---")
    cf = analyzer.cash_flow_analysis()
    for metric, value in cf.items():
        if value is not None:
            print(f"  {metric}: ${value:,.2f}")
    
    # 8. Valuation Summary
    print("\n--- VALUATION SUMMARY ---")
    val = analyzer.valuation_summary()
    for metric, value in val.items():
        if isinstance(value, float):
            if 'value' in metric.lower() or 'fcf' in metric.lower():
                print(f"  {metric}: ${value:,.2f}")
            elif 'period' in metric.lower():
                print(f"  {metric}: {value:.2f} years" if value else f"  {metric}: Not achieved")
            else:
                print(f"  {metric}: {value:.2f}")
    
    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)
