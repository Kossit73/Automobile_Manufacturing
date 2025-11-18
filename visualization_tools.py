"""
Financial Visualization Tools
Generates charts and reports for financial analysis
"""

import pandas as pd
import numpy as np
from financial_model import run_financial_model, generate_financial_statements
from financial_analytics import FinancialAnalyzer
import json
from typing import Dict, List, Tuple


class FinancialVisualizer:
    """Tools for creating financial visualizations and reports"""
    
    def __init__(self, model_data: dict = None):
        """Initialize with model data"""
        if model_data is None:
            self.model_data = run_financial_model()
        else:
            self.model_data = model_data
        
        self.analyzer = FinancialAnalyzer(model_data)
        self.years = list(self.model_data['years'])
        self.cfg = self.model_data['config']
    
    # =====================================================
    # 1. TEXT-BASED CHART GENERATION
    # =====================================================
    
    def simple_bar_chart(self, data: Dict, title: str, width: int = 60) -> str:
        """Create a simple text-based bar chart"""
        chart = f"\n{title}\n" + "=" * (width + 20) + "\n"
        
        max_value = max(data.values())
        for label, value in data.items():
            bar_length = int((value / max_value) * width)
            bar = "█" * bar_length
            chart += f"{str(label):20} | {bar} {value:>15,.0f}\n"
        
        return chart
    
    def generate_revenue_chart(self) -> str:
        """Generate revenue trend chart"""
        revenue_data = {y: self.model_data['revenue'][y] for y in self.years}
        return self.simple_bar_chart(revenue_data, "REVENUE TREND (2026-2030)")
    
    def generate_profit_chart(self) -> str:
        """Generate net profit trend chart"""
        profit_data = {y: self.model_data['net_profit'][y] for y in self.years}
        return self.simple_bar_chart(profit_data, "NET PROFIT TREND (2026-2030)")
    
    def generate_cash_balance_chart(self) -> str:
        """Generate cash balance trend chart"""
        cash_data = {y: self.model_data['cash_balance'][y] for y in self.years}
        return self.simple_bar_chart(cash_data, "CASH BALANCE TREND (2026-2030)")
    
    def generate_margin_trend(self) -> str:
        """Generate profit margin trend"""
        margins = {}
        for y in self.years:
            margin = (self.model_data['net_profit'][y] / self.model_data['revenue'][y]) * 100
            margins[y] = margin
        
        chart = "\nPROFIT MARGIN TREND (%)\n" + "=" * 50 + "\n"
        for y, margin in margins.items():
            bar_length = int(margin * 0.5)
            bar = "▓" * bar_length
            chart += f"{y}: {bar} {margin:.2f}%\n"
        
        return chart
    
    # =====================================================
    # 2. FINANCIAL STATEMENT SUMMARIES
    # =====================================================
    
    def executive_summary(self) -> str:
        """Generate executive summary"""
        val = self.analyzer.valuation_summary()
        cf = self.analyzer.cash_flow_analysis()
        be = self.analyzer.break_even_analysis()
        
        summary = f"""
{'='*80}
EXECUTIVE SUMMARY - {self.cfg.company_name.upper()}
{'='*80}

FINANCIAL OVERVIEW:
  Company Name:          {self.cfg.company_name}
  Planning Period:       {self.years[0]} - {self.years[-1]}
  Initial Investment:    ${self.cfg.equity_investment + self.cfg.loan_amount:,.2f}
  Equity:                ${self.cfg.equity_investment:,.2f}
  Debt:                  ${self.cfg.loan_amount:,.2f}

KEY VALUATION METRICS:
  Enterprise Value:      ${val['enterprise_value']:,.2f}
  Final Equity Value:    ${val['equity_value']:,.2f}
  5-Year ROI:            {val['roi_2030']:.2f}%
  Payback Period:        {val['payback_period_years']:.2f} years
  WACC:                  {val['wacc']*100:.2f}%
  Terminal Growth Rate:  {val['terminal_growth']*100:.2f}%

FINANCIAL PERFORMANCE (2026-2030):
  Cumulative Revenue:    ${sum(self.model_data['revenue'][y] for y in self.years):,.2f}
  Cumulative Net Profit: ${sum(self.model_data['net_profit'][y] for y in self.years):,.2f}
  Cumulative FCF:        ${sum(self.model_data['fcf'][y] for y in self.years):,.2f}
  Final Cash Balance:    ${self.model_data['cash_balance'][self.years[-1]]:,.2f}

CASH FLOW SUMMARY:
  Total Operating CF:    ${cf['total_cfo']:,.2f}
  Total Investing CF:    ${cf['total_cfi']:,.2f}
  Total Financing CF:    ${cf['total_cff']:,.2f}
  Average Annual FCF:    ${cf['fcf_avg']:,.2f}

BREAK-EVEN METRICS:
  Break-even Volume:     {be['break_even_volume']:,.0f} units
  Break-even Revenue:    ${be['break_even_revenue']:,.2f}
  Margin of Safety:      {be['current_margin_of_safety_pct']:.2f}%

GROWTH ANALYSIS:
  {self._growth_summary()}

{'='*80}
"""
        return summary
    
    def _growth_summary(self) -> str:
        """Generate growth summary text"""
        growth_df = self.analyzer.calculate_growth_rates().iloc[1:]
        
        rev_growth = growth_df['Revenue Growth (%)'].mean()
        profit_growth = growth_df['Net Profit Growth (%)'].mean()
        
        return f"""Average Revenue Growth:  {rev_growth:.2f}% per year
  Average Profit Growth:   {profit_growth:.2f}% per year"""
    
    def financial_statement_summary(self) -> str:
        """Generate complete financial statement summary"""
        income_df, cashflow_df, balance_df = generate_financial_statements(self.model_data)
        
        summary = f"""
{'='*80}
COMPREHENSIVE FINANCIAL STATEMENTS SUMMARY
{'='*80}

INCOME STATEMENT HIGHLIGHTS:
{income_df.to_string(index=False)}

CASH FLOW HIGHLIGHTS:
{cashflow_df.to_string(index=False)}

BALANCE SHEET HIGHLIGHTS:
{balance_df.to_string(index=False)}

{'='*80}
"""
        return summary
    
    # =====================================================
    # 3. DETAILED ANALYSIS REPORTS
    # =====================================================
    
    def ratio_analysis_report(self) -> str:
        """Generate detailed ratio analysis report"""
        ratios_df = self.analyzer.calculate_ratios()
        summary = self.analyzer.ratio_summary()
        
        report = f"""
{'='*80}
FINANCIAL RATIO ANALYSIS REPORT
{'='*80}

DETAILED RATIOS BY YEAR:
{ratios_df.to_string(index=False)}

RATIO SUMMARY & ASSESSMENT:

Profitability Analysis:
  Average Net Profit Margin: {summary['Profitability']['Avg Net Margin (%)']:.2f}%
    └─ Industry Benchmark:   15-20% (Strong performance if above 20%)
  
  Average ROE (Return on Equity): {summary['Profitability']['Avg ROE (%)']:.2f}%
    └─ Assessment: {'Excellent' if summary['Profitability']['Avg ROE (%)'] > 20 else 'Good' if summary['Profitability']['Avg ROE (%)'] > 10 else 'Fair'}
  
  Average ROA (Return on Assets): {summary['Profitability']['Avg ROA (%)']:.2f}%
    └─ Assessment: {'Strong' if summary['Profitability']['Avg ROA (%)'] > 10 else 'Moderate' if summary['Profitability']['Avg ROA (%)'] > 5 else 'Weak'}

Liquidity Analysis:
  Average Current Ratio: {summary['Liquidity']['Avg Current Ratio']:.2f}
    └─ Industry Benchmark: 1.5-3.0 (Higher is safer, below 1.0 is risky)
  
  Minimum Current Ratio: {summary['Liquidity']['Min Current Ratio']:.2f}
    └─ Assessment: {'Strong' if summary['Liquidity']['Min Current Ratio'] > 1.5 else 'Acceptable' if summary['Liquidity']['Min Current Ratio'] > 1.0 else 'Concerning'}

Leverage Analysis:
  Average Debt-to-Equity: {summary['Leverage']['Avg Debt-to-Equity']:.2f}
    └─ Industry Benchmark: 0.5-2.0 (Lower is less risky)
  
  Final Debt-to-Equity: {summary['Leverage']['Final Debt-to-Equity']:.2f}
    └─ Trend: {'Improving' if summary['Leverage']['Final Debt-to-Equity'] < summary['Leverage']['Avg Debt-to-Equity'] else 'Deteriorating'}

Efficiency Analysis:
  Average Asset Turnover: {summary['Efficiency']['Avg Asset Turnover']:.2f}
    └─ Assessment: {'Efficient' if summary['Efficiency']['Avg Asset Turnover'] > 1.5 else 'Moderate' if summary['Efficiency']['Avg Asset Turnover'] > 1.0 else 'Below Average'}

{'='*80}
"""
        return report
    
    def scenario_report(self) -> str:
        """Generate scenario analysis report"""
        scenarios_df = self.analyzer.create_standard_scenarios()
        
        report = f"""
{'='*80}
SCENARIO ANALYSIS REPORT
{'='*80}

Comparing Pessimistic, Base Case, and Optimistic scenarios:

{scenarios_df.to_string(index=False)}

KEY INSIGHTS:
"""
        
        # Calculate differences
        base_idx = 1  # Base Case
        pessimistic_idx = 0
        optimistic_idx = 2
        
        ev_diff_pess = scenarios_df.iloc[pessimistic_idx]['Enterprise Value'] - scenarios_df.iloc[base_idx]['Enterprise Value']
        ev_diff_opt = scenarios_df.iloc[optimistic_idx]['Enterprise Value'] - scenarios_df.iloc[base_idx]['Enterprise Value']
        
        report += f"""
1. Enterprise Value Impact:
   └─ Pessimistic vs Base: ${ev_diff_pess:,.2f} ({(ev_diff_pess/scenarios_df.iloc[base_idx]['Enterprise Value']*100):.2f}%)
   └─ Optimistic vs Base: ${ev_diff_opt:,.2f} ({(ev_diff_opt/scenarios_df.iloc[base_idx]['Enterprise Value']*100):.2f}%)
   └─ Range: ${scenarios_df['Enterprise Value'].max() - scenarios_df['Enterprise Value'].min():,.2f}

2. Revenue Impact:
   └─ Base Case 2030: ${scenarios_df.iloc[base_idx]['2030 Revenue']:,.2f}
   └─ Pessimistic: ${scenarios_df.iloc[pessimistic_idx]['2030 Revenue']:,.2f}
   └─ Optimistic: ${scenarios_df.iloc[optimistic_idx]['2030 Revenue']:,.2f}

3. Profit Margin Comparison:
   └─ Pessimistic: {scenarios_df.iloc[pessimistic_idx]['Avg Profit Margin']:.2f}%
   └─ Base Case: {scenarios_df.iloc[base_idx]['Avg Profit Margin']:.2f}%
   └─ Optimistic: {scenarios_df.iloc[optimistic_idx]['Avg Profit Margin']:.2f}%

4. Cash Generation:
   └─ Pessimistic Total FCF: ${scenarios_df.iloc[pessimistic_idx]['Total FCF']:,.2f}
   └─ Base Case Total FCF: ${scenarios_df.iloc[base_idx]['Total FCF']:,.2f}
   └─ Optimistic Total FCF: ${scenarios_df.iloc[optimistic_idx]['Total FCF']:,.2f}

RISK ASSESSMENT:
The valuation ranges from ${scenarios_df['Enterprise Value'].min():,.2f} (pessimistic)
to ${scenarios_df['Enterprise Value'].max():,.2f} (optimistic), with a base case of
${scenarios_df.iloc[base_idx]['Enterprise Value']:,.2f}.

{'='*80}
"""
        return report
    
    # =====================================================
    # 4. EXPORT TO JSON
    # =====================================================
    
    def export_to_json(self, filename: str = "financial_analysis.json") -> str:
        """Export all analysis to JSON"""
        export_data = {
            'company': self.cfg.company_name,
            'period': f"{self.years[0]}-{self.years[-1]}",
            'executive_summary': {
                'enterprise_value': self.model_data['enterprise_value'],
                'final_cash_balance': self.model_data['cash_balance'][self.years[-1]],
                'total_revenue': sum(self.model_data['revenue'][y] for y in self.years),
                'total_net_profit': sum(self.model_data['net_profit'][y] for y in self.years)
            },
            'yearly_data': [
                {
                    'year': y,
                    'revenue': self.model_data['revenue'][y],
                    'cogs': self.model_data['cogs'][y],
                    'opex': self.model_data['opex'][y],
                    'ebitda': self.model_data['ebitda'][y],
                    'net_profit': self.model_data['net_profit'][y],
                    'fcf': self.model_data['fcf'][y],
                    'cash_balance': self.model_data['cash_balance'][y]
                }
                for y in self.years
            ],
            'ratios': self.analyzer.ratio_summary(),
            'scenarios': self.analyzer.create_standard_scenarios().to_dict('records')
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return f"Analysis exported to {filename}"
    
    # =====================================================
    # 5. SENSITIVITY HEATMAP (TEXT BASED)
    # =====================================================
    
    def sensitivity_heatmap(self, params: List[str] = None) -> str:
        """Generate text-based sensitivity analysis heatmap"""
        if params is None:
            params = ['cogs_ratio', 'wacc', 'tax_rate']
        
        heatmap = "\n" + "="*80 + "\n"
        heatmap += "SENSITIVITY ANALYSIS HEATMAP\n"
        heatmap += "="*80 + "\n"
        heatmap += "(Shows % change in Enterprise Value)\n\n"
        
        for param in params:
            sensitivity_df = self.analyzer.sensitivity_analysis(param, range_pct=0.3, steps=7)
            
            heatmap += f"{param.upper().replace('_', ' ')}:\n"
            heatmap += "-" * 80 + "\n"
            
            for _, row in sensitivity_df.iterrows():
                change = row['pct_change']
                ev_change = row['ev_change_pct']
                
                # Create visualization
                if ev_change > 0:
                    bar = "▲" * int(abs(ev_change) / 2)
                    symbol = "+"
                else:
                    bar = "▼" * int(abs(ev_change) / 2)
                    symbol = "-"
                
                heatmap += f"  {change:>6.1f}% change: {symbol}{ev_change:>6.2f}% {bar}\n"
            
            heatmap += "\n"
        
        heatmap += "="*80 + "\n"
        return heatmap

    # =====================================================
    # 6. TABLE-FIRST OUTPUTS FOR STREAMLIT
    # =====================================================

    def executive_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """Return executive summary content as dataframes for UI tables."""
        val = self.analyzer.valuation_summary()
        cf = self.analyzer.cash_flow_analysis()
        be = self.analyzer.break_even_analysis()

        overview_rows = [
            {"Metric": "Company", "Value": self.cfg.company_name},
            {"Metric": "Planning Period", "Value": f"{self.years[0]}–{self.years[-1]}"},
            {"Metric": "Initial Investment", "Value": self.cfg.equity_investment + self.cfg.loan_amount},
            {"Metric": "Equity", "Value": self.cfg.equity_investment},
            {"Metric": "Debt", "Value": self.cfg.loan_amount},
        ]

        valuation_rows = [
            {"Metric": "Enterprise Value", "Value": val["enterprise_value"]},
            {"Metric": "Final Equity Value", "Value": val["equity_value"]},
            {"Metric": "ROI (Final Year)", "Value": val["roi_2030"] / 100},
            {"Metric": "Payback Period (yrs)", "Value": val["payback_period_years"]},
            {"Metric": "WACC", "Value": val["wacc"]},
            {"Metric": "Terminal Growth", "Value": val["terminal_growth"]},
        ]

        performance_rows = [
            {"Metric": "Cumulative Revenue", "Value": sum(self.model_data["revenue"][y] for y in self.years)},
            {"Metric": "Cumulative Net Profit", "Value": sum(self.model_data["net_profit"][y] for y in self.years)},
            {"Metric": "Cumulative FCF", "Value": sum(self.model_data["fcf"][y] for y in self.years)},
            {"Metric": "Final Cash Balance", "Value": self.model_data["cash_balance"][self.years[-1]]},
        ]

        cash_flow_rows = [
            {"Metric": "Total Operating CF", "Value": cf["total_cfo"]},
            {"Metric": "Total Investing CF", "Value": cf["total_cfi"]},
            {"Metric": "Total Financing CF", "Value": cf["total_cff"]},
            {"Metric": "Average Annual FCF", "Value": cf["fcf_avg"]},
        ]

        breakeven_rows = [
            {"Metric": "Break-even Volume", "Value": be["break_even_volume"]},
            {"Metric": "Break-even Revenue", "Value": be["break_even_revenue"]},
            {"Metric": "Margin of Safety", "Value": be["current_margin_of_safety_pct"] / 100},
        ]

        growth_df = self.analyzer.calculate_growth_rates()[["Year", "Revenue Growth (%)", "Net Profit Growth (%)"]]

        return {
            "Overview": pd.DataFrame(overview_rows),
            "Valuation": pd.DataFrame(valuation_rows),
            "Performance": pd.DataFrame(performance_rows),
            "Cash Flow": pd.DataFrame(cash_flow_rows),
            "Break-even": pd.DataFrame(breakeven_rows),
            "Growth": growth_df,
        }

    def financial_statement_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return income, cash flow, and balance sheet tables."""
        return generate_financial_statements(self.model_data)

    def ratio_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return ratio detail and summary tables."""
        ratios_df = self.analyzer.calculate_ratios()
        summary = self.analyzer.ratio_summary()

        summary_rows = []
        for category, metrics in summary.items():
            for name, value in metrics.items():
                summary_rows.append({"Category": category, "Metric": name, "Value": value})

        return ratios_df, pd.DataFrame(summary_rows)

    def scenario_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return scenario comparison table and deltas."""
        scenarios_df = self.analyzer.create_standard_scenarios()

        if len(scenarios_df) >= 3:
            base_idx = 1
            pessimistic_idx = 0
            optimistic_idx = 2

            insights = [
                {
                    "Insight": "Enterprise Value vs Base (Pessimistic)",
                    "Change": scenarios_df.iloc[pessimistic_idx]["Enterprise Value"] - scenarios_df.iloc[base_idx]["Enterprise Value"],
                },
                {
                    "Insight": "Enterprise Value vs Base (Optimistic)",
                    "Change": scenarios_df.iloc[optimistic_idx]["Enterprise Value"] - scenarios_df.iloc[base_idx]["Enterprise Value"],
                },
                {
                    "Insight": "Revenue Range (Final Year)",
                    "Change": scenarios_df["2030 Revenue"].max() - scenarios_df["2030 Revenue"].min(),
                },
                {
                    "Insight": "Profit Margin Range", "Change": scenarios_df["Avg Profit Margin"].max() - scenarios_df["Avg Profit Margin"].min(),
                },
            ]
            insight_df = pd.DataFrame(insights)
        else:
            insight_df = pd.DataFrame()

        return scenarios_df, insight_df

    def sensitivity_table(self, params: List[str] = None) -> pd.DataFrame:
        """Return combined sensitivity results."""
        if params is None:
            params = ['cogs_ratio', 'wacc', 'tax_rate']

        frames = []
        for param in params:
            df_param = self.analyzer.sensitivity_analysis(param, range_pct=0.3, steps=7).copy()
            df_param.insert(0, 'Parameter', param)
            frames.append(df_param)

        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=['Parameter', 'pct_change', 'ev_change', 'ev_change_pct'])

    def chart_tables(self) -> pd.DataFrame:
        """Return a consolidated table for revenue, profit, cash, and margins."""
        data = []
        for y in self.years:
            revenue = self.model_data['revenue'][y]
            margin = (self.model_data['net_profit'][y] / revenue) * 100 if revenue else 0
            data.append({
                'Year': y,
                'Revenue': revenue,
                'Net Profit': self.model_data['net_profit'][y],
                'Cash Balance': self.model_data['cash_balance'][y],
                'Profit Margin (%)': margin,
            })
        return pd.DataFrame(data)


# =====================================================
# REPORT GENERATOR
# =====================================================

class FinancialReportGenerator:
    """Generate comprehensive financial reports"""
    
    def __init__(self, model_data: dict = None):
        self.visualizer = FinancialVisualizer(model_data)
        self.model_data = model_data or run_financial_model()
    
    def generate_full_report(self, filename: str = None) -> str:
        """Generate complete financial analysis report"""
        report = ""
        
        # Title page
        report += f"""
{'█'*80}
█{'COMPREHENSIVE FINANCIAL ANALYSIS REPORT'.center(78)}█
█{self.visualizer.cfg.company_name.center(78)}█
█{'Planning Period: 2026-2030'.center(78)}█
{'█'*80}
"""
        
        # Executive Summary
        report += self.visualizer.executive_summary()
        
        # Charts
        report += self.visualizer.generate_revenue_chart()
        report += self.visualizer.generate_profit_chart()
        report += self.visualizer.generate_cash_balance_chart()
        report += self.visualizer.generate_margin_trend()
        
        # Detailed Analysis
        report += self.visualizer.ratio_analysis_report()
        report += self.visualizer.scenario_report()
        report += self.visualizer.sensitivity_heatmap()
        
        # Footer
        report += "\n" + "="*80 + "\n"
        report += "Report generated by Financial Analysis System\n"
        report += "="*80 + "\n"
        
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
            return f"Report saved to {filename}\n" + report
        
        return report


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    visualizer = FinancialVisualizer()
    report_gen = FinancialReportGenerator()
    
    # Print comprehensive report
    full_report = report_gen.generate_full_report()
    print(full_report)
    
    # Export to JSON
    print("\n" + visualizer.export_to_json())
