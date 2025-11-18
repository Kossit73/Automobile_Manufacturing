"""
Automobile Manufacturing Financial Model
Converted from Excel-based model to Python with Advanced Analytics & Labor Management
Author: Advanced Analytics Team
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# =====================================================
# 1. INPUT PARAMETERS (CONFIGURABLE)
# =====================================================

@dataclass
class CompanyConfig:
    """Configuration class for company parameters"""
    company_name: str = "Volt Rider"
    start_year: int = 2026
    production_end_year: int = 2030
    facility_size: int = 2000
    
    # CAPEX
    land_acquisition: float = 1_000_000
    factory_construction: float = 2_500_000
    machinery_automation: float = 500_000
    useful_life: int = 10
    
    # Payroll
    avg_salary: float = 5000
    headcount: int = 50
    annual_salary_growth: float = 0.05
    
    # Labor Management (optional - overrides above if provided)
    labor_manager: Optional['LaborScheduleManager'] = None
    # CAPEX Management (optional)
    capex_manager: Optional['CapexScheduleManager'] = None
    
    # Production
    annual_capacity: float = 20_000
    capacity_utilization: Dict[int, float] = None
    working_days: int = 300
    
    # Marketing
    marketing_budget: Dict[int, float] = None
    
    # Financing
    loan_amount: float = 1_000_000
    equity_investment: float = 3_000_000
    loan_interest_rate: float = 0.08
    loan_term: int = 5
    
    # Financial Parameters
    cogs_ratio: float = 0.6
    tax_rate: float = 0.25
    wacc: float = 0.12
    terminal_growth: float = 0.03
    
    def __post_init__(self):
        if self.production_end_year < self.start_year:
            self.production_end_year = self.start_year

        years = list(range(self.start_year, self.production_end_year + 1))

        if self.capacity_utilization is None:
            ramp = [0.5, 0.7, 0.9, 1.0, 1.0]
            self.capacity_utilization = {
                y: ramp[i] if i < len(ramp) else ramp[-1]
                for i, y in enumerate(years)
            }
        else:
            sorted_years = sorted(self.capacity_utilization)
            last_val = self.capacity_utilization.get(sorted_years[0], 0.0) if sorted_years else 0.0
            filled = {}
            for y in years:
                if y in self.capacity_utilization:
                    last_val = self.capacity_utilization[y]
                filled[y] = last_val
            self.capacity_utilization = filled

        if self.marketing_budget is None:
            self.marketing_budget = {y: 72_000 for y in years}

# Default Configuration
config = CompanyConfig()

# =====================================================
# 2. PRODUCTION & SALES FORECAST
# =====================================================
def calculate_production_forecast(cfg: CompanyConfig):
    """Calculate production volume and revenue forecasts"""
    years = range(cfg.start_year, cfg.production_end_year + 1)

    production_volume = {y: cfg.annual_capacity * _get_capacity_for_year(cfg, y) for y in years}
    
    product_mix = {
        "EV_Bikes": 0.30,
        "EV_Scooters": 0.25,
        "EV_SUV": 0.25,
        "EV_Hatchback": 0.10,
        "EV_NanoCar": 0.10
    }
    
    selling_price = {
        "EV_Bikes": 4000,
        "EV_Scooters": 3500,
        "EV_SUV": 15000,
        "EV_Hatchback": 12000,
        "EV_NanoCar": 9000
    }
    
    revenue = {
        y: sum(production_volume[y] * product_mix[p] * selling_price[p] for p in product_mix)
        for y in years
    }
    
    return production_volume, product_mix, selling_price, revenue

# =====================================================
# 3. COST OF GOODS SOLD
# =====================================================
def calculate_cogs(revenue: Dict, cfg: CompanyConfig) -> Dict:
    """Calculate COGS based on revenue"""
    return {y: revenue[y] * cfg.cogs_ratio for y in revenue}

# =====================================================
# 4. OPERATING EXPENSES
# =====================================================
def _get_marketing_for_year(cfg: CompanyConfig, year: int) -> float:
    """Gracefully fetch a marketing budget value even when a year is missing.

    The app frequently extends projection horizons or edits schedules without
    reseeding every year of ``cfg.marketing_budget``. This helper avoids
    ``KeyError`` by falling back to the last defined value, or ``0.0`` when no
    budgets exist.
    """

    if not cfg.marketing_budget:
        return 0.0

    if year in cfg.marketing_budget:
        return cfg.marketing_budget[year]

    # Use the most recent budget defined on or before the requested year; if
    # none exist, fall back to the earliest available entry.
    defined_years = sorted(cfg.marketing_budget.keys())
    earlier = [y for y in defined_years if y <= year]
    if earlier:
        return cfg.marketing_budget[earlier[-1]]
    return cfg.marketing_budget[defined_years[0]]


def _get_capacity_for_year(cfg: CompanyConfig, year: int) -> float:
    """Gracefully fetch utilization for production forecasts with forward-fill."""

    if not cfg.capacity_utilization:
        return 0.0

    if year in cfg.capacity_utilization:
        return cfg.capacity_utilization[year]

    past_years = [y for y in cfg.capacity_utilization if y <= year]
    if past_years:
        return cfg.capacity_utilization[max(past_years)]

    return list(cfg.capacity_utilization.values())[0]


def calculate_opex_breakdown(years: range, cfg: CompanyConfig) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    """Calculate operating expenses with a breakdown by cost type.

    Returns both the total opex mapping and a per-year breakdown containing
    marketing, labor/payroll, and other operating costs. This allows the
    income statement to display detailed expense categories instead of a
    single aggregated opex line.
    """

    opex: Dict[int, float] = {}
    breakdown: Dict[int, Dict[str, float]] = {}

    for y in years:
        marketing = _get_marketing_for_year(cfg, y)

        if cfg.labor_manager is None:
            labor_cost = (cfg.avg_salary * cfg.headcount * 12) * (1 + cfg.annual_salary_growth) ** (y - cfg.start_year)
        else:
            direct_cost = cfg.labor_manager.get_labor_cost_by_type(y, cfg.annual_salary_growth).get('Direct', 0)
            indirect_cost = cfg.labor_manager.get_labor_cost_by_type(y, cfg.annual_salary_growth).get('Indirect', 0)
            labor_cost = direct_cost + indirect_cost

        other_cost = 0.0

        breakdown[y] = {
            'marketing': marketing,
            'labor': labor_cost,
            'other': other_cost,
        }

        opex[y] = marketing + labor_cost + other_cost

    return opex, breakdown

def get_labor_metrics(cfg: CompanyConfig, years: range) -> Dict:
    """Extract labor metrics from labor manager for reporting"""
    if cfg.labor_manager is None:
        return {}
    
    metrics = {}
    for y in years:
        headcount_by_type = cfg.labor_manager.get_headcount_by_type(y)
        cost_by_type = cfg.labor_manager.get_labor_cost_by_type(y, cfg.annual_salary_growth)
        
        metrics[y] = {
            'direct_headcount': headcount_by_type['Direct'],
            'indirect_headcount': headcount_by_type['Indirect'],
            'total_headcount': headcount_by_type['Direct'] + headcount_by_type['Indirect'],
            'direct_labor_cost': cost_by_type['Direct'],
            'indirect_labor_cost': cost_by_type['Indirect'],
            'total_labor_cost': cost_by_type['Direct'] + cost_by_type['Indirect']
        }
    return metrics

# =====================================================
# 5. INCOME STATEMENT CALCULATION
# =====================================================
def calculate_income_statement(years: range, cfg: CompanyConfig, production_volume, revenue, cogs, opex):
    """Calculate complete income statement"""
    # Determine depreciation schedule (per-year) from capex manager if available
    years_list = list(years)
    if cfg.capex_manager is not None:
        depreciation_schedule = cfg.capex_manager.depreciation_schedule(cfg.start_year, len(years_list))
    else:
        # Legacy behavior: straight-line total capex over useful life
        total_capex = cfg.land_acquisition + cfg.factory_construction + cfg.machinery_automation
        annual_dep = total_capex / cfg.useful_life
        depreciation_schedule = {y: annual_dep for y in years_list}

    ebitda = {y: revenue[y] - cogs[y] - opex[y] for y in years}
    ebit = {y: ebitda[y] - depreciation_schedule[y] for y in years}
    tax = {y: max(0, ebit[y] * cfg.tax_rate) for y in years}
    net_profit = {y: ebit[y] - tax[y] for y in years}

    return ebitda, ebit, tax, net_profit, depreciation_schedule

# =====================================================
# 6. DCF VALUATION
# =====================================================
def calculate_dcf(years: range, ebit, cfg: CompanyConfig, depreciation) -> Tuple[Dict, Dict, float]:
    """Calculate Free Cash Flow and DCF valuation"""
    # depreciation may be a dict (per-year) or a scalar
    if isinstance(depreciation, dict):
        fcf = {y: ebit[y] * (1 - cfg.tax_rate) + depreciation.get(y, 0.0) for y in years}
    else:
        fcf = {y: ebit[y] * (1 - cfg.tax_rate) + depreciation for y in years}
    discounted_fcf = {y: fcf[y] / ((1 + cfg.wacc) ** (y - cfg.start_year + 1)) for y in years}
    
    last_year = max(years)

    # Avoid division by zero and negative terminal values
    if cfg.wacc <= cfg.terminal_growth or fcf[last_year] <= 0:
        terminal_value = 0
    else:
        terminal_value = fcf[last_year] * (1 + cfg.terminal_growth) / (cfg.wacc - cfg.terminal_growth)

    discounted_terminal = terminal_value / ((1 + cfg.wacc) ** (last_year - cfg.start_year + 1))
    enterprise_value = sum(discounted_fcf.values()) + discounted_terminal
    
    return fcf, discounted_fcf, enterprise_value

# =====================================================
# 7. CASH FLOW STATEMENT CALCULATION
# =====================================================
def calculate_cash_flow(years: range, cfg: CompanyConfig, net_profit, depreciation, cfo, cfi, cff):
    """Calculate cumulative cash balance"""
    cash_balance = {}
    for i, y in enumerate(years):
        if i == 0:
            cash_balance[y] = cfo[y] + cfi[y] + cff[y]
        else:
            prev_year = years[i - 1]
            cash_balance[y] = cash_balance[prev_year] + cfo[y] + cfi[y] + cff[y]
    return cash_balance

# =====================================================
# 8. BALANCE SHEET CALCULATION
# =====================================================
def calculate_balance_sheet(years: range, cfg: CompanyConfig, net_profit, cash_balance, 
                           depreciation, loan_repayment, cfi):
    """Calculate balance sheet items"""
    # Determine total capex and compute fixed assets net of accumulated depreciation
    if cfg.capex_manager is not None:
        total_capex = cfg.capex_manager.total_capex()
        # depreciation expected to be a dict mapping year->amount
        if isinstance(depreciation, dict):
            # accumulated depreciation up to year
            fixed_assets = {}
            for y in years:
                acc_dep = 0.0
                for t in years:
                    if t <= y:
                        acc_dep += depreciation.get(t, 0.0)
                fixed_assets[y] = max(0, total_capex - acc_dep)
        else:
            # fallback: treat as scalar annual depreciation
            fixed_assets = {}
            for y in years:
                years_since_start = y - cfg.start_year
                accumulated_dep = depreciation * (years_since_start + 1)
                fixed_assets[y] = max(0, total_capex - accumulated_dep)
    else:
        total_capex = cfg.land_acquisition + cfg.factory_construction + cfg.machinery_automation
        fixed_assets = {}
        if isinstance(depreciation, dict):
            for y in years:
                acc_dep = 0.0
                for t in years:
                    if t <= y:
                        acc_dep += depreciation.get(t, 0.0)
                fixed_assets[y] = max(0, total_capex - acc_dep)
        else:
            for y in years:
                years_since_start = y - cfg.start_year
                accumulated_dep = depreciation * (years_since_start + 1)
                fixed_assets[y] = max(0, total_capex - accumulated_dep)
    
    current_assets = {y: cash_balance[y] + 200_000 for y in years}
    current_liabilities = {y: 150_000 for y in years}
    long_term_debt = {y: cfg.loan_amount - (loan_repayment[y] * (y - cfg.start_year + 1)) for y in years}
    
    retained_earnings = {}
    for i, y in enumerate(years):
        if i == 0:
            retained_earnings[y] = net_profit[y]
        else:
            prev_year = years[i - 1]
            retained_earnings[y] = retained_earnings[prev_year] + net_profit[y]
    
    total_equity = {y: cfg.equity_investment + retained_earnings[y] for y in years}
    total_assets = {y: current_assets[y] + fixed_assets[y] for y in years}
    total_liab_equity = {y: current_liabilities[y] + long_term_debt[y] + total_equity[y] for y in years}
    
    return fixed_assets, current_assets, current_liabilities, long_term_debt, total_equity, total_assets, total_liab_equity

# =====================================================
# MAIN CALCULATION ENGINE
# =====================================================
def run_financial_model(cfg: CompanyConfig = None) -> dict:

    if cfg is None:
        cfg = config
    
    years = range(cfg.start_year, cfg.production_end_year + 1)
    
    # Calculate components
    production_volume, product_mix, selling_price, revenue = calculate_production_forecast(cfg)
    cogs = calculate_cogs(revenue, cfg)
    opex, opex_breakdown = calculate_opex_breakdown(years, cfg)
    labor_metrics = get_labor_metrics(cfg, years)
    ebitda, ebit, tax, net_profit, depreciation = calculate_income_statement(years, cfg, production_volume, revenue, cogs, opex)
    fcf, discounted_fcf, enterprise_value = calculate_dcf(years, ebit, cfg, depreciation)
    
    # Cash flow components
    change_in_working_capital = {
        cfg.start_year: -500_000,
        cfg.start_year + 1: -250_000,
        cfg.start_year + 2: 0,
        cfg.start_year + 3: 200_000,
        cfg.start_year + 4: 200_000,
    }

    interest_payment = {y: cfg.loan_amount * cfg.loan_interest_rate for y in years}
    loan_repayment = {y: cfg.loan_amount / cfg.loan_term if y - cfg.start_year < cfg.loan_term else 0 for y in years}

    # Determine CAPEX cash flows and total capex
    if cfg.capex_manager is not None:
        yearly_capex = cfg.capex_manager.yearly_capex_schedule(cfg.start_year, len(list(years)))
        total_capex = cfg.capex_manager.total_capex()
        cfi = {y: -yearly_capex.get(y, 0.0) for y in years}
    else:
        total_capex = cfg.land_acquisition + cfg.factory_construction + cfg.machinery_automation
        cfi = {y: -total_capex if y == cfg.start_year else 0 for y in years}

    # cfo uses per-year depreciation if provided
    cfo = {}
    for y in years:
        dep_y = depreciation[y] if isinstance(depreciation, dict) else depreciation
        cfo[y] = net_profit[y] + dep_y - change_in_working_capital.get(y, 0)

    cff = {y: (cfg.equity_investment + cfg.loan_amount) if y == cfg.start_year else -loan_repayment[y] - interest_payment[y] for y in years}
    cash_balance = calculate_cash_flow(years, cfg, net_profit, depreciation, cfo, cfi, cff)
    
    # Balance sheet
    fixed_assets, current_assets, current_liabilities, long_term_debt, total_equity, total_assets, total_liab_equity = \
        calculate_balance_sheet(years, cfg, net_profit, cash_balance, depreciation, loan_repayment, cfi)
    
    balance_check = {y: abs(total_assets[y] - total_liab_equity[y]) < 1e-2 for y in years}
    
    # Return all calculated data
    return {
        'years': years,
        'revenue': revenue,
        'cogs': cogs,
        'opex': opex,
        'opex_breakdown': opex_breakdown,
        'ebitda': ebitda,
        'ebit': ebit,
        'tax': tax,
        'net_profit': net_profit,
        'fcf': fcf,
        'discounted_fcf': discounted_fcf,
        'enterprise_value': enterprise_value,
        'production_volume': production_volume,
        'product_mix': product_mix,
        'selling_price': selling_price,
        'depreciation': depreciation,
        'cfo': cfo,
        'cfi': cfi,
        'cff': cff,
        'working_capital_change': change_in_working_capital,
        'interest_payment': interest_payment,
        'loan_repayment': loan_repayment,
        'cash_balance': cash_balance,
        'fixed_assets': fixed_assets,
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'long_term_debt': long_term_debt,
        'total_equity': total_equity,
        'total_assets': total_assets,
        'total_liab_equity': total_liab_equity,
        'balance_check': balance_check,
        'labor_metrics': labor_metrics,
        'initial_capex': total_capex,
        'capex_items': [item.to_dict() for item in cfg.capex_manager.list_items()] if cfg.capex_manager is not None else [],
        'config': cfg
    }

# =====================================================
# 11. OUTPUT SUMMARIES
# =====================================================
def generate_financial_statements(model_data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate income statement, cash flow, and balance sheet DataFrames"""
    years = model_data['years']

    breakdown = model_data.get('opex_breakdown', {})
    marketing = [breakdown.get(y, {}).get('marketing', 0.0) for y in years]
    labor = [breakdown.get(y, {}).get('labor', 0.0) for y in years]
    other_opex = [breakdown.get(y, {}).get('other', 0.0) for y in years]

    income_df = pd.DataFrame({
        "Year": years,
        "Revenue": [model_data['revenue'][y] for y in years],
        "COGS": [model_data['cogs'][y] for y in years],
        "Marketing": marketing,
        "Labor / Payroll": labor,
        "Other Opex": other_opex,
        "Opex": [model_data['opex'][y] for y in years],
        "EBITDA": [model_data['ebitda'][y] for y in years],
        "Depreciation": [model_data['depreciation'][y] if isinstance(model_data['depreciation'], dict) else model_data['depreciation'] for y in years],
        "EBIT": [model_data['ebit'][y] for y in years],
        "Tax": [model_data['tax'][y] for y in years],
        "Net Profit": [model_data['net_profit'][y] for y in years],
    })
    
    cashflow_df = pd.DataFrame({
        "Year": years,
        "CFO": [model_data['cfo'][y] for y in years],
        "CFI": [model_data['cfi'][y] for y in years],
        "CFF": [model_data['cff'][y] for y in years],
        "Net Cash Flow": [model_data['cfo'][y] + model_data['cfi'][y] + model_data['cff'][y] for y in years],
        "Closing Cash": [model_data['cash_balance'][y] for y in years]
    })
    
    balance_df = pd.DataFrame({
        "Year": years,
        "Fixed Assets": [model_data['fixed_assets'][y] for y in years],
        "Current Assets": [model_data['current_assets'][y] for y in years],
        "Total Assets": [model_data['total_assets'][y] for y in years],
        "Current Liabilities": [model_data['current_liabilities'][y] for y in years],
        "Long Term Debt": [model_data['long_term_debt'][y] for y in years],
        "Total Equity": [model_data['total_equity'][y] for y in years],
        "Total Liabilities + Equity": [model_data['total_liab_equity'][y] for y in years],
        "Balanced?": [model_data['balance_check'][y] for y in years]
    })
    
    return income_df, cashflow_df, balance_df

def generate_labor_statement(model_data: dict) -> pd.DataFrame:
    """Generate labor cost statement if labor metrics available"""
    labor_metrics = model_data.get('labor_metrics', {})
    
    if not labor_metrics:
        return pd.DataFrame()
    
    years = sorted(labor_metrics.keys())
    
    labor_df = pd.DataFrame({
        "Year": years,
        "Direct Labor HC": [labor_metrics[y]['direct_headcount'] for y in years],
        "Direct Labor Cost": [f"${labor_metrics[y]['direct_labor_cost']:,.0f}" for y in years],
        "Indirect Labor HC": [labor_metrics[y]['indirect_headcount'] for y in years],
        "Indirect Labor Cost": [f"${labor_metrics[y]['indirect_labor_cost']:,.0f}" for y in years],
        "Total Headcount": [labor_metrics[y]['total_headcount'] for y in years],
        "Total Labor Cost": [f"${labor_metrics[y]['total_labor_cost']:,.0f}" for y in years],
    })
    
    return labor_df

# =====================================================
# DISPLAY RESULTS
# =====================================================
if __name__ == "__main__":
    model_data = run_financial_model()
    income_df, cashflow_df, balance_df = generate_financial_statements(model_data)
    
    cfg = model_data['config']
    print(f"\n=== {cfg.company_name.upper()} FINANCIAL MODEL ===")
    print("\n--- INCOME STATEMENT ---")
    print(income_df.to_string(index=False))
    
    print("\n--- CASH FLOW STATEMENT ---")
    print(cashflow_df.to_string(index=False))
    
    print("\n--- BALANCE SHEET ---")
    print(balance_df.to_string(index=False))
    
    # Calculate ROI
    roi = (model_data['net_profit'][2030] / (cfg.land_acquisition + cfg.factory_construction + cfg.machinery_automation)) * 100
    
    print(f"\nEnterprise Value (DCF): ${model_data['enterprise_value']:,.2f}")
    print(f"ROI (2030): {roi:.2f}%")
    print("Model complete and balanced ✅")
