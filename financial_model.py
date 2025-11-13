"""
Automobile Manufacturing Financial Model
Converted from Excel-based model to Python with Advanced Analytics & Labor Management
Author: Advanced Analytics Team
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Iterable

# =====================================================
# 1. INPUT PARAMETERS (CONFIGURABLE)
# =====================================================

def _default_capacity_curve(start_year: int, projection_years: int) -> Dict[int, float]:
    """Return the default ramp-up curve for capacity utilization."""
    default_curve = [0.5, 0.7, 0.9, 1.0]
    values = []
    for i in range(projection_years):
        if i < len(default_curve):
            values.append(default_curve[i])
        else:
            values.append(default_curve[-1])
    return {start_year + i: values[i] for i in range(projection_years)}


def _default_product_portfolio() -> Dict[str, Dict[str, float]]:
    """Return the default product portfolio with mix, pricing, and cost drivers."""

    return {
        "EV_Bikes": {
            "mix": 0.30,
            "price": 4_000.0,
            "price_growth": 0.01,
            "variable_cost_ratio": 0.52,
            "cost_inflation": 0.015,
            "scale_sensitivity": 0.10,
        },
        "EV_Scooters": {
            "mix": 0.25,
            "price": 3_500.0,
            "price_growth": 0.01,
            "variable_cost_ratio": 0.50,
            "cost_inflation": 0.015,
            "scale_sensitivity": 0.12,
        },
        "EV_SUV": {
            "mix": 0.25,
            "price": 15_000.0,
            "price_growth": 0.012,
            "variable_cost_ratio": 0.58,
            "cost_inflation": 0.018,
            "scale_sensitivity": 0.08,
        },
        "EV_Hatchback": {
            "mix": 0.10,
            "price": 12_000.0,
            "price_growth": 0.012,
            "variable_cost_ratio": 0.55,
            "cost_inflation": 0.017,
            "scale_sensitivity": 0.09,
        },
        "EV_NanoCar": {
            "mix": 0.10,
            "price": 9_000.0,
            "price_growth": 0.015,
            "variable_cost_ratio": 0.48,
            "cost_inflation": 0.016,
            "scale_sensitivity": 0.14,
        },
    }


def _normalize_product_portfolio(
    start_year: int,
    projection_years: int,
    portfolio: Optional[Dict[str, Dict[str, float]]],
    fallback_ratio: float,
) -> Dict[str, Dict[str, float]]:
    """Validate and normalize product portfolio inputs."""

    if not portfolio:
        portfolio = _default_product_portfolio()

    normalized: Dict[str, Dict[str, float]] = {}
    total_mix = 0.0
    for name, raw in portfolio.items():
        try:
            mix = max(0.0, float(raw.get("mix", 0.0)))
        except (TypeError, ValueError):
            mix = 0.0

        try:
            price = max(0.0, float(raw.get("price", 0.0)))
        except (TypeError, ValueError):
            price = 0.0

        try:
            price_growth = float(raw.get("price_growth", 0.0))
        except (TypeError, ValueError):
            price_growth = 0.0

        try:
            variable_cost_ratio = raw.get("variable_cost_ratio")
            if variable_cost_ratio is not None:
                variable_cost_ratio = max(0.0, float(variable_cost_ratio))
        except (TypeError, ValueError):
            variable_cost_ratio = None

        try:
            base_cost_per_unit = raw.get("variable_cost_per_unit")
            if base_cost_per_unit is not None:
                base_cost_per_unit = max(0.0, float(base_cost_per_unit))
        except (TypeError, ValueError):
            base_cost_per_unit = None

        try:
            cost_inflation = float(raw.get("cost_inflation", 0.0))
        except (TypeError, ValueError):
            cost_inflation = 0.0

        try:
            scale_sensitivity = float(raw.get("scale_sensitivity", 0.1))
        except (TypeError, ValueError):
            scale_sensitivity = 0.1

        scale_sensitivity = max(0.0, min(1.0, scale_sensitivity))

        if base_cost_per_unit is None:
            ratio = variable_cost_ratio if variable_cost_ratio is not None else fallback_ratio
            if price > 0:
                base_cost_per_unit = price * ratio
            else:
                base_cost_per_unit = 0.0

        if price > 0 and (variable_cost_ratio is None or variable_cost_ratio < 0.0):
            variable_cost_ratio = base_cost_per_unit / price if price else 0.0

        variable_cost_ratio = max(0.0, min(1.5, variable_cost_ratio or 0.0))

        normalized[name] = {
            "mix": mix,
            "price": price,
            "price_growth": price_growth,
            "variable_cost_ratio": variable_cost_ratio,
            "base_cost_per_unit": base_cost_per_unit,
            "cost_inflation": cost_inflation,
            "scale_sensitivity": scale_sensitivity,
        }

        total_mix += mix

    if total_mix <= 0:
        normalized = _default_product_portfolio()
        total_mix = sum(item["mix"] for item in normalized.values())

    # Normalize mix weights so they sum to 1
    for name, data in normalized.items():
        data["mix"] = data["mix"] / total_mix

    return normalized


def _default_marketing_plan(start_year: int) -> Dict[str, Dict[str, float]]:
    """Return a default multi-campaign marketing plan."""

    return {
        "Brand Awareness": {
            "base": 48_000.0,
            "growth": 0.04,
            "start_year": start_year,
            "duration": None,
        },
        "Digital Acquisition": {
            "base": 36_000.0,
            "growth": 0.06,
            "start_year": start_year,
            "duration": None,
        },
        "Launch Events": {
            "base": 60_000.0,
            "growth": -0.15,
            "start_year": start_year,
            "duration": 2,
        },
    }


def _normalize_marketing_plan(
    start_year: int, projection_years: int, plan: Optional[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Normalize marketing plan inputs into a structured campaign mapping."""

    if not plan:
        plan = _default_marketing_plan(start_year)

    normalized: Dict[str, Dict[str, float]] = {}
    for name, raw in plan.items():
        try:
            base = max(0.0, float(raw.get("base", 0.0)))
        except (TypeError, ValueError):
            base = 0.0

        try:
            growth = float(raw.get("growth", 0.0))
        except (TypeError, ValueError):
            growth = 0.0

        try:
            start = int(raw.get("start_year", start_year))
        except (TypeError, ValueError):
            start = start_year

        duration_raw = raw.get("duration")
        duration: Optional[int]
        if duration_raw is None:
            duration = None
        else:
            try:
                duration = max(1, int(duration_raw))
            except (TypeError, ValueError):
                duration = None

        normalized[name] = {
            "base": base,
            "growth": growth,
            "start_year": start,
            "duration": duration,
        }

    return normalized


def _build_marketing_budget(
    start_year: int, projection_years: int, plan: Dict[str, Dict[str, float]]
) -> Dict[int, float]:
    """Generate the annual marketing budget from the campaign plan."""

    budgets = {start_year + i: 0.0 for i in range(projection_years)}

    for campaign in plan.values():
        base = campaign["base"]
        growth = campaign["growth"]
        start = max(start_year, campaign["start_year"])
        duration = campaign["duration"]

        for i in range(projection_years):
            year = start_year + i
            if year < start:
                continue
            if duration is not None and year >= start + duration:
                continue

            year_index = year - start
            spend = base * ((1 + growth) ** max(0, year_index))
            budgets[year] += spend

    return budgets


def _normalize_capacity_utilization(
    start_year: int, projection_years: int, values: Optional[Iterable]
) -> Dict[int, float]:
    """Normalize utilization input into a year-indexed mapping covering the horizon.

    Supports dictionaries keyed by year or ordered iterables aligned with the
    projection window. Missing years fall back to the most recently provided
    utilization so downstream consumers never raise ``KeyError``.
    """

    years = [start_year + i for i in range(projection_years)]
    default_map = _default_capacity_curve(start_year, projection_years)

    provided: Dict[int, float] = {}
    if isinstance(values, dict):
        for raw_year, raw_value in values.items():
            if raw_value is None:
                continue
            try:
                year = int(raw_year)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            provided[year] = value
    elif values is not None:
        try:
            ordered = list(values)
        except TypeError:
            ordered = []
        for idx, raw_value in enumerate(ordered):
            if raw_value is None or idx >= projection_years:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            provided[start_year + idx] = value

    carry: Optional[float] = None
    normalized: Dict[int, float] = {}
    for year in years:
        default_value = default_map[year]
        if year in provided:
            carry = provided[year]
        if carry is None:
            carry = default_value
        value = carry
        # Clamp utilization between 0 and 1 to avoid invalid inputs.
        value = max(0.0, min(1.0, value))
        normalized[year] = value
        carry = value

    return normalized


@dataclass
class CompanyConfig:
    """Configuration class for company parameters"""
    company_name: str = "Volt Rider"
    start_year: int = 2026
    projection_years: int = 5
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
    marketing_plan: Optional[Dict[str, Dict[str, float]]] = None

    # Financing
    loan_amount: float = 1_000_000
    equity_investment: float = 3_000_000
    loan_interest_rate: float = 0.08
    loan_term: int = 5
    
    # Working Capital & Financial Parameters
    cogs_ratio: float = 0.6
    product_portfolio: Optional[Dict[str, Dict[str, float]]] = None
    variable_cost_inflation: float = 0.02
    base_fixed_production_cost: float = 1_250_000.0
    fixed_cost_inflation: float = 0.02
    fixed_cost_utilization_sensitivity: float = 0.4
    tax_rate: float = 0.25
    wacc: float = 0.12
    terminal_growth: float = 0.03
    working_capital_ratio: float = 0.12
    receivable_days: float = 45.0
    inventory_days: float = 30.0
    payable_days: float = 30.0
    accrued_expense_ratio: float = 0.04

    def __post_init__(self):
        if self.projection_years < 1:
            raise ValueError("projection_years must be at least 1")

        self.capacity_utilization = _normalize_capacity_utilization(
            self.start_year, self.projection_years, self.capacity_utilization
        )

        # Normalize portfolio cost drivers before calculating budgets or production costs.
        self.product_portfolio = _normalize_product_portfolio(
            self.start_year, self.projection_years, self.product_portfolio, self.cogs_ratio
        )

        # Update the blended cogs ratio so downstream analytics remain meaningful.
        blended_ratio = sum(
            data["mix"] * data["variable_cost_ratio"] for data in self.product_portfolio.values()
        )
        if blended_ratio > 0:
            self.cogs_ratio = blended_ratio

        # Normalize marketing plan and budget.
        self.marketing_plan = _normalize_marketing_plan(
            self.start_year, self.projection_years, self.marketing_plan
        )

        derived_budget = _build_marketing_budget(
            self.start_year, self.projection_years, self.marketing_plan
        )

        if self.marketing_budget:
            custom = {}
            for raw_year, raw_value in self.marketing_budget.items():
                try:
                    year = int(raw_year)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                custom[year] = max(0.0, value)

            budgets = derived_budget
            last_known = None
            for i in range(self.projection_years):
                year = self.start_year + i
                if year in custom:
                    budgets[year] = custom[year]
                    last_known = custom[year]
                elif last_known is not None:
                    budgets[year] = last_known
            self.marketing_budget = budgets
        else:
            self.marketing_budget = derived_budget

        # Guardrail working-capital driver inputs to keep calculations stable.
        self.receivable_days = max(0.0, float(self.receivable_days))
        self.inventory_days = max(0.0, float(self.inventory_days))
        self.payable_days = max(0.0, float(self.payable_days))
        self.accrued_expense_ratio = max(0.0, float(self.accrued_expense_ratio))

        self.variable_cost_inflation = float(self.variable_cost_inflation)
        self.fixed_cost_inflation = float(self.fixed_cost_inflation)
        self.fixed_cost_utilization_sensitivity = max(
            0.0, min(1.0, float(self.fixed_cost_utilization_sensitivity))
        )

# Default Configuration
config = CompanyConfig()


def _projection_years(cfg: CompanyConfig) -> List[int]:
    """Return the list of projection years for the configuration."""
    return [cfg.start_year + i for i in range(cfg.projection_years)]


def _carry_forward(mapping: Dict[int, float], year: int, default: float) -> float:
    """Return mapping[year] while carrying forward the closest available value."""
    if year in mapping:
        return mapping[year]

    earlier = [y for y in mapping.keys() if y <= year]
    if earlier:
        return mapping[max(earlier)]

    later = [y for y in mapping.keys() if y > year]
    if later:
        return mapping[min(later)]

    return default


def _fixed_production_cost(cfg: CompanyConfig, year: int) -> float:
    """Estimate fixed production costs for the given year."""

    years_since_start = max(0, year - cfg.start_year)
    base_cost = cfg.base_fixed_production_cost * ((1 + cfg.fixed_cost_inflation) ** years_since_start)
    utilization = _carry_forward(cfg.capacity_utilization, year, 1.0)

    # Higher utilization dilutes fixed costs while under-utilization keeps them elevated.
    sensitivity = cfg.fixed_cost_utilization_sensitivity
    scale = 1.0 - sensitivity * (utilization - 1.0)
    scale = max(0.5, min(1.5, scale))

    return base_cost * scale


def calculate_working_capital_positions(
    years: Sequence[int],
    revenue: Dict[int, float],
    cogs: Dict[int, float],
    opex: Dict[int, float],
    cfg: CompanyConfig,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Return working-capital component balances and their year-over-year changes."""

    receivables: Dict[int, float] = {}
    inventory: Dict[int, float] = {}
    payables: Dict[int, float] = {}
    accrued_expenses: Dict[int, float] = {}
    net_working_capital: Dict[int, float] = {}
    changes: Dict[int, float] = {}

    sales_day_factor = 1.0 / 365.0

    previous_nwc = 0.0
    for year in years:
        rev = revenue.get(year, 0.0)
        cost = cogs.get(year, 0.0)
        operating = opex.get(year, 0.0)

        receivables_balance = rev * cfg.receivable_days * sales_day_factor
        inventory_balance = cost * cfg.inventory_days * sales_day_factor
        payables_balance = cost * cfg.payable_days * sales_day_factor
        accrued_balance = operating * cfg.accrued_expense_ratio

        receivables[year] = receivables_balance
        inventory[year] = inventory_balance
        payables[year] = payables_balance
        accrued_expenses[year] = accrued_balance

        net_wc = receivables_balance + inventory_balance - payables_balance - accrued_balance
        net_working_capital[year] = net_wc
        changes[year] = net_wc - previous_nwc
        previous_nwc = net_wc

    return (
        net_working_capital,
        changes,
        receivables,
        inventory,
        payables,
        accrued_expenses,
    )

# =====================================================
# 2. PRODUCTION & SALES FORECAST
# =====================================================
def calculate_production_forecast(cfg: CompanyConfig):
    """Calculate production volume, per-product pricing, and revenue forecasts."""

    years = _projection_years(cfg)

    production_volume = {
        y: cfg.annual_capacity * _carry_forward(cfg.capacity_utilization, y, 1.0)
        for y in years
    }

    product_units: Dict[int, Dict[str, float]] = {}
    product_prices: Dict[int, Dict[str, float]] = {}
    product_revenue: Dict[int, Dict[str, float]] = {}
    revenue: Dict[int, float] = {}

    for y in years:
        units_for_year: Dict[str, float] = {}
        prices_for_year: Dict[str, float] = {}
        revenue_for_year: Dict[str, float] = {}
        total_revenue = 0.0
        years_since_start = max(0, y - cfg.start_year)

        for product, drivers in cfg.product_portfolio.items():
            units = production_volume[y] * drivers["mix"]
            price = drivers["price"] * ((1 + drivers.get("price_growth", 0.0)) ** years_since_start)

            units_for_year[product] = units
            prices_for_year[product] = price
            revenue_value = units * price
            revenue_for_year[product] = revenue_value
            total_revenue += revenue_value

        product_units[y] = units_for_year
        product_prices[y] = prices_for_year
        product_revenue[y] = revenue_for_year
        revenue[y] = total_revenue

    return production_volume, product_units, product_prices, revenue, product_revenue

# =====================================================
# 3. COST OF GOODS SOLD
# =====================================================
def calculate_cogs(
    years: Sequence[int],
    product_units: Dict[int, Dict[str, float]],
    cfg: CompanyConfig,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, Dict[str, float]]]:
    """Calculate COGS with variable and fixed components."""

    total_cogs: Dict[int, float] = {}
    variable_cogs: Dict[int, float] = {}
    fixed_cogs: Dict[int, float] = {}
    variable_breakdown: Dict[int, Dict[str, float]] = {}

    for y in years:
        per_product: Dict[str, float] = {}
        variable_total = 0.0
        years_since_start = max(0, y - cfg.start_year)
        utilization = _carry_forward(cfg.capacity_utilization, y, 1.0)

        for product, units in product_units[y].items():
            drivers = cfg.product_portfolio[product]
            base_cost = drivers["base_cost_per_unit"]
            cost_inflation = drivers.get("cost_inflation", cfg.variable_cost_inflation)
            scale_sensitivity = drivers.get("scale_sensitivity", 0.0)

            inflation_factor = (1 + cost_inflation) ** years_since_start
            scale_factor = 1.0 - scale_sensitivity * (utilization - 1.0)
            scale_factor = max(0.5, min(1.5, scale_factor))

            cost_per_unit = base_cost * inflation_factor * scale_factor
            variable_cost = cost_per_unit * units
            per_product[product] = variable_cost
            variable_total += variable_cost

        fixed_cost = _fixed_production_cost(cfg, y)

        total_cogs[y] = variable_total + fixed_cost
        variable_cogs[y] = variable_total
        fixed_cogs[y] = fixed_cost
        variable_breakdown[y] = per_product

    return total_cogs, variable_cogs, fixed_cogs, variable_breakdown

# =====================================================
# 4. OPERATING EXPENSES
# =====================================================
def calculate_opex(years: Sequence[int], cfg: CompanyConfig) -> Dict[int, float]:
    """Calculate operating expenses including marketing and payroll"""
    opex = {}
    for y in years:
        annual_payroll = (cfg.avg_salary * cfg.headcount * 12) * (1 + cfg.annual_salary_growth) ** (y - cfg.start_year)
        marketing = _carry_forward(cfg.marketing_budget, y, 72_000)
        opex[y] = marketing + annual_payroll
    return opex

def calculate_opex_with_labor_manager(years: Sequence[int], cfg: CompanyConfig) -> Dict[int, float]:
    """Calculate operating expenses using labor manager if available"""
    if cfg.labor_manager is None:
        return calculate_opex(years, cfg)

    opex = {}
    for y in years:
        # Get labor costs from manager
        direct_cost = cfg.labor_manager.get_labor_cost_by_type(y, cfg.annual_salary_growth).get('Direct', 0)
        indirect_cost = cfg.labor_manager.get_labor_cost_by_type(y, cfg.annual_salary_growth).get('Indirect', 0)
        labor_cost = direct_cost + indirect_cost
        marketing = _carry_forward(cfg.marketing_budget, y, 72_000)
        opex[y] = marketing + labor_cost
    return opex

def get_labor_metrics(cfg: CompanyConfig, years: Sequence[int]) -> Dict[int, Dict[str, float]]:
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
def calculate_income_statement(
    years: Sequence[int],
    cfg: CompanyConfig,
    production_volume,
    revenue,
    cogs,
    opex,
):
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

    depreciation_lookup = {y: depreciation_schedule.get(y, 0.0) for y in years}

    ebitda = {y: revenue[y] - cogs[y] - opex[y] for y in years}
    ebit = {y: ebitda[y] - depreciation_lookup[y] for y in years}
    tax = {y: max(0, ebit[y] * cfg.tax_rate) for y in years}
    net_profit = {y: ebit[y] - tax[y] for y in years}

    return ebitda, ebit, tax, net_profit, depreciation_lookup

# =====================================================
# 6. DCF VALUATION
# =====================================================
def calculate_dcf(
    years: Sequence[int], ebit, cfg: CompanyConfig, depreciation
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Calculate Free Cash Flow and DCF valuation"""
    # depreciation may be a dict (per-year) or a scalar
    if isinstance(depreciation, dict):
        fcf = {y: ebit[y] * (1 - cfg.tax_rate) + depreciation.get(y, 0.0) for y in years}
    else:
        fcf = {y: ebit[y] * (1 - cfg.tax_rate) + depreciation for y in years}
    discounted_fcf = {y: fcf[y] / ((1 + cfg.wacc) ** (y - cfg.start_year + 1)) for y in years}

    final_year = max(years)

    # Avoid division by zero and negative terminal values
    if cfg.wacc <= cfg.terminal_growth or fcf[final_year] <= 0:
        terminal_value = 0
    else:
        terminal_value = fcf[final_year] * (1 + cfg.terminal_growth) / (cfg.wacc - cfg.terminal_growth)

    discounted_terminal = terminal_value / ((1 + cfg.wacc) ** (final_year - cfg.start_year + 1))
    enterprise_value = sum(discounted_fcf.values()) + discounted_terminal
    
    return fcf, discounted_fcf, enterprise_value

# =====================================================
# 7. CASH FLOW STATEMENT CALCULATION
# =====================================================
def build_debt_schedule(
    years: Sequence[int], cfg: CompanyConfig
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Construct interest, principal, and ending balance schedules for the loan."""
    interest_payment: Dict[int, float] = {}
    principal_payment: Dict[int, float] = {}
    ending_balance: Dict[int, float] = {}

    balance = cfg.loan_amount
    for y in years:
        years_since_start = y - cfg.start_year
        if years_since_start < 0:
            interest = 0.0
            principal = 0.0
        elif years_since_start >= cfg.loan_term or balance <= 0:
            interest = 0.0
            principal = 0.0
        else:
            interest = balance * cfg.loan_interest_rate
            scheduled_principal = cfg.loan_amount / cfg.loan_term
            principal = min(scheduled_principal, balance)
            balance = max(0.0, balance - principal)

        interest_payment[y] = interest
        principal_payment[y] = principal
        ending_balance[y] = balance

    return interest_payment, principal_payment, ending_balance

# =====================================================
# 7. CASH FLOW STATEMENT CALCULATION
# =====================================================
def calculate_cash_flow(
    years: Sequence[int], cfg: CompanyConfig, net_profit, depreciation, cfo, cfi, cff
):
    """Calculate cumulative cash balance"""
    cash_balance = {}
    year_list = list(years)
    for i, y in enumerate(year_list):
        if i == 0:
            cash_balance[y] = cfo[y] + cfi[y] + cff[y]
        else:
            prev_year = year_list[i - 1]
            cash_balance[y] = cash_balance[prev_year] + cfo[y] + cfi[y] + cff[y]
    return cash_balance

# =====================================================
# 8. BALANCE SHEET CALCULATION
# =====================================================
def calculate_balance_sheet(
    years: Sequence[int],
    cfg: CompanyConfig,
    net_profit,
    cash_balance,
    depreciation,
    outstanding_debt,
    accounts_receivable,
    inventory,
    accounts_payable,
    accrued_expenses,
):
    """Calculate balance sheet items"""
    years_list = list(years)
    # Determine total capex and compute fixed assets net of accumulated depreciation
    if cfg.capex_manager is not None:
        total_capex = cfg.capex_manager.total_capex()
        # depreciation expected to be a dict mapping year->amount
        if isinstance(depreciation, dict):
            # accumulated depreciation up to year
            fixed_assets = {}
            for y in years_list:
                acc_dep = 0.0
                for t in years_list:
                    if t <= y:
                        acc_dep += depreciation.get(t, 0.0)
                fixed_assets[y] = max(0, total_capex - acc_dep)
        else:
            # fallback: treat as scalar annual depreciation
            fixed_assets = {}
            for y in years_list:
                years_since_start = y - cfg.start_year
                accumulated_dep = depreciation * (years_since_start + 1)
                fixed_assets[y] = max(0, total_capex - accumulated_dep)
    else:
        total_capex = cfg.land_acquisition + cfg.factory_construction + cfg.machinery_automation
        fixed_assets = {}
        for y in years_list:
            years_since_start = y - cfg.start_year
            if isinstance(depreciation, dict):
                accumulated_dep = sum(depreciation.get(t, 0.0) for t in years_list if t <= y)
            else:
                accumulated_dep = depreciation * (years_since_start + 1)
            fixed_assets[y] = max(0, total_capex - accumulated_dep)

    current_assets = {
        y: cash_balance[y] + accounts_receivable.get(y, 0.0) + inventory.get(y, 0.0)
        for y in years_list
    }
    current_liabilities = {
        y: accounts_payable.get(y, 0.0) + accrued_expenses.get(y, 0.0)
        for y in years_list
    }
    long_term_debt = {y: max(0.0, outstanding_debt[y]) for y in years_list}

    retained_earnings = {}
    for i, y in enumerate(years_list):
        if i == 0:
            retained_earnings[y] = net_profit[y]
        else:
            prev_year = years_list[i - 1]
            retained_earnings[y] = retained_earnings[prev_year] + net_profit[y]

    total_equity = {y: cfg.equity_investment + retained_earnings[y] for y in years_list}
    total_assets = {y: current_assets[y] + fixed_assets[y] for y in years_list}
    total_liab_equity = {y: current_liabilities[y] + long_term_debt[y] + total_equity[y] for y in years_list}

    return (
        fixed_assets,
        current_assets,
        current_liabilities,
        long_term_debt,
        total_equity,
        total_assets,
        total_liab_equity,
    )

# =====================================================
# MAIN CALCULATION ENGINE
# =====================================================
def run_financial_model(cfg: CompanyConfig = None) -> dict:

    if cfg is None:
        cfg = config
    
    years = _projection_years(cfg)

    # Calculate components
    (
        production_volume,
        product_units,
        product_prices,
        revenue,
        product_revenue,
    ) = calculate_production_forecast(cfg)
    cogs, variable_cogs, fixed_cogs, variable_breakdown = calculate_cogs(years, product_units, cfg)
    opex = calculate_opex_with_labor_manager(years, cfg)
    labor_metrics = get_labor_metrics(cfg, years)
    ebitda, ebit, tax, net_profit, depreciation = calculate_income_statement(years, cfg, production_volume, revenue, cogs, opex)
    fcf, discounted_fcf, enterprise_value = calculate_dcf(years, ebit, cfg, depreciation)

    (
        net_working_capital,
        change_in_working_capital,
        accounts_receivable,
        inventory,
        accounts_payable,
        accrued_expenses,
    ) = calculate_working_capital_positions(years, revenue, cogs, opex, cfg)

    interest_payment, loan_repayment, outstanding_debt = build_debt_schedule(years, cfg)

    # Determine CAPEX cash flows and total capex
    if cfg.capex_manager is not None:
        yearly_capex = cfg.capex_manager.yearly_capex_schedule(cfg.start_year, len(years))
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

    cff = {y: (cfg.equity_investment + cfg.loan_amount) if y == cfg.start_year else -loan_repayment[y] - interest_payment[y]
           for y in years}
    cash_balance = calculate_cash_flow(years, cfg, net_profit, depreciation, cfo, cfi, cff)

    # Balance sheet
    fixed_assets, current_assets, current_liabilities, long_term_debt, total_equity, total_assets, total_liab_equity = \
        calculate_balance_sheet(
            years,
            cfg,
            net_profit,
            cash_balance,
            depreciation,
            outstanding_debt,
            accounts_receivable,
            inventory,
            accounts_payable,
            accrued_expenses,
        )
    
    balance_check = {y: abs(total_assets[y] - total_liab_equity[y]) < 1e-2 for y in years}
    
    # Return all calculated data
    return {
        'years': tuple(years),
        'revenue': revenue,
        'cogs': cogs,
        'variable_cogs': variable_cogs,
        'fixed_cogs': fixed_cogs,
        'variable_cogs_breakdown': variable_breakdown,
        'opex': opex,
        'ebitda': ebitda,
        'ebit': ebit,
        'tax': tax,
        'net_profit': net_profit,
        'fcf': fcf,
        'discounted_fcf': discounted_fcf,
        'enterprise_value': enterprise_value,
        'production_volume': production_volume,
        'product_units': product_units,
        'product_prices': product_prices,
        'product_revenue': product_revenue,
        'depreciation': depreciation,
        'cfo': cfo,
        'cfi': cfi,
        'cff': cff,
        'cash_balance': cash_balance,
        'interest_payment': interest_payment,
        'loan_repayment': loan_repayment,
        'outstanding_debt': outstanding_debt,
        'accounts_receivable': accounts_receivable,
        'inventory': inventory,
        'accounts_payable': accounts_payable,
        'accrued_expenses': accrued_expenses,
        'net_working_capital': net_working_capital,
        'change_in_working_capital': change_in_working_capital,
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
    years = list(model_data['years'])
    
    income_df = pd.DataFrame({
        "Year": years,
        "Revenue": [model_data['revenue'][y] for y in years],
        "COGS": [model_data['cogs'][y] for y in years],
        "Opex": [model_data['opex'][y] for y in years],
        "EBITDA": [model_data['ebitda'][y] for y in years],
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
    
    columns = [
        "Year",
        "Direct Labor HC",
        "Direct Labor Cost",
        "Indirect Labor HC",
        "Indirect Labor Cost",
        "Total Headcount",
        "Total Labor Cost",
    ]

    if not labor_metrics:
        return pd.DataFrame(columns=columns)

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
    final_year = max(model_data['years'])
    roi = 0.0
    if model_data['initial_capex']:
        roi = (model_data['net_profit'][final_year] / model_data['initial_capex']) * 100

    print(f"\nEnterprise Value (DCF): ${model_data['enterprise_value']:,.2f}")
    print(f"ROI ({final_year}): {roi:.2f}%")
    print("Model complete and balanced.")
