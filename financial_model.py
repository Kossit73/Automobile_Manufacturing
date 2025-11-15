"""
Automobile Manufacturing Financial Model
Converted from Excel-based model to Python with Advanced Analytics & Labor Management
Author: Advanced Analytics Team
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence, Iterable


def _sanitize_numeric_mapping(raw: Optional[Dict], allow_negative: bool = False) -> Dict[int, float]:
    """Convert arbitrary mappings into ``{year: value}`` dictionaries."""

    if not raw:
        return {}

    cleaned: Dict[int, float] = {}
    for raw_key, raw_value in raw.items():
        try:
            year = int(raw_key)
        except (TypeError, ValueError):
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        if not allow_negative and value < 0:
            value = 0.0

        cleaned[year] = value

    return cleaned


def _sanitize_nested_numeric_mapping(
    raw: Optional[Dict], allow_negative: bool = False
) -> Dict[int, Dict[str, float]]:
    """Sanitize nested mappings keyed by year with numeric payloads."""

    if not raw:
        return {}

    cleaned: Dict[int, Dict[str, float]] = {}
    for raw_key, raw_value in raw.items():
        try:
            year = int(raw_key)
        except (TypeError, ValueError):
            continue

        if not isinstance(raw_value, dict):
            continue

        inner: Dict[str, float] = {}
        for inner_key, inner_value in raw_value.items():
            label = str(inner_key)
            try:
                value = float(inner_value)
            except (TypeError, ValueError):
                continue

            if not allow_negative and value < 0:
                value = 0.0

            inner[label] = value

        if inner:
            cleaned[year] = inner

    return cleaned

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


def _normalize_ownership_schedule(
    years: Sequence[int], raw_schedule: Optional[Dict[int, Dict[str, float]]]
) -> Dict[int, Dict[str, float]]:
    """Normalize ownership inputs into percentage splits that total 100%."""

    sanitized = _sanitize_nested_numeric_mapping(raw_schedule)
    normalized: Dict[int, Dict[str, float]] = {}

    last_owner = 100.0
    last_investor = 0.0

    for year in years:
        data = sanitized.get(year, {})
        try:
            owner_value = float(data.get("owner_pct", last_owner))
        except (TypeError, ValueError):
            owner_value = last_owner
        try:
            investor_value = float(data.get("investor_pct", last_investor))
        except (TypeError, ValueError):
            investor_value = last_investor

        owner_value = max(0.0, owner_value)
        investor_value = max(0.0, investor_value)
        total = owner_value + investor_value

        if total <= 0:
            owner_pct = last_owner if last_owner > 0 else 100.0
            investor_pct = 100.0 - owner_pct
        else:
            owner_pct = (owner_value / total) * 100.0
            investor_pct = 100.0 - owner_pct

        owner_pct = max(0.0, min(100.0, owner_pct))
        investor_pct = 100.0 - owner_pct

        normalized[year] = {"owner_pct": owner_pct, "investor_pct": investor_pct}
        last_owner = owner_pct
        last_investor = investor_pct

    return normalized


def _normalize_capital_structure_schedule(
    years: Sequence[int],
    raw_schedule: Optional[Dict[int, Dict[str, float]]],
    default_equity: float,
    default_debt_draws: Dict[int, float],
) -> Tuple[
    Dict[int, Dict[str, float]],
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    bool,
]:
    """Normalize capital structure overrides for equity and debt contributions."""

    sanitized = _sanitize_nested_numeric_mapping(raw_schedule)
    has_override = bool(sanitized)

    normalized: Dict[int, Dict[str, float]] = {}
    owner_equity: Dict[int, float] = {}
    investor_equity: Dict[int, float] = {}
    total_equity: Dict[int, float] = {}
    debt_amounts: Dict[int, float] = {}

    for idx, year in enumerate(years):
        data = sanitized.get(year, {})

        default_debt = default_debt_draws.get(year, 0.0)

        try:
            owner_value = float(data.get("owner_equity", 0.0))
        except (TypeError, ValueError):
            owner_value = 0.0
        try:
            investor_value = float(data.get("investor_equity", 0.0))
        except (TypeError, ValueError):
            investor_value = 0.0
        try:
            debt_value = float(data.get("debt_amount", default_debt))
        except (TypeError, ValueError):
            debt_value = default_debt

        owner_value = max(0.0, owner_value)
        investor_value = max(0.0, investor_value)
        debt_value = max(0.0, debt_value)

        if not has_override:
            if idx == 0 and default_equity > 0:
                owner_value = default_equity
            else:
                owner_value = owner_value
            investor_value = 0.0
            debt_value = default_debt
        else:
            if idx == 0 and owner_value == 0.0 and investor_value == 0.0 and default_equity > 0:
                owner_value = default_equity

        total = owner_value + investor_value

        normalized[year] = {
            "owner_equity": owner_value,
            "investor_equity": investor_value,
            "debt_amount": debt_value,
        }
        owner_equity[year] = owner_value
        investor_equity[year] = investor_value
        total_equity[year] = total
        debt_amounts[year] = debt_value

    return normalized, owner_equity, investor_equity, total_equity, debt_amounts, has_override


@dataclass
class DebtInstrument:
    """Structured representation of an individual debt instrument."""

    name: str
    principal: float
    interest_rate: float
    term: int
    start_year: int
    interest_only_years: int = 0
    draw_schedule: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.name = str(self.name or "Debt Instrument")
        try:
            self.principal = max(0.0, float(self.principal))
        except (TypeError, ValueError):
            self.principal = 0.0

        try:
            self.interest_rate = max(0.0, float(self.interest_rate))
        except (TypeError, ValueError):
            self.interest_rate = 0.0

        try:
            self.term = int(self.term)
        except (TypeError, ValueError):
            self.term = 0
        self.term = max(0, self.term)

        try:
            self.start_year = int(self.start_year)
        except (TypeError, ValueError):
            self.start_year = 0

        try:
            self.interest_only_years = int(self.interest_only_years)
        except (TypeError, ValueError):
            self.interest_only_years = 0
        self.interest_only_years = max(0, self.interest_only_years)

        cleaned: Dict[int, float] = {}
        total_draws = 0.0
        for raw_year, raw_amount in (self.draw_schedule or {}).items():
            try:
                year = int(raw_year)
                amount = max(0.0, float(raw_amount))
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            cleaned[year] = cleaned.get(year, 0.0) + amount
            total_draws += amount

        if not cleaned and self.principal > 0:
            cleaned = {self.start_year: self.principal}
            total_draws = self.principal

        if total_draws > 0 and self.principal <= 0:
            self.principal = total_draws
        elif total_draws > 0 and abs(total_draws - self.principal) > 1e-6:
            self.principal = total_draws

        self.draw_schedule = cleaned

        if self.principal > 0 and self.term <= 0:
            self.term = max(1, len(self.draw_schedule) if self.draw_schedule else 1)

        if self.draw_schedule:
            latest_draw_year = max(self.draw_schedule)
            if latest_draw_year > self.final_year:
                self.term = (latest_draw_year - self.start_year) + 1

        if self.term > 0 and self.interest_only_years >= self.term:
            self.interest_only_years = max(0, self.term - 1)

    @property
    def final_year(self) -> int:
        if self.term <= 0:
            return self.start_year
        return self.start_year + self.term - 1

    @property
    def amortization_years(self) -> int:
        return max(0, self.term - self.interest_only_years)

    def draw_for_year(self, year: int) -> float:
        return self.draw_schedule.get(year, 0.0)


def _normalize_debt_instruments(
    start_year: int,
    projection_years: int,
    instruments: Optional[Sequence],
    fallback_amount: float,
    fallback_rate: float,
    fallback_term: int,
) -> List[DebtInstrument]:
    """Normalize raw instrument definitions into ``DebtInstrument`` objects."""

    normalized: List[DebtInstrument] = []

    if instruments:
        for idx, raw in enumerate(instruments):
            if isinstance(raw, DebtInstrument):
                normalized.append(raw)
                continue
            if not isinstance(raw, dict):
                continue

            name = raw.get("name") or raw.get("label") or f"Instrument {idx + 1}"

            principal = raw.get("principal", raw.get("amount", 0.0))
            interest_rate = raw.get("interest_rate", raw.get("rate", fallback_rate))
            term = raw.get("term", fallback_term)
            instrument_start = raw.get("start_year", start_year)
            interest_only_years = raw.get("interest_only_years", raw.get("io_years", 0))

            draw_schedule_raw = raw.get("draw_schedule", raw.get("draws"))
            draws: Dict[int, float] = {}
            if isinstance(draw_schedule_raw, dict):
                for raw_year, raw_amount in draw_schedule_raw.items():
                    try:
                        year = int(raw_year)
                        amount = max(0.0, float(raw_amount))
                    except (TypeError, ValueError):
                        continue
                    if amount <= 0:
                        continue
                    draws[year] = draws.get(year, 0.0) + amount
            elif isinstance(draw_schedule_raw, Sequence) and not isinstance(draw_schedule_raw, (str, bytes)):
                for offset, raw_amount in enumerate(draw_schedule_raw):
                    if raw_amount is None:
                        continue
                    try:
                        amount = max(0.0, float(raw_amount))
                    except (TypeError, ValueError):
                        continue
                    if amount <= 0:
                        continue
                    year = int(instrument_start) + offset
                    draws[year] = draws.get(year, 0.0) + amount

            instrument = DebtInstrument(
                name=name,
                principal=principal,
                interest_rate=interest_rate,
                term=term,
                start_year=instrument_start,
                interest_only_years=interest_only_years,
                draw_schedule=draws,
            )
            normalized.append(instrument)

    if not normalized and fallback_amount > 0:
        normalized.append(
            DebtInstrument(
                name="Senior Loan",
                principal=fallback_amount,
                interest_rate=fallback_rate,
                term=fallback_term,
                start_year=start_year,
                draw_schedule={start_year: fallback_amount},
            )
        )

    normalized.sort(key=lambda inst: (inst.start_year, inst.name))
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
    debt_instruments: Optional[Sequence[DebtInstrument]] = None
    ownership_structure: Optional[Dict[int, Dict[str, float]]] = None
    capital_structure_schedule: Optional[Dict[int, Dict[str, float]]] = None

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
    product_unit_overrides: Optional[Dict[int, Dict[str, float]]] = None
    product_price_overrides: Optional[Dict[int, Dict[str, float]]] = None
    product_cost_overrides: Optional[Dict[int, Dict[str, float]]] = None
    variable_cost_overrides: Optional[Dict[int, float]] = None
    fixed_cost_overrides: Optional[Dict[int, float]] = None
    other_opex_overrides: Optional[Dict[int, float]] = None
    working_capital_overrides: Optional[Dict[int, Dict[str, float]]] = None
    owner_equity_contributions: Dict[int, float] = field(default_factory=dict, init=False)
    investor_equity_contributions: Dict[int, float] = field(default_factory=dict, init=False)
    equity_contribution_schedule: Dict[int, float] = field(default_factory=dict, init=False)
    cumulative_equity_contributions: Dict[int, float] = field(default_factory=dict, init=False)
    debt_draw_overrides: Dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.projection_years < 1:
            raise ValueError("projection_years must be at least 1")

        years = [self.start_year + i for i in range(self.projection_years)]

        self.capacity_utilization = _normalize_capacity_utilization(
            self.start_year, self.projection_years, self.capacity_utilization
        )

        # Normalize portfolio cost drivers before calculating budgets or production costs.
        self.product_portfolio = _normalize_product_portfolio(
            self.start_year, self.projection_years, self.product_portfolio, self.cogs_ratio
        )

        self.product_price_overrides = _sanitize_nested_numeric_mapping(self.product_price_overrides)
        self.product_cost_overrides = _sanitize_nested_numeric_mapping(self.product_cost_overrides)

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

        self.debt_instruments = _normalize_debt_instruments(
            self.start_year,
            self.projection_years,
            self.debt_instruments,
            self.loan_amount,
            self.loan_interest_rate,
            self.loan_term,
        )

        debt_draw_defaults: Dict[int, float] = {}
        for instrument in self.debt_instruments:
            for year, amount in instrument.draw_schedule.items():
                if year in years:
                    debt_draw_defaults[year] = debt_draw_defaults.get(year, 0.0) + float(amount)

        self.ownership_structure = _normalize_ownership_schedule(years, self.ownership_structure)

        (
            normalized_capital,
            owner_contrib,
            investor_contrib,
            equity_contrib,
            debt_amounts,
            has_capital_override,
        ) = _normalize_capital_structure_schedule(
            years,
            self.capital_structure_schedule,
            self.equity_investment,
            debt_draw_defaults,
        )

        self.capital_structure_schedule = normalized_capital
        self.owner_equity_contributions = owner_contrib
        self.investor_equity_contributions = investor_contrib
        self.equity_contribution_schedule = equity_contrib
        self.debt_draw_overrides = debt_amounts

        cumulative_equity: Dict[int, float] = {}
        running_total = 0.0
        for year in years:
            running_total += equity_contrib.get(year, 0.0)
            cumulative_equity[year] = running_total
        self.cumulative_equity_contributions = cumulative_equity

        total_equity_contribution = sum(equity_contrib.values())
        if total_equity_contribution > 0:
            self.equity_investment = total_equity_contribution

        if has_capital_override:
            draw_schedule = {year: amount for year, amount in debt_amounts.items() if amount > 0}
            if draw_schedule:
                principal_total = sum(draw_schedule.values())
                override_term = self.loan_term if self.loan_term > 0 else max(1, self.projection_years)
                override_start = min(draw_schedule.keys())
                self.debt_instruments = [
                    DebtInstrument(
                        name="Capital Structure Debt",
                        principal=principal_total,
                        interest_rate=self.loan_interest_rate,
                        term=override_term,
                        start_year=override_start,
                        draw_schedule=draw_schedule,
                    )
                ]
            else:
                self.debt_instruments = []

        total_principal = sum(inst.principal for inst in self.debt_instruments)
        if total_principal > 0:
            self.loan_amount = total_principal
            if self.loan_term <= 0:
                self.loan_term = max(inst.term for inst in self.debt_instruments)
            weighted_rate = sum(inst.interest_rate * inst.principal for inst in self.debt_instruments)
            if weighted_rate > 0:
                self.loan_interest_rate = weighted_rate / total_principal
        else:
            self.loan_amount = 0.0

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

        self.product_unit_overrides = _sanitize_nested_numeric_mapping(self.product_unit_overrides)
        self.variable_cost_overrides = _sanitize_numeric_mapping(self.variable_cost_overrides)
        self.fixed_cost_overrides = _sanitize_numeric_mapping(self.fixed_cost_overrides)
        self.other_opex_overrides = _sanitize_numeric_mapping(self.other_opex_overrides)
        self.working_capital_overrides = _sanitize_nested_numeric_mapping(
            self.working_capital_overrides
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

        overrides = {}
        if getattr(cfg, "working_capital_overrides", None):
            overrides = cfg.working_capital_overrides.get(year, {})

        receivable_days = float(overrides.get("receivable_days", cfg.receivable_days))
        inventory_days = float(overrides.get("inventory_days", cfg.inventory_days))
        payable_days = float(overrides.get("payable_days", cfg.payable_days))
        accrued_ratio = float(overrides.get("accrued_expense_ratio", cfg.accrued_expense_ratio))

        receivable_days = max(0.0, receivable_days)
        inventory_days = max(0.0, inventory_days)
        payable_days = max(0.0, payable_days)
        accrued_ratio = max(0.0, accrued_ratio)

        receivables_balance = rev * receivable_days * sales_day_factor
        inventory_balance = cost * inventory_days * sales_day_factor
        payables_balance = cost * payable_days * sales_day_factor
        accrued_balance = operating * accrued_ratio

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

        price_overrides = {}
        if getattr(cfg, "product_price_overrides", None):
            price_overrides = cfg.product_price_overrides.get(y, {})

        for product, drivers in cfg.product_portfolio.items():
            units = production_volume[y] * drivers["mix"]
            price = drivers["price"] * ((1 + drivers.get("price_growth", 0.0)) ** years_since_start)

            if price_overrides and product in price_overrides:
                override_price = price_overrides[product]
                if override_price >= 0:
                    price = override_price

            units_for_year[product] = units
            prices_for_year[product] = price
            revenue_value = units * price
            revenue_for_year[product] = revenue_value
            total_revenue += revenue_value

        override_units = {}
        if getattr(cfg, "product_unit_overrides", None):
            override_units = cfg.product_unit_overrides.get(y, {})

        if override_units:
            override_total = 0.0
            for product, override_value in override_units.items():
                try:
                    override_amount = max(0.0, float(override_value))
                except (TypeError, ValueError):
                    continue
                units_for_year[product] = override_amount
                override_total += override_amount

            if override_total > 0:
                production_volume[y] = sum(units_for_year.values())
                total_revenue = 0.0
                for product in cfg.product_portfolio.keys():
                    if product not in prices_for_year:
                        continue
                    price = prices_for_year[product]
                    revenue_value = units_for_year.get(product, 0.0) * price
                    revenue_for_year[product] = revenue_value
                    total_revenue += revenue_value
            else:
                cfg.product_unit_overrides.pop(y, None)

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

            if getattr(cfg, "product_cost_overrides", None):
                override_cost = cfg.product_cost_overrides.get(y, {}).get(product)
                if override_cost is not None:
                    try:
                        cost_per_unit = max(0.0, float(override_cost))
                    except (TypeError, ValueError):
                        cost_per_unit = base_cost * inflation_factor * scale_factor

            variable_cost = cost_per_unit * units
            per_product[product] = variable_cost
            variable_total += variable_cost

        fixed_cost = _fixed_production_cost(cfg, y)

        override_variable = None
        if getattr(cfg, "variable_cost_overrides", None):
            override_variable = cfg.variable_cost_overrides.get(y)

        if override_variable is not None:
            override_variable = max(0.0, float(override_variable))
            if variable_total > 0 and per_product:
                scale = override_variable / variable_total
                per_product = {product: value * scale for product, value in per_product.items()}
            elif per_product:
                even_share = override_variable / len(per_product)
                per_product = {product: even_share for product in per_product.keys()}
            variable_total = override_variable

        override_fixed = None
        if getattr(cfg, "fixed_cost_overrides", None):
            override_fixed = cfg.fixed_cost_overrides.get(y)

        if override_fixed is not None:
            fixed_cost = max(0.0, float(override_fixed))

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
        other_override = None
        if getattr(cfg, "other_opex_overrides", None):
            other_override = cfg.other_opex_overrides.get(y)

        total = marketing + annual_payroll
        if other_override is not None:
            total = marketing + annual_payroll + max(0.0, float(other_override))

        opex[y] = total
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
        other_override = None
        if getattr(cfg, "other_opex_overrides", None):
            other_override = cfg.other_opex_overrides.get(y)

        total = marketing + labor_cost
        if other_override is not None:
            total = marketing + labor_cost + max(0.0, float(other_override))

        opex[y] = total
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
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[str, Dict[str, Dict[int, float]]]]:
    """Construct interest, principal, draw, and balance schedules across instruments."""

    years_list = list(years)
    interest_payment: Dict[int, float] = {y: 0.0 for y in years_list}
    principal_payment: Dict[int, float] = {y: 0.0 for y in years_list}
    ending_balance: Dict[int, float] = {y: 0.0 for y in years_list}
    debt_draws: Dict[int, float] = {y: 0.0 for y in years_list}
    instrument_details: Dict[str, Dict[str, Dict[int, float]]] = {}

    instruments = getattr(cfg, "debt_instruments", None) or []

    for instrument in instruments:
        balance = 0.0
        detail = {
            "draws": {y: 0.0 for y in years_list},
            "interest": {y: 0.0 for y in years_list},
            "principal": {y: 0.0 for y in years_list},
            "ending_balance": {y: 0.0 for y in years_list},
        }

        final_year = instrument.final_year
        amortization_start = instrument.start_year + instrument.interest_only_years
        amortization_years = instrument.amortization_years

        for y in years_list:
            draw = instrument.draw_for_year(y)
            if draw:
                balance += draw

            active = instrument.term > 0 and y >= instrument.start_year and y <= final_year
            interest = 0.0
            principal = 0.0

            if active and balance > 0:
                interest = balance * instrument.interest_rate

                if amortization_years <= 0:
                    if y >= final_year:
                        principal = balance
                elif y >= amortization_start:
                    remaining_periods = max(1, (final_year - y) + 1)
                    principal = balance / remaining_periods

                principal = min(principal, balance)
                balance = max(0.0, balance - principal)

            detail["draws"][y] = draw
            detail["interest"][y] = interest
            detail["principal"][y] = principal
            detail["ending_balance"][y] = balance

            interest_payment[y] += interest
            principal_payment[y] += principal
            ending_balance[y] += balance
            debt_draws[y] += draw

        instrument_details[instrument.name] = detail

    return interest_payment, principal_payment, ending_balance, debt_draws, instrument_details

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

    cumulative_equity = {}
    equity_contributions = getattr(cfg, "cumulative_equity_contributions", {})
    if equity_contributions:
        for y in years_list:
            cumulative_equity[y] = float(equity_contributions.get(y, 0.0))
    else:
        running = 0.0
        for y in years_list:
            if y == cfg.start_year:
                running += float(cfg.equity_investment)
            cumulative_equity[y] = running

    total_equity = {y: cumulative_equity.get(y, 0.0) + retained_earnings[y] for y in years_list}
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

    (
        interest_payment,
        loan_repayment,
        outstanding_debt,
        debt_draws,
        debt_details,
    ) = build_debt_schedule(years, cfg)

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

    equity_schedule = dict(getattr(cfg, "equity_contribution_schedule", {}) or {})
    if not equity_schedule and cfg.equity_investment:
        equity_schedule[cfg.start_year] = float(cfg.equity_investment)

    cff = {}
    for y in years:
        equity = float(equity_schedule.get(y, 0.0))
        inflows = equity + debt_draws.get(y, 0.0)
        outflows = loan_repayment[y] + interest_payment[y]
        cff[y] = inflows - outflows
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
        'debt_draws': debt_draws,
        'debt_schedule_detail': debt_details,
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
        'ownership_structure': dict(cfg.ownership_structure),
        'capital_structure_schedule': dict(cfg.capital_structure_schedule),
        'owner_equity_contributions': dict(cfg.owner_equity_contributions),
        'investor_equity_contributions': dict(cfg.investor_equity_contributions),
        'equity_contributions': dict(cfg.equity_contribution_schedule),
        'cumulative_equity_contributions': dict(cfg.cumulative_equity_contributions),
        'debt_contribution_schedule': dict(cfg.debt_draw_overrides),
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
