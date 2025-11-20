"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Interactive labor management, CAPEX scheduling, financial modeling
Author: Advanced Analytics Team
Version: 1.0 (November 2025)
"""

import os
import copy
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

# Import platform modules
from financial_model import (
    run_financial_model,
    CompanyConfig,
    generate_financial_statements,
    generate_labor_statement,
    _get_capacity_for_year,
)
from visualization_tools import FinancialVisualizer
from labor_management import (
    initialize_default_labor_structure, LaborScheduleManager, LaborCostSchedule,
    ProductionLinkedLabor, LaborType, EmploymentStatus, JobCategory
)
from capex_management import initialize_default_capex, CapexScheduleManager
from advanced_analytics import (
    AdvancedSensitivityAnalyzer,
    StressTestEngine,
    MonteCarloSimulator,
    SegmentationAnalyzer,
    WhatIfAnalyzer,
    GoalSeekOptimizer,
    RegressionModeler,
    TimeSeriesAnalyzer,
    RiskAnalyzer,
    PortfolioOptimizer,
    RealOptionsAnalyzer,
    ESGAnalyzer,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Manufacturing Financial Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Automobile manufacturing planner defaults
AUTO_STAGES = ["chassis", "paint", "assembly", "quality"]
AUTO_PLANNER_TEMPLATE = [
    {
        "name": "EV-SUV",
        "defect_rate": 0.03,
        "scrap_allowance": 0.05,
        "target_units": 300,
        "cycle_time": {"chassis": 120, "paint": 90, "assembly": 200, "quality": 60},
        "stations": {"chassis": 3, "paint": 2, "assembly": 4, "quality": 2},
        "hours_per_shift": 8,
        "shifts_per_day": 2,
    },
    {
        "name": "EV-Bikes",
        "defect_rate": 0.025,
        "scrap_allowance": 0.04,
        "target_units": 600,
        "cycle_time": {"chassis": 95, "paint": 70, "assembly": 160, "quality": 50},
        "stations": {"chassis": 4, "paint": 3, "assembly": 5, "quality": 2},
        "hours_per_shift": 8,
        "shifts_per_day": 2,
    },
    {
        "name": "EV-Scooters",
        "defect_rate": 0.02,
        "scrap_allowance": 0.04,
        "target_units": 500,
        "cycle_time": {"chassis": 100, "paint": 80, "assembly": 170, "quality": 55},
        "stations": {"chassis": 3, "paint": 2, "assembly": 4, "quality": 2},
        "hours_per_shift": 8,
        "shifts_per_day": 2,
    },
    {
        "name": "EV-Hatchback",
        "defect_rate": 0.03,
        "scrap_allowance": 0.05,
        "target_units": 350,
        "cycle_time": {"chassis": 110, "paint": 85, "assembly": 190, "quality": 58},
        "stations": {"chassis": 3, "paint": 2, "assembly": 4, "quality": 2},
        "hours_per_shift": 8,
        "shifts_per_day": 2,
    },
    {
        "name": "EV-NanoCar",
        "defect_rate": 0.025,
        "scrap_allowance": 0.045,
        "target_units": 420,
        "cycle_time": {"chassis": 105, "paint": 78, "assembly": 180, "quality": 52},
        "stations": {"chassis": 3, "paint": 2, "assembly": 4, "quality": 2},
        "hours_per_shift": 8,
        "shifts_per_day": 2,
    },
]


def _build_default_auto_planner():
    return copy.deepcopy(AUTO_PLANNER_TEMPLATE)


def _normalize_auto_plan(plan: dict) -> dict:
    normalized = copy.deepcopy(plan)
    normalized.setdefault("defect_rate", 0.0)
    normalized.setdefault("scrap_allowance", 0.0)
    normalized.setdefault("target_units", 0)
    normalized.setdefault("cycle_time", {})
    normalized.setdefault("stations", {})
    normalized.setdefault("hours_per_shift", 8)
    normalized.setdefault("shifts_per_day", 2)
    for stage in AUTO_STAGES:
        normalized["cycle_time"].setdefault(stage, 0)
        normalized["stations"].setdefault(stage, 1)
    return normalized


def _calculate_planner_capacity(plan: dict, working_days: int) -> tuple[dict, str, float]:
    plan = _normalize_auto_plan(plan)
    stage_capacity = {}
    for stage in AUTO_STAGES:
        cycle_minutes = plan["cycle_time"].get(stage, 0) or 0
        stations = plan["stations"].get(stage, 0) or 0
        effective_minutes = cycle_minutes * (1 + plan["defect_rate"] + plan["scrap_allowance"])
        available_minutes = stations * plan["hours_per_shift"] * plan["shifts_per_day"] * 60
        daily_capacity = available_minutes / effective_minutes if effective_minutes > 0 else 0.0
        stage_capacity[stage] = daily_capacity

    bottleneck = min(stage_capacity, key=stage_capacity.get) if stage_capacity else "chassis"
    annual_capacity = stage_capacity.get(bottleneck, 0.0) * working_days
    return stage_capacity, bottleneck, annual_capacity


def _default_asset_start_date(year: int) -> date:
    try:
        return date(year, 1, 1)
    except ValueError:
        return date(datetime.now().year, 1, 1)


def _parse_capex_start_date(value: str, fallback_year: int) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return _default_asset_start_date(fallback_year)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

def initialize_session_state():
    """Initialize or restore session state"""
    default_cfg = CompanyConfig()
    if 'labor_manager' not in st.session_state:
        st.session_state.labor_manager = initialize_default_labor_structure()
    
    if 'capex_manager' not in st.session_state:
        st.session_state.capex_manager = initialize_default_capex(CapexScheduleManager())
    
    if 'financial_model' not in st.session_state:
        st.session_state.financial_model = None

    if 'salary_growth_rate' not in st.session_state:
        st.session_state.salary_growth_rate = 0.05

    if 'production_start_year' not in st.session_state:
        st.session_state.production_start_year = 2026

    if 'production_end_year' not in st.session_state:
        st.session_state.production_end_year = 2030
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None

    if 'ai_settings' not in st.session_state:
        st.session_state.ai_settings = {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "enabled": True,
        }

    if 'owner_equity_pct' not in st.session_state:
        st.session_state.owner_equity_pct = 70.0

    if 'auto_planner' not in st.session_state:
        st.session_state.auto_planner = _build_default_auto_planner()

    if 'marketing_budget_schedule' not in st.session_state or 'marketing_budget_defaults' not in st.session_state:
        st.session_state.marketing_budget_schedule = default_cfg.marketing_budget.copy()
        st.session_state.marketing_budget_defaults = default_cfg.marketing_budget.copy()

    if 'capacity_schedule' not in st.session_state:
        st.session_state.capacity_schedule = default_cfg.capacity_utilization.copy()

    if 'product_mix' not in st.session_state:
        st.session_state.product_mix = default_cfg.product_mix.copy()

    if 'product_base_prices' not in st.session_state:
        st.session_state.product_base_prices = default_cfg.selling_price.copy()

    if 'product_price_overrides' not in st.session_state:
        st.session_state.product_price_overrides = {}

    if 'financing_metadata' not in st.session_state:
        st.session_state.financing_metadata = {}

    # Schedule override containers
    for key in [
        'labor_cost_overrides',
        'fixed_cost_overrides',
        'variable_cost_overrides',
        'other_cost_overrides',
        'loan_repayment_overrides',
        'interest_overrides',
        'financing_cash_flow_overrides',
        'investment_overrides',
        'investment_amounts',
        'investment_default_amounts',
        'asset_overrides',
        'working_capital_overrides',
    ]:
        if key not in st.session_state:
            st.session_state[key] = {}

initialize_session_state()


def _format_currency(val: float) -> str:
    return f"${val:,.0f}"


def _format_statement(df: pd.DataFrame, money_cols):
    formatted = df.copy()
    for col in money_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(_format_currency)
    return formatted


def _ensure_marketing_schedule() -> dict:
    """Ensure the marketing budget schedule covers the active horizon."""

    start_year = st.session_state.production_start_year
    end_year = st.session_state.production_end_year
    schedule = (st.session_state.get('marketing_budget_schedule') or {}).copy()
    defaults = (st.session_state.get('marketing_budget_defaults') or {}).copy()

    if not schedule:
        base_cfg = CompanyConfig(start_year=start_year, production_end_year=end_year)
        schedule = base_cfg.marketing_budget.copy()
        defaults = base_cfg.marketing_budget.copy()

    aligned_schedule = {}
    aligned_defaults = {}

    last_schedule_val = None
    last_default_val = None

    if schedule:
        first_year = min(schedule)
        last_schedule_val = schedule.get(first_year)
    if defaults:
        first_default_year = min(defaults)
        last_default_val = defaults.get(first_default_year)

    fallback_value = last_schedule_val if last_schedule_val is not None else 72_000.0
    fallback_default = last_default_val if last_default_val is not None else 72_000.0

    for year in range(start_year, end_year + 1):
        if year in schedule:
            last_schedule_val = schedule[year]
        if last_schedule_val is None:
            last_schedule_val = fallback_value
        aligned_schedule[year] = last_schedule_val

        if year in defaults:
            last_default_val = defaults[year]
        if last_default_val is None:
            last_default_val = fallback_default
        aligned_defaults[year] = last_default_val

    st.session_state.marketing_budget_schedule.update(aligned_schedule)
    st.session_state.marketing_budget_defaults.update(aligned_defaults)

    return aligned_schedule


def _ensure_capacity_schedule() -> dict:
    """Ensure the capacity utilization schedule matches the active horizon."""

    start_year = st.session_state.production_start_year
    end_year = st.session_state.production_end_year
    schedule = (st.session_state.get('capacity_schedule') or {}).copy()

    if not schedule:
        base_cfg = CompanyConfig(start_year=start_year, production_end_year=end_year)
        schedule = base_cfg.capacity_utilization.copy()

    aligned = {}
    last_val = None
    for year in range(start_year, end_year + 1):
        if year in schedule:
            last_val = schedule[year]
        if last_val is None:
            last_val = 0.5
        aligned[year] = max(0.0, min(last_val, 2.0))

    st.session_state.capacity_schedule = aligned
    return aligned


def _capacity_years() -> List[int]:
    schedule = _ensure_capacity_schedule()
    return sorted(schedule.keys())


def _set_capacity_value(year: int, utilization_pct: float):
    schedule = st.session_state.setdefault('capacity_schedule', {})
    schedule[year] = max(0.0, min(utilization_pct / 100.0, 2.0))
    st.session_state.capacity_schedule = dict(sorted(schedule.items()))
    _ensure_capacity_schedule()


def _remove_capacity_year(year: int):
    schedule = st.session_state.get('capacity_schedule', {})
    schedule.pop(year, None)
    st.session_state.capacity_schedule = schedule
    _ensure_capacity_schedule()


def _increment_capacity_years(years: List[int], increment_pct: float):
    schedule = st.session_state.setdefault('capacity_schedule', {})
    multiplier = 1 + (increment_pct / 100.0)
    for year in years:
        value = schedule.get(year)
        if value is None:
            continue
        schedule[year] = max(0.0, min(value * multiplier, 2.0))
    st.session_state.capacity_schedule = schedule
    _ensure_capacity_schedule()


def _default_owner_pct() -> float:
    return float(st.session_state.get('owner_equity_pct', 70.0))


def _investment_years() -> List[int]:
    base_years = list(range(st.session_state.production_start_year, st.session_state.production_end_year + 1))
    overrides = st.session_state.get('investment_overrides', {})
    extra_years = sorted(set(overrides.keys()) - set(base_years))
    return sorted(base_years + extra_years)


def _get_owner_pct_for_year(year: int) -> float:
    overrides = st.session_state.get('investment_overrides', {})
    return float(overrides.get(year, _default_owner_pct()))


def _set_owner_pct_for_year(year: int, pct: float):
    pct = max(0.0, min(float(pct), 100.0))
    overrides = st.session_state.setdefault('investment_overrides', {})
    overrides[year] = pct
    st.session_state['investment_overrides'] = dict(sorted(overrides.items()))


def _remove_investment_entry(year: int):
    overrides = st.session_state.get('investment_overrides', {})
    if year in overrides:
        overrides.pop(year, None)
        st.session_state['investment_overrides'] = overrides


def _increment_investment_entries(
    years: List[int],
    increment_pct: float,
    owner_amount_increment: float = 0.0,
    investor_amount_increment: float = 0.0,
    debt_amount_increment: float = 0.0,
):
    if not years:
        return
    for year in years:
        updated = _get_owner_pct_for_year(year) + increment_pct
        _set_owner_pct_for_year(year, updated)
    _increment_investment_amounts(years, owner_amount_increment, investor_amount_increment, debt_amount_increment)


def _get_investment_amount_defaults() -> Dict[int, Dict[str, float]]:
    return st.session_state.get('investment_default_amounts', {})


def _get_investment_amount_record(year: int) -> Dict[str, float]:
    defaults = _get_investment_amount_defaults()
    base = defaults.get(year, {"owner_equity": 0.0, "investor_equity": 0.0, "debt_raised": 0.0})
    overrides = st.session_state.get('investment_amounts', {})
    record = overrides.get(year)
    if record:
        return {
            "owner_equity": float(record.get('owner_equity', base['owner_equity'])),
            "investor_equity": float(record.get('investor_equity', base['investor_equity'])),
            "debt_raised": float(record.get('debt_raised', base['debt_raised'])),
        }
    return base


def _set_investment_amount_record(year: int, owner_amount: float, investor_amount: float, debt_amount: float):
    amounts = st.session_state.setdefault('investment_amounts', {})
    amounts[year] = {
        'owner_equity': max(0.0, float(owner_amount)),
        'investor_equity': max(0.0, float(investor_amount)),
        'debt_raised': max(0.0, float(debt_amount)),
    }
    st.session_state['investment_amounts'] = dict(sorted(amounts.items()))


def _remove_investment_amount_record(year: int):
    amounts = st.session_state.get('investment_amounts', {})
    if year in amounts:
        amounts.pop(year, None)
        st.session_state['investment_amounts'] = dict(sorted(amounts.items()))


def _increment_investment_amounts(
    years: List[int], owner_increment: float, investor_increment: float, debt_increment: float
):
    if not years:
        return
    for year in years:
        record = _get_investment_amount_record(year)
        _set_investment_amount_record(
            year,
            record['owner_equity'] + owner_increment,
            record['investor_equity'] + investor_increment,
            record['debt_raised'] + debt_increment,
        )


def _normalize_product_mix():
    mix = st.session_state.setdefault('product_mix', {})
    total = sum(max(0.0, val) for val in mix.values())
    if total <= 0:
        defaults = CompanyConfig().product_mix.copy()
        st.session_state.product_mix = defaults
        return
    for product in list(mix.keys()):
        mix[product] = max(0.0, mix[product]) / total


def _get_product_base_price(product: str) -> float:
    prices = st.session_state.get('product_base_prices') or {}
    if product in prices:
        return prices[product]
    defaults = CompanyConfig().selling_price.copy()
    return defaults.get(product, 0.0)


def _set_product_base_price(product: str, price: float):
    prices = st.session_state.setdefault('product_base_prices', {})
    prices[product] = max(0.0, price)


def _get_product_price(product: str, year: int) -> float:
    overrides = st.session_state.get('product_price_overrides', {})
    product_overrides = overrides.get(product, {})
    if year in product_overrides:
        return product_overrides[year]
    return _get_product_base_price(product)


def _apply_price_override(product: str, year: int, price: float):
    overrides = st.session_state.setdefault('product_price_overrides', {})
    product_overrides = overrides.setdefault(product, {})
    product_overrides[year] = max(0.0, price)


def _remove_price_override(product: str, year: int):
    overrides = st.session_state.get('product_price_overrides', {})
    product_overrides = overrides.get(product)
    if not product_overrides:
        return
    product_overrides.pop(year, None)
    if not product_overrides:
        overrides.pop(product, None)


WC_CATEGORY_OPTIONS = [
    "Receivables",
    "Inventory",
    "Payables",
    "Accrued Expenses",
]

WC_CATEGORY_KEY_MAP = {
    "Receivables": "receivables",
    "Inventory": "inventory",
    "Payables": "payables",
    "Accrued Expenses": "accrued_expenses",
}


def _wc_category_key(label: str) -> str:
    return WC_CATEGORY_KEY_MAP.get(label, label.lower().replace(" ", "_"))


OPEX_CATEGORY_OPTIONS = [
    "Marketing",
    "Labor",
    "Other Operating Cost",
]

OPEX_CATEGORY_KEY_MAP = {
    "Marketing": "marketing",
    "Labor": "labor",
    "Other Operating Cost": "other",
}


def _update_working_capital_entry(year: int, category: str, amount: float):
    """Persist a working-capital override for the selected category."""

    key = _wc_category_key(category)
    overrides = st.session_state.working_capital_overrides.setdefault(key, {})
    overrides[year] = amount


def _remove_working_capital_entry(year: int, category: str):
    """Remove a working-capital override entry."""

    key = _wc_category_key(category)
    if key in st.session_state.working_capital_overrides:
        st.session_state.working_capital_overrides[key].pop(year, None)
        if not st.session_state.working_capital_overrides[key]:
            st.session_state.working_capital_overrides.pop(key, None)


def _update_operating_expense_entry(year: int, category: str, amount: float):
    """Persist an operating expense override based on the selected category."""

    if category == 'Marketing':
        st.session_state.marketing_budget_schedule[year] = amount
    elif category == 'Labor':
        st.session_state.labor_cost_overrides[year] = amount
    else:
        st.session_state.other_cost_overrides[year] = amount


def _remove_operating_expense_entry(year: int, category: str):
    """Remove an override entry for the specified operating expense."""

    if category == 'Marketing':
        default_value = st.session_state.marketing_budget_defaults.get(year, 72_000.0)
        st.session_state.marketing_budget_schedule[year] = default_value
    elif category == 'Labor':
        st.session_state.labor_cost_overrides.pop(year, None)
    else:
        st.session_state.other_cost_overrides.pop(year, None)


FINANCING_COMPONENT_MAP = {
    "Interest": "interest_overrides",
    "Loan Repayment": "loan_repayment_overrides",
    "Cash Flow from Financing": "financing_cash_flow_overrides",
}


def _financing_meta_key(year: int, component: str) -> str:
    return f"{component}:{year}"


def _get_financing_metadata(year: int, component: str) -> dict:
    key = _financing_meta_key(year, component)
    metadata = st.session_state.get('financing_metadata', {}).get(key, {})
    return copy.deepcopy(metadata)


def _persist_financing_metadata(year: int, component: str, metadata: dict):
    if metadata is None:
        return
    store = st.session_state.setdefault('financing_metadata', {})
    store[_financing_meta_key(year, component)] = copy.deepcopy(metadata)


def _remove_financing_metadata(year: int, component: str):
    store = st.session_state.get('financing_metadata')
    if not store:
        return
    store.pop(_financing_meta_key(year, component), None)


def _update_financing_entry(year: int, component: str, amount: float, metadata: dict | None = None):
    """Persist a financing override for the selected component."""

    key = FINANCING_COMPONENT_MAP.get(component)
    if not key:
        return
    overrides = st.session_state.setdefault(key, {})
    overrides[year] = amount
    if metadata is not None:
        _persist_financing_metadata(year, component, metadata)


def _remove_financing_entry(year: int, component: str):
    """Remove a financing override entry for the specified component."""

    key = FINANCING_COMPONENT_MAP.get(component)
    if not key:
        return
    if key in st.session_state:
        st.session_state[key].pop(year, None)
    _remove_financing_metadata(year, component)


def _build_company_config() -> CompanyConfig:
    """Create a CompanyConfig from current session values and overrides."""

    marketing_budget = _ensure_marketing_schedule()
    capacity_schedule = _ensure_capacity_schedule()
    _normalize_product_mix()
    product_mix = copy.deepcopy(st.session_state.get('product_mix', {}))
    base_prices = copy.deepcopy(st.session_state.get('product_base_prices', {}))
    price_overrides = copy.deepcopy(st.session_state.get('product_price_overrides', {}))

    return CompanyConfig(
        start_year=st.session_state.production_start_year,
        production_end_year=st.session_state.production_end_year,
        capacity_utilization=capacity_schedule,
        labor_manager=st.session_state.labor_manager,
        capex_manager=st.session_state.capex_manager,
        labor_cost_overrides=st.session_state.get('labor_cost_overrides', {}),
        opex_overrides=st.session_state.get('fixed_cost_overrides', {}),
        cogs_overrides=st.session_state.get('variable_cost_overrides', {}),
        loan_repayment_overrides=st.session_state.get('loan_repayment_overrides', {}),
        interest_overrides=st.session_state.get('interest_overrides', {}),
        financing_cash_flow_overrides=st.session_state.get('financing_cash_flow_overrides', {}),
        other_cost_overrides=st.session_state.get('other_cost_overrides', {}),
        working_capital_overrides=st.session_state.get('working_capital_overrides', {}),
        marketing_budget=marketing_budget,
        product_mix=product_mix,
        selling_price=base_prices,
        product_price_overrides=price_overrides,
    )


def _editable_schedule(
    title: str,
    df: pd.DataFrame,
    override_key: str,
    value_column: str,
    help_text: str,
):
    """Render an editable schedule with add/remove/save controls."""

    st.markdown(f"#### {title}")
    st.caption(help_text)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"{override_key}_editor",
    )

    if st.button(f"Save {title} Changes", key=f"{override_key}_save"):
        overrides = {}
        for _, row in edited.iterrows():
            try:
                year = int(row.get("Year"))
                overrides[year] = float(row.get(value_column, 0.0))
            except Exception:
                continue

        st.session_state[override_key] = overrides
        st.success(f"Saved {title.lower()} updates. Schedules will refresh with the new values.")


def _render_labor_management_section():
    """Render labor CRUD tools inside the Platform Settings page."""

    st.markdown("#### Labor Position Management")

    tab1, tab2, tab3 = st.tabs(["Current Positions", "Add Position", "Edit Position"])

    with tab1:
        st.markdown("## Current Labor Schedule")

        positions_df = st.session_state.labor_manager.get_position_summary()
        st.dataframe(positions_df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)

        labor_costs = st.session_state.labor_manager.get_labor_cost_by_type(
            2026, st.session_state.salary_growth_rate
        )
        total_cost = labor_costs['Direct'] + labor_costs['Indirect']
        hc = st.session_state.labor_manager.get_total_headcount(2026)

        with col1:
            st.metric("Total Headcount", hc)
        with col2:
            st.metric("Total Annual Cost", f"${total_cost/1e6:.2f}M")
        with col3:
            st.metric("Cost per Employee", f"${total_cost/hc:,.0f}" if hc > 0 else "N/A")

    with tab2:
        st.markdown("## Add New Labor Position")

        col1, col2 = st.columns(2)

        with col1:
            position_name = st.text_input("Position Name", value="New Position")
            labor_type = st.selectbox("Labor Type", [LaborType.DIRECT, LaborType.INDIRECT])
            job_category = st.selectbox("Job Category", list(JobCategory))
            headcount = st.number_input("Headcount", min_value=1, value=1, step=1)
            annual_salary = st.number_input("Annual Salary ($)", min_value=20000, value=40000, step=5000)

        with col2:
            status = st.selectbox("Employment Status", list(EmploymentStatus))
            start_year = st.number_input("Start Year", min_value=2026, value=2026, step=1)
            benefits_percent = st.slider("Benefits % of Salary", 0.0, 1.0, 0.25, step=0.05)
            overtime_hours = st.number_input("Annual Overtime Hours", min_value=0, value=0, step=50)
            training_cost = st.number_input("Annual Training Cost ($)", min_value=0, value=0, step=500)

        if st.button("Add Position"):
            try:
                position_id = st.session_state.labor_manager.add_position(
                    position_name=position_name,
                    labor_type=labor_type,
                    job_category=job_category,
                    headcount=headcount,
                    annual_salary=annual_salary,
                    status=status,
                    start_year=start_year,
                    benefits_percent=benefits_percent,
                    overtime_hours_annual=overtime_hours,
                    training_cost=training_cost,
                )
                st.success(f"Position added. ID: {position_id}")
                st.session_state.last_update = datetime.now()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab3:
        st.markdown("## Edit Labor Position")

        positions = st.session_state.labor_manager.get_all_positions()
        if not positions:
            st.warning("No positions to edit")
            return

        pos_display = {f"{p.position_id} - {p.position_name}": p.position_id for p in positions}
        selected_display = st.selectbox("Select Position", list(pos_display.keys()))
        selected_id = pos_display[selected_display]
        pos = st.session_state.labor_manager.get_position(selected_id)

        col1, col2 = st.columns(2)

        with col1:
            new_name = st.text_input("Position Name", value=pos.position_name)
            new_headcount = st.number_input("Headcount", min_value=0, value=pos.headcount)
            new_salary = st.number_input("Annual Salary", min_value=0, value=int(pos.annual_salary))

        with col2:
            new_status = st.selectbox(
                "Status", list(EmploymentStatus), index=list(EmploymentStatus).index(pos.status)
            )
            new_benefits = st.slider("Benefits %", 0.0, 1.0, pos.benefits_percent)
            new_training = st.number_input("Training Cost", min_value=0, value=int(pos.training_cost_annual))

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Save Changes", key="labor_save_changes"):
                try:
                    st.session_state.labor_manager.edit_position(
                        selected_id,
                        position_name=new_name,
                        headcount=new_headcount,
                        annual_salary=new_salary,
                        status=new_status,
                        benefits_percent=new_benefits,
                        training_cost_annual=new_training,
                    )
                    st.success("Position updated.")
                    st.session_state.last_update = datetime.now()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with action_cols[1]:
            if st.button("Remove Position", key="labor_remove_position"):
                try:
                    st.session_state.labor_manager.remove_position(selected_id)
                    st.success("Position removed.")
                    st.session_state.last_update = datetime.now()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def _render_capex_management_section():
    """Render CAPEX CRUD tools within Platform Settings."""

    st.markdown("#### Capital Expenditure Management")

    tab1, tab2, tab3 = st.tabs(["Current CAPEX", "Add CAPEX", "Edit CAPEX"])

    with tab1:
        st.markdown("### Current CAPEX Schedule")

        assets = st.session_state.capex_manager.list_items()
        items_data = []
        for item in assets:
            items_data.append({
                'ID': item.item_id,
                'Name': item.name,
                'Category': item.category,
                'Acquisition ($M)': f"${item.amount/1e6:.2f}",
                'Asset Additions ($M)': f"${item.asset_additions/1e6:.2f}",
                'Depreciation Rate (%)': f"{item.depreciation_rate * 100:.1f}%",
                'Start Date': item.start_date,
                'Useful Life (years)': item.useful_life,
            })

        if items_data:
            items_df = pd.DataFrame(items_data)
            st.dataframe(items_df, use_container_width=True, hide_index=True)
        else:
            st.info("No capital assets configured")

        col1, col2, col3 = st.columns(3)
        total_capex = st.session_state.capex_manager.total_capex()
        schedule_start = int(st.session_state.production_start_year)
        horizon_years = max(
            1, int(st.session_state.production_end_year - schedule_start + 1)
        )
        deprec_schedule = st.session_state.capex_manager.depreciation_schedule(
            schedule_start, horizon_years
        )

        with col1:
            st.metric("Total CAPEX", f"${total_capex/1e6:.2f}M")
        with col2:
            st.metric("# Assets", len(assets))
        with col3:
            st.metric(
                f"{schedule_start} Depreciation",
                f"${deprec_schedule.get(schedule_start, 0)/1e3:.0f}K",
            )

    with tab2:
        st.markdown("### Add New CAPEX Asset")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Asset Name", value="New Asset")
            category = st.text_input("Asset Category", value="Equipment")
            amount = st.number_input(
                "Acquisition Cost ($)", min_value=10000, value=100000, step=10000
            )
            asset_additions = st.number_input(
                "Asset Additions ($)",
                min_value=0.0,
                value=0.0,
                step=5000.0,
                key="capex_add_asset_additions",
            )
            depreciation_rate_pct = st.number_input(
                "Depreciation Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key="capex_add_dep_rate",
            )
            useful_life = st.number_input("Asset Useful Life (years)", min_value=1, value=10)

        with col2:
            salvage_value = st.number_input(
                "Salvage Value ($)", min_value=0, value=0, step=5000
            )
            start_date_value = st.date_input(
                "Start Date",
                value=_default_asset_start_date(int(st.session_state.production_start_year)),
                key="capex_add_start_date",
            )
            notes = st.text_area("Notes", value="", key="capex_add_notes")

        if st.button("Add CAPEX Asset", key="capex_add_btn"):
            try:
                asset_id = st.session_state.capex_manager.add_item(
                    name=name,
                    amount=amount,
                    start_year=start_date_value.year,
                    useful_life=int(useful_life),
                    salvage_value=salvage_value,
                    category=category,
                    notes=notes,
                    depreciation_rate=depreciation_rate_pct / 100.0,
                    asset_additions=asset_additions,
                    start_date=start_date_value.isoformat(),
                )
                st.success(f"Asset added. ID: {asset_id}")
                st.session_state.last_update = datetime.now()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab3:
        st.markdown("### Edit CAPEX Asset")

        assets = st.session_state.capex_manager.list_items()
        if not assets:
            st.warning("No assets to edit")
            return

        asset_display = {f"{item.item_id} - {item.name}": item.item_id for item in assets}
        selected_display = st.selectbox("Select Asset", list(asset_display.keys()))
        selected_id = asset_display[selected_display]
        asset = st.session_state.capex_manager.get_item(selected_id)

        col1, col2 = st.columns(2)

        with col1:
            new_name = st.text_input("Name", value=asset.name)
            new_category = st.text_input("Category", value=asset.category)
            new_amount = st.number_input("Acquisition Cost ($)", min_value=0, value=int(asset.amount))
            new_asset_additions = st.number_input(
                "Asset Additions ($)",
                min_value=0.0,
                value=float(asset.asset_additions),
                step=5000.0,
                key=f"capex_edit_asset_additions_{selected_id}",
            )
            new_dep_rate = st.number_input(
                "Depreciation Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(asset.depreciation_rate * 100),
                step=0.5,
                key=f"capex_edit_dep_rate_{selected_id}",
            )

        with col2:
            default_date = _parse_capex_start_date(asset.start_date, asset.start_year)
            new_start_date = st.date_input(
                "Start Date",
                value=default_date,
                key=f"capex_edit_start_{selected_id}",
            )
            new_life = st.number_input(
                "Asset Useful Life (years)", min_value=1, value=asset.useful_life
            )
            new_salvage = st.number_input(
                "Salvage Value ($)", min_value=0, value=int(asset.salvage_value)
            )
            new_notes = st.text_area(
                "Notes", value=asset.notes, key=f"capex_edit_notes_{selected_id}"
            )

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Save Changes", key=f"capex_save_changes_{selected_id}"):
                try:
                    st.session_state.capex_manager.edit_item(
                        selected_id,
                        name=new_name,
                        category=new_category,
                        amount=new_amount,
                        asset_additions=new_asset_additions,
                        depreciation_rate=new_dep_rate / 100.0,
                        start_date=new_start_date.isoformat(),
                        start_year=new_start_date.year,
                        useful_life=new_life,
                        salvage_value=new_salvage,
                        notes=new_notes,
                    )
                    st.success("Asset updated.")
                    st.session_state.last_update = datetime.now()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with action_cols[1]:
            if st.button("Remove Asset", key=f"capex_remove_asset_{selected_id}"):
                try:
                    st.session_state.capex_manager.remove_item(selected_id)
                    st.success("Asset removed.")
                    st.session_state.last_update = datetime.now()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# =====================================================
# MAIN NAVIGATION
# =====================================================

tab_platform, tab_dashboard, tab_ai, tab_reports, tab_advanced = st.tabs([
    "Platform Settings",
    "Dashboard",
    "AI & Machine Learning",
    "Reports",
    "Advanced Analytics",
])

# =====================================================
# PAGE 0: PLATFORM SETTINGS
# =====================================================

with tab_platform:
    st.markdown("# Platform Settings")

    st.markdown("### Global Parameters")
    col_start, col_end = st.columns(2)
    with col_start:
        start_year_input = st.number_input(
            "Production Start Year",
            min_value=1900,
            max_value=2100,
            value=int(st.session_state.production_start_year),
            step=1,
        )
    with col_end:
        end_year_input = st.number_input(
            "Production End Year",
            min_value=int(start_year_input),
            max_value=2100,
            value=int(max(st.session_state.production_end_year, start_year_input)),
            step=1,
        )

    st.session_state.production_start_year = int(start_year_input)
    st.session_state.production_end_year = int(end_year_input)

    salary_growth = st.slider(
        "Annual Salary Growth Rate (%)",
        min_value=0,
        max_value=10,
        value=int(st.session_state.salary_growth_rate * 100),
        help="Applied to all labor cost projections",
    )
    st.session_state.salary_growth_rate = salary_growth / 100

    st.markdown("### Platform Info")
    st.info(
        "**Manufacturing Financial Platform v1.0**\n\n"
        "• Labor management (CRUD)\n"
        "• CAPEX scheduling\n"
        "• Financial modeling\n"
        "• Advanced reporting"
    )

    st.markdown("### Financial Model & Valuation")
    fm_run_tab, fm_results_tab = st.tabs(["Run Model", "Results"])

    with fm_run_tab:
        st.markdown("Configure valuation drivers and execute the financial engine. The results will refresh in the sections below and on the Reports page.")

        default_cfg = CompanyConfig()
        col1, col2 = st.columns(2)

        with col1:
            wacc = st.slider("WACC (%)", 0, 20, int(default_cfg.wacc * 100)) / 100
            terminal_growth = st.slider("Terminal Growth Rate (%)", 0, 5, int(default_cfg.terminal_growth * 100)) / 100
            cogs_percent = st.slider("COGS % of Revenue", 0, 100, int(default_cfg.cogs_ratio * 100)) / 100

        with col2:
            tax_rate = st.slider("Tax Rate (%)", 0, 50, int(default_cfg.tax_rate * 100)) / 100
            debt_amount = st.number_input("Debt ($M)", min_value=0, value=int(default_cfg.loan_amount / 1e6)) * 1e6
            interest_rate = st.slider("Interest Rate (%)", 0, 15, int(default_cfg.loan_interest_rate * 100)) / 100

        if st.button("Run Financial Model", key="platform_run_financial_model"):
            try:
                cfg_override = CompanyConfig(
                    start_year=st.session_state.production_start_year,
                    production_end_year=st.session_state.production_end_year,
                    cogs_ratio=cogs_percent,
                    tax_rate=tax_rate,
                    wacc=wacc,
                    terminal_growth=terminal_growth,
                    loan_amount=debt_amount,
                    loan_interest_rate=interest_rate,
                    labor_manager=st.session_state.labor_manager,
                    capex_manager=st.session_state.capex_manager,
                )

                st.session_state.financial_model = run_financial_model(cfg_override)
                st.session_state.last_update = datetime.now()
                st.success("Model executed successfully. Results have been refreshed.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with fm_results_tab:
        if st.session_state.financial_model:
            model = st.session_state.financial_model
            years = list(model["years"])

            st.markdown("#### Financial Highlights")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Enterprise Value", f"${model['enterprise_value']/1e6:.1f}M")
            with col2:
                st.metric("5-Year FCF", f"${sum(model['fcf'].values())/1e6:.1f}M")
            with col3:
                ev_revenue = model['enterprise_value'] / model['revenue'][years[0]] if model['revenue'][years[0]] else 0
                st.metric("EV/Revenue", f"{ev_revenue:.1f}x")
            with col4:
                terminal_revenue = model['revenue'][years[-1]]
                st.metric("Terminal Value", f"${terminal_revenue*5/1e6:.1f}M")

            forecast_df = pd.DataFrame({
                'Year': years,
                'Revenue': [f"${model['revenue'][y]/1e6:.1f}M" for y in years],
                'EBIT': [f"${model['ebit'][y]/1e6:.1f}M" for y in years],
                'FCF': [f"${model['fcf'][y]/1e6:.1f}M" for y in years],
                'Cash': [f"${model['cash_balance'][y]/1e6:.1f}M" for y in years],
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            chart_df = pd.DataFrame({
                'Year': years,
                'Revenue': [model['revenue'][y] for y in years],
                'EBIT': [model['ebit'][y] for y in years],
                'FCF': [model['fcf'][y] for y in years],
            })
            fig = px.line(chart_df, x='Year', y=['Revenue', 'EBIT', 'FCF'], markers=True, title="5-Year Forecast")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the financial model to view forecasts and valuation metrics.")

    st.markdown("### Schedules")
    cfg = _build_company_config()
    model = run_financial_model(cfg)
    years = list(model["years"])
    income_df, cashflow_df, balance_df = generate_financial_statements(model)
    # Build labor cost schedule for editing
    labor_rows = []
    for y in years:
        costs = st.session_state.labor_manager.get_labor_cost_by_type(y, st.session_state.salary_growth_rate)
        headcounts = st.session_state.labor_manager.get_headcount_by_type(y)
        total_hc = headcounts.get('Direct', 0) + headcounts.get('Indirect', 0)
        total_cost = costs.get('Direct', 0) + costs.get('Indirect', 0)
        labor_rows.append(
            {
                'Year': y,
                'Direct Labor Cost': costs.get('Direct', 0),
                'Indirect Labor Cost': costs.get('Indirect', 0),
                'Total Labor Cost': total_cost,
                'Total Headcount': total_hc,
                'Avg Cost per HC': total_cost / total_hc if total_hc else 0.0,
            }
        )

    labor_df = pd.DataFrame(labor_rows)
    if not labor_df.empty and st.session_state.labor_cost_overrides:
        overrides = st.session_state.labor_cost_overrides
        labor_df['Total Labor Cost'] = [overrides.get(y, val) for y, val in zip(labor_df['Year'], labor_df['Total Labor Cost'])]
        labor_df['Avg Cost per HC'] = [
            (labor_df.loc[idx, 'Total Labor Cost'] / labor_df.loc[idx, 'Total Headcount']) if labor_df.loc[idx, 'Total Headcount'] else 0.0
            for idx in labor_df.index
        ]
        labor_df['Override Applied'] = [overrides.get(y) if y in overrides else None for y in labor_df['Year']]

    capex_spend = st.session_state.capex_manager.yearly_capex_schedule(years[0], len(years))
    depreciation_sched = st.session_state.capex_manager.depreciation_schedule(years[0], len(years))
    capex_df = pd.DataFrame(
        {
            "Year": years,
            "CAPEX Spend": [capex_spend.get(y, 0.0) for y in years],
            "Depreciation": [depreciation_sched.get(y, 0.0) for y in years],
        }
    )

    capacity_schedule = _ensure_capacity_schedule()
    production_schedule_df = pd.DataFrame({
        "Year": years,
        "Annual Capacity": [model["config"].annual_capacity for _ in years],
        "Capacity Utilization": [capacity_schedule.get(y, _get_capacity_for_year(cfg, y)) * 100 for y in years],
        "Units Produced": [model["production_volume"][y] for y in years],
        "Revenue": [model["revenue"][y] for y in years],
    })

    start_year = years[0]
    product_mix_df = pd.DataFrame(
        [
            {
                "Product": product,
                "Mix %": mix * 100,
                "Unit Price": _get_product_price(product, start_year),
                "Units (Start Year)": model["production_volume"][start_year] * mix,
                "Revenue (Start Year)": model["production_volume"][start_year] * mix * _get_product_price(product, start_year),
            }
            for product, mix in model["product_mix"].items()
        ]
    )

    pricing_rows = []
    for year in years:
        total_units = model["production_volume"][year]
        for product, mix in model["product_mix"].items():
            unit_price = _get_product_price(product, year)
            units = total_units * mix
            pricing_rows.append(
                {
                    "Year": year,
                    "Product": product,
                    "Mix %": mix * 100,
                    "Unit Price": unit_price,
                    "Units": units,
                    "Revenue": units * unit_price,
                }
            )

    pricing_schedule_df = pd.DataFrame(pricing_rows)

    working_cap_df = pd.DataFrame({
        "Year": years,
        "FCF": [model["fcf"][y] for y in years],
        "Discounted FCF": [model["discounted_fcf"][y] for y in years],
        "Working Capital Change": [model["working_capital_change"].get(y, 0.0) for y in years],
    })

    opex_breakdown_rows = []
    for y in years:
        breakdown = model.get("opex_breakdown", {}).get(y, {})
        opex_breakdown_rows.append(
            {
                "Year": y,
                "Marketing": breakdown.get("marketing", 0.0),
                "Labor": breakdown.get("labor", 0.0),
                "Other Operating Cost": breakdown.get("other", 0.0),
                "Total Opex": model.get("opex", {}).get(y, 0.0),
                "Override Applied": breakdown.get("override_applied"),
            }
        )

    opex_breakdown_df = pd.DataFrame(opex_breakdown_rows)

    wc_positions = model.get("working_capital", {})
    wc_accrual_df = pd.DataFrame(
        {
            "Year": years,
            "Receivables": [wc_positions.get("receivables", {}).get(y, 0.0) for y in years],
            "Inventory": [wc_positions.get("inventory", {}).get(y, 0.0) for y in years],
            "Payables": [wc_positions.get("payables", {}).get(y, 0.0) for y in years],
            "Accrued Expenses": [wc_positions.get("accrued_expenses", {}).get(y, 0.0) for y in years],
            "Net Working Capital": [wc_positions.get("net_working_capital", {}).get(y, 0.0) for y in years],
            "Change in Working Capital": [wc_positions.get("delta_working_capital", {}).get(y, 0.0) for y in years],
        }
    )

    wc_override_labels = {}
    label_lookup = {v: k for k, v in WC_CATEGORY_KEY_MAP.items()}
    for category_key, overrides in st.session_state.working_capital_overrides.items():
        label = label_lookup.get(category_key, category_key.replace('_', ' ').title())
        for year, _ in overrides.items():
            wc_override_labels.setdefault(year, []).append(label)

    if wc_override_labels:
        wc_accrual_df["Override Categories"] = [
            ", ".join(wc_override_labels.get(y, [])) if wc_override_labels.get(y) else ""
            for y in wc_accrual_df["Year"]
        ]

    financing_df = pd.DataFrame({
        "Year": years,
        "Interest": [model["interest_payment"][y] for y in years],
        "Loan Repayment": [model["loan_repayment"][y] for y in years],
        "Long Term Debt": [model["long_term_debt"][y] for y in years],
        "Cash Flow from Financing": [model["cff"][y] for y in years],
    })

    depreciation_values = [model["depreciation"][y] if isinstance(model["depreciation"], dict) else model["depreciation"] for y in years]
    fixed_cost_df = pd.DataFrame({
        "Year": years,
        "Fixed Operating Costs": [model["opex"][y] for y in years],
        "Depreciation": depreciation_values,
    })

    variable_cost_df = pd.DataFrame({
        "Year": years,
        "COGS (Variable)": [model["cogs"][y] for y in years],
    })

    other_cost_df = pd.DataFrame({
        "Year": years,
        "Other Operating Cost": [model["opex_breakdown"].get(y, {}).get("other", 0.0) for y in years],
    })

    debt_outstanding = []
    debt_balance = model["config"].loan_amount
    for y in years:
        debt_balance = max(0.0, debt_balance - model["loan_repayment"].get(y, 0.0))
        debt_outstanding.append(debt_balance)
    debt_schedule_df = pd.DataFrame({
        "Year": years,
        "Interest": [model["interest_payment"].get(y, 0.0) for y in years],
        "Principal": [model["loan_repayment"].get(y, 0.0) for y in years],
        "Ending Balance": debt_outstanding,
    })

    assets_df = pd.DataFrame({
        "Year": years,
        "Fixed Assets": [model["fixed_assets"][y] for y in years],
        "Current Assets": [model["current_assets"][y] for y in years],
        "Total Assets": [model["total_assets"][y] for y in years],
    })

    if st.session_state.asset_overrides:
        assets_df["Total Assets"] = [
            st.session_state.asset_overrides.get(y, val)
            for y, val in zip(assets_df["Year"], assets_df["Total Assets"])
        ]

    assembly_df = pd.DataFrame({
        "Year": years,
        "Annual Capacity": [model["config"].annual_capacity for _ in years],
        "Capacity Utilization": [model["config"].capacity_utilization.get(y, 0.0) for y in years],
        "Units Produced": [model["production_volume"][y] for y in years],
        "Working Days": [model["config"].working_days for _ in years],
    })

    schedule_tabs = st.tabs([
        "Labor Cost",
        "CAPEX",
        "Production Schedule",
        "Operating Expense Breakdown",
        "Working Capital Accruals",
        "Working Capital & FCF",
        "Financing",
        "Fixed Cost",
        "Variable Cost",
        "Other Cost",
        "Debt",
        "Investment",
        "Assets",
        "Automobile & Assembly",
    ])

    (
        tab_labor,
        tab_capex,
        tab_production,
        tab_opex_breakdown,
        tab_wc_accruals,
        tab_wc_fcf,
        tab_financing,
        tab_fixed_cost,
        tab_variable_cost,
        tab_other_cost,
        tab_debt,
        tab_investment,
        tab_assets,
        tab_auto,
    ) = schedule_tabs

    with tab_labor:
        if labor_df.empty:
            st.info("No labor schedule available. Add positions to view costs and headcount.")
        else:
            _editable_schedule(
                "Labor Cost Schedule",
                labor_df,
                "labor_cost_overrides",
                "Total Labor Cost",
                "Edit labor totals inline; add or remove years to reshape the schedule.",
            )

    with tab_capex:
        st.dataframe(
            _format_statement(capex_df, ["CAPEX Spend", "Depreciation"]),
            use_container_width=True,
            hide_index=True,
        )

    with tab_production:
        st.markdown("#### Production Schedule")
        production_display = production_schedule_df.copy()
        production_display["Capacity Utilization"] = production_display["Capacity Utilization"].apply(lambda v: f"{v:.1f}%")
        prod_tabs = st.tabs([
            "Current Schedule",
            "Add Year",
            "Edit Year",
            "Remove Year",
            "Yearly Increment",
        ])
        prod_years = _capacity_years()

        with prod_tabs[0]:
            st.dataframe(
                _format_statement(production_display, ["Revenue"]),
                use_container_width=True,
                hide_index=True,
            )

        with prod_tabs[1]:
            with st.form("production_add_year_form"):
                year_value = st.number_input(
                    "Production Year",
                    min_value=int(years[0]),
                    max_value=int(years[-1]),
                    value=int(years[-1]),
                    step=1,
                )
                util_value = st.slider("Capacity Utilization (%)", 0, 200, 100)
                submitted = st.form_submit_button("Add Production Year")
            if submitted:
                _set_capacity_value(int(year_value), float(util_value))
                st.success("Production year saved. Rerunning model with the updated schedule.")
                st.rerun()

        with prod_tabs[2]:
            if not prod_years:
                st.info("No production years available to edit.")
            else:
                selected_year = st.selectbox("Select Year", prod_years, key="edit_capacity_year")
                current_pct = int(capacity_schedule.get(selected_year, 0.0) * 100)
                with st.form("production_edit_year_form"):
                    updated_pct = st.slider(
                        "Capacity Utilization (%)",
                        0,
                        200,
                        current_pct,
                        key="edit_capacity_slider",
                    )
                    edit_submitted = st.form_submit_button("Save Changes")
                if edit_submitted:
                    _set_capacity_value(int(selected_year), float(updated_pct))
                    st.success("Production year updated. Rerunning model with the new utilization.")
                    st.rerun()

        with prod_tabs[3]:
            if not prod_years:
                st.info("No production years available to remove.")
            else:
                remove_year = st.selectbox("Select Year to Remove", prod_years, key="remove_capacity_year_select")
                if st.button("Remove Year", key="remove_capacity_year_btn"):
                    _remove_capacity_year(int(remove_year))
                    st.success("Production year removed. Rerunning model with the revised schedule.")
                    st.rerun()

        with prod_tabs[4]:
            if not prod_years:
                st.info("No production years available for increments.")
            else:
                with st.form("production_increment_form"):
                    selected_years = st.multiselect(
                        "Select Years",
                        prod_years,
                        default=prod_years,
                        key="increment_capacity_years",
                    )
                    increment_pct = st.number_input(
                        "Increment (%)",
                        min_value=-100.0,
                        max_value=300.0,
                        value=5.0,
                        step=1.0,
                    )
                    increment_submitted = st.form_submit_button("Apply Increment")
                if increment_submitted and selected_years:
                    _increment_capacity_years([int(y) for y in selected_years], float(increment_pct))
                    st.success("Capacity schedule updated. Rerunning model with the incremented values.")
                    st.rerun()

        st.markdown("#### Product Mix & Pricing")
        mix_display = product_mix_df.copy()
        mix_display["Mix %"] = mix_display["Mix %"].apply(lambda v: f"{v:.1f}%")
        mix_tabs = st.tabs([
            "Current Mix",
            "Add Product",
            "Edit Product",
            "Remove Product",
            "Yearly Increment",
        ])
        product_names = sorted(st.session_state.product_mix.keys())

        with mix_tabs[0]:
            st.dataframe(
                _format_statement(mix_display, ["Unit Price", "Revenue (Start Year)"]),
                use_container_width=True,
                hide_index=True,
            )

        with mix_tabs[1]:
            with st.form("add_product_mix_form"):
                product_name = st.text_input("Product Name", value="New Product")
                mix_pct = st.number_input("Mix %", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
                unit_price = st.number_input("Unit Price ($)", min_value=0.0, value=10_000.0, step=500.0)
                add_product_submitted = st.form_submit_button("Add Product")
            if add_product_submitted:
                key = product_name.strip() or "New Product"
                st.session_state.product_mix[key] = mix_pct / 100.0
                _set_product_base_price(key, unit_price)
                st.session_state.product_price_overrides.setdefault(key, {})
                _normalize_product_mix()
                st.success("Product mix updated. Rerunning model with the new product line.")
                st.rerun()

        with mix_tabs[2]:
            if not product_names:
                st.info("No products available to edit.")
            else:
                selected_product = st.selectbox("Select Product", product_names, key="edit_product_mix_select")
                with st.form("edit_product_mix_form"):
                    new_name = st.text_input("Product Name", value=selected_product)
                    mix_pct = st.number_input(
                        "Mix %",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(st.session_state.product_mix.get(selected_product, 0.0) * 100),
                        step=1.0,
                    )
                    unit_price = st.number_input(
                        "Unit Price ($)",
                        min_value=0.0,
                        value=float(_get_product_base_price(selected_product)),
                        step=500.0,
                    )
                    edit_product_submitted = st.form_submit_button("Save Product Changes")
                if edit_product_submitted:
                    target_name = new_name.strip() or selected_product
                    existing_overrides = st.session_state.product_price_overrides.pop(selected_product, {})
                    base_price = st.session_state.product_base_prices.pop(selected_product, unit_price)
                    st.session_state.product_mix.pop(selected_product, None)
                    st.session_state.product_mix[target_name] = mix_pct / 100.0
                    _set_product_base_price(target_name, unit_price)
                    overrides = st.session_state.product_price_overrides.setdefault(target_name, {})
                    overrides.update(existing_overrides)
                    if target_name not in st.session_state.product_base_prices:
                        st.session_state.product_base_prices[target_name] = base_price
                    _normalize_product_mix()
                    st.success("Product mix updated. Rerunning model with the edited product.")
                    st.rerun()

        with mix_tabs[3]:
            if not product_names:
                st.info("No products available to remove.")
            else:
                remove_product = st.selectbox("Select Product to Remove", product_names, key="remove_product_mix_select")
                if st.button("Remove Product", key="remove_product_mix_btn"):
                    st.session_state.product_mix.pop(remove_product, None)
                    st.session_state.product_base_prices.pop(remove_product, None)
                    st.session_state.product_price_overrides.pop(remove_product, None)
                    _normalize_product_mix()
                    st.success("Product removed from the mix. Rerunning model with the updated configuration.")
                    st.rerun()

        with mix_tabs[4]:
            if not product_names:
                st.info("No products available for price increments.")
            else:
                with st.form("product_price_increment_form"):
                    selected_products = st.multiselect(
                        "Select Products",
                        product_names,
                        default=product_names,
                        key="price_increment_products",
                    )
                    increment_pct = st.number_input(
                        "Increment (%)",
                        min_value=-50.0,
                        max_value=200.0,
                        value=5.0,
                        step=1.0,
                    )
                    increment_prices_submitted = st.form_submit_button("Apply Price Increment")
                if increment_prices_submitted and selected_products:
                    multiplier = 1 + (increment_pct / 100.0)
                    for product in selected_products:
                        _set_product_base_price(product, _get_product_base_price(product) * multiplier)
                        overrides = st.session_state.product_price_overrides.get(product, {})
                        for year_key in list(overrides.keys()):
                            overrides[year_key] = max(0.0, overrides[year_key] * multiplier)
                    st.success("Product pricing updated. Rerunning model with the incremented prices.")
                    st.rerun()

        st.markdown("#### Pricing Schedule")
        price_tabs = st.tabs([
            "Current Pricing",
            "Add Price",
            "Edit Price",
            "Remove Price",
            "Yearly Increment",
        ])

        pricing_display = pricing_schedule_df.copy()
        if not pricing_display.empty:
            pricing_display["Mix %"] = pricing_display["Mix %"].apply(lambda v: f"{v:.1f}%")

        with price_tabs[0]:
            if pricing_display.empty:
                st.info("No pricing data available.")
            else:
                st.dataframe(
                    _format_statement(pricing_display, ["Unit Price", "Revenue"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with price_tabs[1]:
            if not product_names:
                st.info("Add at least one product before configuring prices.")
            else:
                with st.form("add_price_override_form"):
                    product_choice = st.selectbox("Product", product_names, key="pricing_add_product")
                    year_choice = st.selectbox("Year", years, key="pricing_add_year")
                    current_price = _get_product_price(product_choice, int(year_choice))
                    unit_price = st.number_input(
                        "Unit Price ($)",
                        min_value=0.0,
                        value=float(current_price),
                        step=500.0,
                    )
                    price_add_submitted = st.form_submit_button("Save Price")
                if price_add_submitted:
                    _apply_price_override(product_choice, int(year_choice), float(unit_price))
                    st.success("Pricing override saved. Rerunning model with the updated prices.")
                    st.rerun()

        with price_tabs[2]:
            override_entries = []
            for product, entries in st.session_state.get('product_price_overrides', {}).items():
                for year_key in sorted(entries.keys()):
                    override_entries.append((product, year_key))
            if not override_entries:
                st.info("No price overrides available to edit.")
            else:
                labels = [f"{prod} - {yr}" for prod, yr in override_entries]
                selection = st.selectbox("Select Price Override", labels, key="edit_price_override_select")
                sel_index = labels.index(selection)
                sel_product, sel_year = override_entries[sel_index]
                with st.form("edit_price_override_form"):
                    unit_price = st.number_input(
                        "Unit Price ($)",
                        min_value=0.0,
                        value=float(_get_product_price(sel_product, sel_year)),
                        step=500.0,
                    )
                    edit_override_submitted = st.form_submit_button("Update Price")
                if edit_override_submitted:
                    _apply_price_override(sel_product, int(sel_year), float(unit_price))
                    st.success("Pricing override updated. Rerunning model with the new value.")
                    st.rerun()

        with price_tabs[3]:
            override_entries = []
            for product, entries in st.session_state.get('product_price_overrides', {}).items():
                for year_key in sorted(entries.keys()):
                    override_entries.append((product, year_key))
            if not override_entries:
                st.info("No price overrides available to remove.")
            else:
                labels = [f"{prod} - {yr}" for prod, yr in override_entries]
                selection = st.selectbox("Select Override to Remove", labels, key="remove_price_override_select")
                sel_index = labels.index(selection)
                sel_product, sel_year = override_entries[sel_index]
                if st.button("Remove Price Override", key="remove_price_override_btn"):
                    _remove_price_override(sel_product, int(sel_year))
                    st.success("Price override removed. Rerunning model with the base pricing.")
                    st.rerun()

        with price_tabs[4]:
            if not product_names:
                st.info("Add at least one product before applying increments.")
            else:
                with st.form("pricing_increment_form"):
                    selected_products = st.multiselect(
                        "Select Products",
                        product_names,
                        default=product_names,
                        key="pricing_increment_products",
                    )
                    selected_years = st.multiselect(
                        "Select Years",
                        years,
                        default=years,
                        key="pricing_increment_years",
                    )
                    increment_pct = st.number_input(
                        "Increment (%)",
                        min_value=-50.0,
                        max_value=200.0,
                        value=3.0,
                        step=1.0,
                    )
                    pricing_increment_submitted = st.form_submit_button("Apply Increment")
                if pricing_increment_submitted and selected_products and selected_years:
                    multiplier = 1 + (increment_pct / 100.0)
                    for product in selected_products:
                        for year_value in selected_years:
                            current_price = _get_product_price(product, int(year_value))
                            _apply_price_override(product, int(year_value), max(0.0, current_price * multiplier))
                    st.success("Pricing schedule updated. Rerunning model with the incremented schedule.")
                    st.rerun()

    with tab_opex_breakdown:
        st.markdown("#### Operating Expense Breakdown")
        if opex_breakdown_df.empty:
            st.info("No operating expense details available. Add costs to view the breakdown.")
        else:
            (
                curr_tab,
                add_tab,
                edit_tab,
                remove_tab,
                increment_tab,
            ) = st.tabs(
                [
                    "Current Operating Expense",
                    "Add Operating Expense",
                    "Edit Operating Expense",
                    "Remove Operating Expense",
                    "Yearly Increment",
                ]
            )

            opex_entries = []
            for _, row in opex_breakdown_df.iterrows():
                year_val = int(row["Year"])
                for category in OPEX_CATEGORY_OPTIONS:
                    opex_entries.append(
                        {
                            "label": f"{year_val} • {category}",
                            "year": year_val,
                            "category": category,
                            "amount": float(row.get(category, 0.0)),
                        }
                    )

            with curr_tab:
                st.dataframe(
                    _format_statement(
                        opex_breakdown_df,
                        ["Marketing", "Labor", "Other Operating Cost", "Total Opex"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            with add_tab:
                st.markdown("## Add Operating Expense")
                with st.form("add_operating_expense_form"):
                    year_choice = st.selectbox("Year", years, key="add_opex_year")
                    category_choice = st.selectbox(
                        "Expense Type",
                        OPEX_CATEGORY_OPTIONS,
                        key="add_opex_category",
                    )
                    amount_value = st.number_input(
                        "Amount ($)",
                        min_value=0.0,
                        value=50000.0,
                        step=5000.0,
                        key="add_opex_amount",
                    )
                    add_submitted = st.form_submit_button("Add Operating Expense")

                if add_submitted:
                    _update_operating_expense_entry(int(year_choice), category_choice, float(amount_value))
                    st.success("Operating expense saved.")
                    st.rerun()

            with edit_tab:
                st.markdown("## Edit Operating Expense")
                if not opex_entries:
                    st.info("No operating expense entries available.")
                else:
                    option = st.selectbox(
                        "Select expense to edit",
                        options=list(range(len(opex_entries))),
                        format_func=lambda idx: f"{opex_entries[idx]['label']} (${opex_entries[idx]['amount']:,.0f})",
                        key="edit_opex_selector",
                    )
                    selected_entry = opex_entries[option]
                    amount_key = f"edit_opex_amount_{selected_entry['year']}_{selected_entry['category']}"
                    new_amount = st.number_input(
                        "Amount ($)",
                        min_value=0.0,
                        value=float(selected_entry['amount']),
                        step=5000.0,
                        key=amount_key,
                    )
                    save_key = f"save_opex_{selected_entry['year']}_{selected_entry['category']}"
                    if st.button("Save Operating Expense", key=save_key):
                        _update_operating_expense_entry(
                            int(selected_entry['year']), selected_entry['category'], float(new_amount)
                        )
                        st.success("Operating expense updated.")
                        st.rerun()

            with remove_tab:
                st.markdown("## Remove Operating Expense")
                if not opex_entries:
                    st.info("No operating expense entries available to remove.")
                else:
                    labels = [f"{entry['label']} (${entry['amount']:,.0f})" for entry in opex_entries]
                    selections = st.multiselect(
                        "Select expenses to remove",
                        options=labels,
                        key="remove_opex_selector",
                    )
                    if st.button("Remove Selected Expenses", key="remove_opex_btn") and selections:
                        for label in selections:
                            idx = labels.index(label)
                            entry = opex_entries[idx]
                            _remove_operating_expense_entry(int(entry['year']), entry['category'])
                        st.success("Selected operating expenses removed.")
                        st.rerun()

            with increment_tab:
                st.markdown("## Yearly Increment Helper")
                with st.form("opex_increment_form"):
                    selected_categories = st.multiselect(
                        "Expense Categories",
                        OPEX_CATEGORY_OPTIONS,
                        default=OPEX_CATEGORY_OPTIONS,
                        key="opex_increment_categories",
                    )
                    selected_years = st.multiselect(
                        "Years",
                        years,
                        default=years,
                        key="opex_increment_years",
                    )
                    increment_pct = st.number_input(
                        "Increment (%)",
                        min_value=-100.0,
                        max_value=200.0,
                        value=5.0,
                        step=1.0,
                        key="opex_increment_pct",
                    )
                    increment_submitted = st.form_submit_button("Apply Increment")

                if increment_submitted and selected_categories and selected_years:
                    multiplier = 1 + (increment_pct / 100.0)
                    for year_value in selected_years:
                        breakdown = model.get("opex_breakdown", {}).get(int(year_value), {})
                        for category in selected_categories:
                            key = OPEX_CATEGORY_KEY_MAP.get(category)
                            base_amount = float(breakdown.get(key, 0.0))
                            new_amount = max(0.0, base_amount * multiplier)
                            _update_operating_expense_entry(int(year_value), category, new_amount)
                    st.success("Operating expense schedule updated with the incremented values.")
                    st.rerun()

    with tab_wc_accruals:
        st.markdown("#### Working-Capital-Driven Accruals")
        if wc_accrual_df.empty:
            st.info("No working-capital data available. Run the model to populate balances.")
        else:
            current_tab, add_tab, edit_tab = st.tabs([
                "Current Working Capital Accruals",
                "Add Working Capital Accruals",
                "Edit Working Capital Accruals",
            ])

            money_columns = [
                "Receivables",
                "Inventory",
                "Payables",
                "Accrued Expenses",
                "Net Working Capital",
                "Change in Working Capital",
            ]

            with current_tab:
                st.dataframe(
                    _format_statement(wc_accrual_df, money_columns),
                    use_container_width=True,
                    hide_index=True,
                )

            with add_tab:
                st.markdown("## Add Working Capital Accrual")
                with st.form("add_wc_accrual_form"):
                    year_choice = st.selectbox("Year", years, key="add_wc_year")
                    category_choice = st.selectbox(
                        "Component",
                        WC_CATEGORY_OPTIONS,
                        key="add_wc_category",
                    )

                    calculated_amount = 0.0
                    details_text = ""
                    selected_year = int(year_choice)

                    if category_choice == "Receivables":
                        receivable_days_input = st.number_input(
                            "Receivables Days",
                            min_value=0,
                            max_value=365,
                            value=int(cfg.receivable_days),
                            step=1,
                            key="add_wc_receivable_days",
                        )
                        base_value = model["revenue"].get(selected_year, 0.0)
                        calculated_amount = base_value * (receivable_days_input / 365.0)
                        details_text = (
                            f"Receivables = Revenue ${base_value:,.0f} × {receivable_days_input}/365"
                        )
                    elif category_choice == "Inventory":
                        inventory_days_input = st.number_input(
                            "Inventory Days",
                            min_value=0,
                            max_value=365,
                            value=int(cfg.inventory_days),
                            step=1,
                            key="add_wc_inventory_days",
                        )
                        base_value = max(0.0, model["cogs"].get(selected_year, 0.0))
                        calculated_amount = base_value * (inventory_days_input / 365.0)
                        details_text = (
                            f"Inventory = COGS ${base_value:,.0f} × {inventory_days_input}/365"
                        )
                    elif category_choice == "Payables":
                        payables_days_input = st.number_input(
                            "Payables Days",
                            min_value=0,
                            max_value=365,
                            value=int(cfg.payable_days),
                            step=1,
                            key="add_wc_payables_days",
                        )
                        base_value = max(0.0, model["cogs"].get(selected_year, 0.0))
                        calculated_amount = base_value * (payables_days_input / 365.0)
                        details_text = (
                            f"Payables = COGS ${base_value:,.0f} × {payables_days_input}/365"
                        )
                    else:
                        default_pct = min(
                            100.0,
                            max(0.0, (cfg.accrued_expense_days / 365.0) * 100.0),
                        )
                        accrued_pct_input = st.number_input(
                            "Accrued Expenses (% of Operating Expense)",
                            min_value=0.0,
                            max_value=100.0,
                            value=round(default_pct, 2),
                            step=0.25,
                            key="add_wc_accrued_pct",
                        )
                        breakdown = model.get("opex_breakdown", {}).get(selected_year, {})
                        opex_total = sum(
                            breakdown.get(k, 0.0)
                            for k in ("marketing", "labor", "other")
                        )
                        calculated_amount = opex_total * (accrued_pct_input / 100.0)
                        details_text = (
                            f"Accrued = OpEx ${opex_total:,.0f} × {accrued_pct_input:.2f}%"
                        )

                    st.caption(details_text or "Amounts are derived from the selected component's driver.")
                    st.info(f"Calculated Amount: ${calculated_amount:,.0f}")

                    add_wc_submitted = st.form_submit_button("Add Working Capital Accrual")

                if add_wc_submitted:
                    _update_working_capital_entry(
                        int(year_choice), category_choice, float(calculated_amount)
                    )
                    st.success("Working-capital accrual saved.")
                    st.rerun()

            with edit_tab:
                st.markdown("## Edit Working Capital Accrual")
                entries = []
                for _, row in wc_accrual_df.iterrows():
                    year_val = int(row["Year"])
                    for col in WC_CATEGORY_OPTIONS:
                        entries.append(
                            {
                                "label": f"{year_val} • {col}",
                                "year": year_val,
                                "category": col,
                                "amount": float(row[col]),
                            }
                        )

                if not entries:
                    st.info("No working-capital entries available to edit.")
                else:
                    option = st.selectbox(
                        "Select entry",
                        options=list(range(len(entries))),
                        format_func=lambda idx: f"{entries[idx]['label']} (${entries[idx]['amount']:,.0f})",
                        key="edit_wc_selector",
                    )
                    selected_entry = entries[option]
                    amount_key = f"edit_wc_amount_{selected_entry['year']}_{selected_entry['category']}"
                    new_amount = st.number_input(
                        "Amount ($)",
                        value=float(selected_entry["amount"]),
                        step=10_000.0,
                        key=amount_key,
                    )

                    col_save, col_remove = st.columns(2)
                    save_key = f"save_wc_{selected_entry['year']}_{selected_entry['category']}"
                    remove_key = f"remove_wc_{selected_entry['year']}_{selected_entry['category']}"

                    with col_save:
                        if st.button("Save Working Capital Accrual", key=save_key):
                            _update_working_capital_entry(
                                int(selected_entry['year']), selected_entry['category'], float(new_amount)
                            )
                            st.success("Working-capital accrual updated.")
                            st.rerun()

                    with col_remove:
                        if st.button("Remove Working Capital Accrual", key=remove_key):
                            _remove_working_capital_entry(
                                int(selected_entry['year']), selected_entry['category']
                            )
                            st.success("Working-capital accrual removed.")
                            st.rerun()

    with tab_wc_fcf:
        st.dataframe(_format_statement(working_cap_df, ["FCF", "Discounted FCF", "Working Capital Change"]), use_container_width=True, hide_index=True)

    with tab_financing:
        st.markdown("#### Financing Schedule")
        if financing_df.empty:
            st.info("Financing data will appear after running the model.")
        else:
            fin_tabs = st.tabs([
                "Current Financing",
                "Add Financing",
                "Edit Financing",
            ])

            with fin_tabs[0]:
                st.dataframe(
                    _format_statement(
                        financing_df,
                        ["Interest", "Loan Repayment", "Long Term Debt", "Cash Flow from Financing"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                financing_metadata_store = st.session_state.get('financing_metadata', {})
                if financing_metadata_store:
                    meta_rows = []
                    for key, meta in financing_metadata_store.items():
                        try:
                            component, year_str = key.split(":")
                            year_val = int(year_str)
                        except ValueError:
                            component = key
                            year_val = None
                        meta_rows.append(
                            {
                                "Year": year_val,
                                "Component": component,
                                "Debt Name": meta.get("debt_name", ""),
                                "Debt Amount": meta.get("debt_amount", 0.0),
                                "Interest Rate (%)": meta.get("interest_rate", 0.0),
                                "Start Date": meta.get("start_date"),
                                "Period of Validity (years)": meta.get("validity_period"),
                                "Grace Period (years)": meta.get("grace_period"),
                            }
                        )

                    meta_df = pd.DataFrame(meta_rows)
                    st.markdown("##### Debt Instrument Details")
                    st.dataframe(
                        _format_statement(
                            meta_df,
                            ["Debt Amount"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            with fin_tabs[1]:
                st.markdown("## Add Financing Entry")
                with st.form("add_financing_entry_form"):
                    year_choice = st.selectbox("Year", years, key="add_fin_year")
                    component_choice = st.selectbox(
                        "Component",
                        list(FINANCING_COMPONENT_MAP.keys()),
                        key="add_fin_component",
                    )
                    default_val = financing_df.loc[
                        financing_df["Year"] == year_choice, component_choice
                    ]
                    default_amount = float(default_val.iloc[0]) if not default_val.empty else 0.0
                    metadata_defaults = _get_financing_metadata(int(year_choice), component_choice)
                    amount_value = st.number_input(
                        "Amount ($)",
                        min_value=-100_000_000.0,
                        value=default_amount,
                        step=10_000.0,
                        key="add_fin_amount",
                    )
                    debt_name = st.text_input(
                        "Debt Name",
                        value=metadata_defaults.get("debt_name", ""),
                        key="add_fin_debt_name",
                    )
                    debt_amount = st.number_input(
                        "Debt Amount ($)",
                        min_value=0.0,
                        value=float(metadata_defaults.get("debt_amount", amount_value if amount_value > 0 else 0.0)),
                        step=10_000.0,
                        key="add_fin_debt_amount",
                    )
                    interest_rate = st.number_input(
                        "Interest Rate (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(metadata_defaults.get("interest_rate", 8.0)),
                        step=0.1,
                        key="add_fin_interest_rate",
                    )
                    start_date = st.number_input(
                        "Start Date",
                        min_value=int(years[0]),
                        max_value=int(years[-1]),
                        value=int(metadata_defaults.get("start_date", year_choice)),
                        step=1,
                        key="add_fin_start_date",
                    )
                    validity_period = st.number_input(
                        "Period of Validity (years)",
                        min_value=0,
                        value=int(metadata_defaults.get("validity_period", 5)),
                        step=1,
                        key="add_fin_validity",
                    )
                    grace_period = st.number_input(
                        "Grace Period (years)",
                        min_value=0,
                        value=int(metadata_defaults.get("grace_period", 0)),
                        step=1,
                        key="add_fin_grace",
                    )
                    add_fin_submitted = st.form_submit_button("Add Financing Entry")

                if add_fin_submitted:
                    metadata_payload = {
                        "debt_name": debt_name.strip(),
                        "debt_amount": float(debt_amount),
                        "interest_rate": float(interest_rate),
                        "start_date": int(start_date),
                        "validity_period": int(validity_period),
                        "grace_period": int(grace_period),
                    }
                    _update_financing_entry(
                        int(year_choice), component_choice, float(amount_value), metadata_payload
                    )
                    st.success("Financing entry saved.")
                    st.rerun()

            with fin_tabs[2]:
                st.markdown("## Edit Financing Entry")
                entries = []
                for _, row in financing_df.iterrows():
                    year_val = int(row["Year"])
                    for component in FINANCING_COMPONENT_MAP.keys():
                        entries.append(
                            {
                                "label": f"{year_val} • {component}",
                                "year": year_val,
                                "component": component,
                                "amount": float(row.get(component, 0.0)),
                            }
                        )

                if not entries:
                    st.info("No financing entries available to edit.")
                else:
                    selection = st.selectbox(
                        "Select financing line", options=list(range(len(entries))),
                        format_func=lambda idx: f"{entries[idx]['label']} (${entries[idx]['amount']:,.0f})",
                        key="edit_fin_selector",
                    )
                    entry = entries[selection]
                    edit_key = f"edit_fin_amount_{entry['year']}_{entry['component']}"
                    new_amount = st.number_input(
                        "Amount ($)",
                        value=float(entry["amount"]),
                        step=10_000.0,
                        key=edit_key,
                    )
                    entry_metadata = _get_financing_metadata(int(entry['year']), entry['component'])
                    debt_name_edit = st.text_input(
                        "Debt Name",
                        value=entry_metadata.get("debt_name", ""),
                        key=f"edit_fin_debt_name_{entry['year']}_{entry['component']}",
                    )
                    debt_amount_edit = st.number_input(
                        "Debt Amount ($)",
                        min_value=0.0,
                        value=float(entry_metadata.get("debt_amount", entry['amount'] if entry['amount'] > 0 else 0.0)),
                        step=10_000.0,
                        key=f"edit_fin_debt_amt_{entry['year']}_{entry['component']}",
                    )
                    interest_rate_edit = st.number_input(
                        "Interest Rate (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(entry_metadata.get("interest_rate", 8.0)),
                        step=0.1,
                        key=f"edit_fin_int_rate_{entry['year']}_{entry['component']}",
                    )
                    start_date_edit = st.number_input(
                        "Start Date",
                        min_value=int(years[0]),
                        max_value=int(years[-1]),
                        value=int(entry_metadata.get("start_date", entry['year'])),
                        step=1,
                        key=f"edit_fin_start_{entry['year']}_{entry['component']}",
                    )
                    validity_edit = st.number_input(
                        "Period of Validity (years)",
                        min_value=0,
                        value=int(entry_metadata.get("validity_period", 5)),
                        step=1,
                        key=f"edit_fin_validity_{entry['year']}_{entry['component']}",
                    )
                    grace_edit = st.number_input(
                        "Grace Period (years)",
                        min_value=0,
                        value=int(entry_metadata.get("grace_period", 0)),
                        step=1,
                        key=f"edit_fin_grace_{entry['year']}_{entry['component']}",
                    )

                    col_save, col_remove = st.columns(2)
                    with col_save:
                        if st.button("Save Financing Entry", key=f"save_fin_{entry['year']}_{entry['component']}"):
                            metadata_payload = {
                                "debt_name": debt_name_edit.strip(),
                                "debt_amount": float(debt_amount_edit),
                                "interest_rate": float(interest_rate_edit),
                                "start_date": int(start_date_edit),
                                "validity_period": int(validity_edit),
                                "grace_period": int(grace_edit),
                            }
                            _update_financing_entry(
                                int(entry['year']), entry['component'], float(new_amount), metadata_payload
                            )
                            st.success("Financing entry updated.")
                            st.rerun()
                    with col_remove:
                        if st.button("Remove Financing Entry", key=f"remove_fin_{entry['year']}_{entry['component']}"):
                            _remove_financing_entry(int(entry['year']), entry['component'])
                            st.success("Financing entry removed.")
                            st.rerun()

    with tab_fixed_cost:
        _editable_schedule(
            "Fixed Cost Schedule",
            fixed_cost_df,
            "fixed_cost_overrides",
            "Fixed Operating Costs",
            "Use the + button to add years or delete rows; edits override computed operating expenses.",
        )

    with tab_variable_cost:
        _editable_schedule(
            "Variable Cost Schedule",
            variable_cost_df,
            "variable_cost_overrides",
            "COGS (Variable)",
            "Edit COGS by year. Added rows expand the projection horizon automatically.",
        )

    with tab_other_cost:
        _editable_schedule(
            "Other Cost Schedule",
            other_cost_df,
            "other_cost_overrides",
            "Other Operating Cost",
            "Adjust other operating costs by year; use add/remove to reshape the horizon.",
        )

    with tab_debt:
        _editable_schedule(
            "Debt Schedule",
            debt_schedule_df,
            "loan_repayment_overrides",
            "Principal",
            "Edit principal repayments; add/remove years to reshape the debt amortization profile.",
        )

    with tab_investment:
        st.markdown("#### Investment Schedule")
        owner_default = float(st.session_state.owner_equity_pct)
        owner_pct = st.slider(
            "Baseline Owner Equity %",
            0.0,
            100.0,
            owner_default,
            key="owner_equity_pct_slider",
        )
        st.session_state.owner_equity_pct = owner_pct
        st.info(
            "Adjust owner percentages per year below; rows without overrides use the baseline value."
        )

        investment_years = _investment_years()
        model_start_year = years[0] if years else (investment_years[0] if investment_years else st.session_state.production_start_year)
        owner_values = [_get_owner_pct_for_year(y) for y in investment_years]
        investor_values = [100.0 - pct for pct in owner_values]
        default_amount_map: Dict[int, Dict[str, float]] = {}
        amount_overrides = st.session_state.get('investment_amounts', {})
        owner_equity_values: List[float] = []
        investor_equity_values: List[float] = []
        debt_values: List[float] = []

        for year, owner_pct, investor_pct in zip(investment_years, owner_values, investor_values):
            base_owner_amt = (
                model["config"].equity_investment * (owner_pct / 100.0)
                if year == model_start_year
                else 0.0
            )
            base_investor_amt = (
                model["config"].equity_investment * (investor_pct / 100.0)
                if year == model_start_year
                else 0.0
            )
            base_debt_amt = model["config"].loan_amount if year == model_start_year else 0.0

            default_amount_map[year] = {
                "owner_equity": base_owner_amt,
                "investor_equity": base_investor_amt,
                "debt_raised": base_debt_amt,
            }

            override = amount_overrides.get(year)
            owner_equity_values.append(float(override.get('owner_equity', base_owner_amt)) if override else base_owner_amt)
            investor_equity_values.append(float(override.get('investor_equity', base_investor_amt)) if override else base_investor_amt)
            debt_values.append(float(override.get('debt_raised', base_debt_amt)) if override else base_debt_amt)

        st.session_state['investment_default_amounts'] = default_amount_map

        investment_df = pd.DataFrame(
            {
                "Year": investment_years,
                "Owner %": owner_values,
                "Investor %": investor_values,
                "Owner Equity": owner_equity_values,
                "Investor Equity": investor_equity_values,
                "Debt Raised": debt_values,
            }
        )

        invest_tabs = st.tabs(
            [
                "Current Schedule",
                "Add Entry",
                "Edit Entry",
                "Remove Entry",
                "Yearly Increment",
            ]
        )

        with invest_tabs[0]:
            st.dataframe(
                _format_statement(investment_df, ["Owner Equity", "Investor Equity", "Debt Raised"]),
                use_container_width=True,
                hide_index=True,
            )

        with invest_tabs[1]:
            with st.form("investment_add_form"):
                year_value = st.number_input(
                    "Investment Year",
                    min_value=int(st.session_state.production_start_year),
                    max_value=int(st.session_state.production_end_year),
                    value=int(investment_years[-1]) if investment_years else int(st.session_state.production_start_year),
                    step=1,
                )
                owner_value = st.slider(
                    "Owner Equity %",
                    0,
                    100,
                    value=int(_get_owner_pct_for_year(int(year_value))),
                    key="investment_add_owner_pct",
                )
                default_amounts = _get_investment_amount_record(int(year_value))
                owner_amount = st.number_input(
                    "Owner Equity Amount ($)",
                    min_value=0.0,
                    value=float(default_amounts.get("owner_equity", 0.0)),
                    step=50000.0,
                    key="investment_add_owner_amount",
                )
                investor_amount = st.number_input(
                    "Investor Equity Amount ($)",
                    min_value=0.0,
                    value=float(default_amounts.get("investor_equity", 0.0)),
                    step=50000.0,
                    key="investment_add_investor_amount",
                )
                debt_amount = st.number_input(
                    "Debt Raised ($)",
                    min_value=0.0,
                    value=float(default_amounts.get("debt_raised", 0.0)),
                    step=50000.0,
                    key="investment_add_debt_amount",
                )
                add_submitted = st.form_submit_button("Add Investment Entry")
            if add_submitted:
                _set_owner_pct_for_year(int(year_value), float(owner_value))
                _set_investment_amount_record(int(year_value), owner_amount, investor_amount, debt_amount)
                st.success("Investment entry added. Rerunning the model with updated ownership splits.")
                st.rerun()

        with invest_tabs[2]:
            if not investment_years:
                st.info("No investment years available to edit.")
            else:
                edit_year = st.selectbox(
                    "Select Year to Edit",
                    investment_years,
                    key="investment_edit_year",
                )
                current_pct = _get_owner_pct_for_year(int(edit_year))
                current_amounts = _get_investment_amount_record(int(edit_year))
                with st.form("investment_edit_form"):
                    updated_pct = st.slider(
                        "Owner Equity %",
                        0,
                        100,
                        value=int(current_pct),
                        key="investment_edit_owner_pct",
                    )
                    owner_amount = st.number_input(
                        "Owner Equity Amount ($)",
                        min_value=0.0,
                        value=float(current_amounts.get("owner_equity", 0.0)),
                        step=50000.0,
                        key=f"investment_edit_owner_amount_{edit_year}",
                    )
                    investor_amount = st.number_input(
                        "Investor Equity Amount ($)",
                        min_value=0.0,
                        value=float(current_amounts.get("investor_equity", 0.0)),
                        step=50000.0,
                        key=f"investment_edit_investor_amount_{edit_year}",
                    )
                    debt_amount = st.number_input(
                        "Debt Raised ($)",
                        min_value=0.0,
                        value=float(current_amounts.get("debt_raised", 0.0)),
                        step=50000.0,
                        key=f"investment_edit_debt_amount_{edit_year}",
                    )
                    edit_submitted = st.form_submit_button("Save Investment Changes")
                if edit_submitted:
                    _set_owner_pct_for_year(int(edit_year), float(updated_pct))
                    _set_investment_amount_record(int(edit_year), owner_amount, investor_amount, debt_amount)
                    st.success("Investment entry updated. Rerunning the model to reflect the change.")
                    st.rerun()

        with invest_tabs[3]:
            if not investment_years:
                st.info("No investment years available to remove.")
            else:
                remove_year = st.selectbox(
                    "Select Year to Remove",
                    investment_years,
                    key="investment_remove_year",
                )
                if st.button("Remove Investment Entry", key="investment_remove_btn"):
                    _remove_investment_entry(int(remove_year))
                    _remove_investment_amount_record(int(remove_year))
                    st.success("Investment entry removed. Default ownership percentages now apply.")
                    st.rerun()

        with invest_tabs[4]:
            if not investment_years:
                st.info("No investment years available for increments.")
            else:
                increment_years = st.multiselect(
                    "Select Years to Increment",
                    investment_years,
                    default=investment_years,
                    key="investment_increment_years",
                )
                increment_value = st.number_input(
                    "Increment (percentage points)",
                    min_value=-100.0,
                    max_value=100.0,
                    value=1.0,
                    step=0.5,
                )
                owner_amount_increment = st.number_input(
                    "Owner Equity Amount Increment ($)",
                    value=0.0,
                    step=10000.0,
                    key="investment_increment_owner_amount",
                )
                investor_amount_increment = st.number_input(
                    "Investor Equity Amount Increment ($)",
                    value=0.0,
                    step=10000.0,
                    key="investment_increment_investor_amount",
                )
                debt_amount_increment = st.number_input(
                    "Debt Raised Increment ($)",
                    value=0.0,
                    step=10000.0,
                    key="investment_increment_debt_amount",
                )
                if st.button("Apply Increment", key="investment_increment_btn"):
                    _increment_investment_entries(
                        [int(y) for y in increment_years],
                        float(increment_value),
                        float(owner_amount_increment),
                        float(investor_amount_increment),
                        float(debt_amount_increment),
                    )
                    st.success("Investment percentages updated. Rerunning the model with the new values.")
                    st.rerun()

    with tab_assets:
        _editable_schedule(
            "Asset Schedule",
            assets_df,
            "asset_overrides",
            "Total Assets",
            "Update asset balances or add projection years to tailor the asset view.",
        )

    with tab_auto:
        st.markdown("#### Automobile Manufacturing Planner")
        planner = [_normalize_auto_plan(p) for p in st.session_state.auto_planner]
        st.session_state.auto_planner = planner

        names = [p["name"] for p in planner]
        selected_idx = st.selectbox(
            "Select automobile to configure",
            options=list(range(len(planner))),
            format_func=lambda i: names[i],
            key="auto_plan_selector",
        )
        selected_plan = planner[selected_idx]

        with st.form(f"auto_plan_form_{selected_idx}"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                name = st.text_input("Automobile name", value=selected_plan["name"], key=f"auto_name_{selected_idx}")
                defect_rate = st.number_input(
                    "Defect rate (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(selected_plan["defect_rate"]),
                    key=f"defect_rate_{selected_idx}",
                )
            with col_b:
                scrap_allowance = st.number_input(
                    "Scrap allowance (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(selected_plan["scrap_allowance"]),
                    key=f"scrap_allowance_{selected_idx}",
                )
            with col_c:
                target_units = st.number_input(
                    "Target units",
                    min_value=0,
                    value=int(selected_plan.get("target_units", 0)),
                    step=50,
                    key=f"target_units_{selected_idx}",
                )

            st.markdown("##### Stage cycle time (minutes per unit)")
            cycle_cols = st.columns(4)
            chassis_cycle = cycle_cols[0].number_input(
                "Chassis",
                min_value=0,
                value=int(selected_plan["cycle_time"]["chassis"]),
                key=f"cycle_chassis_{selected_idx}",
            )
            paint_cycle = cycle_cols[1].number_input(
                "Paint",
                min_value=0,
                value=int(selected_plan["cycle_time"]["paint"]),
                key=f"cycle_paint_{selected_idx}",
            )
            assembly_cycle = cycle_cols[2].number_input(
                "Assembly",
                min_value=0,
                value=int(selected_plan["cycle_time"]["assembly"]),
                key=f"cycle_assembly_{selected_idx}",
            )
            quality_cycle = cycle_cols[3].number_input(
                "Quality",
                min_value=0,
                value=int(selected_plan["cycle_time"]["quality"]),
                key=f"cycle_quality_{selected_idx}",
            )

            st.markdown("##### Stations and shifts")
            station_cols = st.columns(4)
            chassis_stations = station_cols[0].number_input(
                "Chassis stations",
                min_value=1,
                value=int(selected_plan["stations"]["chassis"]),
                key=f"stations_chassis_{selected_idx}",
            )
            paint_stations = station_cols[1].number_input(
                "Paint stations",
                min_value=1,
                value=int(selected_plan["stations"]["paint"]),
                key=f"stations_paint_{selected_idx}",
            )
            assembly_stations = station_cols[2].number_input(
                "Assembly stations",
                min_value=1,
                value=int(selected_plan["stations"]["assembly"]),
                key=f"stations_assembly_{selected_idx}",
            )
            quality_stations = station_cols[3].number_input(
                "Quality stations",
                min_value=1,
                value=int(selected_plan["stations"]["quality"]),
                key=f"stations_quality_{selected_idx}",
            )

            shift_col1, shift_col2 = st.columns(2)
            hours_per_shift = shift_col1.number_input(
                "Hours per shift",
                min_value=1,
                max_value=24,
                value=int(selected_plan.get("hours_per_shift", 8)),
                key=f"hours_per_shift_{selected_idx}",
            )
            shifts_per_day = shift_col2.number_input(
                "Shifts per day",
                min_value=1,
                max_value=4,
                value=int(selected_plan.get("shifts_per_day", 2)),
                key=f"shifts_per_day_{selected_idx}",
            )

            submitted = st.form_submit_button("Calculate capacity")

        if submitted:
            planner[selected_idx] = {
                "name": name.strip() or selected_plan["name"],
                "defect_rate": defect_rate,
                "scrap_allowance": scrap_allowance,
                "target_units": target_units,
                "cycle_time": {
                    "chassis": chassis_cycle,
                    "paint": paint_cycle,
                    "assembly": assembly_cycle,
                    "quality": quality_cycle,
                },
                "stations": {
                    "chassis": chassis_stations,
                    "paint": paint_stations,
                    "assembly": assembly_stations,
                    "quality": quality_stations,
                },
                "hours_per_shift": hours_per_shift,
                "shifts_per_day": shifts_per_day,
            }
            st.session_state.auto_planner = planner
            selected_plan = planner[selected_idx]
            st.success("Planner updated. Capacity recalculated.")

        stage_capacity, bottleneck, annual_capacity = _calculate_planner_capacity(selected_plan, cfg.working_days)

        cap_cols = st.columns(3)
        cap_cols[0].metric("Annual capacity", f"{annual_capacity:,.0f} units", help="Bottleneck-adjusted capacity")
        cap_cols[1].metric("Target units", f"{selected_plan.get('target_units', 0):,.0f}")
        cap_cols[2].metric("Bottleneck stage", bottleneck.title())

        stage_rows = []
        for stage, daily_cap in stage_capacity.items():
            stage_rows.append(
                {
                    "Stage": stage.title(),
                    "Cycle Time (min/unit)": selected_plan["cycle_time"].get(stage, 0),
                    "Stations": selected_plan["stations"].get(stage, 0),
                    "Daily Capacity": daily_cap,
                    "Annual Capacity": daily_cap * cfg.working_days,
                }
            )
        stage_df = pd.DataFrame(stage_rows)
        st.dataframe(stage_df, use_container_width=True, hide_index=True)

        summary_rows = []
        for plan in planner:
            capacities, neck, annual_cap = _calculate_planner_capacity(plan, cfg.working_days)
            summary_rows.append(
                {
                    "Automobile": plan["name"],
                    "Target Units": plan.get("target_units", 0),
                    "Annual Capacity": annual_cap,
                    "Bottleneck": neck.title(),
                    "Utilization vs Target": f"{(annual_cap / plan.get('target_units', 1) * 100):.1f}%" if plan.get("target_units", 0) > 0 else "N/A",
                }
            )

        st.markdown("##### Planner overview")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("### Labor Management")
    _render_labor_management_section()

    st.markdown("### CAPEX Management")
    _render_capex_management_section()

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

with tab_dashboard:
    st.markdown("# Executive Dashboard")
    
    # Run financial model
    cfg = _build_company_config()
    model = run_financial_model(cfg)
    st.session_state.financial_model = model
    years = list(model["years"])
    start_year = years[0]
    
    # Key Metrics
    st.markdown("## Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue_first = model['revenue'][start_year]
        st.metric(f"{start_year} Revenue", f"${revenue_first/1e6:.1f}M", delta="Year 1")

    with col2:
        profit_first = model['net_profit'][start_year]
        margin = (profit_first / revenue_first * 100) if revenue_first > 0 else 0
        st.metric(f"{start_year} Net Profit", f"${profit_first/1e6:.1f}M", delta=f"{margin:.1f}%")

    with col3:
        headcount = st.session_state.labor_manager.get_total_headcount(start_year)
        st.metric("Total Headcount", f"{headcount} employees", delta="Current")

    with col4:
        ev = model['enterprise_value']
        st.metric("Enterprise Value", f"${ev/1e6:.1f}M", delta="DCF")
    
    # Financial Overview
    st.markdown("## 5-Year Financial Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_df = pd.DataFrame({
            'Year': years,
            'Revenue ($M)': [model['revenue'][y]/1e6 for y in years],
            'EBIT ($M)': [model['ebit'][y]/1e6 for y in years],
            'Net Profit ($M)': [model['net_profit'][y]/1e6 for y in years],
            'FCF ($M)': [model['fcf'][y]/1e6 for y in years]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig = px.line(
            forecast_df,
            x='Year',
            y=['Revenue ($M)', 'Net Profit ($M)'],
            markers=True,
            title="Revenue & Profit Forecast",
            labels={'value': 'Amount ($M)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Labor & CAPEX Overview
    st.markdown("## Workforce & Assets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        labor_costs = st.session_state.labor_manager.get_labor_cost_by_type(start_year, st.session_state.salary_growth_rate)
        st.markdown(f"### Labor Costs ({start_year})")
        st.metric("Direct Labor", f"${labor_costs['Direct']/1e6:.2f}M")
        st.metric("Indirect Labor", f"${labor_costs['Indirect']/1e6:.2f}M")
        st.metric("Total Labor", f"${(labor_costs['Direct'] + labor_costs['Indirect'])/1e6:.2f}M")

    with col2:
        hc_types = st.session_state.labor_manager.get_headcount_by_type(start_year)
        st.markdown(f"### Headcount ({start_year})")
        st.metric("Direct Labor HC", f"{hc_types['Direct']} employees")
        st.metric("Indirect Labor HC", f"{hc_types['Indirect']} employees")
        st.metric("Total HC", f"{hc_types['Direct'] + hc_types['Indirect']} employees")
    
    with col3:
        capex_items = st.session_state.capex_manager.list_items()
        total_capex = st.session_state.capex_manager.total_capex()
        st.markdown(f"### Capital Assets ({start_year})")
        st.metric("Total CAPEX", f"${total_capex/1e6:.2f}M")
        st.metric("# Assets", f"{len(capex_items)}")
        dep_first = model['depreciation'].get(start_year, model['depreciation']) if isinstance(model['depreciation'], dict) else model['depreciation']
        st.metric("Annual Depreciation", f"${dep_first/1e3:.0f}K")

    st.markdown("## 5-Year Labor Cost Schedule")

    cost_schedule = LaborCostSchedule(st.session_state.labor_manager)
    schedule_df = cost_schedule.generate_5year_schedule(
        salary_growth=st.session_state.salary_growth_rate
    )

    display_df = schedule_df.copy()
    for col in ['Direct Labor Cost', 'Indirect Labor Cost', 'Total Labor Cost']:
        display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:.2f}M")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    labor_fig = px.line(
        schedule_df,
        x='Year',
        y=['Direct Labor Cost', 'Indirect Labor Cost'],
        markers=True,
        title="Labor Cost Forecast",
        labels={'value': 'Cost ($)', 'variable': 'Labor Type'},
    )
    st.plotly_chart(labor_fig, use_container_width=True)

    st.markdown("## Financial Visualization & Reports")
    visualizer = FinancialVisualizer(model)
    viz_tabs = st.tabs([
        "Executive Summary",
        "Financial Statements",
        "Ratio Analysis",
        "Scenario Report",
        "Sensitivity",
        "Trends",
    ])

    with viz_tabs[0]:
        tables = visualizer.executive_summary_tables()
        for name, df in tables.items():
            st.subheader(name)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with viz_tabs[1]:
        income_df, cashflow_df, balance_df = visualizer.financial_statement_tables()
        st.subheader("Income Statement")
        st.dataframe(income_df, use_container_width=True, hide_index=True)
        st.subheader("Cash Flow Statement")
        st.dataframe(cashflow_df, use_container_width=True, hide_index=True)
        st.subheader("Balance Sheet")
        st.dataframe(balance_df, use_container_width=True, hide_index=True)

    with viz_tabs[2]:
        ratios_df, summary_df = visualizer.ratio_tables()
        st.subheader("Ratios by Year")
        st.dataframe(ratios_df, use_container_width=True, hide_index=True)
        st.subheader("Ratio Highlights")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with viz_tabs[3]:
        scenarios_df, insights_df = visualizer.scenario_tables()
        st.subheader("Scenario Comparison")
        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
        if not insights_df.empty:
            st.subheader("Key Deltas")
            st.dataframe(insights_df, use_container_width=True, hide_index=True)

    with viz_tabs[4]:
        sensitivity_df = visualizer.sensitivity_table()
        st.subheader("Sensitivity Analysis")
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

    with viz_tabs[5]:
        trend_df = visualizer.chart_tables()
        st.subheader("Revenue, Profit, Cash & Margin")
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

# =====================================================
# PAGE 2: AI & MACHINE LEARNING (RAG)
# =====================================================

with tab_ai:
    st.markdown("# AI & Machine Learning")
    st.info(
        "Store your model provider settings and run the standalone `rag_app.py` service "
        "to generate feasibility studies grounded in your financial snapshots and uploaded documents."
    )

    settings = st.session_state.ai_settings

    with st.form("ai_settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            provider = st.text_input("Provider", value=settings.get("provider", "OpenAI"))
            model = st.text_input("Model", value=settings.get("model", "gpt-4o"))
        with col2:
            api_key = st.text_input(
                "API Key",
                value=settings.get("api_key", ""),
                type="password",
                help="Used by the RAG service to call the selected provider.",
            )
            enabled = st.checkbox("Enable RAG Feasibility", value=settings.get("enabled", True))

        saved = st.form_submit_button("Save AI Configuration")

    if saved:
        st.session_state.ai_settings = {
            "provider": provider.strip() or "OpenAI",
            "model": model.strip() or "gpt-4o",
            "api_key": api_key.strip(),
            "enabled": enabled,
        }
        st.success("AI configuration saved for this session.")

    st.markdown("## RAG Feasibility Study Generator")
    st.write(
        "Use the `rag_app.py` service in this repository to ingest documents, capture the "
        "financial snapshot from Excel, and compose grounded feasibility study sections."
    )

    with st.expander("How to run the generator", expanded=True):
        st.markdown(
            "1. Install dependencies: `pip install -r requirements.txt faiss-cpu sentence-transformers fastapi uvicorn pypdf python-docx python-pptx`\n"
            "2. Export your API key: `export OPENAI_API_KEY=...` (or enter it above).\n"
            "3. Start the service: `python rag_app.py`.\n"
            "4. POST to `/collect` with your financial snapshot, `/ingest` with project files, then `/generate` to build the report." 
        )
        st.code("uvicorn rag_app:app --host 0.0.0.0 --port 8000", language="bash")

    st.markdown("### Current Session Status")
    st.write(
        {
            "provider": st.session_state.ai_settings.get("provider"),
            "model": st.session_state.ai_settings.get("model"),
            "api_key_set": bool(st.session_state.ai_settings.get("api_key")),
            "enabled": st.session_state.ai_settings.get("enabled", True),
        }
    )

# =====================================================
# PAGE 5: REPORTS
# =====================================================

with tab_reports:
    st.markdown("# Financial Reports & Exports")
    
    if st.session_state.financial_model:
        model = st.session_state.financial_model
        years = list(model["years"])

        # Summary Report
        st.markdown("## Executive Summary")

        summary_text = f"""
        **{years[0]} Financials:**
        - Revenue: ${model['revenue'][years[0]]/1e6:.1f}M
        - EBIT: ${model['ebit'][years[0]]/1e6:.1f}M
        - Net Profit: ${model['net_profit'][years[0]]/1e6:.1f}M
        - FCF: ${model['fcf'][years[0]]/1e6:.1f}M

        **Valuation:**
        - Enterprise Value: ${model['enterprise_value']/1e6:.1f}M
        - 5-Year FCF: ${sum(model['fcf'].values())/1e6:.1f}M

        **Workforce:**
        - Headcount: {st.session_state.labor_manager.get_total_headcount(years[0])} employees
        - Labor Cost: {(st.session_state.labor_manager.get_labor_cost_by_type(years[0], st.session_state.salary_growth_rate)['Direct'] + st.session_state.labor_manager.get_labor_cost_by_type(years[0], st.session_state.salary_growth_rate)['Indirect'])/1e6:.2f}M
        """

        st.markdown(summary_text)

        statements_tab, schedules_tab, downloads_tab = st.tabs([
            "Financial Statements",
            "Schedules & Drivers",
            "Exports",
        ])

        with statements_tab:
            st.markdown("### Income Statement")
            income_df, cashflow_df, balance_df = generate_financial_statements(model)
            st.dataframe(
                _format_statement(income_df, ["Revenue", "COGS", "Opex", "EBITDA", "EBIT", "Tax", "Net Profit"]),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Cash Flow Statement")
            st.dataframe(
                _format_statement(cashflow_df, ["CFO", "CFI", "CFF", "Net Cash Flow", "Closing Cash"]),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Balance Sheet")
            st.dataframe(
                _format_statement(
                    balance_df,
                    [
                        "Fixed Assets",
                        "Current Assets",
                        "Total Assets",
                        "Current Liabilities",
                        "Long Term Debt",
                        "Total Equity",
                        "Total Liabilities + Equity",
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )

        with schedules_tab:
            st.markdown("### Labor Cost Schedule")
            labor_df = generate_labor_statement(model)
            if labor_df.empty:
                st.info("Labor schedule not available. Add positions to view projected costs.")
            else:
                st.dataframe(labor_df, hide_index=True, use_container_width=True)

            st.markdown("### CAPEX Spend Schedule")
            if st.session_state.capex_manager:
                capex_schedule = st.session_state.capex_manager.yearly_capex_schedule(
                    years[0],
                    len(years),
                )
                capex_df = pd.DataFrame({"Year": years, "CAPEX Spend": [capex_schedule.get(y, 0.0) for y in years]})
                st.dataframe(_format_statement(capex_df, ["CAPEX Spend"]), hide_index=True, use_container_width=True)
            else:
                st.info("No CAPEX manager configured.")

            st.markdown("### Debt Schedule")
            interest = model.get('interest_payment', {})
            principal = model.get('loan_repayment', {})
            outstanding = []
            balance = st.session_state.financial_model['config'].loan_amount
            for y in years:
                balance = max(0.0, balance - principal.get(y, 0.0))
                outstanding.append(balance)
            debt_df = pd.DataFrame(
                {
                    "Year": years,
                    "Interest": [interest.get(y, 0.0) for y in years],
                    "Principal": [principal.get(y, 0.0) for y in years],
                    "Ending Balance": outstanding,
                }
            )
            st.dataframe(
                _format_statement(debt_df, ["Interest", "Principal", "Ending Balance"]),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Working Capital Changes")
            wc_change = model.get('working_capital_change', {})
            wc_df = pd.DataFrame({"Year": years, "Change in Working Capital": [wc_change.get(y, 0.0) for y in years]})
            st.dataframe(_format_statement(wc_df, ["Change in Working Capital"]), hide_index=True, use_container_width=True)

        with downloads_tab:
            st.markdown("### Download Reports")

            export_df = pd.DataFrame({
                'Year': years,
                'Revenue': [model['revenue'][y] for y in years],
                'COGS': [model['cogs'][y] for y in years],
                'OPEX': [model['opex'][y] for y in years],
                'EBIT': [model['ebit'][y] for y in years],
                'Net Profit': [model['net_profit'][y] for y in years],
                'FCF': [model['fcf'][y] for y in years]
            })

            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download Financial Forecast (CSV)",
                csv,
                "financial_forecast.csv",
                "text/csv"
            )

            cost_schedule = LaborCostSchedule(st.session_state.labor_manager)
            labor_schedule = cost_schedule.generate_5year_schedule(salary_growth=st.session_state.salary_growth_rate)

            csv = labor_schedule.to_csv(index=False)
            st.download_button(
                "Download Labor Schedule (CSV)",
                csv,
                "labor_schedule.csv",
                "text/csv"
            )
    else:
        st.info("Run the financial model first to generate reports")

# =====================================================
# PAGE 6: ADVANCED ANALYTICS
# =====================================================

with tab_advanced:
    st.markdown("# Advanced Analytics")

    cfg = _build_company_config()
    model = run_financial_model(cfg)
    years = list(model["years"])
    start_year, final_year = years[0], years[-1]

    st.markdown("## Baseline Snapshot")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.metric("Enterprise Value", _format_currency(model.get("enterprise_value", 0)))
    with kpi_cols[1]:
        st.metric(
            f"Revenue ({start_year})",
            _format_currency(model.get("revenue", {}).get(start_year, 0)),
        )
    with kpi_cols[2]:
        st.metric(
            f"Net Profit ({final_year})",
            _format_currency(model.get("net_profit", {}).get(final_year, 0)),
        )

    # Shared analytics used across the advanced feature tabs
    sens_params = ["annual_capacity", "cogs_ratio", "loan_interest_rate", "loan_amount", "wacc"]
    sens_ranges = {p: 0.2 for p in sens_params}
    sensitivity = AdvancedSensitivityAnalyzer(model, cfg)
    sens_df = sensitivity.pareto_sensitivity(sens_params, sens_ranges)

    stress_engine = StressTestEngine(model, cfg)
    stress_results = stress_engine.extreme_scenarios()
    stress_df = (
        pd.DataFrame.from_dict(stress_results, orient="index")
        .reset_index()
        .rename(columns={"index": "Scenario"})
    )

    mc_engine = MonteCarloSimulator(model, cfg, num_simulations=400)
    mc_stats = mc_engine.run_simulation(
        {
            "cogs_ratio": ("normal", cfg.cogs_ratio, 0.05),
            "wacc": ("normal", cfg.wacc, 0.02),
            "annual_capacity": ("normal", cfg.annual_capacity, cfg.annual_capacity * 0.1),
        }
    )

    summary_rows = []
    labels = {
        "enterprise_values": "Enterprise Value",
        "final_profits": f"Net Profit ({final_year})",
        "final_cash": f"Closing Cash ({final_year})",
        "roi": "ROI (%)",
    }
    for key, label in labels.items():
        stats = mc_stats.get(key, {})
        summary_rows.append(
            {
                "Metric": label,
                "Mean": stats.get("mean", 0.0),
                "P5": stats.get("p5", 0.0),
                "P95": stats.get("p95", 0.0),
            }
        )
    mc_summary_df = pd.DataFrame(summary_rows)

    segment_analyzer = SegmentationAnalyzer(model)
    segment_data = {}
    for product, share in model.get("product_mix", {}).items():
        base_revenue = model.get("revenue", {}).get(start_year, 0) * share
        base_units = model.get("production_volume", {}).get(start_year, 0) * share
        segment_data[product] = {
            "revenue": base_revenue,
            "cost": base_revenue * cfg.cogs_ratio,
            "units": max(1, base_units),
        }
    segment_df = segment_analyzer.segment_analysis(segment_data) if segment_data else pd.DataFrame()

    ts_analyzer = TimeSeriesAnalyzer()
    revenue_series = {y: model.get("revenue", {}).get(y, 0.0) for y in years}
    prod_series = {y: model.get("production_volume", {}).get(y, 0.0) for y in years}
    moving_avg = ts_analyzer.moving_average(revenue_series, window=3)
    seasonality = ts_analyzer.detect_seasonality(revenue_series)
    smoothing = ts_analyzer.simple_exponential_smoothing(revenue_series, forecast_periods=3)

    regression = RegressionModeler.simple_linear_regression(
        list(prod_series.values()), list(revenue_series.values())
    ) if prod_series and revenue_series else {}

    what_if = WhatIfAnalyzer(cfg)
    scenario_rows = []
    sample_adjustments = {
        "Higher Utilization": {"annual_capacity": cfg.annual_capacity * 1.1},
        "Lean Manufacturing": {"cogs_ratio": max(0.05, cfg.cogs_ratio * 0.95)},
        "Marketing Push": {"marketing_budget": {start_year: cfg.marketing_budget.get(start_year, 0) * 1.2}},
    }
    for name, adjustment in sample_adjustments.items():
        scenario_rows.append(what_if.create_scenario(name, adjustment))
    what_if_df = pd.DataFrame(scenario_rows)

    goal_optimizer = GoalSeekOptimizer(cfg)
    goal_target = model.get("enterprise_value", 0) * 1.1
    goal_result = goal_optimizer.find_breakeven_parameter("cogs_ratio", "enterprise_value", goal_target)

    risk_analyzer = RiskAnalyzer()
    ev_distribution = mc_stats.get("enterprise_values", {}).get("distribution", [])
    var_summary = risk_analyzer.calculate_var(ev_distribution.tolist(), 0.95) if len(ev_distribution) else {}
    cvar_summary = risk_analyzer.calculate_cvar(ev_distribution.tolist(), 0.95) if len(ev_distribution) else {}

    portfolio_returns = np.random.normal(0.08, 0.15, (250, 4))
    portfolio_opt = PortfolioOptimizer.optimize_portfolio(portfolio_returns)

    options_value = RealOptionsAnalyzer.expansion_option_value(
        model.get("enterprise_value", 0),
        expansion_cost=model.get("enterprise_value", 0) * 0.1,
        upside_scenario_npv=model.get("enterprise_value", 0) * 1.25,
        probability=0.5,
    )

    esg = ESGAnalyzer.carbon_pricing_impact(base_emissions=10_000, carbon_price=40, emission_reduction_rate=0.05, years=5)
    esg_df = pd.DataFrame.from_dict(esg, orient="index").reset_index().rename(columns={"index": "Year"})

    cov = [[1.0, 0.6], [0.6, 1.0]]
    correlated = np.random.multivariate_normal([0, 0], cov, size=200)
    corr_df = pd.DataFrame(correlated, columns=["FX Shock", "Rate Shock"]).corr()

    analytics_tabs = st.tabs([
        "Sensitivity & Tornado",
        "Stress & Scenarios",
        "Trend & Segmentation",
        "Simulation & Risk",
        "What-If & Goal Seek",
        "Optimization & Valuation",
    ])

    with analytics_tabs[0]:
        st.markdown("### Sensitivity analysis, tornado & spider views")
        if sens_df.empty:
            st.info("No sensitivity results available for the current configuration.")
        else:
            display_cols = ["parameter", "base_value", "low_ev", "high_ev", "impact_pct"]
            sens_view = sens_df[display_cols].rename(
                columns={
                    "parameter": "Parameter",
                    "base_value": "Base",
                    "low_ev": "Low EV",
                    "high_ev": "High EV",
                    "impact_pct": "Impact (%)",
                }
            )
            currency_cols = ["Base", "Low EV", "High EV"]
            formatted = sens_view.copy()
            for col in currency_cols:
                formatted[col] = formatted[col].apply(_format_currency)
            formatted["Impact (%)"] = formatted["Impact (%)"].apply(lambda v: f"{v:.1f}%")
            st.dataframe(formatted, hide_index=True, use_container_width=True)

            tornado_rows = []
            base_ev = model.get("enterprise_value", 0)
            for _, row in sens_df.iterrows():
                tornado_rows.append(
                    {
                        "Parameter": row["parameter"],
                        "Downside": _format_currency(row["low_ev"] - base_ev),
                        "Upside": _format_currency(row["high_ev"] - base_ev),
                        "Range": _format_currency(row["high_ev"] - row["low_ev"]),
                    }
                )
            st.dataframe(pd.DataFrame(tornado_rows), hide_index=True, use_container_width=True)

            # Impact bar chart
            impact_chart = px.bar(
                sens_df,
                x="impact_pct",
                y="parameter",
                orientation="h",
                title="EV Impact by Driver",
                labels={"impact_pct": "Impact (%)", "parameter": "Driver"},
            )
            st.plotly_chart(impact_chart, use_container_width=True)

            # Tornado-style range chart
            tornado_chart = px.bar(
                sens_df.assign(range=sens_df["high_ev"] - sens_df["low_ev"]),
                x="range",
                y="parameter",
                orientation="h",
                title="Tornado View (EV Range)",
                labels={"range": "EV Range", "parameter": "Driver"},
            )
            st.plotly_chart(tornado_chart, use_container_width=True)

    with analytics_tabs[1]:
        st.markdown("### Scenario stress testing & spider metrics")
        if stress_df.empty:
            st.info("No stress scenarios available.")
        else:
            currency_cols = ["enterprise_value", "revenue_2030", "net_profit_2030", "final_cash"]
            stress_view = stress_df.copy()
            for col in currency_cols:
                if col in stress_view.columns:
                    stress_view[col] = stress_view[col].apply(_format_currency)
            if "recovery_probability" in stress_view.columns:
                stress_view["recovery_probability"] = stress_view["recovery_probability"].apply(
                    lambda v: f"{v:.0f}%",
                )
            st.dataframe(stress_view, hide_index=True, use_container_width=True)

            ev_cols = [
                col
                for col in ["enterprise_value", "revenue_2030", "net_profit_2030", "final_cash"]
                if col in stress_df.columns
            ]
            if ev_cols:
                stress_chart = px.bar(
                    stress_df,
                    x="Scenario",
                    y=ev_cols,
                    title="Scenario Outcomes",
                    barmode="group",
                    labels={"Scenario": "Scenario", "value": "Amount", "variable": "Metric"},
                )
                st.plotly_chart(stress_chart, use_container_width=True)

        classification_rows = []
        for y in years:
            debt = model.get("long_term_debt", {}).get(y, 0)
            equity = model.get("total_equity", {}).get(y, 1)
            ratio = debt / equity if equity else 0
            band = "Low" if ratio < 0.5 else ("Medium" if ratio < 1.0 else "High")
            classification_rows.append({"Year": y, "Debt/Equity": ratio, "Risk Band": band})
        st.dataframe(pd.DataFrame(classification_rows), hide_index=True, use_container_width=True)

    with analytics_tabs[2]:
        st.markdown("### Trend, seasonality & segmentation")
        ma_df = pd.DataFrame(
            {
                "Year": moving_avg.get("years", years),
                "Revenue": list(revenue_series.values()),
                "3Y Moving Avg": moving_avg.get("moving_average", []),
            }
        )
        st.dataframe(ma_df, hide_index=True, use_container_width=True)
        seasonality_msg = "Seasonality detected" if seasonality.get("seasonal") else "No strong seasonality"
        st.caption(f"Autocorrelation: {seasonality.get('autocorrelation', 0):.2f} — {seasonality_msg}")

        if not ma_df.empty:
            ma_chart = px.line(
                ma_df,
                x="Year",
                y=[col for col in ma_df.columns if col != "Year"],
                title="Revenue Trend & Moving Average",
                labels={"value": "Amount", "variable": "Series"},
            )
            st.plotly_chart(ma_chart, use_container_width=True)

        forecast_df = pd.DataFrame(
            {
                "Future Year": smoothing.get("future_years", []),
                "SES Forecast": smoothing.get("future_forecast", []),
            }
        )
        if not forecast_df.empty:
            st.dataframe(forecast_df, hide_index=True, use_container_width=True)
            forecast_chart = px.line(
                forecast_df,
                x="Future Year",
                y="SES Forecast",
                markers=True,
                title="Simple Exponential Smoothing Forecast",
                labels={"SES Forecast": "Forecast", "Future Year": "Year"},
            )
            st.plotly_chart(forecast_chart, use_container_width=True)
        else:
            st.info("SES forecast not available for the current series.")

        if not segment_df.empty:
            seg_view = segment_df.copy()
            seg_view["revenue"] = seg_view["revenue"].apply(_format_currency)
            seg_view["cost"] = seg_view["cost"].apply(_format_currency)
            st.dataframe(
                seg_view.rename(columns={"revenue": "Revenue", "cost": "Cost"}),
                hide_index=True,
                use_container_width=True,
            )

            seg_chart = px.bar(
                segment_df,
                x="segment",
                y=[col for col in ["revenue", "cost"] if col in segment_df.columns],
                barmode="group",
                title="Segment Revenue and Cost",
                labels={"segment": "Segment", "value": "Amount", "variable": "Metric"},
            )
            st.plotly_chart(seg_chart, use_container_width=True)
        else:
            st.info("Segmentation results will appear once product mix and revenue are available.")

        if regression:
            st.caption(
                f"Regression (volume→revenue): slope={regression.get('slope',0):.2f}, R²={regression.get('r_squared',0):.2f}"
            )
            rev_vals = list(revenue_series.values())
            prod_vals = list(prod_series.values())
            if rev_vals and prod_vals and len(rev_vals) == len(prod_vals):
                reg_chart = px.scatter(
                    x=prod_vals,
                    y=rev_vals,
                    title="Volume vs Revenue Regression",
                    labels={"x": "Production Units", "y": "Revenue"},
                )
                st.plotly_chart(reg_chart, use_container_width=True)
        else:
            st.info("Regression insights will display after production and revenue data are populated.")

    with analytics_tabs[3]:
        st.markdown("### Monte Carlo, VaR/CVaR & probabilistic valuation")
        formatted_summary = mc_summary_df.copy()
        for idx, row in formatted_summary.iterrows():
            if "ROI" in row["Metric"]:
                formatted_summary.loc[idx, ["Mean", "P5", "P95"]] = [
                    f"{row['Mean']:.1f}%",
                    f"{row['P5']:.1f}%",
                    f"{row['P95']:.1f}%",
                ]
            else:
                formatted_summary.loc[idx, ["Mean", "P5", "P95"]] = [
                    _format_currency(row["Mean"]),
                    _format_currency(row["P5"]),
                    _format_currency(row["P95"]),
                ]
        st.dataframe(formatted_summary, hide_index=True, use_container_width=True)

        if var_summary:
            st.caption(
                f"95% VaR: {_format_currency(var_summary.get('var',0))}; 95% CVaR: {_format_currency(cvar_summary.get('cvar',0))}"
            )
        else:
            st.info("Monte Carlo risk metrics will display after simulations produce a distribution.")
        if len(ev_distribution) > 0:
            hist_fig = px.histogram(
                ev_distribution,
                nbins=30,
                title="Enterprise Value Distribution (Monte Carlo)",
                labels={"value": "Enterprise Value"},
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            st.info("Run simulations to view the enterprise value distribution chart.")

        if len(ev_distribution) > 0:
            ecdf = np.sort(ev_distribution)
            ecdf_chart = px.line(
                x=ecdf,
                y=np.linspace(0, 1, len(ecdf)),
                title="Enterprise Value Cumulative Distribution",
                labels={"x": "Enterprise Value", "y": "Cumulative Probability"},
            )
            st.plotly_chart(ecdf_chart, use_container_width=True)

    with analytics_tabs[4]:
        st.markdown("### What-if analysis, goal seek & tornado/spider helpers")
        if not what_if_df.empty:
            impact_cols = ["enterprise_value", "ev_change", "ev_change_pct", "revenue_2030", "profit_2030"]
            view = what_if_df.drop(columns=[c for c in what_if_df.columns if c not in impact_cols + ["scenario_name"]]).rename(columns={"scenario_name": "Scenario"})
            if "ev_change" in view.columns:
                view["EV Delta"] = view["ev_change"].apply(_format_currency)
            if "ev_change_pct" in view.columns:
                view["EV Delta (%)"] = view["ev_change_pct"].apply(lambda v: f"{v:.1f}%")
            st.dataframe(view, hide_index=True, use_container_width=True)

            if "ev_change" in view.columns:
                ev_chart = px.bar(
                    what_if_df.rename(columns={"scenario_name": "Scenario"}),
                    x="Scenario",
                    y="ev_change",
                    title="What-if Impact on Enterprise Value",
                    labels={"ev_change": "Change", "Scenario": "Scenario"},
                )
                st.plotly_chart(ev_chart, use_container_width=True)
        else:
            st.info("Use the scenario presets to populate what-if results here.")
        if goal_result.get("success"):
            st.caption(
                f"Goal seek: cogs_ratio → {goal_result.get('optimal_value'):.3f} to reach EV {_format_currency(goal_target)}"
            )
        else:
            st.info("Goal seek will display the target input once a feasible solution is found.")

    with analytics_tabs[5]:
        st.markdown("### Optimization, portfolio, ESG, and real options")
        opt_view = {
            "Optimal Weights": ", ".join([f"{w:.2f}" for w in portfolio_opt.get("optimal_weights", [])]),
            "Expected Return": f"{portfolio_opt.get('expected_return',0):.2%}",
            "Volatility": f"{portfolio_opt.get('volatility',0):.2%}",
            "Sharpe Ratio": f"{portfolio_opt.get('sharpe_ratio',0):.2f}",
        }
        st.json(opt_view)

        st.caption(
            f"Real option to expand: option value {_format_currency(options_value.get('option_value',0))} (probability-weighted)."
        )

        if not esg_df.empty:
            esg_df["annual_cost"] = esg_df["annual_cost"].apply(_format_currency)
            esg_df["cumulative_cost"] = esg_df["cumulative_cost"].apply(_format_currency)
            st.dataframe(esg_df.rename(columns={"annual_cost": "Carbon Cost", "cumulative_cost": "Cumulative"}), hide_index=True, use_container_width=True)

        st.markdown("**Correlated shocks (copula-style view)**")
        st.dataframe(corr_df, use_container_width=True)

