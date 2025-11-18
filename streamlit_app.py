"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Interactive labor management, CAPEX scheduling, financial modeling
Author: Advanced Analytics Team
Version: 1.0 (November 2025)
"""

import os
import copy
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import platform modules
from financial_model import (
    run_financial_model,
    CompanyConfig,
    generate_financial_statements,
    generate_labor_statement,
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

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

def initialize_session_state():
    """Initialize or restore session state"""
    if 'labor_manager' not in st.session_state:
        st.session_state.labor_manager = initialize_default_labor_structure()
    
    if 'capex_manager' not in st.session_state:
        st.session_state.capex_manager = initialize_default_capex(CapexScheduleManager())
    
    if 'financial_model' not in st.session_state:
        st.session_state.financial_model = None
    
    if 'salary_growth_rate' not in st.session_state:
        st.session_state.salary_growth_rate = 0.05
    
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

initialize_session_state()


def _format_currency(val: float) -> str:
    return f"${val:,.0f}"


def _format_statement(df: pd.DataFrame, money_cols):
    formatted = df.copy()
    for col in money_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(_format_currency)
    return formatted

# =====================================================
# MAIN NAVIGATION
# =====================================================

tab_platform, tab_dashboard, tab_financial, tab_ai, tab_labor, tab_capex, tab_reports, tab_advanced = st.tabs([
    "Platform Settings",
    "Dashboard",
    "Financial Model",
    "AI & Machine Learning",
    "Labor Management",
    "CAPEX Management",
    "Reports",
    "Advanced Analytics",
])

# =====================================================
# PAGE 0: PLATFORM SETTINGS
# =====================================================

with tab_platform:
    st.markdown("# Platform Settings")

    st.markdown("### Global Parameters")
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

    st.markdown("### Schedules")
    cfg = CompanyConfig(
        labor_manager=st.session_state.labor_manager,
        capex_manager=st.session_state.capex_manager,
    )
    model = run_financial_model(cfg)
    years = list(model["years"])
    income_df, cashflow_df, balance_df = generate_financial_statements(model)
    labor_df = generate_labor_statement(model)

    capex_spend = st.session_state.capex_manager.yearly_capex_schedule(years[0], len(years))
    capex_spend_df = pd.DataFrame({"Year": years, "CAPEX Spend": [capex_spend.get(y, 0.0) for y in years]})

    depreciation_sched = st.session_state.capex_manager.depreciation_schedule(years[0], len(years))
    depreciation_df = pd.DataFrame({"Year": years, "Depreciation": [depreciation_sched.get(y, 0.0) for y in years]})

    production_df = pd.DataFrame({
        "Year": years,
        "Units Produced": [model["production_volume"][y] for y in years],
        "Revenue": [model["revenue"][y] for y in years],
        "COGS": [model["cogs"][y] for y in years],
    })

    working_cap_df = pd.DataFrame({
        "Year": years,
        "FCF": [model["fcf"][y] for y in years],
        "Discounted FCF": [model["discounted_fcf"][y] for y in years],
        "Working Capital Change": [model["working_capital_change"].get(y, 0.0) for y in years],
    })

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
        "Tax": [model["tax"][y] for y in years],
        "Interest": [model["interest_payment"][y] for y in years],
    })
    other_cost_df["Other Costs"] = other_cost_df["Tax"] + other_cost_df["Interest"]

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

    assembly_df = pd.DataFrame({
        "Year": years,
        "Annual Capacity": [model["config"].annual_capacity for _ in years],
        "Capacity Utilization": [model["config"].capacity_utilization.get(y, 0.0) for y in years],
        "Units Produced": [model["production_volume"][y] for y in years],
        "Working Days": [model["config"].working_days for _ in years],
    })

    schedule_tabs = st.tabs([
        "Labor Cost",
        "CAPEX Spend",
        "Depreciation",
        "Production",
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

    with schedule_tabs[0]:
        if labor_df.empty:
            st.info("No labor schedule available. Add positions to view costs and headcount.")
        else:
            st.dataframe(labor_df, use_container_width=True, hide_index=True)

    with schedule_tabs[1]:
        st.dataframe(_format_statement(capex_spend_df, ["CAPEX Spend"]), use_container_width=True, hide_index=True)

    with schedule_tabs[2]:
        st.dataframe(_format_statement(depreciation_df, ["Depreciation"]), use_container_width=True, hide_index=True)

    with schedule_tabs[3]:
        st.dataframe(_format_statement(production_df, ["Revenue", "COGS"]), use_container_width=True, hide_index=True)

    with schedule_tabs[4]:
        st.dataframe(_format_statement(working_cap_df, ["FCF", "Discounted FCF", "Working Capital Change"]), use_container_width=True, hide_index=True)

    with schedule_tabs[5]:
        st.dataframe(_format_statement(financing_df, ["Interest", "Loan Repayment", "Long Term Debt", "Cash Flow from Financing"]), use_container_width=True, hide_index=True)

    with schedule_tabs[6]:
        st.dataframe(_format_statement(fixed_cost_df, ["Fixed Operating Costs", "Depreciation"]), use_container_width=True, hide_index=True)

    with schedule_tabs[7]:
        st.dataframe(_format_statement(variable_cost_df, ["COGS (Variable)"]), use_container_width=True, hide_index=True)

    with schedule_tabs[8]:
        st.dataframe(_format_statement(other_cost_df, ["Tax", "Interest", "Other Costs"]), use_container_width=True, hide_index=True)

    with schedule_tabs[9]:
        st.dataframe(_format_statement(debt_schedule_df, ["Interest", "Principal", "Ending Balance"]), use_container_width=True, hide_index=True)

    with schedule_tabs[10]:
        owner_pct = st.slider("Owner Equity %", 0.0, 100.0, float(st.session_state.owner_equity_pct), key="owner_equity_pct_slider")
        st.session_state.owner_equity_pct = owner_pct
        investor_pct = max(0.0, 100.0 - owner_pct)
        st.info(f"Owner: {owner_pct:.1f}% | Investor: {investor_pct:.1f}%")
        investment_df = pd.DataFrame({
            "Year": years,
            "Owner %": [owner_pct for _ in years],
            "Investor %": [investor_pct for _ in years],
            "Owner Equity": [model["config"].equity_investment * (owner_pct / 100.0) if y == years[0] else 0.0 for y in years],
            "Investor Equity": [model["config"].equity_investment * (investor_pct / 100.0) if y == years[0] else 0.0 for y in years],
            "Debt Raised": [model["config"].loan_amount if y == years[0] else 0.0 for y in years],
        })
        st.dataframe(
            _format_statement(investment_df, ["Owner Equity", "Investor Equity", "Debt Raised"]),
            use_container_width=True,
            hide_index=True,
        )

    with schedule_tabs[11]:
        st.dataframe(_format_statement(assets_df, ["Fixed Assets", "Current Assets", "Total Assets"]), use_container_width=True, hide_index=True)

    with schedule_tabs[12]:
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

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

with tab_dashboard:
    st.markdown("# Executive Dashboard")
    
    # Run financial model
    cfg = CompanyConfig(
        labor_manager=st.session_state.labor_manager,
        capex_manager=st.session_state.capex_manager
    )
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
# PAGE 3: LABOR MANAGEMENT
# =====================================================

with tab_labor:
    st.markdown("# Labor Position Management")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Current Positions", "Add Position", "Edit Position"])
    
    # TAB 1: View Positions
    with tab1:
        st.markdown("## Current Labor Schedule")
        
        positions_df = st.session_state.labor_manager.get_position_summary()
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        labor_costs = st.session_state.labor_manager.get_labor_cost_by_type(2026, st.session_state.salary_growth_rate)
        total_cost = labor_costs['Direct'] + labor_costs['Indirect']
        hc = st.session_state.labor_manager.get_total_headcount(2026)
        
        with col1:
            st.metric("Total Headcount", hc)
        with col2:
            st.metric("Total Annual Cost", f"${total_cost/1e6:.2f}M")
        with col3:
            st.metric("Cost per Employee", f"${total_cost/hc:,.0f}" if hc > 0 else "N/A")
        
        # Labor Cost Schedule
        st.markdown("## 5-Year Labor Cost Schedule")
        
        cost_schedule = LaborCostSchedule(st.session_state.labor_manager)
        schedule_df = cost_schedule.generate_5year_schedule(salary_growth=st.session_state.salary_growth_rate)
        
        # Format for display
        display_df = schedule_df.copy()
        for col in ['Direct Labor Cost', 'Indirect Labor Cost', 'Total Labor Cost']:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:.2f}M")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = px.line(
            schedule_df,
            x='Year',
            y=['Direct Labor Cost', 'Indirect Labor Cost'],
            markers=True,
            title="Labor Cost Forecast (2026-2030)",
            labels={'value': 'Cost ($)', 'variable': 'Labor Type'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Add Position
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
                    training_cost=training_cost
                )
                st.success(f"Position added. ID: {position_id}")
                st.session_state.last_update = datetime.now()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # TAB 3: Edit Position
    with tab3:
        st.markdown("## Edit Labor Position")
        
        positions = st.session_state.labor_manager.get_all_positions()
        if not positions:
            st.warning("No positions to edit")
        else:
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
                new_status = st.selectbox("Status", list(EmploymentStatus), index=list(EmploymentStatus).index(pos.status))
                new_benefits = st.slider("Benefits %", 0.0, 1.0, pos.benefits_percent)
                new_training = st.number_input("Training Cost", min_value=0, value=int(pos.training_cost_annual))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Changes", key="labor_save_changes"):
                    try:
                        st.session_state.labor_manager.edit_position(
                            selected_id,
                            position_name=new_name,
                            headcount=new_headcount,
                            annual_salary=new_salary,
                            status=new_status,
                            benefits_percent=new_benefits,
                            training_cost_annual=new_training
                        )
                        st.success("Position updated.")
                        st.session_state.last_update = datetime.now()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col2:
                if st.button("Remove Position", key="labor_remove_position"):
                    try:
                        st.session_state.labor_manager.remove_position(selected_id)
                        st.success("Position removed.")
                        st.session_state.last_update = datetime.now()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# =====================================================
# PAGE 3: CAPEX MANAGEMENT
# =====================================================

with tab_capex:
    st.markdown("# Capital Expenditure Management")
    
    tab1, tab2, tab3 = st.tabs(["Current Assets", "Add Asset", "Edit Asset"])
    
    # TAB 1: View Assets
    with tab1:
        st.markdown("## Current CAPEX Schedule")
        
        assets = st.session_state.capex_manager.list_items()
        items_data = []
        for item in assets:
            items_data.append({
                'ID': item.item_id,
                'Name': item.name,
                'Category': item.category,
                'Amount ($M)': f"${item.amount/1e6:.2f}",
                'Life (years)': item.useful_life,
                'Start Year': item.start_year
            })
        
        if items_data:
            items_df = pd.DataFrame(items_data)
            st.dataframe(items_df, use_container_width=True, hide_index=True)
        else:
            st.info("No capital assets configured")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        total_capex = st.session_state.capex_manager.total_capex()
        
        with col1:
            st.metric("Total CAPEX", f"${total_capex/1e6:.2f}M")
        with col2:
            st.metric("# Assets", len(assets))
        with col3:
            deprec_schedule = st.session_state.capex_manager.depreciation_schedule(2026, 5)
            st.metric("2026 Depreciation", f"${deprec_schedule.get(2026, 0)/1e3:.0f}K")
    
    # TAB 2: Add Asset
    with tab2:
        st.markdown("## Add New Capital Asset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Asset Name", value="New Asset")
            category = st.text_input("Asset Category", value="Equipment")
            amount = st.number_input("Acquisition Cost ($)", min_value=10000, value=100000, step=10000)
            useful_life = st.number_input("Useful Life (years)", min_value=1, value=10)
        
        with col2:
            salvage_value = st.number_input("Salvage Value ($)", min_value=0, value=0, step=5000)
            start_year = st.number_input("Start Year", min_value=2026, value=2026)
            notes = st.text_area("Notes", value="", key="capex_add_notes")
        
        if st.button("Add Asset"):
            try:
                asset_id = st.session_state.capex_manager.add_item(
                    name=name,
                    amount=amount,
                    start_year=int(start_year),
                    useful_life=int(useful_life),
                    salvage_value=salvage_value,
                    category=category,
                    notes=notes
                )
                st.success(f"Asset added. ID: {asset_id}")
                st.session_state.last_update = datetime.now()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # TAB 3: Edit Asset
    with tab3:
        st.markdown("## Edit Capital Asset")
        
        assets = st.session_state.capex_manager.list_items()
        if not assets:
            st.warning("No assets to edit")
        else:
            asset_display = {f"{item.item_id} - {item.name}": item.item_id for item in assets}
            selected_display = st.selectbox("Select Asset", list(asset_display.keys()))
            selected_id = asset_display[selected_display]
            asset = st.session_state.capex_manager.get_item(selected_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Name", value=asset.name)
                new_category = st.text_input("Category", value=asset.category)
                new_amount = st.number_input("Amount ($)", min_value=0, value=int(asset.amount))
            
            with col2:
                new_life = st.number_input("Useful Life (years)", min_value=1, value=asset.useful_life)
                new_salvage = st.number_input("Salvage Value", min_value=0, value=int(asset.salvage_value))
                new_notes = st.text_area(
                    "Notes", value=asset.notes, key=f"capex_edit_notes_{selected_id}"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Changes", key="capex_save_changes"):
                    try:
                        st.session_state.capex_manager.edit_item(
                            selected_id,
                            name=new_name,
                            category=new_category,
                            amount=new_amount,
                            useful_life=new_life,
                            salvage_value=new_salvage,
                            notes=new_notes
                        )
                        st.success("Asset updated.")
                        st.session_state.last_update = datetime.now()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col2:
                if st.button("Remove Asset", key="capex_remove_asset"):
                    try:
                        st.session_state.capex_manager.remove_item(selected_id)
                        st.success("Asset removed.")
                        st.session_state.last_update = datetime.now()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# =====================================================
# PAGE 4: FINANCIAL MODEL
# =====================================================

with tab_financial:
    st.markdown("# Financial Model & Valuation")
    
    tab1, tab2 = st.tabs(["Run Model", "Results"])
    
    with tab1:
        st.markdown("## Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wacc = st.slider("WACC (%)", 0, 20, 8) / 100
            terminal_growth = st.slider("Terminal Growth Rate (%)", 0, 5, 2) / 100
            revenue_cagr = st.slider("Revenue CAGR (%)", 0, 30, 15) / 100
            cogs_percent = st.slider("COGS % of Revenue", 0, 100, 65) / 100
        
        with col2:
            tax_rate = st.slider("Tax Rate (%)", 0, 50, 21) / 100
            debt_amount = st.number_input("Debt ($M)", min_value=0, value=50) * 1e6
            interest_rate = st.slider("Interest Rate (%)", 0, 15, 5) / 100
            shares_outstanding = st.number_input("Shares Outstanding (M)", min_value=1, value=100) * 1e6
        
        if st.button("Run Financial Model"):
            try:
                cfg = CompanyConfig(
                    revenue_cagr=revenue_cagr,
                    cogs_percent=cogs_percent,
                    tax_rate=tax_rate,
                    wacc=wacc,
                    terminal_growth_rate=terminal_growth,
                    debt_amount=debt_amount,
                    interest_rate=interest_rate,
                    shares_outstanding=shares_outstanding,
                    labor_manager=st.session_state.labor_manager,
                    capex_manager=st.session_state.capex_manager
                )
                
                st.session_state.financial_model = run_financial_model(cfg)
                st.session_state.last_update = datetime.now()
                st.success("Model executed successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        if st.session_state.financial_model:
            model = st.session_state.financial_model
            years = list(model["years"])

            st.markdown("## Financial Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Enterprise Value", f"${model['enterprise_value']/1e6:.1f}M")
            with col2:
                st.metric("5-Year FCF", f"${sum(model['fcf'].values())/1e6:.1f}M")
            with col3:
                ev_revenue = model['enterprise_value'] / model['revenue'][years[0]]
                st.metric("EV/Revenue", f"{ev_revenue:.1f}x")
            with col4:
                terminal_revenue = model['revenue'][years[-1]]
                st.metric("Terminal Value", f"${terminal_revenue*5/1e6:.1f}M")
            
            # Forecast table
            forecast_df = pd.DataFrame({
                'Year': years,
                'Revenue': [f"${model['revenue'][y]/1e6:.1f}M" for y in years],
                'EBIT': [f"${model['ebit'][y]/1e6:.1f}M" for y in years],
                'FCF': [f"${model['fcf'][y]/1e6:.1f}M" for y in years],
                'Cash': [f"${model['cash_balance'][y]/1e6:.1f}M" for y in years]
            })
            
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            
            # Charts
            chart_df = pd.DataFrame({
                'Year': years,
                'Revenue': [model['revenue'][y] for y in years],
                'EBIT': [model['ebit'][y] for y in years],
                'FCF': [model['fcf'][y] for y in years]
            })
            
            fig = px.line(chart_df, x='Year', y=['Revenue', 'EBIT', 'FCF'], markers=True,
                         title="5-Year Forecast")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the model to see results")

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

    cfg = CompanyConfig(
        labor_manager=st.session_state.labor_manager,
        capex_manager=st.session_state.capex_manager,
    )
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

        forecast_df = pd.DataFrame(
            {
                "Future Year": smoothing.get("future_years", []),
                "SES Forecast": smoothing.get("future_forecast", []),
            }
        )
        if not forecast_df.empty:
            st.dataframe(forecast_df, hide_index=True, use_container_width=True)
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
        else:
            st.info("Segmentation results will appear once product mix and revenue are available.")

        if regression:
            st.caption(
                f"Regression (volume→revenue): slope={regression.get('slope',0):.2f}, R²={regression.get('r_squared',0):.2f}"
            )
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

# =====================================================
# FOOTER
# =====================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Quick Links")
    st.markdown("""
    - [GitHub](https://github.com/Kossit73/Automobile_Manufacturing)
    - [Documentation](https://github.com/Kossit73/Automobile_Manufacturing/blob/main/README.md)
    - [Labor Guide](https://github.com/Kossit73/Automobile_Manufacturing/blob/main/LABOR_MANAGEMENT_GUIDE.md)
    """)

with col2:
    st.markdown("### Features")
    st.markdown("""
    - ✓ Labor CRUD management
    - ✓ CAPEX scheduling
    - ✓ Financial modeling
    - ✓ DCF valuation
    - ✓ Advanced reports
    """)

with col3:
    st.markdown("### Info")
    if st.session_state.last_update:
        st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    st.write("Platform v1.0")
    st.write("Python 3.7+")

st.markdown("""
---
**Automobile Manufacturing Financial Platform** © 2025 | [GitHub](https://github.com/Kossit73/Automobile_Manufacturing)
""")
