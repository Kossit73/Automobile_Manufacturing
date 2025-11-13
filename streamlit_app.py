"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Interactive labor management, CAPEX scheduling, financial modeling
Author: Advanced Analytics Team
Version: 1.0 (November 2025)
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, List
from streamlit.delta_generator import DeltaGenerator
import plotly.express as px
import plotly.graph_objects as go

# Import platform modules
from financial_model import (
    run_financial_model,
    CompanyConfig,
    generate_financial_statements,
    generate_labor_statement,
)
from labor_management import (
    initialize_default_labor_structure, LaborScheduleManager, LaborCostSchedule,
    ProductionLinkedLabor, LaborType, EmploymentStatus, JobCategory
)
from capex_management import initialize_default_capex, CapexScheduleManager

# =====================================================
# AI & MACHINE LEARNING CONFIGURATION CONSTANTS
# =====================================================

AI_PROVIDER_OPTIONS = (
    "OpenAI",
    "Azure OpenAI",
    "Anthropic",
    "Google Vertex AI",
    "AWS Bedrock",
)

ML_METHOD_LABELS: Dict[str, str] = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "Gradient Boosting (XGBoost)",
    "prophet": "Prophet Forecasting",
    "lstm": "LSTM Neural Network",
    "arima": "ARIMA Time Series",
}

GEN_AI_FEATURE_LABELS: Dict[str, str] = {
    "summary": "Executive Summary",
    "risk_alerts": "Risk Alerts",
    "opportunities": "Growth Opportunities",
    "scenario_analysis": "Scenario Insights",
    "sensitivity": "Sensitivity Commentary",
}

ML_LABEL_TO_CODE: Dict[str, str] = {label: code for code, label in ML_METHOD_LABELS.items()}
GEN_AI_LABEL_TO_CODE: Dict[str, str] = {
    label: code for code, label in GEN_AI_FEATURE_LABELS.items()
}


def _payload_to_ai_settings(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize persisted payload into session-ready AI settings."""

    payload = payload or {}
    ml_methods = payload.get("ml_methods", ["linear_regression"])
    features = payload.get("generative_features", ["summary"])

    return {
        "enabled": bool(payload.get("enabled", False)),
        "provider": payload.get("provider", "OpenAI"),
        "model": payload.get("model", "gpt-4"),
        "forecast_horizon": int(payload.get("forecast_horizon", 3)),
        "ml_methods": ml_methods if isinstance(ml_methods, list) else ["linear_regression"],
        "generative_features": features if isinstance(features, list) else ["summary"],
        "api_key": payload.get("api_key", ""),
    }


def _ai_settings_to_payload(settings: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Persist AI settings back into a mutable payload dictionary."""

    payload.clear()
    payload.update(
        {
            "enabled": bool(settings.get("enabled", False)),
            "provider": settings.get("provider", "OpenAI"),
            "model": settings.get("model", "gpt-4"),
            "forecast_horizon": int(settings.get("forecast_horizon", 3)),
            "ml_methods": settings.get("ml_methods", ["linear_regression"]),
            "generative_features": settings.get("generative_features", ["summary"]),
            "api_key": settings.get("api_key", ""),
        }
    )


def _rerun() -> None:
    """Trigger a rerun of the Streamlit app to apply updated configuration."""

    st.experimental_rerun()


def _get_model_years(model: Dict[str, Any]) -> List[int]:
    """Return an ordered list of projection years from the financial model."""

    years = model.get("years", [])
    if isinstance(years, range):
        return list(years)
    if isinstance(years, (list, tuple)):
        return list(years)
    return list(years) if years else []


def _series_for_years(data: Any, years: Sequence[int]) -> List[Any]:
    """Normalize model series (dict keyed by year or iterable) into a list."""

    if data is None:
        return []
    if isinstance(data, dict):
        return [data[year] for year in years if year in data]
    if isinstance(data, (list, tuple)):
        return list(data)
    if hasattr(data, "values"):
        try:
            return list(data.values())
        except TypeError:
            pass
    return [data]


def _collect_series(model: Dict[str, Any], keys: Sequence[str], years: Sequence[int]) -> Dict[str, List[Any]]:
    """Convenience helper to gather multiple model series as ordered lists."""

    return {key: _series_for_years(model.get(key), years) for key in keys}


def _get_start_year() -> int:
    """Resolve the active model start year with a safe fallback."""

    default_year = CompanyConfig().start_year
    model = st.session_state.get("financial_model")
    if isinstance(model, dict):
        cfg = model.get("config")
        if isinstance(cfg, CompanyConfig):
            return cfg.start_year
    return default_year


def _render_ai_settings(payload: Dict[str, Any], container: Optional[DeltaGenerator] = None) -> None:
    target = container or st
    settings = st.session_state.setdefault("ai_settings", _payload_to_ai_settings(payload))
    st.session_state.setdefault("ai_api_key", settings.get("api_key", ""))

    provider_options = list(AI_PROVIDER_OPTIONS)
    if settings.get("provider") not in provider_options:
        provider_options.append(settings.get("provider"))

    current_provider = settings.get("provider", "OpenAI")
    try:
        provider_index = provider_options.index(current_provider)
    except ValueError:
        provider_index = 0

    ml_defaults = [
        ML_METHOD_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("ml_methods", ["linear_regression"])
    ]
    feature_defaults = [
        GEN_AI_FEATURE_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("generative_features", ["summary"])
    ]

    form = target.form("ai_settings_form")
    with form:
        enabled = form.checkbox(
            "Enable AI Enhancements",
            value=bool(settings.get("enabled", False)),
            help="Toggle machine-learning forecasts and generative commentary.",
        )
        provider = form.selectbox(
            "Provider",
            provider_options,
            index=provider_index,
            help="Select the API provider powering generative insights.",
        )
        model = form.text_input(
            "Model",
            value=settings.get("model", "gpt-4"),
            help="Name of the deployed model (for example `gpt-4o-mini`).",
        )
        horizon = form.number_input(
            "Forecast Horizon (years)",
            min_value=0,
            max_value=20,
            value=int(settings.get("forecast_horizon", 3)),
            step=1,
            help="Number of additional years used for machine-learning revenue forecasts.",
        )

        ml_selection = form.multiselect(
            "Machine Learning Methods",
            list(ML_METHOD_LABELS.values()),
            default=ml_defaults,
            help="Choose algorithms applied to projected net revenue.",
        )
        feature_selection = form.multiselect(
            "Generative Features",
            list(GEN_AI_FEATURE_LABELS.values()),
            default=feature_defaults,
            help="Pick the narrative focus areas generated by the AI summary.",
        )
        api_key = form.text_input(
            "API Key",
            value=st.session_state.get("ai_api_key", ""),
            type="password",
            help="Store your provider API key securely. Keys are retained only for the current session.",
        )

        submitted = form.form_submit_button("Save AI Configuration")

    if submitted:
        ml_codes = [ML_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower()) for label in ml_selection]
        feature_codes = [
            GEN_AI_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower())
            for label in feature_selection
        ]

        settings.update(
            {
                "enabled": enabled,
                "provider": provider,
                "model": model.strip() or "gpt-4",
                "forecast_horizon": int(horizon),
                "ml_methods": ml_codes or ["linear_regression"],
                "generative_features": feature_codes or ["summary"],
                "api_key": api_key.strip(),
            }
        )
        st.session_state["ai_settings"] = settings
        st.session_state["ai_api_key"] = settings.get("api_key", "")
        _ai_settings_to_payload(settings, payload)
        st.success("AI configuration updated. Rerunning the model with the new settings.")
        _rerun()

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Manufacturing Financial Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    if 'ai_payload' not in st.session_state:
        st.session_state.ai_payload = {}

    if 'salary_growth_rate' not in st.session_state:
        st.session_state.salary_growth_rate = 0.05

    if 'last_update' not in st.session_state:
        st.session_state.last_update = None

initialize_session_state()

PAGE_OPTIONS = [
    "Dashboard",
    "Labor Management",
    "CAPEX Management",
    "Financial Model",
    "Reports",
]

default_page = st.session_state.get("active_page", PAGE_OPTIONS[0])
page_index = PAGE_OPTIONS.index(default_page) if default_page in PAGE_OPTIONS else 0
page = st.radio(
    "Navigate Modules",
    PAGE_OPTIONS,
    horizontal=True,
    index=page_index,
    key="active_page",
    help="Switch between dashboards and management tools",
)

# =====================================================
# SIDEBAR - SETTINGS
# =====================================================

with st.sidebar:
    st.markdown("# Platform Settings")

    st.divider()

    # Global Settings
    st.markdown("### Global Parameters")
    
    salary_growth = st.slider(
        "Annual Salary Growth Rate (%)",
        min_value=0,
        max_value=10,
        value=int(st.session_state.salary_growth_rate * 100),
        help="Applied to all labor cost projections"
    )
    st.session_state.salary_growth_rate = salary_growth / 100

    st.divider()

    # AI & Machine Learning
    st.markdown("### AI & Machine Learning")
    _render_ai_settings(st.session_state.ai_payload)

    st.divider()

    # Platform Info
    st.markdown("### Platform Info")
    st.info(
        "**Manufacturing Financial Platform v1.0**\n\n"
        "- Labor management (CRUD)\n"
        "- CAPEX scheduling\n"
        "- Financial modeling\n"
        "- Advanced reporting\n\n"
        "*Select a module to navigate*"
    )

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

if page == "Dashboard":
    st.markdown("# Executive Dashboard")
    
    # Run financial model
    cfg = CompanyConfig(
        labor_manager=st.session_state.labor_manager,
        capex_manager=st.session_state.capex_manager
    )
    model = run_financial_model(cfg)
    st.session_state.financial_model = model
    years = _get_model_years(model)
    series = _collect_series(
        model,
        ["revenue", "net_profit", "ebit", "fcf", "cash_balance", "depreciation", "cogs", "opex"],
        years,
    )

    # Key Metrics
    st.markdown("## Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        revenue_2026 = series["revenue"][0] if series["revenue"] else 0
        st.metric("2026 Revenue", f"${revenue_2026/1e6:.1f}M", delta="Year 1")

    with col2:
        profit_2026 = series["net_profit"][0] if series["net_profit"] else 0
        margin = (profit_2026/revenue_2026*100) if revenue_2026 > 0 else 0
        st.metric("2026 Net Profit", f"${profit_2026/1e6:.1f}M", delta=f"{margin:.1f}%")
    
    with col3:
        headcount = st.session_state.labor_manager.get_total_headcount(2026)
        st.metric("Total Headcount", f"{headcount} employees", delta="Current")
    
    with col4:
        ev = model['enterprise_value']
        st.metric("Enterprise Value", f"${ev/1e6:.1f}M", delta="DCF")

    # Financial Overview
    st.markdown("## Five-Year Financial Forecast")

    col1, col2 = st.columns(2)

    with col1:
        forecast_df = pd.DataFrame({
            'Year': years,
            'Revenue ($M)': [x/1e6 for x in series['revenue']],
            'EBIT ($M)': [x/1e6 for x in series['ebit']],
            'Net Profit ($M)': [x/1e6 for x in series['net_profit']],
            'FCF ($M)': [x/1e6 for x in series['fcf']]
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
        labor_costs = st.session_state.labor_manager.get_labor_cost_by_type(2026, st.session_state.salary_growth_rate)
        st.markdown("### Labor Costs (2026)")
        st.metric("Direct Labor", f"${labor_costs['Direct']/1e6:.2f}M")
        st.metric("Indirect Labor", f"${labor_costs['Indirect']/1e6:.2f}M")
        st.metric("Total Labor", f"${(labor_costs['Direct'] + labor_costs['Indirect'])/1e6:.2f}M")
    
    with col2:
        hc_types = st.session_state.labor_manager.get_headcount_by_type(2026)
        st.markdown("### Headcount (2026)")
        st.metric("Direct Labor HC", f"{hc_types['Direct']} employees")
        st.metric("Indirect Labor HC", f"{hc_types['Indirect']} employees")
        st.metric("Total HC", f"{hc_types['Direct'] + hc_types['Indirect']} employees")
    
    with col3:
        capex_items = st.session_state.capex_manager.list_items()
        total_capex = st.session_state.capex_manager.total_capex()
        st.markdown("### Capital Assets (2026)")
        st.metric("Total CAPEX", f"${total_capex/1e6:.2f}M")
        st.metric("# Assets", f"{len(capex_items)}")
        depreciation_first_year = series["depreciation"][0] if series["depreciation"] else 0
        st.metric("Annual Depreciation", f"${depreciation_first_year/1e3:.0f}K")

# =====================================================
# PAGE 2: LABOR MANAGEMENT
# =====================================================

elif page == "Labor Management":
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
                st.success(f"Position added (ID: {position_id}).")
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
                if st.button("Save Changes"):
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
                if st.button("Remove Position"):
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

elif page == "CAPEX Management":
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
        
        items_columns = ['ID', 'Name', 'Category', 'Amount ($M)', 'Life (years)', 'Start Year']
        items_df = pd.DataFrame(items_data, columns=items_columns)
        st.dataframe(items_df, use_container_width=True, hide_index=True)
        if not items_data:
            st.caption("No capital assets configured yet. Add assets to populate this schedule.")
        
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

        capex_start = _get_start_year()
        spend_schedule = st.session_state.capex_manager.yearly_capex_schedule(capex_start, 5)
        st.markdown("### CAPEX Spend Schedule")
        spend_years = sorted(spend_schedule.keys()) if spend_schedule else []
        spend_df = pd.DataFrame({
            "Year": spend_years,
            "CAPEX Spend": [spend_schedule[y] for y in spend_years],
        }) if spend_years else pd.DataFrame(columns=["Year", "CAPEX Spend"])
        spend_display = spend_df.copy()
        if not spend_display.empty:
            spend_display["CAPEX Spend"] = spend_display["CAPEX Spend"].apply(lambda x: f"${x/1e6:.2f}M")
        st.dataframe(spend_display, use_container_width=True, hide_index=True)
        if spend_display.empty:
            st.caption("No capital spending scheduled for the selected horizon.")

        depreciation_schedule = st.session_state.capex_manager.depreciation_schedule(capex_start, 5)
        st.markdown("### Depreciation Schedule")
        dep_years = sorted(depreciation_schedule.keys()) if depreciation_schedule else []
        dep_df = pd.DataFrame({
            "Year": dep_years,
            "Depreciation": [depreciation_schedule[y] for y in dep_years],
        }) if dep_years else pd.DataFrame(columns=["Year", "Depreciation"])
        dep_display = dep_df.copy()
        if not dep_display.empty:
            dep_display["Depreciation"] = dep_display["Depreciation"].apply(lambda x: f"${x/1e6:.2f}M")
        st.dataframe(dep_display, use_container_width=True, hide_index=True)
        if dep_display.empty:
            st.caption("Depreciation schedule unavailable for the selected horizon.")

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
            notes = st.text_area("Notes", value="", key="add_asset_notes")
        
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
                st.success(f"Asset added (ID: {asset_id}).")
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
                notes_key = f"edit_asset_notes_{selected_id}"
                new_notes = st.text_area("Notes", value=asset.notes, key=notes_key)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Changes"):
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
                if st.button("Remove Asset"):
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

elif page == "Financial Model":
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
            years = _get_model_years(model)
            series = _collect_series(
                model,
                ["revenue", "ebit", "fcf", "cash_balance"],
                years,
            )

            st.markdown("## Financial Results")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Enterprise Value", f"${model['enterprise_value']/1e6:.1f}M")
            with col2:
                total_fcf = sum(series["fcf"]) if series["fcf"] else 0
                st.metric("5-Year FCF", f"${total_fcf/1e6:.1f}M")
            with col3:
                first_revenue = series["revenue"][0] if series["revenue"] else 0
                ev_revenue = (model['enterprise_value'] / first_revenue) if first_revenue else 0
                st.metric("EV/Revenue", f"{ev_revenue:.1f}x")
            with col4:
                terminal_revenue = series["revenue"][-1] if series["revenue"] else 0
                st.metric("Terminal Value", f"${terminal_revenue*5/1e6:.1f}M")

            # Forecast table
            forecast_df = pd.DataFrame({
                'Year': years,
                'Revenue': [f"${x/1e6:.1f}M" for x in series['revenue']],
                'EBIT': [f"${x/1e6:.1f}M" for x in series['ebit']],
                'FCF': [f"${x/1e6:.1f}M" for x in series['fcf']],
                'Cash': [f"${x/1e6:.1f}M" for x in series['cash_balance']]
            })

            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            # Charts
            chart_df = pd.DataFrame({
                'Year': years,
                'Revenue': series['revenue'],
                'EBIT': series['ebit'],
                'FCF': series['fcf']
            })
            
            fig = px.line(chart_df, x='Year', y=['Revenue', 'EBIT', 'FCF'], markers=True,
                         title="5-Year Forecast")
            st.plotly_chart(fig, use_container_width=True)

            income_df, cashflow_df, balance_df = generate_financial_statements(model)

            st.markdown("### Income Statement Schedule")
            income_display = income_df.copy()
            for col in income_display.columns:
                if col != "Year":
                    income_display[col] = income_display[col].apply(lambda x: f"${x/1e6:.2f}M")
            st.dataframe(income_display, use_container_width=True, hide_index=True)

            st.markdown("### Cash Flow Schedule")
            cash_display = cashflow_df.copy()
            for col in cash_display.columns:
                if col != "Year":
                    cash_display[col] = cash_display[col].apply(lambda x: f"${x/1e6:.2f}M")
            st.dataframe(cash_display, use_container_width=True, hide_index=True)

            st.markdown("### Balance Sheet Schedule")
            balance_display = balance_df.copy()
            for col in balance_display.columns:
                if col not in ("Year", "Balanced?"):
                    balance_display[col] = balance_display[col].apply(lambda x: f"${x/1e6:.2f}M")
            st.dataframe(balance_display, use_container_width=True, hide_index=True)

            debt_df = pd.DataFrame({
                "Year": years,
                "Interest Payment": [model['interest_payment'][y] for y in years],
                "Principal Payment": [model['loan_repayment'][y] for y in years],
                "Ending Balance": [model['outstanding_debt'][y] for y in years],
            })
            debt_display = debt_df.copy()
            for col in ("Interest Payment", "Principal Payment", "Ending Balance"):
                debt_display[col] = debt_display[col].apply(lambda x: f"${x/1e6:.2f}M")

            st.markdown("### Debt Amortization Schedule")
            st.dataframe(debt_display, use_container_width=True, hide_index=True)

            labor_statement = generate_labor_statement(model)
            st.markdown("### Labor Cost Schedule")
            if labor_statement.empty:
                st.caption("Labor schedule data will appear here once labor metrics are generated.")
            st.dataframe(labor_statement, use_container_width=True, hide_index=True)
        else:
            st.info("Run the model to see results")

# =====================================================
# PAGE 5: REPORTS
# =====================================================

elif page == "Reports":
    st.markdown("# Financial Reports & Exports")
    
    if st.session_state.financial_model:
        model = st.session_state.financial_model
        years = _get_model_years(model)
        series = _collect_series(
            model,
            ["revenue", "ebit", "net_profit", "fcf", "cogs", "opex"],
            years,
        )

        # Summary Report
        st.markdown("## Executive Summary")

        first_revenue = series["revenue"][0] if series["revenue"] else 0
        first_ebit = series["ebit"][0] if series["ebit"] else 0
        first_net_profit = series["net_profit"][0] if series["net_profit"] else 0
        first_fcf = series["fcf"][0] if series["fcf"] else 0
        total_fcf = sum(series["fcf"]) if series["fcf"] else 0

        summary_text = f"""
        **2026 Financials:**
        - Revenue: ${first_revenue/1e6:.1f}M
        - EBIT: ${first_ebit/1e6:.1f}M
        - Net Profit: ${first_net_profit/1e6:.1f}M
        - FCF: ${first_fcf/1e6:.1f}M

        **Valuation:**
        - Enterprise Value: ${model['enterprise_value']/1e6:.1f}M
        - 5-Year FCF: ${total_fcf/1e6:.1f}M

        **Workforce:**
        - Headcount: {st.session_state.labor_manager.get_total_headcount(2026)} employees
        - Labor Cost: ${(st.session_state.labor_manager.get_labor_cost_by_type(2026, st.session_state.salary_growth_rate)['Direct'] + st.session_state.labor_manager.get_labor_cost_by_type(2026, st.session_state.salary_growth_rate)['Indirect'])/1e6:.2f}M
        """

        st.markdown(summary_text)

        # Financial summary
        export_df = pd.DataFrame({
            'Year': years,
            'Revenue': series['revenue'],
            'COGS': series['cogs'],
            'OPEX': series['opex'],
            'EBIT': series['ebit'],
            'Net Profit': series['net_profit'],
            'FCF': series['fcf']
        })

        forecast_display = export_df.copy()
        for col in forecast_display.columns:
            if col != "Year":
                forecast_display[col] = forecast_display[col].apply(lambda x: f"${x/1e6:.2f}M")

        st.markdown("## Financial Forecast Schedule")
        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

        # Export buttons
        st.markdown("## Download Reports")

        csv = export_df.to_csv(index=False)
        st.download_button(
            "Download Financial Forecast (CSV)",
            csv,
            "financial_forecast.csv",
            "text/csv"
        )
        
        # Labor summary
        cost_schedule = LaborCostSchedule(st.session_state.labor_manager)
        labor_df = cost_schedule.generate_5year_schedule(salary_growth=st.session_state.salary_growth_rate)

        labor_display = labor_df.copy()
        currency_cols = ["Direct Labor Cost", "Indirect Labor Cost", "Total Labor Cost"]
        for col in currency_cols:
            labor_display[col] = labor_display[col].apply(lambda x: f"${x/1e6:.2f}M")

        st.markdown("## Labor Cost Schedule")
        st.dataframe(labor_display, use_container_width=True, hide_index=True)

        csv = labor_df.to_csv(index=False)
        st.download_button(
            "Download Labor Schedule (CSV)",
            csv,
            "labor_schedule.csv",
            "text/csv"
        )

        capex_start_year = _get_start_year()
        capex_schedule = st.session_state.capex_manager.yearly_capex_schedule(capex_start_year, 5)
        capex_years = sorted(capex_schedule.keys()) if capex_schedule else []
        capex_df = pd.DataFrame({
            "Year": capex_years,
            "CAPEX Spend": [capex_schedule[y] for y in capex_years]
        }) if capex_years else pd.DataFrame(columns=["Year", "CAPEX Spend"])
        capex_display = capex_df.copy()
        if not capex_display.empty:
            capex_display["CAPEX Spend"] = capex_display["CAPEX Spend"].apply(lambda x: f"${x/1e6:.2f}M")

        st.markdown("## CAPEX Spend Schedule")
        if capex_display.empty:
            st.caption("CAPEX spending will populate once capital projects are scheduled.")
        st.dataframe(capex_display, use_container_width=True, hide_index=True)

        debt_df = pd.DataFrame({
            "Year": years,
            "Interest Payment": [model['interest_payment'][y] for y in years],
            "Principal Payment": [model['loan_repayment'][y] for y in years],
            "Ending Balance": [model['outstanding_debt'][y] for y in years],
        })
        debt_display = debt_df.copy()
        for col in ("Interest Payment", "Principal Payment", "Ending Balance"):
            debt_display[col] = debt_display[col].apply(lambda x: f"${x/1e6:.2f}M")

        st.markdown("## Debt Amortization Schedule")
        st.dataframe(debt_display, use_container_width=True, hide_index=True)
    else:
        st.info("Run the financial model first to generate reports")

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
    st.markdown("### Platform Features")
    st.markdown("""
    - Labor CRUD management
    - CAPEX scheduling
    - Financial modeling
    - DCF valuation
    - Advanced reports
    """)

with col3:
    st.markdown("### Platform Information")
    if st.session_state.last_update:
        st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    st.write("Platform v1.0")
    st.write("Python 3.7+")

st.markdown("""
---
**Automobile Manufacturing Financial Platform** (c) 2025 | [GitHub](https://github.com/Kossit73/Automobile_Manufacturing)
""")
