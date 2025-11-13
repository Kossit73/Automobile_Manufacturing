"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Clean, professional interface with horizontal navigation and comprehensive schedules.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from streamlit.delta_generator import DeltaGenerator
import plotly.graph_objects as go

from financial_model import (
    CompanyConfig,
    generate_financial_statements,
    generate_labor_statement,
    run_financial_model,
)
from labor_management import (
    LaborScheduleManager,
    initialize_default_labor_structure,
)
from capex_management import CapexScheduleManager, initialize_default_capex

# ---------------------------------------------------------------------------
# AI & MACHINE LEARNING CONFIGURATION
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SESSION HELPERS
# ---------------------------------------------------------------------------

def _payload_to_ai_settings(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
    st.experimental_rerun()


def _ensure_state() -> None:
    if "labor_manager" not in st.session_state:
        st.session_state["labor_manager"] = initialize_default_labor_structure()

    if "capex_manager" not in st.session_state:
        manager = CapexScheduleManager()
        st.session_state["capex_manager"] = initialize_default_capex(manager)

    if "company_config" not in st.session_state:
        cfg = CompanyConfig()
        cfg.labor_manager = st.session_state["labor_manager"]
        cfg.capex_manager = st.session_state["capex_manager"]
        st.session_state["company_config"] = cfg

    if "ai_payload" not in st.session_state:
        st.session_state["ai_payload"] = {}

    if "ai_settings" not in st.session_state:
        st.session_state["ai_settings"] = _payload_to_ai_settings(st.session_state["ai_payload"])

    _ensure_model()


def _ensure_model() -> None:
    if "financial_model" not in st.session_state:
        _run_model()


def _run_model() -> Dict[str, Any]:
    cfg: CompanyConfig = st.session_state["company_config"]
    cfg.labor_manager = st.session_state["labor_manager"]
    cfg.capex_manager = st.session_state["capex_manager"]
    model = run_financial_model(cfg)
    st.session_state["financial_model"] = model
    st.session_state["last_model_run"] = datetime.now().isoformat()
    return model


# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------

def _projection_years(model: Dict[str, Any]) -> List[int]:
    years = model.get("years", [])
    if isinstance(years, range):
        return list(years)
    if isinstance(years, Sequence):
        return list(years)
    return []


def _config_years(cfg: CompanyConfig) -> List[int]:
    return [cfg.start_year + i for i in range(cfg.projection_years)]


def _safe_dataframe(data: Optional[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=list(columns))
    # ensure consistent column order when a superset is provided
    missing = [col for col in columns if col not in data.columns]
    if missing:
        for col in missing:
            data[col] = ""
    return data[list(columns)]


def _currency_series(values: Iterable[float]) -> List[str]:
    return [f"${value:,.0f}" for value in values]


def _build_labor_cost_schedule(manager: LaborScheduleManager, cfg: CompanyConfig) -> pd.DataFrame:
    years = _config_years(cfg)
    rows = []
    for year in years:
        cost_by_type = manager.get_labor_cost_by_type(year, cfg.annual_salary_growth)
        headcount = manager.get_headcount_by_type(year)
        rows.append(
            {
                "Year": year,
                "Direct Headcount": headcount.get("Direct", 0),
                "Indirect Headcount": headcount.get("Indirect", 0),
                "Total Headcount": headcount.get("Direct", 0) + headcount.get("Indirect", 0),
                "Direct Labor Cost": cost_by_type.get("Direct", 0.0),
                "Indirect Labor Cost": cost_by_type.get("Indirect", 0.0),
                "Total Labor Cost": cost_by_type.get("Direct", 0.0) + cost_by_type.get("Indirect", 0.0),
            }
        )

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Direct Headcount",
                "Indirect Headcount",
                "Total Headcount",
                "Direct Labor Cost",
                "Indirect Labor Cost",
                "Total Labor Cost",
            ]
        )

    schedule["Direct Labor Cost"] = _currency_series(schedule["Direct Labor Cost"])
    schedule["Indirect Labor Cost"] = _currency_series(schedule["Indirect Labor Cost"])
    schedule["Total Labor Cost"] = _currency_series(schedule["Total Labor Cost"])
    return schedule


def _capex_spend_schedule(manager: CapexScheduleManager, cfg: CompanyConfig) -> pd.DataFrame:
    years = _config_years(cfg)
    schedule = manager.yearly_capex_schedule(cfg.start_year, len(years))
    rows = [{"Year": year, "Capital Spend": schedule.get(year, 0.0)} for year in years]
    df = pd.DataFrame(rows)
    df["Capital Spend"] = _currency_series(df["Capital Spend"])
    return df


def _capex_depreciation_schedule(manager: CapexScheduleManager, cfg: CompanyConfig) -> pd.DataFrame:
    years = _config_years(cfg)
    schedule = manager.depreciation_schedule(cfg.start_year, len(years))
    rows = [{"Year": year, "Depreciation": schedule.get(year, 0.0)} for year in years]
    df = pd.DataFrame(rows)
    df["Depreciation"] = _currency_series(df["Depreciation"])
    return df


def _debt_schedule(model: Dict[str, Any]) -> pd.DataFrame:
    years = _projection_years(model)
    rows = []
    for year in years:
        rows.append(
            {
                "Year": year,
                "Debt Draw": model.get("debt_draws", {}).get(year, 0.0),
                "Interest Payment": model.get("interest_payment", {}).get(year, 0.0),
                "Principal Repayment": model.get("loan_repayment", {}).get(year, 0.0),
                "Outstanding Debt": model.get("outstanding_debt", {}).get(year, 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["Year", "Debt Draw", "Interest Payment", "Principal Repayment", "Outstanding Debt"]
        )
    df["Debt Draw"] = _currency_series(df["Debt Draw"])
    df["Interest Payment"] = _currency_series(df["Interest Payment"])
    df["Principal Repayment"] = _currency_series(df["Principal Repayment"])
    df["Outstanding Debt"] = _currency_series(df["Outstanding Debt"])
    return df


def _forecast_schedule(model: Dict[str, Any]) -> pd.DataFrame:
    years = _projection_years(model)
    rows = []
    for year in years:
        rows.append(
            {
                "Year": year,
                "Revenue": model.get("revenue", {}).get(year, 0.0),
                "COGS": model.get("cogs", {}).get(year, 0.0),
                "Operating Expenses": model.get("opex", {}).get(year, 0.0),
                "Net Profit": model.get("net_profit", {}).get(year, 0.0),
                "Closing Cash": model.get("cash_balance", {}).get(year, 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["Year", "Revenue", "COGS", "Operating Expenses", "Net Profit", "Closing Cash"]
        )
    for column in ["Revenue", "COGS", "Operating Expenses", "Net Profit", "Closing Cash"]:
        df[column] = _currency_series(df[column])
    return df


def _dataframe_to_markdown(df: Optional[pd.DataFrame], hide_index: bool = True) -> str:
    df = df if df is not None else pd.DataFrame()
    columns: List[str] = list(getattr(df, "columns", []))
    if not columns:
        return ""

    rows: List[Dict[str, Any]]
    if getattr(df, "empty", True):
        rows = []
    else:
        try:
            rows = df.to_dict(orient="records")  # type: ignore[attr-defined]
        except Exception:
            rows = []

    headers = columns.copy()
    if not hide_index:
        headers = ["index"] + headers

    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    if rows:
        index_values: Sequence[Any] = list(getattr(df, "index", [])) if not hide_index else []
        for position, record in enumerate(rows):
            values = [str(record.get(column, "")) for column in columns]
            if not hide_index:
                index_value = ""
                if position < len(index_values):
                    index_value = str(index_values[position])
                values = [index_value] + values
            lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _render_table(df: Optional[pd.DataFrame], hide_index: bool = True) -> None:
    using_shim = bool(getattr(pd, "USING_PANDAS_SHIM", False))
    if using_shim:
        markdown = _dataframe_to_markdown(df, hide_index=hide_index)
        if markdown:
            st.markdown(markdown)
    else:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def _display_schedule(title: str, df: pd.DataFrame, note: str = "") -> None:
    st.subheader(title)
    if df.empty and note:
        st.write(note)
    _render_table(df, hide_index=True)


# ---------------------------------------------------------------------------
# RENDER FUNCTIONS
# ---------------------------------------------------------------------------

def _render_dashboard(model: Dict[str, Any]) -> None:
    years = _projection_years(model)
    if not years:
        st.info("Run the model to view dashboard insights.")
        return

    income_df, _, _ = generate_financial_statements(model)

    st.markdown("### Key Highlights")
    cols = st.columns(3)
    revenue = model.get("revenue", {}).get(years[0], 0.0)
    net_profit = model.get("net_profit", {}).get(years[-1], 0.0)
    cash = model.get("cash_balance", {}).get(years[-1], 0.0)
    cols[0].metric("Year 1 Revenue", f"${revenue:,.0f}")
    cols[1].metric("Final Year Net Profit", f"${net_profit:,.0f}")
    cols[2].metric("Closing Cash", f"${cash:,.0f}")

    st.markdown("### Revenue and Net Profit Trend")
    revenue_values = [model.get("revenue", {}).get(year, 0.0) for year in years]
    net_profit_values = [model.get("net_profit", {}).get(year, 0.0) for year in years]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=years, y=revenue_values, mode="lines+markers", name="Revenue")
    )
    fig.add_trace(
        go.Scatter(x=years, y=net_profit_values, mode="lines+markers", name="Net Profit")
    )
    fig.update_layout(
        margin=dict(t=20, r=20, b=20, l=20),
        height=360,
        xaxis_title="Year",
        yaxis_title="USD",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Income Statement Snapshot")
    snapshot = income_df.copy()
    snapshot["Revenue"] = _currency_series(snapshot["Revenue"])
    snapshot["Net Profit"] = _currency_series(snapshot["Net Profit"])
    _render_table(snapshot[["Year", "Revenue", "Net Profit"]])


def _render_labor_management() -> None:
    manager: LaborScheduleManager = st.session_state["labor_manager"]
    cfg: CompanyConfig = st.session_state["company_config"]

    st.markdown("### Labor Positions")
    summary = manager.get_position_summary()
    if summary.empty:
        st.write("No labor positions configured.")
    _render_table(summary)

    schedule = _build_labor_cost_schedule(manager, cfg)
    _display_schedule(
        "5-Year Labor Cost Schedule",
        schedule,
        "Labor cost schedule is shown even when no positions are configured.",
    )


def _render_capex_management() -> None:
    manager: CapexScheduleManager = st.session_state["capex_manager"]
    cfg: CompanyConfig = st.session_state["company_config"]

    st.markdown("### Capital Projects")
    items = manager.list_items()
    items_data = [item.to_dict() for item in items]
    items_df = pd.DataFrame(items_data)
    _display_schedule(
        "CAPEX Project Register",
        items_df,
        "Add projects in the CAPEX manager to populate the register.",
    )

    spend_df = _capex_spend_schedule(manager, cfg)
    _display_schedule(
        "CAPEX Spend Schedule",
        spend_df,
        "Annual capital outlays by project year.",
    )

    depreciation_df = _capex_depreciation_schedule(manager, cfg)
    _display_schedule(
        "Depreciation Schedule",
        depreciation_df,
        "Straight-line depreciation calculated from the CAPEX manager.",
    )


def _render_financial_model(model: Dict[str, Any]) -> None:
    income_df, cashflow_df, balance_df = generate_financial_statements(model)
    labor_df = generate_labor_statement(model)
    debt_df = _debt_schedule(model)

    _display_schedule(
        "Income Statement Schedule",
        income_df,
        "Run the model to populate income statement metrics.",
    )
    _display_schedule(
        "Cash Flow Schedule",
        cashflow_df,
        "Operating, investing, and financing cash flow summary.",
    )
    _display_schedule(
        "Balance Sheet Schedule",
        balance_df,
        "Assets, liabilities, and equity projections.",
    )
    _display_schedule(
        "Debt Amortization Schedule",
        debt_df,
        "Loan balance, interest, and repayment projection.",
    )
    _display_schedule(
        "Labor Cost Schedule",
        labor_df,
        "Labor costs are displayed even when no data is available.",
    )


def _render_reports(model: Dict[str, Any]) -> None:
    cfg: CompanyConfig = st.session_state["company_config"]
    labor_df = generate_labor_statement(model)
    capex_manager: CapexScheduleManager = st.session_state["capex_manager"]

    forecast_df = _forecast_schedule(model)
    _display_schedule(
        "Financial Forecast Schedule",
        forecast_df,
        "Revenue, profitability, and liquidity outlook.",
    )

    _display_schedule(
        "Labor Cost Schedule",
        labor_df,
        "Labor schedule replicated for reporting completeness.",
    )

    capex_spend_df = _capex_spend_schedule(capex_manager, cfg)
    _display_schedule(
        "CAPEX Spend Schedule",
        capex_spend_df,
        "Capital expenditures included in consolidated reports.",
    )

    debt_df = _debt_schedule(model)
    _display_schedule(
        "Debt Amortization Schedule",
        debt_df,
        "Debt schedule replicated for reporting completeness.",
    )


def _render_platform_settings() -> None:
    cfg: CompanyConfig = st.session_state["company_config"]
    st.markdown("### Company Configuration")
    company_name = st.text_input("Company Name", cfg.company_name)
    start_year = st.number_input("Model Start Year", value=int(cfg.start_year), step=1)
    projection_years = st.number_input(
        "Projection Years",
        value=int(cfg.projection_years),
        min_value=1,
        max_value=20,
        step=1,
    )
    salary_growth = st.slider(
        "Annual Salary Growth", min_value=0.0, max_value=0.15, value=float(cfg.annual_salary_growth), step=0.01
    )

    if (
        company_name != cfg.company_name
        or int(start_year) != cfg.start_year
        or int(projection_years) != cfg.projection_years
        or salary_growth != cfg.annual_salary_growth
    ):
        cfg.company_name = company_name
        cfg.start_year = int(start_year)
        cfg.projection_years = int(projection_years)
        cfg.annual_salary_growth = float(salary_growth)
        cfg.__post_init__()
        _run_model()
        st.success("Settings updated. Financial model refreshed.")

    st.markdown("### Model Execution")
    st.write(f"Last model refresh: {st.session_state.get('last_model_run', 'Never')}")
    if st.button("Recalculate Model"):
        _run_model()
        st.success("Model recalculated.")


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
        model_name = form.text_input(
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
            GEN_AI_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower()) for label in feature_selection
        ]

        settings.update(
            {
                "enabled": enabled,
                "provider": provider,
                "model": model_name.strip() or "gpt-4",
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


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def _render_navigation() -> str:
    pages = [
        "Dashboard",
        "Labor Management",
        "CAPEX Management",
        "Financial Model",
        "Reports",
        "Platform Settings",
        "AI & Machine Learning",
    ]
    default_index = pages.index(st.session_state.get("active_page", "Dashboard"))
    selected = st.radio(
        "Main Navigation",
        pages,
        index=default_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["active_page"] = selected
    return selected


def main() -> None:
    st.set_page_config(
        page_title="Automobile Manufacturing Financial Platform",
        layout="wide",
    )

    _ensure_state()
    model = st.session_state["financial_model"]

    st.title("Automobile Manufacturing Financial Platform")
    st.caption("Comprehensive labor, CAPEX, and financial planning environment.")

    active_page = _render_navigation()

    if active_page == "Dashboard":
        _render_dashboard(model)
    elif active_page == "Labor Management":
        _render_labor_management()
    elif active_page == "CAPEX Management":
        _render_capex_management()
    elif active_page == "Financial Model":
        _render_financial_model(model)
    elif active_page == "Reports":
        _render_reports(model)
    elif active_page == "Platform Settings":
        _render_platform_settings()
    elif active_page == "AI & Machine Learning":
        _render_ai_settings(st.session_state["ai_payload"])
    else:
        st.warning("Select a module from the navigation bar to begin.")


if __name__ == "__main__":
    main()
