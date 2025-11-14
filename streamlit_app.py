"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Clean, professional interface with horizontal navigation and comprehensive schedules.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime
import re
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
    EmploymentStatus,
    JobCategory,
    LaborPosition,
    LaborScheduleManager,
    LaborType,
    initialize_default_labor_structure,
)
from capex_management import CapexItem, CapexScheduleManager, initialize_default_capex

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


LABOR_TYPE_OPTIONS: Dict[str, LaborType] = {labor_type.value: labor_type for labor_type in LaborType}
JOB_CATEGORY_OPTIONS: Dict[str, JobCategory] = {
    category.value: category for category in JobCategory
}
EMPLOYMENT_STATUS_OPTIONS: Dict[str, EmploymentStatus] = {
    status.value: status for status in EmploymentStatus
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


def _product_label(name: str) -> str:
    """Return a human-friendly label for product keys."""

    return name.replace("_", " ")


def _parse_spend_curve_input(text: str) -> Optional[Dict[int, float]]:
    cleaned = text.strip()
    if not cleaned:
        return None

    entries = re.split(r"[,\n]+", cleaned)
    spend_curve: Dict[int, float] = {}
    for entry in entries:
        part = entry.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                "Spend curve entries must use the 'offset:share' format, e.g., '0:0.6, 1:0.4'."
            )
        offset_text, share_text = part.split(":", 1)
        offset_text = offset_text.strip()
        share_text = share_text.strip()

        try:
            offset = int(offset_text)
        except ValueError as exc:
            raise ValueError("Spend curve offsets must be whole numbers.") from exc

        percentage = share_text.endswith("%")
        if percentage:
            share_text = share_text[:-1]

        try:
            share = float(share_text)
        except ValueError as exc:
            raise ValueError("Spend curve shares must be numeric values.") from exc

        if percentage:
            share /= 100.0

        spend_curve[offset] = share

    if not spend_curve:
        raise ValueError("Provide at least one spend curve allocation or leave the field blank.")

    return spend_curve


def _format_spend_curve(spend_curve: Optional[Dict[int, float]]) -> str:
    if not spend_curve:
        return ""
    return ", ".join(f"{offset}:{share:.4g}" for offset, share in sorted(spend_curve.items()))


def _position_increment_schedule(position: LaborPosition, cfg: CompanyConfig) -> pd.DataFrame:
    years = _config_years(cfg)
    rows: List[Dict[str, Any]] = []
    for year in years:
        if year < position.start_year:
            continue
        if position.end_year is not None and year > position.end_year:
            continue

        years_since_start = year - position.start_year
        growth_factor = (1 + cfg.annual_salary_growth) ** years_since_start
        base_salary = position.annual_salary * growth_factor
        total_cost = position.calculate_annual_cost(growth_factor - 1)
        per_headcount = total_cost / position.headcount if position.headcount else 0.0

        rows.append(
            {
                "Year": year,
                "Base Salary (per HC)": base_salary,
                "Total Compensation (per HC)": per_headcount,
                "Total Role Cost": total_cost,
            }
        )

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Base Salary (per HC)",
                "Total Compensation (per HC)",
                "Total Role Cost",
            ]
        )

    for column in [
        "Base Salary (per HC)",
        "Total Compensation (per HC)",
        "Total Role Cost",
    ]:
        schedule[column] = _currency_series(schedule[column])

    return schedule


def _render_add_position_form(manager: LaborScheduleManager, cfg: CompanyConfig) -> None:
    st.markdown("#### Add Labor Position")
    with st.form("add_position_form"):
        col1, col2 = st.columns(2)
        position_name = col1.text_input("Position Name", key="add_position_name")
        labor_type_labels = list(LABOR_TYPE_OPTIONS.keys())
        labor_type_label = col2.selectbox(
            "Labor Type",
            labor_type_labels,
            index=0 if labor_type_labels else 0,
            key="add_labor_type",
        )

        job_category_labels = list(JOB_CATEGORY_OPTIONS.keys())
        status_labels = list(EMPLOYMENT_STATUS_OPTIONS.keys())

        col3, col4 = st.columns(2)
        job_category_label = col3.selectbox(
            "Job Category",
            job_category_labels,
            index=0 if job_category_labels else 0,
            key="add_job_category",
        )
        status_label = col4.selectbox(
            "Employment Status",
            status_labels,
            index=0 if status_labels else 0,
            key="add_status",
        )

        col5, col6 = st.columns(2)
        headcount = col5.number_input(
            "Headcount", min_value=0, value=1, step=1, key="add_headcount"
        )
        annual_salary = col6.number_input(
            "Annual Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
            key="add_annual_salary",
        )

        col7, col8 = st.columns(2)
        start_year = col7.number_input(
            "Start Year",
            value=int(cfg.start_year),
            step=1,
            key="add_start_year",
        )
        end_year_text = col8.text_input(
            "End Year (leave blank if ongoing)",
            value="",
            key="add_end_year",
        )

        benefits_percent = st.slider(
            "Benefits Percent",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key="add_benefits_percent",
        )

        col9, col10 = st.columns(2)
        overtime_hours = col9.number_input(
            "Overtime Hours (Annual)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key="add_overtime_hours",
        )
        overtime_rate = col10.number_input(
            "Overtime Rate Multiplier",
            min_value=1.0,
            value=1.5,
            step=0.1,
            key="add_overtime_rate",
        )

        col11, col12 = st.columns(2)
        training_cost = col11.number_input(
            "Training Cost (Annual)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="add_training_cost",
        )
        equipment_cost = col12.number_input(
            "Equipment Cost (Annual)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="add_equipment_cost",
        )

        notes = st.text_area("Notes", key="add_notes")
        submitted = st.form_submit_button("Add Position")

    if submitted:
        try:
            end_year = int(end_year_text) if end_year_text.strip() else None
        except ValueError:
            st.error("End year must be a whole number.")
            return

        try:
            manager.add_position(
                position_name=position_name.strip() or "New Position",
                labor_type=LABOR_TYPE_OPTIONS.get(labor_type_label, LaborType.DIRECT),
                job_category=JOB_CATEGORY_OPTIONS.get(job_category_label, JobCategory.ASSEMBLY),
                headcount=int(headcount),
                annual_salary=float(annual_salary),
                status=EMPLOYMENT_STATUS_OPTIONS.get(status_label, EmploymentStatus.ACTIVE),
                start_year=int(start_year),
                end_year=end_year,
                benefits_percent=float(benefits_percent),
                overtime_hours_annual=float(overtime_hours),
                overtime_rate=float(overtime_rate),
                training_cost=float(training_cost),
                equipment_cost=float(equipment_cost),
                notes=notes.strip(),
            )
            _run_model()
            st.success("Labor position added successfully.")
        except ValueError as exc:
            st.error(str(exc))


def _render_position_editor(
    position: LaborPosition, manager: LaborScheduleManager, cfg: CompanyConfig
) -> None:
    header = f"{position.position_id} — {position.position_name}"
    with st.expander(header, expanded=False):
        form_key = f"edit_form_{position.position_id}"
        with st.form(form_key):
            prefix = f"edit_{position.position_id}"

            col1, col2 = st.columns(2)
            position_name = col1.text_input(
                "Position Name",
                value=position.position_name,
                key=f"{prefix}_name",
            )
            labor_type_labels = list(LABOR_TYPE_OPTIONS.keys())
            labor_type_value = position.labor_type.value
            labor_type_index = (
                labor_type_labels.index(labor_type_value)
                if labor_type_value in labor_type_labels
                else 0
            )
            labor_type_label = col2.selectbox(
                "Labor Type",
                labor_type_labels,
                index=labor_type_index,
                key=f"{prefix}_labor_type",
            )

            job_category_labels = list(JOB_CATEGORY_OPTIONS.keys())
            job_category_value = position.job_category.value
            job_category_index = (
                job_category_labels.index(job_category_value)
                if job_category_value in job_category_labels
                else 0
            )
            status_labels = list(EMPLOYMENT_STATUS_OPTIONS.keys())
            status_value = position.status.value
            status_index = (
                status_labels.index(status_value) if status_value in status_labels else 0
            )

            col3, col4 = st.columns(2)
            job_category_label = col3.selectbox(
                "Job Category",
                job_category_labels,
                index=job_category_index,
                key=f"{prefix}_job_category",
            )
            status_label = col4.selectbox(
                "Employment Status",
                status_labels,
                index=status_index,
                key=f"{prefix}_status",
            )

            col5, col6 = st.columns(2)
            headcount = col5.number_input(
                "Headcount",
                min_value=0,
                value=int(position.headcount),
                step=1,
                key=f"{prefix}_headcount",
            )
            annual_salary = col6.number_input(
                "Annual Salary",
                min_value=0.0,
                value=float(position.annual_salary),
                step=1000.0,
                key=f"{prefix}_salary",
            )

            col7, col8 = st.columns(2)
            start_year = col7.number_input(
                "Start Year",
                value=int(position.start_year),
                step=1,
                key=f"{prefix}_start_year",
            )
            end_year_default = "" if position.end_year is None else str(position.end_year)
            end_year_text = col8.text_input(
                "End Year (leave blank if ongoing)",
                value=end_year_default,
                key=f"{prefix}_end_year",
            )

            benefits_percent = st.slider(
                "Benefits Percent",
                min_value=0.0,
                max_value=1.0,
                value=float(position.benefits_percent),
                step=0.01,
                key=f"{prefix}_benefits",
            )

            col9, col10 = st.columns(2)
            overtime_hours = col9.number_input(
                "Overtime Hours (Annual)",
                min_value=0.0,
                value=float(position.overtime_hours_annual),
                step=10.0,
                key=f"{prefix}_overtime_hours",
            )
            overtime_rate = col10.number_input(
                "Overtime Rate Multiplier",
                min_value=1.0,
                value=float(position.overtime_rate),
                step=0.1,
                key=f"{prefix}_overtime_rate",
            )

            col11, col12 = st.columns(2)
            training_cost = col11.number_input(
                "Training Cost (Annual)",
                min_value=0.0,
                value=float(position.training_cost_annual),
                step=500.0,
                key=f"{prefix}_training",
            )
            equipment_cost = col12.number_input(
                "Equipment Cost (Annual)",
                min_value=0.0,
                value=float(position.equipment_cost_annual),
                step=500.0,
                key=f"{prefix}_equipment",
            )

            notes = st.text_area(
                "Notes",
                value=position.notes,
                key=f"{prefix}_notes",
            )

            submitted = st.form_submit_button("Save Changes")

        if submitted:
            try:
                end_year = int(end_year_text) if end_year_text.strip() else None
            except ValueError:
                st.error("End year must be a whole number.")
            else:
                try:
                    manager.edit_position(
                        position.position_id,
                        position_name=position_name.strip() or position.position_name,
                        labor_type=LABOR_TYPE_OPTIONS.get(labor_type_label, position.labor_type),
                        job_category=JOB_CATEGORY_OPTIONS.get(
                            job_category_label, position.job_category
                        ),
                        headcount=int(headcount),
                        annual_salary=float(annual_salary),
                        status=EMPLOYMENT_STATUS_OPTIONS.get(
                            status_label, position.status
                        ),
                        start_year=int(start_year),
                        end_year=end_year,
                        benefits_percent=float(benefits_percent),
                        overtime_hours_annual=float(overtime_hours),
                        overtime_rate=float(overtime_rate),
                        training_cost_annual=float(training_cost),
                        equipment_cost_annual=float(equipment_cost),
                        notes=notes.strip(),
                    )
                    _run_model()
                    st.success("Position updated successfully.")
                except ValueError as exc:
                    st.error(str(exc))

        increment_df = _position_increment_schedule(position, cfg)
        st.markdown("**Yearly Increment Helper**")
        if increment_df.empty:
            st.caption("No projection years available for this position.")
        else:
            _render_table(increment_df)

        remove = st.button(
            "Remove Position",
            key=f"remove_{position.position_id}",
            help="Delete this labor position from the schedule.",
        )
        if remove:
            try:
                manager.remove_position(position.position_id)
                _run_model()
                st.success("Position removed from the schedule.")
            except ValueError as exc:
                st.error(str(exc))

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


def _production_horizon_table(cfg: CompanyConfig) -> pd.DataFrame:
    years = _config_years(cfg)
    records: List[Dict[str, Any]] = []

    for year in years:
        utilization = float(cfg.capacity_utilization.get(year, 0.0))
        utilization = max(0.0, min(1.0, utilization))
        projected_units = cfg.annual_capacity * utilization
        records.append(
            {
                "Year": year,
                "Capacity Utilization": f"{utilization * 100:.1f}%",
                "Projected Units": f"{projected_units:,.0f}",
            }
        )

    return pd.DataFrame(records, columns=["Year", "Capacity Utilization", "Projected Units"])


def _production_capacity_schedule(cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    model = model or {}
    years = _config_years(cfg)
    production_volume: Dict[int, float] = model.get("production_volume", {})

    records: List[Dict[str, Any]] = []
    for year in years:
        utilization = float(cfg.capacity_utilization.get(year, 0.0))
        utilization = max(0.0, min(1.0, utilization))
        annual_capacity = float(cfg.annual_capacity)
        planned_units = float(production_volume.get(year, annual_capacity * utilization))
        available_capacity = max(annual_capacity - planned_units, 0.0)
        records.append(
            {
                "Year": year,
                "Working Days": int(cfg.working_days),
                "Annual Capacity": f"{annual_capacity:,.0f}",
                "Utilization": f"{utilization * 100:.1f}%",
                "Planned Production": f"{planned_units:,.0f}",
                "Available Capacity": f"{available_capacity:,.0f}",
            }
        )

    columns = [
        "Year",
        "Working Days",
        "Annual Capacity",
        "Utilization",
        "Planned Production",
        "Available Capacity",
    ]
    return pd.DataFrame(records, columns=columns)


def _pricing_schedule(cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    model = model or {}
    years = _config_years(cfg)
    product_prices: Dict[int, Dict[str, float]] = model.get("product_prices", {})
    products = list(cfg.product_portfolio.keys())

    columns = ["Year"] + [f"{_product_label(product)} Price" for product in products]
    records: List[Dict[str, Any]] = []

    for year in years:
        row: Dict[str, Any] = {"Year": year}
        prices_for_year = product_prices.get(year, {})
        for product in products:
            row[f"{_product_label(product)} Price"] = float(prices_for_year.get(product, 0.0))
        records.append(row)

    df = pd.DataFrame(records, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns)

    for column in columns[1:]:
        df[column] = _currency_series(df[column])
    return df


def _revenue_schedule(cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    model = model or {}
    years = _config_years(cfg)
    product_revenue: Dict[int, Dict[str, float]] = model.get("product_revenue", {})
    total_revenue: Dict[int, float] = model.get("revenue", {})
    products = list(cfg.product_portfolio.keys())

    columns = ["Year"] + [f"{_product_label(product)} Revenue" for product in products] + [
        "Total Revenue"
    ]
    records: List[Dict[str, Any]] = []

    for year in years:
        row: Dict[str, Any] = {"Year": year}
        revenue_for_year = product_revenue.get(year, {})
        for product in products:
            row[f"{_product_label(product)} Revenue"] = float(revenue_for_year.get(product, 0.0))
        row["Total Revenue"] = float(total_revenue.get(year, 0.0))
        records.append(row)

    df = pd.DataFrame(records, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns)

    for column in columns[1:]:
        df[column] = _currency_series(df[column])
    return df


def _cost_structure_schedule(cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    model = model or {}
    years = _config_years(cfg)

    revenue_map: Dict[int, float] = model.get("revenue", {})
    variable_cogs: Dict[int, float] = model.get("variable_cogs", {})
    fixed_cogs: Dict[int, float] = model.get("fixed_cogs", {})
    opex: Dict[int, float] = model.get("opex", {})
    labor_metrics: Dict[int, Dict[str, float]] = model.get("labor_metrics", {})
    marketing_budget: Dict[int, float] = {}
    if isinstance(getattr(cfg, "marketing_budget", None), dict):
        marketing_budget = {int(year): float(value) for year, value in cfg.marketing_budget.items()}

    columns = [
        "Year",
        "Revenue",
        "Variable Production Cost",
        "Variable % of Revenue",
        "Fixed Manufacturing Cost",
        "Fixed % of Revenue",
        "Marketing Spend",
        "Marketing % of Revenue",
        "Labor Cost",
        "Labor % of Revenue",
        "Other Operating Cost",
        "Other % of Revenue",
        "Total Cost",
        "Total % of Revenue",
    ]

    def _percent(value: float, revenue: float) -> str:
        if revenue <= 0:
            return "—"
        return f"{(value / revenue) * 100:.1f}%"

    rows: List[Dict[str, Any]] = []
    for year in years:
        revenue_value = float(revenue_map.get(year, 0.0))
        variable_cost = float(variable_cogs.get(year, 0.0))
        fixed_cost = float(fixed_cogs.get(year, 0.0))

        if labor_metrics and year in labor_metrics:
            labor_cost = float(labor_metrics[year].get("total_labor_cost", 0.0))
        else:
            years_since_start = max(0, year - cfg.start_year)
            baseline_salary = cfg.avg_salary * cfg.headcount * 12
            labor_cost = baseline_salary * ((1 + cfg.annual_salary_growth) ** years_since_start)

        opex_total = float(opex.get(year, labor_cost))
        marketing_planned = float(marketing_budget.get(year, 0.0))
        marketing_cost = marketing_planned if marketing_planned > 0 else max(0.0, opex_total - labor_cost)
        other_opex = max(0.0, opex_total - labor_cost - marketing_cost)

        labor_cost = max(0.0, labor_cost)
        marketing_cost = max(0.0, marketing_cost)

        total_cost = variable_cost + fixed_cost + marketing_cost + labor_cost + other_opex

        rows.append(
            {
                "Year": year,
                "Revenue": f"${revenue_value:,.0f}",
                "Variable Production Cost": f"${variable_cost:,.0f}",
                "Variable % of Revenue": _percent(variable_cost, revenue_value),
                "Fixed Manufacturing Cost": f"${fixed_cost:,.0f}",
                "Fixed % of Revenue": _percent(fixed_cost, revenue_value),
                "Marketing Spend": f"${marketing_cost:,.0f}",
                "Marketing % of Revenue": _percent(marketing_cost, revenue_value),
                "Labor Cost": f"${labor_cost:,.0f}",
                "Labor % of Revenue": _percent(labor_cost, revenue_value),
                "Other Operating Cost": f"${other_opex:,.0f}",
                "Other % of Revenue": _percent(other_opex, revenue_value),
                "Total Cost": f"${total_cost:,.0f}",
                "Total % of Revenue": _percent(total_cost, revenue_value),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _operating_expense_schedule(cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    model = model or {}
    years = _config_years(cfg)

    revenue_map: Dict[int, float] = model.get("revenue", {})
    opex_map: Dict[int, float] = model.get("opex", {})
    labor_metrics: Dict[int, Dict[str, float]] = model.get("labor_metrics", {})

    marketing_budget: Dict[int, float] = {}
    if isinstance(getattr(cfg, "marketing_budget", None), dict):
        marketing_budget = {int(year): float(value) for year, value in cfg.marketing_budget.items()}

    columns = [
        "Year",
        "Marketing Spend",
        "Labor Cost",
        "Other Operating Expense",
        "Total Operating Expense",
        "Operating Expense % of Revenue",
    ]

    def _percent(value: float, revenue: float) -> str:
        if revenue <= 0:
            return "—"
        return f"{(value / revenue) * 100:.1f}%"

    rows: List[Dict[str, Any]] = []
    for year in years:
        revenue_value = float(revenue_map.get(year, 0.0))
        total_opex = float(opex_map.get(year, 0.0))

        if labor_metrics and year in labor_metrics:
            labor_cost = float(labor_metrics[year].get("total_labor_cost", 0.0))
        else:
            years_since_start = max(0, year - cfg.start_year)
            baseline_salary = cfg.avg_salary * cfg.headcount * 12
            labor_cost = baseline_salary * ((1 + cfg.annual_salary_growth) ** years_since_start)

        planned_marketing = float(marketing_budget.get(year, 0.0))
        marketing_cost = planned_marketing if planned_marketing > 0 else max(0.0, total_opex - labor_cost)

        labor_cost = max(0.0, labor_cost)
        marketing_cost = max(0.0, marketing_cost)

        other_opex = max(0.0, total_opex - labor_cost - marketing_cost)

        if total_opex <= 0.0:
            total_opex = labor_cost + marketing_cost + other_opex

        rows.append(
            {
                "Year": year,
                "Marketing Spend": f"${marketing_cost:,.0f}",
                "Labor Cost": f"${labor_cost:,.0f}",
                "Other Operating Expense": f"${other_opex:,.0f}",
                "Total Operating Expense": f"${total_opex:,.0f}",
                "Operating Expense % of Revenue": _percent(total_opex, revenue_value),
            }
        )

    return pd.DataFrame(rows, columns=columns)


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
    _render_add_position_form(manager, cfg)

    positions = manager.get_all_positions()
    if positions:
        st.markdown("#### Manage Existing Positions")
        for position in positions:
            _render_position_editor(position, manager, cfg)
    else:
        st.info("No labor positions configured. Use the form above to add one.")

    st.markdown("#### Labor Position Summary")
    summary = manager.get_position_summary()
    _render_table(summary)

    schedule = _build_labor_cost_schedule(manager, cfg)
    _display_schedule(
        "5-Year Labor Cost Schedule",
        schedule,
        "Labor cost schedule is shown even when no positions are configured.",
    )


def _render_add_capital_project_form(manager: CapexScheduleManager, cfg: CompanyConfig) -> None:
    st.markdown("#### Add Capital Project")
    with st.form("add_capex_form"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Project Name", key="add_capex_name")
        category = col2.text_input("Category", value="General", key="add_capex_category")

        col3, col4 = st.columns(2)
        amount = col3.number_input(
            "Capital Amount",
            min_value=0.0,
            value=250_000.0,
            step=25_000.0,
            key="add_capex_amount",
        )
        start_year = col4.number_input(
            "Start Year",
            value=int(cfg.start_year),
            step=1,
            key="add_capex_start_year",
        )

        col5, col6 = st.columns(2)
        useful_life = col5.number_input(
            "Useful Life (years)",
            min_value=0,
            value=10,
            step=1,
            key="add_capex_useful_life",
        )
        salvage_value = col6.number_input(
            "Salvage Value",
            min_value=0.0,
            value=0.0,
            step=5_000.0,
            key="add_capex_salvage",
        )

        spend_curve_input = st.text_input(
            "Spend Curve (offset:share, comma or newline separated)",
            key="add_capex_spend_curve",
            help="Example: '0:0.6, 1:0.4' allocates 60% in the start year and 40% the following year.",
        )
        notes = st.text_area("Notes", key="add_capex_notes")

        submitted = st.form_submit_button("Add Project")

    if submitted:
        try:
            spend_curve = _parse_spend_curve_input(spend_curve_input)
        except ValueError as exc:
            st.error(str(exc))
            return

        try:
            manager.add_item(
                name=name.strip() or "New Project",
                amount=float(amount),
                start_year=int(start_year),
                useful_life=int(useful_life),
                salvage_value=float(salvage_value),
                category=category.strip() or "General",
                notes=notes.strip(),
                spend_curve=spend_curve,
            )
            _run_model()
            st.success("Capital project added successfully.")
        except ValueError as exc:
            st.error(str(exc))


def _render_capex_item_editor(
    item: CapexItem, manager: CapexScheduleManager, cfg: CompanyConfig
) -> None:
    header = f"{item.item_id} — {item.name}"
    with st.expander(header, expanded=False):
        form_key = f"edit_capex_form_{item.item_id}"
        prefix = f"capex_{item.item_id}"
        with st.form(form_key):
            col1, col2 = st.columns(2)
            name = col1.text_input(
                "Project Name",
                value=item.name,
                key=f"{prefix}_name",
            )
            category = col2.text_input(
                "Category",
                value=item.category,
                key=f"{prefix}_category",
            )

            col3, col4 = st.columns(2)
            amount = col3.number_input(
                "Capital Amount",
                min_value=0.0,
                value=float(item.amount),
                step=25_000.0,
                key=f"{prefix}_amount",
            )
            start_year = col4.number_input(
                "Start Year",
                value=int(item.start_year),
                step=1,
                key=f"{prefix}_start_year",
            )

            col5, col6 = st.columns(2)
            useful_life = col5.number_input(
                "Useful Life (years)",
                min_value=0,
                value=int(item.useful_life),
                step=1,
                key=f"{prefix}_useful_life",
            )
            salvage_value = col6.number_input(
                "Salvage Value",
                min_value=0.0,
                value=float(item.salvage_value),
                step=5_000.0,
                key=f"{prefix}_salvage",
            )

            spend_curve_default = _format_spend_curve(item.spend_curve)
            spend_curve_input = st.text_input(
                "Spend Curve (offset:share, comma or newline separated)",
                value=spend_curve_default,
                key=f"{prefix}_spend_curve",
                help="Leave blank for single-year spend or specify offsets such as '0:0.7, 1:0.3'.",
            )
            notes = st.text_area(
                "Notes",
                value=item.notes,
                key=f"{prefix}_notes",
            )

            submitted = st.form_submit_button("Save Changes")

        if submitted:
            try:
                spend_curve = _parse_spend_curve_input(spend_curve_input)
            except ValueError as exc:
                st.error(str(exc))
            else:
                try:
                    manager.edit_item(
                        item.item_id,
                        name=name.strip() or item.name,
                        category=category.strip() or item.category,
                        amount=float(amount),
                        start_year=int(start_year),
                        useful_life=int(useful_life),
                        salvage_value=float(salvage_value),
                        notes=notes.strip(),
                        spend_curve=spend_curve,
                    )
                    _run_model()
                    st.success("Capital project updated successfully.")
                except ValueError as exc:
                    st.error(str(exc))

        remove = st.button(
            "Remove Project",
            key=f"remove_capex_{item.item_id}",
            help="Delete this capital project from the plan.",
        )
        if remove:
            try:
                manager.remove_item(item.item_id)
                _run_model()
                st.success("Capital project removed successfully.")
            except ValueError as exc:
                st.error(str(exc))


def _render_capex_management() -> None:
    manager: CapexScheduleManager = st.session_state["capex_manager"]
    cfg: CompanyConfig = st.session_state["company_config"]

    st.markdown("### Capital Projects")
    _render_add_capital_project_form(manager, cfg)

    items = sorted(manager.list_items(), key=lambda record: (record.start_year, record.item_id))
    if items:
        st.markdown("#### Manage Existing Projects")
        for item in items:
            _render_capex_item_editor(item, manager, cfg)
    else:
        st.info("No capital projects configured. Use the form above to add one.")

    items_data = [item.to_dict() for item in items]
    items_df = pd.DataFrame(items_data)
    if not items_df.empty:
        for column in ["amount", "salvage_value"]:
            if column in items_df.columns:
                items_df[column] = _currency_series(items_df[column])
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

    general_tab, labor_tab, capex_tab = st.tabs(
        ["General Settings", "Labor Management", "CAPEX Management"]
    )

    with general_tab:
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
            "Annual Salary Growth",
            min_value=0.0,
            max_value=0.15,
            value=float(cfg.annual_salary_growth),
            step=0.01,
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

        st.markdown("### Production Horizon")
        production_years = _config_years(cfg)
        horizon_form = st.form("production_horizon_form")
        with horizon_form:
            st.write(
                "Adjust annual capacity assumptions and the utilization ramp so production aligns with the forecast horizon."
            )
            annual_capacity = horizon_form.number_input(
                "Annual Capacity (units)",
                value=float(cfg.annual_capacity),
                min_value=0.0,
                step=100.0,
            )
            working_days = horizon_form.number_input(
                "Working Days per Year",
                value=int(cfg.working_days),
                min_value=1,
                max_value=366,
                step=1,
            )
            utilization_inputs: Dict[int, float] = {}
            for year in production_years:
                utilization_inputs[year] = horizon_form.slider(
                    f"Utilization {year}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg.capacity_utilization.get(year, 0.0)),
                    step=0.05,
                )
            save_horizon = horizon_form.form_submit_button("Save Production Horizon")

        if save_horizon:
            cfg.annual_capacity = float(annual_capacity)
            cfg.working_days = int(working_days)
            cfg.capacity_utilization = {
                year: float(utilization_inputs.get(year, 0.0)) for year in production_years
            }
            cfg.__post_init__()
            _run_model()
            st.success("Production horizon updated. Financial model refreshed.")

        st.markdown("#### Production Horizon Schedule")
        horizon_df = _production_horizon_table(cfg)
        _render_table(horizon_df, hide_index=True)

        st.markdown("#### Production Capacity Schedule")
        model: Dict[str, Any] = st.session_state.get("financial_model", {})
        capacity_df = _production_capacity_schedule(cfg, model)
        _render_table(capacity_df, hide_index=True)

        st.markdown("#### Pricing Schedule")
        pricing_df = _pricing_schedule(cfg, model)
        if pricing_df.empty:
            st.write("Run the financial model to populate product pricing across the horizon.")
        _render_table(pricing_df, hide_index=True)

        st.markdown("#### Revenue Schedule")
        revenue_df = _revenue_schedule(cfg, model)
        if revenue_df.empty:
            st.write("Run the financial model to populate revenue projections across the horizon.")
        _render_table(revenue_df, hide_index=True)

        st.markdown("#### Cost Structure Schedule")
        cost_df = _cost_structure_schedule(cfg, model)
        if cost_df.empty:
            st.write("Run the financial model to populate cost structure projections across the horizon.")
        _render_table(cost_df, hide_index=True)

        st.markdown("#### Operating Expense Schedule")
        opex_df = _operating_expense_schedule(cfg, model)
        if opex_df.empty:
            st.write("Run the financial model to populate operating expense projections across the horizon.")
        _render_table(opex_df, hide_index=True)

        st.markdown("#### Debt Amortization Schedule")
        debt_df = _debt_schedule(model)
        if debt_df.empty:
            st.write("Configure debt instruments in the financing settings to populate the schedule.")
        _render_table(debt_df, hide_index=True)

    with labor_tab:
        _render_labor_management()

    with capex_tab:
        _render_capex_management()


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
