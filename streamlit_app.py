"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Clean, professional interface with horizontal navigation and comprehensive schedules.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from scipy import optimize

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
from advanced_analytics import (
    AdvancedSensitivityAnalyzer,
    StressTestEngine,
    TrendDecomposition,
    SegmentationAnalyzer,
    MonteCarloSimulator,
    WhatIfAnalyzer,
    GoalSeekOptimizer,
    RegressionModeler,
    TimeSeriesAnalyzer,
    RiskAnalyzer,
    PortfolioOptimizer,
    RealOptionsAnalyzer,
    MacroeconomicLinker,
    ESGAnalyzer,
    ProbabilisticValuation,
)

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

def _horizon_end(cfg: CompanyConfig) -> int:
    return cfg.start_year + cfg.projection_years - 1


def _ensure_year_in_horizon(cfg: CompanyConfig, year: int) -> None:
    if year > _horizon_end(cfg):
        cfg.projection_years = year - cfg.start_year + 1


def _apply_increment(value: float, increment_pct: float) -> float:
    return value * (1 + increment_pct / 100.0)


def _schedule_toolbar(
    prefix: str,
    years: Sequence[int],
    *,
    label: str = "Schedule Actions",
    allow_add: bool = True,
    allow_remove: bool = True,
) -> Optional[Dict[str, Any]]:
    if not years:
        return None

    triggered: Optional[str] = None
    target_year: Optional[int] = None

    with st.expander(label, expanded=False):
        form_key = f"{prefix}_toolbar_form"
        with st.form(form_key):
            target_year = st.selectbox(
                "Target Year",
                options=list(years),
                key=f"{prefix}_toolbar_year",
            )

            buttons: List[Tuple[str, str]] = []
            if allow_add:
                buttons.append(("add", "Add Row"))
            if allow_remove:
                buttons.append(("remove", "Remove Row"))

            if buttons:
                cols = st.columns(len(buttons))
                for idx, (action_name, button_label) in enumerate(buttons):
                    if cols[idx].form_submit_button(button_label):
                        triggered = action_name

    if triggered and target_year is not None:
        return {"action": triggered, "year": int(target_year)}

    return None


def _schedule_increment_helper(
    prefix: str,
    years: Sequence[int],
    apply_callback: Optional[Callable[[Sequence[int], float]], Optional[str]] = None,
    *,
    label: str = "Yearly Increment Helper",
    help_text: Optional[str] = None,
) -> None:
    if not years or apply_callback is None:
        return

    with st.expander(label, expanded=False):
        if help_text:
            st.caption(help_text)
        with st.form(f"{prefix}_increment_form"):
            selected_years = st.multiselect(
                "Apply to Years",
                options=list(years),
                default=[years[-1]],
                key=f"{prefix}_increment_years",
            )
            increment_value = st.number_input(
                "Increment (%)",
                min_value=-100.0,
                max_value=500.0,
                value=0.0,
                step=1.0,
                key=f"{prefix}_increment_value",
            )
            apply = st.form_submit_button("Apply Increment")

        if apply:
            if not selected_years:
                st.warning("Select at least one year before applying an increment.")
            else:
                try:
                    message = apply_callback([int(year) for year in selected_years], float(increment_value))
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if message:
                        st.success(message)


def _schedule_table_editor(
    key: str,
    df: pd.DataFrame,
    *,
    guidance: Optional[str] = None,
    non_negative_columns: Sequence[str] = (),
    column_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, DeltaGenerator, List[str], bool]:
    if guidance:
        st.caption(guidance)

    editor_kwargs: Dict[str, Any] = {
        "num_rows": "dynamic",
        "use_container_width": True,
        "hide_index": True,
        "key": f"{key}_editor",
    }
    if column_config is not None:
        editor_kwargs["column_config"] = column_config

    edited_df = st.data_editor(df, **editor_kwargs)

    warnings: List[str] = []
    if "Year" in edited_df.columns:
        year_series = edited_df["Year"].dropna()
        if year_series.duplicated().any():
            warnings.append("Duplicate years detected. Ensure each projection year appears once.")
    for column in non_negative_columns:
        if column in edited_df.columns:
            values = edited_df[column].dropna()
            if len(values) > 0 and any(value < 0 for value in values):
                warnings.append(f"{column} cannot be negative.")

    status_placeholder = st.empty()
    has_changes = not edited_df.equals(df)
    if warnings:
        status_placeholder.warning(" • ".join(warnings))
    elif has_changes:
        status_placeholder.info("Changes pending – apply updates to refresh the model.")
    else:
        status_placeholder.caption("No edits pending.")

    return edited_df, status_placeholder, warnings, has_changes

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


def _working_capital_expense_schedule(
    cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    model = model or {}
    years = _config_years(cfg)

    revenue_map: Dict[int, float] = model.get("revenue", {})
    receivables_map: Dict[int, float] = model.get("accounts_receivable", {})
    inventory_map: Dict[int, float] = model.get("inventory", {})
    payables_map: Dict[int, float] = model.get("accounts_payable", {})
    accrued_map: Dict[int, float] = model.get("accrued_expenses", {})
    net_wc_map: Dict[int, float] = model.get("net_working_capital", {})
    change_map: Dict[int, float] = model.get("change_in_working_capital", {})

    columns = [
        "Year",
        "Accounts Receivable",
        "Inventory",
        "Accounts Payable",
        "Accrued Expenses",
        "Net Working Capital",
        "Change in Working Capital",
        "Net Working Capital % of Revenue",
    ]

    def _percent(value: float, revenue: float) -> str:
        if revenue <= 0:
            return "—"
        return f"{(value / revenue) * 100:.1f}%"

    rows: List[Dict[str, Any]] = []
    for year in years:
        revenue_value = float(revenue_map.get(year, 0.0))
        receivables = float(receivables_map.get(year, 0.0))
        inventory = float(inventory_map.get(year, 0.0))
        payables = float(payables_map.get(year, 0.0))
        accrued = float(accrued_map.get(year, 0.0))
        net_wc = float(net_wc_map.get(year, receivables + inventory - payables - accrued))
        change_wc = float(change_map.get(year, 0.0))

        rows.append(
            {
                "Year": year,
                "Accounts Receivable": receivables,
                "Inventory": inventory,
                "Accounts Payable": payables,
                "Accrued Expenses": accrued,
                "Net Working Capital": net_wc,
                "Change in Working Capital": change_wc,
                "Net Working Capital % of Revenue": _percent(net_wc, revenue_value),
            }
        )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    currency_columns = [
        "Accounts Receivable",
        "Inventory",
        "Accounts Payable",
        "Accrued Expenses",
        "Net Working Capital",
        "Change in Working Capital",
    ]
    for column in currency_columns:
        df[column] = _currency_series(df[column])

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


def _render_production_horizon_editor(cfg: CompanyConfig) -> None:
    years = _config_years(cfg)
    if not years:
        return

    st.markdown("##### Edit Production Horizon")

    action = _schedule_toolbar("production_horizon", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            cfg.capacity_utilization.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Production horizon entry removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            baseline = float(cfg.capacity_utilization.get(target_year, 0.0))
            cfg.capacity_utilization[new_year] = max(0.0, min(1.5, baseline))
            cfg.__post_init__()
            _run_model()
            st.success(f"Production horizon row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        utilization = float(cfg.capacity_utilization.get(year, 0.0))
        planned_units = float(cfg.annual_capacity) * utilization
        rows.append(
            {
                "Year": year,
                "Capacity Utilization %": round(utilization * 100, 2),
                "Annual Capacity": float(cfg.annual_capacity),
                "Working Days": int(cfg.working_days),
                "Planned Units": planned_units,
            }
        )

    base_df = pd.DataFrame(rows, columns=[
        "Year",
        "Capacity Utilization %",
        "Annual Capacity",
        "Working Days",
        "Planned Units",
    ])

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "production_horizon",
        base_df,
        guidance="Adjust utilization, capacity, or working days across the projection horizon.",
        non_negative_columns=["Capacity Utilization %", "Annual Capacity", "Working Days", "Planned Units"],
    )

    def _apply_production_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        for year in selected_years:
            baseline = float(cfg.capacity_utilization.get(year, 0.0))
            updated = max(0.0, min(1.5, _apply_increment(baseline, increment_pct)))
            cfg.capacity_utilization[year] = updated
        cfg.__post_init__()
        _run_model()
        return "Increment applied."

    _schedule_increment_helper(
        "production_horizon",
        years,
        _apply_production_increment,
        help_text="Apply a uniform percentage change to utilization across selected years.",
    )

    apply_label = "Apply Production Horizon Updates"
    if st.button(apply_label, key="production_horizon_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            years_list = sanitized["Year"].astype(int).tolist()
            if not years_list:
                raise ValueError("At least one year is required for the production horizon.")

            util_values = sanitized["Capacity Utilization %"].astype(float).tolist()
            annual_capacity_values = sanitized["Annual Capacity"].astype(float).tolist()
            working_days_values = sanitized["Working Days"].astype(int).tolist()

            if len(set(annual_capacity_values)) > 1:
                raise ValueError("Annual capacity must be consistent across the horizon.")
            if len(set(working_days_values)) > 1:
                raise ValueError("Working days must be consistent across the horizon.")

            new_capacity = float(annual_capacity_values[-1])
            new_working_days = int(working_days_values[-1])

            capacity_map: Dict[int, float] = {}
            for year_value, util_pct in zip(years_list, util_values):
                utilization_ratio = max(0.0, min(1.5, float(util_pct) / 100.0))
                capacity_map[int(year_value)] = utilization_ratio

            cfg.capacity_utilization = capacity_map
            cfg.annual_capacity = new_capacity
            cfg.working_days = new_working_days
            cfg.projection_years = max(years_list) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Production horizon updated.")



def _render_production_capacity_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    production_volume: Dict[int, float] = model.get("production_volume", {}) if model else {}

    st.markdown("##### Edit Production Capacity Targets")

    action = _schedule_toolbar("production_capacity", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            cfg.capacity_utilization.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Capacity target removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            baseline_units = production_volume.get(
                target_year,
                float(cfg.annual_capacity) * float(cfg.capacity_utilization.get(target_year, 0.0)),
            )
            if cfg.annual_capacity > 0:
                cfg.capacity_utilization[new_year] = max(
                    0.0, min(1.5, float(baseline_units) / float(cfg.annual_capacity))
                )
            cfg.__post_init__()
            _run_model()
            st.success(f"Capacity row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        planned_units = float(
            production_volume.get(
                year,
                float(cfg.annual_capacity) * float(cfg.capacity_utilization.get(year, 0.0)),
            )
        )
        utilization = float(cfg.capacity_utilization.get(year, 0.0))
        remaining = max(0.0, float(cfg.annual_capacity) - planned_units)
        rows.append(
            {
                "Year": year,
                "Planned Units": planned_units,
                "Implied Utilization %": round(utilization * 100, 2),
                "Remaining Headroom": remaining,
            }
        )

    column_config = {
        "Implied Utilization %": st.column_config.NumberColumn(
            "Implied Utilization %", format="%.2f", disabled=True
        ),
        "Remaining Headroom": st.column_config.NumberColumn(
            "Remaining Headroom", format="%.0f", disabled=True
        ),
    }

    base_df = pd.DataFrame(rows, columns=["Year", "Planned Units", "Implied Utilization %", "Remaining Headroom"])

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "production_capacity",
        base_df,
        guidance="Update planned production to reflect operational targets.",
        non_negative_columns=["Planned Units"],
        column_config=column_config,
    )

    def _apply_capacity_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        if cfg.annual_capacity <= 0:
            raise ValueError("Set an annual capacity before applying increments.")
        for year in selected_years:
            planned_units = float(
                production_volume.get(
                    year,
                    float(cfg.annual_capacity) * float(cfg.capacity_utilization.get(year, 0.0)),
                )
            )
            updated_units = max(0.0, _apply_increment(planned_units, increment_pct))
            cfg.capacity_utilization[year] = max(
                0.0, min(1.5, updated_units / float(cfg.annual_capacity))
            )
        cfg.__post_init__()
        _run_model()
        return "Capacity increments applied."

    _schedule_increment_helper(
        "production_capacity",
        years,
        _apply_capacity_increment,
        help_text="Increment adjusts planned production units and the implied utilization.",
    )

    if st.button("Apply Capacity Updates", key="production_capacity_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            years_list = sanitized["Year"].astype(int).tolist()
            if not years_list:
                raise ValueError("Add at least one year to maintain the capacity schedule.")

            planned_values = sanitized["Planned Units"].astype(float).tolist()
            capacity_map: Dict[int, float] = {}

            if cfg.annual_capacity <= 0:
                raise ValueError("Set a positive annual capacity before configuring targets.")

            for year_value, planned in zip(years_list, planned_values):
                utilization = 0.0
                if cfg.annual_capacity > 0:
                    utilization = max(0.0, min(1.5, float(planned) / float(cfg.annual_capacity)))
                capacity_map[int(year_value)] = utilization

            cfg.capacity_utilization.update(capacity_map)
            cfg.projection_years = max(years_list) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Capacity targets updated.")


def _render_pricing_schedule_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    products = list(cfg.product_portfolio.keys())
    product_prices: Dict[int, Dict[str, float]] = model.get("product_prices", {}) if model else {}

    if not products:
        return

    price_defaults: Dict[int, Dict[str, float]] = {}
    for year in years:
        overrides = {}
        if getattr(cfg, "product_price_overrides", None):
            overrides = cfg.product_price_overrides.get(year, {})
        prices_for_year = product_prices.get(year, {}) if product_prices else {}
        years_since_start = max(0, year - cfg.start_year)
        defaults: Dict[str, float] = {}
        for product in products:
            if overrides and product in overrides:
                defaults[product] = float(overrides[product])
            else:
                current_price = prices_for_year.get(product)
                if current_price is None:
                    base_price = cfg.product_portfolio.get(product, {}).get("price", 0.0)
                    growth = cfg.product_portfolio.get(product, {}).get("price_growth", 0.0)
                    current_price = base_price * ((1 + growth) ** years_since_start)
                defaults[product] = float(current_price)
        price_defaults[year] = defaults

    st.markdown("##### Edit Pricing Schedule")
    action = _schedule_toolbar("pricing_schedule", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "product_price_overrides", None):
                cfg.product_price_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Pricing row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            base_prices = price_defaults.get(target_year, {})
            cfg.product_price_overrides = cfg.product_price_overrides or {}
            cfg.product_price_overrides[new_year] = {
                product: float(value) for product, value in base_prices.items()
            }
            cfg.__post_init__()
            _run_model()
            st.success(f"Pricing row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        row = {"Year": year}
        for product in products:
            row[_product_label(product)] = price_defaults.get(year, {}).get(product, 0.0)
        rows.append(row)

    base_df = pd.DataFrame(rows)

    non_negative_columns = [_product_label(product) for product in products]

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "pricing_schedule",
        base_df,
        guidance="Edit product pricing directly in the table. Add or remove years using the toolbar above.",
        non_negative_columns=non_negative_columns,
    )

    def _apply_pricing_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.product_price_overrides = cfg.product_price_overrides or {}
        for year in selected_years:
            base_prices = price_defaults.get(year, {})
            if not base_prices:
                raise ValueError(f"No pricing data available for {year}.")
            cfg.product_price_overrides[year] = {
                product: max(0.0, _apply_increment(value, increment_pct))
                for product, value in base_prices.items()
            }
        cfg.__post_init__()
        _run_model()
        return "Pricing increments applied."

    _schedule_increment_helper(
        "pricing_schedule",
        years,
        _apply_pricing_increment,
        help_text="Apply a uniform percentage change to prices across selected years.",
    )

    if st.button("Apply Pricing Updates", key="pricing_schedule_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying pricing updates.")

            overrides: Dict[int, Dict[str, float]] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                prices: Dict[str, float] = {}
                for product in products:
                    label = _product_label(product)
                    value = float(row.get(label, 0.0))
                    if value < 0:
                        raise ValueError("Prices cannot be negative.")
                    prices[product] = value
                overrides[year_value] = prices

            cfg.product_price_overrides = overrides
            cfg.projection_years = max(overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Pricing schedule updated.")


def _render_revenue_schedule_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    product_units: Dict[int, Dict[str, float]] = model.get("product_units", {}) if model else {}
    products = list(cfg.product_portfolio.keys())

    if not products:
        return

    unit_defaults: Dict[int, Dict[str, float]] = {}
    for year in years:
        overrides = {}
        if getattr(cfg, "product_unit_overrides", None):
            overrides = cfg.product_unit_overrides.get(year, {})
        units_for_year = product_units.get(year, {}) if product_units else {}
        defaults: Dict[str, float] = {}
        for product in products:
            if overrides and product in overrides:
                defaults[product] = float(overrides[product])
            else:
                defaults[product] = float(units_for_year.get(product, 0.0))
        unit_defaults[year] = defaults

    st.markdown("##### Edit Revenue Schedule")
    action = _schedule_toolbar("revenue_schedule", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "product_unit_overrides", None):
                cfg.product_unit_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Revenue row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            base_units = unit_defaults.get(target_year, {})
            cfg.product_unit_overrides = cfg.product_unit_overrides or {}
            cfg.product_unit_overrides[new_year] = {
                product: float(value) for product, value in base_units.items()
            }
            cfg.__post_init__()
            _run_model()
            st.success(f"Revenue row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        row = {"Year": year}
        for product in products:
            row[_product_label(product)] = unit_defaults.get(year, {}).get(product, 0.0)
        rows.append(row)

    base_df = pd.DataFrame(rows)
    non_negative_columns = [_product_label(product) for product in products]

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "revenue_schedule",
        base_df,
        guidance="Adjust product volumes to influence revenue calculations.",
        non_negative_columns=non_negative_columns,
    )

    def _apply_revenue_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.product_unit_overrides = cfg.product_unit_overrides or {}
        for year in selected_years:
            base_units = unit_defaults.get(year, {})
            if not base_units:
                raise ValueError(f"No unit data available for {year}.")
            cfg.product_unit_overrides[year] = {
                product: max(0.0, _apply_increment(value, increment_pct))
                for product, value in base_units.items()
            }
        cfg.__post_init__()
        _run_model()
        return "Revenue volume increments applied."

    _schedule_increment_helper(
        "revenue_schedule",
        years,
        _apply_revenue_increment,
        help_text="Apply a uniform percentage change to unit volumes across selected years.",
    )

    if st.button("Apply Revenue Updates", key="revenue_schedule_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying revenue updates.")

            overrides: Dict[int, Dict[str, float]] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                units: Dict[str, float] = {}
                for product in products:
                    label = _product_label(product)
                    value = float(row.get(label, 0.0))
                    if value < 0:
                        raise ValueError("Units cannot be negative.")
                    units[product] = value
                overrides[year_value] = units

            cfg.product_unit_overrides = overrides
            cfg.projection_years = max(overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Revenue schedule updated.")


def _render_cost_structure_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    variable_cogs: Dict[int, float] = model.get("variable_cogs", {}) if model else {}
    fixed_cogs: Dict[int, float] = model.get("fixed_cogs", {}) if model else {}

    cost_defaults: Dict[int, Dict[str, float]] = {}
    for year in years:
        variable_default = float(variable_cogs.get(year, 0.0))
        fixed_default = float(fixed_cogs.get(year, 0.0))
        marketing_default = float(cfg.marketing_budget.get(year, 0.0))
        cost_defaults[year] = {
            "variable": variable_default,
            "fixed": fixed_default,
            "marketing": marketing_default,
        }

    st.markdown("##### Edit Cost Structure")
    action = _schedule_toolbar("cost_structure", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "variable_cost_overrides", None):
                cfg.variable_cost_overrides.pop(target_year, None)
            if getattr(cfg, "fixed_cost_overrides", None):
                cfg.fixed_cost_overrides.pop(target_year, None)
            if getattr(cfg, "marketing_budget", None):
                cfg.marketing_budget.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Cost structure row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            defaults = cost_defaults.get(target_year, {"variable": 0.0, "fixed": 0.0, "marketing": 0.0})
            cfg.variable_cost_overrides = cfg.variable_cost_overrides or {}
            cfg.fixed_cost_overrides = cfg.fixed_cost_overrides or {}
            cfg.marketing_budget = cfg.marketing_budget or {}
            cfg.variable_cost_overrides[new_year] = float(defaults.get("variable", 0.0))
            cfg.fixed_cost_overrides[new_year] = float(defaults.get("fixed", 0.0))
            cfg.marketing_budget[new_year] = float(defaults.get("marketing", 0.0))
            cfg.__post_init__()
            _run_model()
            st.success(f"Cost structure row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        defaults = cost_defaults.get(year, {"variable": 0.0, "fixed": 0.0, "marketing": 0.0})
        rows.append(
            {
                "Year": year,
                "Variable Production Cost": defaults.get("variable", 0.0),
                "Fixed Manufacturing Cost": defaults.get("fixed", 0.0),
                "Marketing Spend": defaults.get("marketing", 0.0),
            }
        )

    base_df = pd.DataFrame(rows)
    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "cost_structure",
        base_df,
        guidance="Maintain consistency between variable, fixed, and marketing costs across years.",
        non_negative_columns=["Variable Production Cost", "Fixed Manufacturing Cost", "Marketing Spend"],
    )

    def _apply_cost_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.variable_cost_overrides = cfg.variable_cost_overrides or {}
        cfg.fixed_cost_overrides = cfg.fixed_cost_overrides or {}
        cfg.marketing_budget = cfg.marketing_budget or {}
        for year in selected_years:
            defaults = cost_defaults.get(year, {"variable": 0.0, "fixed": 0.0, "marketing": 0.0})
            cfg.variable_cost_overrides[year] = max(
                0.0, _apply_increment(defaults.get("variable", 0.0), increment_pct)
            )
            cfg.fixed_cost_overrides[year] = max(
                0.0, _apply_increment(defaults.get("fixed", 0.0), increment_pct)
            )
            cfg.marketing_budget[year] = max(
                0.0, _apply_increment(defaults.get("marketing", 0.0), increment_pct)
            )
        cfg.__post_init__()
        _run_model()
        return "Cost structure increments applied."

    _schedule_increment_helper(
        "cost_structure",
        years,
        _apply_cost_increment,
        help_text="Apply a uniform percentage change to variable, fixed, and marketing costs.",
    )

    if st.button("Apply Cost Structure Updates", key="cost_structure_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying cost structure updates.")

            variable_overrides: Dict[int, float] = {}
            fixed_overrides: Dict[int, float] = {}
            marketing_overrides: Dict[int, float] = {}

            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                variable = float(row.get("Variable Production Cost", 0.0))
                fixed = float(row.get("Fixed Manufacturing Cost", 0.0))
                marketing = float(row.get("Marketing Spend", 0.0))
                if variable < 0 or fixed < 0 or marketing < 0:
                    raise ValueError("Cost values cannot be negative.")
                variable_overrides[year_value] = variable
                fixed_overrides[year_value] = fixed
                marketing_overrides[year_value] = marketing

            cfg.variable_cost_overrides = variable_overrides
            cfg.fixed_cost_overrides = fixed_overrides
            cfg.marketing_budget = marketing_overrides
            cfg.projection_years = max(variable_overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Cost structure updated.")


def _render_operating_expense_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    opex_defaults: Dict[int, Dict[str, float]] = {}
    for year in years:
        marketing_default = float(cfg.marketing_budget.get(year, 0.0))
        other_default = 0.0
        if cfg.other_opex_overrides:
            other_default = float(cfg.other_opex_overrides.get(year, 0.0))
        opex_defaults[year] = {"marketing": marketing_default, "other": other_default}

    st.markdown("##### Edit Operating Expenses")
    action = _schedule_toolbar("operating_expense", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "marketing_budget", None):
                cfg.marketing_budget.pop(target_year, None)
            if getattr(cfg, "other_opex_overrides", None):
                cfg.other_opex_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Operating expense row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            defaults = opex_defaults.get(target_year, {"marketing": 0.0, "other": 0.0})
            cfg.marketing_budget = cfg.marketing_budget or {}
            cfg.other_opex_overrides = cfg.other_opex_overrides or {}
            cfg.marketing_budget[new_year] = float(defaults.get("marketing", 0.0))
            cfg.other_opex_overrides[new_year] = float(defaults.get("other", 0.0))
            cfg.__post_init__()
            _run_model()
            st.success(f"Operating expense row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        defaults = opex_defaults.get(year, {"marketing": 0.0, "other": 0.0})
        rows.append(
            {
                "Year": year,
                "Marketing Spend": defaults.get("marketing", 0.0),
                "Other Operating Expense": defaults.get("other", 0.0),
            }
        )

    base_df = pd.DataFrame(rows)
    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "operating_expense",
        base_df,
        guidance="Manage marketing and other operating expense assumptions in one place.",
        non_negative_columns=["Marketing Spend", "Other Operating Expense"],
    )

    def _apply_opex_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.marketing_budget = cfg.marketing_budget or {}
        cfg.other_opex_overrides = cfg.other_opex_overrides or {}
        for year in selected_years:
            defaults = opex_defaults.get(year, {"marketing": 0.0, "other": 0.0})
            cfg.marketing_budget[year] = max(
                0.0, _apply_increment(defaults.get("marketing", 0.0), increment_pct)
            )
            cfg.other_opex_overrides[year] = max(
                0.0, _apply_increment(defaults.get("other", 0.0), increment_pct)
            )
        cfg.__post_init__()
        _run_model()
        return "Operating expense increments applied."

    _schedule_increment_helper(
        "operating_expense",
        years,
        _apply_opex_increment,
        help_text="Apply a uniform percentage change to marketing and other operating expenses.",
    )

    if st.button("Apply Operating Expense Updates", key="operating_expense_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying operating expense updates.")

            marketing_overrides: Dict[int, float] = {}
            other_overrides: Dict[int, float] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                marketing_value = float(row.get("Marketing Spend", 0.0))
                other_value = float(row.get("Other Operating Expense", 0.0))
                if marketing_value < 0 or other_value < 0:
                    raise ValueError("Operating expenses cannot be negative.")
                marketing_overrides[year_value] = marketing_value
                other_overrides[year_value] = other_value

            cfg.marketing_budget = marketing_overrides
            cfg.other_opex_overrides = other_overrides
            cfg.projection_years = max(marketing_overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Operating expenses updated.")


def _render_working_capital_editor(cfg: CompanyConfig) -> None:
    years = _config_years(cfg)
    if not years:
        return

    base_wc_values = {
        "receivable_days": float(cfg.receivable_days),
        "inventory_days": float(cfg.inventory_days),
        "payable_days": float(cfg.payable_days),
        "accrued_expense_ratio": float(cfg.accrued_expense_ratio),
    }

    wc_defaults: Dict[int, Dict[str, float]] = {}
    for year in years:
        base_values = dict(base_wc_values)
        if cfg.working_capital_overrides:
            overrides = cfg.working_capital_overrides.get(year, {})
            base_values.update({
                key: float(value)
                for key, value in overrides.items()
                if key in base_values
            })
        wc_defaults[year] = base_values

    st.markdown("##### Edit Working Capital Drivers")
    action = _schedule_toolbar("working_capital", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "working_capital_overrides", None):
                cfg.working_capital_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Working capital row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            defaults = wc_defaults.get(target_year, base_wc_values)
            cfg.working_capital_overrides = cfg.working_capital_overrides or {}
            cfg.working_capital_overrides[new_year] = {
                "receivable_days": float(defaults.get("receivable_days", 0.0)),
                "inventory_days": float(defaults.get("inventory_days", 0.0)),
                "payable_days": float(defaults.get("payable_days", 0.0)),
                "accrued_expense_ratio": float(defaults.get("accrued_expense_ratio", 0.0)),
            }
            cfg.__post_init__()
            _run_model()
            st.success(f"Working capital row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        defaults = wc_defaults.get(year, base_wc_values)
        rows.append(
            {
                "Year": year,
                "Receivable Days": defaults.get("receivable_days", cfg.receivable_days),
                "Inventory Days": defaults.get("inventory_days", cfg.inventory_days),
                "Payable Days": defaults.get("payable_days", cfg.payable_days),
                "Accrued Expense Ratio": defaults.get("accrued_expense_ratio", cfg.accrued_expense_ratio),
            }
        )

    base_df = pd.DataFrame(rows)

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "working_capital",
        base_df,
        guidance="Adjust receivable, inventory, payable days, and accrued-expense ratios by year.",
        non_negative_columns=["Receivable Days", "Inventory Days", "Payable Days", "Accrued Expense Ratio"],
    )

    def _apply_wc_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.working_capital_overrides = cfg.working_capital_overrides or {}
        for year in selected_years:
            defaults = wc_defaults.get(year, base_wc_values)
            cfg.working_capital_overrides[year] = {
                "receivable_days": max(0.0, _apply_increment(defaults.get("receivable_days", 0.0), increment_pct)),
                "inventory_days": max(0.0, _apply_increment(defaults.get("inventory_days", 0.0), increment_pct)),
                "payable_days": max(0.0, _apply_increment(defaults.get("payable_days", 0.0), increment_pct)),
                "accrued_expense_ratio": max(
                    0.0, _apply_increment(defaults.get("accrued_expense_ratio", 0.0), increment_pct)
                ),
            }
        cfg.__post_init__()
        _run_model()
        return "Working capital increments applied."

    _schedule_increment_helper(
        "working_capital",
        years,
        _apply_wc_increment,
        help_text="Apply percentage changes to working-capital drivers across selected years.",
    )

    if st.button("Apply Working Capital Updates", key="working_capital_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying working-capital updates.")

            overrides: Dict[int, Dict[str, float]] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                receivable = float(row.get("Receivable Days", 0.0))
                inventory = float(row.get("Inventory Days", 0.0))
                payable = float(row.get("Payable Days", 0.0))
                accrued = float(row.get("Accrued Expense Ratio", 0.0))
                if receivable < 0 or inventory < 0 or payable < 0 or accrued < 0:
                    raise ValueError("Working-capital drivers cannot be negative.")
                overrides[year_value] = {
                    "receivable_days": receivable,
                    "inventory_days": inventory,
                    "payable_days": payable,
                    "accrued_expense_ratio": accrued,
                }

            cfg.working_capital_overrides = overrides
            cfg.projection_years = max(overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Working capital drivers updated.")


def _render_debt_schedule_editor(cfg: CompanyConfig) -> None:
    years = _config_years(cfg)
    instruments = list(getattr(cfg, "debt_instruments", []) or [])

    if not years or not instruments:
        return

    draw_defaults: Dict[int, Dict[int, float]] = {}
    for year in years:
        draw_defaults[year] = {
            idx: float(instrument.draw_schedule.get(year, 0.0))
            for idx, instrument in enumerate(instruments)
        }

    st.markdown("##### Edit Debt Schedule")
    action = _schedule_toolbar("debt_schedule", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            for instrument in instruments:
                instrument.draw_schedule.pop(target_year, None)
                instrument.__post_init__()
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.debt_instruments = instruments
            cfg.__post_init__()
            _run_model()
            st.success(f"Debt draws removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = target_year + 1
            _ensure_year_in_horizon(cfg, new_year)
            for idx, instrument in enumerate(instruments):
                base_draw = draw_defaults.get(target_year, {}).get(idx, 0.0)
                instrument.draw_schedule[new_year] = float(base_draw)
                instrument.__post_init__()
            cfg.debt_instruments = instruments
            cfg.__post_init__()
            _run_model()
            st.success(f"Debt schedule row added for {new_year}.")
            return

    column_labels: Dict[int, str] = {}
    rows: List[Dict[str, Any]] = []
    for idx, instrument in enumerate(instruments):
        label = f"{instrument.name or f'Instrument {idx + 1}'} Draw"
        column_labels[idx] = label

    for year in years:
        row: Dict[str, Any] = {"Year": year}
        for idx, instrument in enumerate(instruments):
            label = column_labels[idx]
            row[label] = float(instrument.draw_schedule.get(year, 0.0))
        rows.append(row)

    base_df = pd.DataFrame(rows)

    non_negative_columns = list(column_labels.values())

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "debt_schedule",
        base_df,
        guidance="Edit annual draw amounts for each instrument. Use the toolbar to manage projection years.",
        non_negative_columns=non_negative_columns,
    )

    def _apply_debt_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        for idx, instrument in enumerate(instruments):
            for year in selected_years:
                baseline = instrument.draw_schedule.get(
                    year, draw_defaults.get(year, {}).get(idx, 0.0)
                )
                instrument.draw_schedule[year] = max(0.0, _apply_increment(float(baseline), increment_pct))
            instrument.__post_init__()
        cfg.debt_instruments = instruments
        cfg.__post_init__()
        _run_model()
        return "Debt draw increments applied."

    _schedule_increment_helper(
        "debt_schedule",
        years,
        _apply_debt_increment,
        help_text="Apply percentage changes to all instrument draws in selected years.",
    )

    if st.button("Apply Debt Schedule Updates", key="debt_schedule_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying debt schedule updates.")

            updated_years = sanitized["Year"].astype(int).tolist()
            for idx, instrument in enumerate(instruments):
                label = column_labels[idx]
                values = sanitized[label].astype(float).tolist() if label in sanitized.columns else [0.0] * len(updated_years)
                instrument.draw_schedule = {
                    int(year): max(0.0, float(value)) for year, value in zip(updated_years, values)
                }
                instrument.__post_init__()

            cfg.debt_instruments = instruments
            cfg.projection_years = max(updated_years) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Debt schedule updated.")

    with st.expander("Instrument Parameters", expanded=False):
        st.caption("Update instrument definitions below. Draw amounts are managed in the schedule above.")
        for idx, instrument in enumerate(instruments):
            form_key = f"instrument_form_{idx}"
            with st.form(form_key):
                name = st.text_input(
                    f"Instrument {idx + 1} Name", value=instrument.name, key=f"instrument_name_{idx}"
                )
                principal = st.number_input(
                    "Principal",
                    min_value=0.0,
                    value=float(instrument.principal),
                    step=1000.0,
                    key=f"instrument_principal_{idx}",
                )
                rate = st.number_input(
                    "Interest Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(instrument.interest_rate),
                    step=0.001,
                    format="%.4f",
                    key=f"instrument_rate_{idx}",
                )
                term = st.number_input(
                    "Term (years)",
                    min_value=1,
                    max_value=50,
                    value=int(max(1, instrument.term)),
                    step=1,
                    key=f"instrument_term_{idx}",
                )
                interest_only = st.number_input(
                    "Interest-Only Years",
                    min_value=0,
                    max_value=50,
                    value=int(max(0, instrument.interest_only_years)),
                    step=1,
                    key=f"instrument_io_{idx}",
                )
                start_year_min = cfg.start_year
                start_year_max = cfg.start_year + cfg.projection_years - 1
                start_year_default = int(instrument.start_year)
                if start_year_default < start_year_min or start_year_default > start_year_max:
                    start_year_default = start_year_min

                start_year = st.number_input(
                    "Start Year",
                    min_value=start_year_min,
                    max_value=start_year_max,
                    value=start_year_default,
                    step=1,
                    key=f"instrument_start_{idx}",
                )
                saved = st.form_submit_button("Save Instrument")

            if saved:
                instrument.name = name.strip() or instrument.name
                instrument.principal = float(principal)
                instrument.interest_rate = float(rate)
                instrument.term = int(term)
                instrument.interest_only_years = int(interest_only)
                instrument.start_year = int(start_year)
                instrument.__post_init__()
                cfg.debt_instruments[idx] = instrument
                cfg.__post_init__()
                _run_model()
                st.success(f"Instrument {idx + 1} updated.")

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

def _render_advanced_analytics(model: Dict[str, Any]) -> None:
    cfg: CompanyConfig = st.session_state["company_config"]
    years = _projection_years(model)
    if not years:
        st.info("Run the financial model to explore advanced analytics.")
        return

    st.markdown("### Advanced Analytics Workbench")
    st.write(
        "Stress test the plan, explore probabilistic outcomes, and connect macro drivers "
        "to valuation using the tools below."
    )

    base_ev = float(model.get("enterprise_value", 0.0))
    final_year = years[-1]

    sensitivity_analyzer = AdvancedSensitivityAnalyzer(model, cfg)
    stress_engine = StressTestEngine(model, cfg)
    trend_analyzer = TrendDecomposition(model)
    segmentation_analyzer = SegmentationAnalyzer(model)
    monte_carlo = MonteCarloSimulator(model, cfg, num_simulations=500)
    what_if_analyzer = WhatIfAnalyzer(cfg)
    goal_seeker = GoalSeekOptimizer(cfg)
    ts_analyzer = TimeSeriesAnalyzer()
    risk_analyzer = RiskAnalyzer()
    macro_linker = MacroeconomicLinker()
    esg_analyzer = ESGAnalyzer()
    real_options = RealOptionsAnalyzer()

    product_units_all = model.get("product_units", {})
    product_revenue_all = model.get("product_revenue", {})
    variable_cogs_all = model.get("variable_cogs_breakdown", {})

    units_year = product_units_all.get(final_year, {})
    revenue_year = product_revenue_all.get(final_year, {})
    cogs_year = variable_cogs_all.get(final_year, {})

    segment_payload: Dict[str, Dict[str, float]] = {}
    for product_key, revenue_value in sorted(revenue_year.items()):
        label = _product_label(product_key)
        units = units_year.get(product_key, 0.0)
        cost_value = cogs_year.get(product_key, 0.0)
        segment_payload[label] = {
            "revenue": float(revenue_value),
            "cost": float(cost_value),
            "units": float(units) if units else 1.0,
        }

    segment_df = segmentation_analyzer.segment_analysis(segment_payload) if segment_payload else pd.DataFrame()

    mc_params = {
        "cogs_ratio": ("normal", float(cfg.cogs_ratio), max(0.01, float(cfg.cogs_ratio) * 0.05)),
        "wacc": ("normal", float(cfg.wacc), 0.02),
        "tax_rate": ("normal", float(cfg.tax_rate), 0.015),
    }
    mc_stats = monte_carlo.run_simulation(mc_params)
    ev_distribution = mc_stats.get("enterprise_values", {}).get("distribution", [])
    profit_distribution = mc_stats.get("final_profits", {}).get("distribution", [])

    returns_matrix = np.array([])
    product_returns: Dict[str, List[float]] = {}
    for product_key in revenue_year:
        series: List[float] = []
        previous = None
        for year in years:
            revenue_value = product_revenue_all.get(year, {}).get(product_key, 0.0)
            if previous is not None and previous:
                series.append((revenue_value - previous) / previous)
            previous = revenue_value
        if series:
            product_returns[_product_label(product_key)] = series

    if product_returns:
        returns_matrix = np.array(list(zip(*product_returns.values())))

    tabs = st.tabs([
        "Drivers & Scenarios",
        "Forecast Intelligence",
        "Risk & Portfolio",
        "Strategic & Valuation",
    ])

    with tabs[0]:
        st.subheader("Sensitivity analysis to quantify key driver impacts")
        range_cols = st.columns(4)
        cogs_range = range_cols[0].slider("COGS shock (%)", 5, 40, 20, step=1)
        tax_range = range_cols[1].slider("Tax shock (%)", 5, 30, 10, step=1)
        wacc_range = range_cols[2].slider("WACC shock (%)", 5, 30, 15, step=1)
        capacity_range = range_cols[3].slider("Capacity shock (%)", 5, 50, 20, step=1)

        sensitivity_ranges = {
            "cogs_ratio": cogs_range / 100.0,
            "tax_rate": tax_range / 100.0,
            "wacc": wacc_range / 100.0,
            "annual_capacity": capacity_range / 100.0,
        }
        sensitivity_df = sensitivity_analyzer.pareto_sensitivity(
            ["cogs_ratio", "tax_rate", "wacc", "annual_capacity"], sensitivity_ranges
        )
        if sensitivity_df.empty:
            st.write("Sensitivity results will appear once the model returns baseline metrics.")
        else:
            display = sensitivity_df[
                ["parameter", "impact_pct", "ev_range", "low_ev", "high_ev", "elasticity"]
            ].copy()
            display.rename(columns={
                "parameter": "Parameter",
                "impact_pct": "Impact %",
                "ev_range": "EV Range",
                "low_ev": "Low EV",
                "high_ev": "High EV",
                "elasticity": "Elasticity",
            }, inplace=True)
            _render_table(display, hide_index=True)

        st.subheader("Scenario stress testing for severe shocks")
        stress_results = stress_engine.extreme_scenarios()
        stress_rows = []
        for name, metrics in stress_results.items():
            stress_rows.append(
                {
                    "Scenario": name,
                    "Enterprise Value": metrics.get("enterprise_value", 0.0),
                    "Revenue": metrics.get("revenue_2030", metrics.get("revenue", 0.0)),
                    "Net Profit": metrics.get("net_profit_2030", 0.0),
                    "Closing Cash": metrics.get("final_cash", 0.0),
                    "Recovery Score": metrics.get("recovery_probability", 0.0),
                }
            )
        _render_table(pd.DataFrame(stress_rows), hide_index=True)

        st.subheader("Monte Carlo simulation for probabilistic outcomes")
        mc_rows = []
        for metric, stats_dict in mc_stats.items():
            if not isinstance(stats_dict, dict):
                continue
            mc_rows.append(
                {
                    "Metric": metric.replace("_", " ").title(),
                    "Mean": stats_dict.get("mean", 0.0),
                    "Std Dev": stats_dict.get("std_dev", 0.0),
                    "P5": stats_dict.get("p5", 0.0),
                    "P50": stats_dict.get("median", 0.0),
                    "P95": stats_dict.get("p95", 0.0),
                }
            )
        _render_table(pd.DataFrame(mc_rows), hide_index=True)

        st.subheader("What-if analysis to adjust assumptions interactively")
        with st.form("what_if_form"):
            cogs_override = st.slider(
                "Revised COGS ratio",
                min_value=0.30,
                max_value=0.95,
                value=float(cfg.cogs_ratio),
                step=0.01,
            )
            capacity_shift = st.slider(
                "Capacity change (%)",
                min_value=-25,
                max_value=50,
                value=0,
                step=1,
            )
            wacc_shift = st.slider(
                "WACC change (%)",
                min_value=-20,
                max_value=30,
                value=0,
                step=1,
            )
            run_scenario = st.form_submit_button("Run scenario")

        if run_scenario:
            adjustments = {
                "cogs_ratio": cogs_override,
                "annual_capacity": float(cfg.annual_capacity) * (1 + capacity_shift / 100.0),
                "wacc": float(cfg.wacc) * (1 + wacc_shift / 100.0),
            }
            scenario = what_if_analyzer.create_scenario("Custom What-If", adjustments)
            scenario_df = pd.DataFrame([
                {
                    "Scenario": scenario["scenario_name"],
                    "Enterprise Value": scenario["enterprise_value"],
                    "EV Change": scenario["ev_change"],
                    "EV Change %": scenario["ev_change_pct"],
                    "Final-Year Revenue": scenario.get("revenue_2030", scenario.get("revenue", 0.0)),
                    "Final-Year Profit": scenario.get("profit_2030", 0.0),
                }
            ])
            _render_table(scenario_df, hide_index=True)

        st.subheader("Goal seek routines to meet profitability targets")
        goal_cols = st.columns(2)
        target_metric_label = goal_cols[0].selectbox(
            "Target metric",
            ["enterprise_value", "net_profit"],
            format_func=lambda key: "Enterprise Value" if key == "enterprise_value" else "Net Profit",
        )
        target_value = goal_cols[1].number_input(
            "Target value", value=float(base_ev) if base_ev else 500_000_000.0, step=50_000_000.0
        )
        parameter_label = st.selectbox(
            "Parameter to solve",
            ["cogs_ratio", "wacc", "annual_capacity"],
            format_func=lambda key: {
                "cogs_ratio": "COGS Ratio",
                "wacc": "WACC",
                "annual_capacity": "Annual Capacity",
            }[key],
        )
        if st.button("Run goal seek"):
            result = goal_seeker.find_breakeven_parameter(parameter_label, target_metric_label, target_value)
            _render_table(pd.DataFrame([result]), hide_index=True)

        st.subheader("Tornado charts & spider diagrams for driver visualization")
        if not sensitivity_df.empty:
            tornado_data = []
            for _, row in sensitivity_df.iterrows():
                tornado_data.append(
                    {
                        "Parameter": row["parameter"],
                        "Downside": row["low_ev"] - base_ev,
                        "Upside": row["high_ev"] - base_ev,
                    }
                )
            tornado_df = pd.DataFrame(tornado_data)
            if not tornado_df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        y=tornado_df["Parameter"],
                        x=tornado_df["Upside"],
                        name="Upside",
                        orientation="h",
                        marker_color="#2ca02c",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        y=tornado_df["Parameter"],
                        x=tornado_df["Downside"],
                        name="Downside",
                        orientation="h",
                        marker_color="#d62728",
                    )
                )
                fig.update_layout(barmode="overlay", xaxis_title="Delta vs. Base EV", yaxis_title="Parameter")
                st.plotly_chart(fig, use_container_width=True)

        if stress_rows:
            categories = [
                "Enterprise Value",
                "Net Profit",
                "Closing Cash",
                "Recovery Score",
            ]
            spider_fig = go.Figure()
            for row in stress_rows:
                values = [
                    max(0.0, row["Enterprise Value"] / max(base_ev, 1.0) * 100.0),
                    max(0.0, row["Net Profit"] / 1e6),
                    max(0.0, row["Closing Cash"] / 1e6),
                    row["Recovery Score"],
                ]
                spider_fig.add_trace(
                    go.Scatterpolar(r=values, theta=categories, fill="toself", name=row["Scenario"])
                )
            spider_fig.update_layout(showlegend=True)
            st.plotly_chart(spider_fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Trend and seasonality decomposition")
        revenue_series = {year: model.get("revenue", {}).get(year, 0.0) for year in years}
        decomposition = trend_analyzer.decompose_series(revenue_series)
        trend_df = pd.DataFrame(
            {
                "Year": decomposition["years"],
                "Original": decomposition["original"],
                "Trend": decomposition["trend"],
                "Seasonality": decomposition["seasonality"],
            }
        )
        _render_table(trend_df, hide_index=True)
        inflections = trend_analyzer.identify_inflection_points(revenue_series)
        if inflections:
            st.write(f"Inflection years: {', '.join(str(year) for year in inflections)}")

        st.subheader("Customer and product segmentation analysis")
        _render_table(segment_df, hide_index=True)

        st.subheader("Regression modeling for financial relationships")
        revenue_values = [revenue_series[year] for year in years]
        net_profit_series = [model.get("net_profit", {}).get(year, 0.0) for year in years]
        regression = RegressionModeler.simple_linear_regression(revenue_values, net_profit_series)
        regression_df = pd.DataFrame(
            [
                {
                    "Intercept": regression["intercept"],
                    "Slope": regression["slope"],
                    "R^2": regression["r_squared"],
                }
            ]
        )
        _render_table(regression_df, hide_index=True)
        reg_fig = go.Figure()
        reg_fig.add_trace(go.Scatter(x=revenue_values, y=net_profit_series, mode="markers", name="Actual"))
        reg_fig.add_trace(
            go.Scatter(
                x=revenue_values,
                y=regression["predictions"],
                mode="lines",
                name="Regression",
                line=dict(color="#1f77b4"),
            )
        )
        reg_fig.update_layout(xaxis_title="Revenue", yaxis_title="Net Profit")
        st.plotly_chart(reg_fig, use_container_width=True)

        st.subheader("Time series analysis (SES, moving average, seasonality)")
        ses = ts_analyzer.simple_exponential_smoothing(revenue_series, alpha=0.3, forecast_periods=3)
        ma = ts_analyzer.moving_average(revenue_series, window=3)
        seasonality = ts_analyzer.detect_seasonality(revenue_series)
        forecast_df = pd.DataFrame(
            {
                "Year": years + ses["future_years"],
                "Forecast": ses["historical_forecast"] + ses["future_forecast"],
            }
        )
        _render_table(forecast_df, hide_index=True)
        st.write(
            f"Seasonality detected: {'Yes' if seasonality['seasonal'] else 'No'} (autocorrelation {seasonality['strength']:.2f})"
        )

        st.subheader("Classification models for segment churn risk")
        if not segment_df.empty:
            segment_df = segment_df.copy()
            logits = -1.25 + 0.02 * (100 - segment_df["margin_pct"]) + 0.00000002 * segment_df["cost"]
            probabilities = 1 / (1 + np.exp(-logits))
            segment_df["Churn Probability"] = probabilities
            segment_df["Risk Tier"] = segment_df["Churn Probability"].apply(
                lambda p: "High" if p > 0.65 else ("Medium" if p > 0.35 else "Low")
            )
            classification_df = segment_df[["segment", "margin_pct", "cumulative_revenue_pct", "Churn Probability", "Risk Tier"]]
            classification_df.rename(
                columns={
                    "segment": "Segment",
                    "margin_pct": "Margin %",
                    "cumulative_revenue_pct": "Cumulative Revenue %",
                },
                inplace=True,
            )
            _render_table(classification_df, hide_index=True)
        else:
            st.write("Add product data to generate classification insights.")

    with tabs[2]:
        st.subheader("Value at Risk (VaR) / Conditional VaR")
        if len(profit_distribution) > 0:
            var_result = risk_analyzer.calculate_var(list(profit_distribution), 0.95)
            cvar_result = risk_analyzer.calculate_cvar(list(profit_distribution), 0.95)
            risk_df = pd.DataFrame([{
                "VaR (95%)": var_result["var"],
                "CVaR (95%)": cvar_result["cvar"],
                "Average Loss Beyond VaR": cvar_result["average_loss_beyond_var"],
            }])
            _render_table(risk_df, hide_index=True)
        else:
            st.write("Monte Carlo simulation is required to compute VaR metrics.")

        st.subheader("Stress testing for extreme market shocks")
        if len(profit_distribution) > 0:
            stress_summary = risk_analyzer.stress_test_returns(list(profit_distribution), 0.25)
            _render_table(pd.DataFrame([stress_summary]), hide_index=True)

        st.subheader("Copula-inspired correlation view across risk factors")
        metrics_matrix = []
        metric_labels = ["Revenue", "COGS", "OPEX", "Net Profit"]
        for year in years:
            metrics_matrix.append([
                model.get("revenue", {}).get(year, 0.0),
                model.get("cogs", {}).get(year, 0.0),
                model.get("opex", {}).get(year, 0.0),
                model.get("net_profit", {}).get(year, 0.0),
            ])
        if metrics_matrix:
            correlations = np.corrcoef(np.array(metrics_matrix).T)
            corr_df = pd.DataFrame(correlations, columns=metric_labels, index=metric_labels)
            _render_table(corr_df.reset_index().rename(columns={"index": "Metric"}), hide_index=True)

        st.subheader("Linear and nonlinear optimization to manage resources")
        if not segment_df.empty:
            budget = float(segment_df["cost"].sum()) * 0.5
            costs = segment_df["cost"].to_numpy()
            returns = segment_df["gross_profit"].to_numpy()
            if costs.sum() <= 0 or returns.sum() <= 0:
                st.write("Segment cost and profit data are required for the allocation model.")
            else:
                bounds = [(0, 1) for _ in costs]
                try:
                    linprog_result = optimize.linprog(
                        c=-returns,
                        A_ub=[costs],
                        b_ub=[budget],
                        bounds=bounds,
                        method="highs",
                    )
                    allocation = linprog_result.x if linprog_result.success else np.zeros_like(costs)
                except ValueError:
                    allocation = np.zeros_like(costs)
                allocation_df = segment_df[["segment"]].copy()
                allocation_df["Allocation"] = allocation
                allocation_df["Allocated Cost"] = allocation * costs
                allocation_df["Expected Gross Profit"] = allocation * returns
                _render_table(allocation_df, hide_index=True)
        else:
            st.write("Segment data is required for resource optimization.")

        st.subheader("Portfolio optimization to balance risk and return")
        if returns_matrix.size:
            portfolio_result = PortfolioOptimizer.optimize_portfolio(returns_matrix)
            weights_df = pd.DataFrame(
                {
                    "Product": list(product_returns.keys()),
                    "Optimal Weight": portfolio_result["optimal_weights"],
                }
            )
            portfolio_summary = pd.DataFrame(
                [
                    {
                        "Expected Return": portfolio_result["expected_return"],
                        "Volatility": portfolio_result["volatility"],
                        "Sharpe Ratio": portfolio_result["sharpe_ratio"],
                    }
                ]
            )
            _render_table(weights_df, hide_index=True)
            _render_table(portfolio_summary, hide_index=True)
        else:
            st.write("Additional projection years are required to evaluate portfolio mixes.")

    with tabs[3]:
        st.subheader("Macroeconomic linking for inflation, GDP, and FX")
        gdp_assumption = {
            year: 0.02 + 0.002 * (idx % 3) for idx, year in enumerate(years)
        }
        macro_revenue = macro_linker.apply_gdp_linkage(revenue_values[0], gdp_assumption)
        macro_df = pd.DataFrame(
            {
                "Year": list(macro_revenue.keys()),
                "GDP-Linked Revenue": list(macro_revenue.values()),
                "Inflation Adjusted": [
                    macro_linker.apply_inflation_adjustment(value, 0.025, idx)
                    for idx, value in enumerate(macro_revenue.values())
                ],
            }
        )
        _render_table(macro_df, hide_index=True)

        st.subheader("ESG & sustainability metrics")
        esg_impact = esg_analyzer.carbon_pricing_impact(1200, 40, 0.06, len(years))
        esg_df = pd.DataFrame(
            {
                "Year": [final_year + offset for offset in esg_impact.keys()],
                "Emissions": [entry["emissions"] for entry in esg_impact.values()],
                "Annual Cost": [entry["annual_cost"] for entry in esg_impact.values()],
                "Cumulative Cost": [entry["cumulative_cost"] for entry in esg_impact.values()],
            }
        )
        _render_table(esg_df, hide_index=True)

        st.subheader("Market intelligence integration and demand signals")
        growth_rates = []
        for idx in range(1, len(years)):
            prev = revenue_values[idx - 1]
            curr = revenue_values[idx]
            growth = ((curr - prev) / prev) if prev else 0.0
            growth_rates.append(growth)
        sentiment_series = [50 + rate * 500 for rate in growth_rates]
        market_df = pd.DataFrame(
            {
                "Year": years[1:],
                "Revenue Growth %": [rate * 100 for rate in growth_rates],
                "Sentiment Index": sentiment_series,
                "Industry Forecast %": [2.0 + idx * 0.2 for idx in range(len(growth_rates))],
            }
        )
        _render_table(market_df, hide_index=True)

        st.subheader("Probabilistic valuation and scenario distributions")
        if len(ev_distribution) > 0:
            prob_summary = ProbabilisticValuation.distribution_summary(list(ev_distribution))
            prob_df = pd.DataFrame([prob_summary]).drop(columns=["percentiles"], errors="ignore")
            _render_table(prob_df, hide_index=True)

        st.subheader("Real options analysis for managerial flexibility")
        expansion = real_options.expansion_option_value(base_ev, base_ev * 0.1, base_ev * 1.25)
        abandonment = real_options.abandonment_option_value(base_ev, base_ev * 0.3, base_ev * 0.6)
        _render_table(pd.DataFrame([expansion]), hide_index=True)
        _render_table(pd.DataFrame([abandonment]), hide_index=True)

        st.subheader("Comparative valuation with peer clustering")
        peer_data = pd.DataFrame(
            [
                {"Company": "Volt Rider", "Revenue": revenue_values[-1], "EBITDA Margin": net_profit_series[-1] / revenue_values[-1] if revenue_values[-1] else 0, "EV / Revenue": base_ev / revenue_values[-1] if revenue_values[-1] else 0},
                {"Company": "Peer A", "Revenue": revenue_values[-1] * 0.9, "EBITDA Margin": 0.14, "EV / Revenue": 2.1},
                {"Company": "Peer B", "Revenue": revenue_values[-1] * 1.2, "EBITDA Margin": 0.18, "EV / Revenue": 2.6},
                {"Company": "Peer C", "Revenue": revenue_values[-1] * 0.7, "EBITDA Margin": 0.11, "EV / Revenue": 1.8},
            ]
        )
        peer_data["Cluster"] = pd.qcut(peer_data["EV / Revenue"], 3, labels=["Value", "Balanced", "Growth"])
        _render_table(peer_data, hide_index=True)

        st.subheader("Machine learning–based valuation estimates")
        training_features = np.array([
            [peer_data.iloc[i]["Revenue"], peer_data.iloc[i]["EBITDA Margin"], peer_data.iloc[i]["EV / Revenue"]]
            for i in range(1, len(peer_data))
        ])
        training_targets = np.array([
            peer_data.iloc[i]["Revenue"] * peer_data.iloc[i]["EV / Revenue"] for i in range(1, len(peer_data))
        ])
        ml_result = RegressionModeler.multiple_regression(training_features, training_targets)
        if ml_result.get("success", True):
            coefficients = ml_result.get("coefficients", [])
            intercept = ml_result.get("intercept", 0.0)
            feature_vector = np.array([
                revenue_values[-1],
                net_profit_series[-1] / revenue_values[-1] if revenue_values[-1] else 0.0,
                peer_data.iloc[0]["EV / Revenue"],
            ])
            predicted_value = intercept + np.dot(coefficients, feature_vector)
            ml_df = pd.DataFrame(
                [
                    {
                        "Intercept": intercept,
                        "Revenue Coef": coefficients[0] if len(coefficients) > 0 else 0.0,
                        "Margin Coef": coefficients[1] if len(coefficients) > 1 else 0.0,
                        "Multiple Coef": coefficients[2] if len(coefficients) > 2 else 0.0,
                        "Predicted Enterprise Value": predicted_value,
                    }
                ]
            )
            _render_table(ml_df, hide_index=True)
        else:
            st.write("Insufficient peer data to train valuation model.")


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
        _render_production_horizon_editor(cfg)

        st.markdown("#### Production Capacity Schedule")
        model: Dict[str, Any] = st.session_state.get("financial_model", {})
        capacity_df = _production_capacity_schedule(cfg, model)
        _render_table(capacity_df, hide_index=True)
        _render_production_capacity_editor(cfg, model)

        st.markdown("#### Pricing Schedule")
        pricing_df = _pricing_schedule(cfg, model)
        if pricing_df.empty:
            st.write("Run the financial model to populate product pricing across the horizon.")
        _render_table(pricing_df, hide_index=True)
        _render_pricing_schedule_editor(cfg, model)

        st.markdown("#### Revenue Schedule")
        revenue_df = _revenue_schedule(cfg, model)
        if revenue_df.empty:
            st.write("Run the financial model to populate revenue projections across the horizon.")
        _render_table(revenue_df, hide_index=True)
        _render_revenue_schedule_editor(cfg, model)

        st.markdown("#### Cost Structure Schedule")
        cost_df = _cost_structure_schedule(cfg, model)
        if cost_df.empty:
            st.write("Run the financial model to populate cost structure projections across the horizon.")
        _render_table(cost_df, hide_index=True)
        _render_cost_structure_editor(cfg, model)

        st.markdown("#### Operating Expense Schedule")
        opex_df = _operating_expense_schedule(cfg, model)
        if opex_df.empty:
            st.write("Run the financial model to populate operating expense projections across the horizon.")
        _render_table(opex_df, hide_index=True)
        _render_operating_expense_editor(cfg, model)

        st.markdown("#### Working Capital Expense Schedule")
        working_capital_df = _working_capital_expense_schedule(cfg, model)
        if working_capital_df.empty:
            st.write("Run the financial model to populate working capital projections across the horizon.")
        _render_table(working_capital_df, hide_index=True)
        _render_working_capital_editor(cfg)

        st.markdown("#### Debt Amortization Schedule")
        debt_df = _debt_schedule(model)
        if debt_df.empty:
            st.write("Configure debt instruments in the financing settings to populate the schedule.")
        _render_table(debt_df, hide_index=True)
        _render_debt_schedule_editor(cfg)

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
        "Advanced Analytics",
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
    elif active_page == "Advanced Analytics":
        _render_advanced_analytics(model)
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
