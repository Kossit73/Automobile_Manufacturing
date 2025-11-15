"""
Automobile Manufacturing Financial Platform - Streamlit Web Interface
Clean, professional interface with horizontal navigation and comprehensive schedules.
"""

from __future__ import annotations

import io
import zipfile

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from scipy import optimize

from streamlit.delta_generator import DeltaGenerator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Trigger a Streamlit rerun when running inside the app."""

    # ``st.experimental_rerun`` raises a control-flow exception that should only
    # be triggered when the script executes within Streamlit. During unit tests
    # or direct module execution we skip the rerun to avoid bubbling the
    # exception to the caller.
    if getattr(st, "_is_running_with_streamlit", False):
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


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` between ``minimum`` and ``maximum``."""

    return max(minimum, min(maximum, value))


def _next_available_year(years: Sequence[int], candidate: int) -> int:
    """Return the first year at or after ``candidate`` that is not already present."""

    existing = {int(year) for year in years}
    while candidate in existing:
        candidate += 1
    return candidate


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
                index=len(years) - 1,
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

def _format_display_value(value: Any, field: Dict[str, Any]) -> str:
    if value is None or value == "":
        return "—"

    field_type = field.get("type", "text")
    if field_type == "currency":
        try:
            return f"${float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if field_type == "percent":
        try:
            precision = field.get("precision", 1)
            return f"{float(value):.{precision}f}%"
        except (TypeError, ValueError):
            return str(value)
    if field_type == "float":
        try:
            precision = field.get("precision", 2)
            return f"{float(value):,.{precision}f}"
        except (TypeError, ValueError):
            return str(value)
    if field_type == "int":
        try:
            return f"{int(round(float(value))):,}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _render_inline_row_editor(
    key: str,
    df: pd.DataFrame,
    field_definitions: Sequence[Dict[str, Any]],
    *,
    on_save: Callable[[pd.Series, Dict[str, Any]], Optional[str]],
    row_id_column: str = "Year",
    empty_message: str = "No rows available.",
) -> None:
    """Render a compact inline editor with per-row edit buttons."""

    if df.empty:
        st.info(empty_message)
        return

    status_key = f"{key}_status"
    if status_key in st.session_state:
        st.success(st.session_state.pop(status_key))

    editing_state_key = f"{key}_editing"
    current_editing = st.session_state.get(editing_state_key)

    header_columns = st.columns(len(field_definitions) + 1)
    for idx, field in enumerate(field_definitions):
        label = field.get("label", field.get("column", ""))
        header_columns[idx].markdown(f"**{label}**")
    header_columns[-1].markdown("**Action**")

    row_lookup: Dict[str, pd.Series] = {}

    for _, row in df.iterrows():
        row_id_value = row.get(row_id_column)
        row_key = str(row_id_value)
        row_lookup[row_key] = row

        columns = st.columns(len(field_definitions) + 1)
        for idx, field in enumerate(field_definitions):
            column_name = field.get("column")
            value = row.get(column_name)
            display_value = _format_display_value(value, field)
            columns[idx].markdown(display_value)

        if columns[-1].button("Edit", key=f"{key}_edit_{row_key}"):
            st.session_state[editing_state_key] = row_key
            current_editing = row_key

    if not current_editing:
        return

    row = row_lookup.get(current_editing)
    if row is None:
        st.session_state.pop(editing_state_key, None)
        return

    st.divider()
    st.markdown(f"**Editing {row_id_column}: {row.get(row_id_column)}**")

    updated_values: Dict[str, Any] = {}
    with st.form(key=f"{key}_form_{current_editing}"):
        form_columns = st.columns(len(field_definitions))
        for idx, field in enumerate(field_definitions):
            column_name = field.get("column")
            label = field.get("label", column_name)
            data_key = field.get("data_key", column_name)
            field_type = field.get("type", "text")
            editable = field.get("editable", True)
            value = row.get(column_name)
            col = form_columns[idx]

            if not editable:
                col.markdown(f"**{label}**")
                col.markdown(_format_display_value(value, field))
                continue

            widget_key = f"{key}_{current_editing}_{data_key}"

            if field_type in {"float", "currency", "percent"}:
                base_value = float(value or 0.0)
                input_kwargs: Dict[str, Any] = {
                    "label": label,
                    "key": widget_key,
                    "value": base_value,
                    "step": field.get("step", 1.0),
                }
                if field.get("min") is not None:
                    input_kwargs["min_value"] = float(field["min"])
                if field.get("max") is not None:
                    input_kwargs["max_value"] = float(field["max"])
                if field_type == "currency":
                    input_kwargs.setdefault("step", field.get("step", 1000.0))
                    input_kwargs["format"] = field.get("format", "%.0f")
                elif field_type == "percent":
                    input_kwargs.setdefault("step", field.get("step", 1.0))
                    input_kwargs["format"] = field.get("format", "%.2f")
                else:
                    input_kwargs["format"] = field.get("format", "%.2f")
                updated_values[data_key] = col.number_input(**input_kwargs)
            elif field_type == "int":
                base_value = int(round(float(value or 0)))
                input_kwargs = {
                    "label": label,
                    "key": widget_key,
                    "value": base_value,
                    "step": int(field.get("step", 1)),
                    "format": "%d",
                }
                if field.get("min") is not None:
                    input_kwargs["min_value"] = int(field["min"])
                if field.get("max") is not None:
                    input_kwargs["max_value"] = int(field["max"])
                updated_values[data_key] = col.number_input(**input_kwargs)
            else:
                updated_values[data_key] = col.text_input(
                    label,
                    value=str(value or ""),
                    key=widget_key,
                )

        col_actions = st.columns([1, 1])
        save_clicked = col_actions[0].form_submit_button("Save", type="primary")
        cancel_clicked = col_actions[1].form_submit_button("Cancel")

    if save_clicked:
        try:
            result = on_save(row, updated_values)
        except ValueError as exc:
            st.error(str(exc))
        else:
            if result:
                st.session_state[status_key] = result
            st.session_state.pop(editing_state_key, None)
            _rerun()
    elif cancel_clicked:
        st.session_state.pop(editing_state_key, None)
        _rerun()


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


def _cost_component_summary(
    cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None
) -> Dict[int, Dict[str, Any]]:
    """Return a per-year breakdown of major cost components."""

    model = model or {}
    years = _config_years(cfg)

    revenue_map: Dict[int, float] = model.get("revenue", {})
    variable_cogs: Dict[int, float] = model.get("variable_cogs", {})
    fixed_cogs: Dict[int, float] = model.get("fixed_cogs", {})
    opex: Dict[int, float] = model.get("opex", {})
    labor_metrics: Dict[int, Dict[str, float]] = model.get("labor_metrics", {})
    variable_breakdown: Dict[int, Dict[str, float]] = model.get("variable_cogs_breakdown", {})

    marketing_budget: Dict[int, float] = {}
    if isinstance(getattr(cfg, "marketing_budget", None), dict):
        marketing_budget = {int(year): float(value) for year, value in cfg.marketing_budget.items()}

    summary: Dict[int, Dict[str, Any]] = {}
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
        other_opex = max(0.0, other_opex)

        total_cost = variable_cost + fixed_cost + marketing_cost + labor_cost + other_opex

        summary[year] = {
            "revenue": revenue_value,
            "variable_total": variable_cost,
            "fixed_total": fixed_cost,
            "marketing": marketing_cost,
            "labor": labor_cost,
            "other": other_opex,
            "total": total_cost,
            "variable_breakdown": variable_breakdown.get(year, {}),
        }

    return summary


def _cost_structure_schedule(cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    summary = _cost_component_summary(cfg, model)
    years = _config_years(cfg)

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
        details = summary.get(
            year,
            {
                "revenue": 0.0,
                "variable_total": 0.0,
                "fixed_total": 0.0,
                "marketing": 0.0,
                "labor": 0.0,
                "other": 0.0,
                "total": 0.0,
            },
        )

        revenue_value = float(details["revenue"])
        variable_cost = float(details["variable_total"])
        fixed_cost = float(details["fixed_total"])
        marketing_cost = float(details["marketing"])
        labor_cost = float(details["labor"])
        other_opex = float(details["other"])
        total_cost = float(details["total"])

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


def _variable_production_cost_schedule(
    cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    summary = _cost_component_summary(cfg, model)
    years = _config_years(cfg)
    products = list(cfg.product_portfolio.keys())

    columns = ["Year"]
    for product in products:
        columns.append(f"{_product_label(product)} Variable Cost")
    columns.append("Total Variable Production Cost")

    rows: List[Dict[str, Any]] = []
    variable_breakdown: Dict[int, Dict[str, float]] = {}
    if model:
        variable_breakdown = model.get("variable_cogs_breakdown", {}) or {}

    for year in years:
        details = summary.get(year, {})
        row: Dict[str, Any] = {"Year": year}
        per_product = variable_breakdown.get(year, {})
        for product in products:
            value = float(per_product.get(product, 0.0))
            row[f"{_product_label(product)} Variable Cost"] = value
        row["Total Variable Production Cost"] = float(details.get("variable_total", 0.0))
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns)

    for column in columns[1:]:
        df[column] = _currency_series(df[column])
    return df


def _fixed_manufacturing_cost_schedule(
    cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    summary = _cost_component_summary(cfg, model)
    years = _config_years(cfg)

    rows: List[Dict[str, Any]] = []
    for year in years:
        fixed_value = float(summary.get(year, {}).get("fixed_total", 0.0))
        rows.append({"Year": year, "Fixed Manufacturing Cost": fixed_value})

    df = pd.DataFrame(rows, columns=["Year", "Fixed Manufacturing Cost"])
    if df.empty:
        return pd.DataFrame(columns=["Year", "Fixed Manufacturing Cost"])

    df["Fixed Manufacturing Cost"] = _currency_series(df["Fixed Manufacturing Cost"])
    return df


def _other_operating_cost_schedule(
    cfg: CompanyConfig, model: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    summary = _cost_component_summary(cfg, model)
    years = _config_years(cfg)

    rows: List[Dict[str, Any]] = []
    for year in years:
        other_value = float(summary.get(year, {}).get("other", 0.0))
        rows.append({"Year": year, "Other Operating Cost": other_value})

    df = pd.DataFrame(rows, columns=["Year", "Other Operating Cost"])
    if df.empty:
        return pd.DataFrame(columns=["Year", "Other Operating Cost"])

    df["Other Operating Cost"] = _currency_series(df["Other Operating Cost"])
    return df


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
            new_year = _next_available_year(years, target_year + 1)
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

    def _save_production_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        utilization_pct = float(values.get("Capacity Utilization %", row["Capacity Utilization %"]))
        annual_capacity = float(values.get("Annual Capacity", row["Annual Capacity"]))
        working_days = int(values.get("Working Days", row["Working Days"]))
        if annual_capacity <= 0:
            raise ValueError("Annual capacity must be positive.")
        if working_days <= 0:
            raise ValueError("Working days must be positive.")
        cfg.capacity_utilization[year] = _clamp(utilization_pct / 100.0, 0.0, 1.5)
        cfg.annual_capacity = annual_capacity
        cfg.working_days = working_days
        cfg.__post_init__()
        return f"Production assumptions updated for {year}."

    _render_inline_row_editor(
        "production_horizon_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Capacity Utilization %",
                "label": "Capacity Utilization %",
                "type": "percent",
                "min": 0.0,
                "max": 150.0,
                "step": 1.0,
            },
            {
                "column": "Annual Capacity",
                "label": "Annual Capacity",
                "type": "float",
                "min": 0.0,
                "step": 1000.0,
            },
            {
                "column": "Working Days",
                "label": "Working Days",
                "type": "int",
                "min": 0,
                "max": 400,
                "step": 1,
            },
            {"column": "Planned Units", "label": "Planned Units", "type": "float", "editable": False},
        ],
        on_save=_save_production_row,
        row_id_column="Year",
        empty_message="Add projection years to edit production assumptions.",
    )

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
            new_year = _next_available_year(years, target_year + 1)
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

    def _save_capacity_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        planned_units = max(0.0, float(values.get("Planned Units", row["Planned Units"])))
        if cfg.annual_capacity <= 0:
            raise ValueError("Set an annual capacity before editing planned units.")
        utilization = _clamp(planned_units / float(cfg.annual_capacity), 0.0, 1.5)
        cfg.capacity_utilization[year] = utilization
        cfg.__post_init__()
        return f"Capacity target updated for {year}."

    _render_inline_row_editor(
        "production_capacity_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Planned Units",
                "label": "Planned Units",
                "type": "float",
                "min": 0.0,
                "step": 1000.0,
            },
            {
                "column": "Implied Utilization %",
                "label": "Implied Utilization %",
                "type": "percent",
                "precision": 2,
                "editable": False,
            },
            {
                "column": "Remaining Headroom",
                "label": "Remaining Headroom",
                "type": "float",
                "precision": 0,
                "editable": False,
            },
        ],
        on_save=_save_capacity_row,
        row_id_column="Year",
        empty_message="Add capacity rows to begin editing targets.",
    )

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
            new_year = _next_available_year(years, target_year + 1)
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

    product_labels = {_product_label(product): product for product in products}

    def _save_pricing_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        overrides: Dict[str, float] = {}
        for label, product in product_labels.items():
            price_value = float(values.get(product, row.get(label, 0.0)))
            if price_value < 0:
                raise ValueError("Prices cannot be negative.")
            overrides[product] = price_value
        cfg.product_price_overrides = cfg.product_price_overrides or {}
        cfg.product_price_overrides[year] = overrides
        cfg.__post_init__()
        return f"Pricing updated for {year}."

    pricing_fields: List[Dict[str, Any]] = [
        {"column": "Year", "label": "Year", "type": "int", "editable": False}
    ]
    for label, product in product_labels.items():
        pricing_fields.append(
            {
                "column": label,
                "label": f"{label} Price",
                "type": "currency",
                "min": 0.0,
                "step": 100.0,
                "data_key": product,
            }
        )

    _render_inline_row_editor(
        "pricing_schedule_inline",
        base_df,
        pricing_fields,
        on_save=_save_pricing_row,
        row_id_column="Year",
        empty_message="Add pricing rows to begin editing values.",
    )

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
            new_year = _next_available_year(years, target_year + 1)
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

    unit_labels = {_product_label(product): product for product in products}

    def _save_revenue_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        overrides: Dict[str, float] = {}
        for label, product in unit_labels.items():
            units_value = float(values.get(product, row.get(label, 0.0)))
            if units_value < 0:
                raise ValueError("Units cannot be negative.")
            overrides[product] = units_value
        cfg.product_unit_overrides = cfg.product_unit_overrides or {}
        cfg.product_unit_overrides[year] = overrides
        cfg.__post_init__()
        return f"Unit plan updated for {year}."

    revenue_fields: List[Dict[str, Any]] = [
        {"column": "Year", "label": "Year", "type": "int", "editable": False}
    ]
    for label, product in unit_labels.items():
        revenue_fields.append(
            {
                "column": label,
                "label": f"{label} Units",
                "type": "float",
                "min": 0.0,
                "step": 100.0,
                "data_key": product,
            }
        )

    _render_inline_row_editor(
        "revenue_schedule_inline",
        base_df,
        revenue_fields,
        on_save=_save_revenue_row,
        row_id_column="Year",
        empty_message="Add revenue rows to begin editing unit volumes.",
    )

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
            new_year = _next_available_year(years, target_year + 1)
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

    def _save_cost_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        variable_value = max(0.0, float(values.get("Variable Production Cost", row["Variable Production Cost"])))
        fixed_value = max(0.0, float(values.get("Fixed Manufacturing Cost", row["Fixed Manufacturing Cost"])))
        marketing_value = max(0.0, float(values.get("Marketing Spend", row["Marketing Spend"])))
        cfg.variable_cost_overrides = cfg.variable_cost_overrides or {}
        cfg.fixed_cost_overrides = cfg.fixed_cost_overrides or {}
        cfg.marketing_budget = cfg.marketing_budget or {}
        cfg.variable_cost_overrides[year] = variable_value
        cfg.fixed_cost_overrides[year] = fixed_value
        cfg.marketing_budget[year] = marketing_value
        cfg.__post_init__()
        return f"Cost structure updated for {year}."

    _render_inline_row_editor(
        "cost_structure_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Variable Production Cost",
                "label": "Variable Production Cost",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
            {
                "column": "Fixed Manufacturing Cost",
                "label": "Fixed Manufacturing Cost",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
            {
                "column": "Marketing Spend",
                "label": "Marketing Spend",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
        ],
        on_save=_save_cost_row,
        row_id_column="Year",
        empty_message="Add cost rows to begin editing values.",
    )

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


def _render_variable_cost_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    summary = _cost_component_summary(cfg, model)
    products = list(cfg.product_portfolio.keys())
    variable_breakdown: Dict[int, Dict[str, float]] = model.get("variable_cogs_breakdown", {}) if model else {}

    st.markdown("##### Edit Variable Production Cost")
    action = _schedule_toolbar("variable_cost", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "variable_cost_overrides", None):
                cfg.variable_cost_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Variable cost row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = _next_available_year(years, target_year + 1)
            _ensure_year_in_horizon(cfg, new_year)
            defaults = summary.get(target_year, {})
            cfg.variable_cost_overrides = cfg.variable_cost_overrides or {}
            cfg.variable_cost_overrides[new_year] = float(defaults.get("variable_total", 0.0))
            cfg.__post_init__()
            _run_model()
            st.success(f"Variable cost row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        details = summary.get(year, {})
        per_product = variable_breakdown.get(year, {}) if variable_breakdown else {}
        row: Dict[str, Any] = {
            "Year": year,
            "Total Variable Production Cost": float(details.get("variable_total", 0.0)),
        }
        for product in products:
            label = f"{_product_label(product)} Variable Cost"
            row[label] = float(per_product.get(product, 0.0))
        rows.append(row)

    base_df = pd.DataFrame(rows)
    if base_df.empty:
        st.info("Add production cost rows to begin editing values.")
        return

    def _save_variable_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        total_value = max(
            0.0, float(values.get("Total Variable Production Cost", row["Total Variable Production Cost"]))
        )
        cfg.variable_cost_overrides = cfg.variable_cost_overrides or {}
        cfg.variable_cost_overrides[year] = total_value
        cfg.__post_init__()
        _run_model()
        return f"Variable production cost updated for {year}."

    field_definitions: List[Dict[str, Any]] = [
        {"column": "Year", "label": "Year", "type": "int", "editable": False},
        {
            "column": "Total Variable Production Cost",
            "label": "Total Variable Production Cost",
            "type": "currency",
            "min": 0.0,
            "step": 1000.0,
        },
    ]
    for product in products:
        field_definitions.append(
            {
                "column": f"{_product_label(product)} Variable Cost",
                "label": f"{_product_label(product)} Variable Cost",
                "type": "currency",
                "editable": False,
            }
        )

    _render_inline_row_editor(
        "variable_cost_inline",
        base_df,
        field_definitions,
        on_save=_save_variable_row,
        row_id_column="Year",
        empty_message="Add production cost rows to begin editing values.",
    )

    editable_df = base_df[["Year", "Total Variable Production Cost"]].copy()
    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "variable_cost",
        editable_df,
        guidance="Adjust total variable production cost assumptions by year.",
        non_negative_columns=["Total Variable Production Cost"],
    )

    def _apply_variable_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.variable_cost_overrides = cfg.variable_cost_overrides or {}
        for year in selected_years:
            defaults = summary.get(year, {})
            base_value = float(defaults.get("variable_total", 0.0))
            cfg.variable_cost_overrides[year] = max(0.0, _apply_increment(base_value, increment_pct))
        cfg.__post_init__()
        _run_model()
        return "Variable production cost increments applied."

    _schedule_increment_helper(
        "variable_cost",
        years,
        _apply_variable_increment,
        help_text="Apply a uniform percentage change to variable production cost across selected years.",
    )

    if st.button("Apply Variable Cost Updates", key="variable_cost_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying variable cost updates.")

            overrides: Dict[int, float] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                total_value = float(row.get("Total Variable Production Cost", 0.0))
                if total_value < 0:
                    raise ValueError("Variable production cost cannot be negative.")
                overrides[year_value] = total_value

            cfg.variable_cost_overrides = overrides
            cfg.projection_years = max(overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Variable production cost schedule updated.")


def _render_fixed_cost_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    summary = _cost_component_summary(cfg, model)

    st.markdown("##### Edit Fixed Manufacturing Cost")
    action = _schedule_toolbar("fixed_cost", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "fixed_cost_overrides", None):
                cfg.fixed_cost_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Fixed cost row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = _next_available_year(years, target_year + 1)
            _ensure_year_in_horizon(cfg, new_year)
            defaults = summary.get(target_year, {})
            cfg.fixed_cost_overrides = cfg.fixed_cost_overrides or {}
            cfg.fixed_cost_overrides[new_year] = float(defaults.get("fixed_total", 0.0))
            cfg.__post_init__()
            _run_model()
            st.success(f"Fixed cost row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        rows.append(
            {
                "Year": year,
                "Fixed Manufacturing Cost": float(summary.get(year, {}).get("fixed_total", 0.0)),
            }
        )

    base_df = pd.DataFrame(rows)
    if base_df.empty:
        st.info("Add fixed cost rows to begin editing values.")
        return

    def _save_fixed_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        fixed_value = max(0.0, float(values.get("Fixed Manufacturing Cost", row["Fixed Manufacturing Cost"])))
        cfg.fixed_cost_overrides = cfg.fixed_cost_overrides or {}
        cfg.fixed_cost_overrides[year] = fixed_value
        cfg.__post_init__()
        _run_model()
        return f"Fixed manufacturing cost updated for {year}."

    _render_inline_row_editor(
        "fixed_cost_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Fixed Manufacturing Cost",
                "label": "Fixed Manufacturing Cost",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
        ],
        on_save=_save_fixed_row,
        row_id_column="Year",
        empty_message="Add fixed cost rows to begin editing values.",
    )

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "fixed_cost",
        base_df.copy(),
        guidance="Update fixed manufacturing cost assumptions across the forecast horizon.",
        non_negative_columns=["Fixed Manufacturing Cost"],
    )

    def _apply_fixed_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.fixed_cost_overrides = cfg.fixed_cost_overrides or {}
        for year in selected_years:
            defaults = summary.get(year, {})
            base_value = float(defaults.get("fixed_total", 0.0))
            cfg.fixed_cost_overrides[year] = max(0.0, _apply_increment(base_value, increment_pct))
        cfg.__post_init__()
        _run_model()
        return "Fixed manufacturing cost increments applied."

    _schedule_increment_helper(
        "fixed_cost",
        years,
        _apply_fixed_increment,
        help_text="Apply a uniform percentage change to fixed manufacturing cost across selected years.",
    )

    if st.button("Apply Fixed Cost Updates", key="fixed_cost_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying fixed cost updates.")

            overrides: Dict[int, float] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                fixed_value = float(row.get("Fixed Manufacturing Cost", 0.0))
                if fixed_value < 0:
                    raise ValueError("Fixed manufacturing cost cannot be negative.")
                overrides[year_value] = fixed_value

            cfg.fixed_cost_overrides = overrides
            cfg.projection_years = max(overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Fixed manufacturing cost schedule updated.")


def _render_other_operating_cost_editor(cfg: CompanyConfig, model: Dict[str, Any]) -> None:
    years = _config_years(cfg)
    if not years:
        return

    summary = _cost_component_summary(cfg, model)

    st.markdown("##### Edit Other Operating Cost")
    action = _schedule_toolbar("other_operating_cost", years)
    if action:
        target_year = int(action["year"])
        if action["action"] == "remove":
            if getattr(cfg, "other_opex_overrides", None):
                cfg.other_opex_overrides.pop(target_year, None)
            if target_year == _horizon_end(cfg) and cfg.projection_years > 1:
                cfg.projection_years -= 1
            cfg.__post_init__()
            _run_model()
            st.success(f"Other operating cost row removed for {target_year}.")
            return
        if action["action"] == "add":
            new_year = _next_available_year(years, target_year + 1)
            _ensure_year_in_horizon(cfg, new_year)
            defaults = summary.get(target_year, {})
            cfg.other_opex_overrides = cfg.other_opex_overrides or {}
            cfg.other_opex_overrides[new_year] = float(defaults.get("other", 0.0))
            cfg.__post_init__()
            _run_model()
            st.success(f"Other operating cost row added for {new_year}.")
            return

    rows: List[Dict[str, Any]] = []
    for year in years:
        rows.append(
            {
                "Year": year,
                "Other Operating Cost": float(summary.get(year, {}).get("other", 0.0)),
            }
        )

    base_df = pd.DataFrame(rows)
    if base_df.empty:
        st.info("Add other operating cost rows to begin editing values.")
        return

    def _save_other_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        other_value = max(0.0, float(values.get("Other Operating Cost", row["Other Operating Cost"])))
        cfg.other_opex_overrides = cfg.other_opex_overrides or {}
        cfg.other_opex_overrides[year] = other_value
        cfg.__post_init__()
        _run_model()
        return f"Other operating cost updated for {year}."

    _render_inline_row_editor(
        "other_operating_cost_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Other Operating Cost",
                "label": "Other Operating Cost",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
        ],
        on_save=_save_other_row,
        row_id_column="Year",
        empty_message="Add other operating cost rows to begin editing values.",
    )

    edited_df, status_placeholder, warnings, has_changes = _schedule_table_editor(
        "other_operating_cost",
        base_df.copy(),
        guidance="Update other operating cost assumptions across the forecast horizon.",
        non_negative_columns=["Other Operating Cost"],
    )

    def _apply_other_increment(selected_years: Sequence[int], increment_pct: float) -> Optional[str]:
        if not selected_years:
            raise ValueError("Select at least one year to increment.")
        cfg.other_opex_overrides = cfg.other_opex_overrides or {}
        for year in selected_years:
            defaults = summary.get(year, {})
            base_value = float(defaults.get("other", 0.0))
            cfg.other_opex_overrides[year] = max(0.0, _apply_increment(base_value, increment_pct))
        cfg.__post_init__()
        _run_model()
        return "Other operating cost increments applied."

    _schedule_increment_helper(
        "other_operating_cost",
        years,
        _apply_other_increment,
        help_text="Apply a uniform percentage change to other operating cost across selected years.",
    )

    if st.button("Apply Other Operating Cost Updates", key="other_operating_cost_apply"):
        try:
            sanitized = edited_df.dropna(subset=["Year"])
            sanitized = sanitized.sort_values("Year")
            if sanitized.empty:
                raise ValueError("Add at least one year before applying other operating cost updates.")

            overrides: Dict[int, float] = {}
            for _, row in sanitized.iterrows():
                year_value = int(row["Year"])
                other_value = float(row.get("Other Operating Cost", 0.0))
                if other_value < 0:
                    raise ValueError("Other operating cost cannot be negative.")
                overrides[year_value] = other_value

            cfg.other_opex_overrides = overrides
            cfg.projection_years = max(overrides.keys()) - cfg.start_year + 1
            cfg.__post_init__()
            _run_model()
        except ValueError as exc:
            status_placeholder.error(str(exc))
        else:
            status_placeholder.success("Other operating cost schedule updated.")


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
            new_year = _next_available_year(years, target_year + 1)
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

    def _save_opex_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        marketing_value = max(0.0, float(values.get("Marketing Spend", row["Marketing Spend"])))
        other_value = max(0.0, float(values.get("Other Operating Expense", row["Other Operating Expense"])))
        cfg.marketing_budget = cfg.marketing_budget or {}
        cfg.other_opex_overrides = cfg.other_opex_overrides or {}
        cfg.marketing_budget[year] = marketing_value
        cfg.other_opex_overrides[year] = other_value
        cfg.__post_init__()
        return f"Operating expenses updated for {year}."

    _render_inline_row_editor(
        "operating_expense_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Marketing Spend",
                "label": "Marketing Spend",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
            {
                "column": "Other Operating Expense",
                "label": "Other Operating Expense",
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
            },
        ],
        on_save=_save_opex_row,
        row_id_column="Year",
        empty_message="Add operating expense rows to begin editing values.",
    )

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
            new_year = _next_available_year(years, target_year + 1)
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

    def _save_working_capital_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        receivable = max(0.0, float(values.get("Receivable Days", row["Receivable Days"])))
        inventory = max(0.0, float(values.get("Inventory Days", row["Inventory Days"])))
        payable = max(0.0, float(values.get("Payable Days", row["Payable Days"])))
        accrued = max(0.0, float(values.get("Accrued Expense Ratio", row["Accrued Expense Ratio"])))
        cfg.working_capital_overrides = cfg.working_capital_overrides or {}
        cfg.working_capital_overrides[year] = {
            "receivable_days": receivable,
            "inventory_days": inventory,
            "payable_days": payable,
            "accrued_expense_ratio": accrued,
        }
        cfg.__post_init__()
        return f"Working capital drivers updated for {year}."

    _render_inline_row_editor(
        "working_capital_inline",
        base_df,
        [
            {"column": "Year", "label": "Year", "type": "int", "editable": False},
            {
                "column": "Receivable Days",
                "label": "Receivable Days",
                "type": "float",
                "min": 0.0,
                "step": 5.0,
            },
            {
                "column": "Inventory Days",
                "label": "Inventory Days",
                "type": "float",
                "min": 0.0,
                "step": 5.0,
            },
            {
                "column": "Payable Days",
                "label": "Payable Days",
                "type": "float",
                "min": 0.0,
                "step": 5.0,
            },
            {
                "column": "Accrued Expense Ratio",
                "label": "Accrued Expense Ratio",
                "type": "float",
                "min": 0.0,
                "step": 1.0,
            },
        ],
        on_save=_save_working_capital_row,
        row_id_column="Year",
        empty_message="Add working capital rows to begin editing values.",
    )

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
            new_year = _next_available_year(years, target_year + 1)
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

    def _save_debt_row(row: pd.Series, values: Dict[str, Any]) -> Optional[str]:
        year = int(row["Year"])
        for idx, instrument in enumerate(instruments):
            label = column_labels[idx]
            draw_value = max(0.0, float(values.get(str(idx), row.get(label, 0.0))))
            instrument.draw_schedule[year] = draw_value
            instrument.__post_init__()
        cfg.debt_instruments = instruments
        cfg.__post_init__()
        return f"Debt draws updated for {year}."

    _debt_fields: List[Dict[str, Any]] = [
        {"column": "Year", "label": "Year", "type": "int", "editable": False}
    ]
    for idx, label in column_labels.items():
        _debt_fields.append(
            {
                "column": label,
                "label": label,
                "type": "currency",
                "min": 0.0,
                "step": 1000.0,
                "data_key": str(idx),
            }
        )

    _render_inline_row_editor(
        "debt_schedule_inline",
        base_df,
        _debt_fields,
        on_save=_save_debt_row,
        row_id_column="Year",
        empty_message="Add debt schedule rows to begin editing draws.",
    )

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

def _forecast_schedule(model: Dict[str, Any], formatted: bool = True) -> pd.DataFrame:
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
    if formatted:
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
# REPORTING HELPERS
# ---------------------------------------------------------------------------

def _calculate_cagr(start: float, end: float, periods: int) -> float:
    if periods <= 0 or start <= 0 or end <= 0:
        return 0.0
    try:
        return (end / start) ** (1 / periods) - 1
    except ZeroDivisionError:
        return 0.0


def _reports_kpi_summary(model: Dict[str, Any], years: Sequence[int]) -> List[Dict[str, str]]:
    if not years:
        return []

    first_year, final_year = years[0], years[-1]
    revenue = model.get("revenue", {})
    ebitda = model.get("ebitda", {})
    net_profit = model.get("net_profit", {})
    cash = model.get("cash_balance", {})
    outstanding_debt = model.get("outstanding_debt", {})
    equity = model.get("total_equity", {})

    revenue_start = float(revenue.get(first_year, 0.0))
    revenue_end = float(revenue.get(final_year, 0.0))
    net_profit_end = float(net_profit.get(final_year, 0.0))
    cash_end = float(cash.get(final_year, 0.0))
    ebitda_end = float(ebitda.get(final_year, 0.0))
    debt_end = float(outstanding_debt.get(final_year, 0.0))
    equity_end = float(equity.get(final_year, 0.0))

    cagr = _calculate_cagr(revenue_start, revenue_end, len(years) - 1)
    ebitda_margin = (ebitda_end / revenue_end) if revenue_end else 0.0
    debt_to_equity = (debt_end / equity_end) if equity_end else 0.0

    return [
        {
            "label": "Revenue CAGR",
            "value": f"{cagr * 100:.1f}%",
            "delta": f"${revenue_start:,.0f} → ${revenue_end:,.0f}",
        },
        {
            "label": "EBITDA Margin (final year)",
            "value": f"{ebitda_margin * 100:.1f}%",
            "delta": f"EBITDA ${ebitda_end:,.0f}",
        },
        {
            "label": "Net Profit (final year)",
            "value": f"${net_profit_end:,.0f}",
            "delta": f"Closing Cash ${cash_end:,.0f}",
        },
        {
            "label": "Debt-to-Equity",
            "value": f"{debt_to_equity:.2f}x",
            "delta": f"Debt ${debt_end:,.0f}",
        },
    ]


def _reports_quality_notes(model: Dict[str, Any], capex_manager: CapexScheduleManager, years: Sequence[int]) -> List[str]:
    notes: List[str] = []
    balance_check = model.get("balance_check", {})
    for year in years:
        variance = float(balance_check.get(year, 0.0))
        if abs(variance) > 1.0:
            notes.append(
                f"Balance sheet is off by ${variance:,.0f} in {year}. Review working-capital or debt inputs."
            )
            break

    if not capex_manager.list_items():
        notes.append("No CAPEX projects configured; spend and depreciation schedules show placeholders.")

    if not model.get("labor_metrics"):
        notes.append("Labor metrics unavailable; connect the labor manager to populate workforce schedules.")

    return notes


def _report_controls(years: Sequence[int]) -> Tuple[List[int], bool, bool]:
    st.markdown("### Report Controls")
    filter_col, toggle_col = st.columns([3, 1])
    with filter_col:
        selected_years = st.multiselect(
            "Select Years",
            options=list(years),
            default=list(years),
            key="report_year_filter",
        )
    if not selected_years:
        selected_years = list(years)
    selected_years = sorted(selected_years)

    with toggle_col:
        show_yoy = st.checkbox("YoY Variance", value=True, key="report_show_yoy")
        show_charts = st.checkbox("Show Charts", value=True, key="report_show_charts")

    return selected_years, show_yoy, show_charts


def _filter_schedule_years(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    if df.empty or "Year" not in df.columns:
        return df
    return df[df["Year"].isin(years)]


def _series_map_from_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[int, float]]:
    if df.empty or "Year" not in df.columns:
        return {}

    series_map: Dict[str, Dict[int, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values: Dict[int, float] = {}
        for year, value in zip(df["Year"], df[column]):
            try:
                values[int(year)] = float(value)
            except Exception:
                try:
                    cleaned = str(value).replace("$", "").replace(",", "")
                    values[int(year)] = float(cleaned)
                except Exception:
                    values[int(year)] = 0.0
        series_map[column] = values
    return series_map


def _build_yoy_dataframe(series_map: Dict[str, Dict[int, float]], years: Sequence[int]) -> pd.DataFrame:
    if not series_map or len(years) < 2:
        return pd.DataFrame()

    sorted_years = sorted(years)
    rows: List[Dict[str, Any]] = []
    previous_year: Optional[int] = None
    for year in sorted_years:
        row: Dict[str, Any] = {"Year": year}
        for label, series in series_map.items():
            current_value = float(series.get(year, 0.0))
            if previous_year is None:
                row[f"{label} Δ"] = None
                row[f"{label} Δ%"] = None
            else:
                previous_value = float(series.get(previous_year, 0.0))
                delta = current_value - previous_value
                percent = (delta / previous_value * 100.0) if previous_value else 0.0
                row[f"{label} Δ"] = delta
                row[f"{label} Δ%"] = percent
        rows.append(row)
        previous_year = year

    return pd.DataFrame(rows)


def _format_variance_value(column: str, value: Optional[float]) -> str:
    if value is None:
        return "–"
    if column.endswith("Δ%"):
        return f"{value:+.1f}%"

    normalized = column.lower()
    monetary_tokens = [
        "revenue",
        "cogs",
        "profit",
        "cash",
        "debt",
        "spend",
        "expense",
        "ebitda",
        "interest",
        "principal",
        "draw",
        "capital",
    ]
    if any(token in normalized for token in monetary_tokens):
        return f"${value:+,.0f}"
    if "headcount" in normalized or "units" in normalized:
        return f"{value:+,.0f}"
    return f"{value:+,.2f}"


def _format_yoy_dataframe(yoy_df: pd.DataFrame) -> pd.DataFrame:
    if yoy_df.empty:
        return yoy_df

    formatted = pd.DataFrame({"Year": yoy_df["Year"]})
    for column in yoy_df.columns:
        if column == "Year":
            continue
        formatted[column] = [
            _format_variance_value(column, value) for value in yoy_df[column]
        ]
    return formatted


def _schedule_commentary(series_map: Dict[str, Dict[int, float]], years: Sequence[int]) -> str:
    if not series_map or not years:
        return ""

    sorted_years = sorted(years)
    start_year, end_year = sorted_years[0], sorted_years[-1]
    comments: List[str] = []
    for label, data in series_map.items():
        start_value = float(data.get(start_year, 0.0))
        end_value = float(data.get(end_year, 0.0))
        change = end_value - start_value
        change_text: str
        if start_value:
            pct = change / start_value * 100.0
            change_text = f"{change:+,.0f} ({pct:+.1f}%)"
        else:
            change_text = f"{change:+,.0f}"

        label_lower = label.lower()
        if any(token in label_lower for token in ["headcount", "units"]):
            base_text = f"{start_value:,.0f} → {end_value:,.0f}"
        else:
            base_text = f"${start_value:,.0f} → ${end_value:,.0f}"

        comments.append(f"{label}: {base_text} ({change_text})")

    return " | ".join(comments)


def _build_schedule_chart(title: str, series_map: Dict[str, Dict[int, float]], years: Sequence[int]) -> Optional[go.Figure]:
    if not series_map or not years:
        return None

    sorted_years = sorted(years)
    fig = go.Figure()
    for label, data in series_map.items():
        y_values = [float(data.get(year, 0.0)) for year in sorted_years]
        fig.add_trace(
            go.Scatter(
                name=label,
                x=sorted_years,
                y=y_values,
                mode="lines+markers",
            )
        )

    fig.update_layout(
        margin=dict(t=20, r=20, b=20, l=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Value")
    return fig


def _reports_export_controls(raw_tables: Dict[str, pd.DataFrame], years: Sequence[int]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for title, df in raw_tables.items():
            if df is None or df.empty:
                continue
            export_df = df
            if "Year" in df.columns:
                export_df = df[df["Year"].isin(years)]
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            filename = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".csv"
            archive.writestr(filename, csv_buffer.getvalue())

    st.download_button(
        "Download filtered schedules (ZIP)",
        data=buffer.getvalue(),
        file_name="report_schedules.zip",
        mime="application/zip",
        key="report_export_zip",
    )


def _labor_schedule_raw(model: Dict[str, Any], years: Sequence[int]) -> pd.DataFrame:
    labor_metrics = model.get("labor_metrics", {})
    rows: List[Dict[str, Any]] = []
    for year in years:
        metrics = labor_metrics.get(year, {})
        rows.append(
            {
                "Year": year,
                "Direct Labor Cost": float(metrics.get("direct_labor_cost", 0.0)),
                "Indirect Labor Cost": float(metrics.get("indirect_labor_cost", 0.0)),
                "Total Labor Cost": float(metrics.get("total_labor_cost", 0.0)),
                "Total Headcount": float(metrics.get("total_headcount", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _capex_spend_raw(manager: CapexScheduleManager, cfg: CompanyConfig, years: Sequence[int]) -> pd.DataFrame:
    schedule = manager.yearly_capex_schedule(cfg.start_year, len(years))
    rows = [{"Year": year, "Capital Spend": float(schedule.get(year, 0.0))} for year in years]
    return pd.DataFrame(rows)


def _debt_schedule_raw(model: Dict[str, Any], years: Sequence[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        rows.append(
            {
                "Year": year,
                "Debt Draw": float(model.get("debt_draws", {}).get(year, 0.0)),
                "Interest Payment": float(model.get("interest_payment", {}).get(year, 0.0)),
                "Principal Repayment": float(model.get("loan_repayment", {}).get(year, 0.0)),
                "Outstanding Debt": float(model.get("outstanding_debt", {}).get(year, 0.0)),
            }
        )
    return pd.DataFrame(rows)


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

    income_df, cashflow_df, balance_df = generate_financial_statements(model)
    first_year, final_year = years[0], years[-1]

    revenue_series = model.get("revenue", {})
    net_profit_series = model.get("net_profit", {})
    cogs_series = model.get("cogs", {})
    cfo_series = model.get("cfo", {})
    cfi_series = model.get("cfi", {})
    cff_series = model.get("cff", {})
    cash_series = model.get("cash_balance", {})
    production_volume = model.get("production_volume", {})

    st.markdown("### Executive Highlights")
    highlight_cols = st.columns(4)
    highlight_cols[0].metric("Year 1 Revenue", f"${revenue_series.get(first_year, 0.0):,.0f}")
    highlight_cols[1].metric(
        "Final Year Net Profit", f"${net_profit_series.get(final_year, 0.0):,.0f}"
    )
    highlight_cols[2].metric(
        "Final Year Operating Cash Flow",
        f"${cfo_series.get(final_year, 0.0):,.0f}",
    )
    highlight_cols[3].metric(
        "Closing Cash Balance", f"${cash_series.get(final_year, 0.0):,.0f}"
    )

    st.markdown("### Revenue, Margin, and Volume Trends")
    revenue_values = [revenue_series.get(year, 0.0) for year in years]
    net_profit_values = [net_profit_series.get(year, 0.0) for year in years]
    cogs_values = [cogs_series.get(year, 0.0) for year in years]
    gross_margin_pct = [
        (revenue_series.get(year, 0.0) - cogs_series.get(year, 0.0))
        / revenue_series.get(year, 1.0)
        if revenue_series.get(year, 0.0)
        else 0.0
        for year in years
    ]
    units_values = [production_volume.get(year, 0.0) for year in years]

    trend_fig = make_subplots(specs=[[{"secondary_y": True}]])
    trend_fig.add_trace(
        go.Bar(name="Revenue", x=years, y=revenue_values, marker_color="#2E86AB"),
        secondary_y=False,
    )
    trend_fig.add_trace(
        go.Bar(name="COGS", x=years, y=cogs_values, marker_color="#C0392B"),
        secondary_y=False,
    )
    trend_fig.add_trace(
        go.Scatter(
            name="Net Profit",
            x=years,
            y=net_profit_values,
            mode="lines+markers",
            line=dict(color="#1ABC9C", width=3),
        ),
        secondary_y=True,
    )
    trend_fig.add_trace(
        go.Scatter(
            name="Gross Margin %",
            x=years,
            y=[value * 100 for value in gross_margin_pct],
            mode="lines",
            line=dict(color="#F39C12", dash="dot"),
        ),
        secondary_y=True,
    )
    trend_fig.update_layout(
        barmode="group",
        margin=dict(t=20, r=20, b=20, l=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    trend_fig.update_xaxes(title_text="Year")
    trend_fig.update_yaxes(title_text="USD", secondary_y=False)
    trend_fig.update_yaxes(title_text="Net Profit / Margin", secondary_y=True)
    st.plotly_chart(trend_fig, use_container_width=True)

    units_fig = go.Figure(
        go.Scatter(
            x=years,
            y=units_values,
            mode="lines+markers",
            line=dict(color="#34495E", width=3),
            name="Units Produced",
        )
    )
    units_fig.update_layout(
        margin=dict(t=20, r=20, b=20, l=20),
        height=260,
        xaxis_title="Year",
        yaxis_title="Units",
    )
    st.plotly_chart(units_fig, use_container_width=True)

    st.divider()
    st.markdown("### Cash Flow Overview")
    cash_cols = st.columns(3)
    cash_cols[0].metric(
        "Operating Cash Flow", f"${cfo_series.get(final_year, 0.0):,.0f}", f"vs. ${cfo_series.get(first_year, 0.0):,.0f} in {first_year}"
    )
    cash_cols[1].metric(
        "Investing Cash Flow", f"${cfi_series.get(final_year, 0.0):,.0f}",
        f"vs. ${cfi_series.get(first_year, 0.0):,.0f} in {first_year}",
    )
    cash_cols[2].metric(
        "Financing Cash Flow", f"${cff_series.get(final_year, 0.0):,.0f}",
        f"vs. ${cff_series.get(first_year, 0.0):,.0f} in {first_year}",
    )

    final_index = years.index(final_year)
    opening_cash = cash_series.get(years[final_index - 1], 0.0) if final_index > 0 else 0.0
    closing_cash = cash_series.get(final_year, 0.0)
    waterfall_fig = go.Figure(
        go.Waterfall(
            name="Cash Movement",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=[
                "Opening Cash",
                "Operating Cash Flow",
                "Investing Cash Flow",
                "Financing Cash Flow",
                "Closing Cash",
            ],
            y=[
                opening_cash,
                cfo_series.get(final_year, 0.0),
                cfi_series.get(final_year, 0.0),
                cff_series.get(final_year, 0.0),
                closing_cash,
            ],
            connector={"line": {"color": "#7F8C8D"}},
        )
    )
    waterfall_fig.update_layout(
        showlegend=False,
        margin=dict(t=20, r=20, b=20, l=20),
        height=360,
    )
    st.plotly_chart(waterfall_fig, use_container_width=True)

    st.divider()
    st.markdown("### Working Capital Health")
    ar_values = [model.get("accounts_receivable", {}).get(year, 0.0) for year in years]
    inventory_values = [model.get("inventory", {}).get(year, 0.0) for year in years]
    ap_values = [model.get("accounts_payable", {}).get(year, 0.0) for year in years]
    accrued_values = [model.get("accrued_expenses", {}).get(year, 0.0) for year in years]
    nwc_values = [model.get("net_working_capital", {}).get(year, 0.0) for year in years]
    delta_nwc_values = [model.get("change_in_working_capital", {}).get(year, 0.0) for year in years]

    wc_df = pd.DataFrame(
        {
            "Year": years,
            "Accounts Receivable": _currency_series(ar_values),
            "Inventory": _currency_series(inventory_values),
            "Accounts Payable": _currency_series(ap_values),
            "Accrued Expenses": _currency_series(accrued_values),
            "Net Working Capital": _currency_series(nwc_values),
            "Δ Working Capital": _currency_series(delta_nwc_values),
        }
    )
    _render_table(wc_df, hide_index=True)

    wc_fig = make_subplots(specs=[[{"secondary_y": True}]])
    wc_fig.add_trace(
        go.Bar(
            name="Δ Working Capital",
            x=years,
            y=delta_nwc_values,
            marker_color="#8E44AD",
        ),
        secondary_y=False,
    )
    wc_fig.add_trace(
        go.Scatter(
            name="Net Working Capital",
            x=years,
            y=nwc_values,
            mode="lines+markers",
            line=dict(color="#2980B9", width=3),
        ),
        secondary_y=True,
    )
    wc_fig.update_layout(
        margin=dict(t=20, r=20, b=20, l=20),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    wc_fig.update_xaxes(title_text="Year")
    wc_fig.update_yaxes(title_text="Δ Working Capital", secondary_y=False)
    wc_fig.update_yaxes(title_text="Net Working Capital", secondary_y=True)
    st.plotly_chart(wc_fig, use_container_width=True)

    st.divider()
    st.markdown("### Product Mix and Pricing Insights")
    product_revenue = model.get("product_revenue", {})
    product_prices = model.get("product_prices", {})
    product_names = sorted(
        {product for year_map in product_revenue.values() for product in year_map.keys()}
    )
    mix_col, price_col = st.columns(2)

    with mix_col:
        if product_revenue and product_names:
            mix_fig = go.Figure()
            for product in product_names:
                mix_fig.add_trace(
                    go.Bar(
                        name=_product_label(product),
                        x=years,
                        y=[product_revenue.get(year, {}).get(product, 0.0) for year in years],
                    )
                )
            mix_fig.update_layout(
                barmode="stack",
                margin=dict(t=20, r=20, b=20, l=20),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            mix_fig.update_xaxes(title_text="Year")
            mix_fig.update_yaxes(title_text="Revenue (USD)")
            st.plotly_chart(mix_fig, use_container_width=True)
        else:
            st.info("Run the financial model to populate product mix insights.")

    with price_col:
        if product_prices and product_names:
            price_fig = go.Figure()
            for product in product_names:
                price_fig.add_trace(
                    go.Scatter(
                        name=_product_label(product),
                        x=years,
                        y=[product_prices.get(year, {}).get(product, 0.0) for year in years],
                        mode="lines+markers",
                    )
                )
            price_fig.update_layout(
                margin=dict(t=20, r=20, b=20, l=20),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            price_fig.update_xaxes(title_text="Year")
            price_fig.update_yaxes(title_text="Average Selling Price (USD)")
            st.plotly_chart(price_fig, use_container_width=True)
        else:
            st.info("Run the financial model to populate product pricing insights.")

    st.divider()
    st.markdown("### Labor and Productivity Snapshot")
    labor_metrics = model.get("labor_metrics", {})
    if labor_metrics:
        labor_years = sorted(labor_metrics.keys())
        latest_labor = labor_metrics.get(labor_years[-1], {})
        labor_cards = st.columns(3)
        labor_cards[0].metric(
            "Total Headcount",
            f"{latest_labor.get('total_headcount', 0):,.0f}",
        )
        labor_cards[1].metric(
            "Labor Cost",
            f"${latest_labor.get('total_labor_cost', 0.0):,.0f}",
        )
        per_head = 0.0
        if latest_labor.get("total_headcount"):
            per_head = latest_labor.get("total_labor_cost", 0.0) / max(
                latest_labor.get("total_headcount", 1), 1
            )
        labor_cards[2].metric("Cost per Head", f"${per_head:,.0f}")

        labor_df = pd.DataFrame(
            {
                "Year": labor_years,
                "Direct HC": [labor_metrics[y]["direct_headcount"] for y in labor_years],
                "Indirect HC": [labor_metrics[y]["indirect_headcount"] for y in labor_years],
                "Total HC": [labor_metrics[y]["total_headcount"] for y in labor_years],
                "Direct Labor Cost": _currency_series(
                    [labor_metrics[y]["direct_labor_cost"] for y in labor_years]
                ),
                "Indirect Labor Cost": _currency_series(
                    [labor_metrics[y]["indirect_labor_cost"] for y in labor_years]
                ),
                "Total Labor Cost": _currency_series(
                    [labor_metrics[y]["total_labor_cost"] for y in labor_years]
                ),
            }
        )
        _render_table(labor_df, hide_index=True)
    else:
        st.info("Connect the labor schedule to surface workforce insights on the dashboard.")

    st.divider()
    st.markdown("### Capital Structure and Debt Service")
    outstanding_debt = model.get("outstanding_debt", {})
    loan_repayment = model.get("loan_repayment", {})
    interest_payment = model.get("interest_payment", {})
    debt_draws = model.get("debt_draws", {})

    debt_fig = make_subplots(specs=[[{"secondary_y": True}]])
    debt_fig.add_trace(
        go.Bar(
            name="Debt Draws",
            x=years,
            y=[debt_draws.get(year, 0.0) for year in years],
            marker_color="#1F618D",
        ),
        secondary_y=False,
    )
    debt_fig.add_trace(
        go.Bar(
            name="Principal Paid",
            x=years,
            y=[loan_repayment.get(year, 0.0) for year in years],
            marker_color="#CB4335",
        ),
        secondary_y=False,
    )
    debt_fig.add_trace(
        go.Scatter(
            name="Ending Balance",
            x=years,
            y=[outstanding_debt.get(year, 0.0) for year in years],
            mode="lines+markers",
            line=dict(color="#2C3E50", width=3),
        ),
        secondary_y=True,
    )
    debt_fig.add_trace(
        go.Scatter(
            name="Interest Expense",
            x=years,
            y=[interest_payment.get(year, 0.0) for year in years],
            mode="lines",
            line=dict(color="#7D3C98", dash="dash"),
        ),
        secondary_y=False,
    )
    debt_fig.update_layout(
        barmode="group",
        margin=dict(t=20, r=20, b=20, l=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    debt_fig.update_xaxes(title_text="Year")
    debt_fig.update_yaxes(title_text="Cash Flow (USD)", secondary_y=False)
    debt_fig.update_yaxes(title_text="Outstanding Debt (USD)", secondary_y=True)
    st.plotly_chart(debt_fig, use_container_width=True)

    st.divider()
    st.markdown("### Income Statement Snapshot")
    snapshot = income_df.copy()
    snapshot["Revenue"] = _currency_series(snapshot["Revenue"])
    snapshot["COGS"] = _currency_series(snapshot["COGS"])
    snapshot["Opex"] = _currency_series(snapshot["Opex"])
    snapshot["Net Profit"] = _currency_series(snapshot["Net Profit"])
    _render_table(snapshot[["Year", "Revenue", "COGS", "Opex", "Net Profit"]])

    st.markdown("### Cash Flow Snapshot")
    cash_snapshot = cashflow_df.copy()
    cash_snapshot["CFO"] = _currency_series(cash_snapshot["CFO"])
    cash_snapshot["CFI"] = _currency_series(cash_snapshot["CFI"])
    cash_snapshot["CFF"] = _currency_series(cash_snapshot["CFF"])
    cash_snapshot["Net Cash Flow"] = _currency_series(cash_snapshot["Net Cash Flow"])
    cash_snapshot["Closing Cash"] = _currency_series(cash_snapshot["Closing Cash"])
    _render_table(cash_snapshot)

    st.markdown("### Balance Sheet Snapshot")
    balance_snapshot = balance_df.copy()
    for column in [
        "Fixed Assets",
        "Current Assets",
        "Total Assets",
        "Current Liabilities",
        "Long Term Debt",
        "Total Equity",
        "Total Liabilities + Equity",
    ]:
        balance_snapshot[column] = _currency_series(balance_snapshot[column])
    balance_snapshot["Balanced?"] = [
        "Yes" if value else "Check" for value in balance_snapshot["Balanced?"]
    ]
    _render_table(balance_snapshot)


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
    capex_manager: CapexScheduleManager = st.session_state["capex_manager"]
    years = _projection_years(model)

    if not years:
        st.info("Run the financial model to populate the reports page.")
        return

    st.markdown("### Executive KPI Summary")
    kpis = _reports_kpi_summary(model, years)
    if kpis:
        kpi_columns = st.columns(len(kpis))
        for column, kpi in zip(kpi_columns, kpis):
            column.metric(kpi["label"], kpi["value"], kpi.get("delta", ""))

    st.markdown("### Data Quality & Notes")
    quality_notes = _reports_quality_notes(model, capex_manager, years)
    if quality_notes:
        for note in quality_notes:
            st.write(f"- {note}")
    else:
        st.write("All schedules are aligned with the current configuration.")

    selected_years, show_yoy, show_charts = _report_controls(years)

    forecast_display = _forecast_schedule(model, formatted=True)
    forecast_raw = _forecast_schedule(model, formatted=False)
    labor_display = generate_labor_statement(model)
    labor_raw = _labor_schedule_raw(model, years)
    capex_spend_display = _capex_spend_schedule(capex_manager, cfg)
    capex_spend_raw = _capex_spend_raw(capex_manager, cfg, years)
    debt_display = _debt_schedule(model)
    debt_raw = _debt_schedule_raw(model, years)

    st.markdown("### Export Schedules")
    export_tables = {
        "Financial Forecast Schedule": forecast_raw,
        "Labor Cost Schedule": labor_raw,
        "CAPEX Spend Schedule": capex_spend_raw,
        "Debt Amortization Schedule": debt_raw,
    }
    _reports_export_controls(export_tables, selected_years)

    schedules = [
        {
            "title": "Financial Forecast Schedule",
            "display": forecast_display,
            "note": "Revenue, profitability, and liquidity outlook.",
            "raw": forecast_raw,
            "columns": ["Revenue", "Operating Expenses", "Net Profit", "Closing Cash"],
        },
        {
            "title": "Labor Cost Schedule",
            "display": labor_display,
            "note": "Labor schedule replicated for reporting completeness.",
            "raw": labor_raw,
            "columns": ["Total Labor Cost", "Total Headcount"],
        },
        {
            "title": "CAPEX Spend Schedule",
            "display": capex_spend_display,
            "note": "Capital expenditures included in consolidated reports.",
            "raw": capex_spend_raw,
            "columns": ["Capital Spend"],
        },
        {
            "title": "Debt Amortization Schedule",
            "display": debt_display,
            "note": "Debt schedule replicated for reporting completeness.",
            "raw": debt_raw,
            "columns": ["Debt Draw", "Principal Repayment", "Outstanding Debt", "Interest Payment"],
        },
    ]

    for schedule in schedules:
        filtered_display = _filter_schedule_years(schedule["display"], selected_years)
        _display_schedule(schedule["title"], filtered_display, schedule["note"])

        series_map = _series_map_from_dataframe(schedule["raw"], schedule["columns"])
        commentary = _schedule_commentary(series_map, selected_years)
        if commentary:
            st.caption(commentary)

        if show_yoy:
            yoy_df = _build_yoy_dataframe(series_map, selected_years)
            if not yoy_df.empty:
                st.markdown("_Year-over-year variance_")
                _render_table(_format_yoy_dataframe(yoy_df), hide_index=True)

        if show_charts:
            chart = _build_schedule_chart(schedule["title"], series_map, selected_years)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

        st.divider()


def _render_financial_forecast(model: Dict[str, Any]) -> None:
    st.markdown("## Financial Model Schedules")
    _render_financial_model(model)

    st.markdown("## Reports & Analytics")
    _render_reports(model)


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
                utilization_inputs[year] = horizon_form.number_input(
                    f"Utilization {year}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg.capacity_utilization.get(year, 0.0)),
                    step=0.01,
                    help="Adjust utilization using the +/- controls for the selected year.",
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

        st.markdown("#### Variable Production Cost Schedule")
        variable_cost_df = _variable_production_cost_schedule(cfg, model)
        if variable_cost_df.empty:
            st.write("Run the financial model to populate variable production cost projections across the horizon.")
        _render_table(variable_cost_df, hide_index=True)
        _render_variable_cost_editor(cfg, model)

        st.markdown("#### Fixed Manufacturing Cost Schedule")
        fixed_cost_df = _fixed_manufacturing_cost_schedule(cfg, model)
        if fixed_cost_df.empty:
            st.write("Run the financial model to populate fixed manufacturing cost projections across the horizon.")
        _render_table(fixed_cost_df, hide_index=True)
        _render_fixed_cost_editor(cfg, model)

        st.markdown("#### Other Operating Cost Schedule")
        other_cost_df = _other_operating_cost_schedule(cfg, model)
        if other_cost_df.empty:
            st.write("Run the financial model to populate other operating cost projections across the horizon.")
        _render_table(other_cost_df, hide_index=True)
        _render_other_operating_cost_editor(cfg, model)

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

def _render_navigation() -> Dict[str, DeltaGenerator]:
    """Render the primary navigation as tabs and return the tab containers."""

    pages = [
        "Dashboard",
        "Advanced Analytics",
        "Financial Forecast",
        "Platform Settings",
        "AI & Machine Learning",
    ]

    tabs = st.tabs(pages)
    return {page: tab for page, tab in zip(pages, tabs)}


def main() -> None:
    st.set_page_config(
        page_title="Automobile Manufacturing Financial Platform",
        layout="wide",
    )

    _ensure_state()
    model = st.session_state["financial_model"]

    st.title("Automobile Manufacturing Financial Platform")
    st.caption("Comprehensive labor, CAPEX, and financial planning environment.")

    tabs = _render_navigation()

    with tabs["Dashboard"]:
        _render_dashboard(model)

    with tabs["Advanced Analytics"]:
        _render_advanced_analytics(model)

    with tabs["Financial Forecast"]:
        _render_financial_forecast(model)

    with tabs["Platform Settings"]:
        _render_platform_settings()

    with tabs["AI & Machine Learning"]:
        _render_ai_settings(st.session_state["ai_payload"])


if __name__ == "__main__":
    main()
