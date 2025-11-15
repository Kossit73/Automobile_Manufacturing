# Platform Settings “Add Row” Functionality Review

## Overview
Platform Settings schedules use the shared `_schedule_toolbar` helper to surface an **Add Row** control above each table. When a user selects a year and submits **Add Row**, the toolbar invokes schedule-specific logic to extend the configuration and then reruns the financial model.

## Current Behaviour
- The toolbar only offers years that already exist in the schedule; clicking **Add Row** clones the selected year’s values into the next available year via `_next_available_year` before refreshing the model.【F:streamlit_app.py†L1483-L1509】
- Individual editors (production horizon, cost structure, etc.) handle the new year by copying the preceding year’s assumptions into the relevant override dictionaries and calling `cfg.__post_init__()` to re-normalise the configuration.【F:streamlit_app.py†L1502-L1507】【F:streamlit_app.py†L2182-L2194】

## Pain Points
1. **No chance to define the new row before it is created.** Users must add the row, locate it in the table, and then edit the cloned values in a second step, which increases friction and reruns the model twice for a single change.【F:streamlit_app.py†L1502-L1508】
2. **Only sequential years can be added.** Because `_next_available_year` increments the selected year until it finds a gap, users cannot insert interim years or add multiple future years at once without repeating the entire flow for each addition.【F:streamlit_app.py†L1490-L1504】
3. **Defaults ignore context from other schedules.** Cost-related schedules always seed the new year with the prior year’s totals, even when downstream drivers (e.g., product mix or labour overrides) would suggest different starting points.【F:streamlit_app.py†L2182-L2194】
4. **Minimal feedback about the new row.** Aside from a success toast, the interface does not highlight the inserted row or present its cloned values, making it easy to miss accidental additions—especially on long horizons.【F:streamlit_app.py†L1506-L1508】

## Recommendations
1. **Collect input before adding the row.** Replace the simple form submit with a dialog or inline form that captures the target year and key values (e.g., utilisation, budgets) before writing to the configuration. Persisting the edits in one step would halve the number of model reruns and reduce user churn.
2. **Support batch inserts and custom years.** Extend `_schedule_toolbar` so users can specify how many years to add and the exact year value. Allowing multiple additions in a single action will speed up horizon extensions and avoid repetitive submissions.
3. **Seed smarter defaults.** Instead of blindly copying the previous year, draw initial values from the underlying drivers—such as utilising forecast outputs for production or marketing curves—so new rows start closer to expected targets.
4. **Surface contextual confirmation.** After adding a row, scroll the table to the new entry or display a summary card showing the seeded assumptions. This provides immediate feedback and an obvious entry point for further edits.

Implementing these adjustments will make the Add Row workflow quicker, reduce accidental changes, and better align newly inserted years with the rest of the financial model.
