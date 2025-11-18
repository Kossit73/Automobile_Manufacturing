# Model Workflow Overview

This overview summarizes how the automobile manufacturing financial model assembles forecasts and statements from the shared configuration.

## 1. Configuration and Input Normalization
- `CompanyConfig` centralizes horizon, production capacity, marketing plan, financing, and working-capital drivers.
- During initialization the configuration forward-fills capacity utilization, clamps the production window, and seeds annual marketing budgets.
- Optional overrides for operating expenses, labor cost, COGS, loan repayment, and other costs are normalized into dictionaries so downstream calculations can safely consume them.

## 2. Production, Pricing, and Revenue
- `calculate_production_forecast` translates normalized capacity utilization into total units, applies the product mix and selling prices, and returns per-year revenue across the projection window.

## 3. Cost Structure and Operating Expenses
- `calculate_cogs` builds variable production costs per year, honoring any overrides.
- `calculate_opex_breakdown` splits operating expenses into marketing, labor/payroll, and other operating costs, using the labor manager when attached and applying overrides when present.

## 4. Working Capital and Operating Cash Flow
- `calculate_working_capital_positions` derives receivables, inventory, payables, accrued expenses, net working capital, and year-over-year changes using the configured turnover-day drivers.
- Operating cash flow (`cfo`) adds depreciation back to net profit and subtracts the modeled working-capital deltas to translate earnings into cash.

## 5. Financing, CAPEX, and Cash Movement
- Financing cash flow (`cff`) layers equity and loan funding in the first year and subtracts scheduled principal and interest in later years, respecting repayment overrides.
- Investment cash flow (`cfi`) pulls annual spend from the CAPEX manager when available, otherwise applies the legacy lump-sum assumption.
- `calculate_cash_flow` rolls operating, investing, and financing cash flows into cumulative cash balances across the horizon.

## 6. Financial Statements and Valuation
- `calculate_income_statement` combines revenue, COGS, operating expenses, and depreciation into EBITDA, EBIT, taxes, and net profit.
- `calculate_dcf` converts EBIT into free cash flow, discounts each year, and computes a terminal value to estimate enterprise value.
- `calculate_balance_sheet` builds assets, liabilities, and equity using cash balances, working-capital accounts, fixed assets, and outstanding debt.

## 7. End-to-End Orchestration and Outputs
- `run_financial_model` orchestrates production, cost, working-capital, financing, valuation, and balance-sheet calculations, returning a dictionary of schedules for downstream views.
- Helper functions such as `generate_financial_statements` and `generate_labor_statement` package the outputs into income statement, cash flow, balance sheet, and labor tables for presentation layers.
