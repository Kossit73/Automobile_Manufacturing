# Labor Management System - Complete User Guide

## Overview

The Labor Management System provides enterprise-grade workforce planning, scheduling, and cost management integrated with your financial model. Features include:

- **CRUD Operations**: Create, Read, Update, Delete positions with full validation
- **Direct & Indirect Labor**: Separate tracking for production and support staff
- **Multi-Year Scheduling**: Project labor costs across 5+ years with salary growth
- **Production Linking**: Automatically align headcount to production forecasts
- **Financial Integration**: Labor costs automatically flow into OPEX and valuation
- **What-If Analysis**: Test salary adjustments and staffing scenarios instantly
- **Comprehensive Reporting**: Detailed schedules, summaries, and variance analysis

---

## System Architecture

### 1. Core Components

```
labor_management.py
├── LaborPosition (dataclass)
│   └── Individual position record with all employment details
├── LaborScheduleManager
│   ├── CRUD operations for positions
│   ├── Headcount and cost queries
│   └── Multi-year projections
├── LaborCostSchedule
│   ├── 5-year cost schedules
│   ├── Detailed position breakdown
│   └── Category summaries
├── ProductionLinkedLabor
│   ├── Labor requirement calculations
│   └── Production-aligned headcount
└── LaborVarianceAnalysis
    └── Budget vs. actual comparisons
```

### 2. Integration with Financial Model

The labor system integrates via:

```python
# In CompanyConfig (financial_model.py)
labor_manager: Optional[LaborScheduleManager] = None

# OPEX calculation automatically pulls labor costs if manager provided
opex = calculate_opex_with_labor_manager(years, cfg)
# vs.
opex = calculate_opex(years, cfg)  # Legacy payroll method
```

---

## CRUD OPERATIONS GUIDE

### CREATE - Adding Positions

#### Basic Add

```python
from labor_management import LaborScheduleManager, LaborType, JobCategory

manager = LaborScheduleManager()

# Add assembly line workers
pos_id = manager.add_position(
    position_name="Assembly Line Workers",
    labor_type=LaborType.DIRECT,
    job_category=JobCategory.ASSEMBLY,
    headcount=12,
    annual_salary=36000,
    benefits_percent=0.25,
    overtime_hours_annual=200,
    training_cost=2000
)
# Returns: "POS_D_001"
```

#### Advanced Add with All Parameters

```python
manager.add_position(
    position_name="Robotic System Operators",
    labor_type=LaborType.DIRECT,
    job_category=JobCategory.ASSEMBLY,
    headcount=4,
    annual_salary=55000,
    status=EmploymentStatus.ACTIVE,           # ACTIVE, INACTIVE, SEASONAL
    start_year=2027,                          # When position starts
    end_year=None,                            # When position ends (None = ongoing)
    benefits_percent=0.25,                    # 25% of salary for benefits
    overtime_hours_annual=100,                # Annual overtime hours
    overtime_rate=1.5,                        # 1.5x base rate
    training_cost=8000,                       # Annual training cost
    equipment_cost=5000,                      # Annual equipment cost
    notes="Automation for capacity expansion"
)
```

#### Labor Type Classifications

```
DIRECT LABOR:
├── Assembly
├── Welding
├── Painting
├── Quality Control
└── Material Handling

INDIRECT LABOR:
├── Production Management
├── Quality Assurance
├── Maintenance
├── Planning & Scheduling
├── Supervision
├── Administration
├── Human Resources
├── Finance
└── Sales & Marketing
```

### READ - Querying Positions

#### Get Single Position

```python
pos = manager.get_position('POS_D_001')
print(pos.position_name)        # Assembly Line Workers
print(pos.headcount)             # 12
print(pos.annual_salary)         # 36000
```

#### Get All Positions

```python
# Get all positions
all_positions = manager.get_all_positions()

# Get filtered positions
direct_only = manager.get_all_positions(labor_type=LaborType.DIRECT)
active_only = manager.get_all_positions(status=EmploymentStatus.ACTIVE)
```

#### Get Summary DataFrame

```python
summary_df = manager.get_position_summary()
# Returns DataFrame with all positions:
# Position ID | Position Name | Type | Category | Headcount | Annual Salary | Status | Start Year | End Year
```

#### Headcount Queries

```python
# Get headcount by type for specific year
hc_by_type = manager.get_headcount_by_type(2026)
# Returns: {'Direct': 35, 'Indirect': 13}

# Get total headcount
total = manager.get_total_headcount(2026)
# Returns: 48
```

#### Labor Cost Queries

```python
# Get labor costs by type for specific year
costs = manager.get_labor_cost_by_type(2026, salary_growth=0.05)
# Returns: {'Direct': 1874888, 'Indirect': 897630}

# Get labor costs by job category
category_costs = manager.get_labor_cost_by_category(2026, salary_growth=0.05)
# Returns: {
#   'Assembly': 802812,
#   'Welding': 489800,
#   'Finance': 96000,
#   ...
# }
```

### UPDATE - Editing Positions

#### Edit Single Field

```python
# Increase headcount
manager.edit_position('POS_D_001', headcount=15)

# Increase salary
manager.edit_position('POS_D_001', annual_salary=38000)

# Update status
manager.edit_position('POS_D_001', status=EmploymentStatus.SEASONAL)
```

#### Edit Multiple Fields

```python
manager.edit_position(
    'POS_D_001',
    headcount=16,
    annual_salary=39000,
    benefits_percent=0.28,
    overtime_hours_annual=300,
    training_cost=3000,
    notes="Upgraded tier with expanded benefits"
)
```

#### Editable Fields

```
position_name              - Position title
headcount                  - Number of employees (≥0)
annual_salary              - Base salary (≥0)
status                     - Employment status
end_year                   - End of employment
benefits_percent           - Percent of salary (0-1)
overtime_hours_annual      - Annual OT hours
overtime_rate              - Multiplier (usually 1.5)
training_cost_annual       - Annual training spend
equipment_cost_annual      - Annual equipment spend
notes                      - Position notes
```

#### Validation

```python
# These will raise ValueError:
manager.edit_position('POS_D_001', headcount=-5)           # Negative headcount
manager.edit_position('POS_D_001', annual_salary=-1000)    # Negative salary
manager.edit_position('POS_D_001', benefits_percent=1.5)   # Out of range
```

### DELETE - Removing Positions

#### Hard Delete

```python
# Completely remove position
manager.remove_position('POS_D_001')
# Returns: True
# Output: "✓ Position removed: POS_D_001 - Assembly Line Workers"
```

#### Soft Delete (Mark Inactive)

```python
# Better practice: mark inactive with end date
manager.mark_inactive('POS_D_003', end_year=2028)
# Position keeps history but excludes from future years
# Status: INACTIVE
# End Year: 2028
```

---

## LABOR COST SCHEDULING

### 5-Year Cost Schedule

```python
from labor_management import LaborCostSchedule

schedule = LaborCostSchedule(manager)

# Generate 5-year forecast
df = schedule.generate_5year_schedule(
    start_year=2026,
    salary_growth=0.05  # 5% annual growth
)

# DataFrame columns:
# Year | Direct Labor HC | Direct Labor Cost | Indirect Labor HC | 
# Indirect Labor Cost | Total Headcount | Total Labor Cost | Avg Cost per HC
```

### Detailed Position-Level Schedule

```python
# Get detailed breakdown for specific year
detailed_2026 = schedule.generate_detailed_schedule(2026)

# DataFrame columns:
# Position ID | Position Name | Type | Category | Headcount | Base Salary |
# Total Salary | Benefits | Overtime | Training | Equipment | Total Cost
```

### Category Summary

```python
# Get breakdown by job category for specific year
summary = schedule.generate_category_summary(2026)

# DataFrame columns:
# Job Category | Annual Cost | Percentage of Total
```

---

## PRODUCTION-LINKED LABOR FORECASTING

### Calculate Required Labor

```python
from labor_management import ProductionLinkedLabor

# Direct labor requirement
required = ProductionLinkedLabor.calculate_direct_labor_requirement(
    production_volume=20000,      # Units per year
    labor_hours_per_unit=3.5,     # Hours per unit
    working_days=300,             # Days per year
    hours_per_day=8.0             # Hours per day
)
# Returns: 29.17 (required headcount)
```

### Calculate Indirect/Direct Ratio

```python
# Typical manufacturing: indirect = 30-40% of direct
indirect = ProductionLinkedLabor.calculate_indirect_labor_ratio(
    direct_headcount=30,
    ratio=0.35  # 35% ratio
)
# Returns: 10.5 indirect headcount
```

### Create Full Labor Plan

```python
production_volumes = {
    2026: 10000,
    2027: 14000,
    2028: 18000,
    2029: 20000,
    2030: 20000
}

labor_plan = ProductionLinkedLabor.create_labor_plan(
    production_volumes=production_volumes,
    labor_hours_per_unit=3.5,
    working_days=300,
    hours_per_day=8.0,
    indirect_ratio=0.35
)

# DataFrame shows:
# Year | Production Volume | Direct Labor HC (Required) | 
# Indirect Labor HC (Required) | Total HC (Required)

# 2026: 10,000 units → 14.6 direct, 5.1 indirect, 19.7 total
# 2027: 14,000 units → 20.4 direct, 7.1 indirect, 27.6 total
# 2028: 18,000 units → 26.2 direct, 9.2 indirect, 35.4 total
# 2029: 20,000 units → 29.2 direct, 10.2 indirect, 39.4 total
# 2030: 20,000 units → 29.2 direct, 10.2 indirect, 39.4 total
```

---

## FINANCIAL MODEL INTEGRATION

### Setup Integration

```python
from financial_model import CompanyConfig, run_financial_model
from labor_management import LaborScheduleManager, initialize_default_labor_structure

# Create labor manager
labor_manager = initialize_default_labor_structure()

# Attach to config
cfg = CompanyConfig(labor_manager=labor_manager)

# Run model - labor costs automatically included in OPEX
model_data = run_financial_model(cfg)
```

### Access Labor Metrics in Model

```python
# Labor metrics automatically included for each year
labor_metrics = model_data['labor_metrics']

for year in model_data['years']:
    metrics = labor_metrics[year]
    
    print(f"{year}:")
    print(f"  Direct HC: {metrics['direct_headcount']}")
    print(f"  Indirect HC: {metrics['indirect_headcount']}")
    print(f"  Total HC: {metrics['total_headcount']}")
    print(f"  Direct Labor Cost: ${metrics['direct_labor_cost']:,.0f}")
    print(f"  Indirect Labor Cost: ${metrics['indirect_labor_cost']:,.0f}")
    print(f"  Total Labor Cost: ${metrics['total_labor_cost']:,.0f}")
```

### Generate Labor Statement

```python
from financial_model import generate_labor_statement

labor_df = generate_labor_statement(model_data)
print(labor_df)

# Columns: Year | Direct Labor HC | Direct Labor Cost | Indirect Labor HC |
# Indirect Labor Cost | Total Headcount | Total Labor Cost
```

### Verify Integration

```python
# Labor costs should appear in OPEX
total_opex_2026 = model_data['opex'][2026]

# With labor manager:
# OPEX = Marketing + Labor Costs
# OPEX = $72,000 + $2,635,430 = $2,707,430

# Without labor manager:
# OPEX = Marketing + (Headcount × Salary)
# OPEX = $72,000 + (50 × $5,000 × 12) = $3,072,000
```

---

## WHAT-IF & SCENARIO ANALYSIS

### Salary Adjustment Scenario

```python
# Before
costs_before = manager.get_labor_cost_by_type(2026, 0.05)
total_before = costs_before['Direct'] + costs_before['Indirect']

# Apply 5% increase to all direct labor
direct_positions = manager.get_all_positions(labor_type=LaborType.DIRECT)
for pos in direct_positions:
    new_salary = pos.annual_salary * 1.05
    manager.edit_position(pos.position_id, annual_salary=new_salary)

# After
costs_after = manager.get_labor_cost_by_type(2026, 0.05)
total_after = costs_after['Direct'] + costs_after['Indirect']

impact = total_after - total_before
print(f"Impact: +${impact:,.0f} ({impact/total_before*100:.1f}%)")
```

### Headcount Expansion Scenario

```python
# Scale up assembly headcount by 25%
old_hc = manager.get_position('POS_D_001').headcount
new_hc = int(old_hc * 1.25)

manager.edit_position('POS_D_001', headcount=new_hc)

# Recalculate costs
new_costs = manager.get_labor_cost_by_type(2026, 0.05)
print(f"New labor cost: ${new_costs['Direct'] + new_costs['Indirect']:,.0f}")
```

### Add New Position Scenario

```python
# Add robotics team starting in 2027
manager.add_position(
    position_name="Robotics Engineers",
    labor_type=LaborType.DIRECT,
    job_category=JobCategory.ASSEMBLY,
    headcount=5,
    annual_salary=65000,
    start_year=2027,
    training_cost=15000
)

# View impact on 2027
costs_2027 = manager.get_labor_cost_by_type(2027, 0.05)
```

---

## REPORTING & EXPORT

### Summary Report

```python
from labor_management import LaborCostSchedule

schedule = LaborCostSchedule(manager)

# 5-year summary
print(schedule.generate_5year_schedule().to_string())

# Category breakdown for 2026
print(schedule.generate_category_summary(2026).to_string())
```

### Export to CSV

```python
import pandas as pd

# Export various reports
summary_df = manager.get_position_summary()
summary_df.to_csv('labor_positions.csv', index=False)

cost_schedule = LaborCostSchedule(manager)
cost_schedule.generate_5year_schedule().to_csv('labor_forecast.csv', index=False)

cost_schedule.generate_detailed_schedule(2026).to_csv('labor_detail_2026.csv', index=False)
```

### Export to Excel

```python
# Write multiple sheets to Excel
with pd.ExcelWriter('labor_report.xlsx') as writer:
    manager.get_position_summary().to_excel(writer, sheet_name='Positions', index=False)
    cost_schedule.generate_5year_schedule().to_excel(writer, sheet_name='5-Year Forecast', index=False)
    cost_schedule.generate_detailed_schedule(2026).to_excel(writer, sheet_name='2026 Detail', index=False)
    cost_schedule.generate_category_summary(2026).to_excel(writer, sheet_name='2026 Categories', index=False)
```

---

## BEST PRACTICES

### 1. Organized Position IDs

- Automatically generated: `POS_D_001` (Direct) or `POS_I_006` (Indirect)
- Track positions by category for easy reference

### 2. Salary Growth Rates

```python
# Most calculations default to 5% annual growth
# Customize as needed:

# Conservative 2% growth
costs = manager.get_labor_cost_by_type(2026, salary_growth=0.02)

# Aggressive 8% growth
costs = manager.get_labor_cost_by_type(2026, salary_growth=0.08)
```

### 3. Benefit Packages

```python
# Standard manufacturing
benefits_percent = 0.25  # 25% of salary

# Premium indirect roles
benefits_percent = 0.30  # 30% of salary

# Seasonal/part-time
benefits_percent = 0.15  # 15% of salary
```

### 4. Production Planning

```python
# Always validate labor plan against production
actual_hc = manager.get_total_headcount(2026)
required_hc = ProductionLinkedLabor.calculate_direct_labor_requirement(...)

if actual_hc < required_hc:
    print(f"⚠️ Staffing gap: need {required_hc - actual_hc} more")
elif actual_hc > required_hc * 1.1:
    print(f"⚠️ Over-staffed: {actual_hc - required_hc} excess")
else:
    print(f"✓ Staffing optimized")
```

### 5. Version Control

```python
# Save labor plans with dates for comparison
import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
schedule.generate_5year_schedule().to_csv(f'labor_plan_{timestamp}.csv')
```

---

## TROUBLESHOOTING

### Issue: Position not found

```python
# Error: "Position POS_D_999 not found"
# Solution: Get valid position IDs
positions = manager.get_all_positions()
for pos in positions:
    print(pos.position_id)
```

### Issue: Labor costs don't match expectations

```python
# Check salary growth is applied correctly
costs_no_growth = manager.get_labor_cost_by_type(2026, salary_growth=0.0)
costs_with_growth = manager.get_labor_cost_by_type(2026, salary_growth=0.05)
```

### Issue: Financial model not using labor costs

```python
# Verify labor_manager is attached
cfg = model_data['config']
if cfg.labor_manager is None:
    print("⚠️ Labor manager not attached - using legacy payroll")
else:
    print("✓ Labor manager integrated")
```

---

## API Reference

### LaborScheduleManager

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `add_position()` | See CREATE section | position_id (str) | Add new position |
| `get_position()` | position_id | LaborPosition | Retrieve single position |
| `get_all_positions()` | labor_type, status | List[LaborPosition] | Retrieve filtered positions |
| `get_position_summary()` | - | DataFrame | Summary of all positions |
| `edit_position()` | position_id, **kwargs | bool | Update position fields |
| `remove_position()` | position_id | bool | Delete position |
| `mark_inactive()` | position_id, end_year | bool | Phase out position |
| `get_headcount_by_type()` | year | Dict | HC by Direct/Indirect |
| `get_total_headcount()` | year | int | Total HC for year |
| `get_labor_cost_by_type()` | year, salary_growth | Dict | Costs by Direct/Indirect |
| `get_labor_cost_by_category()` | year, salary_growth | Dict | Costs by job category |

### LaborCostSchedule

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `generate_5year_schedule()` | start_year, salary_growth | DataFrame | Multi-year cost forecast |
| `generate_detailed_schedule()` | year | DataFrame | Position-level detail |
| `generate_category_summary()` | year | DataFrame | Category breakdown |

### ProductionLinkedLabor

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `calculate_direct_labor_requirement()` | production, hours, days, hours/day | float | Required HC |
| `calculate_indirect_labor_ratio()` | direct_hc, ratio | float | Indirect HC needed |
| `create_labor_plan()` | production_vols, params | DataFrame | Full labor plan |

---

## Version Info

- **Version**: 1.0
- **Last Updated**: November 13, 2025
- **Status**: Production Ready ✅
- **Python**: 3.7+
- **Dependencies**: pandas, numpy, dataclasses

---

**For questions or feature requests, see ADVANCED_ANALYTICS_GUIDE.md for additional capabilities.**
