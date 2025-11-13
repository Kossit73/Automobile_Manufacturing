"""
Integrated Labor Management & Financial Model Demonstration
Shows full CRUD operations and model integration
"""

from financial_model import CompanyConfig, run_financial_model, generate_financial_statements, generate_labor_statement
from labor_management import (
    LaborScheduleManager, LaborCostSchedule, ProductionLinkedLabor,
    initialize_default_labor_structure, LaborType, JobCategory, EmploymentStatus
)
import pandas as pd

print("\n" + "="*100)
print("INTEGRATED LABOR MANAGEMENT & FINANCIAL MODEL SYSTEM")
print("="*100)

# =====================================================
# PART 1: LABOR MANAGEMENT CRUD OPERATIONS
# =====================================================

print("\n" + "-"*100)
print("PART 1: LABOR MANAGEMENT CRUD OPERATIONS")
print("-"*100)

# Initialize labor structure
labor_manager = initialize_default_labor_structure()

# Shared financial configuration with labor integration
cfg_with_labor = CompanyConfig(labor_manager=labor_manager)
projection_years = [cfg_with_labor.start_year + i for i in range(cfg_with_labor.projection_years)]
final_year = projection_years[-1]
base_year = cfg_with_labor.start_year

print("\nCURRENT LABOR SCHEDULE:")
print(labor_manager.get_position_summary().to_string(index=False))

# ===== CREATE: Add new positions =====
print("\n\nCREATE OPERATION - Adding new positions:")
print("-" * 100)

pos_id_1 = labor_manager.add_position(
    position_name="Robotic System Operators",
    labor_type=LaborType.DIRECT,
    job_category=JobCategory.ASSEMBLY,
    headcount=4,
    annual_salary=55000,
    benefits_percent=0.25,
    start_year=2027,
    training_cost=8000,
    notes="New automation specialists for capacity expansion"
)

pos_id_2 = labor_manager.add_position(
    position_name="Supply Chain Manager",
    labor_type=LaborType.INDIRECT,
    job_category=JobCategory.PLANNING_SCHEDULING,
    headcount=1,
    annual_salary=72000,
    benefits_percent=0.30,
    training_cost=5000,
    notes="Logistics optimization role"
)

# ===== READ: Query positions =====
print("\n\nREAD OPERATIONS:")
print("-" * 100)

print(f"\nHeadcount Summary ({base_year}):")
hc_by_type = labor_manager.get_headcount_by_type(base_year)
print(f"  Direct Labor:   {hc_by_type['Direct']:3d} employees")
print(f"  Indirect Labor: {hc_by_type['Indirect']:3d} employees")
print(f"  Total:          {labor_manager.get_total_headcount(base_year):3d} employees")

print(f"\nLabor Cost Summary ({base_year}):")
costs_by_type = labor_manager.get_labor_cost_by_type(base_year)
print(f"  Direct Labor Cost:   ${costs_by_type['Direct']:>15,.0f}")
print(f"  Indirect Labor Cost: ${costs_by_type['Indirect']:>15,.0f}")
print(f"  Total Labor Cost:    ${costs_by_type['Direct'] + costs_by_type['Indirect']:>15,.0f}")

print("\nHeadcount Projection ({}-{}):".format(projection_years[0], projection_years[-1]))
for year in projection_years:
    total_hc = labor_manager.get_total_headcount(year)
    print(f"  {year}: {total_hc:3d} employees")

# ===== UPDATE: Edit positions =====
print("\n\nUPDATE OPERATION - Editing positions:")
print("-" * 100)

print(f"\nBefore Edit - Assembly Line Positions:")
old_pos = labor_manager.get_position('POS_D_001')
print(f"  Headcount: {old_pos.headcount}")
print(f"  Annual Salary: ${old_pos.annual_salary:,.0f}")

labor_manager.edit_position(
    'POS_D_001',
    headcount=14,
    annual_salary=38500,
    overtime_hours_annual=250,
    notes="Increased capacity with higher-paid workers"
)

print(f"\nAfter Edit - Assembly Line Positions:")
new_pos = labor_manager.get_position('POS_D_001')
print(f"  Headcount: {new_pos.headcount}")
print(f"  Annual Salary: ${new_pos.annual_salary:,.0f}")
print(f"  Overtime Hours: {new_pos.overtime_hours_annual}")

# ===== DELETE: Mark as inactive =====
print("\n\nDELETE OPERATION - Phasing out positions:")
print("-" * 100)

print(f"\nMarking Painters (POS_D_003) as inactive starting 2028")
print(f"(This represents phase-out, not immediate removal)")
labor_manager.mark_inactive('POS_D_003', end_year=2028)
print(f"  Status updated to: {labor_manager.get_position('POS_D_003').status.value}")
print("  End year set to: 2028")

# =====================================================
# PART 2: LABOR COST SCHEDULES
# =====================================================

print("\n\n" + "-"*100)
print("PART 2: DETAILED LABOR COST SCHEDULES")
print("-"*100)

cost_schedule = LaborCostSchedule(labor_manager)

print("\n5-YEAR LABOR COST SCHEDULE:")
five_year_schedule = cost_schedule.generate_5year_schedule()
print(five_year_schedule.to_string(index=False))

print(f"\nDETAILED LABOR COST SCHEDULE ({base_year}):")
detailed_2026 = cost_schedule.generate_detailed_schedule(base_year)
print(detailed_2026.to_string(index=False))

print(f"\nDETAILED LABOR COST SCHEDULE ({base_year + 1} - With Additions):")
detailed_2027 = cost_schedule.generate_detailed_schedule(base_year + 1)
print(detailed_2027.to_string(index=False))

print(f"\nJOB CATEGORY BREAKDOWN ({base_year}):")
category_summary = cost_schedule.generate_category_summary(base_year)
print(category_summary.to_string(index=False))

# =====================================================
# PART 3: PRODUCTION-LINKED LABOR FORECAST
# =====================================================

print("\n\n" + "-"*100)
print("PART 3: PRODUCTION-LINKED LABOR FORECASTING")
print("-"*100)

production_base = [10000, 14000, 18000, 20000, 20000]
production_volumes = {}
for idx, year in enumerate(projection_years):
    baseline = production_base[idx] if idx < len(production_base) else production_base[-1]
    production_volumes[year] = baseline

print("\nPRODUCTION vs. REQUIRED LABOR:")
labor_plan = ProductionLinkedLabor.create_labor_plan(
    production_volumes=production_volumes,
    labor_hours_per_unit=3.5,
    working_days=300,
    hours_per_day=8.0,
    indirect_ratio=0.35
)
print(labor_plan.to_string(index=False))

print("\nAssumptions:")
print("  - Labor Hours per Unit: 3.5 hours")
print("  - Working Days per Year: 300 days")
print("  - Hours per Working Day: 8.0 hours")
print("  - Indirect/Direct Ratio: 35%")

# =====================================================
# PART 4: INTEGRATION WITH FINANCIAL MODEL
# =====================================================

print("\n\n" + "-"*100)
print("PART 4: FINANCIAL MODEL INTEGRATION")
print("-"*100)

print("\nRunning financial model WITH labor management system...")
model_data = run_financial_model(cfg_with_labor)

print("\nINTEGRATED INCOME STATEMENT:")
income_df, cashflow_df, balance_df = generate_financial_statements(model_data)
print(income_df.to_string(index=False))

print("\nLABOR COST STATEMENT:")
labor_statement = generate_labor_statement(model_data)
print(labor_statement.to_string(index=False))

print("\nKEY METRICS:")
print(f"  Total Revenue ({final_year}):       ${model_data['revenue'][final_year]:>15,.0f}")
print(f"  Total OPEX ({final_year}):          ${model_data['opex'][final_year]:>15,.0f}")
labor_metrics = model_data.get('labor_metrics', {})
print(
    f"  Labor Costs ({final_year}):         ${labor_metrics.get(final_year, {}).get('total_labor_cost', 0):>15,.0f}"
)
print(f"  Net Profit ({final_year}):          ${model_data['net_profit'][final_year]:>15,.0f}")
print(f"  Enterprise Value (DCF):     ${model_data['enterprise_value']:>15,.0f}")

# =====================================================
# PART 5: VARIANCE ANALYSIS & INSIGHTS
# =====================================================

print("\n\n" + "-"*100)
print("PART 5: WORKFORCE PLANNING INSIGHTS")
print("-"*100)

print("\nWorkforce Expansion Plan:")
for year in projection_years:
    total_hc = labor_manager.get_total_headcount(year)
    total_cost = sum(labor_manager.get_labor_cost_by_type(year, 0.05).values())
    avg_cost = total_cost / total_hc if total_hc > 0 else 0
    print(f"  {year}: {total_hc:2d} employees | Total Cost: ${total_cost:>12,.0f} | Avg Cost/HC: ${avg_cost:>10,.0f}")

print("\nDirect vs. Indirect Labor Composition:")
for year in projection_years:
    hc = labor_manager.get_headcount_by_type(year)
    total = hc['Direct'] + hc['Indirect']
    direct_pct = (hc['Direct'] / total * 100) if total > 0 else 0
    indirect_pct = (hc['Indirect'] / total * 100) if total > 0 else 0
    print(f"  {year}: Direct {direct_pct:5.1f}% | Indirect {indirect_pct:5.1f}%")

# =====================================================
# PART 6: EDIT SCENARIOS
# =====================================================

print("\n\n" + "-"*100)
print("PART 6: WHAT-IF SCENARIO - SALARY ADJUSTMENT")
print("-"*100)

print("\nScenario: Increase all direct labor salaries by 5% to improve retention")

# Before scenario
costs_before = labor_manager.get_labor_cost_by_type(base_year, 0.05)
total_before = costs_before['Direct'] + costs_before['Indirect']

# Apply changes
direct_positions = labor_manager.get_all_positions(labor_type=LaborType.DIRECT)
for pos in direct_positions:
    new_salary = pos.annual_salary * 1.05
    labor_manager.edit_position(pos.position_id, annual_salary=new_salary)

# After scenario
costs_after = labor_manager.get_labor_cost_by_type(base_year, 0.05)
total_after = costs_after['Direct'] + costs_after['Indirect']

print(f"\nBefore Adjustment:")
print(f"  Direct Labor Cost: ${costs_before['Direct']:>12,.0f}")
print(f"  Indirect Labor Cost: ${costs_before['Indirect']:>12,.0f}")
print(f"  Total Labor Cost: ${total_before:>15,.0f}")

print(f"\nAfter 5% Direct Labor Salary Increase:")
print(f"  Direct Labor Cost: ${costs_after['Direct']:>12,.0f}")
print(f"  Indirect Labor Cost: ${costs_after['Indirect']:>12,.0f}")
print(f"  Total Labor Cost: ${total_after:>15,.0f}")

increase = total_after - total_before
increase_pct = (increase / total_before * 100) if total_before > 0 else 0
print(f"\nTotal Impact: +${increase:,.0f} ({increase_pct:.2f}%)")

# =====================================================
# SUMMARY
# =====================================================

print("\n\n" + "="*100)
print("INTEGRATION COMPLETE - LABOR MANAGEMENT SYSTEM FULLY OPERATIONAL")
print("="*100)

print("\nSYSTEM CAPABILITIES DEMONSTRATED:")
print("  CREATE operations - Added two new positions")
print("  READ operations - Queried headcount, costs, and projections")
print("  UPDATE operations - Edited existing positions with field validation")
print("  DELETE operations - Marked positions inactive with end dates")
print("  Cost Schedules - Five-year forecasts with detailed breakdowns")
print("  Production Linkage - Aligned labor to production volumes")
print("  Financial Integration - Labor costs flow into operating expenses and the P&L")
print("  What-If Analysis - Scenario testing with immediate impact")

print("\nNEXT STEPS:")
print("  1. Export labor schedules to CSV/Excel")
print("  2. Create dashboard visualizations")
print("  3. Set up real-time monitoring alerts")
print("  4. Integrate with payroll system")
print("  5. Build interactive labor planning UI")

print("\n" + "="*100 + "\n")
