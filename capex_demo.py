"""
CAPEX Integration Demo

Demonstrates: list defaults, add, edit, remove CAPEX items,
attach to `CompanyConfig` and run the financial model to
observe changes to yearly CAPEX (CFI), depreciation, and EV.
"""
from capex_management import CapexScheduleManager, initialize_default_capex
from financial_model import CompanyConfig, run_financial_model

# Create company config (use default start year)
cfg = CompanyConfig()

# Initialize capex manager and defaults
capex_mgr = CapexScheduleManager()
initialize_default_capex(capex_mgr)

print("\n--- INITIAL CAPEX ITEMS ---")
for it in capex_mgr.list_items():
    print(it.to_dict())

# Attach to config and run model (baseline)
cfg.capex_manager = capex_mgr
base_model = run_financial_model(cfg)

print("\n--- BASELINE: YEARLY CAPEX SCHEDULE (first 5 years) ---")
yearly = capex_mgr.yearly_capex_schedule(cfg.start_year, 5)
for y, amt in yearly.items():
    print(f"{y}: ${amt:,.0f}")

print("\n--- BASELINE: AGGREGATE DEPRECIATION SCHEDULE (first 5 years) ---")
dep_sched = capex_mgr.depreciation_schedule(cfg.start_year, 5)
for y, d in dep_sched.items():
    print(f"{y}: ${d:,.0f}")

print("\nBaseline Enterprise Value: ${:,.0f}".format(base_model['enterprise_value']))

# -------------------------
# ADD a new CAPEX item
# -------------------------
print("\n--- ADDING NEW CAPEX ITEM: Solar Array ---")
new_item_id = capex_mgr.add_item(
    name="Solar Array",
    cost=250_000,
    life_years=10,
    start_year=cfg.start_year
)
print("Added item id:", new_item_id)

print("\n--- CAPEX ITEMS AFTER ADD ---")
for it in capex_mgr.list_items():
    print(it.to_dict())

# Re-run model after add
model_after_add = run_financial_model(cfg)
print("\nEnterprise Value After Add: ${:,.0f}".format(model_after_add['enterprise_value']))

# -------------------------
# EDIT an existing CAPEX item
# -------------------------
print("\n--- EDITING FIRST CAPEX ITEM (reduce cost by 10%) ---")
items = capex_mgr.list_items()
if items:
    first = items[0]
    old = first.to_dict()
    new_cost = int(old['cost'] * 0.9)
    capex_mgr.edit_item(first.item_id, cost=new_cost)
    print(f"Edited {first.item_id}: cost {old['cost']} -> {new_cost}")

print("\n--- CAPEX ITEMS AFTER EDIT ---")
for it in capex_mgr.list_items():
    print(it.to_dict())

# Re-run model after edit
model_after_edit = run_financial_model(cfg)
print("\nEnterprise Value After Edit: ${:,.0f}".format(model_after_edit['enterprise_value']))

# -------------------------
# REMOVE a CAPEX item
# -------------------------
print("\n--- REMOVING LAST CAPEX ITEM ---")
items = capex_mgr.list_items()
if items:
    last = items[-1]
    capex_mgr.remove_item(last.item_id)
    print(f"Removed {last.item_id}")

print("\n--- CAPEX ITEMS AFTER REMOVE ---")
for it in capex_mgr.list_items():
    print(it.to_dict())

# Re-run model after remove
model_after_remove = run_financial_model(cfg)
print("\nEnterprise Value After Remove: ${:,.0f}".format(model_after_remove['enterprise_value']))

# Final summary: show CFI and depreciation for final model
print("\n--- FINAL: YEARLY CAPEX SCHEDULE (first 5 years) ---")
yearly_final = capex_mgr.yearly_capex_schedule(cfg.start_year, 5)
for y, amt in yearly_final.items():
    print(f"{y}: ${amt:,.0f}")

print("\n--- FINAL: AGGREGATE DEPRECIATION SCHEDULE (first 5 years) ---")
dep_sched_final = capex_mgr.depreciation_schedule(cfg.start_year, 5)
for y, d in dep_sched_final.items():
    print(f"{y}: ${d:,.0f}")

print("\nFinal Enterprise Value: ${:,.0f}".format(model_after_remove['enterprise_value']))
