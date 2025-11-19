"""
CAPEX Management Module
Features:
- CapexItem dataclass
- CapexScheduleManager with add/edit/remove/list
- Depreciation schedule (straight-line per item)
- Yearly CAPEX schedule and CFI generation for integration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class CapexItem:
    item_id: str
    name: str
    amount: float
    start_year: int
    useful_life: int = 10
    salvage_value: float = 0.0
    category: str = "General"
    depreciation_rate: float = 0.0
    asset_additions: float = 0.0
    start_date: str = ""
    notes: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.start_date:
            self.start_date = f"{self.start_year}-01-01"
        # Guard against negative rates
        if self.depreciation_rate < 0:
            self.depreciation_rate = 0.0
        if self.asset_additions < 0:
            self.asset_additions = 0.0

    def capitalized_cost(self) -> float:
        """Total capitalized amount including asset additions."""
        return self.amount + self.asset_additions

    def depreciation_schedule(self, years: List[int]) -> Dict[int, float]:
        """Return straight-line depreciation per year for the provided years list."""
        schedule = {}
        cost_basis = max(self.capitalized_cost() - self.salvage_value, 0.0)
        if self.useful_life <= 0 or cost_basis <= 0:
            for y in years:
                schedule[y] = 0.0
            return schedule

        if self.depreciation_rate > 0:
            annual_dep = cost_basis * self.depreciation_rate
        else:
            annual_dep = cost_basis / self.useful_life

        remaining = cost_basis
        life_end = self.start_year + self.useful_life - 1
        for y in years:
            if self.start_year <= y <= life_end and remaining > 0:
                charge = min(annual_dep, remaining)
                # Force final year to clear any rounding remainder
                if y == life_end or charge > remaining:
                    charge = remaining
                schedule[y] = charge
                remaining -= charge
            else:
                schedule[y] = 0.0
        return schedule

    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'name': self.name,
            'amount': self.amount,
            'asset_additions': self.asset_additions,
            'start_year': self.start_year,
            'start_date': self.start_date,
            'useful_life': self.useful_life,
            'salvage_value': self.salvage_value,
            'depreciation_rate': self.depreciation_rate,
            'category': self.category,
            'notes': self.notes,
            'created_date': self.created_date,
            'last_modified': self.last_modified
        }

class CapexScheduleManager:
    def __init__(self):
        self.items: Dict[str, CapexItem] = {}
        self._counter = 0

    def add_item(
        self,
        name: str,
        amount: float,
        start_year: int,
        useful_life: int = 10,
        salvage_value: float = 0.0,
        category: str = "General",
        notes: str = "",
        depreciation_rate: float = 0.0,
        asset_additions: float = 0.0,
        start_date: Optional[str] = None,
    ) -> str:
        self._counter += 1
        item_id = f"CAP_{self._counter:03d}"
        item = CapexItem(
            item_id=item_id,
            name=name,
            amount=amount,
            start_year=start_year,
            useful_life=useful_life,
            salvage_value=salvage_value,
            category=category,
            notes=notes,
            depreciation_rate=depreciation_rate,
            asset_additions=asset_additions,
            start_date=start_date or f"{start_year}-01-01",
        )
        self.items[item_id] = item
        print(f"✓ CAPEX item added: {item_id} - {name} (${item.capitalized_cost():,.0f})")
        return item_id

    def get_item(self, item_id: str) -> Optional[CapexItem]:
        return self.items.get(item_id)

    def list_items(self) -> List[CapexItem]:
        return list(self.items.values())

    def edit_item(self, item_id: str, **kwargs) -> bool:
        if item_id not in self.items:
            raise ValueError(f"CAPEX item {item_id} not found")
        item = self.items[item_id]
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
            else:
                raise ValueError(f"Invalid field for CapexItem: {key}")
        if 'start_date' in kwargs and 'start_year' not in kwargs:
            try:
                item.start_year = int(kwargs['start_date'][:4])
            except (ValueError, TypeError, KeyError):
                pass
        item.last_modified = datetime.now().isoformat()
        print(f"✓ CAPEX item updated: {item_id}")
        return True

    def remove_item(self, item_id: str) -> bool:
        if item_id not in self.items:
            raise ValueError(f"CAPEX item {item_id} not found")
        name = self.items[item_id].name
        del self.items[item_id]
        print(f"✓ CAPEX item removed: {item_id} - {name}")
        return True

    def yearly_capex_schedule(self, start_year: int, years: int) -> Dict[int, float]:
        """Return dictionary mapping year -> total CAPEX spend in that year"""
        ylist = [start_year + i for i in range(years)]
        schedule = {y: 0.0 for y in ylist}
        for item in self.items.values():
            if item.start_year in schedule:
                schedule[item.start_year] += item.capitalized_cost()
        return schedule

    def depreciation_schedule(self, start_year: int, years: int) -> Dict[int, float]:
        """Aggregate depreciation schedule across items over the years range"""
        ylist = [start_year + i for i in range(years)]
        total_dep = {y: 0.0 for y in ylist}
        for item in self.items.values():
            item_dep = item.depreciation_schedule(ylist)
            for y in ylist:
                total_dep[y] += item_dep.get(y, 0.0)
        return total_dep

    def total_capex(self) -> float:
        return sum(item.capitalized_cost() for item in self.items.values())

# Helper initializer
def initialize_default_capex(manager: CapexScheduleManager) -> CapexScheduleManager:
    # Add some default capex items
    manager.add_item(
        name="Land Acquisition",
        amount=1_000_000,
        start_year=2026,
        useful_life=30,
        salvage_value=0,
        category="Land",
        depreciation_rate=0.0,
        asset_additions=0.0,
        start_date="2026-01-01",
    )
    manager.add_item(
        name="Factory Construction",
        amount=2_500_000,
        start_year=2026,
        useful_life=30,
        salvage_value=0,
        category="Buildings",
        depreciation_rate=0.0333,
        asset_additions=0.0,
        start_date="2026-02-01",
    )
    manager.add_item(
        name="Machinery & Automation",
        amount=500_000,
        start_year=2026,
        useful_life=10,
        salvage_value=50_000,
        category="Machinery",
        depreciation_rate=0.1,
        asset_additions=0.0,
        start_date="2026-03-01",
    )
    manager.add_item(
        name="Tooling & Fixtures",
        amount=150_000,
        start_year=2027,
        useful_life=7,
        salvage_value=5_000,
        category="Equipment",
        depreciation_rate=0.142857,
        asset_additions=0.0,
        start_date="2027-01-01",
    )
    return manager
