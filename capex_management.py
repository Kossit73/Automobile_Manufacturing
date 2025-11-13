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
    notes: str = ""
    spend_curve: Optional[Dict[int, float]] = None
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    def depreciation_schedule(self, years: List[int]) -> Dict[int, float]:
        """Return straight-line depreciation per year for the provided years list."""
        schedule = {}
        if self.useful_life <= 0:
            # Treat as immediate expensing
            for y in years:
                schedule[y] = 0.0
            return schedule

        # Annual depreciation = (amount - salvage) / useful_life
        annual_dep = (self.amount - self.salvage_value) / self.useful_life
        for y in years:
            # If year is within asset life
            if self.start_year <= y < self.start_year + self.useful_life:
                schedule[y] = annual_dep
            else:
                schedule[y] = 0.0
        return schedule

    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'name': self.name,
            'amount': self.amount,
            'start_year': self.start_year,
            'useful_life': self.useful_life,
            'salvage_value': self.salvage_value,
            'category': self.category,
            'notes': self.notes,
            'spend_curve': self.spend_curve,
            'created_date': self.created_date,
            'last_modified': self.last_modified
        }

class CapexScheduleManager:
    def __init__(self):
        self.items: Dict[str, CapexItem] = {}
        self._counter = 0

    @staticmethod
    def _validate_spend_curve(spend_curve: Dict[int, float]) -> None:
        if any(not isinstance(offset, int) or offset < 0 for offset in spend_curve):
            raise ValueError("Spend curve offsets must be non-negative integers")
        if any(value < 0 for value in spend_curve.values()):
            raise ValueError("Spend curve allocations must be non-negative")
        total_allocation = sum(spend_curve.values())
        if abs(total_allocation - 1.0) > 1e-6:
            raise ValueError("Spend curve allocations must sum to 1.0")

    def add_item(self, name: str, amount: float, start_year: int, useful_life: int = 10,
                 salvage_value: float = 0.0, category: str = "General", notes: str = "",
                 spend_curve: Optional[Dict[int, float]] = None) -> str:
        if spend_curve is not None:
            self._validate_spend_curve(spend_curve)
        self._counter += 1
        item_id = f"CAP_{self._counter:03d}"
        item = CapexItem(item_id=item_id, name=name, amount=amount, start_year=start_year,
                         useful_life=useful_life, salvage_value=salvage_value,
                         category=category, notes=notes, spend_curve=spend_curve)
        self.items[item_id] = item
        print(f"✓ CAPEX item added: {item_id} - {name} (${amount:,.0f})")
        return item_id

    def get_item(self, item_id: str) -> Optional[CapexItem]:
        return self.items.get(item_id)

    def list_items(self) -> List[CapexItem]:
        return list(self.items.values())

    def edit_item(self, item_id: str, **kwargs) -> bool:
        if item_id not in self.items:
            raise ValueError(f"CAPEX item {item_id} not found")
        item = self.items[item_id]
        if 'spend_curve' in kwargs and kwargs['spend_curve'] is not None:
            self._validate_spend_curve(kwargs['spend_curve'])
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
            else:
                raise ValueError(f"Invalid field for CapexItem: {key}")
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
        """Return dictionary mapping year -> total CAPEX spend in that year, honoring spend curves."""
        ylist = [start_year + i for i in range(years)]
        schedule = {y: 0.0 for y in ylist}
        for item in self.items.values():
            if item.spend_curve:
                for offset, share in item.spend_curve.items():
                    target_year = item.start_year + offset
                    allocation = item.amount * share
                    schedule[target_year] = schedule.get(target_year, 0.0) + allocation
            else:
                schedule[item.start_year] = schedule.get(item.start_year, 0.0) + item.amount
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
        return sum(item.amount for item in self.items.values())

# Helper initializer
def initialize_default_capex(manager: CapexScheduleManager) -> CapexScheduleManager:
    # Add some default capex items
    manager.add_item(name="Land Acquisition", amount=1_000_000, start_year=2026, useful_life=30, salvage_value=0, category="Land")
    manager.add_item(name="Factory Construction", amount=2_500_000, start_year=2026, useful_life=30, salvage_value=0, category="Buildings")
    manager.add_item(name="Machinery & Automation", amount=500_000, start_year=2026, useful_life=10, salvage_value=50_000, category="Machinery")
    manager.add_item(name="Tooling & Fixtures", amount=150_000, start_year=2027, useful_life=7, salvage_value=5_000, category="Equipment")
    return manager
