"""
Labor Management Module - Direct & Indirect Labor Scheduling
Integrated with Financial Model
Features: Add, Remove, Edit, Sync with Production Forecasts
Author: Advanced Analytics Team
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

# =====================================================
# 1. ENUMS & DATA STRUCTURES
# =====================================================

class LaborType(Enum):
    """Labor classification"""
    DIRECT = "Direct Labor"
    INDIRECT = "Indirect Labor"

class EmploymentStatus(Enum):
    """Employment status"""
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    SEASONAL = "Seasonal"

class JobCategory(Enum):
    """Job classifications"""
    # Direct Labor
    ASSEMBLY = "Assembly"
    WELDING = "Welding"
    PAINTING = "Painting"
    QUALITY_CONTROL = "Quality Control"
    MATERIAL_HANDLING = "Material Handling"
    
    # Indirect Labor
    PRODUCTION_MANAGEMENT = "Production Management"
    QUALITY_ASSURANCE = "Quality Assurance"
    MAINTENANCE = "Maintenance"
    PLANNING_SCHEDULING = "Planning & Scheduling"
    SUPERVISION = "Supervision"
    ADMINISTRATION = "Administration"
    HUMAN_RESOURCES = "Human Resources"
    FINANCE = "Finance"
    SALES_MARKETING = "Sales & Marketing"

@dataclass
class LaborPosition:
    """Individual labor position record"""
    position_id: str
    position_name: str
    labor_type: LaborType
    job_category: JobCategory
    headcount: int
    annual_salary: float
    status: EmploymentStatus = EmploymentStatus.ACTIVE
    start_year: int = 2026
    end_year: Optional[int] = None
    benefits_percent: float = 0.25  # 25% of salary for benefits
    overtime_hours_annual: float = 0.0  # Annual overtime hours
    overtime_rate: float = 1.5  # 1.5x base rate
    training_cost_annual: float = 0.0
    equipment_cost_annual: float = 0.0
    notes: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate position data"""
        if self.headcount < 0:
            raise ValueError(f"Headcount must be >= 0, got {self.headcount}")
        if self.annual_salary < 0:
            raise ValueError(f"Annual salary must be >= 0, got {self.annual_salary}")
        if not (0 <= self.benefits_percent <= 1):
            raise ValueError(f"Benefits percent must be 0-1, got {self.benefits_percent}")
    
    def calculate_annual_cost(self, salary_growth: float = 0.0) -> float:
        """Calculate total annual cost for this position"""
        base_salary = self.annual_salary * (1 + salary_growth)
        benefits = base_salary * self.benefits_percent
        overtime_cost = self.overtime_hours_annual * (base_salary / 2000) * self.overtime_rate
        return (base_salary + benefits + overtime_cost + self.training_cost_annual + 
                self.equipment_cost_annual) * self.headcount
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'position_id': self.position_id,
            'position_name': self.position_name,
            'labor_type': self.labor_type.value,
            'job_category': self.job_category.value,
            'headcount': self.headcount,
            'annual_salary': self.annual_salary,
            'status': self.status.value,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'benefits_percent': self.benefits_percent,
            'overtime_hours_annual': self.overtime_hours_annual,
            'overtime_rate': self.overtime_rate,
            'training_cost_annual': self.training_cost_annual,
            'equipment_cost_annual': self.equipment_cost_annual,
            'notes': self.notes,
            'created_date': self.created_date,
            'last_modified': self.last_modified
        }

# =====================================================
# 2. LABOR SCHEDULE MANAGER
# =====================================================

class LaborScheduleManager:
    """Manages all labor positions with CRUD operations"""
    
    def __init__(self):
        """Initialize labor schedule"""
        self.positions: Dict[str, LaborPosition] = {}
        self._position_counter = 0
    
    # ===== CREATE =====
    def add_position(self, position_name: str, labor_type: LaborType, job_category: JobCategory,
                     headcount: int, annual_salary: float, status: EmploymentStatus = EmploymentStatus.ACTIVE,
                     start_year: int = 2026, end_year: Optional[int] = None,
                     benefits_percent: float = 0.25, overtime_hours_annual: float = 0.0,
                     overtime_rate: float = 1.5, training_cost: float = 0.0,
                     equipment_cost: float = 0.0, notes: str = "") -> str:
        """
        Add new labor position to schedule
        
        Returns: position_id
        """
        self._position_counter += 1
        position_id = f"POS_{labor_type.name[0]}_{self._position_counter:03d}"
        
        position = LaborPosition(
            position_id=position_id,
            position_name=position_name,
            labor_type=labor_type,
            job_category=job_category,
            headcount=headcount,
            annual_salary=annual_salary,
            status=status,
            start_year=start_year,
            end_year=end_year,
            benefits_percent=benefits_percent,
            overtime_hours_annual=overtime_hours_annual,
            overtime_rate=overtime_rate,
            training_cost_annual=training_cost,
            equipment_cost_annual=equipment_cost,
            notes=notes
        )
        
        self.positions[position_id] = position
        print(f"Position added: {position_id} - {position_name}")
        return position_id
    
    # ===== READ =====
    def get_position(self, position_id: str) -> Optional[LaborPosition]:
        """Retrieve position by ID"""
        return self.positions.get(position_id)
    
    def get_all_positions(self, labor_type: Optional[LaborType] = None,
                         status: Optional[EmploymentStatus] = None) -> List[LaborPosition]:
        """Get all positions, optionally filtered"""
        positions = list(self.positions.values())
        
        if labor_type:
            positions = [p for p in positions if p.labor_type == labor_type]
        if status:
            positions = [p for p in positions if p.status == status]
        
        return positions
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all positions"""
        data = []
        for pos in self.positions.values():
            data.append({
                'Position ID': pos.position_id,
                'Position Name': pos.position_name,
                'Type': pos.labor_type.value,
                'Category': pos.job_category.value,
                'Headcount': pos.headcount,
                'Annual Salary': f"${pos.annual_salary:,.0f}",
                'Status': pos.status.value,
                'Start Year': pos.start_year,
                'End Year': pos.end_year if pos.end_year else 'Ongoing'
            })
        
        return pd.DataFrame(data)
    
    # ===== UPDATE =====
    def edit_position(self, position_id: str, **kwargs) -> bool:
        """
        Edit existing position
        
        Supported fields:
        - position_name, headcount, annual_salary, status, end_year
        - benefits_percent, overtime_hours_annual, overtime_rate
        - training_cost_annual, equipment_cost_annual, notes
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        # Validate before updating
        if 'headcount' in kwargs and kwargs['headcount'] < 0:
            raise ValueError("Headcount must be >= 0")
        if 'annual_salary' in kwargs and kwargs['annual_salary'] < 0:
            raise ValueError("Annual salary must be >= 0")
        if 'benefits_percent' in kwargs:
            pct = kwargs['benefits_percent']
            if not (0 <= pct <= 1):
                raise ValueError(f"Benefits percent must be 0-1, got {pct}")
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)
            else:
                raise ValueError(f"Position does not have field: {key}")
        
        position.last_modified = datetime.now().isoformat()
        print(f"Position updated: {position_id}")
        return True
    
    # ===== DELETE =====
    def remove_position(self, position_id: str) -> bool:
        """Remove position from schedule"""
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position_name = self.positions[position_id].position_name
        del self.positions[position_id]
        print(f"Position removed: {position_id} - {position_name}")
        return True
    
    def mark_inactive(self, position_id: str, end_year: int) -> bool:
        """Mark position as inactive (end of employment)"""
        return self.edit_position(position_id, status=EmploymentStatus.INACTIVE, end_year=end_year)
    
    # ===== QUERY & ANALYSIS =====
    def get_headcount_by_type(self, year: int = 2026) -> Dict[str, int]:
        """Get total headcount by labor type"""
        headcount = {'Direct': 0, 'Indirect': 0}
        
        for pos in self.positions.values():
            if pos.start_year <= year and (pos.end_year is None or pos.end_year >= year):
                if pos.status == EmploymentStatus.ACTIVE or pos.status == EmploymentStatus.SEASONAL:
                    if pos.labor_type == LaborType.DIRECT:
                        headcount['Direct'] += pos.headcount
                    else:
                        headcount['Indirect'] += pos.headcount
        
        return headcount
    
    def get_total_headcount(self, year: int = 2026) -> int:
        """Get total headcount for year"""
        counts = self.get_headcount_by_type(year)
        return counts['Direct'] + counts['Indirect']
    
    def get_labor_cost_by_type(self, year: int, salary_growth: float = 0.05) -> Dict[str, float]:
        """Calculate total labor cost by type"""
        costs = {'Direct': 0.0, 'Indirect': 0.0}
        
        for pos in self.positions.values():
            if pos.start_year <= year and (pos.end_year is None or pos.end_year >= year):
                if pos.status == EmploymentStatus.ACTIVE or pos.status == EmploymentStatus.SEASONAL:
                    years_since_start = year - pos.start_year
                    growth_factor = (1 + salary_growth) ** years_since_start
                    annual_cost = pos.calculate_annual_cost(growth_factor - 1)
                    
                    if pos.labor_type == LaborType.DIRECT:
                        costs['Direct'] += annual_cost
                    else:
                        costs['Indirect'] += annual_cost
        
        return costs
    
    def get_labor_cost_by_category(self, year: int, salary_growth: float = 0.05) -> Dict[str, float]:
        """Calculate total labor cost by job category"""
        costs = {}
        
        for pos in self.positions.values():
            if pos.start_year <= year and (pos.end_year is None or pos.end_year >= year):
                if pos.status == EmploymentStatus.ACTIVE or pos.status == EmploymentStatus.SEASONAL:
                    category = pos.job_category.value
                    years_since_start = year - pos.start_year
                    growth_factor = (1 + salary_growth) ** years_since_start
                    annual_cost = pos.calculate_annual_cost(growth_factor - 1)
                    costs[category] = costs.get(category, 0) + annual_cost
        
        return costs

# =====================================================
# 3. LABOR COST SCHEDULE GENERATOR
# =====================================================

class LaborCostSchedule:
    """Generates detailed labor cost schedules over multi-year period"""
    
    def __init__(self, labor_manager: LaborScheduleManager):
        """Initialize with labor manager"""
        self.labor_manager = labor_manager
    
    def generate_5year_schedule(self, start_year: int = 2026, salary_growth: float = 0.05) -> pd.DataFrame:
        """Generate 5-year labor cost schedule"""
        years = range(start_year, start_year + 5)
        schedule_data = []
        
        for year in years:
            direct_cost = self.labor_manager.get_labor_cost_by_type(year, salary_growth)['Direct']
            indirect_cost = self.labor_manager.get_labor_cost_by_type(year, salary_growth)['Indirect']
            total_cost = direct_cost + indirect_cost
            
            direct_hc = self.labor_manager.get_headcount_by_type(year)['Direct']
            indirect_hc = self.labor_manager.get_headcount_by_type(year)['Indirect']
            total_hc = direct_hc + indirect_hc
            
            schedule_data.append({
                'Year': year,
                'Direct Labor HC': direct_hc,
                'Direct Labor Cost': direct_cost,
                'Indirect Labor HC': indirect_hc,
                'Indirect Labor Cost': indirect_cost,
                'Total Headcount': total_hc,
                'Total Labor Cost': total_cost,
                'Avg Cost per HC': total_cost / total_hc if total_hc > 0 else 0
            })
        
        return pd.DataFrame(schedule_data)
    
    def generate_detailed_schedule(self, year: int) -> pd.DataFrame:
        """Generate detailed position-level schedule for specific year"""
        data = []
        
        for pos in self.labor_manager.get_all_positions():
            if pos.start_year <= year and (pos.end_year is None or pos.end_year >= year):
                if pos.status == EmploymentStatus.ACTIVE or pos.status == EmploymentStatus.SEASONAL:
                    years_since_start = year - pos.start_year
                    salary_growth = 0.05
                    
                    base_salary = pos.annual_salary * (1 + salary_growth) ** years_since_start
                    total_salary = base_salary * pos.headcount
                    benefits = total_salary * pos.benefits_percent
                    overtime_cost = pos.overtime_hours_annual * (base_salary / 2000) * pos.overtime_rate * pos.headcount
                    training = pos.training_cost_annual * pos.headcount
                    equipment = pos.equipment_cost_annual * pos.headcount
                    total_cost = total_salary + benefits + overtime_cost + training + equipment
                    
                    data.append({
                        'Position ID': pos.position_id,
                        'Position Name': pos.position_name,
                        'Type': pos.labor_type.value,
                        'Category': pos.job_category.value,
                        'Headcount': pos.headcount,
                        'Base Salary': base_salary,
                        'Total Salary': total_salary,
                        'Benefits': benefits,
                        'Overtime': overtime_cost,
                        'Training': training,
                        'Equipment': equipment,
                        'Total Cost': total_cost
                    })
        
        return pd.DataFrame(data)
    
    def generate_category_summary(self, year: int) -> pd.DataFrame:
        """Generate summary by job category"""
        category_costs = self.labor_manager.get_labor_cost_by_category(year)
        
        data = []
        for category, cost in sorted(category_costs.items(), key=lambda x: x[1], reverse=True):
            data.append({
                'Job Category': category,
                'Annual Cost': cost,
                'Percentage of Total': f"{cost / sum(category_costs.values()) * 100:.1f}%"
            })
        
        return pd.DataFrame(data)

# =====================================================
# 4. PRODUCTION-LINKED LABOR FORECASTING
# =====================================================

class ProductionLinkedLabor:
    """Links labor requirements to production volumes"""
    
    @staticmethod
    def calculate_direct_labor_requirement(production_volume: float, labor_hours_per_unit: float,
                                          working_days: int = 300, hours_per_day: float = 8.0) -> float:
        """
        Calculate required direct labor headcount based on production
        
        Args:
            production_volume: Annual units to produce
            labor_hours_per_unit: Labor hours required per unit
            working_days: Days available per year
            hours_per_day: Hours worked per day per employee
        
        Returns:
            Required headcount (float, can be fractional)
        """
        total_hours_needed = production_volume * labor_hours_per_unit
        hours_available_per_worker = working_days * hours_per_day
        required_headcount = total_hours_needed / hours_available_per_worker
        return required_headcount
    
    @staticmethod
    def calculate_indirect_labor_ratio(direct_headcount: float, ratio: float = 0.35) -> float:
        """
        Calculate indirect labor as ratio of direct labor
        Typical manufacturing: 30-40% ratio
        
        Args:
            direct_headcount: Number of direct laborers
            ratio: Indirect/Direct labor ratio (default 0.35 = 35%)
        
        Returns:
            Required indirect headcount
        """
        return direct_headcount * ratio
    
    @staticmethod
    def create_labor_plan(production_volumes: Dict[int, float], labor_hours_per_unit: float,
                          working_days: int = 300, hours_per_day: float = 8.0,
                          indirect_ratio: float = 0.35) -> pd.DataFrame:
        """
        Create labor plan based on production forecast
        
        Returns DataFrame with required headcount by year
        """
        plan_data = []
        
        for year in sorted(production_volumes.keys()):
            production = production_volumes[year]
            direct_hc = ProductionLinkedLabor.calculate_direct_labor_requirement(
                production, labor_hours_per_unit, working_days, hours_per_day
            )
            indirect_hc = ProductionLinkedLabor.calculate_indirect_labor_ratio(direct_hc, indirect_ratio)
            
            plan_data.append({
                'Year': year,
                'Production Volume': int(production),
                'Direct Labor HC (Required)': round(direct_hc, 1),
                'Indirect Labor HC (Required)': round(indirect_hc, 1),
                'Total HC (Required)': round(direct_hc + indirect_hc, 1)
            })
        
        return pd.DataFrame(plan_data)

# =====================================================
# 5. VARIANCE ANALYSIS
# =====================================================

class LaborVarianceAnalysis:
    """Analyzes labor cost variances"""
    
    @staticmethod
    def calculate_variances(budgeted: Dict[int, float], actual: Dict[int, float]) -> pd.DataFrame:
        """
        Calculate budget vs. actual variances
        
        Returns DataFrame with variance analysis
        """
        data = []
        
        for year in sorted(budgeted.keys()):
            budget = budgeted[year]
            actual_val = actual.get(year, 0)
            variance = actual_val - budget
            variance_pct = (variance / budget * 100) if budget != 0 else 0
            
            data.append({
                'Year': year,
                'Budgeted': budget,
                'Actual': actual_val,
                'Variance': variance,
                'Variance %': f"{variance_pct:.1f}%",
                'Status': 'Unfavorable' if variance > 0 else 'Favorable'
            })
        
        return pd.DataFrame(data)

# =====================================================
# 6. DEMONSTRATION & TESTING
# =====================================================

def initialize_default_labor_structure() -> LaborScheduleManager:
    """Create default labor structure for Volt Rider"""
    manager = LaborScheduleManager()
    
    # ===== DIRECT LABOR =====
    # Assembly
    manager.add_position(
        position_name="Assembly Line Workers",
        labor_type=LaborType.DIRECT,
        job_category=JobCategory.ASSEMBLY,
        headcount=12,
        annual_salary=36000,
        benefits_percent=0.25,
        overtime_hours_annual=200,
        training_cost=2000
    )
    
    # Welding
    manager.add_position(
        position_name="Welding Specialists",
        labor_type=LaborType.DIRECT,
        job_category=JobCategory.WELDING,
        headcount=8,
        annual_salary=42000,
        benefits_percent=0.25,
        overtime_hours_annual=150,
        training_cost=3000,
        equipment_cost=1000
    )
    
    # Painting
    manager.add_position(
        position_name="Painters",
        labor_type=LaborType.DIRECT,
        job_category=JobCategory.PAINTING,
        headcount=6,
        annual_salary=38000,
        benefits_percent=0.25,
        overtime_hours_annual=100,
        training_cost=1500
    )
    
    # Quality Control
    manager.add_position(
        position_name="QC Inspectors",
        labor_type=LaborType.DIRECT,
        job_category=JobCategory.QUALITY_CONTROL,
        headcount=4,
        annual_salary=40000,
        benefits_percent=0.25,
        training_cost=2000
    )
    
    # Material Handling
    manager.add_position(
        position_name="Material Handlers",
        labor_type=LaborType.DIRECT,
        job_category=JobCategory.MATERIAL_HANDLING,
        headcount=5,
        annual_salary=33000,
        benefits_percent=0.25,
        overtime_hours_annual=250
    )
    
    # ===== INDIRECT LABOR =====
    # Production Management
    manager.add_position(
        position_name="Production Manager",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.PRODUCTION_MANAGEMENT,
        headcount=1,
        annual_salary=65000,
        benefits_percent=0.30,
        training_cost=5000
    )
    
    # Quality Assurance
    manager.add_position(
        position_name="QA Lead",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.QUALITY_ASSURANCE,
        headcount=1,
        annual_salary=58000,
        benefits_percent=0.30,
        training_cost=4000
    )
    
    # Maintenance
    manager.add_position(
        position_name="Maintenance Technicians",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.MAINTENANCE,
        headcount=3,
        annual_salary=45000,
        benefits_percent=0.25,
        training_cost=3000,
        equipment_cost=2000
    )
    
    # Supervision
    manager.add_position(
        position_name="Line Supervisors",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.SUPERVISION,
        headcount=2,
        annual_salary=50000,
        benefits_percent=0.28,
        training_cost=3000
    )
    
    # Administration
    manager.add_position(
        position_name="Administrative Staff",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.ADMINISTRATION,
        headcount=2,
        annual_salary=35000,
        benefits_percent=0.25,
        training_cost=1000
    )
    
    # Finance
    manager.add_position(
        position_name="Finance Manager",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.FINANCE,
        headcount=1,
        annual_salary=70000,
        benefits_percent=0.30,
        training_cost=5000
    )
    
    # Sales & Marketing
    manager.add_position(
        position_name="Sales & Marketing Team",
        labor_type=LaborType.INDIRECT,
        job_category=JobCategory.SALES_MARKETING,
        headcount=2,
        annual_salary=48000,
        benefits_percent=0.28,
        training_cost=2000
    )
    
    return manager

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LABOR MANAGEMENT SYSTEM - DEMONSTRATION")
    print("="*80)
    
    # Initialize labor structure
    labor_manager = initialize_default_labor_structure()
    
    # ===== DISPLAY SUMMARY =====
    print("\n--- LABOR SCHEDULE SUMMARY ---")
    print(labor_manager.get_position_summary().to_string(index=False))
    
    # ===== GENERATE COST SCHEDULES =====
    print("\n--- 5-YEAR LABOR COST SCHEDULE ---")
    cost_schedule = LaborCostSchedule(labor_manager)
    print(cost_schedule.generate_5year_schedule().to_string(index=False))
    
    # ===== PRODUCTION-LINKED LABOR FORECAST =====
    print("\n--- PRODUCTION-LINKED LABOR FORECAST ---")
    production_volumes = {2026: 10000, 2027: 14000, 2028: 18000, 2029: 20000, 2030: 20000}
    labor_plan = ProductionLinkedLabor.create_labor_plan(
        production_volumes=production_volumes,
        labor_hours_per_unit=3.5,
        working_days=300,
        hours_per_day=8.0,
        indirect_ratio=0.35
    )
    print(labor_plan.to_string(index=False))
    
    # ===== DETAILED SCHEDULE FOR 2026 =====
    print("\n--- DETAILED LABOR COST SCHEDULE (2026) ---")
    detailed = cost_schedule.generate_detailed_schedule(2026)
    print(detailed.to_string(index=False))
    
    # ===== CATEGORY SUMMARY =====
    print("\n--- JOB CATEGORY SUMMARY (2026) ---")
    category_summary = cost_schedule.generate_category_summary(2026)
    print(category_summary.to_string(index=False))
    
    # ===== DEMONSTRATE EDIT FUNCTIONALITY =====
    print("\n--- EDIT DEMONSTRATION ---")
    print("Current Assembly Line Headcount: 12")
    pos_id = list(labor_manager.positions.keys())[0]
    labor_manager.edit_position(pos_id, headcount=15, annual_salary=38000)
    print(f"Updated Headcount: 15, Updated Salary: $38,000")
    
    # ===== DEMONSTRATE ADD/REMOVE =====
    print("\n--- ADD/REMOVE DEMONSTRATION ---")
    new_pos_id = labor_manager.add_position(
        position_name="Robotic System Operators",
        labor_type=LaborType.DIRECT,
        job_category=JobCategory.ASSEMBLY,
        headcount=3,
        annual_salary=50000,
        benefits_percent=0.25,
        training_cost=5000,
        notes="New automation role for 2027 expansion"
    )
    
    # Test inactive marking
    print("\nMarking position as inactive at end of 2026...")
    labor_manager.mark_inactive(new_pos_id, 2026)
    
    print("\n" + "="*80)
    print("LABOR MANAGEMENT SYSTEM READY FOR INTEGRATION")
    print("="*80)
