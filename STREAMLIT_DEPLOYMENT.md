# Streamlit App Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## App Features

### 📊 Dashboard
- Real-time key performance indicators (KPIs)
- Revenue and profit forecasts (2026-2030)
- Financial overview with income statement and balance sheet
- Labor and asset summaries

### 👥 Labor Management
**View Positions:**
- Current labor schedule with all positions
- Summary statistics (headcount, costs)
- 5-year labor cost forecasts
- Cost by job category
- Labor cost trends visualization

**Add Position:**
- Create new labor positions with 15+ configurable fields
- Set direct/indirect labor type and job category
- Configure salary, benefits, overtime
- Add training and equipment costs
- Set employment dates and notes

**Edit Position:**
- Modify existing position details
- Update headcount, salary, benefits
- Adjust overtime and training costs
- Mark positions as inactive
- Delete positions entirely

### 🏗️ CAPEX Management
**View Assets:**
- Current capital assets with acquisition costs
- Depreciation methods (Straight-Line, MACRS, Declining Balance, Units of Production)
- 5-year depreciation schedule
- Book value tracking

**Add Asset:**
- Create new capital assets
- Specify acquisition cost and useful life
- Select depreciation method
- Set salvage value
- Track acquisition date

**Edit Asset:**
- Modify asset details and depreciation settings
- Update costs and useful life
- Change depreciation methods
- Remove assets with proper accounting

### 💰 Financial Model
**Run Model:**
- Configure WACC, terminal growth rate, revenue CAGR
- Set COGS percentage, tax rate
- Input debt, interest rate, shares outstanding
- Execute complete financial model

**Results:**
- Income statement (2026-2030)
- DCF valuation summary
- Enterprise and equity value
- Free cash flow forecast
- DCF analysis visualization

**Scenarios:**
- Save different scenarios for comparison
- Document scenario assumptions
- Compare multiple scenarios side-by-side

### 📊 Analytics & Scenarios
**Sensitivity Analysis:**
- Analyze sensitivity to revenue CAGR and COGS
- Generate sensitivity tables
- Identify key value drivers

**Monte Carlo Simulation:**
- Run 100-10,000 simulations
- Vary revenue, COGS, and other parameters
- Calculate confidence intervals
- View probability distributions

**What-If Analysis:**
- Labor cost increase scenarios
- Revenue decline scenarios
- CAPEX expansion scenarios
- Interest rate shock scenarios

### 📈 Reports
**Summary Report:**
- Executive summary with key metrics
- Financial highlights
- Workforce overview
- Capital assets summary

**Detailed Report:**
- Full income statement
- Balance sheet analysis
- Detailed financial metrics

**Export Data:**
- Download labor schedule as CSV
- Export financial model results
- Export CAPEX depreciation schedule

## Global Settings (Sidebar)

- **Annual Salary Growth Rate**: Applied to all labor cost projections
- **Production Forecast**: Set units to produce for 2026-2030
- **Platform Status**: View last update timestamp and version info

## Session State

The app maintains state across interactions:
- Labor manager (positions and configurations)
- CAPEX manager (assets and schedules)
- Financial model results
- Saved scenarios
- Production forecasts
- Last update timestamp

## File Structure

```
streamlit_app.py          # Main Streamlit application
financial_model.py        # Core financial modeling engine
labor_management.py       # Labor management system
capex_management.py       # Capital expenditure management
advanced_analytics.py     # Advanced analytics features
financial_analytics.py    # Basic analytics tools
visualization_tools.py    # Reporting and charting
utils.py                  # Utility functions
requirements.txt          # Python dependencies
```

## Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Streamlit Community Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Select `streamlit_app.py` as main file
5. Deploy!

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t manufacturing-app .
docker run -p 8501:8501 manufacturing-app
```

### AWS/Azure/GCP Deployment
Deploy as containerized application using your cloud provider's container service (ECS, App Service, Cloud Run, etc.)

## Configuration

### Streamlit Config File (`.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#31333F"
font = "sans serif"

[server]
port = 8501
maxUploadSize = 200
```

## Performance Tips

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Session State**: Leverage session state to avoid recalculations
3. **Production Forecast**: Update via sidebar to reflect actual forecasts
4. **Scenario Comparison**: Save multiple scenarios for comparison

## Troubleshooting

**App won't start:**
- Check all dependencies are installed: `pip install -r requirements.txt`
- Verify Python 3.7+ is installed

**Import errors:**
- Ensure all module files (labor_management.py, etc.) are in the same directory
- Check file paths in imports

**Data not persisting:**
- Streamlit reruns entire script when input changes
- Use `st.session_state` to maintain state across reruns

**Performance issues:**
- Reduce number of Monte Carlo simulations
- Use caching decorators for expensive functions
- Simplify visualizations if needed

## Features Summary

✅ Interactive labor management (add/edit/remove positions)
✅ Capital asset management with 4 depreciation methods
✅ Complete financial modeling with DCF valuation
✅ Scenario planning and comparison
✅ Sensitivity analysis
✅ Monte Carlo simulation
✅ What-if analysis
✅ Advanced analytics dashboard
✅ Data export to CSV
✅ Production-ready interface
✅ Professional visualizations
✅ Real-time KPI metrics

## Support & Documentation

- Main Documentation: [README.md](README.md)
- Labor Guide: [LABOR_MANAGEMENT_GUIDE.md](LABOR_MANAGEMENT_GUIDE.md)
- Analytics Guide: [ADVANCED_ANALYTICS_GUIDE.md](ADVANCED_ANALYTICS_GUIDE.md)
- Quick Reference: [LABOR_MANAGEMENT_QUICKREF.md](LABOR_MANAGEMENT_QUICKREF.md)
- GitHub: [Kossit73/Automobile_Manufacturing](https://github.com/Kossit73/Automobile_Manufacturing)

## Version Info

- **Platform Version**: 1.0
- **Release Date**: November 2025
- **Python**: 3.7+
- **Streamlit**: 1.28.0+
- **Status**: Production Ready ✅
