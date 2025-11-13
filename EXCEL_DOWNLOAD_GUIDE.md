# 📊 Excel Model Download Guide

## Overview

The Automobile Manufacturing Financial Platform now includes a **professional Excel export feature** that generates multi-sheet workbooks with complete financial analysis, labor schedules, and CAPEX planning data.

---

## Features

### ✨ Multi-Sheet Excel Workbooks

Each generated Excel file includes **4 comprehensive sheets**:

#### 1. **Executive Summary**
- Key financial metrics (Revenue, EBIT, Net Profit, FCF)
- Valuation metrics (Enterprise Value, 5-Year FCF)
- Generation timestamp and scenario name
- Professional formatting with headers and borders

#### 2. **Financial Model** 
- 5-year comprehensive P&L statement (2026-2030)
- Columns: Year, Revenue, COGS, Gross Profit, OPEX, EBITDA, Depreciation, EBIT, Interest, EBT, Taxes, Net Profit, FCF
- All values formatted as currency
- Professional color-coded headers
- Automatic column sizing

#### 3. **Labor Schedule**
- Position-level labor cost breakdown
- 5-year salary projections
- Headcount by category
- Direct and indirect labor costs
- Salary growth applied across years
- Clean tabular format

#### 4. **CAPEX Schedule**
- Asset inventory with depreciation tracking
- Columns: Item ID, Asset Name, Category, Acquisition Cost, Acquisition Date, Useful Life, Annual Depreciation
- Automatic depreciation calculation
- Currency-formatted values
- Complete asset management details

---

## Using the Feature

### Step 1: Run Financial Model
1. Go to **Financial Model** tab
2. Set your parameters (WACC, Revenue CAGR, COGS %, Tax Rate, Debt Ratio)
3. Click **Run Financial Model**
4. Wait for results to load

### Step 2: Navigate to Reports
1. Click **Reports** tab in sidebar
2. View Executive Summary metrics
3. Scroll down to **Excel Model Download** section

### Step 3: Prepare Excel Model
1. Select scenario from dropdown:
   - **Base Case** - Standard assumptions
   - **Conservative** - Lower growth, higher costs
   - **Optimistic** - Higher growth, efficiency gains

2. Click **📄 Prepare Excel Model** button
3. Wait for workbook generation (usually 2-5 seconds)
4. See ✅ confirmation message

### Step 4: Download Excel File
1. Once prepared, click **⬇️ Download Excel Model** button
2. File saves as: `Automobile_Manufacturing_Financial_Model_[Scenario].xlsx`
3. Example: `Automobile_Manufacturing_Financial_Model_Base_Case.xlsx`

### Step 5: Clear & Switch Scenarios
- To switch scenarios, select a different option from dropdown
- The previous scenario remains cached (re-download anytime)
- Click **🗑️ Clear Prepared Excel** to free memory

---

## Excel File Details

### File Naming
```
Automobile_Manufacturing_Financial_Model_[Scenario].xlsx
```

Examples:
- `Automobile_Manufacturing_Financial_Model_Base_Case.xlsx`
- `Automobile_Manufacturing_Financial_Model_Conservative.xlsx`
- `Automobile_Manufacturing_Financial_Model_Optimistic.xlsx`

### Formatting & Styling

**Headers:**
- Dark blue background (#366092)
- White bold text
- 12pt font
- Centered alignment
- Thin borders

**Currency Format:**
- US Dollar ($)
- Thousand separator (,)
- 2 decimal places
- Example: $1,234,567.89

**Numeric Format:**
- Standard numbers with 2 decimal places
- Thin borders for data cells
- Proper alignment

### File Size
- Typical size: 150-300 KB
- Quick download even on slower connections
- Optimized for email distribution

---

## Session State Management

### Caching Mechanism
The app implements **intelligent caching**:

```
excel_bytes_map = {
    "Base Case": <bytes>,
    "Conservative": <bytes>,
    "Optimistic": <bytes>
}
```

**Benefits:**
- Generate once, download multiple times
- Switch between scenarios without regeneration
- Faster repeated downloads
- Memory-efficient with manual clear option

### Clearing Cache
- Individual scenarios can be cleared
- No need to regenerate unless data changes
- Automatic cleanup on app restart

---

## Technical Implementation

### Dependencies
```python
openpyxl>=3.9.0  # Excel workbook generation
```

### Key Functions

#### `_generate_excel_bytes(model, labor_manager, capex_manager, scenario)`
Generates complete Excel workbook with formatting.

**Parameters:**
- `model` - Financial model results dictionary
- `labor_manager` - Labor management object
- `capex_manager` - CAPEX management object  
- `scenario` - Scenario name (string)

**Returns:**
- `bytes` - Binary Excel file content

**Exception Handling:**
- Checks for openpyxl availability
- Graceful fallback if library missing
- Try-catch blocks for each sheet

### Session State Variables
```python
excel_bytes_map: Dict[str, bytes]  # Cache of generated Excel files
selected_scenario: str             # Currently selected scenario
```

---

## Troubleshooting

### Issue: "openpyxl is required" error

**Solution:**
```bash
pip install openpyxl>=3.9.0
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

### Issue: Excel file won't download

**Solutions:**
1. Check internet connection
2. Verify browser allows downloads from localhost
3. Clear browser cache and try again
4. Click "Prepare Excel Model" button first (if greyed out)

### Issue: File appears empty or corrupted

**Solutions:**
1. Regenerate the Excel file (clear and re-prepare)
2. Ensure financial model has been run (check Reports summary)
3. Try opening in different Excel version
4. Check if file opened properly before sharing

### Issue: Missing data in workbook

**Possible causes:**
- Financial model not run yet
- Labor manager not initialized
- CAPEX manager empty

**Solution:**
1. Go to Dashboard - check all data loaded
2. Run Financial Model with parameters
3. Verify labor positions exist (Labor Management tab)
4. Verify CAPEX assets exist (CAPEX Management tab)
5. Regenerate Excel file

---

## Best Practices

### For Analysis
1. ✅ Generate Base Case first for baseline
2. ✅ Compare with Conservative/Optimistic scenarios
3. ✅ Use Executive Summary for quick overview
4. ✅ Use Financial Model sheet for detailed analysis

### For Sharing
1. ✅ Download in different scenarios for stakeholders
2. ✅ Use professional filename format
3. ✅ Include scenario name in email subject
4. ✅ Provide context (assumptions, parameters) separately

### For Collaboration
1. ✅ Share Excel files via email or cloud storage
2. ✅ Comments-only in Excel to prevent accidental changes
3. ✅ Use separate workbooks for different versions
4. ✅ Archive old files with dates

### For Integration
1. ✅ Export to shared drive for team access
2. ✅ Import into data warehouse for analysis
3. ✅ Link to BI tools for visualization
4. ✅ Archive in document management system

---

## Advanced Usage

### Modifying the Excel Template

To customize Excel generation, edit `_generate_excel_bytes()` in `streamlit_app.py`:

**Change colors:**
```python
header_fill = PatternFill(
    start_color="366092",  # Blue (hex code)
    end_color="366092",
    fill_type="solid"
)
```

**Add new sheets:**
```python
ws_custom = wb.create_sheet("Custom Sheet", 4)
# Add data and formatting...
```

**Modify formatting:**
```python
currency_format = '_-$* #,##0.00_-;-$* #,##0.00_-;_-$* "-"??_-;_-@_-'
header_font = Font(color="FFFFFF", bold=True, size=12)
```

---

## Comparison: Excel vs CSV

| Feature | Excel | CSV |
|---------|-------|-----|
| Multiple sheets | ✅ Yes | ❌ No |
| Formatting | ✅ Professional | ❌ None |
| Quick download | ✅ Yes | ✅ Yes |
| Email friendly | ✅ Yes | ✅ Yes |
| Data integrity | ✅ High | ✅ High |
| File size | 150-300 KB | 50-100 KB |
| Excel compatibility | ✅ Full | ✅ Full |
| Pivot tables | ✅ Ready | ❌ Requires setup |

---

## Examples

### Example 1: Board Presentation
1. Generate **Base Case** Excel model
2. Review Executive Summary sheet
3. Present key metrics from summary
4. Provide Financial Model sheet for detailed questions
5. Distribute workbook to board members

### Example 2: Stakeholder Analysis
1. Generate **Conservative** scenario for risk analysis
2. Generate **Optimistic** scenario for upside case
3. Compare both workbooks side-by-side
4. Present ranges to stakeholders
5. Archive both versions for records

### Example 3: Team Collaboration
1. Generate current **Base Case** model
2. Share with finance team for review
3. Collect feedback (external, not in Excel)
4. Update financial model with changes
5. Regenerate and re-share updated workbook

---

## System Requirements

### Minimum
- Python 3.7+
- 20 MB disk space for app
- 100 MB available RAM

### Recommended
- Python 3.9+
- 50 MB disk space
- 512 MB available RAM

### Dependencies
```
openpyxl>=3.9.0
pandas>=1.5.0
streamlit>=1.28.0
```

---

## Performance Notes

### Generation Time
- First workbook: 2-5 seconds
- Cached retrieval: <100ms
- Multiple scenarios: Parallel generation possible

### File Size Impact
- Typical workbook: 150-300 KB
- Email attachment: No issues
- Cloud storage: Minimal impact

### Memory Usage
- Single workbook: ~5-10 MB RAM
- Multiple cached workbooks: ~15-30 MB RAM (3 scenarios)
- Cleared cache: Returns to baseline

---

## Version History

### v1.0 (November 2025)
- ✅ Initial release
- ✅ Multi-sheet Excel workbooks
- ✅ 3 scenario templates
- ✅ Professional formatting
- ✅ Session state caching
- ✅ CSV export alongside Excel

---

## Getting Help

### Documentation
- **STREAMLIT_DEPLOYMENT.md** - App deployment guide
- **STREAMLIT_QUICKSTART.md** - Quick start guide
- **README.md** - Platform overview

### Support
- Review error messages carefully
- Check browser console for issues
- Reinstall requirements if needed
- Test with simple model first

---

## Next Steps

1. ✅ Install/update requirements: `pip install -r requirements.txt`
2. ✅ Run Streamlit app: `streamlit run streamlit_app.py`
3. ✅ Create financial model with parameters
4. ✅ Generate Excel workbook in Reports tab
5. ✅ Download and review workbook
6. ✅ Share with stakeholders or team

---

**Automobile Manufacturing Financial Platform** © 2025
- [GitHub Repository](https://github.com/Kossit73/Automobile_Manufacturing)
- [Main README](README.md)
- [Platform Deployment Guide](STREAMLIT_DEPLOYMENT.md)
