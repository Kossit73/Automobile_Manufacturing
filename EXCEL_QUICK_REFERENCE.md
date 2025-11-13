# 📊 Excel Download Quick Reference

## One-Line Summary
Generate professional multi-sheet Excel workbooks from your financial model in 3 clicks.

---

## Quick Start (60 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run app
streamlit run streamlit_app.py

# 3. Navigate to Reports tab → Excel Model Download section
```

**That's it!** Your Excel export is ready to use.

---

## 3-Step Usage

### Step 1️⃣: Create Model
- Go to **Financial Model** tab
- Set parameters → Click **Run Financial Model**

### Step 2️⃣: Prepare Excel
- Go to **Reports** tab
- Select scenario (Base Case/Conservative/Optimistic)
- Click **📄 Prepare Excel Model** (wait for ✅ confirmation)

### Step 3️⃣: Download
- Click **⬇️ Download Excel Model**
- File saved: `Automobile_Manufacturing_Financial_Model_[Scenario].xlsx`

---

## What You Get

### 4 Professional Sheets

| Sheet | Contents | Rows |
|-------|----------|------|
| **Executive Summary** | Key metrics, valuation, timestamp | 10 |
| **Financial Model** | 5-year P&L, revenue to FCF | 5 + headers |
| **Labor Schedule** | Headcount, costs by category | 5 + headers |
| **CAPEX Schedule** | Assets, depreciation, lives | N + headers |

---

## File Details

- **Format:** .xlsx (Excel 2016+)
- **Size:** 150-300 KB
- **Scenarios:** Base Case, Conservative, Optimistic
- **Style:** Professional (colors, fonts, borders, currency)
- **Compatibility:** Excel, LibreOffice, Google Sheets

---

## UI Controls

```
Scenario: [Base Case ▼]

[📄 Prepare] [⬇️ Download] [🗑️ Clear]

[CSV Export Buttons Below]
```

**Behavior:**
- Click **Prepare** → Generates workbook (2-5 seconds)
- Click **Download** → Saves to computer
- Click **Clear** → Removes from memory, prepare again if needed
- Switch scenario → Separate workbooks cached

---

## Common Tasks

### Export Base Case
```
1. Scenario dropdown → Base Case (default)
2. Click Prepare button
3. Wait for ✅ confirmation
4. Click Download button
5. File saved automatically
```

### Compare Scenarios
```
1. Prepare Base Case (see above)
2. Switch scenario → Conservative
3. Click Prepare
4. Download Conservative file
5. Open both files side-by-side
6. Repeat for Optimistic
```

### Share with Stakeholders
```
1. Generate all 3 scenarios
2. Download all 3 files
3. Email attachments (each ~200 KB)
4. Provide scenario definitions in email body
5. Request feedback by [date]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Prepare" button grayed out | Run Financial Model first (Financial tab) |
| "openpyxl required" error | Run: `pip install openpyxl` |
| Download not starting | Check browser download settings, try again |
| File appears corrupted | Regenerate by clicking "Clear" then "Prepare" |
| Missing data in Excel | Verify all data entered in Labor/CAPEX tabs first |

---

## Key Features

✅ **Multi-sheet:** 4 sheets in one workbook
✅ **Professional:** Color-coded, formatted, styled
✅ **Fast:** 2-5 second generation, cached retrieval
✅ **Flexible:** 3 scenario templates
✅ **Smart:** Session state caching, manual clear
✅ **Reliable:** Error handling, graceful fallback
✅ **Compatible:** Works with Excel, LibreOffice, Google Sheets
✅ **Portable:** Small file size, email-friendly

---

## What's in Each Sheet

### Executive Summary
```
Automobile Manufacturing Financial Model - Base Case
Generated: 2025-11-13 14:30:45

Key Metrics
2026 Revenue                    $XX,XXX,XXX.XX
2026 EBIT                       $XX,XXX,XXX.XX
2026 Net Profit                 $XX,XXX,XXX.XX
2026 FCF                        $XX,XXX,XXX.XX
Enterprise Value                $XX,XXX,XXX.XX
5-Year Total FCF                $XX,XXX,XXX.XX
```

### Financial Model
```
Year | Revenue | COGS | GP | OPEX | EBITDA | Dep | EBIT | Interest | EBT | Taxes | NP | FCF
2026 |   ...
2027 |   ...
...
2030 |   ...
```

### Labor Schedule
```
Year | Category | Headcount | Direct Cost | Indirect Cost | Total Cost
2026 |    ...
```

### CAPEX Schedule
```
ID | Name | Category | Acquisition Cost | Acq Date | Useful Life | Annual Dep
```

---

## Performance

| Metric | Value |
|--------|-------|
| First generation | 2-5 seconds |
| Cached download | <100ms |
| File size | 150-300 KB |
| Memory per file | ~5-10 MB |
| Max cached files | 3 scenarios |
| Total cache RAM | ~15-30 MB |

---

## Installation

### Requirements
```bash
pip install streamlit>=1.28.0
pip install pandas>=1.5.0
pip install numpy>=1.23.0
pip install plotly>=5.15.0
pip install openpyxl>=3.9.0
```

### Or All at Once
```bash
pip install -r requirements.txt
```

---

## System Requirements

- **Python:** 3.7+ (recommended 3.9+)
- **OS:** Windows, macOS, Linux
- **RAM:** 512 MB minimum (1+ GB recommended)
- **Disk:** 50 MB for app, 1 GB recommended
- **Browser:** Chrome, Firefox, Safari, Edge (all modern versions)

---

## Tips & Tricks

### 🎯 For Analysis
- Generate Base Case first for baseline
- Compare with Conservative/Optimistic scenarios
- Use Executive Summary for quick overview
- Use Financial Model sheet for deep dive

### 💼 For Sharing
- Include scenario name in filename
- Provide assumptions document separately
- Use comments-only mode in Excel to prevent changes
- Archive versions with dates

### ⚡ For Speed
- Multiple scenarios stay cached - just download
- No need to regenerate unless data changes
- Use Clear button to free RAM if needed
- CSV exports available for quick downloads

### 🔧 For Customization
- Edit `_generate_excel_bytes()` in streamlit_app.py
- Modify colors: change header_fill HEX code
- Add sheets: create new ws = wb.create_sheet()
- Adjust formatting: Font, PatternFill, Alignment settings

---

## Excel Keyboard Shortcuts

Once downloaded, use these in Excel:

| Shortcut | Action |
|----------|--------|
| Ctrl+A | Select all |
| Ctrl+F | Find/Replace |
| Ctrl+P | Print |
| Ctrl+Home | Go to cell A1 |
| Ctrl+End | Go to last cell |
| F5 | Go To dialog |
| Ctrl+Right | Next data region |
| Alt+Page Down | Scroll right |

---

## FAQ

**Q: Can I edit the Excel file?**
A: Yes! Downloaded files are fully editable in Excel.

**Q: Does editing the Excel affect the app?**
A: No. Excel files are standalone exports.

**Q: How many scenarios can I generate?**
A: As many as you want - Base/Conservative/Optimistic are defaults, but you can create more.

**Q: Can I schedule automatic exports?**
A: Not yet - manually generate when needed.

**Q: What if I close the browser?**
A: Cache is cleared. Re-run app and regenerate if needed.

**Q: Can I export to other formats?**
A: CSV available in same tab. XML/JSON available via file menu.

---

## Next Steps

1. ✅ Install openpyxl: `pip install -r requirements.txt`
2. ✅ Start app: `streamlit run streamlit_app.py`
3. ✅ Create model with parameters
4. ✅ Go to Reports tab
5. ✅ Generate Excel workbook
6. ✅ Download and explore
7. ✅ Share with stakeholders
8. ✅ Collect feedback

---

## Resources

- 📖 [Full Excel Guide](EXCEL_DOWNLOAD_GUIDE.md)
- 🏗️ [Implementation Details](EXCEL_IMPLEMENTATION_SUMMARY.md)
- 🚀 [Deployment Guide](STREAMLIT_DEPLOYMENT.md)
- ⚡ [Quick Start](STREAMLIT_QUICKSTART.md)
- 📋 [Main README](README.md)

---

## Support

Having issues? Check:
1. Requirements installed: `pip list | grep openpyxl`
2. App running: Terminal shows "Local URL: http://localhost:8501"
3. Financial model generated: Executive Summary visible
4. Browser console: F12 → Console tab for errors
5. Test CSV export first: Usually works if Excel doesn't

---

**Automobile Manufacturing Financial Platform** © 2025
[GitHub](https://github.com/Kossit73/Automobile_Manufacturing)
