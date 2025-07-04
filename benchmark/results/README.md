# SpectroChain-Dental Benchmark Results

This directory contains the results from benchmark runs.

## 📊 File Types

### HTML Reports

- `benchmark_report_*.html` - Interactive HTML reports with visualizations
- `dashboard.html` - Real-time dashboard

### JSON Data

- `benchmark_*.json` - Raw benchmark data
- `performance_*.json` - Performance test results
- `security_*.json` - Security analysis results
- `accuracy_*.json` - Accuracy evaluation results

### Visualizations

- `*.png` - Static charts and graphs
- `*.svg` - Vector graphics
- `confusion_matrices.png` - ML model confusion matrices
- `tps_chart.png` - TPS performance chart
- `stride_chart.png` - Security STRIDE analysis

## 📈 How to View Results

### HTML Reports

Open the HTML files in any modern web browser:

```bash
# Windows
start benchmark_report_*.html

# Linux/Mac
open benchmark_report_*.html
```

### JSON Data

Use any JSON viewer or load in Python:

```python
import json
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)
```

## 🔄 Automated Cleanup

Results older than 30 days are automatically cleaned up to save disk space.
To preserve important results, copy them to a backup location.

## 📋 Results Structure

```
results/
├── benchmark_report_YYYYMMDD_HHMMSS.html    # Main report
├── dashboard.html                            # Interactive dashboard
├── benchmark_YYYYMMDD_HHMMSS.json          # Raw data
├── performance_charts/                       # Performance visualizations
├── security_analysis/                        # Security charts
└── accuracy_plots/                          # ML accuracy plots
```
