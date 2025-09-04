# Lending Club ML Pipeline - Execution Guide

## Quick Start

### 1. Prerequisites Check
```bash
python run_pipeline.py --dry-run
```

### 2. Run the Complete Pipeline
```bash
python run_pipeline.py
```

### 3. Debug Mode (if issues occur)
```bash
python run_pipeline.py --debug
```

## What the Pipeline Does

The pipeline executes these steps automatically:

1. **📊 Data Loading**: Loads quarterly CSV files (2016Q1-Q3 for training, 2016Q4 for validation, 2017Q1 for backtesting)

2. **🔧 Feature Engineering**: Creates 50+ features from loan and borrower data while ensuring listing-time compliance

3. **🤖 Model Training**: Trains a logistic regression model with hyperparameter optimization and probability calibration

4. **💰 Investment Decisions**: Selects optimal loan portfolio under $5,000 budget constraint using lowest-risk strategy

5. **📈 Backtesting**: Evaluates performance on held-out 2017Q1 data and generates comprehensive metrics

6. **📋 Reporting**: Creates detailed HTML report with visualizations and analysis

## Expected Runtime

- **Total Time**: 5-15 minutes (depending on data size and system performance)
- **Data Loading**: 30-60 seconds
- **Feature Engineering**: 1-3 minutes  
- **Model Training**: 2-5 minutes (with hyperparameter search)
- **Investment Decisions**: 10-30 seconds
- **Backtesting**: 30-60 seconds
- **Report Generation**: 10-30 seconds

## Output Files

After successful execution, check these directories:

### 📁 `outputs/reports/`
- `pipeline_report_YYYYMMDD_HHMMSS.html` - Comprehensive HTML report

### 📁 `outputs/models/`
- `model_logistic_YYYYMMDD_HHMMSS.joblib` - Trained model
- `metadata_logistic_YYYYMMDD_HHMMSS.joblib` - Model metadata
- `feature_importance_logistic_YYYYMMDD_HHMMSS.png` - Feature importance plot

### 📁 `outputs/figures/`
- `model_performance_overview.png` - Model metrics overview
- `calibration_analysis.png` - Probability calibration analysis
- `investment_performance.png` - Portfolio performance analysis
- `risk_analysis.png` - Risk distribution analysis

### 📁 `logs/`
- `pipeline_execution.log` - Main execution log
- `operations/data_operations.jsonl` - Data transformation tracking
- `operations/model_training.jsonl` - Model training logs
- `performance/performance_metrics.jsonl` - Performance metrics
- `data_lineage.jsonl` - Complete data lineage

## Key Results to Check

### 1. Model Performance
- **ROC-AUC > 0.65**: Good discrimination between defaults and non-defaults
- **Brier Score < 0.20**: Well-calibrated probability predictions
- **Calibration Slope ≈ 1.0**: Reliable probability estimates

### 2. Investment Performance  
- **Portfolio ROI > 0**: Positive expected returns
- **Default Rate**: Actual vs predicted default rates
- **Budget Utilization**: Efficient use of $5,000 budget
- **Diversification**: Spread across multiple risk grades

### 3. Compliance
- **✅ Listing-Time Only**: No post-origination data used
- **✅ Temporal Ordering**: Proper train/validation/test splits
- **✅ Budget Constraint**: Maximum $5,000 investment respected

## Troubleshooting

### Common Issues

**1. "Data files not found"**
```bash
# Make sure these files exist:
ls data/2016Q*.csv data/2017Q1.csv
```

**2. "Package not found"** 
```bash
pip install -r requirements.txt
```

**3. "Configuration file missing"**
```bash
# Check config file exists:
ls config/pipeline_config.yaml
```

**4. "Memory error during training"**
```bash
# Reduce data size in config (development section):
# test_data_size: 1000
```

### Performance Issues

**Slow execution?**
- Reduce `search_iterations` in config (model section)
- Disable hyperparameter search: `hyperparameter_search: false`
- Use smaller dataset for testing

**Out of memory?**  
- Reduce `chunk_size` in config (performance section)
- Disable feature selection: `max_features: 25`

## Configuration Customization

Edit `config/pipeline_config.yaml` to customize:

### Model Settings
```yaml
model:
  type: "logistic"  # Options: logistic, random_forest, xgboost
  hyperparameter_search: true  # Set to false for faster execution
  search_iterations: 50  # Reduce for faster training
```

### Investment Strategy
```yaml
investment:
  selection_strategy: "lowest_risk"  # Options: lowest_risk, highest_expected_value, balanced_portfolio
  budget_per_quarter: 5000.0
  max_default_probability: 0.25
```

### Output Settings
```yaml
output:
  save_model_artifacts: true
  save_feature_importance: true
  generate_plots: true
```

## Understanding the Results

### The HTML Report
The generated HTML report contains:
- **Executive Summary**: Key metrics and findings
- **Model Performance**: Discrimination and calibration analysis  
- **Investment Analysis**: Portfolio construction and risk metrics
- **Backtest Results**: Out-of-sample performance validation
- **Technical Appendix**: Implementation details and assumptions

### Key Success Metrics
- **Model ROC-AUC ≥ 0.65**: Successfully distinguishes risky loans
- **Portfolio ROI > 0**: Generates positive expected returns
- **Budget Utilization ≈ 100%**: Efficient capital deployment
- **Well-Calibrated Predictions**: Reliable probability estimates

## Next Steps

After successful execution:

1. **📊 Review the HTML Report** in `outputs/reports/` for detailed analysis
2. **🔍 Examine Model Performance** using the generated visualizations
3. **💰 Analyze Investment Decisions** and portfolio composition
4. **📈 Validate Backtesting Results** against expectations
5. **⚙️ Experiment with Different Configurations** to optimize performance

---

**Need Help?** Check the logs in the `logs/` directory for detailed execution information.
