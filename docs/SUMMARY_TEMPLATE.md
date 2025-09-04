# Project Summary Template

*Use this template for your final SUMMARY.md deliverable (≤1 page)*

---

# Lending Club Default Risk Prediction - Summary

## Approach Overview

**Data & Timeline**:
- Training: [specify quarters used]
- Validation: [specify quarter]
- Backtest: [specify quarter]
- Total loans analyzed: [X loans across Y quarters]

**Modeling Strategy**:
- Algorithm: [e.g., Logistic Regression, Random Forest]
- Features: [X listing-time features]
- Validation approach: [temporal split methodology]

## Key Results

### Model Performance
- **ROC-AUC**: [X.XX] on validation set
- **Brier Score**: [X.XX] (lower is better)
- **Calibration**: [brief interpretation, e.g., "well-calibrated across probability ranges" or "overconfident in 0.1-0.3 range"]

### Investment Decision Policy
- **Selection Rule**: [describe your policy, e.g., "Select top 50 loans with lowest predicted default probability"]
- **Budget Utilization**: $5,000 per quarter → [X loans selected on average]

### Backtest Results
- **Selected Default Rate**: [X.X%] vs **Overall Default Rate**: [Y.Y%]
- **ROI Proxy**: [X.X%] average return
- **Risk Reduction**: [X.X percentage points] lower default rate than random selection

## Key Assumptions

1. **ROI Calculation**: [state your assumptions, e.g., "Default occurs at 30% payment completion"]
2. **Missing Data**: [how handled, e.g., "Forward-filled employment length"]
3. **Feature Engineering**: [key assumptions made]

## Top Model Features

1. **[Feature Name]**: [brief interpretation]
2. **[Feature Name]**: [brief interpretation]
3. **[Feature Name]**: [brief interpretation]

**Surprising Finding**: [1-2 sentences about unexpected relationships]

## What Would I Try Next?

1. **[Improvement 1]**: [brief description and expected impact]
2. **[Improvement 2]**: [brief description and expected impact]
3. **[Improvement 3]**: [brief description and expected impact]

## Technical Notes

- **Reproducibility**: Random seed = [X], all dependencies in requirements.txt
- **Data Leakage Check**: ✅ Only listing-time features used
- **Temporal Validation**: ✅ Strict time-ordered splits maintained

---

*Total Time Spent: [X hours]*
