# AI-Era Data Challenge — Lending Club (Intern)

## Project Overview

This project involves building a reproducible machine learning pipeline to predict default risk for Lending Club loans and optimize investment decisions under budget constraints.

## Objective

You are advising a retail investor choosing among newly listed Lending Club loans. Build a small, reproducible pipeline that:

1. **Cleans and explores the data**
2. **Trains a baseline model** to predict default risk at listing time
3. **Converts probabilities into investment choices** under a budget constraint
4. **Backtests the policy** on a held-out quarter

**Philosophy**: Optimize for clarity over perfection. The scope is intentionally intern-level and time-boxed.

## Data

- **Location**: Quarterly CSVs provided in `data/` (2016Q1–2017Q4)
- **Data Dictionary**: Available in `docs/data_dictionary.xlsx`
- **Decision Windows**: Treat each quarter's listings as a separate decision window

### Critical Data Constraint: Listing-Time Only Rule

**⚠️ IMPORTANT**: Use only information known when a loan is first listed. Do not use post-origination outcomes or fields that directly/indirectly reveal them.

**Prohibited Fields** (examples):
- `loan_status`
- `last_pymnt_d`, `last_pymnt_amnt`
- `total_rec_prncp`, `total_rec_int`
- `recoveries`, `collection_recovery_fee`
- `out_prncp`, `next_pymnt_d`
- Any `*_rec_*` fields
- Any `*_pymnt*` fields
- Any `chargeoff*` fields
- Any `settlement*` fields

**Validation Rule**: Ask yourself: "Could this value exist before the first payment was ever made?" If not, it's disallowed.

## Tasks & Requirements

### 1. EDA & Cleaning (20 pts)
- Handle data types, missing values, and obvious outliers
- Document any dropped columns with justification
- Produce a data summary:
  - Number of rows and columns
  - Target variable prevalence
  - Key insights from exploration

### 2. Feature Set - Listing-Time Safe (15 pts)
- Engineer features available at loan listing time:
  - Loan amount, interest rate, term
  - Applicant employment information
  - FICO range
  - Other pre-origination data
- **Deliverable**: Feature provenance table explaining why each feature is valid at listing time

### 3. Baseline Model & Evaluation (20 pts)

#### Model Requirements:
- Train a simple classifier (logistic regression, tree, or GBM)
- Output calibrated probabilities of default (PD)
- Use **time-ordered split**: 
  - Train on earlier quarters
  - Validate on next quarter
  - Example: Train (2016Q1–2016Q3), Validate (2016Q4)

#### Evaluation Metrics:
- **ROC-AUC**
- **Calibration (reliability curve)**
- **Brier score**
- **Calibration interpretation**: Brief explanation (e.g., "over-confident in the 0.1–0.2 bin")

### 4. Decision Policy & Budget (20 pts)
- **Budget Constraint**: $5,000 per quarter
- **Selection Rule**: Define loan selection based on predicted risk
  - Options: Top-K with lowest PD, highest expected value, etc.
- **Documentation**: Spell out your exact rule (threshold or ranking)
- **Reporting**: Count selected loans per quarter

### 5. Backtest (20 pts)

#### Test Setup:
- Apply selection rule to held-out later quarter (e.g., 2017Q1)
- Compare performance against baseline

#### Required Metrics:
- **Default Rate Comparison**: Selected vs. overall default rate
- **ROI Proxy**: Simple return calculation with documented assumptions

**Example ROI Calculation**:
```
ROI_proxy = (collected_payments - principal) / principal

Assumptions:
- If no default: collected_payments ≈ installment * term_months
- If default: collected_payments ≈ 0.30 * installment * term_months
```

### 6. Model Explainability (15 pts)
- Show top features (coefficients or feature importance)
- Identify 1-2 surprising relationships
- Provide business interpretation

### 7. AI-Use Transparency (5 pts)
- Document where AI assistance was used
- Explain how AI outputs were validated
- Examples: boilerplate EDA, code generation, debugging

### 8. Optional Extension (+5 pts)
Add one lightweight text-derived feature:
- Examples: `emp_title` length, contains digits/keywords
- Measure and report impact on model performance

## Deliverables

### Required Files:
1. **Main Analysis**: Single notebook (`.ipynb`) or Python script (`.py`) that runs end-to-end
2. **Summary Report**: `SUMMARY.md` (≤1 page) containing:
   - Approach overview
   - Key metrics and results
   - Assumptions made
   - Decision rule explanation
   - Next steps and improvements
3. **Dependencies**: `requirements.txt` with pinned versions
4. **AI Disclosure**: Documentation of AI assistance usage

### Submission Method:
- GitHub Pull Request with all deliverables

## Guardrails & Quality Checks

### Data Leakage Prevention:
- ✅ Only listing-time features used
- ✅ No post-event/banned fields included
- ✅ Temporal validation: `max(issue_d)` in train < `min(issue_d)` in validation

### Reproducibility:
- ✅ Fixed random seeds where applicable
- ✅ Complete `requirements.txt`
- ✅ Code runs end-to-end locally

### Model Quality:
- ✅ Calibration plot and Brier score included
- ✅ Explicit decision policy with backtesting
- ✅ No random data splits (time-ordered only)

### Common Pitfalls to Avoid:
- ❌ Using random train/test splits instead of temporal
- ❌ Including obvious outcome fields
- ❌ Skipping decision policy or backtest steps
- ❌ Poorly documented assumptions

## Timeline & Scope

- **Time Budget**: 4–6 hours total
- **Focus**: Clarity and reproducibility over perfect optimization
- **Level**: Intern-appropriate scope and complexity

## Scoring Rubric (100 pts)

| Component | Points | Focus Areas |
|-----------|---------|-------------|
| Data hygiene & EDA | 20 | Sensible cleaning, types, missingness, clear documentation |
| Leakage avoidance | 15 | Listing-time discipline, avoiding outcome field traps |
| Modeling & calibration | 20 | Baseline model with PDs, calibration + interpretation |
| Decision & backtest | 20 | Coherent rule, budget application, metrics reporting |
| Reasoning & communication | 15 | Clear SUMMARY.md, trade-offs, next steps |
| AI-use transparency | 5 | Documentation of AI assistance and validation |
| Optional extension | 5 | Text feature with measured effect |

## Technical Notes

- **Class Imbalance**: May use downsampling/upsampling or class weights (document choice)
- **Alternative Approaches**: Different ROI proxies or selection rules acceptable if well-documented
- **Code Quality**: Prioritize readability and clear reasoning over complex optimization

---

*This specification provides the framework for a comprehensive yet manageable data science project focused on practical loan default prediction and investment decision-making.*
