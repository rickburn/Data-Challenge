# Getting Started Guide

This guide provides step-by-step instructions to begin the Lending Club Data Challenge project.

## Prerequisites

- Python 3.8+ installed
- Git for version control
- Jupyter Notebook or preferred IDE
- Basic familiarity with pandas, scikit-learn, and matplotlib

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd /home/rick/repos/LHAI-Data-Challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install development dependencies (includes production deps)
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .

# Set up pre-commit hooks for code quality
pre-commit install
```

### 2. Verify Installation

```bash
# Test package import
python -c "from lending_club.models.data_models import LoanApplication; print('Package installed successfully!')"

# Run initial tests
pytest tests/unit/test_data_models.py -v

# Check code quality setup
pre-commit run --all-files
```

### 3. Initial Data Exploration

```bash
# Check data structure
ls -la data/
wc -l data/*.csv  # Count lines in each file

# Open data dictionary
libreoffice data/data_dictionary.xlsx  # or Excel equivalent

# Check first few lines of data
head -2 data/2016Q1.csv
```

### 4. Project Structure Verification

The package structure is already set up with:

```bash
# Verify structure
tree src/ tests/ -I __pycache__
```

Expected output:
```
src/
└── lending_club/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── data_models.py
    │   └── enums.py
    └── ...
tests/
├── conftest.py
├── unit/
│   └── test_data_models.py
└── ...
```

### 5. Data Safety Check

**CRITICAL**: Before starting analysis, review the listing-time rule:

1. Open `docs/REQUIREMENTS.md`
2. Review the "Prohibited Fields" section  
3. Use the `LoanApplication` Pydantic model to enforce data validation
4. Keep the prohibited fields list handy during feature selection

## Recommended Workflow

### Phase 1: Data Understanding (1-1.5 hours)
1. **Load and inspect** each quarterly CSV file
2. **Review data dictionary** for field definitions
3. **Identify prohibited fields** (post-origination data)
4. **Basic EDA**: distributions, missing values, data types
5. **Document findings** in notebook/script

### Phase 2: Feature Engineering (1-1.5 hours)
1. **Implement feature engineering** in `src/lending_club/features/`
2. **Use Pydantic models** for data validation and type safety
3. **Handle missing values** with documented strategies
4. **Engineer derived features** (e.g., debt-to-income ratios) as model properties
5. **Create feature provenance table**
6. **Write unit tests** for feature engineering functions
7. **Temporal validation**: ensure no future leakage using model constraints

### Phase 3: Modeling (1.5-2 hours)
1. **Use model evaluation utilities** from `src/lending_club/evaluation/`
2. **Time-ordered split**: early quarters → later quarter with date validation
3. **Train baseline model** (start simple: logistic regression)
4. **Calibrate probabilities** using sklearn calibration
5. **Evaluate**: ROC-AUC, Brier score, calibration plot
6. **Feature importance** analysis with model interpretation
7. **Run integration tests** to verify end-to-end pipeline

### Phase 4: Decision Policy (0.5-1 hour)
1. **Define selection rule** for $5,000 budget
2. **Apply to validation set**
3. **Count selected loans** per quarter
4. **Document policy** clearly

### Phase 5: Backtesting (0.5-1 hour)
1. **Apply policy** to held-out quarter
2. **Calculate metrics**: default rate, ROI proxy
3. **Compare** against overall performance
4. **Document assumptions** clearly

### Phase 6: Documentation (0.5-1 hour)
1. **Write SUMMARY.md** using template
2. **Complete AI usage disclosure**
3. **Finalize requirements.txt**
4. **Final reproducibility check**

## Key Checkpoints

- [ ] **Data Leakage Check**: No prohibited fields used
- [ ] **Temporal Validation**: Train dates < validation dates < test dates  
- [ ] **Reproducibility**: Fixed random seeds, complete requirements
- [ ] **Budget Constraint**: $5,000 limit properly applied
- [ ] **Calibration**: Reliability curve and interpretation included
- [ ] **Documentation**: All assumptions and decisions documented

## Common Pitfalls to Avoid

1. **Using post-origination data** → Review prohibited fields list
2. **Random train/test splits** → Use temporal splits only  
3. **Unclear selection rule** → Document budget policy explicitly
4. **Poor calibration** → Include reliability curve analysis
5. **Missing assumptions** → Document ROI calculation assumptions
6. **Non-reproducible results** → Fix seeds, pin dependencies

## Getting Help

- **Requirements**: See `docs/REQUIREMENTS.md` for full specification
- **Structure**: See `docs/PROJECT_STRUCTURE.md` for file organization
- **Templates**: Use `docs/SUMMARY_TEMPLATE.md` and `docs/AI_USAGE_TEMPLATE.md`

## Time Management Tips

- **Total budget**: 4-6 hours
- **Start simple**: Basic logistic regression first
- **Document as you go**: Don't leave documentation to the end
- **Timebox each phase**: Don't over-optimize one component
- **Focus on clarity**: Readable code > perfect model

---

**Ready to start?** Begin with data exploration in your chosen analysis environment!
