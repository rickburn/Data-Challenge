# Lending Club Data Challenge

A professional-grade machine learning package for predicting loan default risk and optimizing investment decisions for retail investors.

## Project Overview

This project builds a reproducible, strongly-typed pipeline that:
- 🔍 **Analyzes** Lending Club loan data from 2016-2017
- 🤖 **Predicts** default risk using only listing-time information  
- 💰 **Optimizes** investment decisions under budget constraints
- 📊 **Backtests** performance on held-out data
- ✅ **Enforces** data integrity with Pydantic models
- 🧪 **Validates** correctness with comprehensive test suite

**Key Constraint**: Only use information available at loan listing time (no post-origination data).

## Package Features

- **Type Safety**: Strongly typed data models using Pydantic
- **Data Validation**: Automatic validation of listing-time constraints
- **Test Coverage**: Comprehensive unit and integration test suite
- **Code Quality**: Pre-commit hooks with linting, formatting, and type checking
- **Modern Packaging**: PEP 518 compliant with `pyproject.toml`

## Quick Start

### Installation

```bash
# Clone the repository
cd /home/rick/repos/LHAI-Data-Challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### Usage

```python
from lending_club.models.data_models import LoanApplication, InvestmentPolicy
from lending_club.models.enums import LoanGrade, HomeOwnership

# Create a validated loan application
loan = LoanApplication(
    id=12345,
    funded_amnt=10000,
    term=36,
    int_rate=12.5,
    sub_grade=LoanGrade.B2,
    home_ownership=HomeOwnership.RENT,
    # ... other required fields
)

# Investment policy with budget constraints
policy = InvestmentPolicy(
    budget_per_quarter=5000,
    max_risk_tolerance=0.15
)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/lending_club --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run with type checking
mypy src/
```

## Documentation

1. **Read the requirements**: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md)
2. **Follow the setup guide**: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)  
3. **Review project structure**: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

## Data

- **Training Data**: 2016Q1 - 2016Q3 quarters
- **Validation**: 2016Q4  
- **Backtest**: 2017Q1+
- **Data Dictionary**: `data/data_dictionary.xlsx`

## Key Requirements

✅ **Time-ordered splits** (no random train/test)  
✅ **Listing-time features only** (no post-origination data)  
✅ **Budget constraint**: $5,000 per quarter  
✅ **Model calibration** with reliability curves  
✅ **Reproducible pipeline** with fixed seeds  

## Expected Deliverables

- [ ] **Main Analysis**: Jupyter notebook or Python script
- [ ] **Summary Report**: `SUMMARY.md` (≤1 page)
- [ ] **Dependencies**: `requirements.txt`
- [ ] **AI Disclosure**: Documentation of AI assistance

## Project Timeline

**Target**: 4-6 hours total
- EDA & Cleaning: 1-1.5h
- Feature Engineering: 1-1.5h  
- Modeling: 1.5-2h
- Decision Policy & Backtest: 1h
- Documentation: 0.5-1h

## Documentation

| Document | Purpose |
|----------|---------|
| [`REQUIREMENTS.md`](docs/REQUIREMENTS.md) | Complete project specification |
| [`GETTING_STARTED.md`](docs/GETTING_STARTED.md) | Setup and workflow guide |
| [`PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | File organization guide |
| [`SUMMARY_TEMPLATE.md`](docs/SUMMARY_TEMPLATE.md) | Template for final report |
| [`AI_USAGE_TEMPLATE.md`](docs/AI_USAGE_TEMPLATE.md) | AI transparency template |

## Contact

For questions about requirements or project scope, please refer to the documentation in the `docs/` directory.

---

*This project is designed as an intern-level data science challenge focused on clarity and reproducibility over perfect optimization.*
