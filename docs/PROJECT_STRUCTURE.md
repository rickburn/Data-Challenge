# Project Structure Guide

This document outlines the expected file organization for the Lending Club Data Challenge project.

## Current Structure

```
LHAI-Data-Challenge/
├── data/                           # Raw data files
│   ├── 2016Q1.csv                 # Training data quarters
│   ├── 2016Q2.csv
│   ├── 2016Q3.csv
│   ├── 2016Q4.csv                 # Validation quarter
│   ├── 2017Q1.csv                 # Backtest quarter
│   ├── 2017Q2.csv
│   ├── 2017Q3.csv
│   ├── 2017Q4.csv
│   └── data_dictionary.xlsx       # Field definitions
├── docs/                          # Project documentation
│   ├── REQUIREMENTS.md           # Main project requirements
│   ├── PROJECT_STRUCTURE.md      # This file
│   └── SUMMARY_TEMPLATE.md       # Template for final summary
└── [TO BE CREATED]               # Your analysis files
```

## Expected Final Structure

After completing the project, your structure should look like:

```
LHAI-Data-Challenge/
├── data/                          # Raw data files (unchanged)
├── docs/                          # Project documentation  
│   ├── REQUIREMENTS.md
│   ├── PROJECT_STRUCTURE.md
│   ├── SUMMARY_TEMPLATE.md
│   ├── AI_USAGE_TEMPLATE.md
│   └── GETTING_STARTED.md
├── src/                          # Source code package
│   └── lending_club/
│       ├── __init__.py
│       ├── py.typed              # Type hints marker
│       ├── models/               # Data models and types
│       │   ├── __init__.py
│       │   ├── data_models.py    # Pydantic models
│       │   └── enums.py         # Enum definitions
│       ├── features/            # Feature engineering
│       │   ├── __init__.py
│       │   └── feature_engineering.py
│       └── evaluation/          # Model evaluation
│           ├── __init__.py
│           └── model_evaluator.py
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── unit/                    # Unit tests
│   │   ├── __init__.py
│   │   └── test_data_models.py
│   ├── integration/             # Integration tests
│   │   ├── __init__.py
│   │   └── test_end_to_end.py
│   └── data/                    # Test data
│       └── __init__.py
├── notebooks/                    # Analysis notebooks (optional)
│   └── lending_club_analysis.ipynb
├── outputs/                      # Generated artifacts
│   ├── figures/
│   │   ├── eda_plots/
│   │   ├── model_evaluation/
│   │   └── backtest_results/
│   └── models/
│       └── baseline_model.pkl
├── pyproject.toml               # Modern Python packaging config
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── pytest.ini                  # Pytest configuration
├── .pre-commit-config.yaml     # Code quality hooks
├── .gitignore                  # Git ignore rules
├── SUMMARY.md                  # Final project summary
└── README.md                   # Project overview
```

## File Descriptions

### Core Package Files
- **src/lending_club/**: Main package directory with strongly typed components
- **models/data_models.py**: Pydantic models for data validation and type safety
- **models/enums.py**: Enumeration classes for categorical data
- **features/feature_engineering.py**: Listing-time feature extraction and validation
- **evaluation/model_evaluator.py**: Model evaluation and calibration utilities

### Testing Infrastructure
- **tests/unit/**: Unit tests for individual components
- **tests/integration/**: End-to-end workflow tests
- **tests/conftest.py**: Shared test fixtures and configuration
- **pytest.ini**: Pytest configuration and markers

### Configuration Files
- **pyproject.toml**: Modern Python packaging configuration (PEP 518)
- **requirements.txt**: Production dependencies (pinned versions)
- **requirements-dev.txt**: Development dependencies
- **pre-commit-config.yaml**: Code quality automation hooks

### Analysis Files
- **Main Analysis**: Either `notebooks/lending_club_analysis.ipynb` OR modular Python scripts
- **SUMMARY.md**: ≤1 page final report (required deliverable)

### Documentation
- **AI_USAGE_TEMPLATE.md**: Template for documenting AI assistance
- **README.md**: Project overview and setup instructions

### Output Artifacts  
- **figures/**: All plots and visualizations
  - EDA plots and data summaries
  - Model calibration curves
  - ROC curves and feature importance
  - Backtest results visualization
- **models/**: Saved trained models for reproducibility

## Recommended Workflow

1. **Environment Setup Phase**
   - Create and activate virtual environment
   - Install development dependencies: `pip install -r requirements-dev.txt`
   - Install package in development mode: `pip install -e .`
   - Set up pre-commit hooks: `pre-commit install`

2. **Development Phase**
   - Implement strongly typed data models in `src/lending_club/models/`
   - Create feature engineering pipeline in `src/lending_club/features/`
   - Build model evaluation utilities in `src/lending_club/evaluation/`
   - Write unit tests as you develop each component

3. **Analysis Phase**  
   - Use notebook OR script-based approach for main analysis
   - Leverage the package modules for consistent, testable code
   - Implement data cleaning with validation using Pydantic models
   - Build and evaluate baseline model with proper calibration

4. **Testing & Quality Phase**
   - Run unit tests: `pytest tests/unit/`
   - Run integration tests: `pytest tests/integration/`
   - Check code quality: `pre-commit run --all-files`
   - Verify type safety: `mypy src/`

5. **Documentation Phase**
   - Write SUMMARY.md using template
   - Complete AI usage disclosure
   - Update README.md with setup instructions
   - Document any package API changes

6. **Final Quality Assurance**
   - Verify reproducibility with fresh environment
   - Check for data leakage using model validators
   - Validate temporal splits with date constraints
   - Review all guardrails compliance

## Key Principles

- **Single Source of Truth**: One main analysis file that runs end-to-end
- **Reproducibility**: All outputs should be regeneratable
- **Clear Organization**: Logical separation of concerns
- **Documentation**: Every decision and assumption documented
- **Time-Ordered Data**: Maintain temporal integrity throughout
