# Technical Requirements — Lending Club ML Pipeline

## Overview

This document specifies the technical implementation requirements for the Lending Club loan default prediction pipeline. These requirements complement the business requirements outlined in `REQUIREMENTS.md`.

## 1. System Architecture Requirements

### 1.1 Technology Stack
- **Python Version**: 3.8+ (recommended: 3.9 or 3.10)
- **Core Libraries**:
  - Data Processing: `pandas>=1.5.0`, `numpy>=1.21.0`
  - Machine Learning: `scikit-learn>=1.1.0`
  - Visualization: `matplotlib>=3.5.0`, `seaborn>=0.11.0`
  - Jupyter: `jupyter>=1.0.0`, `ipykernel>=6.0.0`
- **Optional Libraries**:
  - Advanced ML: `xgboost>=1.6.0`, `lightgbm>=3.3.0`
  - Statistical: `scipy>=1.8.0`, `statsmodels>=0.13.0`

### 1.2 Environment Requirements
- **Development**: Jupyter Notebook or Python script execution
- **Compute**: Single-machine execution (no distributed computing required)
- **Memory**: Minimum 8GB RAM recommended for data processing
- **Storage**: ~2GB for data files and intermediate outputs

### 1.3 Project Structure
```
glowing-octo-spork/
├── data/                     # Raw quarterly CSV files
│   ├── 2016Q1.csv
│   ├── 2016Q2.csv
│   └── ...
├── docs/                     # Documentation
│   ├── REQUIREMENTS.md
│   ├── TECHNICAL_REQUIREMENTS.md
│   ├── data_dictionary.xlsx
│   └── AI_USAGE_TEMPLATE.md
├── src/                      # Source code
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
├── notebooks/                # Jupyter notebooks
│   └── main_analysis.ipynb
├── outputs/                  # Generated artifacts
│   ├── models/
│   ├── figures/
│   └── results/
├── requirements.txt          # Dependency specification
├── SUMMARY.md               # Executive summary
└── README.md                # Setup instructions
```

## 2. Data Pipeline Requirements

### 2.1 Data Ingestion
- **Input Format**: CSV files with quarterly loan data
- **File Size**: Handle files up to 500MB per quarter
- **Encoding**: UTF-8 with fallback handling for encoding issues
- **Error Handling**: Graceful handling of malformed CSV rows

### 2.2 Data Validation
```python
# Required validation checks
def validate_data_constraints():
    """Implement these validation rules"""
    # Temporal ordering validation
    assert max(train_issue_dates) < min(validation_issue_dates)
    
    # Listing-time feature validation
    prohibited_fields = [
        'loan_status', 'last_pymnt_d', 'last_pymnt_amnt',
        'total_rec_prncp', 'total_rec_int', 'recoveries',
        'collection_recovery_fee', 'out_prncp', 'next_pymnt_d'
    ]
    assert not any(field in features for field in prohibited_fields)
    
    # Data quality checks
    assert missing_rate < 0.5  # No feature >50% missing
    assert target_prevalence > 0.01  # Minimum 1% positive class
```

### 2.3 Data Processing Pipeline
- **Memory Management**: Process data in chunks if memory constraints exist
- **Caching**: Implement intermediate result caching for reproducibility
- **Logging**: Comprehensive logging of data transformations
- **Versioning**: Track data processing steps and transformations

## 3. Feature Engineering Requirements

### 3.1 Feature Categories
```python
# Required feature categories with technical specifications
FEATURE_CATEGORIES = {
    'loan_characteristics': [
        'loan_amnt',           # Loan amount ($)
        'int_rate',            # Interest rate (%)
        'installment',         # Monthly payment ($)
        'term',                # Loan term (months)
        'grade',               # Lending Club grade (A-G)
        'sub_grade'            # Lending Club sub-grade
    ],
    'borrower_attributes': [
        'annual_inc',          # Annual income ($)
        'emp_length',          # Employment length (years)
        'home_ownership',      # Home ownership status
        'verification_status', # Income verification
        'dti',                 # Debt-to-income ratio
        'fico_range_low',      # FICO range lower bound
        'fico_range_high'      # FICO range upper bound
    ],
    'credit_history': [
        'delinq_2yrs',         # Delinquencies in past 2 years
        'inq_last_6mths',      # Credit inquiries last 6 months
        'open_acc',            # Number of open credit accounts
        'pub_rec',             # Number of public records
        'revol_bal',           # Revolving credit balance
        'revol_util',          # Revolving credit utilization
        'total_acc'            # Total credit accounts
    ]
}
```

### 3.2 Feature Engineering Technical Specs
- **Missing Value Strategy**: Document and implement consistent approach
- **Encoding**: 
  - Categorical: One-hot encoding or label encoding (document choice)
  - Ordinal: Preserve ordinal relationships (e.g., grades A-G)
- **Scaling**: StandardScaler or MinMaxScaler for continuous features
- **Feature Selection**: Maximum 50 features to maintain interpretability

### 3.3 Feature Validation
```python
# Feature provenance validation
def validate_feature_provenance():
    """Each feature must pass listing-time test"""
    for feature in feature_set:
        assert can_be_known_at_listing_time(feature), f"{feature} fails listing-time test"
        assert not contains_future_information(feature), f"{feature} contains future info"
        assert data_availability_rate(feature) > 0.80, f"{feature} too sparse"
```

## 4. Machine Learning Requirements

### 4.1 Model Architecture
- **Base Models**: Logistic Regression, Random Forest, or Gradient Boosting
- **Complexity**: Maximum 1000 parameters for interpretability
- **Output**: Calibrated probabilities in [0, 1] range
- **Training Time**: Maximum 30 minutes on standard hardware

### 4.2 Model Training Pipeline
```python
# Required training pipeline structure
class ModelTrainingPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def temporal_split(self, data, train_quarters, validation_quarter):
        """Implement strict temporal splitting"""
        pass
        
    def train_model(self, X_train, y_train):
        """Train with cross-validation and hyperparameter tuning"""
        pass
        
    def calibrate_probabilities(self, model, X_val, y_val):
        """Apply Platt scaling or isotonic regression"""
        pass
        
    def validate_model(self, model, X_val, y_val):
        """Compute all required metrics"""
        pass
```

### 4.3 Model Performance Requirements
- **Minimum ROC-AUC**: 0.65 (better than random + reasonable margin)
- **Calibration**: Brier score < 0.20
- **Reliability**: Hosmer-Lemeshow test p-value > 0.05
- **Stability**: Performance variance < 0.05 across validation quarters

## 5. Evaluation and Metrics Requirements

### 5.1 Required Metrics Implementation
```python
# All metrics must be computed with these exact signatures
def compute_roc_auc(y_true, y_pred_proba):
    """ROC-AUC with 95% confidence interval"""
    pass

def compute_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """Reliability diagram with perfect calibration line"""
    pass

def compute_brier_score(y_true, y_pred_proba):
    """Brier score with decomposition into reliability, resolution, uncertainty"""
    pass

def compute_expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """ECE metric for calibration quality"""
    pass
```

### 5.2 Visualization Requirements
- **ROC Curve**: Include AUC value and confidence interval
- **Calibration Plot**: 45-degree line with binned predictions
- **Feature Importance**: Top 10 features with confidence intervals
- **Confusion Matrix**: For optimal threshold selection
- **Distribution Plots**: Predicted probabilities by true class

### 5.3 Backtesting Technical Specs
```python
# Backtesting implementation requirements
class BacktestEngine:
    def __init__(self, budget_per_quarter=5000):
        self.budget = budget_per_quarter
        
    def apply_selection_rule(self, predictions, loan_amounts):
        """Implement budget-constrained loan selection"""
        pass
        
    def compute_roi_proxy(self, selected_loans, actual_outcomes):
        """
        ROI Calculation:
        - No default: collected = installment * term
        - Default: collected = 0.30 * installment * term
        - ROI = (collected - principal) / principal
        """
        pass
        
    def generate_backtest_report(self, results):
        """Standard backtest reporting format"""
        pass
```

## 6. Performance and Scalability Requirements

### 6.1 Computational Performance
- **Data Loading**: < 30 seconds per quarterly file
- **Feature Engineering**: < 5 minutes for all quarters
- **Model Training**: < 30 minutes including hyperparameter tuning
- **Prediction**: < 10 seconds for 50K samples
- **End-to-End Runtime**: < 60 minutes total

### 6.2 Memory Requirements
- **Peak Memory Usage**: < 16GB RAM
- **Data Processing**: Use chunking for files > 1GB
- **Model Storage**: Pickled models < 100MB each

### 6.3 Reproducibility Requirements
```python
# Reproducibility implementation
def ensure_reproducibility():
    """Set all random seeds for reproducible results"""
    import random, numpy as np
    from sklearn.utils import check_random_state
    
    random.seed(42)
    np.random.seed(42)
    # Set sklearn random_state=42 in all estimators
```

## 7. Security and Compliance Requirements

### 7.1 Data Security
- **No PII Storage**: Ensure no personally identifiable information is cached
- **Local Processing**: All data processing must occur locally
- **Data Retention**: Clear data files after processing completion
- **Access Control**: Standard file system permissions

### 7.2 Model Security
- **Model Serialization**: Use pickle with protocol version 4
- **Version Control**: Track model versions and training data
- **Audit Trail**: Log all model training parameters and results

## 8. Testing and Validation Requirements

### 8.1 Unit Testing Requirements
```python
# Required test categories
def test_data_processing():
    """Test data loading, cleaning, and transformation"""
    pass

def test_feature_engineering():
    """Test feature creation and validation"""
    pass

def test_temporal_splitting():
    """Validate no data leakage in temporal splits"""
    pass

def test_model_training():
    """Test model training and calibration"""
    pass

def test_backtesting():
    """Test investment decision and ROI calculation"""
    pass
```

### 8.2 Integration Testing
- **End-to-End**: Complete pipeline execution test
- **Data Validation**: Automated data quality checks
- **Model Validation**: Performance regression tests
- **Reproducibility**: Multiple run consistency tests

### 8.3 Validation Checklist
```python
# Pre-submission validation checklist
VALIDATION_CHECKLIST = [
    "✅ No prohibited features used",
    "✅ Temporal split validation passes", 
    "✅ Model achieves minimum performance thresholds",
    "✅ Calibration metrics within acceptable range",
    "✅ Backtest results are reasonable",
    "✅ Code runs end-to-end without errors",
    "✅ All required outputs are generated",
    "✅ Random seed reproducibility verified"
]
```

## 9. Documentation Requirements

### 9.1 Code Documentation
- **Docstrings**: All functions must have numpy-style docstrings
- **Type Hints**: Use typing annotations for function signatures
- **Comments**: Explain business logic and model assumptions
- **README**: Clear setup and execution instructions

### 9.2 Technical Documentation
- **Model Card**: Formal model documentation with intended use, limitations
- **Feature Documentation**: Provenance table for all features
- **Performance Report**: Detailed metrics and interpretations
- **Assumption Log**: Document all modeling assumptions

## 10. Deployment and Operational Requirements

### 10.1 Environment Setup
```bash
# Required setup commands
python -m venv lending_club_env
source lending_club_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 10.2 Execution Requirements
- **Single Command**: Pipeline should run with single command/notebook execution
- **Progress Indicators**: Show progress for long-running operations
- **Error Recovery**: Graceful handling of common errors
- **Output Generation**: Automated generation of all required deliverables

### 10.3 Monitoring and Logging
```python
# Logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lending_club_pipeline.log'),
        logging.StreamHandler()
    ]
)
```

## 11. Quality Assurance Requirements

### 11.1 Code Quality
- **PEP 8**: Follow Python style guidelines
- **Complexity**: Maximum cyclomatic complexity of 10 per function
- **Dependencies**: Pin all package versions in requirements.txt
- **Error Handling**: Proper exception handling with informative messages

### 11.2 Model Quality Assurance
- **Cross-Validation**: Use time-series cross-validation where applicable
- **Sensitivity Analysis**: Test model stability with different hyperparameters
- **Bias Testing**: Evaluate model fairness across demographic groups if data available
- **Stress Testing**: Test performance on edge cases and outliers

---

*This technical requirements document provides comprehensive implementation specifications for the Lending Club ML pipeline project, ensuring reproducible, maintainable, and high-quality deliverables.*
