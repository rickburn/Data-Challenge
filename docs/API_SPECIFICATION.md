# API Specification — Lending Club ML Pipeline

## Overview

This document defines the programmatic interfaces between different components of the Lending Club ML pipeline. These APIs ensure modularity, testability, and maintainability of the codebase.

## 1. Data Processing Module API

### 1.1 DataLoader Class

```python
from typing import List, Dict, Optional, Tuple
import pandas as pd

class DataLoader:
    """Handles loading and basic validation of quarterly loan data."""
    
    def __init__(self, data_directory: str = "data/"):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        data_directory : str
            Path to directory containing quarterly CSV files
        """
        
    def load_quarter(self, quarter: str) -> pd.DataFrame:
        """
        Load single quarter of loan data.
        
        Parameters:
        -----------
        quarter : str
            Quarter identifier (e.g., "2016Q1")
            
        Returns:
        --------
        pd.DataFrame
            Raw loan data for specified quarter
            
        Raises:
        -------
        FileNotFoundError
            If quarter file doesn't exist
        ValueError
            If data format is invalid
        """
        
    def load_quarters(self, quarters: List[str]) -> pd.DataFrame:
        """
        Load and combine multiple quarters of data.
        
        Parameters:
        -----------
        quarters : List[str]
            List of quarter identifiers
            
        Returns:
        --------
        pd.DataFrame
            Combined loan data with quarter column added
        """
        
    def get_available_quarters(self) -> List[str]:
        """
        Get list of available quarters in data directory.
        
        Returns:
        --------
        List[str]
            Sorted list of available quarters
        """
        
    def validate_data_structure(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate basic data structure and quality.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw loan data
            
        Returns:
        --------
        Dict[str, any]
            Validation report with metrics and issues
        """
```

### 1.2 DataCleaner Class

```python
class DataCleaner:
    """Handles data cleaning and preprocessing."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data cleaner with configuration.
        
        Parameters:
        -----------
        config : Dict, optional
            Cleaning configuration parameters
        """
        
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply cleaning pipeline to loan data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw loan data
            
        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            (cleaned_data, cleaning_report)
        """
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to strategy."""
        
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag outliers."""
        
    def standardize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types across columns."""
        
    def validate_listing_time_constraint(self, df: pd.DataFrame) -> bool:
        """
        Validate that dataset contains only listing-time information.
        
        Returns:
        --------
        bool
            True if all columns are listing-time safe
        """
```

## 2. Feature Engineering Module API

### 2.1 FeatureEngineer Class

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for loan data."""
    
    def __init__(self, 
                 include_text_features: bool = False,
                 max_features: int = 50):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        include_text_features : bool
            Whether to include text-derived features
        max_features : int
            Maximum number of features to create
        """
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit feature engineering pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target variable
            
        Returns:
        --------
        self
        """
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame
            Engineered features
        """
        
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        
    def get_feature_provenance(self) -> pd.DataFrame:
        """
        Get feature provenance documentation.
        
        Returns:
        --------
        pd.DataFrame
            Columns: [feature_name, source_columns, transformation, 
                     listing_time_safe, description]
        """
        
    def create_loan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create loan characteristic features."""
        
    def create_borrower_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create borrower attribute features."""
        
    def create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit history features."""
        
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-derived features (optional)."""
```

### 2.2 FeatureValidator Class

```python
class FeatureValidator:
    """Validates features for listing-time compliance."""
    
    PROHIBITED_PATTERNS = [
        r'.*pymnt.*', r'.*rec_.*', r'chargeoff.*', 
        r'settlement.*', r'collection.*', r'recovery.*'
    ]
    
    PROHIBITED_FIELDS = [
        'loan_status', 'last_pymnt_d', 'next_pymnt_d',
        'out_prncp', 'out_prncp_inv', 'total_rec_prncp'
    ]
    
    def validate_features(self, feature_names: List[str]) -> Dict[str, bool]:
        """
        Validate feature set for listing-time compliance.
        
        Parameters:
        -----------
        feature_names : List[str]
            List of feature names to validate
            
        Returns:
        --------
        Dict[str, bool]
            {feature_name: is_valid} mapping
        """
        
    def generate_validation_report(self, 
                                 feature_names: List[str]) -> pd.DataFrame:
        """Generate detailed validation report."""
```

## 3. Modeling Module API

### 3.1 ModelTrainer Class

```python
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

class ModelTrainer:
    """Handles model training with proper temporal validation."""
    
    def __init__(self, 
                 model_type: str = "logistic",
                 calibration_method: str = "platt",
                 random_state: int = 42):
        """
        Initialize model trainer.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('logistic', 'random_forest', 'xgboost')
        calibration_method : str
            Calibration method ('platt' or 'isotonic')
        random_state : int
            Random seed for reproducibility
        """
        
    def temporal_split(self, 
                      df: pd.DataFrame,
                      train_quarters: List[str],
                      val_quarter: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation split.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with quarter information
        train_quarters : List[str]
            Quarters to use for training
        val_quarter : str
            Quarter to use for validation
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_data, validation_data)
        """
        
    def train_model(self, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series) -> BaseEstimator:
        """
        Train and calibrate model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
            
        Returns:
        --------
        BaseEstimator
            Trained and calibrated model
        """
        
    def hyperparameter_search(self, 
                            X: pd.DataFrame, 
                            y: pd.Series) -> Dict:
        """
        Perform hyperparameter optimization.
        
        Returns:
        --------
        Dict
            Best hyperparameters
        """
        
    def cross_validate_temporal(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              quarters: pd.Series) -> Dict:
        """
        Perform time-series cross-validation.
        
        Returns:
        --------
        Dict
            Cross-validation metrics
        """
```

### 3.2 ModelEvaluator Class

```python
import matplotlib.pyplot as plt
from typing import Tuple

class ModelEvaluator:
    """Comprehensive model evaluation and calibration assessment."""
    
    def __init__(self, output_dir: str = "outputs/"):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation outputs
        """
        
    def evaluate_model(self, 
                      model: BaseEstimator,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Parameters:
        -----------
        model : BaseEstimator
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test targets
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        
    def compute_roc_metrics(self, 
                          y_true: pd.Series, 
                          y_pred_proba: np.array) -> Dict[str, float]:
        """Compute ROC-AUC with confidence intervals."""
        
    def compute_calibration_metrics(self, 
                                  y_true: pd.Series, 
                                  y_pred_proba: np.array) -> Dict[str, float]:
        """Compute calibration metrics (Brier score, ECE, etc.)."""
        
    def plot_roc_curve(self, 
                      y_true: pd.Series, 
                      y_pred_proba: np.array) -> plt.Figure:
        """Generate ROC curve plot."""
        
    def plot_calibration_curve(self, 
                             y_true: pd.Series, 
                             y_pred_proba: np.array,
                             n_bins: int = 10) -> plt.Figure:
        """Generate calibration/reliability plot."""
        
    def plot_feature_importance(self, 
                              model: BaseEstimator,
                              feature_names: List[str]) -> plt.Figure:
        """Generate feature importance plot."""
        
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict) -> str:
        """Generate formatted evaluation report."""
```

## 4. Decision Making Module API

### 4.1 InvestmentDecisionEngine Class

```python
class InvestmentDecisionEngine:
    """Handles investment decision logic and budget constraints."""
    
    def __init__(self, 
                 budget_per_quarter: float = 5000.0,
                 selection_strategy: str = "lowest_risk"):
        """
        Initialize decision engine.
        
        Parameters:
        -----------
        budget_per_quarter : float
            Available budget per quarter
        selection_strategy : str
            Selection strategy ('lowest_risk', 'highest_expected_value')
        """
        
    def make_selection(self, 
                      predictions: pd.DataFrame,
                      loan_amounts: pd.Series) -> pd.DataFrame:
        """
        Select loans based on strategy and budget constraints.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Columns: [loan_id, predicted_probability, ...]
        loan_amounts : pd.Series
            Loan amounts indexed by loan_id
            
        Returns:
        --------
        pd.DataFrame
            Selected loans with selection rationale
        """
        
    def apply_budget_constraint(self, 
                              candidate_loans: pd.DataFrame,
                              loan_amounts: pd.Series) -> pd.DataFrame:
        """Apply budget constraints to loan selection."""
        
    def calculate_expected_value(self, 
                               predicted_prob: float,
                               loan_amount: float,
                               interest_rate: float,
                               term_months: int) -> float:
        """Calculate expected value of loan investment."""
        
    def generate_selection_report(self, 
                                selected_loans: pd.DataFrame) -> Dict:
        """Generate selection summary report."""
```

### 4.2 BacktestEngine Class

```python
class BacktestEngine:
    """Handles backtesting of investment decisions."""
    
    def __init__(self, roi_calculation_method: str = "simple"):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        roi_calculation_method : str
            ROI calculation method ('simple', 'detailed')
        """
        
    def run_backtest(self, 
                    selected_loans: pd.DataFrame,
                    actual_outcomes: pd.DataFrame) -> Dict:
        """
        Run backtest on selected loans.
        
        Parameters:
        -----------
        selected_loans : pd.DataFrame
            Selected loans from decision engine
        actual_outcomes : pd.DataFrame
            Actual loan outcomes
            
        Returns:
        --------
        Dict
            Backtest results and metrics
        """
        
    def calculate_roi_proxy(self, 
                          loan_details: pd.DataFrame,
                          outcomes: pd.DataFrame) -> pd.Series:
        """
        Calculate ROI proxy for loans.
        
        Formula:
        - No default: collected = installment * term
        - Default: collected = 0.30 * installment * term  
        - ROI = (collected - principal) / principal
        
        Parameters:
        -----------
        loan_details : pd.DataFrame
            Loan characteristics
        outcomes : pd.DataFrame
            Actual loan outcomes
            
        Returns:
        --------
        pd.Series
            ROI for each loan
        """
        
    def compare_performance(self, 
                          selected_performance: Dict,
                          benchmark_performance: Dict) -> Dict:
        """Compare selected portfolio vs benchmark."""
        
    def generate_backtest_report(self, results: Dict) -> str:
        """Generate formatted backtest report."""
```

## 5. Pipeline Orchestration API

### 5.1 MLPipeline Class

```python
class MLPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: Dict):
        """
        Initialize ML pipeline.
        
        Parameters:
        -----------
        config : Dict
            Pipeline configuration parameters
        """
        
    def run_full_pipeline(self) -> Dict:
        """
        Execute complete ML pipeline.
        
        Returns:
        --------
        Dict
            Pipeline execution results
        """
        
    def run_data_processing(self) -> Tuple[pd.DataFrame, Dict]:
        """Execute data processing stage."""
        
    def run_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute feature engineering stage."""
        
    def run_model_training(self, 
                          features: pd.DataFrame, 
                          targets: pd.Series) -> BaseEstimator:
        """Execute model training stage."""
        
    def run_evaluation(self, 
                      model: BaseEstimator,
                      test_data: pd.DataFrame) -> Dict:
        """Execute model evaluation stage."""
        
    def run_backtesting(self, 
                       model: BaseEstimator,
                       backtest_data: pd.DataFrame) -> Dict:
        """Execute backtesting stage."""
        
    def save_artifacts(self, results: Dict) -> None:
        """Save all pipeline artifacts."""
        
    def generate_summary_report(self, results: Dict) -> str:
        """Generate executive summary report."""
```

## 6. Configuration and Utilities API

### 6.1 Configuration Schema

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PipelineConfig:
    """Pipeline configuration schema."""
    
    # Data configuration
    data_directory: str = "data/"
    train_quarters: List[str] = None
    validation_quarter: str = None
    backtest_quarter: str = None
    
    # Feature engineering
    max_features: int = 50
    include_text_features: bool = False
    missing_value_strategy: str = "median"
    
    # Model configuration
    model_type: str = "logistic"
    calibration_method: str = "platt"
    hyperparameter_search: bool = True
    random_state: int = 42
    
    # Investment configuration
    budget_per_quarter: float = 5000.0
    selection_strategy: str = "lowest_risk"
    roi_calculation_method: str = "simple"
    
    # Output configuration
    output_directory: str = "outputs/"
    save_artifacts: bool = True
    generate_plots: bool = True
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        pass
        
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from file."""
        pass
```

### 6.2 Utility Functions

```python
def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    pass

def ensure_reproducibility(random_state: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    pass

def create_output_directories(base_dir: str) -> None:
    """Create required output directory structure."""
    pass

def save_model_artifacts(model: BaseEstimator, 
                        metadata: Dict,
                        output_path: str) -> None:
    """Save model with metadata."""
    pass

def load_model_artifacts(model_path: str) -> Tuple[BaseEstimator, Dict]:
    """Load model with metadata."""
    pass
```

## 7. Error Handling and Validation

### 7.1 Custom Exceptions

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DataValidationError(PipelineError):
    """Raised when data validation fails."""
    pass

class FeatureValidationError(PipelineError):
    """Raised when feature validation fails."""
    pass

class ModelTrainingError(PipelineError):
    """Raised when model training fails."""
    pass

class BacktestError(PipelineError):
    """Raised when backtesting fails."""
    pass
```

### 7.2 Validation Utilities

```python
def validate_temporal_split(train_data: pd.DataFrame, 
                          val_data: pd.DataFrame) -> bool:
    """Validate temporal ordering in train/validation split."""
    pass

def validate_feature_compliance(features: List[str]) -> Dict[str, str]:
    """Validate features against prohibited list."""
    pass

def validate_model_performance(metrics: Dict[str, float]) -> bool:
    """Validate model meets minimum performance thresholds."""
    pass
```

## 8. Usage Examples

### 8.1 Basic Pipeline Usage

```python
# Initialize pipeline with configuration
config = PipelineConfig(
    train_quarters=["2016Q1", "2016Q2", "2016Q3"],
    validation_quarter="2016Q4",
    backtest_quarter="2017Q1"
)

pipeline = MLPipeline(config)
results = pipeline.run_full_pipeline()
```

### 8.2 Component Usage

```python
# Use individual components
data_loader = DataLoader("data/")
data = data_loader.load_quarters(["2016Q1", "2016Q2"])

feature_engineer = FeatureEngineer()
features = feature_engineer.fit_transform(data)

model_trainer = ModelTrainer()
model = model_trainer.train_model(features, targets)
```

---

*This API specification provides clear interfaces for all pipeline components, enabling modular development, testing, and maintenance of the Lending Club ML pipeline.*
