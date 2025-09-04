"""
Comprehensive validation utilities for Lending Club ML Pipeline.

This module provides validation functions for data quality, temporal constraints,
feature compliance, and model performance according to cursor rules standards.
"""

import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from .logging_config import log_execution, track_data_transformation


@dataclass
class ValidationResult:
    """
    Structured validation result with detailed reporting.
    
    Attributes
    ----------
    is_valid : bool
        Overall validation status
    validation_type : str
        Type of validation performed
    errors : List[str]
        List of validation errors
    warnings : List[str]
        List of validation warnings
    metrics : Dict[str, Any]
        Validation metrics and measurements
    timestamp : datetime
        Validation timestamp
    """
    
    is_valid: bool
    validation_type: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'is_valid': self.is_valid,
            'validation_type': self.validation_type,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


class DataQualityValidator:
    """
    Comprehensive data quality validation for ML pipeline.
    
    Validates data according to cursor rules standards including:
    - Missing value thresholds
    - Data type consistency  
    - Outlier detection
    - Duplicate detection
    - Statistical distributions
    """
    
    # Validation thresholds from cursor rules
    MAX_MISSING_RATE = 0.50
    MAX_DUPLICATE_RATE = 0.01
    MIN_QUALITY_SCORE = 0.80
    MAX_OUTLIER_RATE = 0.05
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @log_execution
    def validate_data_quality(self, 
                            df: pd.DataFrame,
                            dataset_name: str = "dataset") -> ValidationResult:
        """
        Comprehensive data quality validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate
        dataset_name : str
            Name of dataset for reporting
            
        Returns
        -------
        ValidationResult
            Detailed validation results
        """
        result = ValidationResult(
            is_valid=True,
            validation_type=f"data_quality_{dataset_name}"
        )
        
        self.logger.info(f"Starting data quality validation for {dataset_name}")
        
        # Basic shape validation
        if df.empty:
            result.add_error("Dataset is empty")
            return result
            
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        
        duplicate_rows = df.duplicated().sum()
        duplicate_rate = duplicate_rows / len(df) if len(df) > 0 else 0
        
        # Update metrics
        result.metrics.update({
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_rate': missing_rate,
            'duplicate_rows': duplicate_rows,
            'duplicate_rate': duplicate_rate,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        })
        
        # Validate missing value rate
        if missing_rate > self.MAX_MISSING_RATE:
            result.add_error(
                f"Missing value rate too high: {missing_rate:.3f} > {self.MAX_MISSING_RATE}"
            )
        elif missing_rate > 0.1:
            result.add_warning(
                f"High missing value rate: {missing_rate:.3f}"
            )
            
        # Validate duplicate rate  
        if duplicate_rate > self.MAX_DUPLICATE_RATE:
            result.add_error(
                f"Duplicate rate too high: {duplicate_rate:.3f} > {self.MAX_DUPLICATE_RATE}"
            )
            
        # Check for columns with excessive missing values
        column_missing_rates = df.isnull().mean()
        bad_columns = column_missing_rates[column_missing_rates > self.MAX_MISSING_RATE].index.tolist()
        
        if bad_columns:
            result.add_error(f"Columns with excessive missing values: {bad_columns}")
            
        # Calculate overall quality score
        quality_score = (1 - missing_rate) * (1 - duplicate_rate)
        result.metrics['quality_score'] = quality_score
        
        if quality_score < self.MIN_QUALITY_SCORE:
            result.add_error(
                f"Overall data quality insufficient: {quality_score:.3f} < {self.MIN_QUALITY_SCORE}"
            )
            
        # Log results
        self.logger.info(
            f"Data quality validation completed for {dataset_name}",
            extra=result.to_dict()
        )
        
        return result
    
    @log_execution
    def detect_outliers(self, 
                       df: pd.DataFrame,
                       numeric_columns: Optional[List[str]] = None,
                       method: str = "iqr") -> ValidationResult:
        """
        Detect outliers in numeric columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze
        numeric_columns : List[str], optional
            Columns to check for outliers
        method : str
            Outlier detection method ('iqr', 'zscore')
            
        Returns
        -------
        ValidationResult
            Outlier detection results
        """
        result = ValidationResult(
            is_valid=True,
            validation_type="outlier_detection"
        )
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        outlier_counts = {}
        total_outliers = 0
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (series < lower_bound) | (series > upper_bound)
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = z_scores > 3
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
                
            outlier_count = outliers.sum()
            outlier_rate = outlier_count / len(series)
            
            outlier_counts[col] = {
                'count': outlier_count,
                'rate': outlier_rate,
                'total_values': len(series)
            }
            
            total_outliers += outlier_count
            
            if outlier_rate > self.MAX_OUTLIER_RATE:
                result.add_warning(
                    f"High outlier rate in {col}: {outlier_rate:.3f} > {self.MAX_OUTLIER_RATE}"
                )
                
        # Calculate overall outlier rate
        total_numeric_values = sum(counts['total_values'] for counts in outlier_counts.values())
        overall_outlier_rate = total_outliers / total_numeric_values if total_numeric_values > 0 else 0
        
        result.metrics.update({
            'column_outlier_counts': outlier_counts,
            'total_outliers': total_outliers,
            'overall_outlier_rate': overall_outlier_rate,
            'method': method
        })
        
        return result


class TemporalConstraintValidator:
    """
    Validate temporal constraints critical for preventing data leakage.
    
    Ensures strict temporal ordering in train/validation/test splits
    according to listing-time only rule.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @log_execution
    def validate_temporal_split(self, 
                              train_data: pd.DataFrame,
                              val_data: pd.DataFrame,
                              date_column: str = 'issue_d',
                              test_data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """
        Validate temporal ordering in data splits.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training dataset
        val_data : pd.DataFrame
            Validation dataset
        date_column : str
            Column containing dates
        test_data : pd.DataFrame, optional
            Test dataset
            
        Returns
        -------
        ValidationResult
            Temporal validation results
        """
        result = ValidationResult(
            is_valid=True,
            validation_type="temporal_split_validation"
        )
        
        self.logger.info("Starting temporal split validation")
        
        # Convert date columns to datetime
        try:
            train_dates = pd.to_datetime(train_data[date_column])
            val_dates = pd.to_datetime(val_data[date_column])
            
            if test_data is not None:
                test_dates = pd.to_datetime(test_data[date_column])
        except Exception as e:
            result.add_error(f"Failed to parse dates: {str(e)}")
            return result
            
        # Calculate date ranges
        train_min = train_dates.min()
        train_max = train_dates.max()
        val_min = val_dates.min()
        val_max = val_dates.max()
        
        result.metrics.update({
            'train_date_min': train_min.isoformat(),
            'train_date_max': train_max.isoformat(),
            'val_date_min': val_min.isoformat(),
            'val_date_max': val_max.isoformat(),
            'train_val_gap_days': (val_min - train_max).days
        })
        
        # Validate temporal ordering: train_max < val_min
        if train_max >= val_min:
            result.add_error(
                f"Temporal constraint violated: train_max ({train_max.date()}) >= "
                f"val_min ({val_min.date()})"
            )
            
        # Check for test data temporal ordering
        if test_data is not None:
            test_min = test_dates.min()
            test_max = test_dates.max()
            
            result.metrics.update({
                'test_date_min': test_min.isoformat(),
                'test_date_max': test_max.isoformat(),
                'val_test_gap_days': (test_min - val_max).days
            })
            
            if val_max >= test_min:
                result.add_error(
                    f"Temporal constraint violated: val_max ({val_max.date()}) >= "
                    f"test_min ({test_min.date()})"
                )
                
        # Log results
        self.logger.info(
            "Temporal validation completed",
            extra=result.to_dict()
        )
        
        return result
    
    @log_execution
    def validate_date_consistency(self, 
                                df: pd.DataFrame,
                                date_column: str = 'issue_d') -> ValidationResult:
        """
        Validate date consistency and format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with dates
        date_column : str
            Date column to validate
            
        Returns
        -------
        ValidationResult
            Date consistency validation results
        """
        result = ValidationResult(
            is_valid=True,
            validation_type="date_consistency_validation"
        )
        
        if date_column not in df.columns:
            result.add_error(f"Date column '{date_column}' not found")
            return result
            
        try:
            dates = pd.to_datetime(df[date_column])
            
            # Check for null dates
            null_dates = dates.isnull().sum()
            null_rate = null_dates / len(df)
            
            if null_rate > 0:
                result.add_warning(f"Found {null_dates} null dates ({null_rate:.3%})")
                
            # Check date range reasonableness (for loan data)
            valid_dates = dates.dropna()
            if len(valid_dates) > 0:
                date_min = valid_dates.min()
                date_max = valid_dates.max()
                
                # Reasonable range for Lending Club data
                if date_min.year < 2007:
                    result.add_warning(f"Dates before Lending Club founding: {date_min.date()}")
                    
                if date_max > pd.Timestamp.now():
                    result.add_error(f"Future dates found: {date_max.date()}")
                    
                result.metrics.update({
                    'date_range_min': date_min.isoformat(),
                    'date_range_max': date_max.isoformat(),
                    'date_span_days': (date_max - date_min).days,
                    'null_dates': null_dates,
                    'null_date_rate': null_rate
                })
                
        except Exception as e:
            result.add_error(f"Date parsing failed: {str(e)}")
            
        return result


class FeatureComplianceValidator:
    """
    Validate feature compliance with listing-time only rule.
    
    Critical for preventing data leakage by ensuring only information
    available at loan listing time is used.
    """
    
    # Prohibited patterns and fields from cursor rules
    PROHIBITED_PATTERNS = [
        r'.*pymnt.*',        # Payment-related fields
        r'.*rec_.*',         # Received amounts  
        r'chargeoff.*',      # Charge-off information
        r'settlement.*',     # Settlement data
        r'collection.*',     # Collection information
        r'recovery.*'        # Recovery amounts
    ]
    
    PROHIBITED_FIELDS = [
        'loan_status',       # Final loan outcome
        'last_pymnt_d',      # Last payment date
        'last_pymnt_amnt',   # Last payment amount
        'next_pymnt_d',      # Next payment date
        'total_rec_prncp',   # Total received principal
        'total_rec_int',     # Total received interest
        'recoveries',        # Recovery amounts
        'collection_recovery_fee',  # Collection fees
        'out_prncp',         # Outstanding principal
        'out_prncp_inv'      # Outstanding principal for investors
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @log_execution
    def validate_feature_compliance(self, 
                                  feature_names: List[str],
                                  dataset_name: str = "dataset") -> ValidationResult:
        """
        Validate features against prohibited list.
        
        Parameters
        ----------
        feature_names : List[str]
            List of feature names to validate
        dataset_name : str
            Dataset name for reporting
            
        Returns
        -------
        ValidationResult
            Feature compliance validation results
        """
        result = ValidationResult(
            is_valid=True,
            validation_type=f"feature_compliance_{dataset_name}"
        )
        
        self.logger.info(f"Validating feature compliance for {len(feature_names)} features")
        
        violations = []
        pattern_violations = {}
        
        # Check prohibited fields
        for feature in feature_names:
            if feature in self.PROHIBITED_FIELDS:
                violations.append(feature)
                
        # Check prohibited patterns
        for pattern in self.PROHIBITED_PATTERNS:
            pattern_matches = [f for f in feature_names if re.match(pattern, f, re.IGNORECASE)]
            if pattern_matches:
                pattern_violations[pattern] = pattern_matches
                violations.extend(pattern_matches)
                
        # Remove duplicates
        violations = list(set(violations))
        
        if violations:
            result.add_error(f"Prohibited features detected: {violations}")
            
        result.metrics.update({
            'total_features': len(feature_names),
            'prohibited_features': violations,
            'pattern_violations': pattern_violations,
            'compliance_rate': (len(feature_names) - len(violations)) / len(feature_names),
            'validated_features': [f for f in feature_names if f not in violations]
        })
        
        # Log results
        self.logger.info(
            f"Feature compliance validation completed",
            extra=result.to_dict()
        )
        
        return result
    
    @log_execution
    def generate_feature_provenance_report(self, 
                                         features: Dict[str, Any],
                                         dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Generate feature provenance documentation.
        
        Parameters
        ----------
        features : Dict[str, Any]
            Feature definitions with metadata
        dataset_name : str
            Dataset name
            
        Returns
        -------
        pd.DataFrame
            Feature provenance report
        """
        provenance_data = []
        
        for feature_name, metadata in features.items():
            # Validate feature compliance
            compliance_result = self.validate_feature_compliance([feature_name], dataset_name)
            
            provenance_entry = {
                'feature_name': feature_name,
                'source_columns': metadata.get('source_columns', []),
                'transformation': metadata.get('transformation', 'direct'),
                'listing_time_safe': compliance_result.is_valid,
                'description': metadata.get('description', ''),
                'data_type': metadata.get('data_type', 'unknown'),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            if not compliance_result.is_valid:
                provenance_entry['compliance_errors'] = compliance_result.errors
                
            provenance_data.append(provenance_entry)
            
        provenance_df = pd.DataFrame(provenance_data)
        
        # Track data lineage
        track_data_transformation(
            operation="feature_provenance_generation",
            input_data=features,
            output_data=provenance_df,
            metadata={'dataset_name': dataset_name}
        )
        
        return provenance_df


class ModelPerformanceValidator:
    """
    Validate model performance against minimum thresholds.
    
    Ensures models meet quality standards defined in cursor rules.
    """
    
    # Performance thresholds from cursor rules
    MIN_ROC_AUC = 0.65
    MAX_BRIER_SCORE = 0.20
    MIN_CALIBRATION_PVALUE = 0.05
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @log_execution
    def validate_model_performance(self, 
                                 metrics: Dict[str, float],
                                 model_name: str = "model") -> ValidationResult:
        """
        Validate model performance metrics.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Performance metrics dictionary
        model_name : str
            Model name for reporting
            
        Returns
        -------
        ValidationResult
            Performance validation results
        """
        result = ValidationResult(
            is_valid=True,
            validation_type=f"model_performance_{model_name}"
        )
        
        self.logger.info(f"Validating performance for {model_name}")
        
        # Validate ROC-AUC
        roc_auc = metrics.get('roc_auc', 0.0)
        if roc_auc < self.MIN_ROC_AUC:
            result.add_error(
                f"ROC-AUC below threshold: {roc_auc:.3f} < {self.MIN_ROC_AUC}"
            )
        elif roc_auc < 0.70:
            result.add_warning(f"ROC-AUC could be improved: {roc_auc:.3f}")
            
        # Validate Brier Score
        brier_score = metrics.get('brier_score', 1.0)
        if brier_score > self.MAX_BRIER_SCORE:
            result.add_error(
                f"Brier score above threshold: {brier_score:.3f} > {self.MAX_BRIER_SCORE}"
            )
            
        # Validate calibration
        calibration_pvalue = metrics.get('calibration_pvalue', 0.0)
        if calibration_pvalue < self.MIN_CALIBRATION_PVALUE:
            result.add_warning(
                f"Poor calibration: p-value {calibration_pvalue:.3f} < {self.MIN_CALIBRATION_PVALUE}"
            )
            
        result.metrics.update(metrics)
        
        # Log results  
        self.logger.info(
            f"Model performance validation completed for {model_name}",
            extra=result.to_dict()
        )
        
        return result


# Convenience functions for easy validation
@log_execution
def validate_pipeline_data(train_data: pd.DataFrame,
                         val_data: pd.DataFrame, 
                         feature_names: List[str],
                         test_data: Optional[pd.DataFrame] = None) -> Dict[str, ValidationResult]:
    """
    Comprehensive pipeline data validation.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    val_data : pd.DataFrame
        Validation data
    feature_names : List[str] 
        Feature names to validate
    test_data : pd.DataFrame, optional
        Test data
        
    Returns
    -------
    Dict[str, ValidationResult]
        Validation results for all checks
    """
    results = {}
    
    # Data quality validation
    dq_validator = DataQualityValidator()
    results['train_quality'] = dq_validator.validate_data_quality(train_data, "train")
    results['val_quality'] = dq_validator.validate_data_quality(val_data, "validation")
    
    if test_data is not None:
        results['test_quality'] = dq_validator.validate_data_quality(test_data, "test")
        
    # Temporal constraint validation
    temporal_validator = TemporalConstraintValidator()
    results['temporal'] = temporal_validator.validate_temporal_split(
        train_data, val_data, test_data=test_data
    )
    
    # Feature compliance validation
    feature_validator = FeatureComplianceValidator()
    results['feature_compliance'] = feature_validator.validate_feature_compliance(feature_names)
    
    return results


@log_execution
def validate_model_pipeline(model_metrics: Dict[str, float],
                          model_name: str = "pipeline_model") -> ValidationResult:
    """
    Validate complete model pipeline performance.
    
    Parameters
    ----------
    model_metrics : Dict[str, float]
        Model performance metrics
    model_name : str
        Model identifier
        
    Returns
    -------
    ValidationResult
        Comprehensive model validation results
    """
    validator = ModelPerformanceValidator()
    return validator.validate_model_performance(model_metrics, model_name)


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample data for testing
    np.random.seed(42)
    
    # Training data (older dates)
    train_dates = pd.date_range('2016-01-01', '2016-09-30', freq='D')
    train_data = pd.DataFrame({
        'issue_d': np.random.choice(train_dates, 1000),
        'loan_amnt': np.random.normal(15000, 5000, 1000),
        'int_rate': np.random.normal(0.12, 0.05, 1000),
        'annual_inc': np.random.normal(70000, 30000, 1000)
    })
    
    # Validation data (newer dates)  
    val_dates = pd.date_range('2016-10-01', '2016-12-31', freq='D')
    val_data = pd.DataFrame({
        'issue_d': np.random.choice(val_dates, 300),
        'loan_amnt': np.random.normal(15000, 5000, 300),
        'int_rate': np.random.normal(0.12, 0.05, 300),
        'annual_inc': np.random.normal(70000, 30000, 300)
    })
    
    # Feature names (mix of valid and invalid)
    feature_names = [
        'loan_amnt', 'int_rate', 'annual_inc',  # Valid
        'loan_status', 'last_pymnt_amnt'        # Invalid
    ]
    
    # Test comprehensive validation
    validation_results = validate_pipeline_data(train_data, val_data, feature_names)
    
    print("=== Validation Results Summary ===")
    for validation_type, result in validation_results.items():
        print(f"\n{validation_type.upper()}:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        if result.errors:
            for error in result.errors:
                print(f"    ERROR: {error}")
        if result.warnings:
            for warning in result.warnings:
                print(f"    WARNING: {warning}")
                
    # Test model performance validation
    model_metrics = {
        'roc_auc': 0.72,
        'brier_score': 0.18,
        'calibration_pvalue': 0.12
    }
    
    model_result = validate_model_pipeline(model_metrics, "test_model")
    print(f"\n=== Model Performance Validation ===")
    print(f"Valid: {model_result.is_valid}")
    print(f"Errors: {model_result.errors}")
    print(f"Warnings: {model_result.warnings}")
    
    print("\nValidation testing completed!")
