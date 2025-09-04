"""
Data Loading and Validation Pipeline
====================================

This module handles loading quarterly CSV files and validating data quality
according to the pipeline configuration.

Key Features:
- Loads quarterly CSV files with proper data types
- Enforces temporal constraints (no future leakage)
- Validates data quality metrics
- Handles missing values and outliers
- Ensures listing-time compliance
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from src.utils.logging_config import log_execution, track_data_transformation


class DataLoader:
    """Loads and preprocesses quarterly loan data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader with configuration."""
        self.config = config
        self.input_directory = Path(config.get('input_directory', 'data'))
        self.date_column = config.get('date_column', 'issue_d')
        self.date_format = config.get('date_format', '%Y-%m-%d')
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def load_quarterly_data(self, quarters: List[str]) -> pd.DataFrame:
        """Load data for specified quarters."""
        all_data = []
        
        for quarter in quarters:
            quarter_file = self.input_directory / f"{quarter}.csv"
            
            if not quarter_file.exists():
                raise FileNotFoundError(f"Quarterly data file not found: {quarter_file}")
            
            self.logger.info(f"Loading data for {quarter} from {quarter_file}")
            
            # Load CSV with appropriate data types
            quarter_data = self._load_csv_with_types(quarter_file)
            
            # Add quarter identifier
            quarter_data['data_quarter'] = quarter
            
            # Validate temporal constraints
            self._validate_temporal_constraints(quarter_data, quarter)
            
            all_data.append(quarter_data)
            
            self.logger.info(f"Loaded {len(quarter_data)} rows for {quarter}")
        
        # Combine all quarters
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by issue date to ensure temporal ordering
        if self.date_column in combined_data.columns:
            combined_data = combined_data.sort_values(self.date_column).reset_index(drop=True)
        
        self.logger.info(f"Combined data loaded: {len(combined_data)} rows across {len(quarters)} quarters")
        
        # Track data transformation
        track_data_transformation(
            operation="load_quarterly_data",
            input_data={"quarters": quarters},
            output_data=combined_data,
            metadata={"quarters_loaded": len(quarters)}
        )
        
        return combined_data
    
    def _load_csv_with_types(self, file_path: Path) -> pd.DataFrame:
        """Load CSV with optimized data types."""
        # Define column types for common Lending Club fields
        # Load everything as object first to handle parsing manually
        dtype_mapping = {col: 'object' for col in [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
            'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
            'verification_status', 'loan_status', 'purpose', 'title',
            'zip_code', 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
            'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
            'total_acc', 'fico_range_low', 'fico_range_high', 'issue_d'
        ]}
        
        try:
            # Read CSV with chunking for large files
            chunks = []
            chunk_size = self.config.get('chunk_size', 10000)
            
            for chunk in pd.read_csv(file_path, dtype=dtype_mapping, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            
            data = pd.concat(chunks, ignore_index=True)
            
            # Post-process specific columns
            data = self._post_process_columns(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            # Fallback to basic loading
            return pd.read_csv(file_path, low_memory=False)
    
    def _post_process_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post-process columns after loading."""
        # Convert numeric columns
        numeric_columns = [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
            'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
            'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'fico_range_low', 'fico_range_high'
        ]

        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Convert term to integer (e.g., " 36 months" -> 36)
        if 'term' in data.columns:
            data['term'] = data['term'].astype(str).str.extract('(\d+)').astype('float').astype('Int64')

        # Convert interest rate percentage to decimal (e.g., "9.75%" -> 0.0975)
        if 'int_rate' in data.columns:
            data['int_rate'] = data['int_rate'].astype(str).str.rstrip('%').astype('float') / 100.0

        # Convert revol_util percentage to decimal
        if 'revol_util' in data.columns:
            data['revol_util'] = data['revol_util'].astype(str).str.rstrip('%').astype('float') / 100.0

        # Parse date columns - handle different date formats
        if self.date_column in data.columns:
            data[self.date_column] = pd.to_datetime(data[self.date_column], format='%b-%y', errors='coerce')

        if 'earliest_cr_line' in data.columns:
            data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'], format='%b-%y', errors='coerce')

        # Convert employment length to numeric
        if 'emp_length' in data.columns:
            data['emp_length_years'] = self._parse_emp_length(data['emp_length'])

        return data
    
    def _parse_emp_length(self, emp_length_series: pd.Series) -> pd.Series:
        """Parse employment length into numeric years."""
        emp_length_numeric = emp_length_series.copy()
        
        # Handle various employment length formats
        emp_length_numeric = emp_length_numeric.str.replace('years?', '', regex=True)
        emp_length_numeric = emp_length_numeric.str.replace('year', '')
        emp_length_numeric = emp_length_numeric.str.replace('<', '')
        emp_length_numeric = emp_length_numeric.str.replace('>', '')
        emp_length_numeric = emp_length_numeric.str.strip()
        
        # Convert specific cases
        emp_length_numeric = emp_length_numeric.replace('< 1', '0.5')
        emp_length_numeric = emp_length_numeric.replace('10+', '10')
        
        # Convert to numeric, coercing errors to NaN
        emp_length_numeric = pd.to_numeric(emp_length_numeric, errors='coerce')
        
        return emp_length_numeric
    
    def _validate_temporal_constraints(self, data: pd.DataFrame, quarter: str) -> None:
        """Validate that data adheres to temporal constraints."""
        if self.date_column not in data.columns:
            self.logger.warning(f"Date column {self.date_column} not found in {quarter} data")
            return
        
        # Parse expected quarter dates
        year = int(quarter[:4])
        quarter_num = int(quarter[-1])
        
        # Define quarter date ranges
        quarter_start_month = (quarter_num - 1) * 3 + 1
        quarter_end_month = quarter_num * 3
        
        expected_start = datetime(year, quarter_start_month, 1)
        if quarter_end_month == 12:
            expected_end = datetime(year + 1, 1, 1)
        else:
            expected_end = datetime(year, quarter_end_month + 1, 1)
        
        # Validate date ranges
        valid_dates = data[self.date_column].between(expected_start, expected_end, inclusive='left')
        invalid_count = (~valid_dates).sum()
        
        if invalid_count > 0:
            self.logger.warning(f"{quarter}: {invalid_count} rows with dates outside expected range")
        
        self.logger.info(f"{quarter}: Temporal validation completed")


class DataValidator:
    """Validates data quality according to configuration thresholds."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data validator with configuration."""
        self.config = config
        self.max_missing_rate = config.get('max_missing_rate', 0.50)
        self.max_duplicate_rate = config.get('max_duplicate_rate', 0.01)
        self.min_quality_score = config.get('min_quality_score', 0.80)
        self.max_outlier_rate = config.get('max_outlier_rate', 0.05)
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation."""
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_value_analysis': self._analyze_missing_values(data),
            'duplicate_analysis': self._analyze_duplicates(data),
            'outlier_analysis': self._analyze_outliers(data),
            'data_type_analysis': self._analyze_data_types(data),
            'overall_quality_score': 0.0,
            'quality_issues': [],
            'passed_validation': False
        }
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['overall_quality_score'] = quality_score
        
        # Determine if validation passes
        quality_report['passed_validation'] = quality_score >= self.min_quality_score
        
        # Log quality summary
        if quality_report['passed_validation']:
            self.logger.info(f"✅ Data quality validation PASSED (score: {quality_score:.3f})")
        else:
            self.logger.warning(f"⚠️  Data quality validation FAILED (score: {quality_score:.3f}, "
                               f"minimum required: {self.min_quality_score:.3f})")
        
        return quality_report
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Identify columns with high missing rates
        high_missing_columns = missing_percentages[
            missing_percentages > (self.max_missing_rate * 100)
        ].to_dict()
        
        analysis = {
            'total_missing_cells': missing_counts.sum(),
            'missing_percentage_overall': (missing_counts.sum() / data.size) * 100,
            'columns_with_missing': len(missing_counts[missing_counts > 0]),
            'high_missing_columns': high_missing_columns,
            'missing_by_column': missing_percentages.to_dict()
        }
        
        if high_missing_columns:
            self.logger.warning(f"Columns with high missing rates (>{self.max_missing_rate*100}%): "
                               f"{list(high_missing_columns.keys())}")
        
        return analysis
    
    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows in the dataset."""
        duplicate_mask = data.duplicated()
        duplicate_count = duplicate_mask.sum()
        duplicate_rate = (duplicate_count / len(data)) * 100
        
        analysis = {
            'duplicate_rows': duplicate_count,
            'duplicate_rate_percent': duplicate_rate,
            'exceeds_threshold': duplicate_rate > (self.max_duplicate_rate * 100)
        }
        
        if analysis['exceeds_threshold']:
            self.logger.warning(f"Duplicate rate ({duplicate_rate:.2f}%) exceeds threshold "
                               f"({self.max_duplicate_rate*100}%)")
        
        return analysis
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numeric columns."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        total_outliers = 0
        
        for col in numeric_columns:
            if data[col].notna().sum() < 10:  # Skip columns with too few values
                continue
                
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_rate = (outliers / len(data)) * 100
            
            outlier_analysis[col] = {
                'outlier_count': outliers,
                'outlier_rate_percent': outlier_rate,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            total_outliers += outliers
        
        overall_outlier_rate = (total_outliers / data.size) * 100
        
        analysis = {
            'total_outliers': total_outliers,
            'overall_outlier_rate_percent': overall_outlier_rate,
            'exceeds_threshold': overall_outlier_rate > (self.max_outlier_rate * 100),
            'by_column': outlier_analysis
        }
        
        if analysis['exceeds_threshold']:
            self.logger.warning(f"Overall outlier rate ({overall_outlier_rate:.2f}%) exceeds threshold "
                               f"({self.max_outlier_rate*100}%)")
        
        return analysis
    
    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and their appropriateness."""
        type_counts = data.dtypes.value_counts().to_dict()
        type_counts = {str(k): v for k, v in type_counts.items()}  # Convert dtype objects to strings
        
        # Identify potential type issues
        object_columns = data.select_dtypes(include=['object']).columns
        potential_numeric = []
        
        for col in object_columns:
            if data[col].str.match(r'^[\d.,%-]+$', na=False).sum() > len(data) * 0.8:
                potential_numeric.append(col)
        
        analysis = {
            'data_type_counts': type_counts,
            'potential_type_issues': potential_numeric,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
        }
        
        if potential_numeric:
            self.logger.info(f"Columns that might benefit from numeric conversion: {potential_numeric}")
        
        return analysis
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        score_components = []
        
        # Missing value score (higher is better)
        missing_rate = quality_report['missing_value_analysis']['missing_percentage_overall'] / 100
        missing_score = max(0, 1 - missing_rate / self.max_missing_rate)
        score_components.append(missing_score * 0.3)  # 30% weight
        
        # Duplicate score (higher is better)
        duplicate_rate = quality_report['duplicate_analysis']['duplicate_rate_percent'] / 100
        duplicate_score = max(0, 1 - duplicate_rate / self.max_duplicate_rate)
        score_components.append(duplicate_score * 0.2)  # 20% weight
        
        # Outlier score (higher is better)
        outlier_rate = quality_report['outlier_analysis']['overall_outlier_rate_percent'] / 100
        outlier_score = max(0, 1 - outlier_rate / self.max_outlier_rate)
        score_components.append(outlier_score * 0.2)  # 20% weight
        
        # Completeness score (percentage of non-missing data)
        completeness_score = 1 - (missing_rate / 2)  # Partial credit for missing data
        score_components.append(completeness_score * 0.3)  # 30% weight
        
        # Calculate weighted average
        overall_score = sum(score_components)
        
        return min(1.0, max(0.0, overall_score))  # Clamp to [0, 1]
    
    @log_execution
    def enforce_listing_time_compliance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that violate listing-time constraints."""
        config = self.config
        
        # Get prohibited patterns and fields from configuration
        prohibited_patterns = config.get('prohibited_patterns', [])
        prohibited_fields = config.get('prohibited_fields', [])
        
        # Find columns that match prohibited patterns
        columns_to_remove = set()
        
        for pattern in prohibited_patterns:
            matching_cols = [col for col in data.columns if pd.Series([col]).str.match(pattern, na=False).iloc[0]]
            columns_to_remove.update(matching_cols)
        
        # Add explicitly prohibited fields
        columns_to_remove.update([col for col in prohibited_fields if col in data.columns])
        
        if columns_to_remove:
            self.logger.info(f"Removing {len(columns_to_remove)} prohibited columns for listing-time compliance")
            self.logger.debug(f"Prohibited columns: {sorted(columns_to_remove)}")
            
            # Remove prohibited columns
            clean_data = data.drop(columns=list(columns_to_remove))
            
            # Track the transformation
            track_data_transformation(
                operation="enforce_listing_time_compliance",
                input_data=data,
                output_data=clean_data,
                metadata={
                    'removed_columns': list(columns_to_remove),
                    'remaining_columns': len(clean_data.columns)
                }
            )
            
            return clean_data
        
        self.logger.info("No prohibited columns found - data already compliant")
        return data
