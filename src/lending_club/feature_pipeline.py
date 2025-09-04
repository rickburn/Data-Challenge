"""
Feature Engineering Pipeline
============================

This module handles feature engineering for the Lending Club loan data,
ensuring strict compliance with listing-time constraints.

Key Features:
- Enforces listing-time compliance (no future leakage)
- Creates derived features from loan and borrower attributes
- Handles missing values with configurable strategies
- Scales features for model training
- Provides feature importance and metadata tracking
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

from src.utils.logging_config import log_execution, track_data_transformation


class FeatureEngineer:
    """Creates features from loan data with listing-time compliance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer with configuration."""
        self.config = config
        self.max_features = config.get('max_features', 50)
        self.include_text_features = config.get('include_text_features', False)
        self.missing_value_strategy = config.get('missing_value_strategy', 'median')
        self.scaling_method = config.get('scaling_method', 'standard')
        
        # Feature categories to include
        self.feature_categories = config.get('feature_categories', {})
        
        # Prohibited features for compliance
        self.prohibited_patterns = config.get('prohibited_patterns', [])
        self.prohibited_fields = config.get('prohibited_fields', [])
        
        # Initialize scalers and imputers
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        
        # Feature metadata
        self.feature_names_ = []
        self.feature_metadata_ = {}
        self.prohibited_features_ = []
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def create_features(self, data: pd.DataFrame, align_features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Create feature matrix and target variable from raw loan data."""
        self.logger.info(f"Starting feature engineering on {len(data)} loans")

        # Enforce listing-time compliance first
        clean_data = self._enforce_listing_time_compliance(data)

        # Create target variable
        target = self._create_target_variable(clean_data)

        # Create base features
        features = self._create_base_features(clean_data)

        # Create derived features
        if self.feature_categories.get('derived_features', True):
            derived_features = self._create_derived_features(clean_data)
            features = pd.concat([features, derived_features], axis=1)

        # Create text features if enabled
        if self.include_text_features:
            text_features = self._create_text_features(clean_data)
            features = pd.concat([features, text_features], axis=1)

        # Handle missing values
        features = self._handle_missing_values(features)

        # Handle feature alignment
        if align_features is None:
            # This is the training set - apply feature selection first, then store feature names
            if len(features.columns) > self.max_features:
                features = self._select_features(features, target)

            # Store the final feature names for alignment
            if not hasattr(self, 'training_feature_names'):
                self.training_feature_names = features.columns.tolist()
                self.logger.info(f"Training set finalized with {len(features.columns)} features: {features.columns.tolist()[:5]}...")

        else:
            # This is validation/backtest - first apply feature selection, then align
            if len(features.columns) > self.max_features:
                features = self._select_features(features, target)

            # Now align to training feature set
            self.logger.info(f"Aligning {len(features.columns)} features to training set of {len(align_features)} features")

            # Add missing columns with zeros
            for col in align_features:
                if col not in features.columns:
                    features[col] = 0.0
                    self.logger.debug(f"Added missing column: {col}")

            # Remove extra columns that shouldn't be there
            extra_cols = [col for col in features.columns if col not in align_features]
            if extra_cols:
                self.logger.warning(f"Removing extra columns: {extra_cols}")
                features = features[align_features]

            self.logger.info(f"Feature alignment complete: {len(features.columns)} features")

        # Scale features
        features = self._scale_features(features)

        # Store feature metadata
        self.feature_names_ = list(features.columns)
        self._update_feature_metadata(features)

        self.logger.info(f"Feature engineering completed: {features.shape[1]} features created")

        # Track transformation
        track_data_transformation(
            operation="create_features",
            input_data=data,
            output_data=features,
            metadata={
                'num_features': len(features.columns),
                'target_distribution': target.value_counts().to_dict(),
                'aligned': align_features is not None
            }
        )

        return features, target
    
    def _enforce_listing_time_compliance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that violate listing-time constraints."""
        columns_to_remove = set()
        
        # Find columns matching prohibited patterns
        for pattern in self.prohibited_patterns:
            matching_cols = [col for col in data.columns 
                           if pd.Series([col]).str.contains(pattern, regex=True, na=False).iloc[0]]
            columns_to_remove.update(matching_cols)
        
        # Add explicitly prohibited fields
        columns_to_remove.update([col for col in self.prohibited_fields if col in data.columns])
        
        # Store prohibited features for reporting
        self.prohibited_features_ = list(columns_to_remove)
        
        if columns_to_remove:
            self.logger.info(f"Removing {len(columns_to_remove)} prohibited features for compliance")
            return data.drop(columns=list(columns_to_remove))
        
        return data
    
    def _create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """Create binary target variable indicating loan default."""
        # Define default statuses based on Lending Club loan_status values
        default_statuses = [
            'Charged Off',
            'Default',
            'Does not meet the credit policy. Status:Charged Off',
            'Late (31-120 days)',
            'Late (16-30 days)'
        ]
        
        if 'loan_status' in data.columns:
            target = data['loan_status'].isin(default_statuses).astype(int)
            self.logger.info(f"Target variable created - Default rate: {target.mean():.3f}")
        else:
            # Fallback: create synthetic target for demonstration
            self.logger.warning("loan_status column not found, creating synthetic target")
            # Create target based on credit risk indicators
            risk_score = 0
            
            if 'sub_grade' in data.columns:
                # Higher sub-grades indicate higher risk
                grade_risk = data['sub_grade'].map({
                    'A1': 0.02, 'A2': 0.03, 'A3': 0.04, 'A4': 0.05, 'A5': 0.06,
                    'B1': 0.08, 'B2': 0.10, 'B3': 0.12, 'B4': 0.14, 'B5': 0.16,
                    'C1': 0.18, 'C2': 0.20, 'C3': 0.22, 'C4': 0.24, 'C5': 0.26,
                    'D1': 0.28, 'D2': 0.30, 'D3': 0.32, 'D4': 0.34, 'D5': 0.36,
                    'E1': 0.38, 'E2': 0.40, 'E3': 0.42, 'E4': 0.44, 'E5': 0.46,
                    'F1': 0.48, 'F2': 0.50, 'F3': 0.52, 'F4': 0.54, 'F5': 0.56,
                    'G1': 0.58, 'G2': 0.60, 'G3': 0.62, 'G4': 0.64, 'G5': 0.66
                }).fillna(0.25)  # Default risk for unknown grades
                risk_score += grade_risk
            
            # Create binary target based on risk threshold
            target = (np.random.random(len(data)) < risk_score).astype(int)
            self.logger.info(f"Synthetic target created - Default rate: {target.mean():.3f}")
        
        return pd.Series(target, index=data.index, name='default')
    
    def _create_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create base features from loan characteristics and borrower attributes."""
        features = pd.DataFrame(index=data.index)
        
        # Loan characteristics features
        if self.feature_categories.get('loan_characteristics', True):
            loan_features = self._create_loan_features(data)
            features = pd.concat([features, loan_features], axis=1)
        
        # Borrower attributes features  
        if self.feature_categories.get('borrower_attributes', True):
            borrower_features = self._create_borrower_features(data)
            features = pd.concat([features, borrower_features], axis=1)
        
        # Credit history features
        if self.feature_categories.get('credit_history', True):
            credit_features = self._create_credit_features(data)
            features = pd.concat([features, credit_features], axis=1)
        
        return features
    
    def _create_loan_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from loan characteristics."""
        loan_features = pd.DataFrame(index=data.index)
        
        # Basic loan amount features
        if 'loan_amnt' in data.columns:
            loan_features['loan_amount'] = data['loan_amnt']
            loan_features['loan_amount_log'] = np.log1p(data['loan_amnt'])
        
        if 'funded_amnt' in data.columns:
            loan_features['funded_amount'] = data['funded_amnt']
            if 'loan_amnt' in data.columns:
                loan_features['funding_ratio'] = data['funded_amnt'] / data['loan_amnt'].clip(lower=1)
        
        # Interest rate features
        if 'int_rate' in data.columns:
            loan_features['interest_rate'] = data['int_rate']
            # Interest rate buckets
            loan_features['int_rate_low'] = (data['int_rate'] < 0.10).astype(int)
            loan_features['int_rate_medium'] = ((data['int_rate'] >= 0.10) & (data['int_rate'] < 0.15)).astype(int)
            loan_features['int_rate_high'] = (data['int_rate'] >= 0.15).astype(int)
        
        # Term features
        if 'term' in data.columns:
            loan_features['term'] = data['term']
            loan_features['term_36'] = (data['term'] == 36).astype(int)
            loan_features['term_60'] = (data['term'] == 60).astype(int)
        
        # Installment features
        if 'installment' in data.columns:
            loan_features['installment'] = data['installment']
            if 'annual_inc' in data.columns:
                loan_features['installment_to_income'] = data['installment'] * 12 / data['annual_inc'].clip(lower=1)
        
        # Grade and sub-grade features
        if 'grade' in data.columns:
            grade_dummies = pd.get_dummies(data['grade'], prefix='grade')
            loan_features = pd.concat([loan_features, grade_dummies], axis=1)
        
        if 'sub_grade' in data.columns:
            # Convert sub-grade to numeric risk score
            sub_grade_mapping = {
                'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
                'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
                'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
                'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
                'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
                'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30,
                'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35
            }
            loan_features['sub_grade_numeric'] = data['sub_grade'].map(sub_grade_mapping)
        
        # Purpose features
        if 'purpose' in data.columns:
            purpose_dummies = pd.get_dummies(data['purpose'], prefix='purpose')
            loan_features = pd.concat([loan_features, purpose_dummies], axis=1)
        
        return loan_features
    
    def _create_borrower_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from borrower attributes."""
        borrower_features = pd.DataFrame(index=data.index)
        
        # Annual income features
        if 'annual_inc' in data.columns:
            borrower_features['annual_income'] = data['annual_inc']
            borrower_features['annual_income_log'] = np.log1p(data['annual_inc'])
            
            # Income buckets
            borrower_features['income_low'] = (data['annual_inc'] < 40000).astype(int)
            borrower_features['income_medium'] = ((data['annual_inc'] >= 40000) & 
                                                 (data['annual_inc'] < 80000)).astype(int)
            borrower_features['income_high'] = (data['annual_inc'] >= 80000).astype(int)
        
        # Employment length features
        if 'emp_length_years' in data.columns:
            borrower_features['emp_length_years'] = data['emp_length_years']
            borrower_features['emp_length_missing'] = data['emp_length_years'].isna().astype(int)
            
            # Employment stability indicators
            borrower_features['emp_stable'] = (data['emp_length_years'] >= 5).astype(int)
            borrower_features['emp_new'] = (data['emp_length_years'] < 1).astype(int)
        
        # Home ownership features
        if 'home_ownership' in data.columns:
            home_dummies = pd.get_dummies(data['home_ownership'], prefix='home')
            borrower_features = pd.concat([borrower_features, home_dummies], axis=1)
        
        # Verification status
        if 'verification_status' in data.columns:
            verification_dummies = pd.get_dummies(data['verification_status'], prefix='verified')
            borrower_features = pd.concat([borrower_features, verification_dummies], axis=1)
        
        # Debt-to-income ratio
        if 'dti' in data.columns:
            borrower_features['dti'] = data['dti']
            borrower_features['dti_high'] = (data['dti'] > 20).astype(int)
            borrower_features['dti_very_high'] = (data['dti'] > 30).astype(int)
        
        # State features (top states only to avoid too many categories)
        if 'addr_state' in data.columns:
            top_states = data['addr_state'].value_counts().head(10).index
            for state in top_states:
                borrower_features[f'state_{state}'] = (data['addr_state'] == state).astype(int)
        
        return borrower_features
    
    def _create_credit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from credit history."""
        credit_features = pd.DataFrame(index=data.index)
        
        # Credit length features
        if 'earliest_cr_line' in data.columns and 'issue_d' in data.columns:
            # Calculate credit history length
            credit_history_months = (data['issue_d'] - data['earliest_cr_line']).dt.days / 30.44
            credit_features['credit_history_months'] = credit_history_months
            credit_features['credit_history_years'] = credit_history_months / 12
            
            # Credit history maturity indicators
            credit_features['credit_history_short'] = (credit_history_months < 36).astype(int)
            credit_features['credit_history_long'] = (credit_history_months > 120).astype(int)
        
        # Delinquency features
        if 'delinq_2yrs' in data.columns:
            credit_features['delinq_2yrs'] = data['delinq_2yrs']
            credit_features['has_delinq'] = (data['delinq_2yrs'] > 0).astype(int)
            credit_features['delinq_multiple'] = (data['delinq_2yrs'] > 1).astype(int)
        
        # Inquiries features
        if 'inq_last_6mths' in data.columns:
            credit_features['inq_last_6mths'] = data['inq_last_6mths']
            credit_features['has_recent_inq'] = (data['inq_last_6mths'] > 0).astype(int)
            credit_features['inq_high'] = (data['inq_last_6mths'] > 2).astype(int)
        
        # Account features
        if 'open_acc' in data.columns:
            credit_features['open_acc'] = data['open_acc']
        
        if 'total_acc' in data.columns:
            credit_features['total_acc'] = data['total_acc']
            
            if 'open_acc' in data.columns:
                credit_features['acc_utilization'] = (data['open_acc'] / 
                                                    data['total_acc'].clip(lower=1))
        
        # Public records
        if 'pub_rec' in data.columns:
            credit_features['pub_rec'] = data['pub_rec']
            credit_features['has_pub_rec'] = (data['pub_rec'] > 0).astype(int)
        
        # Revolving credit features
        if 'revol_bal' in data.columns:
            credit_features['revol_bal'] = data['revol_bal']
            credit_features['revol_bal_log'] = np.log1p(data['revol_bal'])
        
        if 'revol_util' in data.columns:
            credit_features['revol_util'] = data['revol_util']
            credit_features['revol_util_high'] = (data['revol_util'] > 75).astype(int)
            credit_features['revol_util_maxed'] = (data['revol_util'] > 90).astype(int)
        
        return credit_features
    
    def _create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from combinations of base features."""
        derived_features = pd.DataFrame(index=data.index)
        
        # Loan-to-income ratios
        if 'loan_amnt' in data.columns and 'annual_inc' in data.columns:
            derived_features['loan_to_income'] = data['loan_amnt'] / data['annual_inc'].clip(lower=1)
            
            # Risk buckets based on loan-to-income
            derived_features['loan_to_income_low'] = (derived_features['loan_to_income'] < 0.2).astype(int)
            derived_features['loan_to_income_high'] = (derived_features['loan_to_income'] > 0.5).astype(int)
        
        # Credit utilization efficiency
        if 'revol_bal' in data.columns and 'annual_inc' in data.columns:
            derived_features['revol_bal_to_income'] = data['revol_bal'] / data['annual_inc'].clip(lower=1)
        
        # Total debt burden (approximation)
        if all(col in data.columns for col in ['installment', 'revol_bal', 'annual_inc']):
            monthly_revol_payment = data['revol_bal'] * 0.03  # Assume 3% minimum payment
            total_monthly_debt = data['installment'] + monthly_revol_payment
            derived_features['total_debt_to_income'] = (total_monthly_debt * 12) / data['annual_inc'].clip(lower=1)
        
        # Credit profile maturity score
        components = []
        if 'earliest_cr_line' in data.columns and 'issue_d' in data.columns:
            credit_age = (data['issue_d'] - data['earliest_cr_line']).dt.days / 365.25
            components.append(np.clip(credit_age / 10, 0, 1))  # Normalize to 10 years
        
        if 'total_acc' in data.columns:
            components.append(np.clip(data['total_acc'] / 50, 0, 1))  # Normalize to 50 accounts
        
        if 'delinq_2yrs' in data.columns:
            components.append(1 - np.clip(data['delinq_2yrs'] / 5, 0, 1))  # Inverse of delinquencies
        
        if components:
            derived_features['credit_maturity_score'] = np.mean(components, axis=0)
        
        # Risk concentration features
        if 'sub_grade' in data.columns and 'revol_util' in data.columns:
            # Combine credit grade risk with utilization risk
            grade_risk_map = {'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4, 'E': 0.5, 'F': 0.6, 'G': 0.7}
            grade_risk = data['sub_grade'].str[0].map(grade_risk_map).fillna(0.35)
            util_risk = np.clip(data['revol_util'] / 100, 0, 1)
            
            derived_features['combined_risk_score'] = (grade_risk + util_risk) / 2
        
        return derived_features
    
    def _create_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from text fields (if enabled)."""
        text_features = pd.DataFrame(index=data.index)
        
        # Job title features
        if 'emp_title' in data.columns:
            emp_title = data['emp_title'].fillna('').str.lower()
            
            # Common job categories
            text_features['job_manager'] = emp_title.str.contains('manager|mgr').astype(int)
            text_features['job_teacher'] = emp_title.str.contains('teacher|education').astype(int)
            text_features['job_nurse'] = emp_title.str.contains('nurse|rn').astype(int)
            text_features['job_driver'] = emp_title.str.contains('driver|truck').astype(int)
            text_features['job_engineer'] = emp_title.str.contains('engineer|tech').astype(int)
            text_features['job_sales'] = emp_title.str.contains('sales|retail').astype(int)
            
            # Job title length (proxy for job complexity)
            text_features['emp_title_length'] = emp_title.str.len()
        
        # Loan title features  
        if 'title' in data.columns:
            title = data['title'].fillna('').str.lower()
            
            # Purpose-related keywords
            text_features['title_debt'] = title.str.contains('debt|consolidat').astype(int)
            text_features['title_credit'] = title.str.contains('credit card').astype(int)
            text_features['title_home'] = title.str.contains('home|house').astype(int)
            text_features['title_car'] = title.str.contains('car|auto').astype(int)
            text_features['title_business'] = title.str.contains('business|startup').astype(int)
            
            # Title complexity
            text_features['title_word_count'] = title.str.split().str.len()
        
        return text_features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration strategy."""
        if features.isnull().sum().sum() == 0:
            self.logger.info("No missing values found in features")
            return features
        
        missing_counts = features.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        self.logger.info(f"Handling missing values in {len(columns_with_missing)} columns "
                        f"using strategy: {self.missing_value_strategy}")
        
        if self.missing_value_strategy == 'drop':
            # Drop rows with any missing values
            features_clean = features.dropna()
            self.logger.info(f"Dropped {len(features) - len(features_clean)} rows with missing values")
            
        else:
            # Use imputation
            strategy_mapping = {
                'median': 'median',
                'mean': 'mean', 
                'mode': 'most_frequent'
            }
            
            sklearn_strategy = strategy_mapping.get(self.missing_value_strategy, 'median')
            
            # Separate numeric and categorical columns
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            categorical_columns = features.select_dtypes(exclude=[np.number]).columns

            features_clean = features.copy()

            # Convert boolean columns to numeric (sklearn doesn't handle boolean well)
            bool_columns = features_clean.select_dtypes(include=['bool']).columns
            for col in bool_columns:
                features_clean[col] = features_clean[col].astype('int64')
                if col in categorical_columns:
                    categorical_columns = categorical_columns.drop(col)
                    numeric_columns = numeric_columns.append(pd.Index([col]))

            # Impute numeric columns
            if len(numeric_columns) > 0:
                numeric_imputer = SimpleImputer(strategy=sklearn_strategy)
                features_clean[numeric_columns] = numeric_imputer.fit_transform(features_clean[numeric_columns])

            # Impute categorical columns with mode
            if len(categorical_columns) > 0:
                # Ensure categorical columns are strings
                for col in categorical_columns:
                    features_clean[col] = features_clean[col].astype(str)

                categorical_imputer = SimpleImputer(strategy='most_frequent')
                features_clean[categorical_columns] = categorical_imputer.fit_transform(features_clean[categorical_columns])
            
            self.logger.info(f"Imputed missing values using {self.missing_value_strategy} strategy")
        
        # Track transformation
        track_data_transformation(
            operation="handle_missing_values",
            input_data=features,
            output_data=features_clean,
            metadata={
                'strategy': self.missing_value_strategy,
                'columns_imputed': len(columns_with_missing)
            }
        )
        
        return features_clean
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select the best features using statistical tests."""
        self.logger.info(f"Selecting top {self.max_features} features from {len(features.columns)}")
        
        # Use SelectKBest with f_classif for feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=self.max_features)
        
        # Ensure target is aligned with features
        aligned_target = target.loc[features.index]
        
        # Fit and transform
        selected_features = self.feature_selector.fit_transform(features, aligned_target)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_feature_names = features.columns[selected_mask]
        
        # Create DataFrame with selected features
        features_selected = pd.DataFrame(
            selected_features, 
            columns=selected_feature_names,
            index=features.index
        )
        
        # Log feature selection results
        feature_scores = self.feature_selector.scores_
        top_features = sorted(zip(selected_feature_names, feature_scores[selected_mask]), 
                            key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Top 5 selected features: {[f[0] for f in top_features[:5]]}")
        
        return features_selected
    
    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale features according to configuration method."""
        if self.scaling_method == 'none':
            self.logger.info("No feature scaling applied")
            return features
        
        self.logger.info(f"Scaling features using {self.scaling_method} method")
        
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {self.scaling_method}, using standard")
            self.scaler = StandardScaler()
        
        # Fit and transform
        scaled_features = self.scaler.fit_transform(features)
        
        # Create DataFrame with scaled features
        features_scaled = pd.DataFrame(
            scaled_features,
            columns=features.columns,
            index=features.index
        )
        
        return features_scaled
    
    def _update_feature_metadata(self, features: pd.DataFrame) -> None:
        """Update feature metadata for reporting."""
        self.feature_metadata_ = {
            'total_features': len(features.columns),
            'feature_names': list(features.columns),
            'missing_value_strategy': self.missing_value_strategy,
            'scaling_method': self.scaling_method,
            'feature_selection_applied': self.feature_selector is not None,
            'selected_from_total': len(features.columns) if self.feature_selector else None
        }
    
    def get_feature_names(self) -> List[str]:
        """Get the names of engineered features."""
        return self.feature_names_
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get feature engineering metadata."""
        return self.feature_metadata_
    
    def get_prohibited_features(self) -> List[str]:
        """Get list of features that were prohibited and removed."""
        return self.prohibited_features_
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessing steps."""
        if not hasattr(self, 'scaler') or self.scaler is None:
            raise ValueError("FeatureEngineer must be fitted before transforming new data")
        
        # Apply the same feature engineering steps
        clean_data = self._enforce_listing_time_compliance(data)
        features = self._create_base_features(clean_data)
        
        if self.feature_categories.get('derived_features', True):
            derived_features = self._create_derived_features(clean_data)
            features = pd.concat([features, derived_features], axis=1)
        
        if self.include_text_features:
            text_features = self._create_text_features(clean_data)
            features = pd.concat([features, text_features], axis=1)
        
        # Handle missing values (using fitted imputer if available)
        features = self._handle_missing_values(features)
        
        # Select features (using fitted selector if available)
        if self.feature_selector is not None:
            # Ensure we have the same features as during training
            missing_features = set(self.feature_names_) - set(features.columns)
            if missing_features:
                self.logger.warning(f"Missing features in new data: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    features[feat] = 0
            
            # Reorder columns to match training
            features = features[self.feature_names_]
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        return features_scaled
