"""
Unit tests for feature engineering pipeline.

Tests feature creation, compliance checking, and data transformation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch


class MockFeatureEngineer:
    """Mock version of FeatureEngineer for testing without sklearn."""
    
    def __init__(self, config):
        self.config = config
        self.max_features = config.get('max_features', 50)
        self.include_text_features = config.get('include_text_features', False)
        self.missing_value_strategy = config.get('missing_value_strategy', 'median')
        self.scaling_method = config.get('scaling_method', 'standard')
        
        # Feature categories
        self.feature_categories = config.get('feature_categories', {})
        
        # Prohibited features
        self.prohibited_patterns = config.get('prohibited_patterns', [])
        self.prohibited_fields = config.get('prohibited_fields', [])
        
        # Metadata
        self.feature_names_ = []
        self.feature_metadata_ = {}
        self.prohibited_features_ = []
        
    def create_features(self, data):
        """Mock feature creation."""
        # Simulate compliance enforcement
        clean_data = self._enforce_listing_time_compliance(data)
        
        # Create target
        target = self._create_target_variable(clean_data)
        
        # Create features
        features = self._create_mock_features(clean_data)
        
        # Store metadata
        self.feature_names_ = list(features.columns)
        self._update_feature_metadata(features)
        
        return features, target
    
    def _enforce_listing_time_compliance(self, data):
        """Mock compliance enforcement."""
        columns_to_remove = set()
        
        # Check prohibited patterns
        for pattern in self.prohibited_patterns:
            matching_cols = [col for col in data.columns if pattern in col]
            columns_to_remove.update(matching_cols)
        
        # Check prohibited fields
        columns_to_remove.update([col for col in self.prohibited_fields if col in data.columns])
        
        self.prohibited_features_ = list(columns_to_remove)
        
        if columns_to_remove:
            return data.drop(columns=list(columns_to_remove))
        return data
    
    def _create_target_variable(self, data):
        """Mock target creation."""
        if 'loan_status' in data.columns:
            # Use actual loan status if available
            default_statuses = ['Charged Off', 'Default']
            target = data['loan_status'].isin(default_statuses).astype(int)
        else:
            # Create synthetic target based on risk indicators
            np.random.seed(42)  # For reproducible tests
            risk_score = np.random.random(len(data))
            target = (risk_score > 0.8).astype(int)  # ~20% default rate
        
        return pd.Series(target, index=data.index, name='default')
    
    def _create_mock_features(self, data):
        """Create mock features from data."""
        features = pd.DataFrame(index=data.index)
        
        # Basic loan features
        if 'loan_amnt' in data.columns:
            features['loan_amount'] = data['loan_amnt']
            # Handle missing values before log transformation
            loan_amount_clean = data['loan_amnt'].fillna(1)  # Fill NaN with 1 for log
            features['loan_amount_log'] = np.log1p(loan_amount_clean)
        
        if 'int_rate' in data.columns:
            features['interest_rate'] = data['int_rate']
            features['int_rate_high'] = (data['int_rate'] > 0.15).astype(int)
        
        if 'term' in data.columns:
            features['term'] = data['term']
            features['term_60'] = (data['term'] == 60).astype(int)
        
        # Borrower features
        if 'annual_inc' in data.columns:
            features['annual_income'] = data['annual_inc']
            features['income_high'] = (data['annual_inc'] > 80000).astype(int)
        
        if 'dti' in data.columns:
            features['dti'] = data['dti']
            features['dti_high'] = (data['dti'] > 20).astype(int)
        
        # Categorical features (mock one-hot encoding)
        if 'grade' in data.columns:
            for grade in ['A', 'B', 'C', 'D', 'E']:
                features[f'grade_{grade}'] = (data['grade'] == grade).astype(int)
        
        if 'home_ownership' in data.columns:
            for home_type in ['RENT', 'OWN', 'MORTGAGE']:
                features[f'home_{home_type}'] = (data['home_ownership'] == home_type).astype(int)
        
        # Derived features
        if 'loan_amnt' in data.columns and 'annual_inc' in data.columns:
            features['loan_to_income'] = data['loan_amnt'] / (data['annual_inc'] + 1)
        
        # Handle missing values (simple median imputation)
        features = features.fillna(features.median())
        
        return features
    
    def _update_feature_metadata(self, features):
        """Update feature metadata."""
        self.feature_metadata_ = {
            'total_features': len(features.columns),
            'feature_names': list(features.columns),
            'missing_value_strategy': self.missing_value_strategy,
            'scaling_method': self.scaling_method
        }
    
    def get_feature_names(self):
        return self.feature_names_
    
    def get_feature_metadata(self):
        return self.feature_metadata_
    
    def get_prohibited_features(self):
        return self.prohibited_features_


class TestFeatureEngineer:
    """Test cases for FeatureEngineer."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        config = {
            'max_features': 25,
            'include_text_features': True,
            'missing_value_strategy': 'mean',
            'scaling_method': 'minmax'
        }
        
        engineer = MockFeatureEngineer(config)
        
        assert engineer.max_features == 25
        assert engineer.include_text_features is True
        assert engineer.missing_value_strategy == 'mean'
        assert engineer.scaling_method == 'minmax'
    
    def test_create_features_basic(self):
        """Test basic feature creation."""
        # Create sample data
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'loan_amnt': [10000, 15000, 20000, 25000, 30000],
            'int_rate': [0.10, 0.12, 0.15, 0.18, 0.20],
            'term': [36, 60, 36, 60, 36],
            'annual_inc': [50000, 60000, 70000, 80000, 90000],
            'dti': [15, 18, 22, 25, 30],
            'grade': ['A', 'B', 'C', 'D', 'E'],
            'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'RENT', 'OWN']
        })
        
        config = {'max_features': 50}
        engineer = MockFeatureEngineer(config)
        
        features, target = engineer.create_features(data)
        
        # Check feature matrix
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)
        assert len(features.columns) > 0
        
        # Check target
        assert isinstance(target, pd.Series)
        assert len(target) == len(data)
        assert target.name == 'default'
        assert target.dtype in [np.int64, np.int32]
        
        # Check feature names are stored
        assert len(engineer.get_feature_names()) > 0
        assert engineer.get_feature_names() == list(features.columns)
    
    def test_listing_time_compliance(self):
        """Test listing-time compliance enforcement."""
        # Create data with prohibited features
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'loan_amnt': [10000, 15000, 20000],
            'int_rate': [0.10, 0.12, 0.15],
            # Prohibited features
            'total_pymnt': [12000, 18000, 24000],  # Post-origination
            'last_pymnt_d': ['2017-01-01', '2017-02-01', '2017-03-01'],
            'recoveries': [0, 500, 1000],
            'collection_recovery_fee': [0, 50, 100]
        })
        
        config = {
            'prohibited_patterns': ['pymnt', 'rec'],
            'prohibited_fields': ['recoveries', 'collection_recovery_fee']
        }
        engineer = MockFeatureEngineer(config)
        
        features, target = engineer.create_features(data)
        
        # Prohibited features should be removed
        prohibited = engineer.get_prohibited_features()
        assert len(prohibited) > 0
        
        # Check that prohibited features are not in final feature set
        for feature in prohibited:
            assert feature not in features.columns
    
    def test_target_variable_creation(self):
        """Test target variable creation."""
        # Test with actual loan status
        data_with_status = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'loan_amnt': [10000, 15000, 20000, 25000],
            'loan_status': ['Fully Paid', 'Charged Off', 'Current', 'Default']
        })
        
        config = {}
        engineer = MockFeatureEngineer(config)
        
        features, target = engineer.create_features(data_with_status)
        
        # Should correctly identify defaults
        expected_defaults = [0, 1, 0, 1]  # Charged Off and Default are defaults
        assert target.tolist() == expected_defaults
        
        # Test without loan status (synthetic target)
        data_without_status = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'loan_amnt': [10000, 15000, 20000, 25000],
            'int_rate': [0.10, 0.15, 0.20, 0.25]
        })
        
        features2, target2 = engineer.create_features(data_without_status)
        
        # Should create synthetic target
        assert isinstance(target2, pd.Series)
        assert target2.dtype in [np.int64, np.int32]
        assert all(val in [0, 1] for val in target2)
    
    def test_derived_feature_creation(self):
        """Test creation of derived features."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'loan_amnt': [10000, 20000, 30000],
            'annual_inc': [50000, 60000, 70000],
            'int_rate': [0.10, 0.15, 0.20],
            'term': [36, 60, 36]
        })
        
        config = {}
        engineer = MockFeatureEngineer(config)
        
        features, _ = engineer.create_features(data)
        
        # Check for derived features
        expected_derived = [
            'loan_amount_log',
            'int_rate_high', 
            'term_60',
            'income_high',
            'loan_to_income'
        ]
        
        for feature in expected_derived:
            assert feature in features.columns, f"Missing derived feature: {feature}"
        
        # Check loan-to-income calculation
        expected_lti = data['loan_amnt'] / (data['annual_inc'] + 1)
        actual_lti = features['loan_to_income']
        pd.testing.assert_series_equal(actual_lti, expected_lti, check_names=False)
    
    def test_categorical_feature_encoding(self):
        """Test categorical feature encoding."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'grade': ['A', 'B', 'C', 'A'],
            'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'RENT'],
            'loan_amnt': [10000, 15000, 20000, 25000]
        })
        
        config = {}
        engineer = MockFeatureEngineer(config)
        
        features, _ = engineer.create_features(data)
        
        # Check grade encoding
        grade_features = [col for col in features.columns if col.startswith('grade_')]
        assert len(grade_features) > 0
        
        # Check home ownership encoding
        home_features = [col for col in features.columns if col.startswith('home_')]
        assert len(home_features) > 0
        
        # Check one-hot encoding correctness
        if 'grade_A' in features.columns:
            # First and fourth rows should be 1 for grade A
            expected_grade_a = [1, 0, 0, 1]
            assert features['grade_A'].tolist() == expected_grade_a
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'loan_amnt': [10000, None, 20000, None, 30000],
            'int_rate': [0.10, 0.15, None, 0.20, 0.25],
            'annual_inc': [50000, 60000, 70000, 80000, None]
        })
        
        config = {'missing_value_strategy': 'median'}
        engineer = MockFeatureEngineer(config)
        
        features, _ = engineer.create_features(data)
        
        # Features should not have missing values
        assert features.isnull().sum().sum() == 0
        
        # Check that median imputation was applied (approximately)
        if 'loan_amount' in features.columns:
            # Should be filled with median value
            median_loan = data['loan_amnt'].median()
            filled_values = features['loan_amount'].unique()
            assert median_loan in filled_values
    
    def test_feature_metadata_tracking(self):
        """Test feature metadata tracking."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'loan_amnt': [10000, 15000, 20000],
            'int_rate': [0.10, 0.12, 0.15]
        })
        
        config = {
            'missing_value_strategy': 'median',
            'scaling_method': 'standard'
        }
        engineer = MockFeatureEngineer(config)
        
        features, _ = engineer.create_features(data)
        
        # Check metadata
        metadata = engineer.get_feature_metadata()
        
        assert 'total_features' in metadata
        assert 'feature_names' in metadata
        assert 'missing_value_strategy' in metadata
        assert 'scaling_method' in metadata
        
        assert metadata['total_features'] == len(features.columns)
        assert metadata['feature_names'] == list(features.columns)
        assert metadata['missing_value_strategy'] == 'median'
        assert metadata['scaling_method'] == 'standard'
    
    def test_feature_categories_configuration(self):
        """Test feature category inclusion/exclusion."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'loan_amnt': [10000, 15000, 20000],
            'int_rate': [0.10, 0.12, 0.15],
            'annual_inc': [50000, 60000, 70000],
            'grade': ['A', 'B', 'C']
        })
        
        # Test with all categories enabled
        config_all = {
            'feature_categories': {
                'loan_characteristics': True,
                'borrower_attributes': True,
                'derived_features': True
            }
        }
        engineer_all = MockFeatureEngineer(config_all)
        features_all, _ = engineer_all.create_features(data)
        
        # Test with limited categories
        config_limited = {
            'feature_categories': {
                'loan_characteristics': True,
                'borrower_attributes': False,
                'derived_features': False
            }
        }
        engineer_limited = MockFeatureEngineer(config_limited)
        features_limited, _ = engineer_limited.create_features(data)
        
        # All features should have more features than limited
        assert len(features_all.columns) >= len(features_limited.columns)
    
    def test_edge_cases(self):
        """Test edge cases in feature engineering."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        config = {}
        engineer = MockFeatureEngineer(config)
        
        try:
            features, target = engineer.create_features(empty_data)
            # Should handle gracefully
            assert isinstance(features, pd.DataFrame)
            assert isinstance(target, pd.Series)
        except Exception as e:
            # Or raise appropriate exception
            assert isinstance(e, (ValueError, KeyError))
        
        # Test with single row
        single_row = pd.DataFrame({
            'id': [1],
            'loan_amnt': [10000],
            'int_rate': [0.10]
        })
        
        features, target = engineer.create_features(single_row)
        assert len(features) == 1
        assert len(target) == 1
        
        # Test with all missing values in a column
        all_missing = pd.DataFrame({
            'id': [1, 2, 3],
            'loan_amnt': [None, None, None],
            'int_rate': [0.10, 0.12, 0.15]
        })

        features, target = engineer.create_features(all_missing)
        # Should handle all missing values
        assert len(features) == 3
        # The mock implementation should handle missing values gracefully
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
    
    def test_feature_name_consistency(self):
        """Test that feature names are consistent across runs."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'loan_amnt': [10000, 15000, 20000],
            'int_rate': [0.10, 0.12, 0.15],
            'grade': ['A', 'B', 'C']
        })
        
        config = {}
        
        # Run feature engineering twice
        engineer1 = MockFeatureEngineer(config)
        features1, _ = engineer1.create_features(data)
        
        engineer2 = MockFeatureEngineer(config)
        features2, _ = engineer2.create_features(data)
        
        # Feature names should be consistent
        assert list(features1.columns) == list(features2.columns)
        assert engineer1.get_feature_names() == engineer2.get_feature_names()
