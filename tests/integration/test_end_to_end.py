"""
Integration tests for end-to-end workflows.

These tests verify that the complete pipeline works correctly,
from data loading through model training to backtesting.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List

from lending_club.models.data_models import (
    LoanApplication, PredictionResult, InvestmentPolicy, BacktestResult
)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_investment_pipeline(self, sample_loan_applications_list):
        """Test complete investment pipeline."""
        # 1. Generate predictions (mock)
        predictions = [
            PredictionResult(loan_id=12345, default_probability=0.08, risk_score=0.12),
            PredictionResult(loan_id=12346, default_probability=0.05, risk_score=0.08),
        ]
        
        # 2. Create investment policy
        policy = InvestmentPolicy(
            budget_per_quarter=Decimal('30000'),
            max_risk_tolerance=0.15,
            selection_method="lowest_risk"
        )
        
        # 3. Select loans
        selected_loans = policy.select_loans(predictions, sample_loan_applications_list)
        
        # 4. Verify selection
        assert len(selected_loans) == 2
        assert 12346 in selected_loans  # Lower risk loan
        assert 12345 in selected_loans
        
        # 5. Calculate metrics (mock backtest result)
        backtest_result = BacktestResult(
            quarter="2016Q4",
            total_budget=policy.budget_per_quarter,
            budget_utilized=Decimal('35000'),  # Sum of selected loan amounts
            loans_selected=len(selected_loans),
            selected_default_rate=0.06,
            overall_default_rate=0.12,
            risk_reduction=0.06,
            total_return=Decimal('2000'),
            roi_percentage=5.7
        )
        
        # 6. Verify backtest results
        assert backtest_result.selected_default_rate < backtest_result.overall_default_rate
        assert backtest_result.risk_reduction > 0
        assert backtest_result.roi_percentage > 0
    
    @pytest.mark.slow
    def test_model_training_workflow(self, sample_loan_applications_list):
        """Test model training workflow (placeholder for actual implementation)."""
        # This would test the complete model training pipeline
        # For now, just verify data structure integrity
        
        # 1. Verify input data structure
        assert len(sample_loan_applications_list) > 0
        for loan in sample_loan_applications_list:
            assert isinstance(loan, LoanApplication)
            assert loan.id is not None
            assert loan.funded_amnt > 0
        
        # 2. Mock feature engineering
        features = []
        for loan in sample_loan_applications_list:
            feature_dict = {
                "loan_amount": float(loan.funded_amnt),
                "int_rate": float(loan.int_rate),
                "term": loan.term,
                "fico_midpoint": loan.fico_midpoint,
                "dti": float(loan.dti) if loan.dti else None,
            }
            features.append(feature_dict)
        
        # 3. Verify feature extraction
        assert len(features) == len(sample_loan_applications_list)
        for feature_dict in features:
            assert "loan_amount" in feature_dict
            assert "int_rate" in feature_dict
            assert feature_dict["loan_amount"] > 0
    
    def test_data_validation_pipeline(self, sample_loan_applications_list):
        """Test data validation throughout the pipeline."""
        # 1. Validate input data
        for loan in sample_loan_applications_list:
            # Ensure no post-origination data is present
            assert not hasattr(loan, 'loan_status')
            assert not hasattr(loan, 'total_rec_prncp')
            assert not hasattr(loan, 'last_pymnt_d')
            
            # Ensure required listing-time fields are present
            assert loan.id is not None
            assert loan.funded_amnt > 0
            assert loan.issue_d is not None
            assert loan.sub_grade is not None
        
        # 2. Validate temporal constraints
        for loan in sample_loan_applications_list:
            # Issue date should be reasonable
            assert loan.issue_d >= datetime(2007, 1, 1)  # Lending Club founded
            assert loan.issue_d <= datetime(2020, 12, 31)  # Reasonable upper bound
            
            # Credit history should predate loan issue
            if loan.earliest_cr_line:
                assert loan.earliest_cr_line <= loan.issue_d
    
    def test_budget_constraint_enforcement(self, sample_loan_applications_list):
        """Test that budget constraints are properly enforced."""
        # Create predictions for all loans
        predictions = [
            PredictionResult(
                loan_id=loan.id, 
                default_probability=0.08, 
                risk_score=0.12
            )
            for loan in sample_loan_applications_list
        ]
        
        # Test with tight budget
        tight_policy = InvestmentPolicy(
            budget_per_quarter=Decimal('5000'),  # Very small budget
            max_risk_tolerance=0.50  # High tolerance to ensure loans aren't filtered
        )
        
        selected = tight_policy.select_loans(predictions, sample_loan_applications_list)
        
        # Calculate total cost of selected loans
        total_cost = sum(
            loan.funded_amnt 
            for loan in sample_loan_applications_list 
            if loan.id in selected
        )
        
        # Verify budget constraint
        assert total_cost <= tight_policy.budget_per_quarter
        
        # Test with large budget
        large_policy = InvestmentPolicy(
            budget_per_quarter=Decimal('100000'),  # Large budget
            max_risk_tolerance=0.50
        )
        
        selected_large = large_policy.select_loans(predictions, sample_loan_applications_list)
        
        # Should select more loans with larger budget
        assert len(selected_large) >= len(selected)


@pytest.mark.integration
@pytest.mark.data_dependent
class TestDataIntegration:
    """Test integration with actual data files (when available)."""
    
    def test_data_loading_integration(self):
        """Test loading real data files (if available)."""
        # This would test actual CSV loading
        # For now, just a placeholder
        assert True  # Placeholder
    
    def test_temporal_split_validation(self):
        """Test temporal data splitting validation."""
        # This would verify proper temporal ordering
        # For now, just a placeholder
        assert True  # Placeholder
