"""
Unit tests for data models.

These tests verify that the Pydantic models correctly validate data,
enforce business rules, and provide expected behavior.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from pydantic import ValidationError

from lending_club.models.data_models import (
    LoanApplication, LoanOutcome, PredictionResult, 
    InvestmentPolicy, BacktestResult, ModelMetrics
)
from lending_club.models.enums import (
    LoanGrade, HomeOwnership, LoanPurpose, LoanStatus,
    VerificationStatus, ApplicationType
)


class TestLoanApplication:
    """Test cases for LoanApplication model."""
    
    def test_valid_loan_application_creation(self, sample_loan_application):
        """Test creating a valid loan application."""
        assert sample_loan_application.id == 12345
        assert sample_loan_application.funded_amnt == Decimal('10000')
        assert sample_loan_application.term == 36
        assert sample_loan_application.sub_grade == LoanGrade.B2
        assert sample_loan_application.home_ownership == HomeOwnership.RENT
    
    def test_fico_range_validation(self):
        """Test FICO range validation."""
        # Valid range
        loan_data = {
            "id": 1,
            "funded_amnt": Decimal('10000'),
            "term": 36,
            "int_rate": Decimal('12.5'),
            "installment": Decimal('333.33'),
            "sub_grade": LoanGrade.B2,
            "home_ownership": HomeOwnership.RENT,
            "annual_inc": Decimal('75000'),
            "verification_status": VerificationStatus.VERIFIED,
            "zip_code": "12345",
            "addr_state": "CA",
            "issue_d": datetime(2016, 1, 1),
            "purpose": LoanPurpose.DEBT_CONSOLIDATION,
            "fico_range_low": 700,
            "fico_range_high": 720,
            "application_type": ApplicationType.INDIVIDUAL
        }
        
        loan = LoanApplication(**loan_data)
        assert loan.fico_range_low == 700
        assert loan.fico_range_high == 720
        
        # Invalid range (high < low)
        loan_data["fico_range_high"] = 680
        with pytest.raises(ValidationError):
            LoanApplication(**loan_data)
    
    def test_term_validation(self):
        """Test loan term validation."""
        loan_data = {
            "id": 1,
            "funded_amnt": Decimal('10000'),
            "term": 48,  # Invalid term
            "int_rate": Decimal('12.5'),
            "installment": Decimal('333.33'),
            "sub_grade": LoanGrade.B2,
            "home_ownership": HomeOwnership.RENT,
            "annual_inc": Decimal('75000'),
            "verification_status": VerificationStatus.VERIFIED,
            "zip_code": "12345",
            "addr_state": "CA",
            "issue_d": datetime(2016, 1, 1),
            "purpose": LoanPurpose.DEBT_CONSOLIDATION,
            "application_type": ApplicationType.INDIVIDUAL
        }
        
        with pytest.raises(ValidationError):
            LoanApplication(**loan_data)
    
    def test_joint_application_validation(self):
        """Test joint application validation."""
        loan_data = {
            "id": 1,
            "funded_amnt": Decimal('10000'),
            "term": 36,
            "int_rate": Decimal('12.5'),
            "installment": Decimal('333.33'),
            "sub_grade": LoanGrade.B2,
            "home_ownership": HomeOwnership.RENT,
            "annual_inc": Decimal('75000'),
            "verification_status": VerificationStatus.VERIFIED,
            "zip_code": "12345",
            "addr_state": "CA",
            "issue_d": datetime(2016, 1, 1),
            "purpose": LoanPurpose.DEBT_CONSOLIDATION,
            "application_type": ApplicationType.JOINT,
            # Missing joint annual income - should fail
        }
        
        with pytest.raises(ValidationError):
            LoanApplication(**loan_data)
        
        # Valid joint application
        loan_data["annual_inc_joint"] = Decimal('150000')
        loan = LoanApplication(**loan_data)
        assert loan.application_type == ApplicationType.JOINT
        assert loan.annual_inc_joint == Decimal('150000')
    
    def test_credit_age_calculation(self, sample_loan_application):
        """Test credit age calculation property."""
        credit_age = sample_loan_application.credit_age_years
        assert credit_age is not None
        assert credit_age > 0
        # Should be about 6 years (2010 to 2016)
        assert 5.5 < credit_age < 6.5
    
    def test_fico_midpoint_calculation(self, sample_loan_application):
        """Test FICO midpoint calculation."""
        fico_mid = sample_loan_application.fico_midpoint
        assert fico_mid == 710.0  # (700 + 720) / 2
    
    def test_loan_to_income_ratio(self, sample_loan_application):
        """Test loan-to-income ratio calculation."""
        ratio = sample_loan_application.loan_to_income_ratio
        assert ratio is not None
        expected_ratio = float(Decimal('10000') / Decimal('75000'))
        assert abs(ratio - expected_ratio) < 0.001
    
    def test_missing_optional_fields(self):
        """Test handling of missing optional fields."""
        minimal_data = {
            "id": 1,
            "funded_amnt": Decimal('10000'),
            "term": 36,
            "int_rate": Decimal('12.5'),
            "installment": Decimal('333.33'),
            "sub_grade": LoanGrade.B2,
            "home_ownership": HomeOwnership.RENT,
            "verification_status": VerificationStatus.VERIFIED,
            "zip_code": "12345",
            "addr_state": "CA",
            "issue_d": datetime(2016, 1, 1),
            "purpose": LoanPurpose.DEBT_CONSOLIDATION,
            "application_type": ApplicationType.INDIVIDUAL
        }
        
        loan = LoanApplication(**minimal_data)
        assert loan.id == 1
        assert loan.annual_inc is None
        assert loan.fico_range_low is None
        assert loan.credit_age_years is None


class TestLoanOutcome:
    """Test cases for LoanOutcome model."""
    
    def test_default_detection(self, sample_loan_outcome_default):
        """Test default detection."""
        assert sample_loan_outcome_default.is_default is True
        assert sample_loan_outcome_default.loan_status == LoanStatus.CHARGED_OFF
    
    def test_non_default_detection(self, sample_loan_outcome_good):
        """Test non-default detection."""
        assert sample_loan_outcome_good.is_default is False
        assert sample_loan_outcome_good.loan_status == LoanStatus.FULLY_PAID


class TestPredictionResult:
    """Test cases for PredictionResult model."""
    
    def test_risk_tier_categorization(self):
        """Test risk tier categorization."""
        # Low risk
        pred_low = PredictionResult(loan_id=1, default_probability=0.03, risk_score=0.05)
        assert pred_low.risk_tier == "LOW"
        
        # Medium risk
        pred_med = PredictionResult(loan_id=2, default_probability=0.10, risk_score=0.15)
        assert pred_med.risk_tier == "MEDIUM"
        
        # High risk
        pred_high = PredictionResult(loan_id=3, default_probability=0.20, risk_score=0.25)
        assert pred_high.risk_tier == "HIGH"
        
        # Very high risk
        pred_very_high = PredictionResult(loan_id=4, default_probability=0.40, risk_score=0.50)
        assert pred_very_high.risk_tier == "VERY_HIGH"
    
    def test_probability_bounds_validation(self):
        """Test probability bounds validation."""
        # Valid probability
        pred = PredictionResult(loan_id=1, default_probability=0.5, risk_score=0.3)
        assert pred.default_probability == 0.5
        
        # Invalid probability (> 1)
        with pytest.raises(ValidationError):
            PredictionResult(loan_id=1, default_probability=1.5, risk_score=0.3)
        
        # Invalid probability (< 0)
        with pytest.raises(ValidationError):
            PredictionResult(loan_id=1, default_probability=-0.1, risk_score=0.3)


class TestInvestmentPolicy:
    """Test cases for InvestmentPolicy model."""
    
    def test_default_policy_creation(self):
        """Test creating default investment policy."""
        policy = InvestmentPolicy()
        assert policy.budget_per_quarter == Decimal('5000')
        assert policy.max_risk_tolerance == 0.15
        assert policy.selection_method == "lowest_risk"
    
    def test_loan_selection_by_risk(self, sample_loan_applications_list):
        """Test loan selection by lowest risk."""
        policy = InvestmentPolicy(
            budget_per_quarter=Decimal('30000'),
            max_risk_tolerance=0.20
        )
        
        predictions = [
            PredictionResult(loan_id=12345, default_probability=0.08, risk_score=0.12),
            PredictionResult(loan_id=12346, default_probability=0.05, risk_score=0.08),
        ]
        
        selected = policy.select_loans(predictions, sample_loan_applications_list)
        
        # Should select loan with lower risk first (12346, then 12345)
        assert len(selected) == 2
        assert selected[0] == 12346  # Lower risk
        assert selected[1] == 12345
    
    def test_budget_constraint(self, sample_loan_applications_list):
        """Test budget constraint enforcement."""
        policy = InvestmentPolicy(
            budget_per_quarter=Decimal('15000'),  # Only enough for one loan
            max_risk_tolerance=0.20
        )
        
        predictions = [
            PredictionResult(loan_id=12345, default_probability=0.08, risk_score=0.12),
            PredictionResult(loan_id=12346, default_probability=0.05, risk_score=0.08),
        ]
        
        selected = policy.select_loans(predictions, sample_loan_applications_list)
        
        # Should only select one loan due to budget constraint
        assert len(selected) == 1
        assert selected[0] == 12346  # Lower risk loan selected
    
    def test_risk_tolerance_filtering(self, sample_loan_applications_list):
        """Test risk tolerance filtering."""
        policy = InvestmentPolicy(
            budget_per_quarter=Decimal('50000'),
            max_risk_tolerance=0.06  # Very low tolerance
        )
        
        predictions = [
            PredictionResult(loan_id=12345, default_probability=0.08, risk_score=0.12),  # Too risky
            PredictionResult(loan_id=12346, default_probability=0.05, risk_score=0.08),  # Acceptable
        ]
        
        selected = policy.select_loans(predictions, sample_loan_applications_list)
        
        # Should only select the low-risk loan
        assert len(selected) == 1
        assert selected[0] == 12346


class TestBacktestResult:
    """Test cases for BacktestResult model."""
    
    def test_budget_utilization_calculation(self):
        """Test budget utilization rate calculation."""
        result = BacktestResult(
            quarter="2017Q1",
            total_budget=Decimal('5000'),
            budget_utilized=Decimal('4500'),
            loans_selected=3,
            selected_default_rate=0.08,
            overall_default_rate=0.12,
            risk_reduction=0.04,
            total_return=Decimal('500'),
            roi_percentage=11.1
        )
        
        assert result.budget_utilization_rate == 0.9  # 4500/5000
    
    def test_average_loan_amount_calculation(self):
        """Test average loan amount calculation."""
        result = BacktestResult(
            quarter="2017Q1",
            total_budget=Decimal('5000'),
            budget_utilized=Decimal('4500'),
            loans_selected=3,
            selected_default_rate=0.08,
            overall_default_rate=0.12,
            risk_reduction=0.04,
            total_return=Decimal('500'),
            roi_percentage=11.1
        )
        
        assert result.average_loan_amount == Decimal('1500')  # 4500/3
    
    def test_zero_loans_selected(self):
        """Test handling when no loans are selected."""
        result = BacktestResult(
            quarter="2017Q1",
            total_budget=Decimal('5000'),
            budget_utilized=Decimal('0'),
            loans_selected=0,
            selected_default_rate=0.0,
            overall_default_rate=0.12,
            risk_reduction=0.0,
            total_return=Decimal('0'),
            roi_percentage=0.0
        )
        
        assert result.average_loan_amount == Decimal('0')


class TestModelMetrics:
    """Test cases for ModelMetrics model."""
    
    def test_calibration_assessment(self):
        """Test calibration assessment."""
        # Well calibrated model
        metrics_good = ModelMetrics(
            roc_auc=0.75,
            brier_score=0.15,
            log_loss=0.45,
            calibration_slope=1.05,
            calibration_intercept=-0.02
        )
        assert metrics_good.is_well_calibrated is True
        
        # Poorly calibrated model
        metrics_bad = ModelMetrics(
            roc_auc=0.75,
            brier_score=0.15,
            log_loss=0.45,
            calibration_slope=0.5,  # Too far from 1
            calibration_intercept=0.15  # Too far from 0
        )
        assert metrics_bad.is_well_calibrated is False
