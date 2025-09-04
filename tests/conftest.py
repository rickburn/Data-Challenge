"""
Pytest configuration and shared fixtures for the Lending Club test suite.

This module defines common test fixtures and configurations that are
shared across all test modules.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Generator, List, Dict, Any

from lending_club.models.data_models import LoanApplication, LoanOutcome, PredictionResult
from lending_club.models.enums import (
    LoanGrade, HomeOwnership, LoanPurpose, LoanStatus, 
    VerificationStatus, ApplicationType
)


@pytest.fixture
def sample_loan_application() -> LoanApplication:
    """Create a sample loan application for testing."""
    return LoanApplication(
        id=12345,
        funded_amnt=Decimal('10000'),
        term=36,
        int_rate=Decimal('12.50'),
        installment=Decimal('333.33'),
        sub_grade=LoanGrade.B2,
        emp_title="Software Engineer",
        emp_length=None,  # Will be handled by model
        home_ownership=HomeOwnership.RENT,
        annual_inc=Decimal('75000'),
        verification_status=VerificationStatus.VERIFIED,
        zip_code="12345",
        addr_state="CA",
        issue_d=datetime(2016, 1, 1),
        purpose=LoanPurpose.DEBT_CONSOLIDATION,
        title="Debt consolidation",
        dti=Decimal('18.50'),
        delinq_2yrs=0,
        earliest_cr_line=datetime(2010, 1, 1),
        fico_range_low=700,
        fico_range_high=720,
        inq_last_6mths=1,
        mths_since_last_delinq=None,
        mths_since_last_record=None,
        open_acc=8,
        pub_rec=0,
        revol_bal=Decimal('5000'),
        revol_util=Decimal('25.5'),
        total_acc=15,
        application_type=ApplicationType.INDIVIDUAL
    )


@pytest.fixture
def sample_joint_loan_application() -> LoanApplication:
    """Create a sample joint loan application for testing."""
    return LoanApplication(
        id=12346,
        funded_amnt=Decimal('25000'),
        term=60,
        int_rate=Decimal('10.50'),
        installment=Decimal('537.89'),
        sub_grade=LoanGrade.A3,
        emp_title="Manager",
        home_ownership=HomeOwnership.MORTGAGE,
        annual_inc=Decimal('100000'),
        verification_status=VerificationStatus.SOURCE_VERIFIED,
        zip_code="90210",
        addr_state="CA",
        issue_d=datetime(2016, 2, 1),
        purpose=LoanPurpose.HOME_IMPROVEMENT,
        title="Home improvement project",
        dti=Decimal('15.0'),
        delinq_2yrs=0,
        earliest_cr_line=datetime(2008, 5, 1),
        fico_range_low=750,
        fico_range_high=780,
        inq_last_6mths=0,
        open_acc=12,
        pub_rec=0,
        revol_bal=Decimal('8000'),
        revol_util=Decimal('15.2'),
        total_acc=20,
        application_type=ApplicationType.JOINT,
        annual_inc_joint=Decimal('150000'),
        dti_joint=Decimal('12.0'),
        verification_status_joint=VerificationStatus.VERIFIED
    )


@pytest.fixture
def sample_loan_outcome_good() -> LoanOutcome:
    """Create a sample loan outcome for a fully paid loan."""
    return LoanOutcome(
        loan_id=12345,
        loan_status=LoanStatus.FULLY_PAID,
        total_payments=Decimal('12000'),
        total_principal=Decimal('10000'),
        total_interest=Decimal('2000'),
        recoveries=None,
        collection_recovery_fee=None
    )


@pytest.fixture
def sample_loan_outcome_default() -> LoanOutcome:
    """Create a sample loan outcome for a defaulted loan."""
    return LoanOutcome(
        loan_id=12346,
        loan_status=LoanStatus.CHARGED_OFF,
        total_payments=Decimal('3000'),
        total_principal=Decimal('2500'),
        total_interest=Decimal('500'),
        recoveries=Decimal('500'),
        collection_recovery_fee=Decimal('50')
    )


@pytest.fixture
def sample_prediction_result() -> PredictionResult:
    """Create a sample prediction result."""
    return PredictionResult(
        loan_id=12345,
        default_probability=0.08,
        risk_score=0.12,
        confidence_interval={"lower": 0.05, "upper": 0.12},
        feature_contributions={
            "fico_midpoint": -0.15,
            "dti": 0.08,
            "revol_util": 0.05
        }
    )


@pytest.fixture
def sample_loan_applications_list(
    sample_loan_application: LoanApplication,
    sample_joint_loan_application: LoanApplication
) -> List[LoanApplication]:
    """Create a list of sample loan applications."""
    return [sample_loan_application, sample_joint_loan_application]


@pytest.fixture
def sample_prediction_results_list() -> List[PredictionResult]:
    """Create a list of sample prediction results."""
    return [
        PredictionResult(loan_id=12345, default_probability=0.08, risk_score=0.12),
        PredictionResult(loan_id=12346, default_probability=0.05, risk_score=0.08),
        PredictionResult(loan_id=12347, default_probability=0.15, risk_score=0.20),
        PredictionResult(loan_id=12348, default_probability=0.25, risk_score=0.35),
    ]


@pytest.fixture
def test_data_directory() -> str:
    """Return path to test data directory."""
    return "tests/data"


@pytest.fixture
def mock_csv_data() -> Dict[str, Any]:
    """Create mock CSV data for testing data loading."""
    return {
        "id": [1, 2, 3],
        "funded_amnt": [10000, 15000, 8000],
        "term": [36, 60, 36],
        "int_rate": [12.5, 8.9, 15.2],
        "installment": [333.33, 287.50, 275.88],
        "sub_grade": ["B2", "A1", "C3"],
        "home_ownership": ["RENT", "OWN", "MORTGAGE"],
        "annual_inc": [75000, 120000, 45000],
        "verification_status": ["Verified", "Not Verified", "Source Verified"],
        "zip_code": ["12345", "67890", "54321"],
        "addr_state": ["CA", "NY", "TX"],
        "issue_d": ["2016-01-01", "2016-01-01", "2016-01-01"],
        "purpose": ["debt_consolidation", "home_improvement", "credit_card"],
        "dti": [18.5, 12.0, 25.8],
        "fico_range_low": [700, 780, 650],
        "fico_range_high": [720, 800, 670],
        "application_type": ["Individual", "Individual", "Individual"]
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"  
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "data_dependent: mark test as requiring real data files"
    )
