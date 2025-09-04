"""
Strongly typed data models for Lending Club loan analysis.

This module defines Pydantic models that enforce type safety and data validation
throughout the machine learning pipeline.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from .enums import (
    LoanGrade, HomeOwnership, LoanPurpose, LoanStatus, 
    VerificationStatus, EmploymentLength, ApplicationType, InitialListStatus
)


class LoanApplication(BaseModel):
    """
    Represents a loan application with only listing-time information.
    
    This model enforces the critical constraint that only data available
    at loan listing time is included (no post-origination fields).
    """
    
    # Core loan characteristics (always available at listing)
    id: int = Field(..., description="Unique loan identifier")
    funded_amnt: Decimal = Field(..., gt=0, description="Loan amount funded")
    term: int = Field(..., description="Loan term in months")
    int_rate: Decimal = Field(..., ge=0, le=100, description="Interest rate percentage")
    installment: Decimal = Field(..., gt=0, description="Monthly payment amount")
    sub_grade: LoanGrade = Field(..., description="Loan sub-grade (A1-G5)")
    
    # Borrower information (listing-time only)
    emp_title: Optional[str] = Field(None, description="Employment title")
    emp_length: Optional[EmploymentLength] = Field(None, description="Employment length")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    annual_inc: Optional[Decimal] = Field(None, ge=0, description="Annual income")
    verification_status: VerificationStatus = Field(..., description="Income verification status")
    
    # Geographic and demographic
    zip_code: str = Field(..., min_length=5, max_length=5, description="ZIP code")
    addr_state: str = Field(..., min_length=2, max_length=2, description="State code")
    
    # Loan metadata
    issue_d: datetime = Field(..., description="Loan issue date")
    purpose: LoanPurpose = Field(..., description="Loan purpose")
    title: Optional[str] = Field(None, description="Loan title provided by borrower")
    
    # Credit history (available at listing)
    dti: Optional[Decimal] = Field(None, ge=0, description="Debt-to-income ratio")
    delinq_2yrs: Optional[int] = Field(None, ge=0, description="Delinquencies in past 2 years")
    earliest_cr_line: Optional[datetime] = Field(None, description="Earliest credit line date")
    fico_range_low: Optional[int] = Field(None, ge=300, le=850, description="FICO range low")
    fico_range_high: Optional[int] = Field(None, ge=300, le=850, description="FICO range high")
    inq_last_6mths: Optional[int] = Field(None, ge=0, description="Credit inquiries last 6 months")
    mths_since_last_delinq: Optional[int] = Field(None, ge=0, description="Months since last delinquency")
    mths_since_last_record: Optional[int] = Field(None, ge=0, description="Months since last public record")
    open_acc: Optional[int] = Field(None, ge=0, description="Number of open credit accounts")
    pub_rec: Optional[int] = Field(None, ge=0, description="Number of public records")
    revol_bal: Optional[Decimal] = Field(None, ge=0, description="Revolving balance")
    revol_util: Optional[Decimal] = Field(None, ge=0, le=100, description="Revolving utilization rate")
    total_acc: Optional[int] = Field(None, ge=0, description="Total number of credit accounts")
    
    # Application details
    initial_list_status: Optional[InitialListStatus] = Field(None, description="Initial listing status")
    application_type: ApplicationType = Field(default=ApplicationType.INDIVIDUAL, description="Application type")
    
    # Joint application fields (if applicable)
    annual_inc_joint: Optional[Decimal] = Field(None, ge=0, description="Joint annual income")
    dti_joint: Optional[Decimal] = Field(None, ge=0, description="Joint debt-to-income ratio")
    verification_status_joint: Optional[VerificationStatus] = Field(None, description="Joint income verification")
    
    # Extended credit attributes (listing-time safe)
    acc_now_delinq: Optional[int] = Field(None, ge=0, description="Current delinquent accounts")
    tot_coll_amt: Optional[Decimal] = Field(None, ge=0, description="Total collection amounts")
    tot_cur_bal: Optional[Decimal] = Field(None, ge=0, description="Total current balance")
    collections_12_mths_ex_med: Optional[int] = Field(None, ge=0, description="Collections last 12 months")
    mths_since_last_major_derog: Optional[int] = Field(None, ge=0, description="Months since last major derogatory")
    
    # Additional credit metrics (if available at listing)
    open_acc_6m: Optional[int] = Field(None, ge=0, description="Accounts opened in last 6 months")
    open_act_il: Optional[int] = Field(None, ge=0, description="Currently active installment accounts")
    open_il_12m: Optional[int] = Field(None, ge=0, description="Installment accounts opened last 12 months")
    open_il_24m: Optional[int] = Field(None, ge=0, description="Installment accounts opened last 24 months")
    mths_since_rcnt_il: Optional[int] = Field(None, ge=0, description="Months since most recent installment")
    total_bal_il: Optional[Decimal] = Field(None, ge=0, description="Total installment balance")
    il_util: Optional[Decimal] = Field(None, ge=0, description="Installment utilization")
    
    @validator('fico_range_high')
    def validate_fico_range(cls, v, values):
        """Ensure FICO high >= FICO low."""
        if v is not None and 'fico_range_low' in values and values['fico_range_low'] is not None:
            if v < values['fico_range_low']:
                raise ValueError('FICO range high must be >= FICO range low')
        return v
    
    @validator('term')
    def validate_term(cls, v):
        """Ensure term is a valid loan term."""
        if v not in [36, 60]:
            raise ValueError('Loan term must be 36 or 60 months')
        return v
    
    @root_validator
    def validate_joint_application(cls, values):
        """Validate joint application fields are consistent."""
        app_type = values.get('application_type')
        joint_fields = ['annual_inc_joint', 'dti_joint', 'verification_status_joint']
        
        if app_type == ApplicationType.JOINT:
            # Joint applications should have joint income
            if not values.get('annual_inc_joint'):
                raise ValueError('Joint applications must have joint annual income')
        elif app_type == ApplicationType.INDIVIDUAL:
            # Individual applications shouldn't have joint fields
            if any(values.get(field) for field in joint_fields):
                raise ValueError('Individual applications should not have joint fields')
        
        return values
    
    @property
    def credit_age_years(self) -> Optional[float]:
        """Calculate credit history age in years."""
        if self.earliest_cr_line and self.issue_d:
            delta = self.issue_d - self.earliest_cr_line
            return delta.days / 365.25
        return None
    
    @property
    def fico_midpoint(self) -> Optional[float]:
        """Calculate FICO score midpoint."""
        if self.fico_range_low and self.fico_range_high:
            return (self.fico_range_low + self.fico_range_high) / 2.0
        return None
    
    @property
    def loan_to_income_ratio(self) -> Optional[float]:
        """Calculate loan amount to annual income ratio."""
        income = self.annual_inc_joint if self.application_type == ApplicationType.JOINT else self.annual_inc
        if income and income > 0:
            return float(self.funded_amnt / income)
        return None

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True


class LoanOutcome(BaseModel):
    """
    Represents the outcome of a loan (for backtesting only).
    
    This data is NOT available at listing time and should only be used
    for evaluation and backtesting purposes.
    """
    
    loan_id: int = Field(..., description="Loan identifier")
    loan_status: LoanStatus = Field(..., description="Final loan status")
    
    # Payment information (post-origination only)
    total_payments: Optional[Decimal] = Field(None, ge=0, description="Total payments received")
    total_principal: Optional[Decimal] = Field(None, ge=0, description="Total principal received")
    total_interest: Optional[Decimal] = Field(None, ge=0, description="Total interest received")
    
    # Recovery information (if defaulted)
    recoveries: Optional[Decimal] = Field(None, ge=0, description="Recovery amount")
    collection_recovery_fee: Optional[Decimal] = Field(None, ge=0, description="Collection fees")
    
    @property
    def is_default(self) -> bool:
        """Returns True if the loan defaulted."""
        return self.loan_status.is_default
    
    @property
    def roi(self) -> Optional[float]:
        """Calculate return on investment."""
        if self.total_payments is not None:
            # Simple ROI calculation - can be enhanced with assumptions
            return float((self.total_payments - self.loan_id) / self.loan_id)
        return None


class PredictionResult(BaseModel):
    """Model prediction result."""
    
    loan_id: int = Field(..., description="Loan identifier")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted default probability")
    risk_score: float = Field(..., description="Risk score (higher = riskier)")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Prediction confidence")
    feature_contributions: Optional[Dict[str, float]] = Field(None, description="Feature contributions to prediction")
    
    @property
    def risk_tier(self) -> str:
        """Categorize risk level."""
        if self.default_probability < 0.05:
            return "LOW"
        elif self.default_probability < 0.15:
            return "MEDIUM"
        elif self.default_probability < 0.30:
            return "HIGH"
        else:
            return "VERY_HIGH"


class InvestmentPolicy(BaseModel):
    """Investment decision policy configuration."""
    
    budget_per_quarter: Decimal = Field(default=Decimal('5000'), gt=0, description="Investment budget per quarter")
    max_risk_tolerance: float = Field(default=0.15, ge=0, le=1, description="Maximum acceptable default probability")
    min_expected_return: Optional[float] = Field(None, description="Minimum expected return threshold")
    diversification_limits: Optional[Dict[str, int]] = Field(None, description="Portfolio diversification limits")
    
    # Selection strategy
    selection_method: str = Field(default="lowest_risk", description="Selection method: 'lowest_risk', 'highest_return', 'risk_adjusted'")
    max_loans_per_selection: Optional[int] = Field(None, gt=0, description="Maximum loans to select")
    
    def select_loans(self, predictions: List[PredictionResult], loan_applications: List[LoanApplication]) -> List[int]:
        """
        Select loans based on the investment policy.
        
        Args:
            predictions: List of prediction results
            loan_applications: List of loan applications
            
        Returns:
            List of selected loan IDs
        """
        # Filter by risk tolerance
        eligible = [p for p in predictions if p.default_probability <= self.max_risk_tolerance]
        
        if not eligible:
            return []
        
        # Sort by selection method
        if self.selection_method == "lowest_risk":
            eligible.sort(key=lambda x: x.default_probability)
        elif self.selection_method == "highest_return":
            eligible.sort(key=lambda x: -x.risk_score)  # Higher risk score = higher expected return
        
        # Apply budget constraint
        selected_ids = []
        remaining_budget = self.budget_per_quarter
        
        loan_dict = {loan.id: loan for loan in loan_applications}
        
        for prediction in eligible:
            if prediction.loan_id in loan_dict:
                loan = loan_dict[prediction.loan_id]
                if loan.funded_amnt <= remaining_budget:
                    selected_ids.append(prediction.loan_id)
                    remaining_budget -= loan.funded_amnt
                    
                    if self.max_loans_per_selection and len(selected_ids) >= self.max_loans_per_selection:
                        break
        
        return selected_ids


class BacktestResult(BaseModel):
    """Results from backtesting an investment policy."""
    
    quarter: str = Field(..., description="Quarter tested (e.g., '2017Q1')")
    total_budget: Decimal = Field(..., description="Total budget allocated")
    budget_utilized: Decimal = Field(..., description="Actual budget used")
    loans_selected: int = Field(..., description="Number of loans selected")
    
    # Performance metrics
    selected_default_rate: float = Field(..., description="Default rate of selected loans")
    overall_default_rate: float = Field(..., description="Overall default rate in quarter")
    risk_reduction: float = Field(..., description="Risk reduction vs random selection")
    
    # Financial metrics
    total_return: Decimal = Field(..., description="Total return amount")
    roi_percentage: float = Field(..., description="Return on investment percentage")
    
    # Risk metrics
    sharpe_ratio: Optional[float] = Field(None, description="Risk-adjusted return metric")
    max_drawdown: Optional[float] = Field(None, description="Maximum portfolio drawdown")
    
    @property
    def budget_utilization_rate(self) -> float:
        """Calculate budget utilization rate."""
        return float(self.budget_utilized / self.total_budget)
    
    @property
    def average_loan_amount(self) -> Decimal:
        """Calculate average loan amount selected."""
        if self.loans_selected > 0:
            return self.budget_utilized / self.loans_selected
        return Decimal('0')


class FeatureImportance(BaseModel):
    """Feature importance from model training."""
    
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    importance_type: str = Field(default="coefficient", description="Type of importance measure")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval for importance")


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""
    
    roc_auc: float = Field(..., ge=0, le=1, description="ROC-AUC score")
    brier_score: float = Field(..., ge=0, le=1, description="Brier score")
    log_loss: float = Field(..., ge=0, description="Log loss")
    
    # Calibration metrics
    calibration_slope: float = Field(..., description="Calibration slope")
    calibration_intercept: float = Field(..., description="Calibration intercept")
    hosmer_lemeshow_p_value: Optional[float] = Field(None, description="Hosmer-Lemeshow test p-value")
    
    # Classification metrics at default threshold
    precision: Optional[float] = Field(None, ge=0, le=1, description="Precision")
    recall: Optional[float] = Field(None, ge=0, le=1, description="Recall")
    f1_score: Optional[float] = Field(None, ge=0, le=1, description="F1 score")
    
    feature_importance: List[FeatureImportance] = Field(default_factory=list, description="Feature importance rankings")
    
    @property
    def is_well_calibrated(self) -> bool:
        """Check if model is well calibrated."""
        # Simple heuristic: calibration slope close to 1, intercept close to 0
        return abs(self.calibration_slope - 1.0) < 0.2 and abs(self.calibration_intercept) < 0.1
