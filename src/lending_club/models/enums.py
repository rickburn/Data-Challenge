"""
Enumeration classes for Lending Club loan data.

This module defines strongly typed enums for categorical fields to ensure
data integrity and type safety throughout the pipeline.
"""

from enum import Enum
from typing import Optional


class LoanGrade(str, Enum):
    """Loan grade categories (A-G with sub-grades)."""
    A1 = "A1"
    A2 = "A2" 
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"


class HomeOwnership(str, Enum):
    """Home ownership status."""
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanPurpose(str, Enum):
    """Purpose of the loan."""
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    SMALL_BUSINESS = "small_business"
    CAR = "car"
    VACATION = "vacation"
    WEDDING = "wedding"
    HOUSE = "house"
    MOVING = "moving"
    EDUCATIONAL = "educational"
    RENEWABLE_ENERGY = "renewable_energy"
    OTHER = "other"


class LoanStatus(str, Enum):
    """Loan status (target variable)."""
    CURRENT = "Current"
    FULLY_PAID = "Fully Paid"
    CHARGED_OFF = "Charged Off"
    LATE_31_120 = "Late (31-120 days)"
    IN_GRACE_PERIOD = "In Grace Period"
    LATE_16_30 = "Late (16-30 days)"
    DEFAULT = "Default"
    
    @property
    def is_default(self) -> bool:
        """Returns True if this status represents a default."""
        return self in {
            LoanStatus.CHARGED_OFF,
            LoanStatus.DEFAULT,
            LoanStatus.LATE_31_120
        }


class VerificationStatus(str, Enum):
    """Income verification status."""
    VERIFIED = "Verified"
    SOURCE_VERIFIED = "Source Verified"
    NOT_VERIFIED = "Not Verified"


class EmploymentLength(str, Enum):
    """Employment length categories."""
    LESS_THAN_1_YEAR = "< 1 year"
    ONE_YEAR = "1 year"
    TWO_YEARS = "2 years"
    THREE_YEARS = "3 years"
    FOUR_YEARS = "4 years"
    FIVE_YEARS = "5 years"
    SIX_YEARS = "6 years"
    SEVEN_YEARS = "7 years"
    EIGHT_YEARS = "8 years"
    NINE_YEARS = "9 years"
    TEN_PLUS_YEARS = "10+ years"
    
    @property
    def numeric_years(self) -> float:
        """Convert employment length to numeric years for modeling."""
        mapping = {
            self.LESS_THAN_1_YEAR: 0.5,
            self.ONE_YEAR: 1.0,
            self.TWO_YEARS: 2.0,
            self.THREE_YEARS: 3.0,
            self.FOUR_YEARS: 4.0,
            self.FIVE_YEARS: 5.0,
            self.SIX_YEARS: 6.0,
            self.SEVEN_YEARS: 7.0,
            self.EIGHT_YEARS: 8.0,
            self.NINE_YEARS: 9.0,
            self.TEN_PLUS_YEARS: 12.0,  # Assumption for 10+ years
        }
        return mapping[self]


class ApplicationType(str, Enum):
    """Application type."""
    INDIVIDUAL = "Individual"
    JOINT = "Joint App"


class InitialListStatus(str, Enum):
    """Initial listing status."""
    WHOLE = "w"
    FRACTIONAL = "f"
