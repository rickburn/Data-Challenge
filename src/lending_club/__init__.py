"""
Lending Club ML Pipeline Package
================================

A comprehensive machine learning pipeline for predicting loan default risk
and optimizing investment decisions.

Main Components:
- data_pipeline: Data loading and validation
- feature_pipeline: Feature engineering with listing-time compliance
- model_pipeline: Model training and calibration
- investment_pipeline: Investment decision making under constraints
- evaluation_pipeline: Backtesting and performance evaluation
- reporting: Comprehensive report generation
"""

from .data_pipeline import DataLoader, DataValidator
from .feature_pipeline import FeatureEngineer
from .model_pipeline import ModelTrainer, ModelCalibrator
from .investment_pipeline import InvestmentDecisionMaker
from .evaluation_pipeline import BacktestEvaluator
from .reporting import ReportGenerator

__version__ = "1.0.0"
__author__ = "Lending Club ML Team"

__all__ = [
    "DataLoader",
    "DataValidator", 
    "FeatureEngineer",
    "ModelTrainer",
    "ModelCalibrator",
    "InvestmentDecisionMaker",
    "BacktestEvaluator",
    "ReportGenerator"
]