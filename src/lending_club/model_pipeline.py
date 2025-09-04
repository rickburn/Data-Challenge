"""
Model Training and Calibration Pipeline
=======================================

This module handles model training, hyperparameter optimization, and probability
calibration for the Lending Club default prediction task.

Key Features:
- Supports multiple model types (logistic, random forest, XGBoost, LightGBM)
- Hyperparameter optimization with cross-validation
- Probability calibration using Platt scaling or isotonic regression
- Comprehensive model evaluation and validation
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging_config import log_execution, track_data_transformation

# Optional dependencies for advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# GPU detection
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
    if HAS_CUDA:
        print(f"✅ GPU acceleration available: {GPU_COUNT} GPU(s) detected")
        for i in range(GPU_COUNT):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  No GPU detected, using CPU only")
except ImportError:
    HAS_CUDA = False
    GPU_COUNT = 0
    print("⚠️  PyTorch not available for GPU detection")


class ModelTrainer:
    """Trains ML models with hyperparameter optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer with configuration."""
        self.config = config
        self.model_type = config.get('type', 'logistic')
        self.random_state = config.get('random_state', 42)
        self.hyperparameter_search = config.get('hyperparameter_search', True)
        self.cv_folds = config.get('cv_folds', 5)
        self.search_iterations = config.get('search_iterations', 100)

        # Hardware configuration
        hardware_config = config.get('hardware', {})
        self.n_jobs = hardware_config.get('n_jobs', -1)
        self.max_threads = hardware_config.get('max_threads', 8)
        self.use_gpu = hardware_config.get('use_gpu', False)
        self.gpu_memory_limit = hardware_config.get('gpu_memory_limit', 0.8)

        # Performance thresholds
        self.min_roc_auc = config.get('min_roc_auc', 0.65)
        self.max_brier_score = config.get('max_brier_score', 0.20)

        # Model and training results
        self.model = None
        self.best_params_ = {}
        self.training_metrics_ = {}
        self.feature_importance_ = {}

        self.logger = logging.getLogger(__name__)

        # Validate GPU configuration
        if self.use_gpu and not HAS_CUDA:
            self.logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
            self.use_gpu = False
        elif self.use_gpu and GPU_COUNT == 0:
            self.logger.warning("GPU requested but no GPUs detected. Falling back to CPU.")
            self.use_gpu = False
        elif self.use_gpu:
            self.logger.info(f"GPU acceleration enabled: {GPU_COUNT} GPU(s) available")

        # Log hardware configuration
        self.logger.info(f"Hardware config: n_jobs={self.n_jobs}, max_threads={self.max_threads}, use_gpu={self.use_gpu}")
    
    @log_execution
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """Train model with hyperparameter optimization."""
        self.logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        
        # Get base model and parameter grid
        base_model = self._get_base_model()
        param_grid = self._get_parameter_grid()
        
        if self.hyperparameter_search and param_grid:
            self.logger.info("Performing hyperparameter optimization...")
            self.model = self._perform_hyperparameter_search(
                base_model, param_grid, X_train, y_train
            )
        else:
            self.logger.info("Training with default parameters...")
            self.model = base_model.fit(X_train, y_train)
            self.best_params_ = base_model.get_params()
        
        # Evaluate model performance
        self._evaluate_model_performance(X_train, y_train, X_val, y_val)
        
        # Extract feature importance
        self._extract_feature_importance(X_train.columns)
        
        # Validate performance thresholds
        self._validate_performance_thresholds()
        
        self.logger.info("Model training completed successfully")
        
        # Track training
        track_data_transformation(
            operation="train_model",
            input_data={'X_train_shape': X_train.shape, 'y_train_distribution': y_train.value_counts().to_dict()},
            output_data={'model_type': self.model_type, 'best_params': self.best_params_},
            metadata=self.training_metrics_
        )
        
        return self.model
    
    def _get_base_model(self) -> Any:
        """Get base model according to configuration."""
        if self.model_type == 'logistic':
            model_config = self.config.get('logistic_regression', {})
            return LogisticRegression(
                random_state=self.random_state,
                **model_config
            )
        
        elif self.model_type == 'random_forest':
            model_config = self.config.get('random_forest', {})
            # Add thread configuration
            model_config['n_jobs'] = self.n_jobs
            return RandomForestClassifier(
                random_state=self.random_state,
                **model_config
            )

        elif self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")

            model_config = self.config.get('xgboost', {})

            # Add GPU configuration if enabled
            if self.use_gpu:
                model_config.update({
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'gpu_id': 0
                })
                self.logger.info("XGBoost GPU acceleration enabled")

            # Add thread configuration
            model_config['n_jobs'] = self.n_jobs

            return xgb.XGBClassifier(
                random_state=self.random_state,
                **model_config
            )

        elif self.model_type == 'lightgbm':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

            model_config = self.config.get('lightgbm', {})

            # Add GPU configuration if enabled
            if self.use_gpu:
                model_config.update({
                    'device': 'gpu',
                    'gpu_device_id': 0
                })
                self.logger.info("LightGBM GPU acceleration enabled")

            # Add thread configuration
            model_config['n_jobs'] = self.n_jobs

            return lgb.LGBMClassifier(
                random_state=self.random_state,
                **model_config
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for model optimization."""
        if not self.hyperparameter_search:
            return {}
        
        if self.model_type == 'logistic':
            return {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [15, 31, 63],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        return {}
    
    def _perform_hyperparameter_search(self, base_model: Any, param_grid: Dict[str, List[Any]],
                                     X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform hyperparameter search using cross-validation."""
        # Use stratified k-fold to maintain class balance
        cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Perform grid search with configured threading
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=self.max_threads,  # Use configured max threads instead of -1
            verbose=1
        )

        self.logger.info(f"Starting hyperparameter search with {self.max_threads} threads")
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params_ = grid_search.best_params_
        best_score = grid_search.best_score_
        
        self.logger.info(f"Best cross-validation ROC-AUC: {best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model_performance(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Evaluate model performance on training and validation sets."""
        # Training set evaluation
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba),
            'brier_score': brier_score_loss(y_train, y_train_proba),
            'log_loss': log_loss(y_train, y_train_proba)
        }
        
        # Validation set evaluation  
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred),
            'roc_auc': roc_auc_score(y_val, y_val_proba),
            'brier_score': brier_score_loss(y_val, y_val_proba),
            'log_loss': log_loss(y_val, y_val_proba)
        }
        
        # Store metrics
        self.training_metrics_ = {
            'train': train_metrics,
            'validation': val_metrics
        }
        
        # Log key metrics
        self.logger.info(f"Training ROC-AUC: {train_metrics['roc_auc']:.4f}")
        self.logger.info(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        self.logger.info(f"Training Brier Score: {train_metrics['brier_score']:.4f}")
        self.logger.info(f"Validation Brier Score: {val_metrics['brier_score']:.4f}")
    
    def _extract_feature_importance(self, feature_names: List[str]) -> None:
        """Extract feature importance from trained model."""
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models - use absolute coefficients
            importances = np.abs(self.model.coef_[0])
        else:
            self.logger.warning("Model does not support feature importance extraction")
            return
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Store top features
        self.feature_importance_ = {
            'all_features': feature_importance,
            'top_10_features': dict(sorted_importance[:10]),
            'feature_names': feature_names,
            'importance_values': importances
        }
        
        # Log top features
        top_5 = [f"{name}: {importance:.4f}" for name, importance in sorted_importance[:5]]
        self.logger.info(f"Top 5 features: {top_5}")
    
    def _validate_performance_thresholds(self) -> None:
        """Validate model meets minimum performance thresholds."""
        val_metrics = self.training_metrics_.get('validation', {})
        
        roc_auc = val_metrics.get('roc_auc', 0)
        brier_score = val_metrics.get('brier_score', 1)
        
        issues = []
        
        if roc_auc < self.min_roc_auc:
            issues.append(f"ROC-AUC ({roc_auc:.4f}) below minimum threshold ({self.min_roc_auc})")
        
        if brier_score > self.max_brier_score:
            issues.append(f"Brier Score ({brier_score:.4f}) above maximum threshold ({self.max_brier_score})")
        
        if issues:
            warning_msg = "Model performance issues detected: " + "; ".join(issues)
            self.logger.warning(warning_msg)
        else:
            self.logger.info("✅ Model meets all performance thresholds")
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters from training."""
        return self.best_params_
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training and validation metrics."""
        return self.training_metrics_
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance analysis."""
        return self.feature_importance_
    
    def save_model_artifacts(self, model: Any, metadata: Dict[str, Any], 
                           output_path: Path) -> None:
        """Save model and associated metadata."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this model
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = output_path / f"model_{self.model_type}_{timestamp}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = output_path / f"metadata_{self.model_type}_{timestamp}.joblib"
        joblib.dump(metadata, metadata_path)
        
        # Save feature importance plot
        if self.feature_importance_:
            self._save_feature_importance_plot(output_path, timestamp)
        
        self.logger.info(f"Model artifacts saved to {output_path}")
    
    def _save_feature_importance_plot(self, output_path: Path, timestamp: str) -> None:
        """Save feature importance visualization."""
        try:
            top_features = self.feature_importance_['top_10_features']
            
            plt.figure(figsize=(10, 6))
            features, importances = zip(*list(top_features.items()))
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Feature Importances - {self.model_type.title()} Model')
            plt.tight_layout()
            
            plot_path = output_path / f"feature_importance_{self.model_type}_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save feature importance plot: {e}")


class ModelCalibrator:
    """Calibrates model probabilities for better reliability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model calibrator with configuration."""
        self.config = config
        self.calibration_method = config.get('calibration_method', 'platt')
        self.enable_calibration = config.get('enable_calibration', True)
        self.min_calibration_pvalue = config.get('min_calibration_pvalue', 0.05)
        
        # Calibration results
        self.calibrated_model = None
        self.calibration_metrics_ = {}
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def calibrate_model(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """Calibrate model probabilities."""
        if not self.enable_calibration:
            self.logger.info("Calibration disabled - returning original model")
            return model
        
        self.logger.info(f"Calibrating model using {self.calibration_method} method")
        
        # Choose calibration method
        method = 'sigmoid' if self.calibration_method == 'platt' else 'isotonic'
        
        # Create calibrated classifier (sklearn 1.2+ API)
        calibrated_classifier = CalibratedClassifierCV(
            estimator=model,  # Changed from base_estimator to estimator for sklearn 1.2+
            method=method,
            cv='prefit'  # Use prefit since we're passing validation data
        )
        
        # Fit calibration on validation set
        self.calibrated_model = calibrated_classifier.fit(X_val, y_val)
        
        # Evaluate calibration quality
        self._evaluate_calibration_quality(model, X_val, y_val)
        
        self.logger.info("Model calibration completed")
        
        return self.calibrated_model
    
    def _evaluate_calibration_quality(self, original_model: Any, X_val: pd.DataFrame, 
                                    y_val: pd.Series) -> None:
        """Evaluate calibration quality using reliability diagrams."""
        # Get predictions from both models
        original_proba = original_model.predict_proba(X_val)[:, 1]
        calibrated_proba = self.calibrated_model.predict_proba(X_val)[:, 1]
        
        # Calculate calibration curves
        original_fraction_pos, original_mean_pred = calibration_curve(
            y_val, original_proba, n_bins=10
        )
        calibrated_fraction_pos, calibrated_mean_pred = calibration_curve(
            y_val, calibrated_proba, n_bins=10
        )
        
        # Calculate calibration metrics
        original_brier = brier_score_loss(y_val, original_proba)
        calibrated_brier = brier_score_loss(y_val, calibrated_proba)
        
        # Calculate Expected Calibration Error (ECE)
        original_ece = self._calculate_expected_calibration_error(y_val, original_proba)
        calibrated_ece = self._calculate_expected_calibration_error(y_val, calibrated_proba)
        
        # Calculate calibration slope and intercept
        calibrated_slope, calibrated_intercept = self._calculate_calibration_slope_intercept(
            y_val, calibrated_proba
        )
        
        # Store calibration metrics
        self.calibration_metrics_ = {
            'original_brier_score': original_brier,
            'calibrated_brier_score': calibrated_brier,
            'brier_score_improvement': original_brier - calibrated_brier,
            'original_ece': original_ece,
            'calibrated_ece': calibrated_ece,
            'ece_improvement': original_ece - calibrated_ece,
            'calibration_slope': calibrated_slope,
            'calibration_intercept': calibrated_intercept,
            'calibration_curves': {
                'original': {
                    'fraction_positives': original_fraction_pos.tolist(),
                    'mean_predicted': original_mean_pred.tolist()
                },
                'calibrated': {
                    'fraction_positives': calibrated_fraction_pos.tolist(),
                    'mean_predicted': calibrated_mean_pred.tolist()
                }
            }
        }
        
        # Log calibration improvements
        self.logger.info(f"Brier Score improvement: {self.calibration_metrics_['brier_score_improvement']:.4f}")
        self.logger.info(f"ECE improvement: {self.calibration_metrics_['ece_improvement']:.4f}")
        self.logger.info(f"Calibration slope: {calibrated_slope:.4f} (ideal: 1.0)")
        self.logger.info(f"Calibration intercept: {calibrated_intercept:.4f} (ideal: 0.0)")
    
    def _calculate_expected_calibration_error(self, y_true: pd.Series, y_prob: np.ndarray,
                                            n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_calibration_slope_intercept(self, y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, float]:
        """Calculate calibration slope and intercept using logistic regression."""
        from sklearn.linear_model import LogisticRegression
        
        # Convert probabilities to log-odds (logits)
        epsilon = 1e-15  # Small epsilon to avoid log(0)
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Fit logistic regression: y_true ~ logits
        calibration_model = LogisticRegression()
        calibration_model.fit(logits.reshape(-1, 1), y_true)
        
        slope = calibration_model.coef_[0][0]
        intercept = calibration_model.intercept_[0]
        
        return slope, intercept
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get calibration quality metrics."""
        return self.calibration_metrics_
    
    def plot_calibration_curve(self, output_path: Path, timestamp: str = None) -> None:
        """Plot calibration curve comparison."""
        if not self.calibration_metrics_:
            self.logger.warning("No calibration metrics available for plotting")
            return
        
        try:
            curves = self.calibration_metrics_['calibration_curves']
            
            plt.figure(figsize=(10, 8))
            
            # Plot perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            # Plot original model calibration
            original = curves['original']
            plt.plot(original['mean_predicted'], original['fraction_positives'], 
                    'o-', label='Original Model', linewidth=2)
            
            # Plot calibrated model calibration  
            calibrated = curves['calibrated']
            plt.plot(calibrated['mean_predicted'], calibrated['fraction_positives'],
                    's-', label='Calibrated Model', linewidth=2)
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot (Reliability Curve)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = (
                f"ECE Improvement: {self.calibration_metrics_['ece_improvement']:.4f}\n"
                f"Brier Score Improvement: {self.calibration_metrics_['brier_score_improvement']:.4f}\n"
                f"Calibration Slope: {self.calibration_metrics_['calibration_slope']:.3f}\n"
                f"Calibration Intercept: {self.calibration_metrics_['calibration_intercept']:.3f}"
            )
            plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save plot
            if timestamp is None:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            plot_path = output_path / f"calibration_curve_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Calibration curve saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save calibration curve: {e}")
