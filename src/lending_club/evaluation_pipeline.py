"""
Backtesting and Evaluation Pipeline
===================================

This module handles backtesting of investment strategies and comprehensive
evaluation of model performance and financial returns.

Key Features:
- Backtesting on held-out temporal data
- ROI calculation and performance metrics
- Model calibration evaluation
- Benchmark comparisons (random selection, market average)
- Comprehensive visualization and reporting
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils.logging_config import log_execution, track_data_transformation
from src.lending_club.investment_pipeline import InvestmentDecision


class BacktestEvaluator:
    """Evaluates investment strategy performance through backtesting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backtest evaluator with configuration."""
        self.config = config
        self.required_metrics = config.get('required_metrics', [
            'roc_auc', 'brier_score', 'calibration_slope', 'calibration_intercept',
            'expected_calibration_error'
        ])
        
        # Backtesting parameters
        self.backtest_metrics = config.get('backtest_metrics', [
            'default_rate_comparison', 'roi_proxy', 'sharpe_ratio', 'maximum_drawdown'
        ])
        
        # Benchmarks
        self.benchmarks = config.get('benchmarks', {
            'random_selection': True,
            'market_average': True
        })
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def evaluate_backtest(self, predictions: np.ndarray, actual_outcomes: pd.Series,
                         investment_decisions: Dict[str, Any], 
                         loan_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive backtesting evaluation."""
        self.logger.info(f"Starting backtest evaluation on {len(predictions)} loans")
        
        # Model performance evaluation
        model_metrics = self._evaluate_model_performance(predictions, actual_outcomes)
        
        # Investment performance evaluation  
        investment_metrics = self._evaluate_investment_performance(
            investment_decisions, actual_outcomes, loan_data
        )
        
        # Calibration evaluation
        calibration_metrics = self._evaluate_calibration_quality(predictions, actual_outcomes)
        
        # Benchmark comparisons
        benchmark_metrics = self._evaluate_benchmarks(
            predictions, actual_outcomes, loan_data, investment_decisions
        )
        
        # Combined results
        backtest_results = {
            'model_performance': model_metrics,
            'investment_performance': investment_metrics,
            'calibration_quality': calibration_metrics,
            'benchmark_comparisons': benchmark_metrics,
            'summary_metrics': self._create_summary_metrics(
                model_metrics, investment_metrics, calibration_metrics
            )
        }
        
        self.logger.info("Backtest evaluation completed")
        
        # Track evaluation
        track_data_transformation(
            operation="evaluate_backtest",
            input_data={
                'prediction_count': len(predictions),
                'investment_count': len(investment_decisions['selected_loans'])
            },
            output_data=backtest_results['summary_metrics'],
            metadata={'evaluation_timestamp': pd.Timestamp.now().isoformat()}
        )
        
        return backtest_results
    
    def _evaluate_model_performance(self, predictions: np.ndarray, 
                                  actual_outcomes: pd.Series) -> Dict[str, Any]:
        """Evaluate model prediction performance."""
        # Ensure predictions and outcomes are aligned
        actual_outcomes = actual_outcomes.iloc[:len(predictions)]
        
        # Basic classification metrics
        model_metrics = {
            'roc_auc': roc_auc_score(actual_outcomes, predictions),
            'brier_score': brier_score_loss(actual_outcomes, predictions),
            'log_loss': self._safe_log_loss(actual_outcomes, predictions)
        }
        
        # Prediction distribution analysis
        model_metrics.update({
            'prediction_mean': np.mean(predictions),
            'prediction_std': np.std(predictions),
            'prediction_min': np.min(predictions),
            'prediction_max': np.max(predictions),
            'actual_default_rate': actual_outcomes.mean()
        })
        
        # Add threshold-based metrics
        threshold_metrics = self._evaluate_threshold_metrics(predictions, actual_outcomes)
        model_metrics.update(threshold_metrics)
        
        self.logger.info(f"Model ROC-AUC: {model_metrics['roc_auc']:.4f}")
        self.logger.info(f"Model Brier Score: {model_metrics['brier_score']:.4f}")
        
        return model_metrics
    
    def _evaluate_investment_performance(self, investment_decisions: Dict[str, Any],
                                       actual_outcomes: pd.Series, 
                                       loan_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate actual investment performance."""
        selected_loans = investment_decisions['selected_loans']
        
        if not selected_loans:
            return {
                'total_investment': 0,
                'actual_defaults': 0,
                'default_rate': 0,
                'roi_proxy': 0,
                'total_interest_income': 0,
                'total_losses': 0,
                'net_return': 0
            }
        
        # Extract loan IDs and map to outcomes
        selected_loan_ids = [loan.loan_id for loan in selected_loans]
        investment_amounts = {loan.loan_id: loan.investment_amount for loan in selected_loans}
        
        # Find corresponding outcomes (this is simplified - in practice would need proper ID matching)
        # For now, use positional matching assuming same order
        selected_outcomes = actual_outcomes.iloc[:len(selected_loans)]
        selected_loan_data = loan_data.iloc[:len(selected_loans)]
        
        # Calculate actual performance
        total_investment = sum(loan.investment_amount for loan in selected_loans)
        actual_defaults = selected_outcomes.sum()
        default_rate = actual_defaults / len(selected_loans)
        
        # Calculate returns
        total_interest_income = 0
        total_losses = 0
        
        for i, (loan, outcome) in enumerate(zip(selected_loans, selected_outcomes)):
            if outcome == 0:  # No default
                # Calculate interest income (simplified)
                annual_rate = loan.interest_rate
                term_years = loan.term / 12.0
                interest_income = loan.investment_amount * annual_rate * term_years
                total_interest_income += interest_income
            else:  # Default
                # Calculate loss with recovery
                recovery_rate = 0.30  # From config
                loss = loan.investment_amount * (1 - recovery_rate)
                total_losses += loss
        
        net_return = total_interest_income - total_losses
        roi_proxy = net_return / total_investment if total_investment > 0 else 0
        
        investment_metrics = {
            'total_investment': total_investment,
            'loans_selected': len(selected_loans),
            'actual_defaults': actual_defaults,
            'default_rate': default_rate,
            'total_interest_income': total_interest_income,
            'total_losses': total_losses,
            'net_return': net_return,
            'roi_proxy': roi_proxy,
            'annualized_return': roi_proxy * (12 / np.mean([loan.term for loan in selected_loans]))
        }
        
        # Risk-adjusted metrics
        if len(selected_loans) > 1:
            returns_by_loan = []
            for i, (loan, outcome) in enumerate(zip(selected_loans, selected_outcomes)):
                if outcome == 0:
                    loan_return = (loan.interest_rate * (loan.term / 12.0))
                else:
                    loan_return = -0.70  # 70% loss on default (30% recovery)
                returns_by_loan.append(loan_return)
            
            returns_array = np.array(returns_by_loan)
            investment_metrics.update({
                'return_volatility': np.std(returns_array),
                'sharpe_ratio': np.mean(returns_array) / (np.std(returns_array) + 1e-8),
                'maximum_drawdown': np.min(returns_array) if len(returns_array) > 0 else 0
            })
        
        self.logger.info(f"Portfolio default rate: {default_rate:.3f}")
        self.logger.info(f"ROI proxy: {roi_proxy:.3f}")
        
        return investment_metrics
    
    def _evaluate_calibration_quality(self, predictions: np.ndarray,
                                    actual_outcomes: pd.Series) -> Dict[str, Any]:
        """Evaluate calibration quality of probability predictions."""
        actual_outcomes = actual_outcomes.iloc[:len(predictions)]
        
        # Calculate calibration curve
        fraction_pos, mean_pred = calibration_curve(
            actual_outcomes, predictions, n_bins=10
        )
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_expected_calibration_error(actual_outcomes, predictions)
        
        # Calculate calibration slope and intercept
        slope, intercept = self._calculate_calibration_slope_intercept(
            actual_outcomes, predictions
        )
        
        # Hosmer-Lemeshow test for calibration
        hl_statistic, hl_pvalue = self._hosmer_lemeshow_test(
            actual_outcomes, predictions
        )
        
        calibration_metrics = {
            'expected_calibration_error': ece,
            'calibration_slope': slope,
            'calibration_intercept': intercept,
            'hosmer_lemeshow_statistic': hl_statistic,
            'hosmer_lemeshow_pvalue': hl_pvalue,
            'well_calibrated': hl_pvalue > 0.05,  # Null hypothesis: well calibrated
            'calibration_curve': {
                'fraction_positives': fraction_pos.tolist(),
                'mean_predicted': mean_pred.tolist()
            }
        }
        
        self.logger.info(f"Expected Calibration Error: {ece:.4f}")
        self.logger.info(f"Calibration slope: {slope:.3f} (ideal: 1.0)")
        self.logger.info(f"Hosmer-Lemeshow p-value: {hl_pvalue:.4f}")
        
        return calibration_metrics
    
    def _evaluate_benchmarks(self, predictions: np.ndarray, actual_outcomes: pd.Series,
                           loan_data: pd.DataFrame, investment_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance against benchmarks."""
        benchmark_results = {}
        
        if self.benchmarks.get('random_selection', False):
            random_metrics = self._evaluate_random_selection_benchmark(
                actual_outcomes, loan_data, investment_decisions
            )
            benchmark_results['random_selection'] = random_metrics
        
        if self.benchmarks.get('market_average', False):
            market_metrics = self._evaluate_market_average_benchmark(
                actual_outcomes, loan_data, investment_decisions
            )
            benchmark_results['market_average'] = market_metrics
        
        return benchmark_results
    
    def _evaluate_random_selection_benchmark(self, actual_outcomes: pd.Series,
                                           loan_data: pd.DataFrame,
                                           investment_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate random loan selection benchmark."""
        selected_count = len(investment_decisions['selected_loans'])
        total_investment = investment_decisions['portfolio_metrics']['total_investment']
        
        # Simulate random selection multiple times
        n_simulations = 100
        random_results = []
        
        np.random.seed(42)  # For reproducible results
        
        for _ in range(n_simulations):
            # Randomly select same number of loans
            random_indices = np.random.choice(
                len(actual_outcomes), size=min(selected_count, len(actual_outcomes)), 
                replace=False
            )
            
            random_outcomes = actual_outcomes.iloc[random_indices]
            random_default_rate = random_outcomes.mean()
            
            # Estimate ROI (simplified)
            avg_interest_rate = loan_data.iloc[random_indices]['int_rate'].mean() if 'int_rate' in loan_data.columns else 0.12
            random_roi = (1 - random_default_rate) * avg_interest_rate - random_default_rate * 0.70
            
            random_results.append({
                'default_rate': random_default_rate,
                'roi_proxy': random_roi
            })
        
        # Calculate statistics across simulations
        default_rates = [r['default_rate'] for r in random_results]
        roi_proxies = [r['roi_proxy'] for r in random_results]
        
        return {
            'mean_default_rate': np.mean(default_rates),
            'std_default_rate': np.std(default_rates),
            'mean_roi_proxy': np.mean(roi_proxies),
            'std_roi_proxy': np.std(roi_proxies),
            'simulations_run': n_simulations
        }
    
    def _evaluate_market_average_benchmark(self, actual_outcomes: pd.Series,
                                         loan_data: pd.DataFrame,
                                         investment_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market average benchmark."""
        # Use overall market performance as benchmark
        market_default_rate = actual_outcomes.mean()
        
        # Estimate market ROI
        if 'int_rate' in loan_data.columns:
            avg_market_rate = loan_data['int_rate'].mean()
            market_roi = (1 - market_default_rate) * avg_market_rate - market_default_rate * 0.70
        else:
            market_roi = 0.05  # Assume 5% market return
        
        return {
            'market_default_rate': market_default_rate,
            'market_roi_proxy': market_roi,
            'data_points': len(actual_outcomes)
        }
    
    def _evaluate_threshold_metrics(self, predictions: np.ndarray, 
                                  actual_outcomes: pd.Series) -> Dict[str, Any]:
        """Evaluate metrics at various probability thresholds."""
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
        threshold_metrics = {}
        
        for threshold in thresholds:
            predicted_positive = (predictions >= threshold).astype(int)
            
            if np.sum(predicted_positive) > 0:
                precision = np.sum((predicted_positive == 1) & (actual_outcomes == 1)) / np.sum(predicted_positive)
                recall = np.sum((predicted_positive == 1) & (actual_outcomes == 1)) / np.sum(actual_outcomes)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                threshold_metrics[f'threshold_{threshold}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'selected_fraction': np.mean(predicted_positive)
                }
        
        return threshold_metrics
    
    def _calculate_expected_calibration_error(self, y_true: pd.Series, y_prob: np.ndarray,
                                            n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_calibration_slope_intercept(self, y_true: pd.Series, 
                                             y_prob: np.ndarray) -> Tuple[float, float]:
        """Calculate calibration slope and intercept."""
        from sklearn.linear_model import LogisticRegression
        
        # Convert probabilities to log-odds
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Fit calibration model
        cal_model = LogisticRegression(fit_intercept=True)
        cal_model.fit(logits.reshape(-1, 1), y_true)
        
        slope = cal_model.coef_[0][0]
        intercept = cal_model.intercept_[0]
        
        return slope, intercept
    
    def _hosmer_lemeshow_test(self, y_true: pd.Series, y_prob: np.ndarray,
                            n_bins: int = 10) -> Tuple[float, float]:
        """Perform Hosmer-Lemeshow goodness-of-fit test."""
        # Create bins based on predicted probabilities
        bin_indices = np.digitize(y_prob, np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        
        hl_statistic = 0
        for bin_idx in range(1, n_bins + 1):
            in_bin = (bin_indices == bin_idx)
            
            if np.sum(in_bin) > 0:
                observed_positive = np.sum(y_true[in_bin])
                observed_negative = np.sum(in_bin) - observed_positive
                expected_positive = np.sum(y_prob[in_bin])
                expected_negative = np.sum(in_bin) - expected_positive
                
                # Add to chi-square statistic
                if expected_positive > 0:
                    hl_statistic += (observed_positive - expected_positive) ** 2 / expected_positive
                if expected_negative > 0:
                    hl_statistic += (observed_negative - expected_negative) ** 2 / expected_negative
        
        # Calculate p-value (chi-square distribution with n_bins - 2 degrees of freedom)
        p_value = 1 - stats.chi2.cdf(hl_statistic, n_bins - 2)
        
        return hl_statistic, p_value
    
    def _safe_log_loss(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        """Calculate log loss with numerical stability."""
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    
    def _create_summary_metrics(self, model_metrics: Dict[str, Any],
                              investment_metrics: Dict[str, Any],
                              calibration_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create high-level summary metrics."""
        return {
            'model_auc': model_metrics.get('roc_auc', 0),
            'model_brier_score': model_metrics.get('brier_score', 1),
            'calibration_error': calibration_metrics.get('expected_calibration_error', 1),
            'portfolio_default_rate': investment_metrics.get('default_rate', 0),
            'portfolio_roi': investment_metrics.get('roi_proxy', 0),
            'sharpe_ratio': investment_metrics.get('sharpe_ratio', 0),
            'well_calibrated': calibration_metrics.get('well_calibrated', False),
            'total_investment': investment_metrics.get('total_investment', 0),
            'loans_selected': investment_metrics.get('loans_selected', 0)
        }
    
    @log_execution
    def generate_evaluation_plots(self, backtest_results: Dict[str, Any],
                                investment_summary: Dict[str, Any],
                                output_path: Path) -> None:
        """Generate comprehensive evaluation plots."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Set style for plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Model Performance Overview
            self._plot_model_performance_overview(backtest_results, output_path)
            
            # 2. Calibration Analysis
            self._plot_calibration_analysis(backtest_results, output_path)
            
            # 3. Investment Performance
            self._plot_investment_performance(backtest_results, investment_summary, output_path)
            
            # 4. Benchmark Comparisons
            self._plot_benchmark_comparisons(backtest_results, output_path)
            
            # 5. Risk Analysis
            self._plot_risk_analysis(backtest_results, investment_summary, output_path)
            
            self.logger.info(f"Evaluation plots saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation plots: {e}")
    
    def _plot_model_performance_overview(self, backtest_results: Dict[str, Any],
                                       output_path: Path) -> None:
        """Plot model performance overview."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        model_metrics = backtest_results['model_performance']
        
        # ROC AUC Score
        ax1.bar(['ROC AUC'], [model_metrics.get('roc_auc', 0)], color='skyblue')
        ax1.set_ylim(0, 1)
        ax1.set_title('ROC AUC Score')
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Random')
        ax1.legend()
        
        # Brier Score
        ax2.bar(['Brier Score'], [model_metrics.get('brier_score', 0)], color='lightcoral')
        ax2.set_ylim(0, 0.5)
        ax2.set_title('Brier Score (Lower is Better)')
        
        # Prediction Distribution
        predictions_mean = model_metrics.get('prediction_mean', 0)
        predictions_std = model_metrics.get('prediction_std', 0)
        ax3.bar(['Mean', 'Std Dev'], [predictions_mean, predictions_std], 
               color=['lightgreen', 'orange'])
        ax3.set_title('Prediction Distribution')
        ax3.set_ylabel('Probability')
        
        # Actual vs Predicted Default Rate
        actual_rate = model_metrics.get('actual_default_rate', 0)
        predicted_rate = predictions_mean
        ax4.bar(['Actual', 'Predicted'], [actual_rate, predicted_rate], 
               color=['darkblue', 'lightblue'])
        ax4.set_title('Default Rate Comparison')
        ax4.set_ylabel('Default Rate')
        
        plt.suptitle('Model Performance Overview', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_analysis(self, backtest_results: Dict[str, Any],
                                 output_path: Path) -> None:
        """Plot calibration analysis."""
        calibration_metrics = backtest_results['calibration_quality']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration Curve
        curve_data = calibration_metrics.get('calibration_curve', {})
        if curve_data:
            fraction_pos = curve_data['fraction_positives']
            mean_pred = curve_data['mean_predicted']
            
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax1.plot(mean_pred, fraction_pos, 'o-', label='Model Calibration', linewidth=2)
            ax1.set_xlabel('Mean Predicted Probability')
            ax1.set_ylabel('Fraction of Positives')
            ax1.set_title('Calibration Plot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Calibration Metrics
        metrics = [
            ('ECE', calibration_metrics.get('expected_calibration_error', 0)),
            ('Slope', calibration_metrics.get('calibration_slope', 1)),
            ('Intercept', calibration_metrics.get('calibration_intercept', 0))
        ]
        
        metric_names, metric_values = zip(*metrics)
        colors = ['red' if abs(v - ideal) > 0.1 else 'green' 
                 for v, ideal in zip(metric_values, [0, 1, 0])]
        
        bars = ax2.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Calibration Metrics')
        ax2.set_ylabel('Value')
        
        # Add ideal values as text
        ax2.text(0, -0.1, 'Ideal: 0', ha='center', transform=ax2.transData)
        ax2.text(1, 1.1, 'Ideal: 1', ha='center', transform=ax2.transData)
        ax2.text(2, -0.1, 'Ideal: 0', ha='center', transform=ax2.transData)
        
        plt.suptitle('Model Calibration Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_investment_performance(self, backtest_results: Dict[str, Any],
                                   investment_summary: Dict[str, Any],
                                   output_path: Path) -> None:
        """Plot investment performance analysis."""
        investment_metrics = backtest_results['investment_performance']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio vs Market Default Rate
        portfolio_default = investment_metrics.get('default_rate', 0)
        market_default = backtest_results.get('benchmark_comparisons', {}).get(
            'market_average', {}).get('market_default_rate', portfolio_default * 1.2)
        
        ax1.bar(['Portfolio', 'Market Average'], [portfolio_default, market_default],
               color=['darkgreen', 'lightgray'])
        ax1.set_title('Default Rate Comparison')
        ax1.set_ylabel('Default Rate')
        
        # ROI Comparison
        portfolio_roi = investment_metrics.get('roi_proxy', 0)
        market_roi = backtest_results.get('benchmark_comparisons', {}).get(
            'market_average', {}).get('market_roi_proxy', 0.05)
        
        ax2.bar(['Portfolio', 'Market Average'], [portfolio_roi, market_roi],
               color=['darkblue', 'lightgray'])
        ax2.set_title('ROI Comparison')
        ax2.set_ylabel('ROI Proxy')
        
        # Investment Allocation
        total_investment = investment_metrics.get('total_investment', 0)
        budget = investment_summary.get('budget_allocated', 5000)
        unused_budget = budget - total_investment
        
        sizes = [total_investment, unused_budget]
        labels = [f'Invested\n${total_investment:.0f}', f'Cash\n${unused_budget:.0f}']
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Budget Utilization')
        
        # Risk-Return Profile
        sharpe_ratio = investment_metrics.get('sharpe_ratio', 0)
        return_vol = investment_metrics.get('return_volatility', 0)
        
        ax4.scatter([return_vol], [portfolio_roi], s=200, c='red', label='Portfolio')
        ax4.scatter([return_vol * 1.5], [market_roi], s=200, c='gray', label='Market')
        ax4.set_xlabel('Return Volatility')
        ax4.set_ylabel('Expected Return')
        ax4.set_title(f'Risk-Return Profile\n(Sharpe Ratio: {sharpe_ratio:.3f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Investment Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'investment_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_benchmark_comparisons(self, backtest_results: Dict[str, Any],
                                  output_path: Path) -> None:
        """Plot benchmark comparisons."""
        benchmark_data = backtest_results.get('benchmark_comparisons', {})
        
        if not benchmark_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Selection Benchmark
        if 'random_selection' in benchmark_data:
            random_data = benchmark_data['random_selection']
            portfolio_default = backtest_results['investment_performance'].get('default_rate', 0)
            random_default_mean = random_data.get('mean_default_rate', 0)
            random_default_std = random_data.get('std_default_rate', 0)
            
            ax1.bar(['Portfolio', 'Random Mean'], [portfolio_default, random_default_mean],
                   yerr=[0, random_default_std], color=['green', 'orange'], alpha=0.7)
            ax1.set_title('Default Rate vs Random Selection')
            ax1.set_ylabel('Default Rate')
            
            # ROI comparison
            portfolio_roi = backtest_results['investment_performance'].get('roi_proxy', 0)
            random_roi_mean = random_data.get('mean_roi_proxy', 0)
            random_roi_std = random_data.get('std_roi_proxy', 0)
            
            ax2.bar(['Portfolio', 'Random Mean'], [portfolio_roi, random_roi_mean],
                   yerr=[0, random_roi_std], color=['blue', 'orange'], alpha=0.7)
            ax2.set_title('ROI vs Random Selection')
            ax2.set_ylabel('ROI Proxy')
        
        plt.suptitle('Benchmark Comparisons', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'benchmark_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_analysis(self, backtest_results: Dict[str, Any],
                          investment_summary: Dict[str, Any],
                          output_path: Path) -> None:
        """Plot risk analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Grade Distribution
        grade_dist = investment_summary.get('diversification_metrics', {}).get('grade_distribution', {})
        if grade_dist:
            grades, percentages = zip(*grade_dist.items())
            ax1.pie(percentages, labels=grades, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Investment by Grade')
        
        # Term Distribution
        term_dist = investment_summary.get('diversification_metrics', {}).get('term_distribution', {})
        if term_dist:
            terms, percentages = zip(*term_dist.items())
            ax2.pie(percentages, labels=[f'{t} months' for t in terms], autopct='%1.1f%%', startangle=90)
            ax2.set_title('Investment by Term')
        
        # Risk Metrics Summary
        risk_profile = investment_summary.get('portfolio_risk_profile', {})
        avg_risk = risk_profile.get('avg_default_probability', 0)
        risk_range = risk_profile.get('risk_range', {})
        min_risk = risk_range.get('min', 0)
        max_risk = risk_range.get('max', 0)
        
        ax3.bar(['Min', 'Average', 'Max'], [min_risk, avg_risk, max_risk], 
               color=['green', 'orange', 'red'])
        ax3.set_title('Portfolio Risk Distribution')
        ax3.set_ylabel('Default Probability')
        
        # Performance Summary
        performance = investment_summary.get('expected_performance', {})
        metrics = ['Expected Return', 'Sharpe Ratio']
        values = [performance.get('avg_expected_return', 0), 
                 performance.get('portfolio_sharpe_ratio', 0)]
        
        ax4.bar(metrics, values, color=['darkblue', 'darkgreen'])
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Value')
        
        plt.suptitle('Portfolio Risk Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
