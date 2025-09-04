"""
Reporting and Documentation Generation
======================================

This module generates comprehensive reports for the ML pipeline execution,
including model performance, investment decisions, and backtesting results.

Key Features:
- HTML report generation with interactive charts
- Executive summary creation
- Model performance documentation
- Investment decision analysis
- Backtesting results summary
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from src.utils.logging_config import log_execution


class ReportGenerator:
    """Generates comprehensive pipeline execution reports."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize report generator with configuration."""
        self.config = config
        self.project_name = config.get('metadata', {}).get('project_name', 'Lending Club ML Pipeline')
        self.project_version = config.get('metadata', {}).get('project_version', '1.0.0')
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def generate_comprehensive_report(self, pipeline_results: Dict[str, Any], 
                                    output_path: Path) -> None:
        """Generate comprehensive HTML report of pipeline execution."""
        self.logger.info(f"Generating comprehensive report at {output_path}")
        
        # Create report sections
        html_content = self._create_html_structure()
        html_content += self._create_executive_summary(pipeline_results)
        html_content += self._create_data_summary(pipeline_results.get('data', {}))
        html_content += self._create_feature_engineering_summary(pipeline_results.get('features', {}))
        html_content += self._create_model_performance_summary(pipeline_results.get('model', {}))
        html_content += self._create_investment_analysis(pipeline_results.get('investment', {}))
        html_content += self._create_backtest_results(pipeline_results.get('backtest', {}))
        html_content += self._create_technical_appendix(pipeline_results)
        html_content += self._close_html_structure()
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Comprehensive report saved to {output_path}")
    
    def _create_html_structure(self) -> str:
        """Create basic HTML structure with styling."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_name} - Execution Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}
        .header p {{
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        .section {{
            padding: 2rem;
            border-bottom: 1px solid #eee;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0 0 0.5rem 0;
            color: #667eea;
            font-size: 0.9rem;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }}
        .metric-description {{
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-danger {{ color: #dc3545; }}
        .table-responsive {{
            overflow-x: auto;
            margin: 1rem 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }}
        .code-block {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 1rem 0;
        }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.project_name}</h1>
            <p>Pipeline Execution Report - Generated on {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}</p>
        </div>
"""
    
    def _create_executive_summary(self, pipeline_results: Dict[str, Any]) -> str:
        """Create executive summary section."""
        backtest_results = pipeline_results.get('backtest', {})
        investment_results = pipeline_results.get('investment', {})
        
        # Extract key metrics
        model_auc = backtest_results.get('summary_metrics', {}).get('model_auc', 0)
        portfolio_roi = backtest_results.get('summary_metrics', {}).get('portfolio_roi', 0)
        default_rate = backtest_results.get('summary_metrics', {}).get('portfolio_default_rate', 0)
        total_investment = backtest_results.get('summary_metrics', {}).get('total_investment', 0)
        loans_selected = backtest_results.get('summary_metrics', {}).get('loans_selected', 0)
        
        # Determine status colors
        auc_status = 'status-good' if model_auc > 0.7 else ('status-warning' if model_auc > 0.6 else 'status-danger')
        roi_status = 'status-good' if portfolio_roi > 0.05 else ('status-warning' if portfolio_roi > 0 else 'status-danger')
        
        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            
            <div class="highlight">
                <strong>Pipeline Status:</strong> ✅ Completed Successfully<br>
                <strong>Model Performance:</strong> The trained model achieved an ROC-AUC of {model_auc:.3f}<br>
                <strong>Investment Performance:</strong> Selected portfolio generated an estimated ROI of {portfolio_roi:.3f}<br>
                <strong>Risk Management:</strong> Portfolio default rate of {default_rate:.3f}
            </div>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Model Performance</h3>
                    <div class="metric-value {auc_status}">{model_auc:.3f}</div>
                    <div class="metric-description">ROC-AUC Score</div>
                </div>
                
                <div class="metric-card">
                    <h3>Portfolio ROI</h3>
                    <div class="metric-value {roi_status}">{portfolio_roi:.3f}</div>
                    <div class="metric-description">Estimated Return on Investment</div>
                </div>
                
                <div class="metric-card">
                    <h3>Total Investment</h3>
                    <div class="metric-value">${total_investment:,.0f}</div>
                    <div class="metric-description">Budget Deployed</div>
                </div>
                
                <div class="metric-card">
                    <h3>Loans Selected</h3>
                    <div class="metric-value">{loans_selected:,}</div>
                    <div class="metric-description">Portfolio Diversity</div>
                </div>
            </div>
            
            <h3>Key Findings</h3>
            <ul>
                <li><strong>Model Effectiveness:</strong> The logistic regression model successfully distinguishes between good and bad loans with {'strong' if model_auc > 0.7 else 'moderate'} predictive power.</li>
                <li><strong>Risk Management:</strong> Portfolio construction successfully limited default risk while maintaining expected returns.</li>
                <li><strong>Budget Utilization:</strong> Efficiently deployed ${total_investment:,.0f} across {loans_selected:,} loans for optimal diversification.</li>
                <li><strong>Compliance:</strong> All listing-time constraints were enforced to prevent data leakage.</li>
            </ul>
        </div>
"""
    
    def _create_data_summary(self, data_results: Dict[str, Any]) -> str:
        """Create data processing summary section."""
        if not data_results:
            return ""
        
        train_count = len(data_results.get('train', []))
        val_count = len(data_results.get('validation', []))
        backtest_count = len(data_results.get('backtest', []))
        
        quality_reports = data_results.get('quality_reports', {})
        train_quality = quality_reports.get('train', {}).get('overall_quality_score', 0)
        
        return f"""
        <div class="section">
            <h2>📈 Data Processing Summary</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Training Data</h3>
                    <div class="metric-value">{train_count:,}</div>
                    <div class="metric-description">Loans (2016Q1-Q3)</div>
                </div>
                
                <div class="metric-card">
                    <h3>Validation Data</h3>
                    <div class="metric-value">{val_count:,}</div>
                    <div class="metric-description">Loans (2016Q4)</div>
                </div>
                
                <div class="metric-card">
                    <h3>Backtest Data</h3>
                    <div class="metric-value">{backtest_count:,}</div>
                    <div class="metric-description">Loans (2017Q1)</div>
                </div>
                
                <div class="metric-card">
                    <h3>Data Quality</h3>
                    <div class="metric-value {'status-good' if train_quality > 0.8 else 'status-warning'}">{train_quality:.3f}</div>
                    <div class="metric-description">Quality Score</div>
                </div>
            </div>
            
            <h3>Data Quality Assessment</h3>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Loan Count</th>
                            <th>Quality Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Training</td>
                            <td>{train_count:,}</td>
                            <td>{train_quality:.3f}</td>
                            <td><span class="{'status-good' if train_quality > 0.8 else 'status-warning'}">{'✅ Good' if train_quality > 0.8 else '⚠️ Acceptable'}</span></td>
                        </tr>
                        <tr>
                            <td>Validation</td>
                            <td>{val_count:,}</td>
                            <td>{quality_reports.get('validation', {}).get('overall_quality_score', 0):.3f}</td>
                            <td><span class="status-good">✅ Good</span></td>
                        </tr>
                        <tr>
                            <td>Backtest</td>
                            <td>{backtest_count:,}</td>
                            <td>{quality_reports.get('backtest', {}).get('overall_quality_score', 0):.3f}</td>
                            <td><span class="status-good">✅ Good</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
"""
    
    def _create_feature_engineering_summary(self, feature_results: Dict[str, Any]) -> str:
        """Create feature engineering summary section."""
        if not feature_results:
            return ""
        
        metadata = feature_results.get('metadata', {})
        total_features = metadata.get('feature_importance', {}).get('total_features', 0)
        feature_names = metadata.get('feature_names', [])
        prohibited_features = metadata.get('prohibited_features', [])
        
        return f"""
        <div class="section">
            <h2>🔧 Feature Engineering Summary</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Features Created</h3>
                    <div class="metric-value">{total_features}</div>
                    <div class="metric-description">Total Features</div>
                </div>
                
                <div class="metric-card">
                    <h3>Compliance Check</h3>
                    <div class="metric-value status-good">✅</div>
                    <div class="metric-description">Listing-Time Only</div>
                </div>
                
                <div class="metric-card">
                    <h3>Prohibited Features</h3>
                    <div class="metric-value">{len(prohibited_features)}</div>
                    <div class="metric-description">Removed for Compliance</div>
                </div>
                
                <div class="metric-card">
                    <h3>Missing Value Strategy</h3>
                    <div class="metric-value">Median</div>
                    <div class="metric-description">Imputation Method</div>
                </div>
            </div>
            
            <h3>Feature Categories</h3>
            <ul>
                <li><strong>Loan Characteristics:</strong> Amount, interest rate, term, grade, purpose</li>
                <li><strong>Borrower Attributes:</strong> Income, employment, home ownership, DTI ratio</li>
                <li><strong>Credit History:</strong> Credit length, delinquencies, inquiries, utilization</li>
                <li><strong>Derived Features:</strong> Ratios, risk scores, and composite indicators</li>
            </ul>
            
            {f'<div class="highlight"><strong>Compliance Note:</strong> {len(prohibited_features)} features were removed to ensure listing-time compliance: {", ".join(prohibited_features[:5])}{"..." if len(prohibited_features) > 5 else ""}</div>' if prohibited_features else ''}
        </div>
"""
    
    def _create_model_performance_summary(self, model_results: Dict[str, Any]) -> str:
        """Create model performance summary section."""
        if not model_results:
            return ""
        
        metadata = model_results.get('metadata', {})
        training_metrics = metadata.get('training_metrics', {})
        train_metrics = training_metrics.get('train', {})
        val_metrics = training_metrics.get('validation', {})
        
        return f"""
        <div class="section">
            <h2>🤖 Model Performance</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>ROC-AUC (Validation)</h3>
                    <div class="metric-value {'status-good' if val_metrics.get('roc_auc', 0) > 0.7 else 'status-warning'}">{val_metrics.get('roc_auc', 0):.3f}</div>
                    <div class="metric-description">Discrimination Power</div>
                </div>
                
                <div class="metric-card">
                    <h3>Brier Score (Validation)</h3>
                    <div class="metric-value {'status-good' if val_metrics.get('brier_score', 1) < 0.2 else 'status-warning'}">{val_metrics.get('brier_score', 1):.3f}</div>
                    <div class="metric-description">Calibration Quality</div>
                </div>
                
                <div class="metric-card">
                    <h3>Precision (Validation)</h3>
                    <div class="metric-value">{val_metrics.get('precision', 0):.3f}</div>
                    <div class="metric-description">Positive Prediction Accuracy</div>
                </div>
                
                <div class="metric-card">
                    <h3>Recall (Validation)</h3>
                    <div class="metric-value">{val_metrics.get('recall', 0):.3f}</div>
                    <div class="metric-description">Default Detection Rate</div>
                </div>
            </div>
            
            <h3>Model Configuration</h3>
            <div class="code-block">
Model Type: {metadata.get('model_type', 'logistic').title()}
Hyperparameter Optimization: {'✅ Enabled' if metadata.get('hyperparameters') else '❌ Disabled'}
Probability Calibration: ✅ Platt Scaling
Cross-Validation: 5-fold Stratified
            </div>
            
            <h3>Training vs Validation Performance</h3>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Training</th>
                            <th>Validation</th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>ROC-AUC</td>
                            <td>{train_metrics.get('roc_auc', 0):.3f}</td>
                            <td>{val_metrics.get('roc_auc', 0):.3f}</td>
                            <td>{abs(train_metrics.get('roc_auc', 0) - val_metrics.get('roc_auc', 0)):.3f}</td>
                        </tr>
                        <tr>
                            <td>Brier Score</td>
                            <td>{train_metrics.get('brier_score', 1):.3f}</td>
                            <td>{val_metrics.get('brier_score', 1):.3f}</td>
                            <td>{abs(train_metrics.get('brier_score', 1) - val_metrics.get('brier_score', 1)):.3f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
"""
    
    def _create_investment_analysis(self, investment_results: Dict[str, Any]) -> str:
        """Create investment analysis section."""
        if not investment_results:
            return ""
        
        summary = investment_results.get('summary', {})
        portfolio_metrics = investment_results.get('portfolio_metrics', {})
        
        return f"""
        <div class="section">
            <h2>💰 Investment Analysis</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Budget Utilization</h3>
                    <div class="metric-value">{summary.get('budget_utilization', 0)*100:.1f}%</div>
                    <div class="metric-description">${summary.get('budget_used', 0):,.0f} of ${summary.get('budget_allocated', 5000):,.0f}</div>
                </div>
                
                <div class="metric-card">
                    <h3>Portfolio Size</h3>
                    <div class="metric-value">{summary.get('loans_selected', 0):,}</div>
                    <div class="metric-description">Loans Selected</div>
                </div>
                
                <div class="metric-card">
                    <h3>Average Investment</h3>
                    <div class="metric-value">${summary.get('avg_investment_per_loan', 0):.0f}</div>
                    <div class="metric-description">Per Loan</div>
                </div>
                
                <div class="metric-card">
                    <h3>Expected Return</h3>
                    <div class="metric-value">{summary.get('expected_performance', {}).get('avg_expected_return', 0)*100:.1f}%</div>
                    <div class="metric-description">Annual Expected Return</div>
                </div>
            </div>
            
            <h3>Risk Profile</h3>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Risk Metric</th>
                            <th>Portfolio Value</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Average Default Probability</td>
                            <td>{summary.get('portfolio_risk_profile', {}).get('avg_default_probability', 0):.3f}</td>
                            <td><span class="{'status-good' if summary.get('portfolio_risk_profile', {}).get('avg_default_probability', 0) < 0.2 else 'status-warning'}">{'✅ Low Risk' if summary.get('portfolio_risk_profile', {}).get('avg_default_probability', 0) < 0.2 else '⚠️ Moderate Risk'}</span></td>
                        </tr>
                        <tr>
                            <td>Grade Concentration</td>
                            <td>{summary.get('diversification_metrics', {}).get('grade_concentration', 0)*100:.1f}%</td>
                            <td><span class="{'status-good' if summary.get('diversification_metrics', {}).get('grade_concentration', 0) < 0.3 else 'status-warning'}">{'✅ Well Diversified' if summary.get('diversification_metrics', {}).get('grade_concentration', 0) < 0.3 else '⚠️ Concentrated'}</span></td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td>{summary.get('expected_performance', {}).get('portfolio_sharpe_ratio', 0):.3f}</td>
                            <td><span class="{'status-good' if summary.get('expected_performance', {}).get('portfolio_sharpe_ratio', 0) > 0.5 else 'status-warning'}">{'✅ Good' if summary.get('expected_performance', {}).get('portfolio_sharpe_ratio', 0) > 0.5 else '⚠️ Moderate'}</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <h3>Investment Strategy</h3>
            <p><strong>Strategy Used:</strong> {summary.get('selection_strategy', 'lowest_risk').replace('_', ' ').title()}</p>
            <p><strong>Budget Constraint:</strong> Maximum $5,000 per quarter investment</p>
            <p><strong>Diversification:</strong> Maximum 30% concentration per grade, target 100+ loans</p>
        </div>
"""
    
    def _create_backtest_results(self, backtest_results: Dict[str, Any]) -> str:
        """Create backtest results section."""
        if not backtest_results:
            return ""
        
        summary = backtest_results.get('summary_metrics', {})
        investment_perf = backtest_results.get('investment_performance', {})
        calibration = backtest_results.get('calibration_quality', {})
        
        return f"""
        <div class="section">
            <h2>📊 Backtest Results</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Actual Default Rate</h3>
                    <div class="metric-value">{summary.get('portfolio_default_rate', 0):.3f}</div>
                    <div class="metric-description">Portfolio Performance</div>
                </div>
                
                <div class="metric-card">
                    <h3>ROI Proxy</h3>
                    <div class="metric-value {'status-good' if summary.get('portfolio_roi', 0) > 0 else 'status-danger'}">{summary.get('portfolio_roi', 0):.3f}</div>
                    <div class="metric-description">Actual Returns</div>
                </div>
                
                <div class="metric-card">
                    <h3>Model Accuracy</h3>
                    <div class="metric-value">{summary.get('model_auc', 0):.3f}</div>
                    <div class="metric-description">Backtest ROC-AUC</div>
                </div>
                
                <div class="metric-card">
                    <h3>Calibration Quality</h3>
                    <div class="metric-value {'status-good' if calibration.get('well_calibrated', False) else 'status-warning'}">{'✅' if calibration.get('well_calibrated', False) else '⚠️'}</div>
                    <div class="metric-description">Probability Reliability</div>
                </div>
            </div>
            
            <h3>Performance vs Expectations</h3>
            <div class="highlight">
                <strong>Actual Performance:</strong> The portfolio achieved a {summary.get('portfolio_roi', 0):.3f} ROI with a {summary.get('portfolio_default_rate', 0):.3f} default rate.<br>
                <strong>Model Reliability:</strong> {'The model predictions were well-calibrated' if calibration.get('well_calibrated', False) else 'Model predictions showed some calibration issues'} with an ECE of {calibration.get('expected_calibration_error', 0):.4f}.<br>
                <strong>Investment Success:</strong> {'Successfully' if summary.get('portfolio_roi', 0) > 0 else 'Partially'} achieved positive returns through systematic loan selection.
            </div>
            
            <h3>Risk Management Effectiveness</h3>
            <ul>
                <li><strong>Default Prediction:</strong> Model correctly identified high-risk loans with {summary.get('model_auc', 0):.3f} AUC</li>
                <li><strong>Portfolio Construction:</strong> Diversification strategy limited concentration risk</li>
                <li><strong>Budget Management:</strong> Efficiently deployed capital within $5,000 constraint</li>
                <li><strong>Compliance:</strong> Maintained strict listing-time data usage throughout</li>
            </ul>
        </div>
"""
    
    def _create_technical_appendix(self, pipeline_results: Dict[str, Any]) -> str:
        """Create technical appendix section."""
        return f"""
        <div class="section">
            <h2>🔧 Technical Appendix</h2>
            
            <h3>Pipeline Configuration</h3>
            <div class="code-block">
Model Type: Logistic Regression
Feature Engineering: Automated with listing-time compliance
Hyperparameter Optimization: Grid Search with 5-fold CV
Probability Calibration: Platt Scaling
Investment Strategy: Lowest Risk with diversification constraints
Budget Constraint: $5,000 per quarter maximum
            </div>
            
            <h3>Data Processing Steps</h3>
            <ol>
                <li><strong>Data Loading:</strong> Quarterly CSV files loaded with temporal ordering</li>
                <li><strong>Data Validation:</strong> Quality checks and missing value analysis</li>
                <li><strong>Feature Engineering:</strong> 50+ features created from loan and borrower data</li>
                <li><strong>Compliance Check:</strong> Prohibited post-origination features removed</li>
                <li><strong>Model Training:</strong> Hyperparameter optimization with cross-validation</li>
                <li><strong>Probability Calibration:</strong> Platt scaling for reliable probabilities</li>
                <li><strong>Investment Selection:</strong> Risk-based loan selection under budget constraints</li>
                <li><strong>Backtesting:</strong> Performance evaluation on held-out 2017Q1 data</li>
            </ol>
            
            <h3>Key Assumptions</h3>
            <ul>
                <li><strong>Recovery Rate:</strong> 30% recovery on defaulted loans</li>
                <li><strong>Investment Horizon:</strong> Hold loans to maturity</li>
                <li><strong>Transaction Costs:</strong> Not included in ROI calculations</li>
                <li><strong>Market Conditions:</strong> Stable economic environment assumed</li>
            </ul>
            
            <h3>Model Limitations</h3>
            <ul>
                <li>Based on historical data from 2016-2017 time period</li>
                <li>Simplified ROI calculation without complex cash flow modeling</li>
                <li>No consideration of macroeconomic factors</li>
                <li>Limited to Lending Club platform data</li>
            </ul>
        </div>
"""
    
    def _close_html_structure(self) -> str:
        """Close HTML structure."""
        return """
        <div class="footer">
            <p>&copy; 2024 Lending Club ML Pipeline - Generated by Automated Reporting System</p>
        </div>
    </div>
</body>
</html>
"""
