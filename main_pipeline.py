#!/usr/bin/env python3
"""
Lending Club ML Pipeline - Main Execution Script
=================================================

This script orchestrates the complete machine learning pipeline for 
predicting loan default risk and optimizing investment decisions.

Features:
- Loads and validates quarterly loan data
- Engineers features with listing-time compliance
- Trains and calibrates ML models
- Makes investment decisions under budget constraints
- Performs backtesting on held-out data
- Comprehensive logging and progress tracking

Usage:
    python main_pipeline.py [--config config/pipeline_config.yaml]
"""

import argparse
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Import our custom utilities
from src.utils.logging_config import setup_pipeline_logging, log_execution
from src.utils.progress_tracker import PipelineProgressTracker as ProgressTracker
from src.utils.validation import validate_pipeline_data

# Import pipeline components (we'll implement these)
from src.lending_club.data_pipeline import DataLoader, DataValidator
from src.lending_club.feature_pipeline import FeatureEngineer
from src.lending_club.model_pipeline import ModelTrainer, ModelCalibrator
from src.lending_club.investment_pipeline import InvestmentDecisionMaker
from src.lending_club.evaluation_pipeline import BacktestEvaluator


class PipelineConfig:
    """Configuration loader and validator for the ML pipeline."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['data', 'features', 'model', 'investment', 'evaluation']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config['data']
        if not data_config.get('train_quarters'):
            raise ValueError("No training quarters specified in configuration")
        
        if not data_config.get('validation_quarter'):
            raise ValueError("No validation quarter specified in configuration")
        
        # Validate budget constraint
        investment_config = self.config['investment']
        if investment_config.get('budget_per_quarter', 0) <= 0:
            raise ValueError("Budget per quarter must be positive")
        
        logging.info("Configuration validation completed successfully")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        return self.config.get(section, {})


class MLPipeline:
    """Main ML Pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config = PipelineConfig(config_path)
        self.logger = None
        self.progress = None
        self.results = {}
        
    @log_execution
    def setup_infrastructure(self) -> None:
        """Setup logging, progress tracking, and output directories."""
        # Setup comprehensive logging system
        log_config = self.config.get_section('logging')
        self.logger = setup_pipeline_logging(
            log_directory=log_config.get('log_directory', 'logs'),
            log_level=log_config.get('level', 'INFO')
        )
        
        # Create output directories
        output_config = self.config.get_section('output')
        for dir_key in ['models_directory', 'figures_directory', 'reports_directory', 'data_directory']:
            directory = Path(output_config.get(dir_key, f'outputs/{dir_key.replace("_directory", "")}'))
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup progress tracking
        self.progress = ProgressTracker(total_steps=6, description="Lending Club ML Pipeline")
        
        logging.info("Infrastructure setup completed")
    
    @log_execution
    def load_and_validate_data(self) -> Dict[str, Any]:
        """Load and validate quarterly loan data."""
        data_config = self.config.get_section('data')
        validation_config = self.config.get_section('validation')
        
        # Initialize data loader
        data_loader = DataLoader(data_config)
        data_validator = DataValidator(validation_config)
        
        # Load training data
        train_data = data_loader.load_quarterly_data(data_config['train_quarters'])
        validation_data = data_loader.load_quarterly_data([data_config['validation_quarter']])
        backtest_data = data_loader.load_quarterly_data([data_config['backtest_quarter']])
        
        # Validate data quality and stop if any dataset fails
        print("🔍 Validating data quality...")
        validation_config = self.config.get_section('validation')
        min_quality_score = validation_config.get('min_quality_score', 0.8)

        for dataset_name, dataset in [
            ('training', train_data),
            ('validation', validation_data),
            ('backtest', backtest_data)
        ]:
            quality_report = data_validator.validate_data_quality(dataset)
            quality_score = quality_report.get('overall_quality_score', 0)
            passed_validation = quality_report.get('passed_validation', False)

            if passed_validation:
                print(f"   ✅ {dataset_name.title()} data quality PASSED (score: {quality_score:.3f})")
                logging.info(f"{dataset_name.title()} data quality PASSED: {quality_score:.3f}")
            else:
                error_msg = (f"❌ {dataset_name.title()} data quality FAILED (score: {quality_score:.3f}, "
                           f"minimum required: {min_quality_score:.3f})")
                print(f"   {error_msg}")
                logging.error(error_msg)
                raise ValueError(f"Data quality validation failed for {dataset_name} dataset. "
                               f"Score: {quality_score:.3f}, Required: {min_quality_score:.3f}")

        print("✅ All datasets passed quality validation")

        # Combine and return data dictionary
        data_dict = {
            'train': train_data,
            'validation': validation_data,
            'backtest': backtest_data,
            'quality_reports': {
                'train': data_validator.validate_data_quality(train_data),
                'validation': data_validator.validate_data_quality(validation_data),
                'backtest': data_validator.validate_data_quality(backtest_data)
            }
        }
        
        logging.info(f"Data loading completed - Train: {len(train_data)}, "
                    f"Validation: {len(validation_data)}, Backtest: {len(backtest_data)} rows")
        
        return data_dict
    
    @log_execution
    def engineer_features(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features with listing-time compliance."""
        feature_config = self.config.get_section('features')
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(feature_config)
        
        # Engineer features for each dataset with alignment
        engineered_data = {}

        # First, create training features to establish the feature schema
        logging.info("Creating training features to establish schema...")
        train_data = data_dict['train']
        train_features, train_target = feature_engineer.create_features(train_data)
        training_feature_names = feature_engineer.training_feature_names

        engineered_data['train'] = {
            'features': train_features,
            'target': train_target,
            'raw_data': train_data
        }
        logging.info(f"Training features finalized: {len(training_feature_names)} columns")

        # Now create and align validation and backtest features
        for dataset_name in ['validation', 'backtest']:
            raw_data = data_dict[dataset_name]
            logging.info(f"Creating and aligning {dataset_name} features...")

            # Create features and align to training feature set
            features, target = feature_engineer.create_features(raw_data, align_features=training_feature_names)

            engineered_data[dataset_name] = {
                'features': features,
                'target': target,
                'raw_data': raw_data
            }
            logging.info(f"{dataset_name.title()} features aligned: {len(features.columns)} columns")

            logging.info(f"{dataset_name.title()} features engineered: "
                        f"{features.shape[1]} features, {len(features)} samples")
        
        # Store feature metadata
        engineered_data['metadata'] = {
            'feature_names': feature_engineer.get_feature_names(),
            'feature_importance': feature_engineer.get_feature_metadata(),
            'prohibited_features': feature_engineer.get_prohibited_features()
        }
        
        return engineered_data
    
    @log_execution  
    def train_and_calibrate_model(self, engineered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train and calibrate the ML model."""
        model_config = self.config.get_section('model')
        
        # Initialize model trainer and calibrator
        model_trainer = ModelTrainer(model_config)
        model_calibrator = ModelCalibrator(model_config)
        
        # Prepare training data
        X_train = engineered_data['train']['features']
        y_train = engineered_data['train']['target']
        X_val = engineered_data['validation']['features']
        y_val = engineered_data['validation']['target']
        
        # Train model with hyperparameter optimization
        model = model_trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Calibrate probabilities
        calibrated_model = model_calibrator.calibrate_model(model, X_val, y_val)
        
        # Generate model metadata
        model_metadata = {
            'model_type': model_config.get('type', 'logistic'),
            'hyperparameters': model_trainer.get_best_params(),
            'training_metrics': model_trainer.get_training_metrics(),
            'calibration_metrics': model_calibrator.get_calibration_metrics(),
            'feature_importance': model_trainer.get_feature_importance()
        }
        
        # Save model artifacts
        output_config = self.config.get_section('output')
        model_trainer.save_model_artifacts(
            calibrated_model, 
            model_metadata,
            Path(output_config.get('models_directory', 'outputs/models'))
        )
        
        return {
            'model': calibrated_model,
            'metadata': model_metadata
        }
    
    @log_execution
    def make_investment_decisions(self, model_dict: Dict[str, Any], 
                                engineered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investment decisions under budget constraints."""
        investment_config = self.config.get_section('investment')
        
        # Initialize investment decision maker
        decision_maker = InvestmentDecisionMaker(investment_config)
        
        # Make decisions for validation set
        model = model_dict['model']
        validation_features = engineered_data['validation']['features']
        validation_raw = engineered_data['validation']['raw_data']
        
        # Get model predictions
        risk_scores = model.predict_proba(validation_features)[:, 1]  # Probability of default
        
        # Apply investment policy
        investment_decisions = decision_maker.select_investments(
            risk_scores=risk_scores,
            loan_data=validation_raw,
            budget=investment_config.get('budget_per_quarter', 5000.0)
        )
        
        # Generate decision summary
        decision_summary = decision_maker.generate_decision_summary(investment_decisions)
        
        logging.info(f"Investment decisions generated: "
                    f"{len(investment_decisions['selected_loans'])} loans selected, "
                    f"${decision_summary['total_investment']:.2f} invested")
        
        return {
            'decisions': investment_decisions,
            'summary': decision_summary,
            'risk_scores': risk_scores
        }
    
    @log_execution
    def run_backtest(self, model_dict: Dict[str, Any], 
                    engineered_data: Dict[str, Any],
                    investment_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Perform backtesting on held-out data."""
        evaluation_config = self.config.get_section('evaluation')
        
        # Initialize backtest evaluator
        backtest_evaluator = BacktestEvaluator(evaluation_config)
        
        # Prepare backtest data
        model = model_dict['model']
        backtest_features = engineered_data['backtest']['features']
        backtest_target = engineered_data['backtest']['target']
        backtest_raw = engineered_data['backtest']['raw_data']
        
        # Generate predictions for backtest period
        backtest_predictions = model.predict_proba(backtest_features)[:, 1]
        
        # Apply investment strategy to backtest period
        investment_config = self.config.get_section('investment')
        decision_maker = InvestmentDecisionMaker(investment_config)
        
        backtest_investments = decision_maker.select_investments(
            risk_scores=backtest_predictions,
            loan_data=backtest_raw,
            budget=investment_config.get('budget_per_quarter', 5000.0)
        )
        
        # Evaluate backtest performance
        backtest_results = backtest_evaluator.evaluate_backtest(
            predictions=backtest_predictions,
            actual_outcomes=backtest_target,
            investment_decisions=backtest_investments,
            loan_data=backtest_raw
        )
        
        # Generate evaluation plots
        output_config = self.config.get_section('output')
        figures_dir = Path(output_config.get('figures_directory', 'outputs/figures'))
        
        backtest_evaluator.generate_evaluation_plots(
            backtest_results, 
            investment_decisions['summary'],
            figures_dir
        )
        
        logging.info(f"Backtest completed - ROI: {backtest_results.get('roi_proxy', 'N/A'):.3f}, "
                    f"Default Rate: {backtest_results.get('default_rate', 'N/A'):.3f}")
        
        return backtest_results
    
    @log_execution
    def generate_final_report(self, all_results: Dict[str, Any]) -> None:
        """Generate comprehensive final report."""
        output_config = self.config.get_section('output')
        reports_dir = Path(output_config.get('reports_directory', 'outputs/reports'))
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime(output_config.get('timestamp_format', '%Y-%m-%d_%H-%M-%S'))
        
        # Generate HTML report
        report_path = reports_dir / f"pipeline_report_{timestamp}.html"
        
        # Create comprehensive report content
        from src.lending_club.reporting import ReportGenerator
        
        report_generator = ReportGenerator(self.config.config)
        report_generator.generate_comprehensive_report(
            all_results, 
            report_path
        )
        
        logging.info(f"Final report generated: {report_path}")
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete ML pipeline."""
        try:
            # Step 1: Infrastructure Setup (outside progress context)
            self.setup_infrastructure()

            # Now use progress tracking for the rest of the pipeline
            with self.progress:
                print("\n📊 Pipeline Progress:")
                print("=" * 50)

                # Step 2: Data Loading and Validation
                print("📥 Step 2/6: Loading and validating data...")
                logging.info("Starting data loading and validation phase")
                self.progress.update(step_description="Loading and validating data...")
                data_dict = self.load_and_validate_data()
                self.results['data'] = data_dict
                print(f"   ✅ Data loaded: {data_dict['train'].shape[0]} train, {data_dict['validation'].shape[0]} validation, {data_dict['backtest'].shape[0]} backtest samples")
                logging.info(f"Data loading completed: {data_dict['train'].shape[0]} train, {data_dict['validation'].shape[0]} validation, {data_dict['backtest'].shape[0]} backtest samples")

                # Step 3: Feature Engineering
                print("🔧 Step 3/6: Engineering features...")
                logging.info("Starting feature engineering phase")
                self.progress.update(step_description="Engineering features...")
                engineered_data = self.engineer_features(data_dict)
                self.results['features'] = engineered_data
                train_features = engineered_data['train']['features']
                print(f"   ✅ Features engineered: {train_features.shape[1]} features from {train_features.shape[0]} samples")
                logging.info(f"Feature engineering completed: {train_features.shape[1]} features from {train_features.shape[0]} samples")

                # Step 4: Model Training and Calibration
                print("🤖 Step 4/6: Training and calibrating model...")
                logging.info("Starting model training and calibration phase")
                self.progress.update(step_description="Training and calibrating model...")
                model_dict = self.train_and_calibrate_model(engineered_data)
                self.results['model'] = model_dict
                model_type = model_dict.get('metadata', {}).get('model_type', 'unknown')
                auc_score = model_dict.get('metadata', {}).get('training_metrics', {}).get('validation_auc', 'N/A')
                print(f"   ✅ Model trained: {model_type} with AUC = {auc_score}")
                logging.info(f"Model training completed: {model_type} with AUC = {auc_score}")

                # Step 5: Investment Decisions
                print("💰 Step 5/6: Making investment decisions...")
                logging.info("Starting investment decision phase")
                self.progress.update(step_description="Making investment decisions...")
                investment_decisions = self.make_investment_decisions(model_dict, engineered_data)
                self.results['investment'] = investment_decisions
                print(f"   ✅ Investment decisions made: {len(investment_decisions.get('selected_loans', []))} loans selected")
                logging.info(f"Investment decisions completed: {len(investment_decisions.get('selected_loans', []))} loans selected")

                # Step 6: Backtesting
                print("📈 Step 6/6: Running backtest evaluation...")
                logging.info("Starting backtest evaluation phase")
                self.progress.update(step_description="Running backtest evaluation...")
                backtest_results = self.run_backtest(model_dict, engineered_data, investment_decisions)
                self.results['backtest'] = backtest_results
                roi = backtest_results.get('overall_roi', 0) * 100
                print(f"   ✅ Backtest completed: {roi:.1f}% ROI")
                logging.info(f"Backtest evaluation completed: {roi:.1f}% ROI")
            
            # Generate final report
            self.generate_final_report(self.results)
            
            logging.info("🎉 Pipeline completed successfully!")
            
            # Print summary to console
            self._print_pipeline_summary()
            
            return self.results
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _print_pipeline_summary(self) -> None:
        """Print a summary of pipeline results to console."""
        print("\n" + "="*70)
        print("🎯 LENDING CLUB ML PIPELINE - EXECUTION SUMMARY")
        print("="*70)
        
        # Data summary
        data_results = self.results.get('data', {})
        if data_results:
            print(f"📊 Data Loaded:")
            print(f"  • Training: {len(data_results.get('train', []))} loans")
            print(f"  • Validation: {len(data_results.get('validation', []))} loans") 
            print(f"  • Backtest: {len(data_results.get('backtest', []))} loans")
        
        # Model summary
        model_results = self.results.get('model', {})
        if model_results and 'metadata' in model_results:
            metadata = model_results['metadata']
            print(f"\n🤖 Model Performance:")
            training_metrics = metadata.get('training_metrics', {})
            for metric, value in training_metrics.items():
                print(f"  • {metric}: {value:.4f}")
        
        # Investment summary
        investment_results = self.results.get('investment', {})
        if investment_results and 'summary' in investment_results:
            summary = investment_results['summary']
            print(f"\n💰 Investment Decisions:")
            print(f"  • Loans Selected: {summary.get('loans_selected', 0)}")
            print(f"  • Total Investment: ${summary.get('total_investment', 0):.2f}")
            print(f"  • Average Risk Score: {summary.get('avg_risk_score', 0):.4f}")
        
        # Backtest summary
        backtest_results = self.results.get('backtest', {})
        if backtest_results:
            print(f"\n📈 Backtest Results:")
            print(f"  • Default Rate: {backtest_results.get('default_rate', 0):.3f}")
            print(f"  • ROI Proxy: {backtest_results.get('roi_proxy', 0):.3f}")
            print(f"  • Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.3f}")
        
        print(f"\n📁 Outputs saved to:")
        print(f"  • Models: outputs/models/")
        print(f"  • Figures: outputs/figures/") 
        print(f"  • Reports: outputs/reports/")
        print(f"  • Logs: logs/")
        
        print("="*70)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Lending Club ML Pipeline - Predicting Default Risk and Optimizing Investments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pipeline.py                                    # Use default config
  python main_pipeline.py --config config/pipeline_config.yaml  # Use specific config
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config/pipeline_config.yaml',
        help='Path to pipeline configuration file (default: config/pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Create and run pipeline
        pipeline = MLPipeline(config_path=args.config)
        results = pipeline.run()
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Check outputs/ directory for results")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
