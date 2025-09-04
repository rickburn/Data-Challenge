#!/usr/bin/env python3
"""
Demonstration of Comprehensive Logging and Progress Tracking System.

This script showcases the complete logging and progress tracking capabilities
implemented according to cursor rules standards.

Usage:
    python examples/demo_logging_and_progress.py
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging_config import (
    setup_pipeline_logging, 
    log_execution,
    track_data_transformation
)
from utils.progress_tracker import (
    PipelineProgressTracker,
    BatchProgressTracker,
    track_operation,
    PerformanceMonitor
)
from utils.validation import (
    validate_pipeline_data,
    DataQualityValidator
)


def demonstrate_logging_system():
    """Demonstrate comprehensive logging capabilities."""
    print("🔥 Setting up comprehensive logging system...")
    
    # Initialize logging system
    logger_system = setup_pipeline_logging(
        log_directory="logs",
        log_level="INFO"
    )
    
    print(f"✅ Logging initialized with execution ID: {logger_system.execution_id}")
    print(f"📁 Log files created in: {logger_system.log_directory}")
    
    return logger_system


@log_execution
def sample_data_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample data processing function with automatic logging.
    
    This function demonstrates how the @log_execution decorator
    automatically captures function entry, execution time, and results.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data processing transformation")
    
    # Simulate some processing time
    time.sleep(0.1)
    
    # Create processed version
    processed_df = df.copy()
    processed_df['processed_column'] = processed_df.iloc[:, 0] * 2
    processed_df['timestamp'] = datetime.now()
    
    # This will be automatically tracked for data lineage
    track_data_transformation(
        operation="sample_data_processing",
        input_data=df,
        output_data=processed_df,
        metadata={'processing_time': 0.1, 'transformation': 'multiplication_by_2'}
    )
    
    logger.info("Data processing transformation completed")
    
    return processed_df


def demonstrate_progress_tracking():
    """Demonstrate advanced progress tracking capabilities."""
    print("\n🚀 Demonstrating Progress Tracking System...")
    
    # Basic progress tracking
    print("\n1️⃣ Basic Progress Tracking:")
    with track_operation("Basic Data Processing", total_steps=20) as progress:
        for i in range(20):
            time.sleep(0.05)  # Simulate work
            progress.update(step_description=f"Processing item {i+1}")
    
    # Nested progress tracking
    print("\n2️⃣ Nested Progress Tracking:")
    with track_operation("Complex ML Pipeline", total_steps=3) as main_progress:
        
        # Stage 1: Data Loading
        main_progress.update(step_description="Starting data loading")
        with main_progress.create_nested_tracker(15, "Loading Data") as load_progress:
            for i in range(15):
                time.sleep(0.02)
                load_progress.update(step_description=f"Loading file {i+1}")
        
        # Stage 2: Feature Engineering  
        main_progress.update(step_description="Starting feature engineering")
        with main_progress.create_nested_tracker(25, "Feature Engineering") as feat_progress:
            for i in range(25):
                time.sleep(0.01)
                feat_progress.update(step_description=f"Creating feature {i+1}")
                
        # Stage 3: Model Training
        main_progress.update(step_description="Starting model training")
        time.sleep(0.2)  # Simulate training
    
    # Batch processing
    print("\n3️⃣ Batch Processing:")
    
    def process_data_batch(start_idx: int, end_idx: int) -> dict:
        """Process a batch of data."""
        time.sleep(0.05)  # Simulate processing
        return {
            'processed_items': end_idx - start_idx,
            'batch_start': start_idx,
            'batch_end': end_idx
        }
    
    batch_tracker = BatchProgressTracker(
        total_items=500,
        batch_size=25,
        description="Batch Data Processing"
    )
    
    results = batch_tracker.process_batch(process_data_batch)
    print(f"✅ Processed {len(results)} batches successfully")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n⚡ Demonstrating Performance Monitoring...")
    
    # Monitor a data-intensive operation
    with PerformanceMonitor("Data Processing Operation"):
        # Create large dataset
        data = pd.DataFrame({
            'col1': np.random.randn(50000),
            'col2': np.random.randn(50000), 
            'col3': np.random.randn(50000)
        })
        
        # Perform computationally expensive operations
        result = data.groupby(pd.cut(data['col1'], bins=10)).agg({
            'col2': ['mean', 'std', 'count'],
            'col3': ['min', 'max', 'median']
        })
        
        # More processing
        correlation_matrix = data.corr()
        
        time.sleep(0.1)  # Simulate additional processing
        
    print("✅ Performance monitoring completed - check logs for detailed metrics")


def demonstrate_data_validation():
    """Demonstrate comprehensive data validation."""
    print("\n🔍 Demonstrating Data Validation System...")
    
    # Create sample datasets
    np.random.seed(42)
    
    # Training data (older dates)
    train_dates = pd.date_range('2016-01-01', '2016-09-30', freq='D')
    train_data = pd.DataFrame({
        'issue_d': np.random.choice(train_dates, 1000),
        'loan_amnt': np.random.normal(15000, 5000, 1000),
        'int_rate': np.random.normal(0.12, 0.05, 1000),
        'annual_inc': np.random.normal(70000, 30000, 1000),
        'loan_status': ['Current'] * 800 + ['Default'] * 200  # Prohibited field!
    })
    
    # Add some missing values and outliers
    train_data.loc[50:100, 'annual_inc'] = np.nan
    train_data.loc[0:5, 'loan_amnt'] = 1000000  # Outliers
    
    # Validation data (newer dates)
    val_dates = pd.date_range('2016-10-01', '2016-12-31', freq='D')  
    val_data = pd.DataFrame({
        'issue_d': np.random.choice(val_dates, 300),
        'loan_amnt': np.random.normal(15000, 5000, 300),
        'int_rate': np.random.normal(0.12, 0.05, 300),
        'annual_inc': np.random.normal(70000, 30000, 300),
        'loan_status': ['Current'] * 250 + ['Default'] * 50
    })
    
    # Feature names including prohibited ones
    feature_names = [
        'loan_amnt', 'int_rate', 'annual_inc',  # Valid features
        'loan_status', 'last_pymnt_amnt'        # Prohibited features
    ]
    
    # Run comprehensive validation
    validation_results = validate_pipeline_data(train_data, val_data, feature_names)
    
    print("\n📊 Validation Results Summary:")
    print("=" * 50)
    
    for validation_type, result in validation_results.items():
        status = "✅ PASS" if result.is_valid else "❌ FAIL" 
        print(f"{validation_type.upper()}: {status}")
        
        if result.errors:
            print("  🚨 ERRORS:")
            for error in result.errors:
                print(f"    • {error}")
                
        if result.warnings:
            print("  ⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"    • {warning}")
                
        print(f"  📈 Key Metrics: {result.metrics}")
        print()


def demonstrate_complete_pipeline():
    """Demonstrate complete pipeline with logging, progress, and validation."""
    print("\n🎯 Demonstrating Complete Pipeline Integration...")
    
    with track_operation("Complete ML Pipeline Demo", total_steps=5) as pipeline_progress:
        
        # Step 1: Data Generation
        pipeline_progress.update(step_description="Generating sample data")
        
        with PerformanceMonitor("Data Generation"):
            sample_data = pd.DataFrame({
                'feature_1': np.random.randn(10000),
                'feature_2': np.random.randn(10000),
                'feature_3': np.random.randn(10000),
                'target': np.random.choice([0, 1], 10000)
            })
        
        # Step 2: Data Processing with logging
        pipeline_progress.update(step_description="Processing data")
        processed_data = sample_data_processing(sample_data)
        
        # Step 3: Data Quality Validation
        pipeline_progress.update(step_description="Validating data quality")
        validator = DataQualityValidator()
        quality_result = validator.validate_data_quality(processed_data, "processed_sample")
        
        if quality_result.is_valid:
            print("✅ Data quality validation passed")
        else:
            print("❌ Data quality issues detected")
            
        # Step 4: Feature Engineering Simulation
        pipeline_progress.update(step_description="Engineering features")
        with track_operation("Feature Engineering", total_steps=10, enable_progress_bar=False) as feat_progress:
            for i in range(10):
                time.sleep(0.02)
                feat_progress.update(f"Creating feature {i+1}")
        
        # Step 5: Model Training Simulation  
        pipeline_progress.update(step_description="Training model")
        with PerformanceMonitor("Model Training Simulation"):
            # Simulate model training
            time.sleep(0.2)
            
            # Simulate model performance metrics
            model_metrics = {
                'roc_auc': 0.72,
                'brier_score': 0.18,
                'calibration_slope': 1.05,
                'calibration_intercept': -0.02
            }
    
    print("\n🏁 Complete pipeline demonstration finished!")
    print("📋 Check the logs/ directory for detailed execution logs")
    print("📊 All operations have been tracked with full traceability")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("🎪 LENDING CLUB ML PIPELINE - LOGGING & PROGRESS DEMO")
    print("=" * 70)
    
    # Set up logging system
    logger_system = demonstrate_logging_system()
    
    # Demonstrate each capability
    demonstrate_progress_tracking()
    demonstrate_performance_monitoring() 
    demonstrate_data_validation()
    demonstrate_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Execution logs available at: {logger_system.log_directory}")
    print(f"🆔 Execution ID: {logger_system.execution_id}")
    print("\nKey files generated:")
    print("  • logs/pipeline_execution.log - Main execution log")
    print("  • logs/operations/data_operations.jsonl - Data operation tracking") 
    print("  • logs/operations/model_training.jsonl - Model operation tracking")
    print("  • logs/performance/performance_metrics.jsonl - Performance metrics")
    print("  • logs/data_lineage.jsonl - Complete data lineage tracking")
    print("\n🔍 Review these files to see the comprehensive logging in action!")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("examples", exist_ok=True)
    
    # Run demonstration
    main()
