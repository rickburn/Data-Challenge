#!/usr/bin/env python3
"""
Lending Club ML Pipeline Runner
===============================

Simple runner script for executing the complete ML pipeline.
This script provides a user-friendly interface and handles common issues.

Usage:
    python run_pipeline.py                    # Run with default settings
    python run_pipeline.py --debug            # Run with debug logging
    python run_pipeline.py --dry-run          # Validate setup without execution
"""

import sys
import argparse
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages (import_name, display_name)
    required_packages = [
        ('pandas', 'pandas'), ('numpy', 'numpy'), ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'), ('seaborn', 'seaborn'), ('pydantic', 'pydantic'), ('yaml', 'yaml')
    ]

    for import_name, display_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            issues.append(f"Missing required package: {display_name}")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        issues.append("Data directory 'data/' not found")
    else:
        required_files = ['2016Q1.csv', '2016Q2.csv', '2016Q3.csv', '2016Q4.csv', '2017Q1.csv']
        for file in required_files:
            if not (data_dir / file).exists():
                issues.append(f"Required data file missing: {file}")
    
    # Check config file
    config_file = Path("config/pipeline_config.yaml")
    if not config_file.exists():
        issues.append("Configuration file 'config/pipeline_config.yaml' not found")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Please resolve these issues before running the pipeline.")
        return False
    
    print("✅ All requirements met!")
    return True

def setup_environment():
    """Setup environment for pipeline execution."""
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Create output directories
    output_dirs = ["outputs", "outputs/models", "outputs/figures", "outputs/reports", "outputs/data", "logs", "logs/daily", "logs/operations", "logs/performance"]
    for dir_name in output_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    # Clear log files for fresh run
    print("🧹 Clearing previous log files...")
    log_files = [
        "logs/pipeline_execution.log",
        "logs/data_lineage.jsonl",
        "logs/operations/data_operations.jsonl",
        "logs/operations/model_training.jsonl",
        "logs/performance/performance_metrics.jsonl"
    ]

    for log_file in log_files:
        try:
            Path(log_file).unlink(missing_ok=True)
            print(f"   ✅ Cleared {log_file}")
        except Exception as e:
            print(f"   ⚠️  Could not clear {log_file}: {e}")

    print("✅ Environment setup complete\n")
    print("📋 Log Files:")
    print(f"   📄 Comprehensive log: logs/pipeline_comprehensive.log")
    print(f"   📊 Data lineage: logs/data_lineage.jsonl")
    print(f"   ⚡ Performance: logs/performance/performance_metrics.jsonl")
    print(f"   🔧 Operations: logs/operations/")
    print()

    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print("🎮 GPU Status:")
            print(f"   ✅ GPU acceleration available: {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print("   💡 To enable GPU: Set 'use_gpu: true' in config/pipeline_config.yaml")
        else:
            print("🎮 GPU Status:")
            print("   ⚠️  No GPU detected - using CPU only")
    except ImportError:
        print("🎮 GPU Status:")
        print("   ⚠️  PyTorch not available for GPU detection")
        print("   💡 Install PyTorch to enable GPU detection")

    print()

def main():
    """Main runner function."""
    parser = argparse.ArgumentParser(
        description="Run the Lending Club ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Standard execution
  python run_pipeline.py --debug            # Debug mode with verbose logging
  python run_pipeline.py --dry-run          # Check setup without running
  python run_pipeline.py --gpu              # Enable GPU acceleration
  python run_pipeline.py --threads 4        # Use max 4 threads
  python run_pipeline.py --gpu --threads 8  # GPU + 8 threads
        """
    )
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Check setup without executing pipeline')
    parser.add_argument('--config', default='config/pipeline_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration (if available)')
    parser.add_argument('--threads', type=int, default=None,
                       help='Maximum number of threads for training (default: 8)')
    
    args = parser.parse_args()
    
    print("🚀 Lending Club ML Pipeline Runner")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    if args.dry_run:
        print("✅ Dry run completed - setup is valid!")
        return 0
    
    try:
        # Import and run pipeline
        print("\n📦 Loading pipeline components...")
        from main_pipeline import MLPipeline

        print("🏗️  Initializing pipeline...")
        pipeline = MLPipeline(config_path=args.config)

        # Override configuration with command-line arguments
        if args.gpu or args.threads:
            print("⚙️  Applying command-line overrides...")
            if args.gpu:
                pipeline.config.config['model']['hardware']['use_gpu'] = True
                print("   ✅ GPU acceleration enabled via --gpu flag")
            if args.threads:
                pipeline.config.config['model']['hardware']['max_threads'] = args.threads
                print(f"   ✅ Max threads set to {args.threads} via --threads flag")

        print("▶️  Starting pipeline execution...")
        results = pipeline.run()
        
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the outputs/ directory for results")
        print("📊 View the HTML report in outputs/reports/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️  Pipeline execution interrupted by user")
        return 130
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install -r requirements.txt")
        return 1
    
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("💡 Check that all data files are in the correct locations")
        return 1
    
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print("💡 Run with --debug flag for detailed error information")
        return 1

if __name__ == "__main__":
    sys.exit(main())
