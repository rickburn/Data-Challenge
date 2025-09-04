# Examples Directory

This directory contains example scripts demonstrating the comprehensive logging, progress tracking, and validation systems implemented for the Lending Club ML Pipeline.

## Available Examples

### 1. `demo_logging_and_progress.py`

Comprehensive demonstration of all logging and progress tracking capabilities:

- **Comprehensive Logging System**: Shows how to set up structured logging with multiple handlers
- **Progress Tracking**: Demonstrates basic, nested, and batch progress tracking
- **Performance Monitoring**: Shows memory and timing monitoring
- **Data Validation**: Complete data quality and compliance validation
- **Complete Pipeline Integration**: End-to-end example combining all systems

#### Running the Demo

```bash
# From the project root directory
python examples/demo_logging_and_progress.py
```

#### Expected Output

The demo will:
1. ✅ Set up comprehensive logging system
2. 🚀 Show various progress tracking methods
3. ⚡ Demonstrate performance monitoring
4. 🔍 Run data validation with sample data
5. 🎯 Execute complete integrated pipeline

#### Generated Files

After running, check these directories:
- `logs/` - All execution logs with complete traceability
- `logs/pipeline_execution.log` - Main execution log
- `logs/operations/` - Structured operation logs (JSON format)
- `logs/performance/` - Performance metrics
- `logs/data_lineage.jsonl` - Complete data transformation lineage

## Features Demonstrated

### Logging Capabilities
- ✅ Multiple log handlers (console, file, structured JSON)
- ✅ Automatic function execution logging with `@log_execution` decorator
- ✅ Data lineage tracking for all transformations
- ✅ Performance metrics capture
- ✅ Error handling with full context

### Progress Tracking
- ✅ Basic progress bars with ETA calculation
- ✅ Nested progress tracking for complex operations
- ✅ Batch processing with automatic progress
- ✅ Memory usage and timing monitoring
- ✅ Integration with logging system

### Data Validation
- ✅ Comprehensive data quality checks
- ✅ Temporal constraint validation (critical for ML pipelines)
- ✅ Feature compliance validation (listing-time rule enforcement)
- ✅ Model performance validation
- ✅ Structured validation reporting

### Integration Features
- ✅ Complete traceability of all operations
- ✅ Automatic checkpoint creation
- ✅ Performance bottleneck detection
- ✅ Configuration-driven validation thresholds
- ✅ Error recovery and detailed error reporting

## Cursor Rules Compliance

All examples follow the cursor rules standards:

- **Type Hints**: All functions have complete type annotations
- **Docstrings**: Comprehensive numpy-style documentation
- **Logging**: Every operation is logged with full context
- **Progress Tracking**: All long-running operations show progress
- **Error Handling**: Comprehensive exception handling
- **Reproducibility**: Fixed random seeds and deterministic behavior
- **Data Lineage**: Complete transformation tracking
- **Performance Monitoring**: Resource usage tracking

## Next Steps

After reviewing these examples:

1. 📖 Study the logging output in the `logs/` directory
2. 🔧 Adapt the patterns to your specific pipeline components
3. 🧪 Run the validation examples to understand data quality requirements
4. 🚀 Integrate these systems into your ML pipeline development

## Requirements

To run the examples, ensure you have:

```bash
pip install pandas numpy tqdm psutil pyyaml
```

## Support

These examples demonstrate the complete implementation of cursor rules standards for:
- Comprehensive traceability
- Professional progress tracking  
- Robust data validation
- Performance monitoring
- Error handling and recovery

Use these patterns throughout your ML pipeline development for production-ready code quality.
