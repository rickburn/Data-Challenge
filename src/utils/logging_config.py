"""
Comprehensive logging configuration for Lending Club ML Pipeline.

This module provides centralized logging setup with multiple handlers,
structured formatting, and complete traceability for all operations.
"""

import os
import json
import sys
import uuid
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps


class PipelineLogger:
    """
    Centralized logging system for complete pipeline traceability.
    
    Features:
    - Multiple log handlers for different operations
    - Structured logging with JSON format options
    - Data lineage tracking
    - Operation metadata capture
    - Performance metrics logging
    """
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize pipeline logging system.
        
        Parameters
        ----------
        log_directory : str
            Directory to store all log files
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Create subdirectories for organized logging
        (self.log_directory / "daily").mkdir(exist_ok=True)
        (self.log_directory / "operations").mkdir(exist_ok=True)
        (self.log_directory / "performance").mkdir(exist_ok=True)
        
        self.execution_id = str(uuid.uuid4())
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure comprehensive logging system."""
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | '
            'exec_id=%(exec_id)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        json_formatter = JSONFormatter()
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Console handler with detailed format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        console_handler.addFilter(ExecutionContextFilter(self.execution_id))
        root_logger.addHandler(console_handler)
        
        # Main pipeline log file
        main_log_file = self.log_directory / "pipeline_execution.log"
        main_handler = logging.FileHandler(main_log_file)
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(detailed_formatter)
        main_handler.addFilter(ExecutionContextFilter(self.execution_id))
        root_logger.addHandler(main_handler)
        
        # Data operations log (JSON format for structured analysis)
        data_log_file = self.log_directory / "operations" / "data_operations.jsonl"
        data_handler = logging.FileHandler(data_log_file)
        data_handler.setLevel(logging.INFO)
        data_handler.setFormatter(json_formatter)
        data_handler.addFilter(ExecutionContextFilter(self.execution_id))
        
        # Add named logger for data operations
        data_logger = logging.getLogger("data_operations")
        data_logger.addHandler(data_handler)
        data_logger.propagate = False  # Don't duplicate to root logger
        
        # Model training log
        model_log_file = self.log_directory / "operations" / "model_training.jsonl"
        model_handler = logging.FileHandler(model_log_file)
        model_handler.setLevel(logging.INFO)
        model_handler.setFormatter(json_formatter)
        model_handler.addFilter(ExecutionContextFilter(self.execution_id))
        
        # Add named logger for model operations
        model_logger = logging.getLogger("model_training")
        model_logger.addHandler(model_handler)
        model_logger.propagate = False
        
        # Performance log
        perf_log_file = self.log_directory / "performance" / "performance_metrics.jsonl"
        perf_handler = logging.FileHandler(perf_log_file)
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(json_formatter)
        perf_handler.addFilter(ExecutionContextFilter(self.execution_id))
        
        # Add named logger for performance
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
        
        # Create comprehensive main log file with detailed INFO output
        main_log_file = self.log_directory / "pipeline_comprehensive.log"
        main_handler = logging.FileHandler(main_log_file)
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | '
            'exec_id=%(exec_id)s | %(message)s'
        ))
        main_handler.addFilter(ExecutionContextFilter(self.execution_id))

        # Add to root logger so ALL logs go to the main file
        root_logger = logging.getLogger()
        root_logger.addHandler(main_handler)
        root_logger.setLevel(logging.INFO)

        # Ensure all existing loggers also propagate to root
        logging.getLogger("data_operations").propagate = True
        logging.getLogger("model_training").propagate = True
        logging.getLogger("performance").propagate = True

        # Log successful setup
        logging.getLogger(__name__).info(
            f"Logging system initialized | exec_id={self.execution_id} | "
            f"log_dir={self.log_directory} | main_log={main_log_file.name}"
        )


class ExecutionContextFilter(logging.Filter):
    """Add execution context to all log records."""
    
    def __init__(self, execution_id: str):
        super().__init__()
        self.execution_id = execution_id
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Add execution ID to log record."""
        record.exec_id = self.execution_id
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'execution_id': getattr(record, 'exec_id', 'unknown')
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str)


class DataLineageTracker:
    """
    Track data transformations and lineage throughout the pipeline.
    
    Maintains complete audit trail of all data operations including:
    - Input/output data characteristics
    - Transformation operations
    - Data quality metrics
    - Processing timestamps
    """
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.lineage_file = self.log_directory / "data_lineage.jsonl"
        
    def track_operation(self, 
                       operation: str,
                       input_data: Any,
                       output_data: Any,
                       metadata: Optional[Dict] = None) -> None:
        """
        Track a data transformation operation.
        
        Parameters
        ----------
        operation : str
            Description of the operation performed
        input_data : Any
            Input data object (typically pandas DataFrame)
        output_data : Any
            Output data object
        metadata : Dict, optional
            Additional operation metadata
        """
        import hashlib
        import pandas as pd
        
        lineage_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'execution_id': getattr(logging.getLogger().handlers[0].filters[0], 'execution_id', 'unknown'),
            'input': self._get_data_characteristics(input_data),
            'output': self._get_data_characteristics(output_data),
            'metadata': metadata or {}
        }
        
        # Add data quality metrics if applicable
        if hasattr(input_data, 'shape') and hasattr(output_data, 'shape'):
            lineage_entry['transformation'] = {
                'rows_changed': output_data.shape[0] - input_data.shape[0],
                'columns_changed': output_data.shape[1] - input_data.shape[1],
                'data_size_change': self._estimate_memory_change(input_data, output_data)
            }
            
        # Write to lineage file
        with open(self.lineage_file, 'a') as f:
            f.write(json.dumps(lineage_entry, default=str) + '\n')
            
        # Also log to data operations logger
        data_logger = logging.getLogger("data_operations")
        data_logger.info("Data transformation tracked", extra=lineage_entry)
        
    def _get_data_characteristics(self, data: Any) -> Dict:
        """Extract characteristics from data object."""
        import pandas as pd
        import numpy as np
        
        if data is None:
            return {'type': 'None', 'shape': None, 'hash': None}
            
        characteristics = {
            'type': type(data).__name__,
            'hash': self._safe_hash(data)
        }
        
        # Add pandas-specific info
        if isinstance(data, pd.DataFrame):
            characteristics.update({
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().sum(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
            })
        elif isinstance(data, pd.Series):
            characteristics.update({
                'shape': data.shape,
                'dtype': str(data.dtype),
                'missing_values': data.isnull().sum(),
                'unique_values': data.nunique()
            })
        elif hasattr(data, 'shape'):
            characteristics['shape'] = data.shape
            
        return characteristics
        
    def _safe_hash(self, data: Any) -> str:
        """Generate safe hash for data object."""
        import hashlib
        
        try:
            if hasattr(data, 'values'):
                # For pandas objects, use values
                data_str = str(data.values.tobytes())
            else:
                data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "hash_failed"
            
    def _estimate_memory_change(self, input_data: Any, output_data: Any) -> Dict:
        """Estimate memory usage change between input and output."""
        try:
            input_mem = input_data.memory_usage(deep=True).sum() if hasattr(input_data, 'memory_usage') else 0
            output_mem = output_data.memory_usage(deep=True).sum() if hasattr(output_data, 'memory_usage') else 0
            
            return {
                'input_memory_mb': input_mem / (1024 * 1024),
                'output_memory_mb': output_mem / (1024 * 1024),
                'memory_delta_mb': (output_mem - input_mem) / (1024 * 1024)
            }
        except Exception:
            return {'memory_change': 'calculation_failed'}


def log_execution(func):
    """
    Decorator for comprehensive function execution logging.
    
    Automatically logs:
    - Function entry with parameters
    - Execution time and performance
    - Success/failure status
    - Exception details if errors occur
    - Data lineage for data transformation functions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        # Log function entry
        logger.info(
            f"ENTRY: {func.__name__}",
            extra={
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'start_time': start_time.isoformat()
            }
        )
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log successful completion
            logger.info(
                f"SUCCESS: {func.__name__}",
                extra={
                    'function': func.__name__,
                    'duration_seconds': execution_time,
                    'result_type': type(result).__name__,
                    'status': 'success'
                }
            )
            
            # Track data lineage if this is a data transformation
            if hasattr(result, 'shape') and len(args) > 0 and hasattr(args[0], 'shape'):
                lineage_tracker = DataLineageTracker()
                lineage_tracker.track_operation(
                    operation=func.__name__,
                    input_data=args[0],
                    output_data=result,
                    metadata={'duration_seconds': execution_time}
                )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log error with full context
            logger.error(
                f"ERROR: {func.__name__}",
                extra={
                    'function': func.__name__,
                    'duration_seconds': execution_time,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'status': 'failed'
                },
                exc_info=True
            )
            raise
            
    return wrapper


def setup_pipeline_logging(log_directory: str = "logs", 
                         log_level: str = "INFO") -> PipelineLogger:
    """
    Initialize comprehensive pipeline logging.
    
    Parameters
    ----------
    log_directory : str
        Directory for log files
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns
    -------
    PipelineLogger
        Configured logging system
    """
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    # Initialize pipeline logger
    pipeline_logger = PipelineLogger(log_directory)
    
    # Log system information
    import sys
    import platform
    
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'working_directory': os.getcwd(),
        'execution_id': pipeline_logger.execution_id
    }
    
    logging.getLogger(__name__).info(
        "Pipeline logging system initialized",
        extra=system_info
    )
    
    return pipeline_logger


# Initialize global lineage tracker
_lineage_tracker = None

def get_lineage_tracker() -> DataLineageTracker:
    """Get global data lineage tracker instance."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = DataLineageTracker()
    return _lineage_tracker


def track_data_transformation(operation: str, 
                            input_data: Any, 
                            output_data: Any,
                            metadata: Optional[Dict] = None) -> None:
    """
    Convenience function for tracking data transformations.
    
    Parameters
    ----------
    operation : str
        Description of the transformation
    input_data : Any
        Input data
    output_data : Any
        Output data  
    metadata : Dict, optional
        Additional metadata
    """
    tracker = get_lineage_tracker()
    tracker.track_operation(operation, input_data, output_data, metadata)


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    
    # Initialize logging
    logger_system = setup_pipeline_logging()
    
    # Example data transformation with logging
    @log_execution
    def example_data_processing(df: pd.DataFrame) -> pd.DataFrame:
        """Example function with comprehensive logging."""
        logger = logging.getLogger(__name__)
        logger.info("Processing data transformation")
        
        # Simulate processing
        processed_df = df.copy()
        processed_df['new_column'] = processed_df['existing_column'] * 2
        
        return processed_df
    
    # Test the logging system
    sample_data = pd.DataFrame({'existing_column': [1, 2, 3, 4, 5]})
    result = example_data_processing(sample_data)
    
    logging.getLogger(__name__).info("Logging system test completed successfully")
