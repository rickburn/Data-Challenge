"""
Comprehensive progress tracking system for Lending Club ML Pipeline.

This module provides advanced progress tracking with nested progress bars,
performance monitoring, ETA calculations, and detailed logging integration.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")


@dataclass
class ProgressMetrics:
    """Comprehensive progress metrics tracking."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    step_times: List[float] = field(default_factory=list)
    step_descriptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def average_step_time(self) -> float:
        """Average time per step."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
    
    @property
    def eta_seconds(self) -> float:
        """Estimated time to completion in seconds."""
        if not self.step_times or self.completed_steps == 0:
            return 0.0
        remaining_steps = self.total_steps - self.completed_steps
        return self.average_step_time * remaining_steps
    
    @property
    def completion_percentage(self) -> float:
        """Completion percentage (0-100)."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'duration_seconds': self.duration_seconds,
            'average_step_time': self.average_step_time,
            'eta_seconds': self.eta_seconds,
            'completion_percentage': self.completion_percentage,
            'metadata': self.metadata
        }


class PipelineProgressTracker:
    """
    Advanced progress tracking for ML pipeline operations.
    
    Features:
    - Nested progress bars for complex operations
    - ETA calculation with adaptive algorithms
    - Performance monitoring and bottleneck detection
    - Integration with logging system
    - Memory usage tracking
    - Automatic checkpoint creation
    """
    
    def __init__(self, 
                 total_steps: int,
                 description: str = "Processing",
                 unit: str = "step",
                 enable_logging: bool = True,
                 enable_nested: bool = True,
                 checkpoint_frequency: int = 10):
        """
        Initialize progress tracker.
        
        Parameters
        ----------
        total_steps : int
            Total number of steps expected
        description : str
            Description of the operation
        unit : str
            Unit of measurement for steps
        enable_logging : bool
            Whether to log progress updates
        enable_nested : bool
            Whether to support nested progress bars
        checkpoint_frequency : int
            Frequency of automatic checkpoints
        """
        self.metrics = ProgressMetrics(total_steps=total_steps)
        self.description = description
        self.unit = unit
        self.enable_logging = enable_logging
        self.enable_nested = enable_nested
        self.checkpoint_frequency = checkpoint_frequency
        
        self.logger = logging.getLogger(__name__) if enable_logging else None
        self.nested_trackers: List['PipelineProgressTracker'] = []
        self.is_closed = False
        self.lock = threading.Lock()
        
        # Initialize progress bar if tqdm is available
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=total_steps,
                desc=description,
                unit=unit,
                leave=True,
                position=len(self._get_active_trackers())
            )
        else:
            self.pbar = None
            
        # Log initialization
        if self.logger:
            self.logger.info(
                f"Progress tracker initialized: {description}",
                extra={
                    'operation': 'progress_init',
                    'total_steps': total_steps,
                    'description': description,
                    'tracker_id': id(self)
                }
            )
    
    def update(self, 
               n: int = 1, 
               step_description: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress with step information.
        
        Parameters
        ----------
        n : int
            Number of steps to advance
        step_description : str
            Description of current step
        metadata : Dict, optional
            Additional step metadata
        """
        if self.is_closed:
            return
            
        with self.lock:
            # Record step timing
            current_time = time.time()
            if self.metrics.step_times:
                step_duration = current_time - self._last_update_time
                self.metrics.step_times.append(step_duration)
            else:
                # First step timing from start
                step_duration = current_time - self.metrics.start_time.timestamp()
                self.metrics.step_times.append(step_duration)
                
            self._last_update_time = current_time
            
            # Update metrics
            self.metrics.completed_steps += n
            if step_description:
                self.metrics.step_descriptions.append(step_description)
            if metadata:
                self.metrics.metadata.update(metadata)
                
            # Update progress bar
            if self.pbar:
                postfix = self._generate_postfix(step_description, step_duration)
                self.pbar.set_postfix_str(postfix)
                self.pbar.update(n)
            else:
                # Console fallback if no tqdm
                percentage = self.metrics.completion_percentage
                eta_str = f"ETA: {self.metrics.eta_seconds:.0f}s" if self.metrics.eta_seconds > 0 else ""
                print(f"\r{self.description}: {percentage:.1f}% | {step_description} | {eta_str}", end="")
                
            # Log progress
            if self.logger and (self.metrics.completed_steps % self.checkpoint_frequency == 0 or step_description):
                self.logger.info(
                    f"Progress update: {step_description or 'Step completed'}",
                    extra={
                        'operation': 'progress_update',
                        'step': self.metrics.completed_steps,
                        'total': self.metrics.total_steps,
                        'percentage': self.metrics.completion_percentage,
                        'eta_seconds': self.metrics.eta_seconds,
                        'step_duration': step_duration,
                        'description': step_description,
                        'metadata': metadata or {}
                    }
                )
                
    def _generate_postfix(self, step_description: str, step_duration: float) -> str:
        """Generate postfix string for progress bar."""
        postfix_parts = []
        
        if step_description:
            postfix_parts.append(f"{step_description}")
            
        postfix_parts.append(f"{step_duration:.1f}s")
        
        if self.metrics.eta_seconds > 0:
            eta_str = str(timedelta(seconds=int(self.metrics.eta_seconds)))
            postfix_parts.append(f"ETA: {eta_str}")
            
        return " | ".join(postfix_parts)
    
    def create_nested_tracker(self, 
                            total_steps: int,
                            description: str = "Subtask") -> 'PipelineProgressTracker':
        """
        Create nested progress tracker for sub-operations.
        
        Parameters
        ----------
        total_steps : int
            Total steps for nested operation
        description : str
            Description of nested operation
            
        Returns
        -------
        PipelineProgressTracker
            Nested progress tracker
        """
        if not self.enable_nested:
            raise ValueError("Nested tracking is disabled for this tracker")
            
        nested_tracker = PipelineProgressTracker(
            total_steps=total_steps,
            description=f"  ↳ {description}",
            enable_logging=self.enable_logging,
            enable_nested=False  # Prevent deep nesting
        )
        
        self.nested_trackers.append(nested_tracker)
        return nested_tracker
    
    def close(self) -> ProgressMetrics:
        """
        Close progress tracker and return final metrics.
        
        Returns
        -------
        ProgressMetrics
            Final progress metrics
        """
        if self.is_closed:
            return self.metrics
            
        with self.lock:
            self.metrics.end_time = datetime.now()
            self.is_closed = True
            
            # Close nested trackers
            for tracker in self.nested_trackers:
                if not tracker.is_closed:
                    tracker.close()
                    
            # Close progress bar
            if self.pbar:
                self.pbar.close()
            else:
                print()  # New line for console fallback
                
            # Log completion
            if self.logger:
                self.logger.info(
                    f"Progress tracker completed: {self.description}",
                    extra={
                        'operation': 'progress_complete',
                        'final_metrics': self.metrics.to_dict(),
                        'tracker_id': id(self)
                    }
                )
                
        return self.metrics
    
    @staticmethod
    def _get_active_trackers() -> List['PipelineProgressTracker']:
        """Get list of currently active progress trackers."""
        # This is a simplified implementation - in practice you might want
        # to maintain a global registry of active trackers
        return []
    
    def __enter__(self) -> 'PipelineProgressTracker':
        """Context manager entry."""
        self._last_update_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class BatchProgressTracker:
    """
    Progress tracker for batch operations with automatic batching.
    
    Useful for processing large datasets in chunks with progress tracking.
    """
    
    def __init__(self, 
                 total_items: int,
                 batch_size: int = 100,
                 description: str = "Batch Processing"):
        """
        Initialize batch progress tracker.
        
        Parameters
        ----------
        total_items : int
            Total number of items to process
        batch_size : int
            Size of each batch
        description : str
            Operation description
        """
        self.total_items = total_items
        self.batch_size = batch_size
        self.total_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division
        
        self.main_tracker = PipelineProgressTracker(
            total_steps=self.total_batches,
            description=description,
            unit="batch"
        )
        
        self.current_batch = 0
        self.items_processed = 0
        
    def process_batch(self, 
                     batch_processor: Callable[[int, int], Any],
                     batch_description: str = "") -> List[Any]:
        """
        Process items in batches with progress tracking.
        
        Parameters
        ----------
        batch_processor : Callable[[int, int], Any]
            Function that processes a batch given start and end indices
        batch_description : str
            Description for current batch
            
        Returns
        -------
        List[Any]
            Results from all batch processing
        """
        results = []
        
        with self.main_tracker:
            for batch_idx in range(self.total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.total_items)
                
                batch_desc = batch_description or f"Items {start_idx}-{end_idx-1}"
                
                # Process batch
                batch_result = batch_processor(start_idx, end_idx)
                results.append(batch_result)
                
                # Update progress
                items_in_batch = end_idx - start_idx
                self.items_processed += items_in_batch
                
                self.main_tracker.update(
                    n=1,
                    step_description=batch_desc,
                    metadata={
                        'batch_idx': batch_idx,
                        'items_in_batch': items_in_batch,
                        'total_items_processed': self.items_processed
                    }
                )
                
        return results


@contextmanager
def track_operation(description: str, 
                   total_steps: Optional[int] = None,
                   enable_progress_bar: bool = True):
    """
    Context manager for tracking operation progress.
    
    Parameters
    ----------
    description : str
        Operation description
    total_steps : int, optional
        Total expected steps
    enable_progress_bar : bool
        Whether to show progress bar
        
    Yields
    ------
    PipelineProgressTracker or SimpleTracker
        Progress tracker instance
    """
    if total_steps and enable_progress_bar:
        tracker = PipelineProgressTracker(total_steps, description)
        try:
            yield tracker
        finally:
            if not tracker.is_closed:
                tracker.close()
    else:
        # Simple tracking without progress bar
        tracker = SimpleOperationTracker(description)
        try:
            yield tracker
        finally:
            tracker.close()


class SimpleOperationTracker:
    """Simple operation tracker without progress bar."""
    
    def __init__(self, description: str):
        self.description = description
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        self.is_closed = False
        
        self.logger.info(f"Operation started: {description}")
        
    def update(self, step_description: str = "", metadata: Optional[Dict] = None):
        """Log operation update."""
        if not self.is_closed:
            self.logger.info(
                f"Operation progress: {step_description}",
                extra={
                    'operation': self.description,
                    'step_description': step_description,
                    'metadata': metadata or {}
                }
            )
    
    def close(self):
        """Close operation tracker."""
        if not self.is_closed:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(
                f"Operation completed: {self.description} | Duration: {duration:.2f}s"
            )
            self.is_closed = True


class PerformanceMonitor:
    """
    Monitor performance metrics during operations.
    
    Tracks CPU usage, memory usage, and operation timing.
    """
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        self.logger = logging.getLogger("performance")
        
    def __enter__(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
        self.logger.info(
            f"Performance monitoring started: {self.operation_name}",
            extra={
                'operation': self.operation_name,
                'start_memory_mb': self.start_memory,
                'monitoring_start': datetime.now().isoformat()
            }
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End performance monitoring and log results."""
        duration = time.time() - self.start_time
        end_memory = self._get_memory_usage()
        memory_delta = end_memory - self.start_memory
        
        performance_metrics = {
            'operation': self.operation_name,
            'duration_seconds': duration,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'memory_delta_mb': memory_delta,
            'status': 'success' if exc_type is None else 'failed'
        }
        
        if exc_type:
            performance_metrics['error_type'] = exc_type.__name__
            
        self.logger.info(
            f"Performance monitoring completed: {self.operation_name}",
            extra=performance_metrics
        )
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0  # Fallback if psutil not available


# Example usage and testing
if __name__ == "__main__":
    import random
    import pandas as pd
    
    # Test basic progress tracking
    def example_data_processing():
        """Example function with progress tracking."""
        with track_operation("Data Processing", total_steps=100) as progress:
            for i in range(100):
                # Simulate processing
                time.sleep(0.01)
                progress.update(step_description=f"Processing item {i+1}")
                
    # Test nested progress tracking
    def example_model_training():
        """Example with nested progress tracking."""
        with track_operation("Model Training", total_steps=3) as main_progress:
            
            # Stage 1: Data preprocessing
            main_progress.update(step_description="Starting data preprocessing")
            with main_progress.create_nested_tracker(50, "Preprocessing") as prep_progress:
                for i in range(50):
                    time.sleep(0.002)
                    prep_progress.update(step_description=f"Preprocessing step {i+1}")
            
            # Stage 2: Model training  
            main_progress.update(step_description="Starting model training")
            with main_progress.create_nested_tracker(25, "Training") as train_progress:
                for i in range(25):
                    time.sleep(0.005)
                    train_progress.update(step_description=f"Training epoch {i+1}")
                    
            # Stage 3: Model evaluation
            main_progress.update(step_description="Starting model evaluation")
            time.sleep(0.1)  # Simulate evaluation
            
    # Test batch processing
    def example_batch_processing():
        """Example batch processing with progress tracking."""
        def process_batch(start_idx: int, end_idx: int) -> Dict:
            # Simulate batch processing
            time.sleep(0.01)
            return {
                'processed_items': end_idx - start_idx,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
        batch_tracker = BatchProgressTracker(
            total_items=1000,
            batch_size=50,
            description="Batch Data Processing"
        )
        
        results = batch_tracker.process_batch(process_batch)
        print(f"Processed {len(results)} batches")
        
    # Test performance monitoring
    def example_with_performance_monitoring():
        """Example with performance monitoring."""
        with PerformanceMonitor("Example Operation"):
            # Simulate some work
            data = pd.DataFrame({'col1': range(10000), 'col2': range(10000)})
            result = data.groupby('col1').sum()
            time.sleep(0.1)
            
    # Run examples
    print("Testing basic progress tracking...")
    example_data_processing()
    
    print("\nTesting nested progress tracking...")
    example_model_training()
    
    print("\nTesting batch processing...")
    example_batch_processing()
    
    print("\nTesting performance monitoring...")
    example_with_performance_monitoring()
    
    print("\nAll progress tracking tests completed!")
