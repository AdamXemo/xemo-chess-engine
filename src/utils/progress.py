"""
Progress bar utilities for training and data loading.

Uses rich.Progress for professional ML-style progress bars with
nested bars and live metric updates.
"""

from typing import Optional, Dict, Any
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box


class TrainingProgress:
    """
    Training progress manager with nested progress bars and live metrics.
    
    Features:
    - Nested epoch and batch progress bars
    - Live metric updates (loss, MAE, etc.)
    - Samples per second calculation
    - Time remaining estimation
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize training progress manager.
        
        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()
        
        # Create progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False
        )
        
        self.epoch_task = None
        self.batch_task = None
        self.metrics_table = None
        self.live = None
        
        self.current_metrics = {}
    
    def start(self):
        """Start the progress display."""
        self.progress.start()
    
    def stop(self):
        """Stop the progress display."""
        self.progress.stop()
    
    def create_epoch_bar(self, total_epochs: int, description: str = "Training"):
        """
        Create epoch-level progress bar.
        
        Args:
            total_epochs: Total number of epochs
            description: Description text
            
        Returns:
            Task ID for the epoch bar
        """
        self.epoch_task = self.progress.add_task(
            f"[cyan]{description}",
            total=total_epochs
        )
        return self.epoch_task
    
    def create_batch_bar(self, total_batches: int, description: str = "Batches"):
        """
        Create batch-level progress bar (nested under epoch).
        
        Args:
            total_batches: Total number of batches
            description: Description text
            
        Returns:
            Task ID for the batch bar
        """
        self.batch_task = self.progress.add_task(
            f"  [green]{description}",
            total=total_batches
        )
        return self.batch_task
    
    def update_epoch(self, advance: int = 1):
        """
        Update epoch progress.
        
        Args:
            advance: Number of epochs to advance
        """
        if self.epoch_task is not None:
            self.progress.update(self.epoch_task, advance=advance)
    
    def update_batch(self, advance: int = 1):
        """
        Update batch progress.
        
        Args:
            advance: Number of batches to advance
        """
        if self.batch_task is not None:
            self.progress.update(self.batch_task, advance=advance)
    
    def reset_batch_bar(self, total_batches: int):
        """
        Reset batch progress bar for new epoch.
        
        Args:
            total_batches: Total number of batches
        """
        if self.batch_task is not None:
            self.progress.reset(self.batch_task, total=total_batches)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update live metrics display.
        
        Args:
            metrics: Dictionary of metrics to display
        """
        self.current_metrics.update(metrics)
        
        # Update batch task description with metrics
        if self.batch_task is not None:
            metric_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in list(self.current_metrics.items())[:3]  # Show first 3 metrics
            ])
            self.progress.update(
                self.batch_task,
                description=f"  [green]Batches[/green] ({metric_str})"
            )
    
    def finish(self):
        """Finish and close progress bars."""
        self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class DataLoadProgress:
    """
    Simple progress bar for data loading operations.
    
    Lighter weight than TrainingProgress for simple tasks.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize data loading progress.
        
        Args:
            console: Rich console instance
        """
        self.console = console or Console()
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        self.task = None
    
    def start(self, description: str, total: int):
        """
        Start progress bar.
        
        Args:
            description: Task description
            total: Total items to process
            
        Returns:
            Task ID
        """
        self.progress.start()
        self.task = self.progress.add_task(description, total=total)
        return self.task
    
    def update(self, advance: int = 1):
        """
        Update progress.
        
        Args:
            advance: Number of items completed
        """
        if self.task is not None:
            self.progress.update(self.task, advance=advance)
    
    def finish(self):
        """Finish and stop progress bar."""
        self.progress.stop()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class SimpleProgress:
    """
    Very simple progress wrapper for basic tasks.
    
    Good for short operations where full progress bar is overkill.
    """
    
    def __init__(self, description: str, total: int, console: Optional[Console] = None):
        """
        Initialize simple progress.
        
        Args:
            description: Task description
            total: Total items
            console: Rich console
        """
        self.console = console or Console()
        self.description = description
        self.total = total
        self.current = 0
        
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        self.task = None
    
    def __enter__(self):
        """Start progress."""
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=self.total)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress."""
        self.progress.stop()
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current += n
        if self.task is not None:
            self.progress.update(self.task, advance=n)
    
    def set_description(self, description: str):
        """
        Update description.
        
        Args:
            description: New description
        """
        if self.task is not None:
            self.progress.update(self.task, description=description)

