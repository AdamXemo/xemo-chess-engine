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
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box


class TrainingProgress:
    """
    Training progress manager with clean epoch/batch display.
    
    Shows:
    - Epoch progress bar (overall training progress)
    - Current epoch batch progress (resets each epoch)
    - Live metrics (loss, lr)
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize training progress manager.
        
        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()
        
        # Create progress display with cleaner columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
            transient=False,  # Keep completed bars visible
        )
        
        self.epoch_task = None
        self.batch_task = None
        self.current_metrics = {}
        self.current_epoch = 0
        self.total_epochs = 0
    
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
        self.total_epochs = total_epochs
        self.epoch_task = self.progress.add_task(
            f"[cyan]{description}",
            total=total_epochs
        )
        return self.epoch_task
    
    def create_batch_bar(self, total_batches: int, description: str = "Epoch"):
        """
        Create or reset batch-level progress bar.
        
        On first call, creates the bar. On subsequent calls, resets it.
        
        Args:
            total_batches: Total number of batches
            description: Description text (e.g. "Epoch 1")
            
        Returns:
            Task ID for the batch bar
        """
        if self.batch_task is None:
            # First time: create the bar
            self.batch_task = self.progress.add_task(
                f"  [green]{description}",
                total=total_batches
            )
        else:
            # Subsequent: reset and update description
            self.progress.reset(self.batch_task, total=total_batches)
            self.progress.update(
                self.batch_task,
                description=f"  [green]{description}",
                completed=0
            )
        
        # Clear metrics for new epoch
        self.current_metrics = {}
        
        return self.batch_task
    
    def update_epoch(self, advance: int = 1):
        """
        Update epoch progress.
        
        Args:
            advance: Number of epochs to advance
        """
        if self.epoch_task is not None:
            self.current_epoch += advance
            self.progress.update(self.epoch_task, advance=advance)
    
    def update_batch(self, advance: int = 1):
        """
        Update batch progress.
        
        Args:
            advance: Number of batches to advance
        """
        if self.batch_task is not None:
            self.progress.update(self.batch_task, advance=advance)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update live metrics display in batch bar description.
        
        Args:
            metrics: Dictionary of metrics to display
        """
        self.current_metrics.update(metrics)
        
        if self.batch_task is not None:
            # Format metrics nicely
            metric_parts = []
            for k, v in list(self.current_metrics.items())[:2]:  # Show top 2 metrics
                if isinstance(v, float):
                    if abs(v) < 0.001:
                        metric_parts.append(f"{k}: {v:.2e}")
                    else:
                        metric_parts.append(f"{k}: {v:.4f}")
                else:
                    metric_parts.append(f"{k}: {v}")
            
            metric_str = " | ".join(metric_parts)
            
            # Get current epoch from description or use stored value
            epoch_num = self.current_epoch + 1
            
            self.progress.update(
                self.batch_task,
                description=f"  [green]Epoch {epoch_num}[/green] ({metric_str})"
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
