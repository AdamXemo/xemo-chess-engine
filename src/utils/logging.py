"""
Experiment logging utility combining terminal and file logging.

Provides unified logging interface with rich terminal output and
structured file logging for post-analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

from .formatting import (
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_metric_table,
    print_config,
)


class ExperimentLogger:
    """
    Unified logger for experiments combining terminal and file logging.
    
    Features:
    - Rich terminal output with colors and formatting
    - Structured file logging for experiments
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR)
    - Experiment-specific log files
    """
    
    def __init__(
        self,
        name: str = "chess_ai",
        log_level: int = logging.INFO,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize the experiment logger.
        
        Args:
            name: Logger name
            log_level: Overall log level
            console_level: Terminal output level
            file_level: File output level
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        self.console = Console()
        self.console_level = console_level
        self.file_level = file_level
        
        # Setup console handler with rich
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False
        )
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.file_handler = None
        self.experiment_name = None
        self.log_file = None
    
    def setup_experiment(
        self,
        experiment_name: str,
        log_dir: str = "experiments/logs"
    ):
        """
        Setup experiment-specific file logging.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{experiment_name}_{timestamp}.log"
        self.log_file = log_path / log_filename
        
        # Setup file handler
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
        
        self.file_handler = logging.FileHandler(self.log_file, mode='w')
        self.file_handler.setLevel(self.file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(file_formatter)
        self.logger.addHandler(self.file_handler)
        
        self.info(f"Experiment: {experiment_name}")
        self.info(f"Log file: {self.log_file}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def section(self, title: str):
        """
        Print a section header.
        
        Args:
            title: Section title
        """
        print_section(title)
        self.info(f"=== {title} ===")
    
    def header(self, text: str):
        """
        Print a header box.
        
        Args:
            text: Header text
        """
        print_header(text)
        self.info(f"### {text} ###")
    
    def success(self, message: str):
        """
        Log and print success message.
        
        Args:
            message: Success message
        """
        print_success(message)
        self.info(f"SUCCESS: {message}")
    
    def error_msg(self, message: str):
        """
        Log and print error message.
        
        Args:
            message: Error message
        """
        print_error(message)
        self.error(message)
    
    def warning_msg(self, message: str):
        """
        Log and print warning message.
        
        Args:
            message: Warning message
        """
        print_warning(message)
        self.warning(message)
    
    def info_msg(self, message: str):
        """
        Log and print info message.
        
        Args:
            message: Info message
        """
        print_info(message)
        self.info(message)
    
    def table(self, data: Dict[str, Any], title: Optional[str] = None):
        """
        Print and log a metrics table.
        
        Args:
            data: Dictionary of metrics
            title: Table title
        """
        print_metric_table(data, title=title)
        
        # Log to file
        if title:
            self.info(f"--- {title} ---")
        for key, value in data.items():
            self.info(f"  {key}: {value}")
    
    def config(self, config: Dict[str, Any], title: str = "Configuration"):
        """
        Print and log configuration.
        
        Args:
            config: Configuration dictionary
            title: Section title
        """
        print_config(config, title=title)
        
        # Log to file
        self.info(f"--- {title} ---")
        
        def log_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    log_dict(value, full_key)
                else:
                    self.info(f"  {full_key}: {value}")
        
        log_dict(config)
    
    def metric(self, name: str, value: Any, step: Optional[int] = None):
        """
        Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if step is not None:
            self.info(f"[Step {step}] {name}: {value}")
        else:
            self.info(f"{name}: {value}")
    
    def metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/epoch number
        """
        prefix = f"[Step {step}] " if step is not None else ""
        for name, value in metrics.items():
            self.info(f"{prefix}{name}: {value}")
    
    def close(self):
        """Close file handlers."""
        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)


# Global logger instance
_global_logger: Optional[ExperimentLogger] = None


def get_logger(name: str = "chess_ai") -> ExperimentLogger:
    """
    Get or create global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        ExperimentLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(name)
    return _global_logger


def setup_experiment(experiment_name: str, log_dir: str = "experiments/logs"):
    """
    Setup experiment logging (convenience function).
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
    """
    logger = get_logger()
    logger.setup_experiment(experiment_name, log_dir)
    return logger

