"""
Formatting utilities for beautiful terminal output.

Uses the rich library for professional ML-style output.
"""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


# Global console instance
console = Console()


# Color scheme
COLORS = {
    'info': 'cyan',
    'success': 'green',
    'warning': 'yellow',
    'error': 'red',
    'highlight': 'magenta',
    'muted': 'dim',
}


def print_header(text: str, style: str = 'bold cyan'):
    """
    Print a centered header with box borders.
    
    Args:
        text: Header text
        style: Rich style string
    """
    panel = Panel(
        Text(text, justify='center', style=style),
        box=box.ROUNDED,
        border_style='cyan',
        padding=(0, 2)
    )
    console.print(panel)


def print_section(text: str):
    """
    Print a section divider.
    
    Args:
        text: Section title
    """
    console.print()
    console.rule(f"[bold cyan]{text}[/bold cyan]", style='cyan')
    console.print()


def print_success(text: str):
    """
    Print a success message with checkmark.
    
    Args:
        text: Success message
    """
    console.print(f"[green]✓[/green] {text}")


def print_error(text: str):
    """
    Print an error message with X mark.
    
    Args:
        text: Error message
    """
    console.print(f"[red]✗[/red] {text}")


def print_warning(text: str):
    """
    Print a warning message with warning symbol.
    
    Args:
        text: Warning message
    """
    console.print(f"[yellow]⚠[/yellow] {text}")


def print_info(text: str):
    """
    Print an info message.
    
    Args:
        text: Info message
    """
    console.print(f"[cyan]ℹ[/cyan] {text}")


def print_metric_table(metrics: Dict[str, Any], title: Optional[str] = "Metrics"):
    """
    Print metrics as a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Table title
    """
    table = Table(title=title, box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta", justify="right")
    
    for name, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)
        
        table.add_row(name, formatted_value)
    
    console.print(table)


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """
    Pretty-print configuration as a table.
    
    Args:
        config: Configuration dictionary
        title: Table title
    """
    table = Table(title=title, box=box.SIMPLE, show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    def add_rows(d: Dict, prefix: str = ""):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                add_rows(value, full_key)
            else:
                table.add_row(full_key, str(value))
    
    add_rows(config)
    console.print(table)


def print_comparison_table(data: Dict[str, Dict[str, Any]], title: str = "Comparison"):
    """
    Print a comparison table with multiple columns.
    
    Args:
        data: Dictionary where keys are column names and values are dicts of metrics
        title: Table title
        
    Example:
        data = {
            'Model A': {'loss': 0.23, 'accuracy': 0.89},
            'Model B': {'loss': 0.19, 'accuracy': 0.92}
        }
    """
    if not data:
        return
    
    # Get all unique metric names
    all_metrics = set()
    for metrics in data.values():
        all_metrics.update(metrics.keys())
    
    table = Table(title=title, box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan")
    
    for model_name in data.keys():
        table.add_column(model_name, style="magenta", justify="right")
    
    for metric in sorted(all_metrics):
        row = [metric]
        for model_metrics in data.values():
            value = model_metrics.get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, int):
                row.append(f"{value:,}")
            else:
                row.append(str(value))
        table.add_row(*row)
    
    console.print(table)


def print_model_summary(
    model_name: str,
    parameters: int,
    input_shape: tuple,
    output_shape: tuple,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Print a model architecture summary.
    
    Args:
        model_name: Name of the model
        parameters: Total parameters
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        additional_info: Optional additional information
    """
    table = Table(title=f"Model: {model_name}", box=box.DOUBLE, show_header=False)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="white")
    
    table.add_row("Total Parameters", f"{parameters:,}")
    table.add_row("Input Shape", str(input_shape))
    table.add_row("Output Shape", str(output_shape))
    
    if additional_info:
        for key, value in additional_info.items():
            table.add_row(key, str(value))
    
    console.print(table)


def print_status(message: str, status: str = 'info'):
    """
    Print a status message with appropriate symbol and color.
    
    Args:
        message: Status message
        status: One of 'info', 'success', 'warning', 'error'
    """
    symbols = {
        'info': ('ℹ', 'cyan'),
        'success': ('✓', 'green'),
        'warning': ('⚠', 'yellow'),
        'error': ('✗', 'red'),
    }
    
    symbol, color = symbols.get(status, ('•', 'white'))
    console.print(f"[{color}]{symbol}[/{color}] {message}")


def print_separator(char: str = '─', width: Optional[int] = None):
    """
    Print a horizontal separator line.
    
    Args:
        char: Character to use for separator
        width: Width of separator (None = full terminal width)
    """
    if width is None:
        console.print(f"[dim]{char * console.width}[/dim]")
    else:
        console.print(f"[dim]{char * width}[/dim]")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.0f}m"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


def format_number(num: float, precision: int = 2) -> str:
    """
    Format a number with appropriate precision and separators.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if abs(num) >= 1000:
        return f"{num:,.{precision}f}"
    elif abs(num) >= 1:
        return f"{num:.{precision}f}"
    else:
        return f"{num:.{precision+2}f}"

