"""
Utility functions for data processing and analysis.
"""

from typing import Any, Union
import os
from pathlib import Path


def validate_number(value: Any) -> bool:
    """Validate if a value can be converted to a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """Format a number for display."""
    if isinstance(number, int) or (isinstance(number, float) and number.is_integer()):
        return str(int(number))
    return f"{number:.{decimal_places}f}"


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning 0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def clean_string(text: str) -> str:
    """Clean a string by removing extra whitespace and converting to lowercase."""
    return text.strip().lower()


def is_file_readable(filepath: Union[str, Path]) -> bool:
    """Check if a file exists and is readable."""
    try:
        path_obj = Path(filepath)
        return path_obj.exists() and path_obj.is_file() and os.access(path_obj, os.R_OK)
    except (OSError, TypeError):
        return False
