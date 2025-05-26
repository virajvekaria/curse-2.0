"""
Data Analysis Module

This module provides functionality for analyzing data from various sources.
"""

from typing import List, Dict, Any
import statistics


class DataAnalyzer:
    """A class for analyzing data and generating insights."""

    def __init__(self, data: List[List[str]]):
        """Initialize the analyzer with data."""
        self.data = data
        self.headers = data[0] if data else []
        self.rows = data[1:] if len(data) > 1 else []

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the data by category."""
        if not self.rows:
            return {}

        summary = {}
        for row in self.rows:
            if len(row) >= 2:
                category = row[0]
                try:
                    value = int(row[1])
                    if category not in summary:
                        summary[category] = 0
                    summary[category] += value
                except (ValueError, IndexError):
                    continue

        return summary

    def calculate_statistics(self, column_index: int = 1) -> Dict[str, float]:
        """Calculate basic statistics for a numeric column."""
        if not self.rows:
            return {}

        values = []
        for row in self.rows:
            try:
                if len(row) > column_index:
                    values.append(float(row[column_index]))
            except (ValueError, IndexError):
                continue

        if not values:
            return {}

        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'mode': statistics.mode(values) if len(set(values)) < len(values) else None,
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }