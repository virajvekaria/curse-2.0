"""
CSV Processing Module

This module provides functionality for loading and processing CSV files.
"""

import csv
from typing import List


class CSVProcessor:
    """A class for processing CSV files."""

    def __init__(self, filename: str):
        """Initialize the CSV processor with a filename."""
        self.filename = filename
        self.data = []

    def load_data(self) -> List[List[str]]:
        """Load data from the CSV file."""
        try:
            with open(self.filename, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                self.data = list(reader)
            print(f"Data loaded from {self.filename}")
            return self.data
        except FileNotFoundError:
            print(f"File {self.filename} not found")
            return []
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def get_headers(self) -> List[str]:
        """Get the header row from the CSV data."""
        return self.data[0] if self.data else []

    def get_data_rows(self) -> List[List[str]]:
        """Get all data rows (excluding header)."""
        return self.data[1:] if len(self.data) > 1 else []