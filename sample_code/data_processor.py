"""
Data processing utilities for handling various data formats.
"""

import json
import csv
from typing import List, Dict, Any
from pathlib import Path


class DataProcessor:
    """A class for processing different types of data files."""
    
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        self.processed_files = []
    
    def read_json(self, file_path: str) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            self.processed_files.append(file_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
    
    def write_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """Write data to a JSON file."""
        try:
            with open(file_path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing JSON file: {e}")
            return False
    
    def read_csv(self, file_path: str, delimiter=',') -> List[Dict[str, str]]:
        """Read a CSV file and return a list of dictionaries."""
        data = []
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    data.append(row)
            self.processed_files.append(file_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def filter_data(self, data: List[Dict], key: str, value: Any) -> List[Dict]:
        """Filter data based on a key-value pair."""
        return [item for item in data if item.get(key) == value]
    
    def aggregate_data(self, data: List[Dict], group_by: str, 
                      aggregate_field: str, operation='sum') -> Dict[str, float]:
        """Aggregate data by grouping and performing operations."""
        groups = {}
        
        for item in data:
            group_key = item.get(group_by)
            if group_key not in groups:
                groups[group_key] = []
            
            try:
                value = float(item.get(aggregate_field, 0))
                groups[group_key].append(value)
            except (ValueError, TypeError):
                continue
        
        # Perform aggregation
        result = {}
        for group, values in groups.items():
            if operation == 'sum':
                result[group] = sum(values)
            elif operation == 'avg':
                result[group] = sum(values) / len(values) if values else 0
            elif operation == 'max':
                result[group] = max(values) if values else 0
            elif operation == 'min':
                result[group] = min(values) if values else 0
            elif operation == 'count':
                result[group] = len(values)
        
        return result


def validate_email(email: str) -> bool:
    """Simple email validation function."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text
