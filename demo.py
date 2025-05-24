"""
Demo script to showcase the code analysis system capabilities.
"""

from pathlib import Path
from analyzer import CodebaseAnalyzer
from storage import VectorDatabase
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def create_sample_code():
    """Create some sample Python files for demonstration."""
    
    # Create a sample directory structure
    sample_dir = Path("sample_code")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample Python file 1: Calculator
    calculator_code = '''"""
A simple calculator module with basic arithmetic operations.
"""

class Calculator:
    """A calculator class that performs basic arithmetic operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers and return the result."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        """Subtract b from a and return the result."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers and return the result."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide a by b and return the result."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        """Return the calculation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear the calculation history."""
        self.history.clear()


def calculate_factorial(n):
    """Calculate the factorial of a number recursively."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * calculate_factorial(n - 1)


def fibonacci_sequence(n):
    """Generate the first n numbers in the Fibonacci sequence."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence
'''
    
    # Sample Python file 2: Data processor
    data_processor_code = '''"""
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
'''
    
    # Write the sample files
    with open(sample_dir / "calculator.py", "w") as f:
        f.write(calculator_code)
    
    with open(sample_dir / "data_processor.py", "w") as f:
        f.write(data_processor_code)
    
    console.print(f"[green]Created sample code in {sample_dir}/[/green]")
    return sample_dir


def run_demo():
    """Run a comprehensive demo of the code analysis system."""
    
    console.print(Panel.fit(
        "[bold blue]Code Analysis System Demo[/bold blue]\n"
        "This demo showcases the key features of our minimal Cursor-like system",
        title="Welcome"
    ))
    
    # Step 1: Create sample code
    console.print("\n[bold]Step 1: Creating sample code...[/bold]")
    sample_dir = create_sample_code()
    
    # Step 2: Initialize analyzer
    console.print("\n[bold]Step 2: Initializing analyzer...[/bold]")
    analyzer = CodebaseAnalyzer()
    
    # Step 3: Analyze the sample codebase
    console.print("\n[bold]Step 3: Analyzing codebase...[/bold]")
    results = analyzer.analyze_path(sample_dir, recursive=True)
    
    # Display analysis results
    table = Table(title="Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Files Processed", str(results['files_processed']))
    table.add_row("Chunks Extracted", str(results['chunks_extracted']))
    table.add_row("Embeddings Generated", str(results['embeddings_generated']))
    table.add_row("Merkle Hash", results['merkle_hash'][:16] + "..." if results['merkle_hash'] else "None")
    
    console.print(table)
    
    # Step 4: Demonstrate search capabilities
    console.print("\n[bold]Step 4: Demonstrating semantic search...[/bold]")
    
    search_queries = [
        "mathematical operations",
        "file processing",
        "data validation",
        "error handling"
    ]
    
    db = VectorDatabase()
    
    for query in search_queries:
        console.print(f"\n[cyan]Searching for: '{query}'[/cyan]")
        results = db.search_by_text(query, n_results=3)
        
        if results:
            search_table = Table()
            search_table.add_column("Similarity", style="green")
            search_table.add_column("Type", style="cyan")
            search_table.add_column("Name", style="bold")
            search_table.add_column("Description", style="dim")
            
            for result in results:
                metadata = result['metadata']
                similarity = f"{result['similarity']:.3f}"
                chunk_type = metadata.get('type', 'unknown')
                name = metadata.get('name', 'unnamed')
                description = result['document']
                
                search_table.add_row(similarity, chunk_type, name, description)
            
            console.print(search_table)
        else:
            console.print("[yellow]No results found[/yellow]")
    
    # Step 5: Show database statistics
    console.print("\n[bold]Step 5: Database statistics...[/bold]")
    stats = db.get_statistics()
    
    stats_table = Table(title="Database Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Chunks", str(stats['total_chunks']))
    stats_table.add_row("Average Content Length", f"{stats['avg_content_length']:.1f} chars")
    
    for chunk_type, count in stats['chunk_types'].items():
        stats_table.add_row(f"  {chunk_type.title()} Chunks", str(count))
    
    console.print(stats_table)
    
    # Step 6: Demonstrate change detection
    console.print("\n[bold]Step 6: Demonstrating change detection...[/bold]")
    
    # Modify one of the sample files
    calculator_file = sample_dir / "calculator.py"
    with open(calculator_file, "a") as f:
        f.write('''

def power(base, exponent):
    """Calculate base raised to the power of exponent."""
    return base ** exponent
''')
    
    console.print("[yellow]Modified calculator.py by adding a power function...[/yellow]")
    
    # Re-analyze to detect changes
    new_results = analyzer.analyze_path(sample_dir, recursive=True)
    
    changes_table = Table(title="Detected Changes")
    changes_table.add_column("Change Type", style="cyan")
    changes_table.add_column("Count", style="green")
    
    for change_type, chunk_ids in new_results['changes'].items():
        changes_table.add_row(change_type.title(), str(len(chunk_ids)))
    
    console.print(changes_table)
    
    console.print("\n[bold green]Demo completed successfully![/bold green]")
    console.print("\n[dim]You can now use the CLI tool with: python main.py --help[/dim]")


if __name__ == "__main__":
    run_demo()
