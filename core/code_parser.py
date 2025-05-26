"""Code parsing utilities for extracting meaningful code chunks."""

import ast
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import core.config as config


@dataclass
class CodeChunk:
    """Represents a meaningful unit of code."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'import', 'global'
    name: str
    parent: Optional[str] = None  # For methods, the parent class name
    docstring: Optional[str] = None
    complexity_score: int = 0


class PythonParser:
    """Parser for Python code using AST."""

    def __init__(self):
        self.chunks: List[CodeChunk] = []

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a Python file and extract code chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            lines = content.splitlines()

            self.chunks = []
            self._extract_chunks(tree, lines, content)

            return self.chunks

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []

    def _extract_chunks(self, tree: ast.AST, lines: List[str], full_content: str):
        """Extract meaningful chunks from AST."""

        # Process only top-level nodes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Top-level function, no parent class
                self._process_function(node, lines, full_content)
            elif isinstance(node, ast.ClassDef):
                # Process class and its methods inside
                self._process_class(node, lines, full_content)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._process_import(node, lines)


    def _process_function(self, node: ast.FunctionDef, lines: List[str], full_content: str, parent_class: str = None):
        """Process a function definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get the actual content
        content = '\n'.join(lines[start_line-1:end_line])

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Calculate complexity (simple metric: number of decision points)
        complexity = self._calculate_complexity(node)

        chunk = CodeChunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type='method' if parent_class else 'function',
            name=node.name,
            parent=parent_class,
            docstring=docstring,
            complexity_score=complexity
        )

        self.chunks.append(chunk)

    def _process_class(self, node: ast.ClassDef, lines: List[str], full_content: str):
        """Process a class definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get class content (without methods, we'll process them separately)
        class_header_end = start_line
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                break
            class_header_end = getattr(child, 'end_lineno', child.lineno)

        content = '\n'.join(lines[start_line-1:class_header_end])
        docstring = ast.get_docstring(node)

        chunk = CodeChunk(
            content=content,
            start_line=start_line,
            end_line=class_header_end,
            chunk_type='class',
            name=node.name,
            docstring=docstring
        )

        self.chunks.append(chunk)

        # Process methods within the class
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self._process_function(child, lines, full_content, parent_class=node.name)

    def _process_import(self, node: ast.AST, lines: List[str]):
        """Process import statements."""
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line)

        content = '\n'.join(lines[start_line-1:end_line])

        chunk = CodeChunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type='import',
            name='imports'
        )

        self.chunks.append(chunk)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


class UniversalParser:
    """Parser that can handle multiple programming languages."""

    def __init__(self):
        self.parsers = {
            '.py': PythonParser()
        }

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a file based on its extension."""
        suffix = file_path.suffix.lower()

        if suffix in self.parsers:
            return self.parsers[suffix].parse_file(file_path)
        else:
            # Fallback: treat as plain text and create simple chunks
            return self._parse_as_text(file_path)

    def _parse_as_text(self, file_path: Path) -> List[CodeChunk]:
        """Fallback parser for unsupported file types."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple chunking by functions/classes using regex
            chunks = []
            lines = content.splitlines()

            # This is a very basic implementation
            chunk = CodeChunk(
                content=content,
                start_line=1,
                end_line=len(lines),
                chunk_type='file',
                name=file_path.stem
            )
            chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Error parsing {file_path} as text: {e}")
            return []
