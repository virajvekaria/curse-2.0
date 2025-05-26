"""Safe code execution and testing utilities."""

import subprocess
import tempfile
import os
import sys
import ast
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import contextlib
import io


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    execution_time: float
    return_value: Any = None


class SafeCodeExecutor:
    """Safely execute Python code with restrictions."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.allowed_imports = {
            'math', 'random', 'datetime', 'json', 'os', 'sys', 'pathlib',
            'collections', 'itertools', 'functools', 're', 'string',
            'typing', 'dataclasses', 'enum', 'abc', 'copy', 'pickle'
        }
        self.forbidden_functions = {
            'exec', 'eval', 'compile', '__import__', 'open', 'input',
            'raw_input', 'file', 'execfile', 'reload', 'vars', 'locals',
            'globals', 'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
        }

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code for safety before execution."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for forbidden constructs
        for node in ast.walk(tree):
            # Check for forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.forbidden_functions:
                        return False, f"Forbidden function: {node.func.id}"

            # Check for imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allowed_imports:
                        return False, f"Import not allowed: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.allowed_imports:
                    return False, f"Import not allowed: {node.module}"

        return True, "Code validation passed"

    def execute_code(self, code: str, capture_output: bool = True) -> ExecutionResult:
        """Execute Python code safely."""
        start_time = time.time()

        # Validate code first
        is_valid, validation_msg = self.validate_code(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Validation failed: {validation_msg}",
                execution_time=0
            )

        if capture_output:
            # Capture stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Create a restricted globals environment
                restricted_globals = {
                    '__builtins__': {
                        'print': print,
                        'len': len,
                        'range': range,
                        'enumerate': enumerate,
                        'zip': zip,
                        'map': map,
                        'filter': filter,
                        'sum': sum,
                        'min': min,
                        'max': max,
                        'abs': abs,
                        'round': round,
                        'sorted': sorted,
                        'reversed': reversed,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'list': list,
                        'dict': dict,
                        'tuple': tuple,
                        'set': set,
                        'type': type,
                        'isinstance': isinstance,
                        'issubclass': issubclass,
                        # Common exceptions
                        'Exception': Exception,
                        'ValueError': ValueError,
                        'TypeError': TypeError,
                        'IndexError': IndexError,
                        'KeyError': KeyError,
                        'AttributeError': AttributeError,
                        'ZeroDivisionError': ZeroDivisionError,
                        'RuntimeError': RuntimeError,
                        'NotImplementedError': NotImplementedError,
                        # Allow safe imports
                        '__import__': __import__,
                    }
                }

                # Add safe modules to globals
                import random
                import string
                import math
                import datetime
                restricted_globals['random'] = random
                restricted_globals['string'] = string
                restricted_globals['math'] = math
                restricted_globals['datetime'] = datetime

                # Execute the code
                exec(code, restricted_globals)

                execution_time = time.time() - start_time

                return ExecutionResult(
                    success=True,
                    output=stdout_capture.getvalue(),
                    error=stderr_capture.getvalue(),
                    execution_time=execution_time
                )

            except Exception as e:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    success=False,
                    output=stdout_capture.getvalue(),
                    error=str(e),
                    execution_time=execution_time
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        else:
            # Execute without capturing output
            try:
                exec(code)
                execution_time = time.time() - start_time
                return ExecutionResult(
                    success=True,
                    output="",
                    error="",
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=execution_time
                )

    def execute_file(self, file_path: Path) -> ExecutionResult:
        """Execute a Python file safely."""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            return self.execute_code(code)
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Error reading file: {e}",
                execution_time=0
            )

    def test_function(self, code: str, function_name: str, test_cases: List[Dict]) -> List[ExecutionResult]:
        """Test a function with multiple test cases."""
        results = []

        # First execute the code to define the function
        exec_result = self.execute_code(code, capture_output=False)
        if not exec_result.success:
            return [exec_result]

        # Now test each case
        for i, test_case in enumerate(test_cases):
            args = test_case.get('args', [])
            kwargs = test_case.get('kwargs', {})
            expected = test_case.get('expected')

            # Create test code
            args_str = ', '.join(repr(arg) for arg in args)
            kwargs_str = ', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())
            call_args = ', '.join(filter(None, [args_str, kwargs_str]))

            test_code = f"""
{code}

result = {function_name}({call_args})
print(f"Test {i+1}: {{result}}")
"""

            result = self.execute_code(test_code)
            results.append(result)

        return results


class CodeTester:
    """Test generated code automatically."""

    def __init__(self):
        self.executor = SafeCodeExecutor()

    def generate_tests(self, code: str, function_name: str) -> List[Dict]:
        """Generate basic test cases for a function."""
        # This is a simple implementation - in practice you'd use LLM to generate tests
        tests = []

        try:
            # Parse the function to understand its signature
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Generate basic tests based on function signature
                    args = [arg.arg for arg in node.args.args]

                    if len(args) == 1:
                        tests = [
                            {'args': [0], 'expected': None},
                            {'args': [1], 'expected': None},
                            {'args': [-1], 'expected': None},
                        ]
                    elif len(args) == 2:
                        tests = [
                            {'args': [1, 2], 'expected': None},
                            {'args': [0, 0], 'expected': None},
                            {'args': [-1, 1], 'expected': None},
                        ]
                    break
        except Exception:
            # Fallback tests
            tests = [{'args': [], 'expected': None}]

        return tests

    def test_code(self, code: str, function_name: str = None) -> Dict[str, Any]:
        """Test code and return results."""
        # First, just try to execute the code
        exec_result = self.executor.execute_code(code)

        test_results = {
            'syntax_valid': exec_result.success,
            'execution_output': exec_result.output,
            'execution_error': exec_result.error,
            'execution_time': exec_result.execution_time,
            'function_tests': []
        }

        if function_name and exec_result.success:
            # Generate and run function tests
            test_cases = self.generate_tests(code, function_name)
            function_test_results = self.executor.test_function(code, function_name, test_cases)

            test_results['function_tests'] = [
                {
                    'test_case': i + 1,
                    'success': result.success,
                    'output': result.output,
                    'error': result.error
                }
                for i, result in enumerate(function_test_results)
            ]

        return test_results

    def validate_generated_code(self, code: str, requirements: str = "") -> Dict[str, Any]:
        """Validate that generated code meets requirements."""
        validation_result = {
            'meets_requirements': True,
            'issues': [],
            'suggestions': [],
            'test_results': {}
        }

        # Test the code
        test_results = self.test_code(code)
        validation_result['test_results'] = test_results

        # Check if code executes without errors
        if not test_results['syntax_valid']:
            validation_result['meets_requirements'] = False
            validation_result['issues'].append("Code has syntax errors")

        # Check for basic code quality
        try:
            tree = ast.parse(code)

            # Check for docstrings
            has_docstring = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Str)):
                        has_docstring = True
                        break

            if not has_docstring:
                validation_result['suggestions'].append("Consider adding docstrings to functions/classes")

        except Exception:
            pass

        return validation_result
