"""
Augment-style iterative coding engine that writes, executes, reads terminal, and self-corrects.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.llm_integration import LLMManager
from core.code_executor import SafeCodeExecutor, ExecutionResult
from core.analyzer import CodebaseAnalyzer


@dataclass
class IterationResult:
    """Result of a single iteration in the coding loop."""
    iteration: int
    code_generated: str
    execution_result: ExecutionResult
    terminal_output: str
    error_analysis: str
    correction_needed: bool
    success: bool


@dataclass
class AugmentSession:
    """Complete Augment-style coding session with iterations."""
    prompt: str
    codebase_path: Path
    iterations: List[IterationResult]
    final_success: bool
    total_iterations: int
    final_code: str
    execution_time: float


class AugmentEngine:
    """Augment-style iterative coding engine with self-correction."""

    def __init__(self):
        self.console = Console()
        self.llm_manager = LLMManager()
        self.executor = SafeCodeExecutor()
        self.analyzer = CodebaseAnalyzer()
        self.max_iterations = 5
        self.current_session: Optional[AugmentSession] = None

    def augment_code(self, prompt: str, codebase_path: Path, execute_in_terminal: bool = True) -> AugmentSession:
        """
        Main Augment-style coding function:
        1. Understand codebase
        2. Generate code
        3. Execute and read terminal
        4. If errors, analyze and correct
        5. Repeat until success or max iterations
        """
        start_time = time.time()

        self.console.print(Panel(
            f"🤖 **AUGMENT ENGINE ACTIVATED**\n\n"
            f"📁 Codebase: {codebase_path}\n"
            f"🎯 Prompt: {prompt}\n"
            f"🔄 Max Iterations: {self.max_iterations}\n"
            f"⚡ Terminal Execution: {'Yes' if execute_in_terminal else 'No'}",
            title="🚀 Augment Coding Session",
            border_style="bold blue"
        ))

        # Step 1: Understand the codebase
        codebase_context = self._understand_codebase(codebase_path)

        # Initialize session
        session = AugmentSession(
            prompt=prompt,
            codebase_path=codebase_path,
            iterations=[],
            final_success=False,
            total_iterations=0,
            final_code="",
            execution_time=0.0
        )
        self.current_session = session

        # Step 2: Iterative coding loop
        current_code = ""
        error_context = ""

        for iteration in range(1, self.max_iterations + 1):
            self.console.print(f"\n[bold cyan]🔄 ITERATION {iteration}[/bold cyan]")

            # Generate/correct code
            if iteration == 1:
                code = self._generate_initial_code(prompt, codebase_context)
            else:
                code = self._correct_code(current_code, error_context, prompt, codebase_context)

            # Execute the code
            if execute_in_terminal:
                execution_result, terminal_output = self._execute_in_terminal(code, codebase_path)
            else:
                execution_result = self.executor.execute_code(code)
                terminal_output = execution_result.output

            # Analyze results
            success = execution_result.success and not self._has_runtime_errors(terminal_output)

            if not success:
                error_analysis = self._analyze_errors(code, execution_result, terminal_output)
                correction_needed = True
            else:
                error_analysis = "Code executed successfully!"
                correction_needed = False

            # Record iteration
            iteration_result = IterationResult(
                iteration=iteration,
                code_generated=code,
                execution_result=execution_result,
                terminal_output=terminal_output,
                error_analysis=error_analysis,
                correction_needed=correction_needed,
                success=success
            )
            session.iterations.append(iteration_result)

            # Display iteration results
            self._display_iteration_result(iteration_result)

            if success:
                self.console.print(f"[bold green]✅ SUCCESS in iteration {iteration}![/bold green]")
                session.final_success = True
                session.final_code = code
                break
            else:
                current_code = code
                error_context = f"Previous attempt failed with: {error_analysis}"
                self.console.print(f"[yellow]⚠️  Iteration {iteration} failed, trying again...[/yellow]")

        # Finalize session
        session.total_iterations = len(session.iterations)
        session.execution_time = time.time() - start_time

        if not session.final_success:
            self.console.print(f"[red]❌ Failed to solve after {self.max_iterations} iterations[/red]")
            session.final_code = current_code  # Use last attempt

        self._display_session_summary(session)
        return session

    def _understand_codebase(self, codebase_path: Path) -> str:
        """Understand the codebase structure and context."""
        self.console.print("[bold blue]🧠 Understanding codebase...[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=None)

            # Analyze the codebase
            analysis_result = self.analyzer.analyze_path(codebase_path, recursive=True)

            # Get file structure
            files_info = []
            for root, dirs, files in os.walk(codebase_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
                        rel_path = os.path.relpath(os.path.join(root, file), codebase_path)
                        files_info.append(rel_path)

            progress.update(task, description="Building context...")

            # Create comprehensive context
            context = f"""
CODEBASE ANALYSIS:
- Files processed: {analysis_result['files_processed']}
- Code chunks: {analysis_result['chunks_extracted']}
- Embeddings: {analysis_result['embeddings_generated']}

KEY FILES:
{chr(10).join(f"- {file}" for file in files_info[:10])}

STRUCTURE: {len(files_info)} total code files
"""

        self.console.print("[green]✓ Codebase understanding complete[/green]")
        return context

    def _generate_initial_code(self, prompt: str, codebase_context: str) -> str:
        """Generate initial code based on prompt and codebase context."""
        self.console.print("[bold yellow]🎨 Generating initial code...[/bold yellow]")

        generation_prompt = f"""You are an expert programmer working with an existing codebase.

CODEBASE CONTEXT:
{codebase_context}

USER REQUEST: {prompt}

Generate complete, executable Python code that:
1. Integrates well with the existing codebase
2. Follows the project's patterns and style
3. Includes proper error handling
4. Has clear documentation
5. Can be executed immediately

IMPORTANT: Generate ONLY the Python code, no explanations or markdown formatting.
The code should be ready to save to a file and execute."""

        code = self.llm_manager.generate_code(generation_prompt)

        # Clean up the code (remove markdown if present)
        if code.startswith('```python'):
            code = code[9:]
        if code.endswith('```'):
            code = code[:-3]

        return code.strip()

    def _execute_in_terminal(self, code: str, codebase_path: Path) -> Tuple[ExecutionResult, str]:
        """Execute code in terminal and capture all output."""
        self.console.print("[bold green]⚡ Executing in terminal...[/bold green]")

        # Create temporary file
        temp_file = codebase_path / "temp_augment_code.py"

        try:
            # Ensure directory exists
            codebase_path.mkdir(parents=True, exist_ok=True)

            # Check for required packages and install them
            required_packages = self._extract_required_packages(code)
            if required_packages:
                self.console.print(f"[dim]Installing required packages: {required_packages}[/dim]")
                install_success = self._install_packages(required_packages)
                if not install_success:
                    self.console.print("[yellow]⚠️ Package installation failed, proceeding with standard library only[/yellow]")

            # Write code to file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)

            # Verify file was created
            if not temp_file.exists():
                raise FileNotFoundError(f"Failed to create temp file: {temp_file}")

            # Execute in terminal with enhanced environment
            result = subprocess.run(
                ['python', str(temp_file.absolute())],
                cwd=str(codebase_path.absolute()),
                capture_output=True,
                text=True,
                timeout=30,
                env=self._get_enhanced_env()
            )

            # Create execution result
            execution_result = ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time=0.0
            )

            terminal_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nReturn Code: {result.returncode}"

            return execution_result, terminal_output

        except subprocess.TimeoutExpired:
            execution_result = ExecutionResult(
                success=False,
                output="",
                error="Execution timed out after 30 seconds",
                execution_time=30.0
            )
            return execution_result, "TIMEOUT: Code execution exceeded 30 seconds"

        except Exception as e:
            execution_result = ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0.0
            )
            return execution_result, f"EXECUTION ERROR: {e}"

        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

    def _has_runtime_errors(self, terminal_output: str) -> bool:
        """Check if terminal output indicates runtime errors."""
        error_indicators = [
            'Traceback (most recent call last)',
            'Error:', 'Exception:', 'SyntaxError:', 'NameError:',
            'TypeError:', 'ValueError:', 'ImportError:', 'ModuleNotFoundError:'
        ]

        return any(indicator in terminal_output for indicator in error_indicators)

    def _analyze_errors(self, code: str, execution_result: ExecutionResult, terminal_output: str) -> str:
        """Analyze errors and provide detailed analysis."""
        self.console.print("[bold red]🔍 Analyzing errors...[/bold red]")

        # Check for common issues first
        if "can't open file" in terminal_output or "No such file or directory" in terminal_output:
            return """FILE CREATION ERROR: The temporary file could not be created or found.
This is likely a system/path issue, not a code issue.
The code itself appears to be syntactically correct.
Try removing any problematic imports or file references."""

        if "ModuleNotFoundError" in terminal_output or "ImportError" in terminal_output:
            # Extract the specific module that failed
            import_error_module = "unknown"
            if "No module named" in terminal_output:
                start = terminal_output.find("No module named '") + 17
                end = terminal_output.find("'", start)
                if start > 16 and end > start:
                    import_error_module = terminal_output[start:end]

            return f"""IMPORT ERROR: The code is trying to import '{import_error_module}' which is not available.
SOLUTION: Rewrite the code to use only Python standard library modules.
- Replace pandas with built-in csv module
- Replace external libraries with standard library equivalents
- Implement functionality from scratch using basic Python
- Use io.StringIO instead of pandas.compat.StringIO
- Create simple data structures instead of DataFrames"""

        if "pd.compat.StringIO" in terminal_output or "pandas" in terminal_output:
            return """PANDAS ERROR: The code is using pandas which is not available.
SOLUTION: Rewrite to use Python standard library:
- Use csv.reader() instead of pd.read_csv()
- Use io.StringIO() instead of pd.compat.StringIO()
- Use lists and dictionaries instead of DataFrames
- Implement statistics calculations manually or use statistics module"""

        if "NotImplementedError" in terminal_output:
            return """LOGIC ERROR: The code is calling a base class method that raises NotImplementedError.
SOLUTION: Use the specific algorithm classes instead of the base class.
- Replace SortingAlgorithm(data).sort() with algorithm(data).sort()
- Make sure to instantiate the correct algorithm class for each test"""

        analysis_prompt = f"""Analyze this code execution failure and provide specific guidance for fixing it:

CODE THAT FAILED:
```python
{code}
```

EXECUTION RESULT:
- Success: {execution_result.success}
- Error: {execution_result.error}

TERMINAL OUTPUT:
{terminal_output}

Provide a concise analysis of:
1. What exactly went wrong
2. The root cause of the error
3. Specific steps to fix it
4. Any missing imports or dependencies

Be specific and actionable."""

        return self.llm_manager.generate_code(analysis_prompt)

    def _correct_code(self, previous_code: str, error_context: str, original_prompt: str, codebase_context: str) -> str:
        """Generate corrected code based on previous errors."""
        self.console.print("[bold yellow]🔧 Correcting code based on errors...[/bold yellow]")

        # Apply specific fixes based on error type
        if "IMPORT ERROR" in error_context or "ModuleNotFoundError" in error_context:
            return self._fix_import_errors(previous_code, original_prompt)
        elif "PANDAS ERROR" in error_context:
            return self._fix_pandas_errors(previous_code, original_prompt)
        else:
            # General correction
            return self._general_correction(previous_code, error_context, original_prompt, codebase_context)

    def _fix_import_errors(self, previous_code: str, original_prompt: str) -> str:
        """Fix import errors by regenerating code with standard library only."""
        self.console.print("[dim]Applying smart import error fixes...[/dim]")

        # Instead of trying to patch the broken code, regenerate it completely
        # with explicit constraints about using only standard library

        regeneration_prompt = f"""CRITICAL: The previous code failed because it used external libraries that are not available.

ORIGINAL REQUEST: {original_prompt}

STRICT REQUIREMENTS:
- Use ONLY Python standard library (math, random, json, csv, etc.)
- NO external libraries: NO numpy, pandas, sklearn, tensorflow, keras, torch, pygame, etc.
- Implement everything from scratch using basic Python
- For neural networks: implement matrix operations manually with lists
- For data: generate synthetic data in the code
- For math operations: use basic Python math operations

Generate COMPLETE, WORKING Python code that implements the request using ONLY standard library:"""

        # Generate completely new code with strict constraints
        new_code = self.llm_manager.generate_code(regeneration_prompt)

        # Clean up any remaining problematic imports
        lines = new_code.split('\n')
        fixed_lines = []
        removed_imports = []

        for line in lines:
            # Track and remove ALL problematic imports
            if 'import' in line and any(bad in line for bad in [
                'nltk', 'pandas', 'numpy', 'sklearn', 'augment_', 'pygame',
                'tensorflow', 'torch', 'keras', 'cv2', 'matplotlib', 'scipy',
                'seaborn', 'plotly', 'bokeh', 'dash'
            ]):
                removed_imports.append(line.strip())
                continue

            # Fix common import-related function calls
            if 'from augment_' in line or 'import augment_' in line:
                continue  # Skip these entirely

            # Replace problematic function calls with simple implementations
            line = line.replace('word_tokenize(', 'simple_tokenize(')
            line = line.replace('SentimentIntensityAnalyzer()', 'simple_sentiment_score(')
            line = line.replace('pd.', '')
            line = line.replace('np.', '')
            line = line.replace('tf.', '')
            line = line.replace('torch.', '')

            fixed_lines.append(line)

        # Add replacement functions at the top if needed
        if any('simple_tokenize' in line for line in fixed_lines):
            tokenize_func = '''
def simple_tokenize(text):
    """Simple word tokenization."""
    import re
    return re.findall(r'\\b\\w+\\b', text.lower())
'''
            fixed_lines.insert(0, tokenize_func)

        if any('simple_sentiment_score' in line for line in fixed_lines):
            sentiment_func = '''
def simple_sentiment_score(text):
    """Simple sentiment analysis."""
    positive = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy']
    negative = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'horrible']

    words = simple_tokenize(text)
    pos_count = sum(1 for word in words if word in positive)
    neg_count = sum(1 for word in words if word in negative)

    if pos_count + neg_count == 0:
        return 0.0
    return (pos_count - neg_count) / (pos_count + neg_count)
'''
            fixed_lines.insert(0, sentiment_func)

        self.console.print(f"[dim]Removed imports: {removed_imports}[/dim]")
        return '\n'.join(fixed_lines)

    def _fix_pandas_errors(self, previous_code: str, original_prompt: str) -> str:
        """Fix pandas-related errors by using standard library."""
        self.console.print("[dim]Applying pandas error fixes...[/dim]")

        # Replace pandas with csv module
        fixed_code = previous_code.replace('import pandas as pd', 'import csv\nimport io')
        fixed_code = fixed_code.replace('pd.read_csv(pd.compat.StringIO(', 'list(csv.reader(io.StringIO(')
        fixed_code = fixed_code.replace('pd.read_csv(', 'list(csv.reader(')

        return fixed_code

    def _general_correction(self, previous_code: str, error_context: str, original_prompt: str, codebase_context: str) -> str:
        """Advanced correction using intelligent code analysis and LLM."""
        self.console.print("[dim]Applying intelligent error correction...[/dim]")

        # First, try smart pattern-based fixes
        smart_fixed = self._apply_smart_fixes(previous_code, error_context)
        if smart_fixed != previous_code:
            self.console.print("[dim]Applied smart pattern fixes[/dim]")
            return smart_fixed

        # Perform deep code analysis
        code_analysis = self._analyze_code_structure(previous_code)
        error_analysis = self._analyze_error_type(error_context)

        # Create intelligent correction prompt
        correction_prompt = f"""You are an expert Python programmer. Analyze and fix this code systematically.

ORIGINAL TASK: {original_prompt}

FAILED CODE:
```python
{previous_code}
```

ERROR DETAILS: {error_context}

CODE ANALYSIS:
{code_analysis}

ERROR TYPE: {error_analysis}

CORRECTION STRATEGY:
1. Identify the exact cause of the error
2. Check if functions/methods are defined before being called
3. Verify all imports are correct
4. Ensure proper syntax and indentation
5. Fix ONLY the specific error, don't add extra functionality

Generate the corrected Python code that fixes the specific error:"""

        corrected_code = self.llm_manager.generate_code(correction_prompt)

        # Clean up the code
        corrected_code = self._clean_generated_code(corrected_code)

        return corrected_code

    def _analyze_code_structure(self, code: str) -> str:
        """Analyze the structure of the code to understand what's defined."""
        import ast
        import re

        analysis = []

        try:
            # Parse the AST to find defined functions and classes
            tree = ast.parse(code)

            functions = []
            classes = []
            methods = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    # Find methods in this class
                    class_methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_methods.append(item.name)
                    methods[node.name] = class_methods

            analysis.append(f"DEFINED FUNCTIONS: {functions}")
            analysis.append(f"DEFINED CLASSES: {classes}")
            for class_name, class_methods in methods.items():
                analysis.append(f"METHODS IN {class_name}: {class_methods}")

        except SyntaxError as e:
            analysis.append(f"SYNTAX ERROR: {e}")

        # Also check for function calls
        function_calls = re.findall(r'(\w+)\s*\(', code)
        method_calls = re.findall(r'\.(\w+)\s*\(', code)

        analysis.append(f"FUNCTION CALLS FOUND: {list(set(function_calls))}")
        analysis.append(f"METHOD CALLS FOUND: {list(set(method_calls))}")

        return "\n".join(analysis)

    def _analyze_error_type(self, error_context: str) -> str:
        """Analyze the type of error and provide specific guidance."""
        error_type = "UNKNOWN"
        guidance = ""

        if "NameError" in error_context:
            if "is not defined" in error_context:
                error_type = "UNDEFINED_FUNCTION_OR_VARIABLE"
                guidance = "A function or variable is being called/used but not defined. Check if the function exists or needs to be imported."

        elif "AttributeError" in error_context:
            if "has no attribute" in error_context:
                error_type = "UNDEFINED_METHOD"
                guidance = "A method is being called on an object that doesn't have that method. Check the class definition."

        elif "ModuleNotFoundError" in error_context:
            error_type = "MISSING_IMPORT"
            guidance = "A module is being imported that doesn't exist. Either install it or use standard library alternatives."

        elif "SyntaxError" in error_context:
            error_type = "SYNTAX_ERROR"
            guidance = "There's a syntax error in the code. Check indentation, parentheses, and Python syntax."

        elif "IndentationError" in error_context:
            error_type = "INDENTATION_ERROR"
            guidance = "Incorrect indentation. Python requires consistent indentation."

        elif "TypeError" in error_context:
            error_type = "TYPE_ERROR"
            guidance = "Wrong type being used. Check function arguments and return types."

        return f"{error_type}: {guidance}"

    def _apply_smart_fixes(self, code: str, error_context: str) -> str:
        """Apply pattern-based smart fixes for common errors."""
        fixed_code = code

        # Fix missing imports
        if "NameError: name 'math'" in error_context:
            if "import math" not in fixed_code:
                fixed_code = "import math\n" + fixed_code

        if "NameError: name 'random'" in error_context:
            if "import random" not in fixed_code:
                fixed_code = "import random\n" + fixed_code

        # Fix NotImplementedError by replacing base class calls
        if "NotImplementedError" in error_context:
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                if "SortingAlgorithm(data).sort()" in line:
                    lines[i] = line.replace("SortingAlgorithm(data).sort()", "algorithm(data).sort()")
            fixed_code = '\n'.join(lines)

        # Fix module import errors by removing problematic imports
        if "ModuleNotFoundError" in error_context and "augment_" in error_context:
            lines = fixed_code.split('\n')
            fixed_lines = []
            for line in lines:
                if not ("from augment_" in line or "import augment_" in line):
                    fixed_lines.append(line)
            fixed_code = '\n'.join(fixed_lines)

        return fixed_code

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code from LLM."""
        # Remove markdown formatting
        if code.startswith('```python'):
            code = code[9:]
        if code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]

        # Remove common LLM artifacts
        code = code.replace('```python', '').replace('```', '')

        # Strip whitespace
        return code.strip()

    def _display_iteration_result(self, result: IterationResult):
        """Display the results of a single iteration."""
        status_color = "green" if result.success else "red"
        status_icon = "✅" if result.success else "❌"

        self.console.print(f"\n[{status_color}]{status_icon} Iteration {result.iteration} Result[/{status_color}]")

        # Show code preview
        code_preview = result.code_generated[:200] + "..." if len(result.code_generated) > 200 else result.code_generated
        syntax = Syntax(code_preview, "python", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Generated Code"))

        # Show execution output
        if result.terminal_output:
            output_preview = result.terminal_output[:300] + "..." if len(result.terminal_output) > 300 else result.terminal_output
            self.console.print(Panel(output_preview, title="Terminal Output"))

        # Show error analysis if needed
        if not result.success:
            self.console.print(Panel(result.error_analysis, title="Error Analysis", border_style="red"))

    def _display_session_summary(self, session: AugmentSession):
        """Display final session summary."""
        title = "🎉 SESSION COMPLETED" if session.final_success else "⚠️ SESSION INCOMPLETE"
        color = "green" if session.final_success else "yellow"

        summary = f"""
[bold {color}]{title}[/bold {color}]

📊 **Session Statistics:**
• Total Iterations: {session.total_iterations}
• Final Success: {'✅ Yes' if session.final_success else '❌ No'}
• Execution Time: {session.execution_time:.2f} seconds
• Final Code Length: {len(session.final_code)} characters

🎯 **Original Prompt:** {session.prompt}

📁 **Codebase:** {session.codebase_path}
"""

        self.console.print(Panel(summary, border_style=color))

        if session.final_success and session.final_code:
            self.console.print("\n[bold green]🏆 FINAL WORKING CODE:[/bold green]")
            syntax = Syntax(session.final_code, "python", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="Final Code"))

    def save_final_code(self, session: AugmentSession, filename: str = None) -> Path:
        """Save the final working code to a file."""
        if not session.final_success:
            raise ValueError("Cannot save code from unsuccessful session")

        if not filename:
            # Generate filename based on prompt
            safe_prompt = "".join(c for c in session.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"augment_{safe_prompt.replace(' ', '_').lower()}.py"

        file_path = session.codebase_path / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(session.final_code)

        self.console.print(f"[green]💾 Final code saved to: {file_path}[/green]")
        return file_path

    def get_session_report(self, session: AugmentSession) -> str:
        """Generate a detailed report of the coding session."""
        report = f"""
# Augment Coding Session Report

## Session Overview
- **Prompt**: {session.prompt}
- **Codebase**: {session.codebase_path}
- **Success**: {'✅ Yes' if session.final_success else '❌ No'}
- **Total Iterations**: {session.total_iterations}
- **Execution Time**: {session.execution_time:.2f} seconds

## Iteration Details
"""

        for iteration in session.iterations:
            report += f"""
### Iteration {iteration.iteration}
- **Success**: {'✅' if iteration.success else '❌'}
- **Code Length**: {len(iteration.code_generated)} characters
- **Error Analysis**: {iteration.error_analysis[:100]}...
"""

        if session.final_success:
            report += f"""
## Final Working Code
```python
{session.final_code}
```
"""

        return report

    def _extract_required_packages(self, code: str) -> List[str]:
        """Extract required packages from import statements."""
        import re

        # Common package mappings
        package_mappings = {
            'numpy': 'numpy',
            'np': 'numpy',
            'pandas': 'pandas',
            'pd': 'pandas',
            'matplotlib': 'matplotlib',
            'plt': 'matplotlib',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'requests': 'requests',
            'bs4': 'beautifulsoup4',
            'pygame': 'pygame',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'psutil': 'psutil'
        }

        required_packages = set()

        # Find import statements
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+)\s+as\s+\w+'
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if match in package_mappings:
                    required_packages.add(package_mappings[match])

        # Filter out standard library modules
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'csv', 'math', 'random',
            'collections', 'itertools', 'functools', 're', 'pathlib', 'typing',
            'threading', 'queue', 'subprocess', 'io', 'abc', 'dataclasses'
        }

        return [pkg for pkg in required_packages if pkg not in stdlib_modules]

    def _install_packages(self, packages: List[str]) -> bool:
        """Install required packages using pip."""
        try:
            for package in packages:
                self.console.print(f"[dim]Installing {package}...[/dim]")
                result = subprocess.run(
                    ['pip', 'install', package],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    self.console.print(f"[yellow]Failed to install {package}: {result.stderr}[/yellow]")
                    return False
            return True
        except Exception as e:
            self.console.print(f"[yellow]Package installation error: {e}[/yellow]")
            return False

    def _get_enhanced_env(self):
        """Get enhanced environment variables for subprocess execution."""
        import os
        env = os.environ.copy()

        # Ensure Python can find installed packages
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = env['PYTHONPATH']

        return env
