"""
Intelligent Augment Engine - Redesigned for complex tasks.

This system:
1. Plans the complete architecture before coding
2. Validates dependencies and structure
3. Generates complete, self-contained solutions
4. Uses intelligent error correction with full context understanding
"""

import ast
import re
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

@dataclass
class TaskPlan:
    """Complete plan for implementing a task."""
    task_description: str
    required_components: List[str]  # Classes, functions, etc.
    required_imports: List[str]     # External libraries needed
    implementation_order: List[str] # Order to implement components
    dependencies: Dict[str, List[str]]  # Component dependencies
    estimated_complexity: int      # 1-10 scale
    file_structure: Dict[str, List[str]] = None  # filename -> components mapping

@dataclass
class ComponentSpec:
    """Specification for a single component."""
    name: str
    type: str  # 'class', 'function', 'variable'
    signature: str
    dependencies: List[str]
    implementation: str = ""

@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    missing_components: List[str]
    undefined_references: List[str]
    import_issues: List[str]
    syntax_errors: List[str]
    suggestions: List[str]

class IntelligentAugmentEngine:
    """Redesigned Augment Engine that actually works for complex tasks."""

    def __init__(self):
        self.console = Console()
        self.max_iterations = 5
        self.current_plan: Optional[TaskPlan] = None
        self.implemented_components: Set[str] = set()

    def augment_code_intelligent(self, prompt: str, codebase_path: Path) -> Dict[str, Any]:
        """
        Main method: Intelligently implement complex tasks.

        Process:
        1. Analyze and plan the complete task
        2. Validate the plan
        3. Generate complete implementation
        4. Validate before execution
        5. Execute and fix issues intelligently
        """
        self.console.print(Panel(
            f"🧠 **INTELLIGENT AUGMENT ENGINE**\n\n"
            f"Task: {prompt}\n"
            f"Working Directory: {codebase_path}\n\n"
            f"Process:\n"
            f"1. 📋 Analyze and create complete implementation plan\n"
            f"2. 🔍 Validate plan and dependencies\n"
            f"3. 🎨 Generate complete, self-contained code\n"
            f"4. ✅ Pre-validate code structure\n"
            f"5. ⚡ Execute with intelligent error handling",
            title="🚀 Intelligent Augment Engine",
            border_style="bold blue"
        ))

        start_time = time.time()

        try:
            # Step 0: Read and understand existing codebase
            self.console.print("📖 Reading and understanding existing codebase...")
            codebase_context = self._read_codebase(codebase_path)

            # Step 1: Create comprehensive plan with codebase context
            self.console.print("📋 Creating comprehensive implementation plan...")
            plan = self._create_task_plan(prompt, codebase_context)
            self.current_plan = plan

            # Display the detailed plan
            self._display_plan(plan)

            # Step 2: Validate plan feasibility
            self.console.print("🔍 Validating plan feasibility...")
            plan_validation = self._validate_plan(plan)
            if not plan_validation['feasible']:
                return self._create_failure_result(f"Plan validation failed: {plan_validation['issues']}")

            # Display validation results
            self._display_validation_results(plan_validation)

            # Step 2.5: Install required dependencies
            if plan.required_imports:
                self.console.print("📦 Installing required dependencies...")
                installation_result = self._install_dependencies(plan.required_imports)
                if not installation_result['success']:
                    self.console.print(f"[yellow]⚠️ Some dependencies failed to install: {installation_result['failed']}[/yellow]")
                    self.console.print("[yellow]Proceeding with available packages...[/yellow]")

            # Step 3: Generate multi-file implementation
            self.console.print("🎨 Generating multi-file implementation...")
            generated_files = self._generate_multi_file_implementation(plan, codebase_context)

            # Step 4: Create files in the codebase
            self.console.print("📁 Creating files in codebase...")
            created_files = self._create_files_in_codebase(generated_files, codebase_path)

            # Step 5: Execute main file with intelligent error handling
            main_file = self._find_main_file(created_files)
            if main_file:
                self.console.print("⚡ Executing with intelligent monitoring...")
                execution_result = self._execute_main_file(main_file, codebase_path)
            else:
                execution_result = {'success': True, 'final_code': 'Multi-file project created', 'iterations': 0, 'output': 'Files created successfully'}

            end_time = time.time()

            return {
                'success': execution_result['success'],
                'final_code': execution_result['final_code'],
                'execution_time': end_time - start_time,
                'plan': plan,
                'iterations': execution_result['iterations'],
                'output': execution_result['output']
            }

        except Exception as e:
            return self._create_failure_result(f"Engine error: {str(e)}")

    def _display_plan(self, plan: TaskPlan):
        """Display the detailed implementation plan."""
        from rich.table import Table

        self.console.print(Panel(
            f"📋 **IMPLEMENTATION PLAN**\n\n"
            f"🎯 **Task**: {plan.task_description}\n"
            f"🔢 **Estimated Complexity**: {plan.estimated_complexity}/10\n"
            f"📦 **Components**: {len(plan.required_components)}\n"
            f"📚 **Imports**: {len(plan.required_imports)}",
            title="🧠 Task Analysis",
            border_style="bold cyan"
        ))

        # Components table
        if plan.required_components:
            components_table = Table(title="📦 Required Components")
            components_table.add_column("Order", style="cyan", no_wrap=True)
            components_table.add_column("Component", style="green")
            components_table.add_column("Dependencies", style="yellow")

            for i, component in enumerate(plan.implementation_order, 1):
                deps = plan.dependencies.get(component, [])
                deps_str = ", ".join(deps) if deps else "None"
                components_table.add_row(str(i), component, deps_str)

            self.console.print(components_table)

        # Imports table
        if plan.required_imports:
            imports_table = Table(title="📚 Required Imports")
            imports_table.add_column("Import", style="blue")

            for imp in plan.required_imports:
                imports_table.add_row(imp)

            self.console.print(imports_table)

    def _display_validation_results(self, validation: Dict[str, Any]):
        """Display plan validation results."""
        if validation['feasible']:
            self.console.print("[bold green]✅ Plan is feasible and ready for implementation![/bold green]")
        else:
            self.console.print("[bold yellow]⚠️ Plan has some issues but proceeding...[/bold yellow]")

        if validation['issues']:
            self.console.print("\n[bold yellow]Issues identified:[/bold yellow]")
            for issue in validation['issues']:
                self.console.print(f"  • {issue}")

        if validation['recommendations']:
            self.console.print("\n[bold blue]Recommendations:[/bold blue]")
            for rec in validation['recommendations']:
                self.console.print(f"  • {rec}")

        self.console.print()  # Add spacing

    def _read_codebase(self, codebase_path: Path) -> Dict[str, Any]:
        """Read and understand the existing codebase."""
        codebase_context = {
            'files': {},
            'structure': {},
            'patterns': [],
            'existing_functions': [],
            'existing_classes': [],
            'imports_used': set(),
            'file_count': 0
        }

        try:
            # Read Python files in the codebase
            python_files = list(codebase_path.glob('**/*.py'))
            codebase_context['file_count'] = len(python_files)

            for file_path in python_files[:10]:  # Limit to first 10 files to avoid overwhelming
                try:
                    relative_path = file_path.relative_to(codebase_path)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Store file content
                    codebase_context['files'][str(relative_path)] = content

                    # Extract code patterns
                    self._extract_code_patterns(content, codebase_context)

                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
                    continue

            # Analyze project structure
            codebase_context['structure'] = self._analyze_project_structure(codebase_path)

            self.console.print(f"📖 Read {len(codebase_context['files'])} files, found {len(codebase_context['existing_functions'])} functions, {len(codebase_context['existing_classes'])} classes")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not fully read codebase: {e}[/yellow]")

        return codebase_context

    def _extract_code_patterns(self, content: str, context: Dict[str, Any]):
        """Extract patterns from code content."""
        import ast
        import re

        try:
            # Parse the AST to extract functions and classes
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    context['existing_functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    context['existing_classes'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        context['imports_used'].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        context['imports_used'].add(node.module)

        except SyntaxError:
            # If we can't parse, extract patterns with regex
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            classes = re.findall(r'class\s+(\w+)\s*[:\(]', content)
            imports = re.findall(r'(?:import|from)\s+(\w+)', content)

            context['existing_functions'].extend(functions)
            context['existing_classes'].extend(classes)
            context['imports_used'].update(imports)

    def _analyze_project_structure(self, codebase_path: Path) -> Dict[str, Any]:
        """Analyze the overall project structure."""
        structure = {
            'has_main': False,
            'has_utils': False,
            'has_tests': False,
            'directories': [],
            'config_files': []
        }

        try:
            # Check for common files
            if (codebase_path / 'main.py').exists():
                structure['has_main'] = True
            if (codebase_path / 'utils.py').exists():
                structure['has_utils'] = True

            # Check for test directories
            test_dirs = ['tests', 'test', '__tests__']
            for test_dir in test_dirs:
                if (codebase_path / test_dir).exists():
                    structure['has_tests'] = True
                    break

            # List directories
            for item in codebase_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    structure['directories'].append(item.name)

            # Check for config files
            config_patterns = ['*.json', '*.yaml', '*.yml', '*.toml', '*.ini']
            for pattern in config_patterns:
                structure['config_files'].extend([f.name for f in codebase_path.glob(pattern)])

        except Exception:
            pass

        return structure

    def _create_task_plan(self, prompt: str, codebase_context: Dict[str, Any] = None) -> TaskPlan:
        """Create a comprehensive plan for the task using LLM intelligence."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        # Create codebase context summary
        codebase_summary = ""
        if codebase_context and codebase_context['file_count'] > 0:
            existing_functions = list(set(codebase_context['existing_functions']))[:10]  # Limit to 10
            existing_classes = list(set(codebase_context['existing_classes']))[:10]
            existing_imports = list(codebase_context['imports_used'])[:10]

            codebase_summary = f"""
EXISTING CODEBASE CONTEXT:
- Files: {codebase_context['file_count']} Python files
- Existing Functions: {', '.join(existing_functions) if existing_functions else 'None'}
- Existing Classes: {', '.join(existing_classes) if existing_classes else 'None'}
- Current Imports: {', '.join(existing_imports) if existing_imports else 'None'}
- Project Structure: {codebase_context['structure']}

INTEGRATION GUIDELINES:
- Reuse existing functions/classes where appropriate
- Follow existing code patterns and naming conventions
- Avoid duplicating existing functionality
- Consider extending existing classes rather than creating new ones
- Use similar import patterns as existing code
"""

        # Use LLM to intelligently analyze the task and create a plan
        planning_prompt = f"""Analyze this programming task and create a detailed implementation plan: "{prompt}"
{codebase_summary}
TASK: Create a JSON plan with the following structure:
{{
    "required_components": ["list of functions/classes needed"],
    "required_imports": ["list of Python packages/libraries needed"],
    "dependencies": {{"component": ["list of dependencies"]}},
    "estimated_complexity": 1-10,
    "task_type": "description of task type",
    "file_structure": {{"filename.py": ["components to include in this file"]}}
}}

RULES:
- Use ANY Python libraries that would make the task easier and better
- Include both standard library (math, random, csv, json, os, sys, etc.) AND external libraries
- For data analysis: pandas, numpy, matplotlib, seaborn, scipy
- For web scraping: requests, beautifulsoup4, selenium
- For machine learning: scikit-learn, tensorflow, pytorch
- For web development: flask, django, fastapi
- For image processing: pillow, opencv-python
- Components should be actual function/class names that make sense
- Dependencies should show which components depend on others
- Complexity: 1-3=simple, 4-6=medium, 7-10=complex
- Include a main function to demonstrate functionality
- Choose the BEST libraries for the task, don't limit yourself
- Plan for multiple files when appropriate (separate concerns, utilities, main logic)
- Consider existing codebase patterns and avoid conflicts

Examples:
- Calculator: {{"required_components": ["Calculator", "add", "subtract", "multiply", "divide", "main"], "required_imports": ["math"], "dependencies": {{"main": ["Calculator", "add", "subtract", "multiply", "divide"]}}, "file_structure": {{"calculator.py": ["Calculator", "add", "subtract", "multiply", "divide"], "main.py": ["main"]}}}}
- Data analysis: {{"required_components": ["DataProcessor", "load_data", "analyze_data", "visualize_results", "main"], "required_imports": ["pandas", "numpy", "matplotlib", "seaborn"], "dependencies": {{"analyze_data": ["load_data"], "visualize_results": ["analyze_data"], "main": ["DataProcessor", "load_data", "analyze_data", "visualize_results"]}}, "file_structure": {{"data_processor.py": ["DataProcessor", "load_data", "analyze_data"], "visualization.py": ["visualize_results"], "main.py": ["main"]}}}}

Return ONLY the JSON plan:"""

        try:
            plan_response = llm_manager.generate_code(planning_prompt)
            plan_response = plan_response[5:]

            # Clean and parse the JSON response
            plan_response = plan_response.strip()
            if plan_response.startswith('```json'):
                plan_response = plan_response[7:]
            if plan_response.endswith('```'):
                plan_response = plan_response[:-3]

            import json
            plan_data = json.loads(plan_response)

            required_components = plan_data.get('required_components', [])
            required_imports = plan_data.get('required_imports', [])
            dependencies = plan_data.get('dependencies', {})
            estimated_complexity = plan_data.get('estimated_complexity', 3)
            file_structure = plan_data.get('file_structure', {})

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fallback to simple analysis if LLM planning fails
            self.console.print(f"[yellow]LLM planning failed ({e}), using fallback analysis[/yellow]")
            required_components, required_imports, dependencies, estimated_complexity = self._fallback_task_analysis(prompt)
            file_structure = {}

        return TaskPlan(
            task_description=prompt,
            required_components=required_components,
            required_imports=required_imports,
            implementation_order=self._determine_implementation_order(required_components, dependencies),
            dependencies=dependencies,
            estimated_complexity=estimated_complexity,
            file_structure=file_structure
        )

    def _fallback_task_analysis(self, prompt: str) -> Tuple[List[str], List[str], Dict[str, List[str]], int]:
        """Fallback analysis when LLM planning fails."""
        prompt_lower = prompt.lower()

        # Simple keyword-based analysis
        if 'calculator' in prompt_lower:
            return (['Calculator', 'add', 'subtract', 'multiply', 'divide', 'main'],
                   [],
                   {'main': ['Calculator', 'add', 'subtract', 'multiply', 'divide']},
                   3)
        elif 'data' in prompt_lower and 'csv' in prompt_lower:
            return (['DataProcessor', 'load_data', 'process_data', 'main'],
                   ['csv'],
                   {'process_data': ['load_data'], 'main': ['DataProcessor', 'load_data', 'process_data']},
                   5)
        else:
            # Generic fallback
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', prompt)
            meaningful_words = [word for word in words if len(word) > 3 and word.lower() not in
                              ['create', 'simple', 'with', 'that', 'this', 'make', 'build', 'using']]
            components = meaningful_words[:2] + ['main'] if meaningful_words else ['Task', 'main']
            return (components, [], {'main': components[:-1]}, 3)

    def _determine_implementation_order(self, components: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Determine the correct order to implement components based on dependencies."""
        ordered = []
        remaining = set(components)

        while remaining:
            # Find components with no unmet dependencies
            ready = []
            for comp in remaining:
                deps = dependencies.get(comp, [])
                if all(dep in ordered or dep not in components for dep in deps):
                    ready.append(comp)

            if not ready:
                # Break circular dependencies by picking the first one
                ready = [list(remaining)[0]]

            ordered.extend(ready)
            remaining -= set(ready)

        return ordered

    def _estimate_complexity(self, prompt: str) -> int:
        """Estimate task complexity dynamically using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        complexity_prompt = f"""Estimate the programming complexity of this task on a scale of 1-10: "{prompt}"

COMPLEXITY SCALE:
1-3: Simple (basic functions, simple calculations, hello world)
4-6: Medium (data structures, file I/O, basic algorithms)
7-8: Complex (multi-component systems, advanced algorithms)
9-10: Very Complex (neural networks, complex simulations)

Consider:
- Number of components needed
- Algorithm complexity
- Data processing requirements
- Integration complexity
- Standard library limitations

Return ONLY a single number (1-10):"""

        try:
            complexity_response = llm_manager.generate_code(complexity_prompt)
            complexity = int(complexity_response.strip())
            return max(1, min(complexity, 10))  # Ensure it's between 1-10
        except (ValueError, Exception):
            # Fallback to simple heuristic
            return self._fallback_complexity_estimation(prompt)

    def _fallback_complexity_estimation(self, prompt: str) -> int:
        """Fallback complexity estimation when LLM fails."""
        prompt_lower = prompt.lower()
        base_complexity = 3

        # Simple heuristic based on keywords
        if any(term in prompt_lower for term in ['neural', 'machine learning', 'ai']):
            base_complexity = 8
        elif any(term in prompt_lower for term in ['web scraping', 'api', 'database']):
            base_complexity = 6
        elif any(term in prompt_lower for term in ['game', 'gui', 'visualization']):
            base_complexity = 6
        elif any(term in prompt_lower for term in ['algorithm', 'sort', 'search']):
            base_complexity = 5
        elif any(term in prompt_lower for term in ['data', 'csv', 'json']):
            base_complexity = 4

        # Adjust for prompt length and detail
        if len(prompt) > 100:
            base_complexity += 1
        if len(prompt.split()) > 15:
            base_complexity += 1

        return min(base_complexity, 10)

    def _validate_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Validate if the plan is feasible using intelligent analysis."""
        issues = []

        # Dynamic complexity validation
        if plan.estimated_complexity > 8:
            issues.append(f"High complexity ({plan.estimated_complexity}/10) - may need simplification")

        # Check for circular dependencies
        for comp, deps in plan.dependencies.items():
            if comp in deps:
                issues.append(f"Circular dependency detected: {comp} depends on itself")

        # Dynamic import validation using LLM
        import_issues = self._validate_imports_dynamically(plan.required_imports)
        issues.extend(import_issues)

        return {
            'feasible': len(issues) == 0 or plan.estimated_complexity <= 6,
            'issues': issues,
            'recommendations': self._get_plan_recommendations(plan)
        }

    def _validate_imports_dynamically(self, imports: List[str]) -> List[str]:
        """Dynamically validate imports using LLM knowledge."""
        from core.llm_integration import LLMManager

        if not imports:
            return []

        llm_manager = LLMManager()

        validation_prompt = f"""Validate these Python imports for safety and availability: {imports}

RULES:
- Allow both standard library AND external libraries
- Standard library: math, random, csv, json, os, sys, urllib, re, collections, etc.
- Popular external libraries: pandas, numpy, requests, beautifulsoup4, matplotlib, etc.
- Block dangerous/malicious packages: os-sys, system-tools, malware, virus, hack, exploit
- Block packages that might cause system issues

For each import, respond with:
- "OK: import_name" if it's safe and reasonable
- "ISSUE: import_name - reason" if it's dangerous or problematic

Imports to validate: {', '.join(imports)}

Response format (one per line):"""

        try:
            validation_response = llm_manager.generate_code(validation_prompt)
            issues = []

            for line in validation_response.strip().split('\n'):
                if line.strip().startswith('ISSUE:'):
                    issues.append(line.strip()[6:])  # Remove "ISSUE: " prefix

            return issues
        except Exception:
            # Fallback to simple validation
            return self._fallback_import_validation(imports)

    def _fallback_import_validation(self, imports: List[str]) -> List[str]:
        """Fallback import validation when LLM fails - now allows external libraries."""
        # Only block truly dangerous packages
        dangerous_patterns = [
            'os-sys', 'system-tools', 'malware', 'virus', 'hack', 'exploit',
            'subprocess-tools', 'system-admin', 'root-access'
        ]

        issues = []
        for imp in imports:
            for pattern in dangerous_patterns:
                if pattern in imp.lower():
                    issues.append(f"Dangerous import detected: {imp} - blocked for security")
                    break

        return issues

    def _get_python_builtins(self) -> set:
        """Get Python built-in functions dynamically."""
        import builtins
        return set(dir(builtins))

    def _get_plan_recommendations(self, plan: TaskPlan) -> List[str]:
        """Get recommendations for improving the plan."""
        recommendations = []

        if plan.estimated_complexity > 7:
            recommendations.append("Consider breaking into smaller subtasks")

        if len(plan.required_components) > 10:
            recommendations.append("Large number of components - ensure clear interfaces")

        if not plan.dependencies:
            recommendations.append("No dependencies detected - verify component isolation")

        return recommendations

    def _install_dependencies(self, required_imports: List[str]) -> Dict[str, Any]:
        """Install required Python packages automatically."""
        import subprocess
        import sys

        # Filter out standard library modules
        standard_library = {
            'math', 'random', 'csv', 'json', 'os', 'sys', 'time', 'datetime',
            'collections', 're', 'itertools', 'functools', 'operator', 'pathlib',
            'urllib', 'http', 'socket', 'threading', 'multiprocessing', 'asyncio',
            'sqlite3', 'pickle', 'base64', 'hashlib', 'hmac', 'secrets', 'uuid',
            'logging', 'argparse', 'configparser', 'tempfile', 'shutil', 'glob',
            'fnmatch', 'linecache', 'textwrap', 'string', 'io', 'struct'
        }

        # Safety check - block potentially dangerous packages
        dangerous_packages = {
            'os-sys', 'system-tools', 'malware', 'virus', 'hack', 'exploit'
        }

        # Package name mapping for common cases
        package_mapping = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'bs4': 'beautifulsoup4'
        }

        external_packages = set()  # Use set to avoid duplicates
        for package in required_imports:
            # Clean package name (remove .submodule parts)
            base_package = package.split('.')[0]

            if base_package not in standard_library and base_package not in dangerous_packages:
                # Map to correct package name if needed
                actual_package = package_mapping.get(base_package, base_package)
                external_packages.add(actual_package)

        external_packages = list(external_packages)  # Convert back to list

        if not external_packages:
            return {'success': True, 'installed': [], 'failed': []}

        self.console.print(f"📦 Installing packages: {', '.join(external_packages)}")

        installed = []
        failed = []

        for package in external_packages:
            try:
                self.console.print(f"  📥 Installing {package}...")

                # Use pip to install the package
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout per package
                )

                if result.returncode == 0:
                    installed.append(package)
                    self.console.print(f"  ✅ {package} installed successfully")
                else:
                    failed.append(package)
                    self.console.print(f"  ❌ Failed to install {package}: {result.stderr[:100]}...")

            except subprocess.TimeoutExpired:
                failed.append(package)
                self.console.print(f"  ⏰ {package} installation timed out")
            except Exception as e:
                failed.append(package)
                self.console.print(f"  ❌ Error installing {package}: {str(e)[:100]}...")

        success = len(failed) == 0
        return {
            'success': success,
            'installed': installed,
            'failed': failed
        }

    def _generate_complete_implementation(self, plan: TaskPlan) -> str:
        """Generate complete, self-contained implementation using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        # Create a comprehensive prompt for the LLM with FULL plan details
        dependencies_info = ""
        if plan.dependencies:
            dependencies_info = "\nDEPENDENCIES:\n"
            for component, deps in plan.dependencies.items():
                if deps:
                    dependencies_info += f"- {component} depends on: {', '.join(deps)}\n"
                else:
                    dependencies_info += f"- {component} has no dependencies\n"

        imports_info = ""
        if plan.required_imports:
            imports_info = f"\nREQUIRED IMPORTS (standard library only):\n{', '.join(plan.required_imports)}\n"

        prompt = f"""⚠️ CRITICAL: RETURN ONLY PYTHON CODE - NO TEXT, NO EXPLANATIONS, NO MARKDOWN ⚠️

YOUR RESPONSE WILL BE EXECUTED DIRECTLY AS PYTHON CODE!

Task: {plan.task_description}

TASK ANALYSIS:
- Estimated Complexity: {plan.estimated_complexity}/10
- Total Components: {len(plan.required_components)}
{imports_info}{dependencies_info}

IMPLEMENTATION PLAN:
Required components to implement: {', '.join(plan.required_components)}
Implementation order: {', '.join(plan.implementation_order)}

CRITICAL REQUIREMENTS:
- Write COMPLETE, SELF-CONTAINED Python code that runs immediately
- Use the specified libraries from the plan - they will be automatically installed
- Import and use the libraries listed in the required imports
- DEFINE ALL functions and classes that are used
- Include ALL data generation, processing, and output in one file
- The code must execute successfully from start to finish
- If you need data, GENERATE it in the code or use sample data
- Include proper error handling and docstrings
- Add a main section that demonstrates the functionality
- NO interactive input() calls - use predefined test data instead
- NO infinite loops or user interaction - make it fully automated
- Include print statements to show the results
- Follow the dependency order: implement components with no dependencies first
- Ensure each component properly uses its dependencies as specified
- Use the BEST practices for each library (pandas for data, requests for web, etc.)

⚠️ IMPORTANT: START YOUR RESPONSE WITH PYTHON CODE IMMEDIATELY - NO EXPLANATORY TEXT!
⚠️ DO NOT WRITE "Here's the code:" or "```python" or ANY TEXT BEFORE THE CODE!
⚠️ YOUR FIRST CHARACTER MUST BE PYTHON CODE (import, def, class, or #)!

RETURN ONLY EXECUTABLE PYTHON CODE:"""

        # Generate the complete code using LLM
        generated_code = llm_manager.generate_code(prompt)

        # Clean up the response to ensure it's only code
        cleaned_code = self._extract_pure_code(generated_code)

        return cleaned_code

    def _extract_pure_code(self, response: str) -> str:
        """Extract pure Python code from LLM response, removing any explanatory text."""
        lines = response.strip().split('\n')
        code_lines = []
        code_started = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines at the beginning
            if not code_started and not stripped:
                continue

            # Skip common explanatory text patterns
            if not code_started and (
                stripped.lower().startswith(('here', 'this', 'the following', 'below', 'i will', 'let me', 'sure', 'certainly')) or
                stripped.lower().startswith(('```python', '```', 'python', 'code:')) or
                'here is' in stripped.lower() or
                'here\'s' in stripped.lower() or
                'the code' in stripped.lower() or
                'solution' in stripped.lower()
            ):
                continue

            # Detect start of Python code
            if not code_started and (
                stripped.startswith(('#', 'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')) or
                stripped.startswith(('print(', 'return ', '@')) or
                '=' in stripped or
                stripped.endswith(':')
            ):
                code_started = True

            # Once code starts, include everything except markdown endings
            if code_started:
                if stripped == '```' or stripped.startswith('```'):
                    break  # End of code block
                code_lines.append(line)

        # Join the code lines
        clean_code = '\n'.join(code_lines).strip()

        # Final cleanup - remove any remaining markdown
        if clean_code.startswith('```python'):
            clean_code = clean_code[9:].strip()
        elif clean_code.startswith('```'):
            clean_code = clean_code[3:].strip()

        if clean_code.endswith('```'):
            clean_code = clean_code[:-3].strip()

        return clean_code

    def _generate_multi_file_implementation(self, plan: TaskPlan, codebase_context: Dict[str, Any]) -> Dict[str, str]:
        """Generate multiple files based on the plan's file structure."""
        generated_files = {}

        # If no file structure is specified, create a default one
        if not plan.file_structure:
            plan.file_structure = self._create_default_file_structure(plan)

        # Generate each file
        for filename, components in plan.file_structure.items():
            self.console.print(f"  📄 Generating {filename}...")

            # Create codebase context for this file
            file_context = self._create_file_context(filename, components, plan, codebase_context)

            # Generate the file content
            file_content = self._generate_single_file(filename, components, plan, file_context)

            # Validate the generated file content
            validation = self._validate_code_structure(file_content)
            if not validation.is_valid:
                self.console.print(f"  ⚠️ {filename} has validation issues, fixing...")
                file_content = self._fix_structural_issues(file_content, validation)

            generated_files[filename] = file_content

        return generated_files

    def _create_default_file_structure(self, plan: TaskPlan) -> Dict[str, List[str]]:
        """Create a default file structure when none is provided."""
        file_structure = {}

        # Separate main from other components
        main_components = [comp for comp in plan.required_components if comp.lower() == 'main']
        other_components = [comp for comp in plan.required_components if comp.lower() != 'main']

        if len(other_components) <= 3:
            # Small project - put everything in main.py
            file_structure['main.py'] = plan.required_components
        else:
            # Larger project - separate into logical files
            if any('data' in comp.lower() or 'process' in comp.lower() for comp in other_components):
                data_components = [comp for comp in other_components if 'data' in comp.lower() or 'process' in comp.lower()]
                file_structure['data_processor.py'] = data_components
                other_components = [comp for comp in other_components if comp not in data_components]

            if any('visual' in comp.lower() or 'plot' in comp.lower() or 'chart' in comp.lower() for comp in other_components):
                viz_components = [comp for comp in other_components if 'visual' in comp.lower() or 'plot' in comp.lower() or 'chart' in comp.lower()]
                file_structure['visualization.py'] = viz_components
                other_components = [comp for comp in other_components if comp not in viz_components]

            if other_components:
                file_structure['utils.py'] = other_components

            if main_components:
                file_structure['main.py'] = main_components

        return file_structure

    def _create_file_context(self, filename: str, components: List[str], plan: TaskPlan, codebase_context: Dict[str, Any]) -> str:
        """Create context information for generating a specific file."""
        context_parts = []

        # Add existing codebase context
        if codebase_context and codebase_context['file_count'] > 0:
            context_parts.append(f"EXISTING CODEBASE: {codebase_context['file_count']} files")
            if codebase_context['existing_functions']:
                context_parts.append(f"Existing functions: {', '.join(list(set(codebase_context['existing_functions']))[:5])}")

        # Add file-specific context
        context_parts.append(f"FILE: {filename}")
        context_parts.append(f"COMPONENTS TO IMPLEMENT: {', '.join(components)}")

        # Add dependency information for these components
        deps_info = []
        for comp in components:
            if comp in plan.dependencies:
                deps = plan.dependencies[comp]
                if deps:
                    deps_info.append(f"{comp} depends on: {', '.join(deps)}")

        if deps_info:
            context_parts.append("DEPENDENCIES:")
            context_parts.extend(deps_info)

        return "\n".join(context_parts)

    def _generate_single_file(self, filename: str, components: List[str], plan: TaskPlan, file_context: str) -> str:
        """Generate content for a single file."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        # Determine if this is the main file
        is_main_file = filename.lower() == 'main.py' or 'main' in components

        # Create imports section
        imports_section = ""
        if plan.required_imports:
            imports_section = f"REQUIRED IMPORTS: {', '.join(plan.required_imports)}"

        # Create inter-file imports
        inter_file_imports = ""
        if plan.file_structure and len(plan.file_structure) > 1:
            other_files = [f for f in plan.file_structure.keys() if f != filename]
            if other_files and not is_main_file:
                inter_file_imports = f"INTER-FILE IMPORTS: May need to import from: {', '.join([f.replace('.py', '') for f in other_files])}"
            elif is_main_file:
                inter_file_imports = f"INTER-FILE IMPORTS: Import from: {', '.join([f.replace('.py', '') for f in other_files])}"

        prompt = f"""⚠️ CRITICAL: RETURN ONLY PYTHON CODE - NO TEXT, NO EXPLANATIONS, NO MARKDOWN ⚠️

YOUR RESPONSE WILL BE EXECUTED DIRECTLY AS PYTHON CODE!

Generate Python code for file: {filename}

TASK: {plan.task_description}
{file_context}

{imports_section}
{inter_file_imports}

COMPONENTS TO IMPLEMENT IN THIS FILE:
{', '.join(components)}

CRITICAL REQUIREMENTS:
- Write COMPLETE, WORKING Python code for this specific file
- Use the specified libraries from the plan - they will be automatically installed
- DEFINE ALL functions and classes listed in components
- Include proper error handling and docstrings
- If this is main.py, include if __name__ == "__main__": block
- If this is not main.py, focus on the specific components only
- Use proper imports (both external libraries and inter-file imports)
- Make the code modular and well-organized
- Include print statements for demonstration if this is main.py

⚠️ IMPORTANT: START YOUR RESPONSE WITH PYTHON CODE IMMEDIATELY - NO EXPLANATORY TEXT!
⚠️ DO NOT WRITE "Here's the code:" or "```python" or ANY TEXT BEFORE THE CODE!
⚠️ YOUR FIRST CHARACTER MUST BE PYTHON CODE (import, def, class, or #)!

RETURN ONLY EXECUTABLE PYTHON CODE:"""

        # Generate the file content
        generated_code = llm_manager.generate_code(prompt)

        # Clean up the response
        cleaned_code = self._extract_pure_code(generated_code)

        return cleaned_code

    def _create_files_in_codebase(self, generated_files: Dict[str, str], codebase_path: Path) -> List[Path]:
        """Create the generated files in the codebase directory."""
        created_files = []

        for filename, content in generated_files.items():
            file_path = codebase_path / filename

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                created_files.append(file_path)
                self.console.print(f"  ✅ Created {filename}")

            except Exception as e:
                self.console.print(f"  ❌ Failed to create {filename}: {e}")

        return created_files

    def _find_main_file(self, created_files: List[Path]) -> Optional[Path]:
        """Find the main file to execute."""
        for file_path in created_files:
            if file_path.name.lower() == 'main.py':
                return file_path

        # If no main.py, return the first file
        return created_files[0] if created_files else None

    def _execute_main_file(self, main_file: Path, codebase_path: Path) -> Dict[str, Any]:
        """Execute the main file with intelligent error handling."""
        iterations = 0
        max_iterations = 3

        while iterations < max_iterations:
            iterations += 1

            try:
                # Execute the main file using relative path from codebase directory
                main_file_name = main_file.name  # Just the filename, not full path
                result = subprocess.run(
                    ['python', main_file_name],
                    cwd=str(codebase_path.absolute()),  # Ensure absolute path for cwd
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    # Success!
                    return {
                        'success': True,
                        'final_code': f'Multi-file project with main: {main_file.name}',
                        'iterations': iterations,
                        'output': result.stdout
                    }
                else:
                    # Error occurred
                    error_output = result.stderr
                    stdout_output = result.stdout
                    self.console.print(f"[yellow]Iteration {iterations} failed:[/yellow]")
                    self.console.print(f"[red]STDERR: {error_output[:300]}[/red]")
                    if stdout_output:
                        self.console.print(f"[blue]STDOUT: {stdout_output[:200]}[/blue]")
                    self.console.print(f"[yellow]Working directory: {codebase_path.absolute()}[/yellow]")
                    self.console.print(f"[yellow]Executing: python {main_file_name}[/yellow]")

                    if iterations < max_iterations:
                        # Try to fix the error by regenerating the main file
                        self._fix_main_file_error(main_file, error_output)

            except subprocess.TimeoutExpired:
                self.console.print(f"[yellow]Iteration {iterations} timed out[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[yellow]Execution error: {e}[/yellow]")
                break

        return {
            'success': False,
            'final_code': f'Multi-file project (execution failed): {main_file.name}',
            'iterations': iterations,
            'output': 'Failed after maximum iterations'
        }

    def _fix_main_file_error(self, main_file: Path, error_output: str):
        """Attempt to fix errors in the main file."""
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                current_content = f.read()

            # Use the existing error fixing logic
            fixed_content = self._intelligent_error_fix(current_content, error_output)

            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            self.console.print(f"  🔧 Applied fix to {main_file.name}")

        except Exception as e:
            self.console.print(f"  ❌ Could not fix {main_file.name}: {e}")

    def _generate_component(self, component_name: str, plan: TaskPlan) -> str:
        """Generate code for a specific component using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        prompt = f"""Generate COMPLETE Python code for the component: {component_name}

Task context: {plan.task_description}

CRITICAL REQUIREMENTS:
- Write COMPLETE, WORKING code for this specific component
- Use ONLY Python standard library (math, random, csv, json, etc.)
- NO external libraries like tensorflow, sklearn, pandas, etc.
- DEFINE ALL functions and classes completely
- Include proper error handling and docstrings
- Make it self-contained and functional

Generate ONLY the Python code for this component:"""

        return llm_manager.generate_code(prompt)

    def _generate_main_function(self, plan: TaskPlan) -> str:
        """Generate the main execution function using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        prompt = f"""Generate COMPLETE Python main function for: {plan.task_description}

CRITICAL REQUIREMENTS:
- Write a complete main() function and if __name__ == "__main__": block
- Use ONLY the components that will be defined: {', '.join(plan.required_components)}
- Use ONLY Python standard library
- Include proper demonstration of all functionality
- Add print statements to show progress and results
- Make it self-contained and executable

Generate ONLY the main function code:"""

        return llm_manager.generate_code(prompt)

    def _generate_web_scraper_component(self, component_name: str) -> str:
        """Generate web scraper components using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        prompt = f"""Generate COMPLETE Python code for web scraper component: {component_name}

CRITICAL REQUIREMENTS:
- Use ONLY Python standard library (urllib, re, json, etc.)
- NO external libraries like requests, beautifulsoup, etc.
- Write complete, working code for web scraping
- Include proper error handling
- Make it self-contained and functional

Generate ONLY the Python code for this web scraper component:"""

        return llm_manager.generate_code(prompt)

    def _generate_data_analysis_component(self, component_name: str) -> str:
        """Generate data analysis components using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        prompt = f"""Generate COMPLETE Python code for data analysis component: {component_name}

CRITICAL REQUIREMENTS:
- Use ONLY Python standard library (csv, json, math, statistics, etc.)
- NO external libraries like pandas, numpy, matplotlib, etc.
- Write complete, working code for data analysis
- Include proper error handling and data processing
- Make it self-contained and functional

Generate ONLY the Python code for this data analysis component:"""

        return llm_manager.generate_code(prompt)

    def _generate_game_component(self, component_name: str) -> str:
        """Generate game components using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        prompt = f"""Generate COMPLETE Python code for game component: {component_name}

CRITICAL REQUIREMENTS:
- Write complete, working code for a simple console game
- Include game logic, rendering, and input handling
- Make it self-contained and functional
- Include proper error handling

Generate ONLY the Python code for this game component:"""

        return llm_manager.generate_code(prompt)

    def _generate_generic_component(self, component_name: str) -> str:
        """Generate a generic component using LLM."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        prompt = f"""Generate COMPLETE Python code for component: {component_name}

CRITICAL REQUIREMENTS:
- Use ONLY Python standard library
- Write complete, working code for this component
- Include proper error handling and docstrings
- Make it self-contained and functional
- If it's a class (starts with capital), include __init__ and main methods
- If it's a function, make it complete and working

Generate ONLY the Python code for this component:"""

        return llm_manager.generate_code(prompt)

    def _validate_code_structure(self, code: str) -> ValidationResult:
        """Validate code structure before execution."""
        missing_components = []
        undefined_references = []
        import_issues = []
        syntax_errors = []
        suggestions = []

        try:
            # Parse the code to check syntax
            tree = ast.parse(code)

            # Find all defined functions and classes
            defined_functions = set()
            defined_classes = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)

            # Find all function calls and class instantiations
            called_functions = set()
            used_classes = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_functions.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # Method calls - check if the object exists
                        if isinstance(node.func.value, ast.Name):
                            # This is a method call, we'll check the class later
                            pass
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    # Variable/class usage
                    if node.id[0].isupper():  # Likely a class
                        used_classes.add(node.id)

            # Check for undefined functions using dynamic built-ins detection
            python_builtins = self._get_python_builtins()

            # Add common math functions that need to be imported
            math_functions = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'pow', 'ceil', 'floor'}

            for func in called_functions:
                if func not in defined_functions and func not in python_builtins:
                    if func in math_functions:
                        undefined_references.append(f"Function '{func}' requires 'import math' or 'from math import {func}'")
                    else:
                        undefined_references.append(f"Function '{func}' is called but not defined")

            # Check for undefined classes
            for cls in used_classes:
                if cls not in defined_classes:
                    undefined_references.append(f"Class '{cls}' is used but not defined")

        except SyntaxError as e:
            syntax_errors.append(f"Syntax error: {e}")

        # Check imports dynamically
        import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith('import')]
        if import_lines:
            import_validation = self._validate_imports_dynamically([line.split()[-1] for line in import_lines])
            import_issues.extend(import_validation)

        # Generate suggestions
        if undefined_references:
            suggestions.append("Consider implementing missing functions/classes or using standard library alternatives")
        if import_issues:
            suggestions.append("Replace complex imports with standard library equivalents")

        is_valid = not (missing_components or undefined_references or import_issues or syntax_errors)

        return ValidationResult(
            is_valid=is_valid,
            missing_components=missing_components,
            undefined_references=undefined_references,
            import_issues=import_issues,
            syntax_errors=syntax_errors,
            suggestions=suggestions
        )

    def _fix_structural_issues(self, code: str, validation: ValidationResult) -> str:
        """Fix structural issues in the code using intelligent analysis."""
        fixed_code = code

        # Fix import issues dynamically
        if validation.import_issues:
            fixed_code = self._fix_import_issues_dynamically(fixed_code, validation.import_issues)

        # Add missing functions dynamically
        if validation.undefined_references:
            fixed_code = self._add_missing_functions_dynamically(fixed_code, validation.undefined_references)

        return fixed_code

    def _fix_import_issues_dynamically(self, code: str, import_issues: List[str]) -> str:
        """Fix import issues using LLM intelligence."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        fix_prompt = f"""Fix these import issues in Python code by replacing with standard library alternatives:

Issues: {import_issues}

Original code:
{code}

RULES:
- Replace problematic imports with standard library equivalents
- Comment out imports that can't be replaced
- Ensure the code still works with standard library only

Return the fixed code:"""

        try:
            fixed_code = llm_manager.generate_code(fix_prompt)
            return fixed_code
        except Exception:
            # Fallback to simple commenting out
            for issue in import_issues:
                for line in code.split('\n'):
                    if 'import' in line and any(word in issue.lower() for word in line.lower().split()):
                        code = code.replace(line, f"# {line}  # Commented out: {issue}")
            return code

    def _add_missing_functions_dynamically(self, code: str, undefined_refs: List[str]) -> str:
        """Add missing functions using LLM intelligence."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        function_prompt = f"""Add missing function implementations to this Python code:

Missing functions: {undefined_refs}

Original code:
{code}

RULES:
- Add simple, working implementations for missing functions
- Use only standard library
- Make functions compatible with how they're being called
- Add functions at the top of the code

Return the complete fixed code:"""

        try:
            fixed_code = llm_manager.generate_code(function_prompt)
            return fixed_code
        except Exception:
            # Fallback to simple stub functions
            for ref in undefined_refs:
                if "Function '" in ref and "' is called" in ref:
                    func_name = ref.split("'")[1]
                    stub_func = f'''
def {func_name}(*args, **kwargs):
    """Auto-generated stub function."""
    print(f"Called {func_name} with args: {{args}}, kwargs: {{kwargs}}")
    return None
'''
                    code = stub_func + "\n" + code
            return code

    def _execute_with_intelligent_handling(self, code: str, codebase_path: Path) -> Dict[str, Any]:
        """Execute code with intelligent error handling."""
        iterations = 0
        max_iterations = 3
        current_code = code

        while iterations < max_iterations:
            iterations += 1

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(current_code)
                temp_file = f.name

            try:
                # Execute the code
                result = subprocess.run(
                    ['python', temp_file],
                    cwd=str(codebase_path),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    # Success!
                    return {
                        'success': True,
                        'final_code': current_code,
                        'iterations': iterations,
                        'output': result.stdout
                    }
                else:
                    # Error occurred
                    error_output = result.stderr
                    self.console.print(f"[yellow]Iteration {iterations} failed: {error_output[:200]}...[/yellow]")

                    if iterations < max_iterations:
                        # Try to fix the error
                        current_code = self._intelligent_error_fix(current_code, error_output)

            except subprocess.TimeoutExpired:
                self.console.print(f"[yellow]Iteration {iterations} timed out[/yellow]")
                if iterations < max_iterations:
                    # Add timeout handling
                    current_code = self._add_timeout_handling(current_code)

            finally:
                # Clean up temp file
                try:
                    Path(temp_file).unlink()
                except:
                    pass

        return {
            'success': False,
            'final_code': current_code,
            'iterations': iterations,
            'output': 'Failed after maximum iterations'
        }

    def _intelligent_error_fix(self, code: str, error_output: str) -> str:
        """Apply intelligent fixes based on error output using plan context."""
        from core.llm_integration import LLMManager

        llm_manager = LLMManager()

        # Use LLM to intelligently fix the error with plan context
        plan_context = ""
        if self.current_plan:
            plan_context = f"""
ORIGINAL PLAN CONTEXT:
- Task: {self.current_plan.task_description}
- Components: {', '.join(self.current_plan.required_components)}
- Dependencies: {self.current_plan.dependencies}
- Complexity: {self.current_plan.estimated_complexity}/10
"""

        fix_prompt = f"""⚠️ CRITICAL: RETURN ONLY PYTHON CODE - NO TEXT, NO EXPLANATIONS ⚠️

YOUR RESPONSE WILL BE EXECUTED DIRECTLY AS PYTHON CODE!

Fix this Python code error using the plan context:

{plan_context}
ERROR OUTPUT:
{error_output}

CURRENT CODE:
{code}

RULES:
- Fix the specific error mentioned
- Use the specified libraries from the plan (they are installed)
- Maintain the original plan's component structure
- Ensure all planned components are properly implemented
- Make the code executable and complete

⚠️ IMPORTANT: START YOUR RESPONSE WITH PYTHON CODE IMMEDIATELY!
⚠️ DO NOT WRITE "Here's the fixed code:" or ANY TEXT BEFORE THE CODE!
⚠️ YOUR FIRST CHARACTER MUST BE PYTHON CODE!

RETURN ONLY THE COMPLETE FIXED PYTHON CODE:"""

        try:
            fixed_code = llm_manager.generate_code(fix_prompt)
            # Clean up the response to ensure it's only code
            cleaned_fixed_code = self._extract_pure_code(fixed_code)
            return cleaned_fixed_code
        except Exception:
            # Fallback to simple pattern-based fixes
            return self._fallback_error_fix(code, error_output)

    def _fallback_error_fix(self, code: str, error_output: str) -> str:
        """Fallback error fixing when LLM fails."""
        if "NameError" in error_output and "is not defined" in error_output:
            # Extract the undefined name
            import re
            match = re.search(r"name '(\w+)' is not defined", error_output)
            if match:
                undefined_name = match.group(1)

                # Add a simple implementation
                if undefined_name == 'exp':
                    # Add math import for exp function
                    if 'import math' not in code and 'from math import' not in code:
                        fix = "import math\n\n"
                        # Replace exp with math.exp
                        code = code.replace('exp(', 'math.exp(')
                        return fix + code
                elif undefined_name == 'show_progress':
                    fix = '''
def show_progress(current, total, message="Progress"):
    percent = (current / total) * 100
    print(f"{message}: {percent:.1f}%")

'''
                    return fix + code
                elif undefined_name.endswith('_data') or undefined_name.startswith('load_'):
                    # Use string concatenation to avoid syntax issues
                    fix = f"\ndef {undefined_name}():\n    \"\"\"Auto-generated function.\"\"\"\n    return []\n\n"
                    return fix + code

        elif "ModuleNotFoundError" in error_output:
            # Replace problematic imports
            if 'tensorflow' in error_output:
                code = code.replace('import tensorflow as tf', '# tensorflow not available')
            elif 'torch' in error_output:
                code = code.replace('import torch', '# torch not available')

        return code

    def _add_timeout_handling(self, code: str) -> str:
        """Add timeout handling to prevent infinite loops."""
        # Add a simple timeout mechanism that works on Windows
        timeout_code = '''
import sys
import time

# Simple timeout mechanism for Windows compatibility
start_time = time.time()
MAX_EXECUTION_TIME = 25  # 25 seconds

def check_timeout():
    if time.time() - start_time > MAX_EXECUTION_TIME:
        print("Execution timed out!")
        sys.exit(1)

'''
        return timeout_code + code

    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create a failure result."""
        return {
            'success': False,
            'final_code': '',
            'execution_time': 0,
            'plan': None,
            'iterations': 0,
            'output': f'Error: {error_message}'
        }