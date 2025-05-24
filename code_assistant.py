"""Intelligent code assistant that combines analysis, generation, and execution."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from analyzer import CodebaseAnalyzer
from storage import VectorDatabase
from llm_integration import LLMManager, CodeChange
from code_executor import SafeCodeExecutor, CodeTester
from crypto import PathEncryption


class IntelligentCodeAssistant:
    """AI-powered code assistant that can understand and modify codebases."""

    def __init__(self):
        self.analyzer = CodebaseAnalyzer()
        self.vector_db = VectorDatabase()
        self.llm_manager = LLMManager()
        self.executor = SafeCodeExecutor()
        self.tester = CodeTester()
        self.encryption = PathEncryption()
        self.console = Console()

    def understand_request(self, request: str, codebase_path: Path = None) -> Dict[str, Any]:
        """Understand what the user wants to do."""
        # Get relevant context from codebase
        context = ""
        if codebase_path:
            context = self._get_codebase_context(request, codebase_path)

        # Analyze the request
        analysis_prompt = f"""Analyze this coding request and determine:
1. What type of task is this? (create_new, modify_existing, debug, explain, test)
2. What files might be involved?
3. What programming concepts are needed?
4. Is this a safe request to fulfill?

Request: {request}

Context: {context}

Return a JSON response with: task_type, complexity, safety_level, estimated_files, concepts_needed"""

        analysis = self.llm_manager.generate_code(analysis_prompt)

        return {
            'request': request,
            'analysis': analysis,
            'context': context,
            'codebase_path': str(codebase_path) if codebase_path else None
        }

    def _get_codebase_context(self, request: str, codebase_path: Path, max_chunks: int = 5) -> str:
        """Get relevant context from the codebase for the request."""
        try:
            # Search for relevant code chunks
            relevant_chunks = self.vector_db.search_by_text(request, n_results=max_chunks)

            context_parts = []
            for chunk in relevant_chunks:
                metadata = chunk['metadata']
                chunk_info = f"File: {metadata.get('name', 'unknown')}\n"
                chunk_info += f"Type: {metadata.get('type', 'unknown')}\n"
                chunk_info += f"Description: {chunk.get('document', '')}\n"
                context_parts.append(chunk_info)

            return "\n---\n".join(context_parts)
        except Exception as e:
            return f"Error getting context: {e}"

    def generate_code(self, request: str, codebase_path: Path = None) -> Dict[str, Any]:
        """Generate code based on the request."""
        # Understand the request first
        understanding = self.understand_request(request, codebase_path)

        # Generate the code
        generated_code = self.llm_manager.generate_code(request, understanding['context'])

        # Test the generated code
        test_results = self.tester.validate_generated_code(generated_code, request)

        return {
            'request': request,
            'generated_code': generated_code,
            'test_results': test_results,
            'understanding': understanding
        }

    def modify_codebase(self, request: str, codebase_path: Path, dry_run: bool = True) -> Dict[str, Any]:
        """Modify the codebase based on the request."""
        # Analyze the codebase first if not already done
        if not self.vector_db.get_statistics()['total_chunks']:
            self.console.print("[yellow]Analyzing codebase first...[/yellow]")
            self.analyzer.analyze_path(codebase_path, recursive=True)

        # Get context and plan changes
        context = self._get_codebase_context(request, codebase_path, max_chunks=10)
        planned_changes = self.llm_manager.plan_changes(request, context)

        results = {
            'request': request,
            'planned_changes': planned_changes,
            'executed_changes': [],
            'dry_run': dry_run
        }

        if not dry_run:
            # Execute the planned changes
            for change in planned_changes:
                try:
                    change_result = self._execute_change(change, codebase_path)
                    results['executed_changes'].append(change_result)
                except Exception as e:
                    results['executed_changes'].append({
                        'change': change,
                        'success': False,
                        'error': str(e)
                    })

        return results

    def _execute_change(self, change: CodeChange, base_path: Path) -> Dict[str, Any]:
        """Execute a single code change."""
        file_path = base_path / change.file_path

        if change.action == 'create':
            # Create new file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(change.content)

            return {
                'change': change,
                'success': True,
                'message': f"Created {file_path}"
            }

        elif change.action == 'modify':
            # Modify existing file
            if not file_path.exists():
                return {
                    'change': change,
                    'success': False,
                    'error': f"File {file_path} does not exist"
                }

            # For now, replace entire file content
            # In a more sophisticated version, you'd do targeted edits
            with open(file_path, 'w') as f:
                f.write(change.content)

            return {
                'change': change,
                'success': True,
                'message': f"Modified {file_path}"
            }

        elif change.action == 'delete':
            # Delete file
            if file_path.exists():
                file_path.unlink()
                return {
                    'change': change,
                    'success': True,
                    'message': f"Deleted {file_path}"
                }
            else:
                return {
                    'change': change,
                    'success': False,
                    'error': f"File {file_path} does not exist"
                }

        return {
            'change': change,
            'success': False,
            'error': f"Unknown action: {change.action}"
        }

    def explain_code(self, code_query: str, codebase_path: Path = None) -> str:
        """Explain code or answer questions about the codebase."""
        if codebase_path:
            # Search for relevant code
            relevant_chunks = self.vector_db.search_by_text(code_query, n_results=3)

            if relevant_chunks:
                # Get the actual code content for the most relevant chunk
                best_chunk = relevant_chunks[0]
                chunk_id = best_chunk['chunk_id']

                # Try to get the actual code content
                # This is simplified - in practice you'd decrypt the path and read the file
                code_content = "# Code content would be retrieved here"

                return self.llm_manager.analyze_code(code_content, code_query)
            else:
                return "No relevant code found in the codebase."
        else:
            return "Please specify a codebase path to analyze."

    def debug_code(self, code: str, error_description: str = "") -> Dict[str, Any]:
        """Debug code and suggest fixes."""
        # First, try to execute the code to see what happens
        exec_result = self.executor.execute_code(code)

        # Prepare debugging prompt
        debug_prompt = f"""Debug this code and suggest fixes:

Code:
```python
{code}
```

Execution result:
- Success: {exec_result.success}
- Output: {exec_result.output}
- Error: {exec_result.error}

{f"Additional context: {error_description}" if error_description else ""}

Provide:
1. What's wrong with the code
2. How to fix it
3. The corrected code
"""

        debug_analysis = self.llm_manager.generate_code(debug_prompt)

        return {
            'original_code': code,
            'execution_result': exec_result,
            'debug_analysis': debug_analysis,
            'error_description': error_description
        }

    def interactive_session(self, codebase_path: Path = None):
        """Start an interactive coding session."""
        self.console.print(Panel.fit(
            "[bold blue]🤖 Intelligent Code Assistant[/bold blue]\n"
            "I can help you understand, generate, modify, and debug code!\n\n"
            "Commands:\n"
            "- 'generate <description>' - Generate new code\n"
            "- 'modify <description>' - Modify existing codebase\n"
            "- 'explain <question>' - Explain code or concepts\n"
            "- 'debug <code>' - Debug problematic code\n"
            "- 'test <code>' - Test code execution\n"
            "- 'quit' - Exit",
            title="Welcome"
        ))

        if codebase_path:
            self.console.print(f"[green]Working with codebase: {codebase_path}[/green]")
            # Analyze codebase if not already done
            stats = self.vector_db.get_statistics()
            if stats['total_chunks'] == 0:
                self.console.print("[yellow]Analyzing codebase...[/yellow]")
                self.analyzer.analyze_path(codebase_path, recursive=True)
                self.console.print("[green]Codebase analysis complete![/green]")

        while True:
            try:
                user_input = input("\n🤖 What would you like me to help with? ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[blue]Goodbye! Happy coding! 👋[/blue]")
                    break

                if user_input.startswith('generate '):
                    request = user_input[9:]
                    result = self.generate_code(request, codebase_path)
                    self._display_generation_result(result)

                elif user_input.startswith('modify '):
                    request = user_input[7:]
                    if codebase_path:
                        result = self.modify_codebase(request, codebase_path, dry_run=True)
                        self._display_modification_result(result)
                    else:
                        self.console.print("[red]Please specify a codebase path for modifications[/red]")

                elif user_input.startswith('explain '):
                    question = user_input[8:]
                    explanation = self.explain_code(question, codebase_path)
                    self.console.print(Panel(explanation, title="Explanation"))

                elif user_input.startswith('debug '):
                    code = user_input[6:]
                    result = self.debug_code(code)
                    self._display_debug_result(result)

                elif user_input.startswith('test '):
                    code = user_input[5:]
                    exec_result = self.executor.execute_code(code)
                    self._display_execution_result(exec_result)

                else:
                    self.console.print("[yellow]Unknown command. Try 'generate', 'modify', 'explain', 'debug', or 'test'[/yellow]")

            except KeyboardInterrupt:
                self.console.print("\n[blue]Goodbye! Happy coding! 👋[/blue]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    def _display_generation_result(self, result: Dict[str, Any]):
        """Display code generation results."""
        code = result['generated_code']

        # Display the generated code
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Generated Code"))

        # Display test results
        test_results = result.get('test_results', {})
        if test_results.get('syntax_valid', False):
            self.console.print("[green]✅ Code syntax is valid[/green]")
        elif 'execution_error' in test_results:
            self.console.print(f"[red]❌ Syntax error: {test_results['execution_error']}[/red]")

    def _display_modification_result(self, result: Dict[str, Any]):
        """Display codebase modification results."""
        changes = result['planned_changes']

        table = Table(title="Planned Changes (Dry Run)")
        table.add_column("File", style="cyan")
        table.add_column("Action", style="green")
        table.add_column("Description", style="dim")

        for change in changes:
            table.add_row(change.file_path, change.action, change.description)

        self.console.print(table)

        if result['dry_run']:
            self.console.print("[yellow]This was a dry run. Use 'modify --execute' to apply changes.[/yellow]")

    def _display_debug_result(self, result: Dict[str, Any]):
        """Display debugging results."""
        exec_result = result['execution_result']

        if not exec_result.success:
            self.console.print(f"[red]❌ Error: {exec_result.error}[/red]")

        self.console.print(Panel(result['debug_analysis'], title="Debug Analysis"))

    def _display_execution_result(self, result):
        """Display code execution results."""
        if result.success:
            self.console.print("[green]✅ Code executed successfully[/green]")
            if result.output:
                self.console.print(f"Output:\n{result.output}")
        else:
            self.console.print(f"[red]❌ Execution failed: {result.error}[/red]")
