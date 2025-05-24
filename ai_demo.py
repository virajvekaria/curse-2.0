"""
Enhanced demo showcasing the AI-powered code assistant capabilities.
"""

from pathlib import Path
from code_assistant import IntelligentCodeAssistant
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def demo_code_generation():
    """Demonstrate AI code generation."""
    console.print(Panel.fit(
        "[bold blue]🤖 AI Code Generation Demo[/bold blue]\n"
        "Watch as the AI generates code from natural language!",
        title="Demo 1"
    ))
    
    assistant = IntelligentCodeAssistant()
    
    # Example requests
    requests = [
        "Create a function to calculate the factorial of a number",
        "Write a class for managing a simple todo list",
        "Generate a function to validate email addresses using regex"
    ]
    
    for i, request in enumerate(requests, 1):
        console.print(f"\n[bold cyan]Request {i}:[/bold cyan] {request}")
        
        try:
            result = assistant.generate_code(request)
            
            # Display the generated code
            code = result['generated_code']
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"Generated Code {i}"))
            
            # Show test results
            test_results = result['test_results']
            if test_results['syntax_valid']:
                console.print("[green]✅ Code syntax is valid[/green]")
            else:
                console.print(f"[red]❌ Syntax error: {test_results['execution_error']}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Note: Set OPENAI_API_KEY environment variable for actual AI generation[/yellow]")


def demo_code_testing():
    """Demonstrate safe code execution and testing."""
    console.print(Panel.fit(
        "[bold blue]🧪 Safe Code Testing Demo[/bold blue]\n"
        "Testing generated code safely in a sandboxed environment!",
        title="Demo 2"
    ))
    
    assistant = IntelligentCodeAssistant()
    
    # Test some sample code
    test_codes = [
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"Fibonacci(10) = {fibonacci(10)}")
""",
        """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

for i in range(2, 20):
    if is_prime(i):
        print(f"{i} is prime")
""",
        """
# This will cause an error
def divide_by_zero():
    return 10 / 0

divide_by_zero()
"""
    ]
    
    for i, code in enumerate(test_codes, 1):
        console.print(f"\n[bold cyan]Testing Code {i}:[/bold cyan]")
        
        # Show the code
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"Code to Test {i}"))
        
        # Execute and show results
        result = assistant.executor.execute_code(code)
        
        if result.success:
            console.print("[green]✅ Code executed successfully[/green]")
            if result.output:
                console.print(f"[dim]Output:[/dim]\n{result.output}")
        else:
            console.print(f"[red]❌ Execution failed: {result.error}[/red]")
        
        console.print(f"[dim]Execution time: {result.execution_time:.3f}s[/dim]")


def demo_code_debugging():
    """Demonstrate AI-powered debugging."""
    console.print(Panel.fit(
        "[bold blue]🐛 AI Debugging Demo[/bold blue]\n"
        "Watch the AI analyze and fix buggy code!",
        title="Demo 3"
    ))
    
    assistant = IntelligentCodeAssistant()
    
    # Buggy code examples
    buggy_codes = [
        """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: division by zero if empty list

print(calculate_average([]))
""",
        """
def find_max(lst):
    max_val = 0  # Bug: assumes all numbers are positive
    for num in lst:
        if num > max_val:
            max_val = num
    return max_val

print(find_max([-5, -2, -10]))
""",
        """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)  # Bug: no handling for negative numbers

print(factorial(-5))  # This will cause infinite recursion
"""
    ]
    
    for i, code in enumerate(buggy_codes, 1):
        console.print(f"\n[bold cyan]Debugging Code {i}:[/bold cyan]")
        
        # Show the buggy code
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"Buggy Code {i}"))
        
        try:
            # First execute to see the error
            result = assistant.executor.execute_code(code)
            
            if not result.success:
                console.print(f"[red]❌ Error detected: {result.error}[/red]")
                
                # Now debug it
                debug_result = assistant.debug_code(code, result.error)
                console.print(Panel(debug_result['debug_analysis'], title="AI Debug Analysis"))
            else:
                console.print("[yellow]Code executed without errors (but may have logical bugs)[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Demo error: {e}[/red]")
            console.print("[yellow]Note: Set OPENAI_API_KEY for actual AI debugging[/yellow]")


def demo_interactive_features():
    """Show interactive features."""
    console.print(Panel.fit(
        "[bold blue]💬 Interactive Features[/bold blue]\n"
        "The system supports interactive coding sessions!\n\n"
        "Available commands:\n"
        "• [cyan]python main.py generate 'create a sorting function'[/cyan]\n"
        "• [cyan]python main.py test 'print(\"Hello World\")'[/cyan]\n"
        "• [cyan]python main.py debug 'broken_code_here'[/cyan]\n"
        "• [cyan]python main.py chat[/cyan] - Start interactive session\n"
        "• [cyan]python main.py modify 'add error handling' ./codebase[/cyan]\n"
        "• [cyan]python main.py explain 'how does this work?' -c ./codebase[/cyan]",
        title="Demo 4"
    ))


def main():
    """Run the complete AI demo."""
    console.print(Panel.fit(
        "[bold green]🚀 AI-Powered Code Assistant Demo[/bold green]\n"
        "This demo showcases the enhanced capabilities:\n\n"
        "✅ [bold]Code Understanding[/bold] - Analyzes your codebase\n"
        "✅ [bold]AI Code Generation[/bold] - Creates code from natural language\n"
        "✅ [bold]Safe Execution[/bold] - Tests code in sandboxed environment\n"
        "✅ [bold]Intelligent Debugging[/bold] - Finds and fixes bugs\n"
        "✅ [bold]Codebase Modification[/bold] - Makes targeted changes\n"
        "✅ [bold]Interactive Chat[/bold] - Natural language coding sessions",
        title="Welcome to the Enhanced System!"
    ))
    
    try:
        # Demo 1: Code Generation
        demo_code_generation()
        
        input("\nPress Enter to continue to the next demo...")
        
        # Demo 2: Code Testing
        demo_code_testing()
        
        input("\nPress Enter to continue to the next demo...")
        
        # Demo 3: Debugging
        demo_code_debugging()
        
        input("\nPress Enter to see interactive features...")
        
        # Demo 4: Interactive Features
        demo_interactive_features()
        
        console.print("\n[bold green]🎉 Demo Complete![/bold green]")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("1. Set your OPENAI_API_KEY environment variable for full AI capabilities")
        console.print("2. Try: [cyan]python main.py chat[/cyan] for an interactive session")
        console.print("3. Use: [cyan]python main.py generate 'your request here'[/cyan] to generate code")
        console.print("4. Analyze your own codebase: [cyan]python main.py analyze /path/to/your/code[/cyan]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted. Thanks for watching![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        console.print("[yellow]This is expected if you haven't set up API keys yet.[/yellow]")


if __name__ == "__main__":
    main()
