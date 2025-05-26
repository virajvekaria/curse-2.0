"""
Code Analysis System - A minimal implementation inspired by Cursor's codebase understanding.

This system:
1. Parses code structurally into meaningful chunks (functions, classes, etc.)
2. Creates cryptographic fingerprints for change detection
3. Organizes chunks in a Merkle tree for efficient change tracking
4. Generates semantic embeddings for code understanding
5. Stores embeddings with encrypted file paths for privacy
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Fix Windows Unicode issues
import sys
if sys.platform == "win32":
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
import time
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.analyzer import CodebaseAnalyzer
from core.storage import VectorDatabase
from core.embeddings import EmbeddingGenerator
from core.code_executor import SafeCodeExecutor
from core.augment_engine import AugmentEngine
from core.intelligent_augment_engine import IntelligentAugmentEngine
import core.config as config

console = Console()


@click.group()
def cli():
    """Code Analysis System - Understand your codebase like Cursor does."""
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('--recursive', '-r', is_flag=True, help='Analyze directory recursively')
@click.option('--force', '-f', is_flag=True, help='Force re-analysis of all files')
@click.option('--extensions', '-e', help='Comma-separated list of file extensions to analyze')
def analyze(path: Path, recursive: bool, force: bool, extensions: str):
    """Analyze a file or directory and build the knowledge base."""

    console.print(f"[bold blue]Analyzing codebase at: {path}[/bold blue]")

    # Initialize analyzer
    analyzer = CodebaseAnalyzer()

    # Parse extensions if provided
    if extensions:
        ext_list = [ext.strip() for ext in extensions.split(',')]
        if not all(ext.startswith('.') for ext in ext_list):
            ext_list = [f'.{ext}' if not ext.startswith('.') else ext for ext in ext_list]
        config.SUPPORTED_EXTENSIONS = set(ext_list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Analyze the codebase
        task = progress.add_task("Analyzing codebase...", total=None)

        start_time = time.time()
        results = analyzer.analyze_path(path, recursive=recursive, force_reanalysis=force)
        end_time = time.time()

        progress.update(task, completed=True)

    # Display results
    display_analysis_results(results, end_time - start_time)


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results to return')
@click.option('--type', '-t', help='Filter by chunk type (function, class, method)')
@click.option('--min-similarity', '-s', default=0.0, help='Minimum similarity threshold')
def search(query: str, limit: int, type: str, min_similarity: float):
    """Search for code chunks using semantic similarity."""

    console.print(f"[bold blue]Searching for: '{query}'[/bold blue]")

    # Initialize components
    db = VectorDatabase()
    embedding_gen = EmbeddingGenerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Searching...", total=None)

        # Search using text query (ChromaDB will handle the embedding)
        results = db.search_by_text(query, n_results=limit)

        # Filter by type if specified
        if type:
            results = [r for r in results if r['metadata'].get('type') == type]

        # Filter by similarity threshold
        results = [r for r in results if r['similarity'] >= min_similarity]

        progress.update(task, completed=True)

    # Display results
    display_search_results(results, query)


@cli.command()
@click.option('--type', '-t', help='Filter by chunk type')
@click.option('--complexity', '-c', type=int, help='Filter by complexity score')
@click.option('--parent', '-p', help='Filter by parent class')
def list_chunks(type: str, complexity: int, parent: str):
    """List code chunks with optional filtering."""

    db = VectorDatabase()

    # Build filter criteria
    filters = {}
    if type:
        filters['type'] = type
    if complexity is not None:
        filters['complexity'] = complexity
    if parent:
        filters['parent'] = parent

    console.print("[bold blue]Listing code chunks...[/bold blue]")

    if filters:
        results = db.filter_by_metadata(filters)
    else:
        # Get all chunks (limited to avoid overwhelming output)
        all_chunks = []
        for chunk_id, metadata in list(db.embeddings_data.items())[:50]:
            all_chunks.append({
                'chunk_id': chunk_id,
                'metadata': metadata,
                'document': metadata.get('document', '')
            })
        results = all_chunks

    display_chunk_list(results, filters)


@cli.command()
def stats():
    """Show database statistics."""

    console.print("[bold blue]Database Statistics[/bold blue]")

    db = VectorDatabase()
    stats = db.get_statistics()

    # Create statistics table
    table = Table(title="Codebase Analysis Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(stats['total_chunks']))
    table.add_row("Average Content Length", f"{stats['avg_content_length']:.1f} chars")

    # Chunk types
    for chunk_type, count in stats['chunk_types'].items():
        table.add_row(f"  {chunk_type.title()} Chunks", str(count))

    # Complexity distribution
    if stats['complexity_distribution']:
        avg_complexity = sum(k * v for k, v in stats['complexity_distribution'].items()) / sum(stats['complexity_distribution'].values())
        table.add_row("Average Complexity", f"{avg_complexity:.1f}")

    console.print(table)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reset the database?')
def reset():
    """Reset the entire database (destructive operation)."""

    db = VectorDatabase()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Resetting database...", total=None)
        success = db.reset_database()
        progress.update(task, completed=True)

    if success:
        console.print("[bold green]Database reset successfully![/bold green]")
    else:
        console.print("[bold red]Failed to reset database![/bold red]")


@cli.command()
@click.argument('request')
@click.option('--codebase', '-c', type=click.Path(exists=True, path_type=Path), help='Codebase path for context')
@click.option('--provider', '-p', default='ollama', help='LLM provider (ollama, openai)')
def generate(request: str, codebase: Path, provider: str):
    """Generate code based on natural language request."""

    console.print(f"[bold blue]Generating code for: '{request}'[/bold blue]")

    engine = AugmentEngine()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Generating code...", total=None)
        result = engine.process_command(request)
        progress.update(task, completed=True)

    console.print(Panel(result.get('generated_code', 'No code generated'), title="Generated Code"))


# Commands that require IntelligentCodeAssistant are temporarily disabled
# until the module is available


@cli.command()
@click.argument('code')
def test(code: str):
    """Test code execution safely."""

    console.print("[bold blue]Testing code execution...[/bold blue]")

    executor = SafeCodeExecutor()
    result = executor.execute_code(code)

    if result.success:
        console.print("[green]SUCCESS: Code executed successfully[/green]")
        if result.output:
            console.print(f"Output:\n{result.output}")
        console.print(f"Execution time: {result.execution_time:.3f}s")
    else:
        console.print(f"[red]ERROR: Execution failed: {result.error}[/red]")


# Chat command temporarily disabled


def display_analysis_results(results: Dict[str, Any], duration: float):
    """Display the results of codebase analysis."""

    table = Table(title="Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Files Processed", str(results.get('files_processed', 0)))
    table.add_row("Chunks Extracted", str(results.get('chunks_extracted', 0)))
    table.add_row("Embeddings Generated", str(results.get('embeddings_generated', 0)))
    table.add_row("Processing Time", f"{duration:.2f} seconds")

    # Changes summary
    changes = results.get('changes', {})
    if changes:
        table.add_row("Files Added", str(len(changes.get('added', []))))
        table.add_row("Files Modified", str(len(changes.get('modified', []))))
        table.add_row("Files Deleted", str(len(changes.get('deleted', []))))

    console.print(table)

    # Show Merkle tree hash if available
    if 'merkle_hash' in results:
        console.print(f"\n[bold]Merkle Tree Hash:[/bold] {results['merkle_hash'][:16]}...")


def display_search_results(results: List[Dict], query: str):
    """Display search results in a formatted table."""

    if not results:
        console.print(f"[yellow]No results found for '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Similarity", style="green")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Parent", style="dim")
    table.add_column("Lines", style="dim")
    table.add_column("Complexity", style="yellow")

    for result in results:
        metadata = result['metadata']
        similarity = f"{result['similarity']:.3f}"
        chunk_type = metadata.get('type', 'unknown')
        name = metadata.get('name', 'unnamed')
        parent = metadata.get('parent', '')
        lines = f"{metadata.get('start_line', 0)}-{metadata.get('end_line', 0)}"
        complexity = str(metadata.get('complexity', 0))

        table.add_row(similarity, chunk_type, name, parent, lines, complexity)

    console.print(table)


def display_chunk_list(results: List[Dict], filters: Dict):
    """Display a list of code chunks."""

    if not results:
        console.print("[yellow]No chunks found matching the criteria[/yellow]")
        return

    filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()]) if filters else "all chunks"

    table = Table(title=f"Code Chunks ({filter_desc})")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Parent", style="dim")
    table.add_column("Lines", style="dim")
    table.add_column("Complexity", style="yellow")
    table.add_column("Length", style="green")

    for result in results:
        metadata = result['metadata']
        chunk_type = metadata.get('type', 'unknown')
        name = metadata.get('name', 'unnamed')
        parent = metadata.get('parent', '')
        lines = f"{metadata.get('start_line', 0)}-{metadata.get('end_line', 0)}"
        complexity = str(metadata.get('complexity', 0))
        length = str(metadata.get('content_length', 0))

        table.add_row(chunk_type, name, parent, lines, complexity, length)

    console.print(table)


# Agentic commands temporarily disabled until AgenticCoder module is available


@cli.command()
@click.argument('prompt')
@click.argument('directory', type=click.Path(exists=True, path_type=Path))
@click.option('--provider', '-p', default='ollama', help='LLM provider (ollama, openai)')
@click.option('--max-iterations', '-i', default=5, help='Maximum iterations for self-correction')
@click.option('--save', '-s', is_flag=True, help='Save final code to file')
@click.option('--report', '-r', is_flag=True, help='Generate session report')
@click.option('--intelligent/--no-intelligent', default=True, help='Use intelligent planning engine (default: True)')
def augment(prompt: str, directory: Path, provider: str, max_iterations: int, save: bool, report: bool, intelligent: bool):
    """
    TRUE AUGMENT-STYLE CODING: Understand codebase, write code, execute, read terminal,
    self-correct based on errors, and iterate until success.

    This is the core Augment Code behavior that makes it so powerful.
    """
    console.print(Panel(
        f"🤖 **AUGMENT CODE ENGINE**\n\n"
        f"This will:\n"
        f"1. 🧠 Understand your entire codebase\n"
        f"2. 🎨 Generate code for your prompt\n"
        f"3. ⚡ Execute the code in terminal\n"
        f"4. 📖 Read terminal output and errors\n"
        f"5. 🔧 Self-correct based on errors\n"
        f"6. 🔄 Repeat until success (max {max_iterations} iterations)\n\n"
        f"📁 Directory: {directory}\n"
        f"🎯 Prompt: {prompt}\n"
        f"🤖 Provider: {provider}",
        title="🚀 Augment Engine",
        border_style="bold blue"
    ))

    try:
        if intelligent:
            # Use the Intelligent Augment Engine with proper planning
            console.print("[bold green]🧠 Using Intelligent Planning Engine[/bold green]")
            engine = IntelligentAugmentEngine()
            result = engine.augment_code_intelligent(prompt, directory)

            # Convert to session format for compatibility
            class IntelligentSession:
                def __init__(self, result):
                    self.final_success = result['success']
                    self.final_code = result['final_code']
                    self.total_iterations = result['iterations']
                    self.execution_time = result['execution_time']
                    self.plan = result.get('plan')
                    self.output = result['output']

            session = IntelligentSession(result)
        else:
            # Use the regular Augment Engine
            console.print("[bold blue]⚡ Using Regular Augment Engine[/bold blue]")
            engine = AugmentEngine()
            engine.max_iterations = max_iterations
            engine.llm_manager.set_provider(provider)
            session = engine.augment_code(prompt, directory, execute_in_terminal=True)

        # Save final code if requested and successful
        if save and session.final_success:
            if intelligent:
                # Save code for intelligent engine
                import re
                safe_prompt = re.sub(r'[^\w\s-]', '', prompt.replace(' ', '_'))[:50]
                filename = f"augment_{safe_prompt}.py"
                saved_path = directory / filename
                with open(saved_path, 'w', encoding='utf-8') as f:
                    f.write(session.final_code)
                console.print(f"[green]💾 Code saved to: {saved_path}[/green]")
            else:
                saved_path = engine.save_final_code(session)
                console.print(f"[green]💾 Code saved to: {saved_path}[/green]")

        # Generate report if requested
        if report and not intelligent:  # Regular engine has report functionality
            report_content = engine.get_session_report(session)
            report_path = directory / f"augment_session_report_{int(time.time())}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            console.print(f"[blue]📊 Session report saved to: {report_path}[/blue]")

        # Final status
        if session.final_success:
            console.print(Panel(
                f"🎉 **SUCCESS!**\n\n"
                f"✅ Code generated and executed successfully\n"
                f"🔄 Completed in {session.total_iterations} iteration(s)\n"
                f"⏱️  Total time: {session.execution_time:.2f} seconds\n\n"
                f"The code is working and ready to use!",
                title="🏆 Augment Session Complete",
                border_style="bold green"
            ))
        else:
            console.print(Panel(
                f"⚠️  **INCOMPLETE**\n\n"
                f"❌ Could not generate working code\n"
                f"🔄 Tried {session.total_iterations} iteration(s)\n"
                f"⏱️  Total time: {session.execution_time:.2f} seconds\n\n"
                f"Check the error analysis above for details.",
                title="⚠️ Augment Session Incomplete",
                border_style="bold yellow"
            ))

    except Exception as e:
        console.print(f"[red]❌ Augment session failed: {e}[/red]")
        raise


if __name__ == '__main__':
    cli()
