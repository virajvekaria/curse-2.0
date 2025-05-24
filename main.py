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
from pathlib import Path
import time
from typing import List, Dict, Any

from analyzer import CodebaseAnalyzer
from storage import VectorDatabase
from embeddings import EmbeddingGenerator
from code_assistant import IntelligentCodeAssistant
from llm_integration import LLMManager
from code_executor import SafeCodeExecutor, CodeTester
import config

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
@click.option('--provider', '-p', default='openai', help='LLM provider (openai, ollama)')
def generate(request: str, codebase: Path, provider: str):
    """Generate code based on natural language request."""

    console.print(f"[bold blue]Generating code for: '{request}'[/bold blue]")

    assistant = IntelligentCodeAssistant()
    assistant.llm_manager.set_provider(provider)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Generating code...", total=None)
        result = assistant.generate_code(request, codebase)
        progress.update(task, completed=True)

    assistant._display_generation_result(result)


@cli.command()
@click.argument('request')
@click.argument('codebase_path', type=click.Path(exists=True, path_type=Path))
@click.option('--execute', is_flag=True, help='Actually execute the changes (default is dry run)')
@click.option('--provider', '-p', default='openai', help='LLM provider (openai, ollama)')
def modify(request: str, codebase_path: Path, execute: bool, provider: str):
    """Modify codebase based on natural language request."""

    console.print(f"[bold blue]Planning modifications for: '{request}'[/bold blue]")

    assistant = IntelligentCodeAssistant()
    assistant.llm_manager.set_provider(provider)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("Planning changes...", total=None)
        result = assistant.modify_codebase(request, codebase_path, dry_run=not execute)
        progress.update(task, completed=True)

    assistant._display_modification_result(result)


@cli.command()
@click.argument('question')
@click.option('--codebase', '-c', type=click.Path(exists=True, path_type=Path), help='Codebase path for context')
@click.option('--provider', '-p', default='openai', help='LLM provider (openai, ollama)')
def explain(question: str, codebase: Path, provider: str):
    """Explain code or answer questions about the codebase."""

    console.print(f"[bold blue]Explaining: '{question}'[/bold blue]")

    assistant = IntelligentCodeAssistant()
    assistant.llm_manager.set_provider(provider)

    explanation = assistant.explain_code(question, codebase)
    console.print(Panel(explanation, title="Explanation"))


@cli.command()
@click.argument('code')
@click.option('--error-desc', '-e', help='Description of the error you\'re seeing')
@click.option('--provider', '-p', default='openai', help='LLM provider (openai, ollama)')
def debug(code: str, error_desc: str, provider: str):
    """Debug code and get suggestions for fixes."""

    console.print("[bold blue]Debugging code...[/bold blue]")

    assistant = IntelligentCodeAssistant()
    assistant.llm_manager.set_provider(provider)

    result = assistant.debug_code(code, error_desc or "")
    assistant._display_debug_result(result)


@cli.command()
@click.argument('code')
def test(code: str):
    """Test code execution safely."""

    console.print("[bold blue]Testing code execution...[/bold blue]")

    executor = SafeCodeExecutor()
    result = executor.execute_code(code)

    if result.success:
        console.print("[green]✅ Code executed successfully[/green]")
        if result.output:
            console.print(f"Output:\n{result.output}")
        console.print(f"Execution time: {result.execution_time:.3f}s")
    else:
        console.print(f"[red]❌ Execution failed: {result.error}[/red]")


@cli.command()
@click.option('--codebase', '-c', type=click.Path(exists=True, path_type=Path), help='Codebase path to work with')
@click.option('--provider', '-p', default='openai', help='LLM provider (openai, ollama)')
def chat(codebase: Path, provider: str):
    """Start an interactive AI coding session."""

    assistant = IntelligentCodeAssistant()
    assistant.llm_manager.set_provider(provider)
    assistant.interactive_session(codebase)


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


if __name__ == '__main__':
    cli()
