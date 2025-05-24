# AI-Powered Code Analysis & Assistant System

A complete AI coding assistant inspired by Cursor that can understand, generate, modify, and debug code. This system combines advanced codebase analysis with LLM integration to provide intelligent coding assistance.

## Features

🔍 **Structural Code Parsing**
- Extracts meaningful code units (functions, classes, methods)
- Supports Python with extensible architecture for other languages
- Calculates complexity metrics and extracts documentation

🔐 **Cryptographic Fingerprinting**
- SHA-256 hashing of code chunks for change detection
- Merkle tree organization for hierarchical change tracking
- Encrypted file paths for privacy protection

🧠 **Semantic Understanding**
- Generates vector embeddings for code chunks
- Semantic similarity search capabilities
- Metadata-rich storage for advanced filtering

⚡ **Efficient Change Detection**
- Incremental updates - only processes changed files
- Millisecond-level change detection via hash comparison
- Parallel processing for large codebases

🔒 **Privacy-First Design**
- Encrypted file paths in storage
- No actual code content stored in embeddings database
- Local-first operation (no remote dependencies)

🤖 **AI-Powered Assistance**
- Natural language code generation
- Intelligent code modification and refactoring
- AI-powered debugging and error analysis
- Interactive coding sessions
- Context-aware suggestions

🧪 **Safe Code Execution**
- Sandboxed code testing environment
- Automatic validation and safety checks
- Real-time execution feedback
- Test case generation and validation

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Demos

```bash
# Basic analysis demo
python demo.py

# AI-powered features demo
python ai_demo.py
```

These will demonstrate all system capabilities including AI features.

### 2. Analyze Your Codebase

```bash
# Analyze current directory
python main.py analyze .

# Analyze specific directory recursively
python main.py analyze /path/to/your/code -r

# Analyze only specific file types
python main.py analyze . -e "py,js,ts"

# Force re-analysis of all files
python main.py analyze . -f
```

### 3. Search Your Code

```bash
# Semantic search
python main.py search "database connection"
python main.py search "error handling" --limit 10

# Filter by type
python main.py search "validation" --type function

# Set similarity threshold
python main.py search "authentication" --min-similarity 0.7
```

### 4. Browse Code Chunks

```bash
# List all chunks
python main.py list-chunks

# Filter by type
python main.py list-chunks --type class

# Filter by complexity
python main.py list-chunks --complexity 5

# Filter by parent class
python main.py list-chunks --parent "DataProcessor"
```

### 5. View Statistics

```bash
python main.py stats
```

### 6. AI-Powered Code Generation

```bash
# Generate code from natural language
python main.py generate "create a function to sort a list of dictionaries by a key"

# Generate with codebase context
python main.py generate "add error handling to the calculator" --codebase ./my_project
```

### 7. Intelligent Code Modification

```bash
# Plan modifications (dry run)
python main.py modify "add logging to all functions" ./my_project

# Execute modifications
python main.py modify "refactor the database connection code" ./my_project --execute
```

### 8. AI Debugging

```bash
# Debug problematic code
python main.py debug "def divide(a, b): return a/b" --error-desc "division by zero"
```

### 9. Safe Code Testing

```bash
# Test code execution safely
python main.py test "print('Hello, World!')"
```

### 10. Interactive AI Session

```bash
# Start interactive coding session
python main.py chat

# With codebase context
python main.py chat --codebase ./my_project
```

## System Architecture

### Core Components

1. **Parser** (`parser.py`)
   - AST-based code parsing
   - Extracts functions, classes, methods, imports
   - Calculates complexity metrics
   - Extensible for multiple languages

2. **Fingerprinting** (`fingerprint.py`)
   - SHA-256 content hashing
   - Merkle tree construction
   - Change detection algorithms
   - Fingerprint persistence

3. **Embeddings** (`embeddings.py`)
   - Semantic vector generation
   - Similarity calculations
   - Batch processing for efficiency
   - Fallback for offline operation

4. **Storage** (`storage.py`)
   - ChromaDB vector database
   - Metadata filtering
   - Similarity search
   - Privacy-preserving storage

5. **Analyzer** (`analyzer.py`)
   - Orchestrates the entire pipeline
   - Parallel file processing
   - Change detection coordination
   - Statistics collection

### Data Flow

```
Code Files → Parser → Chunks → Fingerprinting → Embeddings → Vector DB
                ↓              ↓                    ↓
            AST Analysis   Hash + Merkle      Semantic Vectors
```

## Configuration

Edit `config.py` to customize:

- **File Extensions**: Which file types to analyze
- **Chunk Sizes**: Minimum/maximum code chunk sizes
- **Embedding Model**: Which model to use for embeddings
- **Ignore Patterns**: Files/directories to skip
- **Database Path**: Where to store the vector database

## Privacy & Security

- **Encrypted Paths**: File paths are encrypted before storage
- **No Code Storage**: Only embeddings and metadata stored, not actual code
- **Local Operation**: Everything runs locally, no data sent to external services
- **Secure Keys**: Encryption keys stored with restricted permissions

## Performance

- **Parallel Processing**: Multi-threaded file analysis
- **Incremental Updates**: Only processes changed files
- **Efficient Storage**: Optimized vector database operations
- **Memory Management**: Batch processing to handle large codebases

## Extending the System

### Adding New Languages

1. Create a new parser class in `parser.py`
2. Implement the parsing logic for your language
3. Add the file extension to `config.py`
4. Register the parser in `UniversalParser`

### Custom Embedding Models

1. Modify `embeddings.py` to use your preferred model
2. Update the embedding dimension in `config.py`
3. Ensure the model can handle code-specific text

### Additional Metadata

1. Extend the `CodeChunk` dataclass in `parser.py`
2. Update the fingerprinting logic in `fingerprint.py`
3. Modify storage schema in `storage.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Errors**: Check file permissions for database directory
3. **Memory Issues**: Reduce batch size in `config.py`
4. **Slow Performance**: Enable parallel processing and check ignore patterns

### Debug Mode

Set environment variable for verbose output:
```bash
export DEBUG=1
python main.py analyze .
```

## Comparison with Cursor

| Feature | This Implementation | Cursor |
|---------|-------------------|---------|
| Structural Parsing | ✅ AST-based | ✅ Advanced |
| Fingerprinting | ✅ SHA-256 + Merkle | ✅ Proprietary |
| Embeddings | ✅ Local models | ✅ TurboPuffer |
| Privacy | ✅ Encrypted paths | ✅ Full encryption |
| Change Detection | ✅ Hash-based | ✅ Millisecond |
| Languages | 🔄 Python + extensible | ✅ Multi-language |
| Scale | 🔄 Medium codebases | ✅ Enterprise |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Inspired by the Cursor AI code editor and their innovative approach to codebase understanding.
