# Code Analysis System

A comprehensive code analysis system using structural parsing, cryptographic hashing, Merkle trees, and semantic embeddings for understanding codebases.

## Architecture

### Core Modules

- **`core/config.py`** - Configuration management and settings
- **`core/code_parser.py`** - Code parsing utilities for extracting meaningful chunks
- **`core/crypto.py`** - Cryptographic utilities for path encryption and data protection
- **`core/fingerprint.py`** - Fingerprinting and change detection using cryptographic hashes
- **`core/embeddings.py`** - Semantic embedding generation for code understanding
- **`core/storage.py`** - Vector database operations for storing and querying embeddings
- **`core/analyzer.py`** - Main analyzer orchestrating the entire pipeline
- **`core/llm_integration.py`** - LLM integration for code generation and modification
- **`core/code_executor.py`** - Code execution and testing utilities
- **`core/augment_engine.py`** - Main augment engine for code analysis and generation
- **`core/intelligent_augment_engine.py`** - Enhanced augment engine with recursive features

### Main Interface

- **`main.py`** - CLI interface for the code analysis system

## Features

- **Code Parsing**: Extract functions, classes, methods, and imports from source code
- **Cryptographic Security**: Encrypt file paths and generate secure hashes
- **Change Detection**: Use Merkle trees for hierarchical change detection
- **Semantic Understanding**: Generate embeddings for code similarity and search
- **LLM Integration**: Automatic code generation and modification capabilities
- **Vector Storage**: Efficient storage and retrieval of code embeddings

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the system:
```bash
python main.py
```

## Usage

The system can analyze codebases, generate embeddings, detect changes, and integrate with LLMs for code generation and modification.

### Basic Analysis
```python
from core.analyzer import CodebaseAnalyzer

analyzer = CodebaseAnalyzer()
results = analyzer.analyze_path("path/to/code")
```

### Code Generation
```python
from core.augment_engine import AugmentEngine

engine = AugmentEngine()
result = engine.process_command("create a function that calculates fibonacci")
```

## Configuration

Configuration is managed through `core/config.py`. Key settings include:

- Supported file extensions
- Ignore patterns for files/directories
- Database paths
- LLM integration settings
- Encryption settings

## Key Components

1. **Parser**: Extracts meaningful code chunks using AST parsing
2. **Fingerprinting**: Creates cryptographic fingerprints for change detection
3. **Embeddings**: Generates semantic embeddings for code understanding
4. **Storage**: Vector database for efficient similarity search
5. **LLM Integration**: Connects to language models for code generation
6. **Augment Engine**: Orchestrates the entire analysis and generation pipeline