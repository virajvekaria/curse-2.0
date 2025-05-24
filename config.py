"""Configuration settings for the code analysis system."""

import os
from pathlib import Path

# Database settings
DB_PATH = Path("./codebase_db")
EMBEDDINGS_MODEL = "microsoft/codebert-base"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"}

# Chunk settings
MIN_CHUNK_SIZE = 50  # Minimum characters for a code chunk
MAX_CHUNK_SIZE = 2000  # Maximum characters for a code chunk

# Encryption settings
ENCRYPTION_KEY_FILE = ".encryption_key"

# Embedding settings
EMBEDDING_DIMENSION = 768  # CodeBERT embedding dimension
BATCH_SIZE = 32

# File patterns to ignore
IGNORE_PATTERNS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store"
}
