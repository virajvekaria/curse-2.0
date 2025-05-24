"""Semantic embedding generation for code understanding."""

import numpy as np
from typing import List, Dict, Optional
import hashlib
import tiktoken
from dataclasses import dataclass
import config
from code_parser import CodeChunk


@dataclass
class CodeEmbedding:
    """Represents a semantic embedding of a code chunk."""
    chunk_id: str
    embedding: np.ndarray
    metadata: Dict
    token_count: int


class EmbeddingGenerator:
    """Generates semantic embeddings for code chunks."""

    def __init__(self, model_name: str = config.EMBEDDINGS_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            # Use simple hash-based embeddings for this demo
            # In production, you'd want to use a proper transformer model
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print(f"Using simple hash-based embeddings for demo")
            self.model = None  # We'll use hash-based embeddings
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to simple text-based embeddings")
            self.model = None
            self.tokenizer = None

    def generate_embedding(self, chunk: CodeChunk, chunk_id: str) -> Optional[CodeEmbedding]:
        """Generate semantic embedding for a code chunk."""
        try:
            # Prepare text for embedding
            text = self._prepare_text(chunk)

            # Use simple hash-based embedding for this demo
            embedding = self._simple_embedding(text)

            # Count tokens
            token_count = len(self.tokenizer.encode(text)) if self.tokenizer else len(text.split())

            # Prepare metadata
            metadata = {
                'name': chunk.name,
                'type': chunk.chunk_type,
                'parent': chunk.parent,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'complexity': chunk.complexity_score,
                'has_docstring': chunk.docstring is not None,
                'content_length': len(chunk.content)
            }

            return CodeEmbedding(
                chunk_id=chunk_id,
                embedding=embedding,
                metadata=metadata,
                token_count=token_count
            )

        except Exception as e:
            print(f"Error generating embedding for chunk {chunk_id}: {e}")
            return None

    def _prepare_text(self, chunk: CodeChunk) -> str:
        """Prepare code chunk text for embedding generation."""
        parts = []

        # Add type and name information
        if chunk.chunk_type and chunk.name:
            parts.append(f"{chunk.chunk_type} {chunk.name}")

        # Add parent context for methods
        if chunk.parent:
            parts.append(f"in class {chunk.parent}")

        # Add docstring if available
        if chunk.docstring:
            parts.append(f"Documentation: {chunk.docstring}")

        # Add the actual code content
        parts.append(f"Code: {chunk.content}")

        return " ".join(parts)

    def _simple_embedding(self, text: str, dimension: int = 384) -> np.ndarray:
        """Generate a simple hash-based embedding as fallback."""
        # This is a very basic approach - in practice you'd want a proper model
        import hashlib

        # Create multiple hash values to form a vector
        embedding = []
        for i in range(dimension // 32):  # SHA256 gives 32 bytes
            hash_input = f"{text}_{i}".encode()
            hash_bytes = hashlib.sha256(hash_input).digest()
            # Convert bytes to normalized floats
            hash_floats = [b / 255.0 - 0.5 for b in hash_bytes]
            embedding.extend(hash_floats)

        # Pad or truncate to exact dimension
        if len(embedding) < dimension:
            embedding.extend([0.0] * (dimension - len(embedding)))
        else:
            embedding = embedding[:dimension]

        return np.array(embedding, dtype=np.float32)

    def batch_generate_embeddings(self, chunks_with_ids: List[tuple]) -> List[CodeEmbedding]:
        """Generate embeddings for multiple chunks efficiently."""
        embeddings = []

        # Use individual generation for simplicity
        for chunk, chunk_id in chunks_with_ids:
            embedding = self.generate_embedding(chunk, chunk_id)
            if embedding:
                embeddings.append(embedding)

        return embeddings

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def find_similar_chunks(self, query_embedding: np.ndarray,
                          embeddings: List[CodeEmbedding],
                          top_k: int = 5) -> List[tuple]:
        """Find the most similar code chunks to a query embedding."""
        similarities = []

        for code_embedding in embeddings:
            similarity = self.calculate_similarity(query_embedding, code_embedding.embedding)
            similarities.append((similarity, code_embedding))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        return similarities[:top_k]
