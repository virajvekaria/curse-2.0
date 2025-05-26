"""Semantic embedding generation for code understanding."""

import numpy as np
from typing import List, Dict, Optional
import tiktoken
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
import core.config as config
from core.code_parser import CodeChunk


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
        """Load the embedding model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"Loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to simple text-based embeddings")
            self.model = None
            self.tokenizer = None

    @torch.no_grad()
    def _transformer_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using transformer model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state.squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=0)
            sum_mask = torch.clamp(mask_expanded.sum(dim=0), min=1e-9)
            embedding = sum_hidden / sum_mask
        else:
            raise ValueError("Unexpected model output structure.")

        return embedding.cpu().numpy()

    def _simple_embedding(self, text: str, dimension: int = 384) -> np.ndarray:
        """Generate a simple hash-based embedding as fallback."""
        import hashlib
        embedding = []
        for i in range(dimension // 32):
            hash_input = f"{text}_{i}".encode()
            hash_bytes = hashlib.sha256(hash_input).digest()
            hash_floats = [b / 255.0 - 0.5 for b in hash_bytes]
            embedding.extend(hash_floats)
        if len(embedding) < dimension:
            embedding.extend([0.0] * (dimension - len(embedding)))
        else:
            embedding = embedding[:dimension]
        return np.array(embedding, dtype=np.float32)

    def _prepare_text(self, chunk: CodeChunk) -> str:
        """Prepare code chunk text for embedding generation."""
        parts = []
        if chunk.chunk_type and chunk.name:
            parts.append(f"{chunk.chunk_type} {chunk.name}")
        if chunk.parent:
            parts.append(f"in class {chunk.parent}")
        if chunk.docstring:
            parts.append(f"Documentation: {chunk.docstring}")
        parts.append(f"Code: {chunk.content}")
        return " ".join(parts)

    def generate_embedding(self, chunk: CodeChunk, chunk_id: str) -> Optional[CodeEmbedding]:
        """Generate semantic embedding for a code chunk."""
        try:
            text = self._prepare_text(chunk)
            if self.model and self.tokenizer:
                embedding = self._transformer_embedding(text)
                token_count = len(self.tokenizer.encode(text))
            else:
                embedding = self._simple_embedding(text)
                token_count = len(text.split())

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

    def batch_generate_embeddings(self, chunks_with_ids: List[tuple]) -> List[CodeEmbedding]:
        """Generate embeddings for multiple chunks efficiently."""
        embeddings = []
        for chunk, chunk_id in chunks_with_ids:
            embedding = self.generate_embedding(chunk, chunk_id)
            if embedding:
                embeddings.append(embedding)
        return embeddings

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
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
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
