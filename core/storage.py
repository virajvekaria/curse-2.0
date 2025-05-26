"""Vector database operations for storing and querying code embeddings."""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pickle
import core.config as config
from core.embeddings import CodeEmbedding
from core.crypto import PathEncryption


class VectorDatabase:
    """Manages storage and retrieval of code embeddings using simple file storage."""

    def __init__(self, db_path: str = str(config.DB_PATH)):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        # Simple file-based storage
        self.embeddings_file = self.db_path / "embeddings.json"
        self.vectors_file = self.db_path / "vectors.pkl"

        # Load existing data
        self.embeddings_data = self._load_embeddings()
        self.vectors_data = self._load_vectors()

        self.encryption = PathEncryption()

    def _load_embeddings(self) -> Dict[str, Dict]:
        """Load embeddings metadata from file."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        return {}

    def _load_vectors(self) -> Dict[str, np.ndarray]:
        """Load vector data from file."""
        if self.vectors_file.exists():
            try:
                with open(self.vectors_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading vectors: {e}")
        return {}

    def _save_embeddings(self):
        """Save embeddings metadata to file."""
        try:
            with open(self.embeddings_file, 'w') as f:
                json.dump(self.embeddings_data, f, indent=2)
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def _save_vectors(self):
        """Save vector data to file."""
        try:
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(self.vectors_data, f)
        except Exception as e:
            print(f"Error saving vectors: {e}")

    def store_embeddings(self, embeddings: List[CodeEmbedding]) -> bool:
        """Store code embeddings in the vector database."""
        try:
            if not embeddings:
                return True

            for embedding in embeddings:
                # Store metadata
                metadata = {
                    'name': embedding.metadata['name'],
                    'type': embedding.metadata['type'],
                    'parent': embedding.metadata.get('parent', ''),
                    'start_line': embedding.metadata['start_line'],
                    'end_line': embedding.metadata['end_line'],
                    'complexity': embedding.metadata['complexity'],
                    'has_docstring': embedding.metadata['has_docstring'],
                    'content_length': embedding.metadata['content_length'],
                    'token_count': embedding.token_count
                }

                # Create a searchable document text
                doc_text = f"{metadata['type']} {metadata['name']}"
                if metadata['parent']:
                    doc_text += f" in {metadata['parent']}"

                metadata['document'] = doc_text
                self.embeddings_data[embedding.chunk_id] = metadata

                # Store vector
                self.vectors_data[embedding.chunk_id] = embedding.embedding

            # Save to files
            self._save_embeddings()
            self._save_vectors()

            print(f"Stored {len(embeddings)} embeddings in vector database")
            return True

        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False

    def search_similar(self, query_embedding: np.ndarray,
                      n_results: int = 5,
                      filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar code chunks."""
        try:
            similarities = []

            # Calculate similarities with all stored vectors
            for chunk_id, vector in self.vectors_data.items():
                metadata = self.embeddings_data.get(chunk_id, {})

                # Apply filters if specified
                if filter_metadata:
                    skip = False
                    for key, value in filter_metadata.items():
                        if metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, vector)

                similarities.append({
                    'chunk_id': chunk_id,
                    'similarity': similarity,
                    'metadata': metadata,
                    'document': metadata.get('document', '')
                })

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:n_results]

        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def search_by_text(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Search for code chunks using text query."""
        try:
            # Simple text-based search
            results = []
            query_lower = query_text.lower()

            for chunk_id, metadata in self.embeddings_data.items():
                score = 0

                # Search in document text
                document = metadata.get('document', '') or ''
                if query_lower in document.lower():
                    score += 1.0

                # Search in name
                name = metadata.get('name', '') or ''
                if query_lower in name.lower():
                    score += 0.8

                # Search in type
                chunk_type = metadata.get('type', '') or ''
                if query_lower in chunk_type.lower():
                    score += 0.6

                # Search in parent
                parent = metadata.get('parent', '') or ''
                if query_lower in parent.lower():
                    score += 0.4

                if score > 0:
                    results.append({
                        'chunk_id': chunk_id,
                        'similarity': score,
                        'metadata': metadata,
                        'document': metadata.get('document', '')
                    })

            # Sort by score and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:n_results]

        except Exception as e:
            print(f"Error in text search: {e}")
            return []

    def filter_by_metadata(self, filters: Dict[str, Any]) -> List[Dict]:
        """Filter code chunks by metadata criteria."""
        try:
            results = []

            for chunk_id, metadata in self.embeddings_data.items():
                # Check if all filter criteria match
                match = True
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        match = False
                        break

                if match:
                    results.append({
                        'chunk_id': chunk_id,
                        'metadata': metadata,
                        'document': metadata.get('document', '')
                    })

            return results

        except Exception as e:
            print(f"Error filtering by metadata: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a specific chunk by its ID."""
        try:
            if chunk_id in self.embeddings_data:
                metadata = self.embeddings_data[chunk_id]
                embedding = self.vectors_data.get(chunk_id)

                return {
                    'chunk_id': chunk_id,
                    'metadata': metadata,
                    'document': metadata.get('document', ''),
                    'embedding': embedding.tolist() if embedding is not None else None
                }

            return None

        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
            return None

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete specific chunks from the database."""
        try:
            for chunk_id in chunk_ids:
                if chunk_id in self.embeddings_data:
                    del self.embeddings_data[chunk_id]
                if chunk_id in self.vectors_data:
                    del self.vectors_data[chunk_id]

            # Save changes
            self._save_embeddings()
            self._save_vectors()

            print(f"Deleted {len(chunk_ids)} chunks from database")
            return True

        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            count = len(self.embeddings_data)

            stats = {
                'total_chunks': count,
                'chunk_types': {},
                'complexity_distribution': {},
                'avg_content_length': 0
            }

            if count > 0:
                total_length = 0
                for metadata in self.embeddings_data.values():
                    # Count chunk types
                    chunk_type = metadata.get('type', 'unknown')
                    stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1

                    # Count complexity
                    complexity = metadata.get('complexity', 0)
                    stats['complexity_distribution'][complexity] = stats['complexity_distribution'].get(complexity, 0) + 1

                    # Sum content lengths
                    total_length += metadata.get('content_length', 0)

                stats['avg_content_length'] = total_length / count

            return stats

        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {'total_chunks': 0}

    def reset_database(self) -> bool:
        """Reset the entire database (use with caution)."""
        try:
            # Clear all data
            self.embeddings_data = {}
            self.vectors_data = {}

            # Remove files
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            if self.vectors_file.exists():
                self.vectors_file.unlink()

            print("Database reset successfully")
            return True

        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
