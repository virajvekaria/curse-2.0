"""
Core analyzer that orchestrates the entire code analysis pipeline.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Set
import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.code_parser import UniversalParser, CodeChunk
from core.fingerprint import FingerprintManager, ChunkFingerprint
from core.embeddings import EmbeddingGenerator, CodeEmbedding
from core.storage import VectorDatabase
import core.config as config


class CodebaseAnalyzer:
    """Main analyzer that coordinates parsing, fingerprinting, and embedding generation."""

    def __init__(self):
        self.parser = UniversalParser()
        self.fingerprint_manager = FingerprintManager()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'files_processed': 0,
            'chunks_extracted': 0,
            'embeddings_generated': 0,
            'files_skipped': 0,
            'errors': 0
        }

    def analyze_path(self, path: Path, recursive: bool = True,
                    force_reanalysis: bool = False) -> Dict[str, Any]:
        """Analyze a file or directory path."""

        self._reset_stats()

        if path.is_file():
            files_to_process = [path]
        else:
            files_to_process = self._discover_files(path, recursive)

        print(f"Found {len(files_to_process)} files to analyze")

        # Process files
        all_changes = {'added': [], 'modified': [], 'deleted': [], 'unchanged': []}

        # Use thread pool for parallel processing
        max_workers = min(4, len(files_to_process))  # Limit concurrent processing

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_file, file_path, force_reanalysis): file_path
                for file_path in files_to_process
            }

            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_changes = future.result()
                    if file_changes:
                        # Merge changes
                        for change_type, chunk_ids in file_changes.items():
                            all_changes[change_type].extend(chunk_ids)

                        with self._lock:
                            self.stats['files_processed'] += 1
                    else:
                        with self._lock:
                            self.stats['files_skipped'] += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    with self._lock:
                        self.stats['errors'] += 1

        # Build Merkle tree for change detection
        merkle_root = self.fingerprint_manager.build_merkle_tree(path if path.is_dir() else path.parent)

        # Save fingerprints
        self.fingerprint_manager.save_fingerprints()

        # Prepare results
        results = {
            'files_processed': self.stats['files_processed'],
            'chunks_extracted': self.stats['chunks_extracted'],
            'embeddings_generated': self.stats['embeddings_generated'],
            'files_skipped': self.stats['files_skipped'],
            'errors': self.stats['errors'],
            'changes': all_changes,
            'merkle_hash': merkle_root.combined_hash if merkle_root else None
        }

        return results

    def _process_file(self, file_path: Path, force_reanalysis: bool = False) -> Dict[str, List[str]]:
        """Process a single file."""

        try:
            # Check if file should be processed
            if not self._should_process_file(file_path):
                return None

            # Parse the file
            chunks = self.parser.parse_file(file_path)
            if not chunks:
                return None

            with self._lock:
                self.stats['chunks_extracted'] += len(chunks)

            # Update fingerprints and detect changes
            changes = self.fingerprint_manager.update_fingerprints(chunks, file_path)

            # Generate embeddings for new or modified chunks
            chunks_to_embed = []

            if force_reanalysis:
                # Re-embed all chunks
                chunks_to_embed = [(chunk, fp.chunk_id) for chunk, fp in
                                 zip(chunks, [self.fingerprint_manager.create_fingerprint(c, file_path) for c in chunks])]
            else:
                # Only embed new or modified chunks
                for chunk in chunks:
                    fp = self.fingerprint_manager.create_fingerprint(chunk, file_path)
                    if fp.chunk_id in changes['added'] or fp.chunk_id in changes['modified']:
                        chunks_to_embed.append((chunk, fp.chunk_id))

            if chunks_to_embed:
                # Generate embeddings
                embeddings = self.embedding_generator.batch_generate_embeddings(chunks_to_embed)

                # Store embeddings
                if embeddings:
                    self.vector_db.store_embeddings(embeddings)

                    with self._lock:
                        self.stats['embeddings_generated'] += len(embeddings)

            return changes

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def _discover_files(self, root_path: Path, recursive: bool) -> List[Path]:
        """Discover files to analyze in a directory."""

        files = []

        if recursive:
            for root, dirs, filenames in os.walk(root_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore_path(Path(root) / d)]

                for filename in filenames:
                    file_path = Path(root) / filename
                    if self._should_process_file(file_path):
                        files.append(file_path)
        else:
            # Only process files in the immediate directory
            for file_path in root_path.iterdir():
                if file_path.is_file() and self._should_process_file(file_path):
                    files.append(file_path)

        return files

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed."""

        # Check if path should be ignored
        if self._should_ignore_path(file_path):
            return False

        # Check file extension
        if file_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return False

        # Check file size (skip very large files)
        try:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                print(f"Skipping large file: {file_path}")
                return False
        except OSError:
            return False

        return True

    def _should_ignore_path(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns."""

        path_str = str(path)
        path_name = path.name

        for pattern in config.IGNORE_PATTERNS:
            if fnmatch.fnmatch(path_name, pattern) or fnmatch.fnmatch(path_str, pattern):
                return True

        return False

    def _reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'files_processed': 0,
            'chunks_extracted': 0,
            'embeddings_generated': 0,
            'files_skipped': 0,
            'errors': 0
        }

    def analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file and return detailed results."""

        if not self._should_process_file(file_path):
            return {'error': 'File should not be processed'}

        try:
            # Parse the file
            chunks = self.parser.parse_file(file_path)

            # Create fingerprints
            fingerprints = []
            for chunk in chunks:
                fp = self.fingerprint_manager.create_fingerprint(chunk, file_path)
                fingerprints.append(fp)

            # Generate embeddings
            chunks_with_ids = [(chunk, fp.chunk_id) for chunk, fp in zip(chunks, fingerprints)]
            embeddings = self.embedding_generator.batch_generate_embeddings(chunks_with_ids)

            return {
                'file_path': str(file_path),
                'chunks': len(chunks),
                'fingerprints': len(fingerprints),
                'embeddings': len(embeddings),
                'chunk_details': [
                    {
                        'name': chunk.name,
                        'type': chunk.chunk_type,
                        'lines': f"{chunk.start_line}-{chunk.end_line}",
                        'complexity': chunk.complexity_score,
                        'content_length': len(chunk.content)
                    }
                    for chunk in chunks
                ]
            }

        except Exception as e:
            return {'error': str(e)}

    def get_similar_chunks(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Find chunks similar to the given query."""

        return self.vector_db.search_by_text(query_text, n_results=limit)

    def get_chunk_details(self, chunk_id: str) -> Dict:
        """Get detailed information about a specific chunk."""

        # Get from vector database
        chunk_data = self.vector_db.get_chunk_by_id(chunk_id)

        if not chunk_data:
            return {'error': 'Chunk not found'}

        # Get fingerprint information
        fingerprint = self.fingerprint_manager.fingerprints.get(chunk_id)

        result = {
            'chunk_id': chunk_id,
            'metadata': chunk_data['metadata'],
            'document': chunk_data['document']
        }

        if fingerprint:
            try:
                # Decrypt file path for display
                file_path = self.fingerprint_manager.encryption.decrypt_path(fingerprint.file_path_encrypted)
                result['file_path'] = file_path
                result['content_hash'] = fingerprint.content_hash
                result['last_modified'] = fingerprint.last_modified
            except Exception:
                result['file_path'] = '[encrypted]'

        return result
