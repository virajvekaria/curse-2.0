"""Fingerprinting and change detection system using cryptographic hashes."""

import hashlib
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from code_parser import CodeChunk
from crypto import PathEncryption, generate_chunk_id


@dataclass
class ChunkFingerprint:
    """Fingerprint of a code chunk."""
    chunk_id: str
    content_hash: str
    file_path_encrypted: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    parent: Optional[str]
    last_modified: float
    complexity_score: int


class MerkleNode:
    """Node in a Merkle tree for hierarchical change detection."""

    def __init__(self, name: str, is_file: bool = False):
        self.name = name
        self.is_file = is_file
        self.children: Dict[str, 'MerkleNode'] = {}
        self.chunk_hashes: List[str] = []
        self.combined_hash: Optional[str] = None

    def add_child(self, name: str, node: 'MerkleNode'):
        """Add a child node."""
        self.children[name] = node

    def add_chunk_hash(self, chunk_hash: str):
        """Add a chunk hash to this node."""
        self.chunk_hashes.append(chunk_hash)

    def calculate_hash(self) -> str:
        """Calculate the combined hash for this node."""
        all_hashes = []

        # Add chunk hashes
        all_hashes.extend(sorted(self.chunk_hashes))

        # Add children hashes
        for child_name in sorted(self.children.keys()):
            child_hash = self.children[child_name].calculate_hash()
            all_hashes.append(f"{child_name}:{child_hash}")

        # Combine all hashes
        combined = ''.join(all_hashes)
        self.combined_hash = hashlib.sha256(combined.encode()).hexdigest()
        return self.combined_hash


class FingerprintManager:
    """Manages fingerprints and change detection for the codebase."""

    def __init__(self, fingerprint_file: str = "fingerprints.json"):
        self.fingerprint_file = Path(fingerprint_file)
        self.encryption = PathEncryption()
        self.fingerprints: Dict[str, ChunkFingerprint] = {}
        self.merkle_root: Optional[MerkleNode] = None
        self.load_fingerprints()

    def load_fingerprints(self):
        """Load existing fingerprints from disk."""
        if self.fingerprint_file.exists():
            try:
                with open(self.fingerprint_file, 'r') as f:
                    data = json.load(f)

                self.fingerprints = {}
                for chunk_id, fp_data in data.get('fingerprints', {}).items():
                    self.fingerprints[chunk_id] = ChunkFingerprint(**fp_data)

            except Exception as e:
                print(f"Error loading fingerprints: {e}")
                self.fingerprints = {}

    def save_fingerprints(self):
        """Save fingerprints to disk."""
        data = {
            'fingerprints': {
                chunk_id: asdict(fp) for chunk_id, fp in self.fingerprints.items()
            },
            'merkle_root_hash': self.merkle_root.combined_hash if self.merkle_root else None,
            'last_updated': time.time()
        }

        with open(self.fingerprint_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_fingerprint(self, chunk: CodeChunk, file_path: Path) -> ChunkFingerprint:
        """Create a fingerprint for a code chunk."""
        content_hash = self.encryption.hash_content(chunk.content)
        encrypted_path = self.encryption.encrypt_path(str(file_path))

        chunk_id = generate_chunk_id(
            str(file_path),
            chunk.start_line,
            chunk.end_line,
            content_hash
        )

        return ChunkFingerprint(
            chunk_id=chunk_id,
            content_hash=content_hash,
            file_path_encrypted=encrypted_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            name=chunk.name,
            parent=chunk.parent,
            last_modified=time.time(),
            complexity_score=chunk.complexity_score
        )

    def update_fingerprints(self, chunks: List[CodeChunk], file_path: Path) -> Dict[str, str]:
        """Update fingerprints for chunks and return change summary."""
        changes = {
            'added': [],
            'modified': [],
            'deleted': [],
            'unchanged': []
        }

        # Get current chunk IDs for this file
        file_path_str = str(file_path)
        current_chunk_ids = set()

        # Process each chunk
        for chunk in chunks:
            fingerprint = self.create_fingerprint(chunk, file_path)
            current_chunk_ids.add(fingerprint.chunk_id)

            if fingerprint.chunk_id in self.fingerprints:
                old_fp = self.fingerprints[fingerprint.chunk_id]
                if old_fp.content_hash != fingerprint.content_hash:
                    changes['modified'].append(fingerprint.chunk_id)
                else:
                    changes['unchanged'].append(fingerprint.chunk_id)
            else:
                changes['added'].append(fingerprint.chunk_id)

            self.fingerprints[fingerprint.chunk_id] = fingerprint

        # Find deleted chunks (chunks that were in this file but no longer exist)
        for chunk_id, fp in list(self.fingerprints.items()):
            try:
                decrypted_path = self.encryption.decrypt_path(fp.file_path_encrypted)
                if decrypted_path == file_path_str and chunk_id not in current_chunk_ids:
                    changes['deleted'].append(chunk_id)
                    del self.fingerprints[chunk_id]
            except Exception:
                # If we can't decrypt the path, skip it
                continue

        return changes

    def build_merkle_tree(self, root_path: Path) -> MerkleNode:
        """Build a Merkle tree from the current fingerprints."""
        self.merkle_root = MerkleNode("root")

        # Group fingerprints by file path
        file_groups: Dict[str, List[ChunkFingerprint]] = {}

        for fp in self.fingerprints.values():
            try:
                file_path = self.encryption.decrypt_path(fp.file_path_encrypted)
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append(fp)
            except Exception:
                continue

        # Build tree structure
        for file_path, fingerprints in file_groups.items():
            self._add_file_to_tree(self.merkle_root, file_path, fingerprints, root_path)

        # Calculate hashes
        self.merkle_root.calculate_hash()
        return self.merkle_root

    def _add_file_to_tree(self, root: MerkleNode, file_path: str,
                         fingerprints: List[ChunkFingerprint], root_path: Path):
        """Add a file and its chunks to the Merkle tree."""
        try:
            rel_path = Path(file_path).relative_to(root_path)
            parts = rel_path.parts

            current_node = root

            # Navigate/create directory structure
            for part in parts[:-1]:  # All but the filename
                if part not in current_node.children:
                    current_node.children[part] = MerkleNode(part)
                current_node = current_node.children[part]

            # Add file node
            filename = parts[-1]
            if filename not in current_node.children:
                current_node.children[filename] = MerkleNode(filename, is_file=True)

            file_node = current_node.children[filename]

            # Add chunk hashes to file node
            for fp in fingerprints:
                file_node.add_chunk_hash(fp.content_hash)

        except Exception as e:
            print(f"Error adding file to tree: {e}")

    def detect_changes(self, old_merkle_hash: Optional[str]) -> bool:
        """Detect if there are changes by comparing Merkle tree hashes."""
        if not self.merkle_root:
            return True

        current_hash = self.merkle_root.combined_hash
        return old_merkle_hash != current_hash

    def get_file_changes(self, file_path: Path) -> Dict[str, List[str]]:
        """Get detailed changes for a specific file."""
        file_path_str = str(file_path)
        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }

        for chunk_id, fp in self.fingerprints.items():
            try:
                decrypted_path = self.encryption.decrypt_path(fp.file_path_encrypted)
                if decrypted_path == file_path_str:
                    # This is a simplified version - in practice you'd compare with previous state
                    changes['added'].append(f"{fp.name} ({fp.chunk_type})")
            except Exception:
                continue

        return changes
