"""Cryptographic utilities for path encryption and data protection."""

import os
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import config


class PathEncryption:
    """Handles encryption and decryption of file paths for privacy protection."""
    
    def __init__(self, key_file: str = config.ENCRYPTION_KEY_FILE):
        self.key_file = Path(key_file)
        self.cipher = self._get_or_create_cipher()
    
    def _get_or_create_cipher(self) -> Fernet:
        """Get existing cipher or create a new one."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Make key file read-only for security
            os.chmod(self.key_file, 0o600)
        
        return Fernet(key)
    
    def encrypt_path(self, file_path: str) -> str:
        """Encrypt a file path."""
        path_bytes = str(file_path).encode('utf-8')
        encrypted = self.cipher.encrypt(path_bytes)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_path(self, encrypted_path: str) -> str:
        """Decrypt a file path."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_path.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def hash_content(self, content: str) -> str:
        """Generate a cryptographic hash of content."""
        digest = hashes.Hash(hashes.SHA256())
        digest.update(content.encode('utf-8'))
        return digest.finalize().hex()


def generate_chunk_id(file_path: str, start_line: int, end_line: int, content_hash: str) -> str:
    """Generate a unique ID for a code chunk."""
    chunk_info = f"{file_path}:{start_line}:{end_line}:{content_hash}"
    digest = hashes.Hash(hashes.SHA256())
    digest.update(chunk_info.encode('utf-8'))
    return digest.finalize().hex()[:16]  # Use first 16 chars for readability
