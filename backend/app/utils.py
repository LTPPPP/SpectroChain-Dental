import hashlib

def calculate_hash(file_content: bytes) -> str:
    """
    Calculates the SHA-256 hash of the given file content.

    Args:
        file_content: The content of the file in bytes.

    Returns:
        The hex digest of the SHA-256 hash as a string.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_content)
    return sha256_hash.hexdigest() 