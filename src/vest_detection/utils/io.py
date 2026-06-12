"""I/O utility functions."""

from pathlib import Path
from typing import List


def list_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """List files in a directory, optionally filtered by extension."""
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    files = list(path.iterdir())
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    return sorted(files)
