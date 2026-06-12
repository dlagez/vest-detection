"""Utility functions for I/O operations."""

from pathlib import Path


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, create if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
