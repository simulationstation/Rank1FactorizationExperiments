"""I/O helpers for pipeline artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return the Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_text(path: str | Path, content: str) -> Path:
    """Write text content to a file and return the Path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def list_files(path: str | Path, patterns: Iterable[str]) -> list[Path]:
    """List files matching glob patterns within a directory."""
    base = Path(path)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(base.glob(pattern)))
    return files
