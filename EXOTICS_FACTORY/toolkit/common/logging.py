"""Logging helpers for pipelines."""
from __future__ import annotations

import logging


def configure_logger(name: str = "exotics_factory", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a simple format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
