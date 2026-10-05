"""Structured logging configuration.

Image contents are never logged. Only metadata that helps operators diagnose a
request: dimensions, stage timings, and detection counts.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

LOGGER_NAME = "floorplanto3d"

# Attributes LogRecord always carries; anything else is ours to format.
_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"asctime", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def configure_logging(
    level: str | int | None = None, *, json_output: bool | None = None
) -> logging.Logger:
    """Configure and return the package logger.

    Args:
        level: Log level. Defaults to ``FLOORPLANTO3D_LOG_LEVEL`` or ``INFO``.
        json_output: Emit JSON lines. Defaults to ``FLOORPLANTO3D_LOG_FORMAT=json``.

    Returns:
        The configured ``floorplanto3d`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if level is None:
        level = os.environ.get("FLOORPLANTO3D_LOG_LEVEL", "INFO")

    if json_output is None:
        json_output = os.environ.get("FLOORPLANTO3D_LOG_FORMAT", "").lower() == "json"

    # Replace our own handlers only, so host applications keep theirs.
    for handler in list(logger.handlers):
        if getattr(handler, "_floorplanto3d", False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    handler._floorplanto3d = True  # type: ignore[attr-defined]

    logger.addHandler(handler)
    logger.setLevel(level)
    # Keep our records out of the root logger to avoid duplicate output.
    logger.propagate = False

    # These libraries are extremely chatty at INFO.
    for noisy in ("PIL", "urllib3", "matplotlib", "shapely", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger."""
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")