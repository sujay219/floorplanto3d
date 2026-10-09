"""Library exception hierarchy.

Every error carries a machine-readable ``code`` so the HTTP layer can map
failures onto structured responses without inspecting messages.
"""

from __future__ import annotations

from typing import Any


class FloorPlanError(Exception):
    """Base class for all errors raised by this library."""

    code = "floorplan_error"
    status_code = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


# --- Input errors -------------------------------------------------------


class InvalidImageError(FloorPlanError):
    """The supplied image could not be decoded or is empty."""

    code = "invalid_image"
    status_code = 422


class UnsupportedImageFormatError(InvalidImageError):
    """The image format is not one we decode."""

    code = "unsupported_image_format"
    status_code = 415


class ImageTooLargeError(InvalidImageError):
    """The image exceeds the configured pixel or dimension limits."""

    code = "image_too_large"
    status_code = 413


class MissingImageError(FloorPlanError):
    """No image was supplied."""

    code = "missing_image"
    status_code = 400


# --- Processing errors --------------------------------------------------


class InsufficientGeometryError(FloorPlanError):
    """No usable architectural geometry could be recovered."""

    code = "insufficient_geometry"
    status_code = 422


class ProcessingError(FloorPlanError):
    """An unexpected failure occurred inside the pipeline."""

    code = "processing_error"
    status_code = 500