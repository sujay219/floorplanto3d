"""Image loading with validation."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from floorplanto3d.errors import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
)

logger = logging.getLogger(__name__)

# Formats we are willing to decode.
ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "TIFF", "BMP", "GIF", "WEBP"})

# Guard rails against decompression bombs and absurd inputs.
DEFAULT_MAX_PIXELS = 50_000_000  # ~50 MP
DEFAULT_MAX_DIMENSION = 12_000


def load_image(
    source: str | Path | bytes | bytearray | Image.Image,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> tuple[np.ndarray, int, int, str | None]:
    """Load an image and return it as an RGB array.

    Args:
        source: File path, raw bytes, or a PIL image.
        max_pixels: Reject images larger than this many pixels.
        max_dimension: Reject images with a side longer than this.

    Returns:
        ``(rgb_array, width, height, detected_format)``.

    Raises:
        InvalidImageError: The bytes could not be decoded as an image.
        UnsupportedImageFormatError: The format is not in :data:`ALLOWED_FORMATS`.
        ImageTooLargeError: The image exceeds the size limits.
    """
    logger.debug("Loading image from %s", type(source).__name__)

    image = _open(source)
    already_decoded = isinstance(source, Image.Image)

    # An already-decoded PIL image carries no container format; only encoded
    # input can be checked against the allow-list.
    detected_format = (image.format or "").upper() or None
    if not already_decoded and detected_format not in ALLOWED_FORMATS:
        raise UnsupportedImageFormatError(
            f"Image format {detected_format or 'unknown'} is not supported",
            format=detected_format,
            supported=sorted(ALLOWED_FORMATS),
        )

    width, height = image.size

    if width <= 0 or height <= 0:
        raise InvalidImageError("Image has zero width or height")

    if width > max_dimension or height > max_dimension:
        raise ImageTooLargeError(
            f"Image dimension {width}x{height} exceeds the maximum of {max_dimension}",
            width=width,
            height=height,
            max_dimension=max_dimension,
        )

    if width * height > max_pixels:
        raise ImageTooLargeError(
            f"Image has {width * height} pixels which exceeds the maximum of {max_pixels}",
            width=width,
            height=height,
            max_pixels=max_pixels,
        )

    # Palette images with transparency are common for floor plans; flatten them.
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise InvalidImageError(f"Expected an RGB image, got shape {array.shape}")

    logger.info("Loaded image %dx%d (%s)", width, height, detected_format)
    return array, width, height, detected_format


def _open(source: str | Path | bytes | bytearray | Image.Image) -> Image.Image:
    """Open a source into a PIL image without decoding pixel data yet."""
    try:
        if isinstance(source, Image.Image):
            # Copy so we never mutate a caller-owned image.
            image = source.copy()
            image.format = source.format
            return image
        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise InvalidImageError("Image data is empty")
            return Image.open(io.BytesIO(bytes(source)))
        path = Path(source)
        if not path.exists():
            raise InvalidImageError(f"Image file not found: {path}")
        return Image.open(path)
    except InvalidImageError:
        raise
    except UnidentifiedImageError as exc:
        raise InvalidImageError("File is not a recognisable image") from exc
    except OSError as exc:
        raise InvalidImageError(f"Could not read image: {exc}") from exc


def encode_png(array: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes. Used by tests and fixtures."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()