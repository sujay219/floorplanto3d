"""Image preprocessing: binarisation, rectification and deskewing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from floorplanto3d.processing.rectify import rectify_image

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Outputs of the preprocessing stage."""

    gray: np.ndarray
    binary: np.ndarray
    scale: float = 1.0
    rotation_deg: float = 0.0
    deskewed: bool = False
    rectified: bool = False
    quad: np.ndarray | None = None
    homography: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB array to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an RGB or grayscale image, got shape {image.shape}")
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Estimate a small global skew angle in degrees.

    Floor plan scans are frequently slightly rotated. This uses the
    minAreaRect orientation of the ink distribution, which is cheap and robust
    for near-axis-aligned drawings.
    """
    ink = (gray < 128).astype(np.uint8)
    if ink.sum() == 0:
        return 0.0

    coords = np.column_stack(np.nonzero(ink))
    # Too few ink pixels for a meaningful orientation.
    if len(coords) < 10:
        return 0.0

    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[2]

    # Normalise into (-45, 45].
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if not np.isfinite(angle) or abs(angle) > max_angle:
        return 0.0
    return float(angle)


def rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a grayscale image about its centre, preserving size."""
    if abs(angle_deg) < 1e-3:
        return gray
    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    border = int(np.ceil(max(height, width) * 0.5))
    matrix[0, 2] += border - center[0]
    matrix[1, 2] += border - center[1]
    return cv2.warpAffine(
        gray,
        matrix,
        (width + 2 * border, height + 2 * border),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )


def _binarise(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    """Adaptive threshold into dark-ink-on-white."""
    block_size = max(3, block_size | 1)  # must be odd and >= 3
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c,
    )


def preprocess(
    image: np.ndarray,
    *,
    deskew: bool = True,
    rectify: bool = True,
    block_size: int = 15,
    c: int = 10,
) -> PreprocessResult:
    """Binarise a floor plan scan into dark-ink-on-white.

    Adaptive thresholding is used rather than a global Otsu threshold because
    scanned plans routinely contain both shaded and unshaded regions.

    Skew is corrected before thresholding. When the plan has a rectangular
    outline, the outline quadrilateral is warped onto an exact rectangle,
    absorbing arbitrary rotation and perspective skew at once. Otherwise a
    small global rotation is removed as a fallback.

    Args:
        image: RGB array.
        deskew: Correct a small global rotation when rectification does not
            apply.
        rectify: Warp a detected rectangular outline onto an exact rectangle.
        block_size: Odd neighbourhood size for adaptive thresholding.
        c: Bias subtracted from the local mean.

    Returns:
        A :class:`PreprocessResult`.
    """
    warnings: list[str] = []
    gray = to_gray(image)

    # Mild blur suppresses JPEG ringing without erasing thin wall strokes.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    rotation_deg = 0.0
    rectified = False
    quad = None
    homography = None

    if rectify:
        result = rectify_image(gray)
        if result.applied:
            gray = result.gray
            rectified = True
            quad = result.quad
            homography = result.homography

    if not rectified and deskew:
        rotation_deg = estimate_skew(gray)
        if abs(rotation_deg) >= 0.2:
            gray = rotate(gray, rotation_deg)
            logger.info("Deskewed image by %.2f degrees", rotation_deg)
        else:
            rotation_deg = 0.0

    binary = _binarise(gray, block_size, c)

    ink_ratio = float(np.count_nonzero(binary)) / binary.size
    logger.info(
        "Binarised: ink ratio %.3f, rotation %.2f deg, rectified %s",
        ink_ratio,
        rotation_deg,
        rectified,
    )

    if ink_ratio > 0.6:
        warnings.append(
            "ink coverage above 60%; the image may be inverted or too dense"
        )
    elif ink_ratio < 0.002:
        warnings.append("ink coverage below 0.2%; the image may be blank")

    # Close pinholes so strokes stay connected, without bridging door gaps.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    return PreprocessResult(
        gray=gray,
        binary=binary,
        rotation_deg=rotation_deg,
        deskewed=abs(rotation_deg) >= 0.2,
        rectified=rectified,
        quad=quad,
        homography=homography,
        warnings=warnings,
    )