"""Perspective rectification of skewed floor plans.

A photographed or carelessly scanned plan is often rotated and slightly
keystoned, so the building outline is a generic quadrilateral rather than an
axis-aligned rectangle. This layer detects the outer quadrilateral of the
plan, computes a homography, and warps the image so the outline lands on an
exact rectangle. Rotation and perspective skew are corrected in one step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# The outline must dominate the page and be nearly rectangular, otherwise
# rectifying into a rectangle would distort the plan instead of straightening
# it (e.g. L-shaped buildings or pages holding several separate plans).
_MIN_OUTLINE_AREA_FRACTION = 0.02
_MIN_OUTLINE_INK_SHARE = 0.5
_MIN_QUAD_SIMILARITY = 0.85
_MIN_RECT_SIDE = 10
_CANVAS_MARGIN = 4


@dataclass
class RectifyResult:
    """Outcome of the rectification stage."""

    gray: np.ndarray
    applied: bool = False
    quad: np.ndarray | None = None
    homography: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order four corners clockwise starting at the top-left corner."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = points[:, 1] - points[:, 0]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def _approx_quad(points: np.ndarray) -> np.ndarray | None:
    """Reduce a convex outline to exactly four corners, or ``None``."""
    perimeter = cv2.arcLength(points, True)
    for scale in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(points, scale * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def find_outline_quad(mask: np.ndarray) -> np.ndarray | None:
    """Find the building outline of an ink mask as four ordered corners.

    ``mask`` is a binary image with non-zero ink on a zero background. The
    returned corners are ordered top-left, top-right, bottom-right,
    bottom-left. ``None`` is returned when no convincing rectangular outline
    exists.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    total_ink = sum(cv2.contourArea(c) for c in contours)
    if area < _MIN_OUTLINE_AREA_FRACTION * mask.size:
        return None
    if area < _MIN_OUTLINE_INK_SHARE * total_ink:
        return None

    # The convex hull tolerates door gaps in the outer wall: the hull restores
    # the true corners even when the stroke ring is broken.
    quad = _approx_quad(cv2.convexHull(contour))
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

    quad_area = abs(cv2.contourArea(quad))
    similarity = min(area, quad_area) / max(area, quad_area)
    if similarity < _MIN_QUAD_SIMILARITY:
        return None

    return _order_corners(quad)


def rectify_image(gray: np.ndarray) -> RectifyResult:
    """Straighten a skewed plan so its outline becomes an exact rectangle.

    The plan's outer quadrilateral is mapped onto an axis-aligned rectangle
    sized to the outline's longer opposite sides, so a rotated or keystoned
    house comes out upright and rectangular. When no rectangular outline can
    be found the image is returned unchanged and ``applied`` is ``False``.
    """
    ink = ((gray < 128).astype(np.uint8)) * 255
    quad = find_outline_quad(ink)
    if quad is None:
        return RectifyResult(gray=gray)

    width = int(
        round(
            max(
                np.linalg.norm(quad[0] - quad[1]),
                np.linalg.norm(quad[3] - quad[2]),
            )
        )
    )
    height = int(
        round(
            max(
                np.linalg.norm(quad[0] - quad[3]),
                np.linalg.norm(quad[1] - quad[2]),
            )
        )
    )
    if width < _MIN_RECT_SIDE or height < _MIN_RECT_SIDE:
        return RectifyResult(gray=gray)

    m = _CANVAS_MARGIN
    dst = np.float32(
        [
            [m, m],
            [width + m, m],
            [width + m, height + m],
            [m, height + m],
        ]
    )
    homography = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        gray,
        homography,
        (width + 2 * m, height + 2 * m),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )

    logger.info("Rectified outline quad onto %dx%d", width + 2 * m, height + 2 * m)
    return RectifyResult(
        gray=warped,
        applied=True,
        quad=quad,
        homography=homography,
    )


__all__ = ["RectifyResult", "find_outline_quad", "rectify_image"]
