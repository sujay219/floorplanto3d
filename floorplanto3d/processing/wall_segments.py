"""Wall segmentation of the Step B structure masks.

Splits the horizontal and vertical structure masks into individual wall
strokes by connected components and measures each one (length, mean
thickness, area, bounding box) together with its outline polygon. This is
a measurement view of the masks for thresholding UIs: a client can drop
strokes below a thickness or length threshold and redraw the survivors,
and with no threshold the strokes tile the ``04_combined.png`` mask
exactly (combined = horizontal ∪ vertical).

Nothing here is classified or removed: every stroke of the masks is
returned, diagonals are absent because the Step B masks are axis-aligned
by construction, and the coordinate space is the normalized working image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class WallSegment:
    """One measured wall stroke of a Step B structure mask."""

    id: str
    orientation: str
    length: float
    thickness: float
    area: int
    bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[float, float], ...]


@dataclass
class WallSegmentationResult:
    """Outputs of wall segmentation."""

    segments: list[WallSegment]
    report: dict[str, Any]


def segment_walls(
    horizontal: Image.Image,
    vertical: Image.Image,
) -> WallSegmentationResult:
    """Segment the Step B masks into measured wall strokes.

    Args:
        horizontal: Step B horizontal structures (255 = ink).
        vertical: Step B vertical structures (255 = ink).

    Returns:
        A :class:`WallSegmentationResult` with one :class:`WallSegment`
        per connected component of either mask. Coordinates are in the
        normalized working image's pixel space.
    """
    segments: list[WallSegment] = []
    for orientation, mask in (("horizontal", horizontal), ("vertical", vertical)):
        binary = _mask_array(mask)
        segments.extend(_measure_components(binary, orientation))

    for index, segment in enumerate(segments, start=1):
        segment.id = f"seg_{index:03d}"

    report = {
        "phase": {"number": 3, "name": "wall_segments"},
        "coordinate_space": "normalized",
        "summary": {
            "total_segments": len(segments),
            "horizontal_segments": sum(1 for s in segments if s.orientation == "horizontal"),
            "vertical_segments": sum(1 for s in segments if s.orientation == "vertical"),
        },
    }
    return WallSegmentationResult(segments=segments, report=report)


def segment_to_dict(segment: WallSegment) -> dict[str, Any]:
    """The JSON form of one :class:`WallSegment`."""
    x, y, width, height = segment.bbox
    return {
        "id": segment.id,
        "orientation": segment.orientation,
        "length": round(segment.length, 2),
        "thickness": round(segment.thickness, 2),
        "area": segment.area,
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "polygon": [{"x": round(px, 2), "y": round(py, 2)} for px, py in segment.polygon],
    }


def _mask_array(mask: Image.Image) -> np.ndarray:
    """Boolean foreground array of a binary mask image."""
    return np.asarray(mask.convert("L")) > 0


def _measure_components(binary: np.ndarray, orientation: str) -> list[WallSegment]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    segments: list[WallSegment] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        length = float(max(width, height))
        thickness = float(area) / length if length else 0.0
        region = labels[y : y + height, x : x + width] == label
        segments.append(
            WallSegment(
                id="",
                orientation=orientation,
                length=length,
                thickness=thickness,
                area=area,
                bbox=(x, y, width, height),
                polygon=_outline(region, x, y),
            )
        )
    return segments


def _outline(
    region: np.ndarray, offset_x: int, offset_y: int
) -> tuple[tuple[float, float], ...]:
    """Outer contour of a component region, offset to image coordinates."""
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return ()
    largest = max(contours, key=cv2.contourArea)
    return tuple(
        (float(px + offset_x), float(py + offset_y)) for px, py in largest.reshape(-1, 2)
    )
