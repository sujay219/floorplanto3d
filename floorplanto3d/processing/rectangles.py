"""Phase 2: rectangle detection.

Reads the normalized working image produced by Phase 1 and reports
rectangular *candidates* at multiple scales: small structures (tables,
furniture), medium structures (beds, fixtures) and large structures (rooms,
balconies). Detection is purely geometric; no semantic label is assigned to
any rectangle and no assumption is made that a rectangle is furniture, a
room or anything else.

Nothing is removed from any image. The overlays are drawn on a copy of the
original image so the original content stays visible underneath, and the
normalized image is only ever read. Candidates are merged neither into walls
nor rooms here: this layer stops at "what looks rectangular", so a later
phase can decide what to suppress.

Detection methods (recorded per rectangle):

- ``contour_polygon``: ink contour approximated to a convex 4-gon
  (``approxPolyDP`` epsilon ladder) and measured as a rotated rectangle.
- ``contour_min_area_rect``: ink contour whose rotated minimum-area
  rectangle it fills well, but which has no clean 4-gon approximation.
- ``region_component``: sealed background component (morphological closing at
  several scales) whose minimum-area rectangle it fills well, which surfaces
  enclosed spaces such as rooms whose outlines carry door gaps.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from floorplanto3d.processing.image import load_image, open_image
from floorplanto3d.processing.normalize import png_bytes

logger = logging.getLogger(__name__)

CvRect = tuple[tuple[float, float], tuple[float, float], float]
Corners = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]

# Ink thresholding: Otsu on the normalized image (Phase 1 guarantees a light
# background with dark ink), inverted so 255 = ink.
INK_THRESHOLD = "otsu"

# Candidate acceptance. Area bounds are fractions of the working image area
# so nothing is tuned to one floor plan; the side floor kills slivers.
MIN_AREA_FRACTION = 0.0002
MAX_AREA_FRACTION = 0.98
MIN_SIDE_FRACTION = 0.004
MIN_SIDE_FLOOR_PX = 4

# A candidate must fill its rotated rectangle at least this well (shape area
# divided by rectangle area). Purely geometric quality, not confidence in
# any semantic interpretation.
MIN_RECTANGULARITY = 0.70

# Polygon approximation ladder (fractions of the contour perimeter), tried
# from fine to coarse until a convex 4-gon appears.
APPROX_EPSILON_SCALES = (0.01, 0.02, 0.04, 0.08)

# Region detection: morphological closing scales (fractions of the shorter
# image side) that seal gaps in outlines before background components are
# measured. Small gaps, jittered strokes and wider openings each need their
# own scale; near-identical findings collapse in the relationship pass.
REGION_CLOSE_SCALES = (0.005, 0.015, 0.035)

# Size categories, as fractions of the image area. Small ≈ tables/furniture,
# medium ≈ beds/fixtures, large ≈ rooms/balconies. Thresholds only drive the
# colour coding and the separate views; nothing is filtered out by category.
SMALL_AREA_FRACTION = 0.01
LARGE_AREA_FRACTION = 0.05

# Relationship detection. "Duplicate" = same structure measured twice (two
# methods, two closing scales, or the inner and outer face of one drawn
# outline). "Nested" = one rectangle genuinely inside another. Both are
# reported only; no candidate is discarded.
DUPLICATE_IOU = 0.85
DUPLICATE_CONTAINMENT = 0.95
DUPLICATE_AREA_RATIO = 0.75
NESTED_CONTAINMENT = 0.90

# Overlay drawing (RGB). Outlines only, never filled, so the floor plan
# underneath stays readable.
CATEGORY_COLORS = {
    "small": (240, 140, 50),
    "medium": (40, 110, 220),
    "large": (215, 50, 60),
}
OVERLAY_TEXT_COLOR = (255, 255, 255)
OVERLAY_LEGEND_BG = (255, 255, 255)
OVERLAY_LEGEND_TEXT = (32, 32, 32)
LINE_THICKNESS_FRACTION = 0.002
LINE_THICKNESS_MIN_PX = 2
FONT_SCALE_FRACTION = 0.006
FONT_SCALE_MIN = 0.5
LABEL_GAP_PX = 4

# Saved artifact filenames.
RECTANGLES_OVERLAY_FILENAME = "rectangles_overlay.png"
SMALL_RECTANGLES_FILENAME = "small_rectangles.png"
MEDIUM_RECTANGLES_FILENAME = "medium_rectangles.png"
LARGE_RECTANGLES_FILENAME = "large_rectangles.png"
RECTANGLES_FILENAME = "rectangles.json"
RECTANGLE_REPORT_FILENAME = "rectangle_report.json"

METHOD_PRIORITY = ("contour_polygon", "contour_min_area_rect", "region_component")


@dataclass
class DetectedRectangle:
    """One rectangular candidate, measured in original-image pixel space.

    ``width`` is always the longer side and ``angle_deg`` its direction,
    folded to ``(-90, 90]`` degrees (0 = axis-aligned horizontal, 90 =
    axis-aligned vertical), positive clockwise in image coordinates.
    """

    corners: Corners
    bbox: tuple[int, int, int, int]
    width: float
    height: float
    area: float
    aspect_ratio: float
    angle_deg: float
    axis_aligned: bool
    method: str
    quality_score: float
    category: str
    cv_rect: CvRect = field(repr=False)
    id: str = ""
    duplicate_of: str | None = None
    nested_in: str | None = None


@dataclass
class RectangleDetectionResult:
    """Outputs of Phase 2 (rectangle detection)."""

    rectangles: list[DetectedRectangle]
    overlay: Image.Image
    small_view: Image.Image
    medium_view: Image.Image
    large_view: Image.Image
    metadata: dict[str, Any]
    report: dict[str, Any]


@dataclass
class _Candidate:
    """Internal detection output before measurement and labelling."""

    points: np.ndarray
    cv_rect: CvRect
    method: str
    quality: float
    source_area: float


@dataclass
class _Measured:
    """Rotated-rectangle measurement of a candidate."""

    corners: np.ndarray
    cv_rect: CvRect
    width: float
    height: float
    angle_deg: float
    axis_aligned: bool


def detect_rectangles(
    normalized: str | Path | bytes | bytearray | Image.Image,
    original: str | Path | bytes | bytearray | Image.Image,
) -> RectangleDetectionResult:
    """Run Phase 2 on a floor plan.

    Args:
        normalized: The normalized working image produced by Phase 1. It is
            only read; detection never modifies it.
        original: The original floor plan image, used purely as the visual
            reference the overlays are drawn on. Its pixels are never
            modified either.

    Returns:
        A :class:`RectangleDetectionResult` with every candidate (unique
        IDs), the combined overlay, the three per-category views, the
        ``rectangles.json`` document and the ``rectangle_report.json``
        summary.
    """
    normalized_image = open_image(normalized)
    original_image = open_image(original)
    gray = _gray_array(normalized_image)
    base_rgb, original_width, original_height, _ = load_image(original_image)

    mask, otsu_value = _ink_mask(gray)
    image_area = float(gray.shape[0] * gray.shape[1])
    min_area = MIN_AREA_FRACTION * image_area
    max_area = MAX_AREA_FRACTION * image_area
    min_side = max(
        MIN_SIDE_FLOOR_PX, round(MIN_SIDE_FRACTION * min(gray.shape[0], gray.shape[1]))
    )

    candidates = _contour_candidates(mask, min_area, max_area, min_side)
    candidates += _region_candidates(mask, min_area, max_area, min_side)

    scale_x = original_width / normalized_image.width
    scale_y = original_height / normalized_image.height
    rectangles = _finalize(candidates, scale_x, scale_y, original_width * original_height)
    duplicates, nested = _classify_relationships(rectangles)

    representatives = [rect for rect in rectangles if rect.duplicate_of is None]
    overlay = _draw_view(
        base_rgb,
        representatives,
        title="Phase 2 · rectangle candidates",
        counts=_count_categories(representatives),
    )
    small_view = _draw_view(
        base_rgb, representatives, title="small candidates", only="small"
    )
    medium_view = _draw_view(
        base_rgb, representatives, title="medium candidates", only="medium"
    )
    large_view = _draw_view(
        base_rgb, representatives, title="large candidates", only="large"
    )

    metadata = _build_metadata(rectangles, original_image, normalized_image)
    report = _build_report(
        rectangles,
        duplicates,
        nested,
        original_image,
        normalized_image,
        {
            "ink_threshold": INK_THRESHOLD,
            "otsu_threshold": round(float(otsu_value), 2),
            "min_area_px": round(min_area, 1),
            "max_area_px": round(max_area, 1),
            "min_side_px": min_side,
            "min_rectangularity": MIN_RECTANGULARITY,
            "approx_epsilon_scales": list(APPROX_EPSILON_SCALES),
            "region_close_scales": list(REGION_CLOSE_SCALES),
            "small_area_fraction_max": SMALL_AREA_FRACTION,
            "large_area_fraction_min": LARGE_AREA_FRACTION,
            "duplicate_iou": DUPLICATE_IOU,
            "duplicate_containment": DUPLICATE_CONTAINMENT,
            "duplicate_area_ratio": DUPLICATE_AREA_RATIO,
            "nested_containment": NESTED_CONTAINMENT,
            "methods": list(METHOD_PRIORITY),
            "coordinate_space": "original",
            "scale_from_normalized": round(scale_x, 6),
        },
    )

    logger.info(
        "rectangle detection complete: %d candidates (%d drawn), %d duplicates, "
        "%d nested",
        len(rectangles),
        len(representatives),
        len(duplicates),
        len(nested),
    )
    return RectangleDetectionResult(
        rectangles=rectangles,
        overlay=overlay,
        small_view=small_view,
        medium_view=medium_view,
        large_view=large_view,
        metadata=metadata,
        report=report,
    )


def save_artifacts(
    result: RectangleDetectionResult, directory: str | Path
) -> dict[str, Path]:
    """Write the Phase 2 artifacts into ``directory``.

    Writes ``rectangles_overlay.png``, ``small_rectangles.png``,
    ``medium_rectangles.png``, ``large_rectangles.png``, ``rectangles.json``
    and ``rectangle_report.json``, and returns the output paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "overlay": directory / RECTANGLES_OVERLAY_FILENAME,
        "small": directory / SMALL_RECTANGLES_FILENAME,
        "medium": directory / MEDIUM_RECTANGLES_FILENAME,
        "large": directory / LARGE_RECTANGLES_FILENAME,
        "rectangles": directory / RECTANGLES_FILENAME,
        "report": directory / RECTANGLE_REPORT_FILENAME,
    }
    outputs["overlay"].write_bytes(png_bytes(result.overlay))
    outputs["small"].write_bytes(png_bytes(result.small_view))
    outputs["medium"].write_bytes(png_bytes(result.medium_view))
    outputs["large"].write_bytes(png_bytes(result.large_view))
    outputs["rectangles"].write_text(
        json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8"
    )
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def _gray_array(image: Image.Image) -> np.ndarray:
    """8-bit luma of an image via the project loader."""
    rgb, _, _, _ = load_image(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _ink_mask(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Binary ink mask (255 = ink) and the Otsu threshold used."""
    value, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask, float(value)


def _contour_candidates(
    mask: np.ndarray, min_area: float, max_area: float, min_side: float
) -> list[_Candidate]:
    """Rectangle candidates from the ink contours themselves."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        points = contour.reshape(-1, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        if min(rect[1]) < min_side or rect[1][0] * rect[1][1] <= 0:
            continue

        quad = _approximate_quad(contour, cv2.arcLength(contour, True))
        if quad is not None:
            quad_area = abs(float(cv2.contourArea(quad)))
            quality = min(area, quad_area) / quad_area if quad_area > 0 else 0.0
            if quality >= MIN_RECTANGULARITY:
                candidates.append(
                    _Candidate(
                        points=quad,
                        cv_rect=cv2.minAreaRect(quad),
                        method="contour_polygon",
                        quality=quality,
                        source_area=area,
                    )
                )
                continue

        quality = area / (rect[1][0] * rect[1][1])
        if quality >= MIN_RECTANGULARITY:
            candidates.append(
                _Candidate(
                    points=points,
                    cv_rect=rect,
                    method="contour_min_area_rect",
                    quality=min(quality, 1.0),
                    source_area=area,
                )
            )
    return candidates


def _approximate_quad(contour: np.ndarray, perimeter: float) -> np.ndarray | None:
    """A convex 4-gon approximation of a contour, or ``None``.

    The epsilon ladder runs fine to coarse; once the approximation drops
    below four vertices no coarser epsilon can produce a quadrilateral.
    """
    for scale in APPROX_EPSILON_SCALES:
        approx = cv2.approxPolyDP(contour, scale * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            return None
    return None


def _region_candidates(
    mask: np.ndarray, min_area: float, max_area: float, min_side: float
) -> list[_Candidate]:
    """Rectangle candidates from sealed background components.

    Morphological closing on the ink mask seals gaps in drawn outlines; every
    enclosed background component is then measured as a possible rectangle.
    Components touching the image border are the outside of the drawing and
    are skipped.
    """
    height, width = mask.shape
    candidates: list[_Candidate] = []
    for scale in REGION_CLOSE_SCALES:
        kernel_size = _odd_kernel(scale * min(height, width))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            255 - closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if x <= 0 or y <= 0 or x + box_width >= width or y + box_height >= height:
                continue
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            points = contour.reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(points)
            if min(rect[1]) < min_side or rect[1][0] * rect[1][1] <= 0:
                continue
            quality = area / (rect[1][0] * rect[1][1])
            if quality >= MIN_RECTANGULARITY:
                candidates.append(
                    _Candidate(
                        points=points,
                        cv_rect=rect,
                        method="region_component",
                        quality=min(quality, 1.0),
                        source_area=area,
                    )
                )
    return candidates


def _odd_kernel(size: float) -> int:
    """An odd kernel side of at least 3 px."""
    value = max(3, round(size))
    return value if value % 2 == 1 else value + 1


def _measure(points: np.ndarray) -> _Measured:
    """Rotated minimum-area rectangle of a point set, canonically folded."""
    rect = cv2.minAreaRect(points.astype(np.float32))
    corners = cv2.boxPoints(rect).astype(np.float64)

    lengths = [
        math.hypot(
            float(corners[(index + 1) % 4][0] - corners[index][0]),
            float(corners[(index + 1) % 4][1] - corners[index][1]),
        )
        for index in range(4)
    ]
    long_edge = max(range(4), key=lambda index: lengths[index])
    width = lengths[long_edge]
    height = lengths[(long_edge + 1) % 4]

    edge = corners[(long_edge + 1) % 4] - corners[long_edge]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0]))) % 180.0
    if angle > 90.0:
        angle -= 180.0
    axis_aligned = abs(angle) < 0.5 or abs(abs(angle) - 90.0) < 0.5

    return _Measured(
        corners=corners,
        cv_rect=rect,
        width=width,
        height=height,
        angle_deg=angle,
        axis_aligned=axis_aligned,
    )


def _finalize(
    candidates: list[_Candidate], scale_x: float, scale_y: float, image_area: float
) -> list[DetectedRectangle]:
    """Measure every candidate, move it to original-image space and ID it."""
    rectangles: list[DetectedRectangle] = []
    for candidate in candidates:
        measured = _measure(candidate.points)
        corners = tuple(
            (float(x) * scale_x, float(y) * scale_y) for x, y in measured.corners
        )
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        width = measured.width * scale_x
        height = measured.height * scale_x
        area = width * height
        (center, size, angle) = measured.cv_rect
        rectangles.append(
            DetectedRectangle(
                corners=corners,  # type: ignore[arg-type]
                bbox=(
                    round(min(xs)),
                    round(min(ys)),
                    round(max(xs) - min(xs)),
                    round(max(ys) - min(ys)),
                ),
                width=width,
                height=height,
                area=area,
                aspect_ratio=width / max(height, 1e-9),
                angle_deg=measured.angle_deg,
                axis_aligned=measured.axis_aligned,
                method=candidate.method,
                quality_score=round(candidate.quality, 4),
                category=_category(area, image_area),
                cv_rect=(
                    (center[0] * scale_x, center[1] * scale_y),
                    (size[0] * scale_x, size[1] * scale_x),
                    angle,
                ),
            )
        )

    rectangles.sort(key=lambda r: (-r.area, r.bbox[1], r.bbox[0], r.method))
    for index, rectangle in enumerate(rectangles):
        rectangle.id = f"rect_{index + 1:03d}"
    return rectangles


def _category(area: float, image_area: float) -> str:
    """Size category of a rectangle as a fraction of the image area."""
    fraction = area / max(image_area, 1e-9)
    if fraction < SMALL_AREA_FRACTION:
        return "small"
    if fraction < LARGE_AREA_FRACTION:
        return "medium"
    return "large"


def _classify_relationships(
    rectangles: list[DetectedRectangle],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flag duplicate and nested rectangles and describe them for the report."""
    duplicates: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []

    preference = sorted(
        rectangles,
        key=lambda r: (
            -r.quality_score,
            -r.area,
            METHOD_PRIORITY.index(r.method),
        ),
    )
    for index, rect in enumerate(preference):
        if rect.duplicate_of is not None:
            continue
        for other in preference[index + 1 :]:
            if other.duplicate_of is not None:
                continue
            if _is_duplicate(rect, other):
                other.duplicate_of = rect.id
                duplicates.append(
                    {
                        "representative": rect.id,
                        "duplicate": other.id,
                        "method": other.method,
                        "iou": round(_iou(rect, other), 4),
                        "area_ratio": round(
                            min(rect.area, other.area) / max(rect.area, other.area), 4
                        ),
                    }
                )

    representatives = [rect for rect in rectangles if rect.duplicate_of is None]
    parents: dict[str, DetectedRectangle] = {}
    for outer in representatives:
        for inner in representatives:
            if inner is outer or outer.area <= inner.area:
                continue
            if _is_duplicate(outer, inner):
                continue
            containment = _containment(outer, inner)
            if containment < NESTED_CONTAINMENT:
                continue
            parent = parents.get(inner.id)
            if parent is None or outer.area < parent.area:
                parents[inner.id] = outer
            nested.append(
                {
                    "outer": outer.id,
                    "inner": inner.id,
                    "containment": round(containment, 4),
                }
            )
    for rect in rectangles:
        parent = parents.get(rect.id)
        if parent is not None:
            rect.nested_in = parent.id
    return duplicates, nested


def _is_duplicate(a: DetectedRectangle, b: DetectedRectangle) -> bool:
    """True when both rectangles describe the same structure."""
    if _iou(a, b) >= DUPLICATE_IOU:
        return True
    ratio = min(a.area, b.area) / max(a.area, b.area)
    return _containment(a, b) >= DUPLICATE_CONTAINMENT and ratio >= DUPLICATE_AREA_RATIO


def _iou(a: DetectedRectangle, b: DetectedRectangle) -> float:
    """Intersection over union of two rotated rectangles."""
    intersection = _intersection_area(a, b)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def _containment(a: DetectedRectangle, b: DetectedRectangle) -> float:
    """Fraction of the smaller rectangle covered by the larger one."""
    intersection = _intersection_area(a, b)
    smaller = min(a.area, b.area)
    return intersection / smaller if smaller > 0 else 0.0


def _intersection_area(a: DetectedRectangle, b: DetectedRectangle) -> float:
    """Exact area shared by two rotated rectangles, with an AABB pre-filter."""
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    if ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay:
        return 0.0
    code, region = cv2.rotatedRectangleIntersection(a.cv_rect, b.cv_rect)
    if code == cv2.INTERSECT_NONE or region is None or len(region) < 3:
        return 0.0
    return abs(float(cv2.contourArea(region)))


def _record(rectangle: DetectedRectangle) -> dict[str, Any]:
    """The ``rectangles.json`` entry for one candidate."""
    record: dict[str, Any] = {
        "id": rectangle.id,
        "category": rectangle.category,
        "bbox": {
            "x": rectangle.bbox[0],
            "y": rectangle.bbox[1],
            "width": rectangle.bbox[2],
            "height": rectangle.bbox[3],
        },
        "corners": [{"x": round(x, 2), "y": round(y, 2)} for x, y in rectangle.corners],
        "width": round(rectangle.width, 2),
        "height": round(rectangle.height, 2),
        "area": round(rectangle.area, 2),
        "aspect_ratio": round(rectangle.aspect_ratio, 4),
        "angle_deg": round(rectangle.angle_deg, 3),
        "axis_aligned": rectangle.axis_aligned,
        "method": rectangle.method,
        "quality_score": rectangle.quality_score,
    }
    if rectangle.duplicate_of is not None:
        record["duplicate_of"] = rectangle.duplicate_of
    if rectangle.nested_in is not None:
        record["nested_in"] = rectangle.nested_in
    return record


def _build_metadata(
    rectangles: list[DetectedRectangle],
    original: Image.Image,
    normalized: Image.Image,
) -> dict[str, Any]:
    """The ``rectangles.json`` document."""
    return {
        "phase": {"number": 2, "name": "rectangle_detection"},
        "image": {
            "width": original.width,
            "height": original.height,
            "normalized_width": normalized.width,
            "normalized_height": normalized.height,
            "coordinate_space": "original",
        },
        "rectangles": [_record(rectangle) for rectangle in rectangles],
    }


def _build_report(
    rectangles: list[DetectedRectangle],
    duplicates: list[dict[str, Any]],
    nested: list[dict[str, Any]],
    original: Image.Image,
    normalized: Image.Image,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """The ``rectangle_report.json`` document."""
    by_method: dict[str, int] = {}
    for rectangle in rectangles:
        by_method[rectangle.method] = by_method.get(rectangle.method, 0) + 1
    representatives = [rect for rect in rectangles if rect.duplicate_of is None]
    return {
        "phase": {"number": 2, "name": "rectangle_detection"},
        "summary": {
            "total_rectangles": len(rectangles),
            "drawn_rectangles": len(representatives),
            "by_category": _count_categories(rectangles),
            "by_method": by_method,
            "duplicate_count": len(duplicates),
            "nested_count": len(nested),
        },
        "duplicates": duplicates,
        "nested": nested,
        "input": {
            "original_width": original.width,
            "original_height": original.height,
            "normalized_width": normalized.width,
            "normalized_height": normalized.height,
        },
        "parameters": parameters,
    }


def _count_categories(rectangles: list[DetectedRectangle]) -> dict[str, int]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for rectangle in rectangles:
        counts[rectangle.category] += 1
    return counts


def _draw_view(
    base_rgb: np.ndarray,
    rectangles: list[DetectedRectangle],
    *,
    title: str,
    counts: dict[str, int] | None = None,
    only: str | None = None,
) -> Image.Image:
    """Outlines and IDs drawn on a copy of the original image."""
    canvas = base_rgb.copy()
    height, width = canvas.shape[:2]
    thickness = max(LINE_THICKNESS_MIN_PX, round(LINE_THICKNESS_FRACTION * min(height, width)))
    font_scale = max(FONT_SCALE_MIN, FONT_SCALE_FRACTION * min(height, width))

    selected = [rect for rect in rectangles if only is None or rect.category == only]
    for rectangle in selected:
        color = CATEGORY_COLORS[rectangle.category]
        points = np.array(rectangle.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
        _draw_label(canvas, rectangle, color, font_scale, max(1, thickness // 2))

    if counts is None:
        counts = _count_categories(selected)
    rows = [f"{title}  ·  {len(selected)} drawn"]
    rows += [
        f"{name}: {counts[name]}"
        for name in ("small", "medium", "large")
        if only is None or counts[name] > 0
    ]
    _draw_legend(canvas, rows, font_scale, thickness)
    return Image.fromarray(canvas)


def _draw_label(
    canvas: np.ndarray,
    rectangle: DetectedRectangle,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """ID chip at the topmost corner of a rectangle, clamped to the image."""
    height, width = canvas.shape[:2]
    text = rectangle.id
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    top = min(rectangle.corners, key=lambda point: point[1])
    x0 = int(top[0]) - text_width // 2
    y0 = int(top[1]) - text_height - baseline - 2 * LABEL_GAP_PX
    if y0 < 0:
        y0 = int(top[1]) + 2 * LABEL_GAP_PX
    x0 = min(max(x0, 0), max(0, width - text_width - 2 * LABEL_GAP_PX))
    y0 = min(max(y0, 0), max(0, height - text_height - baseline - 2 * LABEL_GAP_PX))

    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + text_width + 2 * LABEL_GAP_PX, y0 + text_height + baseline + 2 * LABEL_GAP_PX),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        (x0 + LABEL_GAP_PX, y0 + text_height + LABEL_GAP_PX),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        OVERLAY_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def _draw_legend(
    canvas: np.ndarray,
    rows: list[str],
    font_scale: float,
    thickness: int,
) -> None:
    """Colour legend and per-category counts in the top-left corner."""
    sizes = [
        cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for row in rows
    ]
    row_height = max(size[1] for size in sizes) + 2 * LABEL_GAP_PX
    chip_width = max(size[0] for size in sizes) + row_height + 4 * LABEL_GAP_PX
    chip_height = row_height * len(rows) + 2 * LABEL_GAP_PX

    cv2.rectangle(
        canvas,
        (LABEL_GAP_PX, LABEL_GAP_PX),
        (LABEL_GAP_PX + chip_width, LABEL_GAP_PX + chip_height),
        OVERLAY_LEGEND_BG,
        -1,
    )
    for index, row in enumerate(rows):
        y = LABEL_GAP_PX + row_height * (index + 1) - LABEL_GAP_PX
        text_x = 3 * LABEL_GAP_PX
        if index > 0:
            name = row.split(":")[0]
            cv2.rectangle(
                canvas,
                (text_x, y - row_height // 2),
                (text_x + row_height // 2, y + row_height // 4),
                CATEGORY_COLORS[name],
                -1,
            )
            text_x += row_height
        cv2.putText(
            canvas,
            row,
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            OVERLAY_LEGEND_TEXT,
            thickness,
            cv2.LINE_AA,
        )
