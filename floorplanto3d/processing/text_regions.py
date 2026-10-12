"""Phase 2, Step C2: proposed text and dimension-annotation regions.

A floor plan mixes architectural strokes with *drawn text*: room labels,
dimension strings (``11'0"X11'0"``), badges and rotated corridor notes.
This step proposes regions that look like text and annotations so later
steps can consider them, without touching a single source pixel and
without deciding anything about wall candidates.

Everything produced here is a **proposal**. There is no OCR in the
environment and no labelled ground truth, so a region is "text-like" only
in the geometric sense: small repeated marks with consistent size, dense
short strokes and regular spacing along a line. Furniture outlines and
architectural boundaries are rejected where geometry allows — large
boxes, thick strokes and long uninterrupted runs do not look like glyph
groups — but a proposal can still be wrong and is labelled as such.

The pipeline is:

1. Mark extraction. The normalized image is separated into marks of both
   polarities: dark marks on light ground (the Step A adaptive method,
   reused) and light marks on dark ground (the light pixels enclosed by
   dark marks — the holes of the dark mask). Room labels here are light
   grey on white and white on dark room fills, so one polarity alone
   would miss half the text; hole extraction keeps white glyphs from
   being swallowed by the bright background halo.
2. Connected components. Each mark gets dimensions, fill ratio and an
   estimated stroke width (median distance transform), plus a stroke
   count across the component: glyphs are several short strokes in a
   small box, while a furniture rail or wall is one or two long ones.
3. Spatial grouping. Glyph-like components are grouped into horizontal or
   vertical lines (rotated labels are lines too), small fragments such as
   quote marks are absorbed into their line, and each line becomes one
   proposed region with a quality score written out as evidence.

The mask, the overlay and the documents all keep the original image's
pixel grid (Phase 1 preserves the aspect ratio and here the scale is 1:1);
regions are reported in original-image pixel space. No pixel is erased,
inpaint or altered, no wall candidate is rejected or reclassified, and no
output claims a region *is* text — only that it looks like it.
"""

from __future__ import annotations

import argparse
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
from floorplanto3d.processing.normalize import normalize_image, png_bytes
from floorplanto3d.processing.wall_detection import ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C

logger = logging.getLogger(__name__)

# Mark extraction. Dark marks reuse the Step A adaptive threshold
# (Gaussian local mean minus C), which copes with the uneven backgrounds
# of rendered plans; light marks are the holes those dark marks enclose.

# Component geometry. Text on this class of plan is rendered at a wide
# range of sizes (rotated corridor notes, large room labels, small
# dimension strings), so the accepted boxes are generous; thickness and
# stroke count do the discriminating instead.
MIN_COMPONENT_AREA_PX = 8
MIN_COMPONENT_SIDE_PX = 3
GLYPH_MAX_HEIGHT_PX = 90.0
GLYPH_MAX_WIDTH_PX = 420.0
GLYPH_MAX_ASPECT = 22.0
GLYPH_MAX_STROKE_WIDTH_PX = 6.5
GLYPH_MIN_FILL_RATIO = 0.05

# Stroke density. A glyph group carries several short strokes across its
# line; long straight outlines carry one or two. Long thin components
# (rails, wall lines, dimension rules) must show enough stroke runs to be
# text-like, while compact merged word blobs are judged by grouping.
TEXT_STROKE_WIDTH_RANGE_PX = (1.2, 4.5)
MIN_STROKE_RUNS_ELONGATED = 3
ELONGATED_MIN_EXTENT_PX = 100.0
ELONGATED_MAX_CROSS_PX = 20.0
ELONGATED_MIN_ASPECT = 4.0

# Spatial grouping. Components merge into a line when they agree in size,
# sit on a shared axis and are within a plausible letter gap. The group
# must stay coherent (perpendicular spread) so scattered texture cannot
# chain into one giant region. Tiny fragments (quote marks, dots) join an
# existing line when they fall close to it.
GROUP_MAX_HEIGHT_RATIO = 2.4
GROUP_AXIS_TOLERANCE = 0.55
GROUP_MAX_GAP_PX = 20.0
GROUP_MAX_GAP_CROSS_FRACTION = 0.75
GROUP_COHERENCE_SPREAD_FRACTION = 0.55
ABSORB_MAX_HEIGHT_FRACTION = 0.55
ABSORB_PADDING_FRACTION = 0.35

# Region acceptance. A grouped line is only proposed when it carries real
# text evidence: a plausible type size, a shared axis, compatible heights
# and enough strokes. Everything below the gate is left out entirely —
# that is filtering proposals, never filtering wall candidates.
MIN_REGION_GLYPH_HEIGHT_PX = 6.0
MIN_REGION_ALIGNMENT = 0.5
MIN_REGION_HEIGHT_CONSISTENCY = 0.35
MIN_REGION_SINGLE_STROKE_RUNS = 3
MIN_REGION_CROSS_RUNS = 2

# Region roles (proposed only). Large glyphs read as room labels; strings
# with narrow marks and superscript fragments read as dimension strings.
ROLE_ROOM_LABEL_MIN_HEIGHT_PX = 22.0
ROLE_DIMENSION_NARROW_ASPECT = 1.6
ROLE_DIMENSION_NARROW_FRACTION = 0.3
ROLE_DIMENSION_QUOTE_HEIGHT_FRACTION = 0.55

# Region quality score (proposal quality, never a correctness claim).
QUALITY_WEIGHTS = {
    "glyph_evidence": 0.3,
    "height_consistency": 0.2,
    "alignment": 0.2,
    "stroke_density": 0.2,
    "spacing_regularity": 0.1,
}

# Overlay drawing (RGB). Highlight is blended over a copy of the source;
# the source image itself is never written to.
HIGHLIGHT_COLOR = (236, 76, 120)
HIGHLIGHT_ALPHA = 0.22
OVERLAY_BOX_THICKNESS = 2
OVERLAY_FONT_SCALE = 0.45
OVERLAY_FONT_THICKNESS = 1
OVERLAY_TEXT_COLOR = (255, 255, 255)
OVERLAY_LABEL_BG = (236, 76, 120)
OVERLAY_BANNER_BG = (255, 255, 255)
OVERLAY_BANNER_TEXT = (32, 32, 32)
OVERLAY_LABEL_GAP_PX = 4

# Saved artifact filenames.
TEXT_REGIONS_MASK_FILENAME = "text_regions_mask.png"
TEXT_REGIONS_OVERLAY_FILENAME = "text_regions_overlay.png"
TEXT_REGIONS_FILENAME = "text_regions.json"
TEXT_DETECTION_REPORT_FILENAME = "text_detection_report.json"


@dataclass
class TextRegion:
    """One proposed text/annotation region, in original-image pixel space."""

    id: str
    role: str
    detection_method: str
    polarity: str
    bbox: tuple[int, int, int, int]
    component_count: int
    ink_pixels: int
    fill_ratio: float
    glyph_height: float
    stroke_width: float
    stroke_runs: int
    height_consistency: float
    alignment: float
    spacing_regularity: float
    quality: float
    quality_evidence: dict[str, float]
    evidence: list[str]
    uncertainty: list[str]
    component_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class TextRegionDetectionResult:
    """Outputs of Phase 2, Step C2."""

    regions: list[TextRegion]
    mask: Image.Image
    overlay: Image.Image
    document: dict[str, Any]
    report: dict[str, Any]


@dataclass
class _Component:
    """One connected mark in the normalized working space."""

    bbox: tuple[int, int, int, int]
    area: int
    fill_ratio: float
    stroke_width: float
    stroke_runs: int
    cross_runs: int
    polarity: str


@dataclass
class _Line:
    """A spatial group of components that behaves like one line of text."""

    components: list[_Component]
    axis: str  # "h" for horizontal lines, "v" for vertical (rotated) lines


def detect_text_regions(
    normalized: str | Path | bytes | bytearray | Image.Image,
    original: str | Path | bytes | bytearray | Image.Image,
) -> TextRegionDetectionResult:
    """Run Step C2 on one floor plan.

    Args:
        normalized: The Phase 1 normalized working image. Only read.
        original: The original floor plan image. Defines the reported
            coordinate space and the visual base of the overlay; its
            pixels are never modified.

    Returns:
        A :class:`TextRegionDetectionResult` with every proposed region,
        the proposed binary mask, the overlay and the two JSON documents.
        Nothing here is confirmed text, and no pixel of any input is
        changed.
    """
    original_image = open_image(original)
    base_rgb, original_width, original_height, _ = load_image(original_image)
    normalized_image = open_image(normalized)
    gray = cv2.cvtColor(np.asarray(normalized_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    mask_height, mask_width = gray.shape[:2]

    dark_marks = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )
    light_marks = _enclosed_marks(dark_marks)

    components: list[_Component] = []
    for polarity, marks in (("dark_on_light", dark_marks), ("light_on_dark", light_marks)):
        components.extend(_mark_components(marks, polarity))

    regions: list[TextRegion] = []
    for polarity in ("dark_on_light", "light_on_dark"):
        subset = [c for c in components if c.polarity == polarity]
        for line in _group_into_lines(subset):
            region = _measure_line(line)
            if region is not None:
                regions.append(region)

    regions = _dedupe_identical(regions)
    regions.sort(key=lambda r: (round((r.bbox[1] + r.bbox[3] / 2.0), 2), r.bbox[0], r.bbox[2]))
    for index, region in enumerate(regions, start=1):
        region.id = f"text_{index:03d}"

    scale_x = original_width / max(1, mask_width)
    scale_y = original_height / max(1, mask_height)
    regions = [_to_original_space(region, scale_x, scale_y) for region in regions]

    mask = _build_mask(regions, original_width, original_height)
    overlay = _draw_overlay(base_rgb, regions)
    document = _build_document(regions, original_image, normalized_image, mask)
    report = _build_report(regions, original_image, normalized_image, components, scale_x, scale_y)

    logger.info(
        "text-region detection (Step C2) complete: %d proposed regions "
        "(%d room-label-like, %d dimension-like)",
        report["summary"]["total_regions"],
        report["summary"]["by_role"].get("room_label_like", 0),
        report["summary"]["by_role"].get("dimension_like", 0),
    )
    return TextRegionDetectionResult(
        regions=regions,
        mask=mask,
        overlay=overlay,
        document=document,
        report=report,
    )


def save_artifacts(
    result: TextRegionDetectionResult, directory: str | Path
) -> dict[str, Path]:
    """Write the Step C2 artifacts into ``directory``.

    Writes ``text_regions_mask.png``, ``text_regions_overlay.png``,
    ``text_regions.json`` and ``text_detection_report.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "mask": directory / TEXT_REGIONS_MASK_FILENAME,
        "overlay": directory / TEXT_REGIONS_OVERLAY_FILENAME,
        "regions": directory / TEXT_REGIONS_FILENAME,
        "report": directory / TEXT_DETECTION_REPORT_FILENAME,
    }
    outputs["mask"].write_bytes(png_bytes(result.mask))
    outputs["overlay"].write_bytes(png_bytes(result.overlay))
    outputs["regions"].write_text(
        json.dumps(result.document, indent=2) + "\n", encoding="utf-8"
    )
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


# --- component extraction ---------------------------------------------


def _enclosed_marks(dark_marks: np.ndarray) -> np.ndarray:
    """Light marks enclosed by dark marks (the holes of the dark mask).

    White text on a dark room fill sits inside the fill's dark pixels, so
    it appears as a hole of the dark mask. Working from the holes keeps
    such glyphs whole: the bright background halo that surrounds every
    dark stroke is border-connected and therefore excluded, where a plain
    inverted threshold would swallow the glyphs into it.
    """
    light = np.where(dark_marks > 0, 0, 255).astype(np.uint8)
    flooded = light.copy()
    flood_mask = np.zeros((light.shape[0] + 2, light.shape[1] + 2), dtype=np.uint8)
    seed = _border_background_seed(light)
    cv2.floodFill(flooded, flood_mask, seed, 128)
    return np.where(flooded == 255, 255, 0).astype(np.uint8)


def _border_background_seed(light: np.ndarray) -> tuple[int, int]:
    """A background pixel on the image border to start the flood from."""
    height, width = light.shape[:2]
    for x in range(width):
        if light[0, x] > 0:
            return (x, 0)
        if light[height - 1, x] > 0:
            return (x, height - 1)
    for y in range(height):
        if light[y, 0] > 0:
            return (0, y)
        if light[y, width - 1] > 0:
            return (width - 1, y)
    return (0, 0)


def _mark_components(marks: np.ndarray, polarity: str) -> list[_Component]:
    """Connected components of one polarity's mark mask."""
    binary = (marks > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    components: list[_Component] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        if min(w, h) < MIN_COMPONENT_SIDE_PX and area < 20:
            continue
        if h > GLYPH_MAX_HEIGHT_PX or w > GLYPH_MAX_WIDTH_PX:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > GLYPH_MAX_ASPECT:
            continue
        fill_ratio = area / max(1, w * h)
        if fill_ratio < GLYPH_MIN_FILL_RATIO:
            continue
        component_mask = labels[y : y + h, x : x + w] == index
        stroke_width = 2.0 * float(np.median(distance[y : y + h, x : x + w][component_mask]))
        if stroke_width > GLYPH_MAX_STROKE_WIDTH_PX:
            continue
        runs_h = _stroke_runs(component_mask, "h")
        runs_v = _stroke_runs(component_mask, "v")
        stroke_runs = max(runs_h, runs_v)
        cross_runs = min(runs_h, runs_v)
        extent, cross = max(w, h), min(w, h)
        if (
            extent >= ELONGATED_MIN_EXTENT_PX
            and cross <= ELONGATED_MAX_CROSS_PX
            and aspect >= ELONGATED_MIN_ASPECT
            and stroke_runs < MIN_STROKE_RUNS_ELONGATED
        ):
            continue
        components.append(
            _Component(
                bbox=(x, y, w, h),
                area=area,
                fill_ratio=round(fill_ratio, 4),
                stroke_width=round(stroke_width, 2),
                stroke_runs=stroke_runs,
                cross_runs=cross_runs,
                polarity=polarity,
            )
        )
    return components


def _stroke_runs(component_mask: np.ndarray, axis: str) -> int:
    """Max ink runs crossing the component (its stroke density).

    The busiest scanline is used: a merged word blob has dense bands where
    a scanline crosses many glyph strokes, while a rail or rule shows one
    or two runs on every scanline.
    """
    height, width = component_mask.shape[:2]
    if axis == "h":
        scans = [round(height * f) for f in (0.15, 0.325, 0.5, 0.675, 0.85)]
    else:
        scans = [round(width * f) for f in (0.15, 0.325, 0.5, 0.675, 0.85)]
    counts: list[int] = []
    for position in scans:
        if axis == "h":
            line = component_mask[max(0, min(height - 1, position)), :]
        else:
            line = component_mask[:, max(0, min(width - 1, position))]
        runs = 0
        inside = False
        for value in line:
            if value and not inside:
                runs += 1
            inside = bool(value)
        counts.append(runs)
    return int(max(counts)) if counts else 0


# --- spatial grouping --------------------------------------------------


def _group_into_lines(components: list[_Component]) -> list[_Line]:
    """Group glyph-like components into horizontal or vertical lines."""
    open_lines: list[_Line] = []
    used: set[int] = set()
    for index, component in enumerate(components):
        if index in used:
            continue
        used.add(index)
        line = _Line(components=[component], axis="h" if component.bbox[2] >= component.bbox[3] else "v")
        open_lines.append(line)

    changed = True
    while changed:
        changed = False
        for i, line in enumerate(open_lines):
            for j, other in enumerate(open_lines):
                if i == j:
                    continue
                merged = _try_merge_lines(line, other)
                if merged is not None:
                    open_lines[i] = merged
                    del open_lines[j]
                    changed = True
                    break
            if changed:
                break

    final: list[_Line] = []
    for line in open_lines:
        line = _absorb_fragments(line, components)
        final.append(line)
    return final


def _line_box(line: _Line) -> tuple[int, int, int, int]:
    xs = [c.bbox[0] for c in line.components]
    ys = [c.bbox[1] for c in line.components]
    x1 = [c.bbox[0] + c.bbox[2] for c in line.components]
    y1 = [c.bbox[1] + c.bbox[3] for c in line.components]
    return (min(xs), min(ys), max(x1) - min(xs), max(y1) - min(ys))


def _try_merge_lines(a: _Line, b: _Line) -> _Line | None:
    for axis in ("h", "v"):
        if not _boxes_compatible(a, b, axis):
            continue
        merged = _Line(components=a.components + b.components, axis=axis)
        if _line_coherent(merged.components, axis):
            return merged
    return None


def _line_coherent(components: list[_Component], axis: str) -> bool:
    """A group is a line only while its members share one axis band.

    Without this check, pairwise merging chains scattered texture marks
    transitively into one giant incoherent region.
    """
    if axis == "h":
        centers = [c.bbox[1] + c.bbox[3] / 2.0 for c in components]
        cross = [c.bbox[3] for c in components]
    else:
        centers = [c.bbox[0] + c.bbox[2] / 2.0 for c in components]
        cross = [c.bbox[2] for c in components]
    spread = float(np.std(centers))
    return spread <= GROUP_COHERENCE_SPREAD_FRACTION * max(1.0, float(np.median(cross)))


def _boxes_compatible(a: _Line, b: _Line, axis: str) -> bool:
    ax, ay, aw, ah = _line_box(a)
    bx, by, bw, bh = _line_box(b)
    if axis == "h":
        gap = max(ax, bx) - min(ax + aw, bx + bw)
        offset = abs((ay + ah / 2.0) - (by + bh / 2.0))
        size_a, size_b = ah, bh
    else:
        gap = max(ay, by) - min(ay + ah, by + bh)
        offset = abs((ax + aw / 2.0) - (bx + bw / 2.0))
        size_a, size_b = aw, bw
    max_gap = max(GROUP_MAX_GAP_PX, GROUP_MAX_GAP_CROSS_FRACTION * max(size_a, size_b))
    if gap > max_gap:
        return False
    if offset > GROUP_AXIS_TOLERANCE * max(size_a, size_b):
        return False
    ratio = max(size_a, size_b) / max(1e-6, min(size_a, size_b))
    return ratio <= GROUP_MAX_HEIGHT_RATIO


def _absorb_fragments(line: _Line, components: list[_Component]) -> _Line:
    """Pull quote-like fragments and dots into a nearby line."""
    x, y, w, h = _line_box(line)
    pad_x = ABSORB_PADDING_FRACTION * w
    pad_y = ABSORB_PADDING_FRACTION * h
    members = {id(c) for c in line.components}
    for component in components:
        if id(component) in members or component.polarity != line.components[0].polarity:
            continue
        cx, cy, cw, ch = component.bbox
        if ch > ABSORB_MAX_HEIGHT_FRACTION * h and line.axis == "h":
            continue
        if cw > ABSORB_MAX_HEIGHT_FRACTION * w and line.axis == "v":
            continue
        if x - pad_x <= cx and cx + cw <= x + w + pad_x and y - pad_y <= cy and cy + ch <= y + h + pad_y:
            if _line_coherent(line.components + [component], line.axis):
                line.components.append(component)
                members.add(id(component))
    return line


# --- region measurement -----------------------------------------------


def _measure_line(line: _Line) -> TextRegion | None:
    """Measure one grouped line into a proposed text region (or None)."""
    components = sorted(line.components, key=lambda c: (c.bbox[1], c.bbox[0]))
    if not components:
        return None
    x, y, w, h = _line_box(line)
    major = [c for c in components if c.bbox[3] >= ABSORB_MAX_HEIGHT_FRACTION * h] or components
    heights = [float(c.bbox[3]) if line.axis == "h" else float(c.bbox[2]) for c in major]
    glyph_height = float(np.median(heights))
    mean_height = float(np.mean(heights)) if heights else 0.0
    height_consistency = (
        max(0.0, 1.0 - float(np.std(heights)) / max(1e-6, mean_height)) if heights else 0.0
    )

    if line.axis == "h":
        centers = [c.bbox[1] + c.bbox[3] / 2.0 for c in major]
        spread = float(np.std(centers))
    else:
        centers = [c.bbox[0] + c.bbox[2] / 2.0 for c in major]
        spread = float(np.std(centers))
    alignment = max(0.0, 1.0 - spread / max(1e-6, 0.5 * max(glyph_height, 1.0)))

    stroke_runs = max(c.stroke_runs for c in components)
    stroke_width = float(np.median([c.stroke_width for c in major]))
    lo, hi = TEXT_STROKE_WIDTH_RANGE_PX
    if lo <= stroke_width <= hi:
        width_score = 1.0
    else:
        width_score = max(0.0, 1.0 - min(abs(stroke_width - lo), abs(stroke_width - hi)) / max(1e-6, lo))
    stroke_density = min(1.0, stroke_runs / 4.0) * width_score

    if glyph_height < MIN_REGION_GLYPH_HEIGHT_PX:
        return None
    if alignment < MIN_REGION_ALIGNMENT:
        return None
    if height_consistency < MIN_REGION_HEIGHT_CONSISTENCY:
        return None
    if max(c.cross_runs for c in components) < MIN_REGION_CROSS_RUNS:
        return None
    if len(components) == 1 and stroke_runs < MIN_REGION_SINGLE_STROKE_RUNS:
        return None

    gaps = _gaps(components, line.axis)
    if len(gaps) >= 2 and float(np.mean(gaps)) > 0:
        gap_cv = float(np.std(gaps)) / float(np.mean(gaps))
        spacing_regularity = max(0.0, 1.0 - gap_cv)
    else:
        spacing_regularity = 0.5

    glyph_evidence = min(1.0, (len(components) - 1) / 3.0) if len(components) > 1 else 0.0
    glyph_evidence = max(glyph_evidence, min(1.0, stroke_runs / 5.0))

    quality_evidence = {
        "glyph_evidence": round(glyph_evidence, 4),
        "height_consistency": round(height_consistency, 4),
        "alignment": round(alignment, 4),
        "stroke_density": round(stroke_density, 4),
        "spacing_regularity": round(spacing_regularity, 4),
    }
    quality = round(
        sum(QUALITY_WEIGHTS[name] * value for name, value in quality_evidence.items()), 4
    )

    role = _propose_role(components, glyph_height, line.axis)
    evidence, uncertainty = _explain(
        components, line.axis, stroke_runs, stroke_width, height_consistency, alignment
    )
    return TextRegion(
        id="",
        role=role,
        detection_method=f"connected_components[{components[0].polarity}] + stroke_density + spatial_grouping[{line.axis}]",
        polarity=components[0].polarity,
        bbox=(x, y, w, h),
        component_count=len(components),
        ink_pixels=sum(c.area for c in components),
        fill_ratio=round(sum(c.area for c in components) / max(1, w * h), 4),
        glyph_height=round(glyph_height, 2),
        stroke_width=round(stroke_width, 2),
        stroke_runs=stroke_runs,
        height_consistency=round(height_consistency, 4),
        alignment=round(alignment, 4),
        spacing_regularity=round(spacing_regularity, 4),
        quality=quality,
        quality_evidence=quality_evidence,
        evidence=evidence,
        uncertainty=uncertainty,
        component_bboxes=[c.bbox for c in components],
    )


def _gaps(components: list[_Component], axis: str) -> list[float]:
    ordered = sorted(components, key=lambda c: c.bbox[0] if axis == "h" else c.bbox[1])
    gaps: list[float] = []
    for a, b in zip(ordered, ordered[1:]):
        if axis == "h":
            gap = b.bbox[0] - (a.bbox[0] + a.bbox[2])
        else:
            gap = b.bbox[1] - (a.bbox[1] + a.bbox[3])
        gaps.append(float(max(0, gap)))
    return gaps


def _propose_role(components: list[_Component], glyph_height: float, axis: str) -> str:
    """Propose a role for a region: room label, dimension string or text."""
    narrow = 0
    quote_like = 0
    for component in components:
        cw, ch = component.bbox[2], component.bbox[3]
        extent = cw if axis == "h" else ch
        cross = ch if axis == "h" else cw
        if extent / max(1.0, cross) <= 1.0 / ROLE_DIMENSION_NARROW_ASPECT:
            narrow += 1
        if cross <= ROLE_DIMENSION_QUOTE_HEIGHT_FRACTION * glyph_height:
            quote_like += 1
    total = len(components)
    if total >= 2 and (
        narrow / total >= ROLE_DIMENSION_NARROW_FRACTION or quote_like >= 1
    ) and glyph_height < ROLE_ROOM_LABEL_MIN_HEIGHT_PX:
        return "dimension_like"
    if glyph_height >= ROLE_ROOM_LABEL_MIN_HEIGHT_PX:
        return "room_label_like"
    return "text_line"


def _explain(
    components: list[_Component],
    axis: str,
    stroke_runs: int,
    stroke_width: float,
    height_consistency: float,
    alignment: float,
) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    uncertainty: list[str] = []
    if len(components) >= 3:
        evidence.append(
            f"{len(components)} similar small components grouped into one "
            f"{'horizontal' if axis == 'h' else 'vertical (rotated)'} line"
        )
    elif len(components) == 2:
        evidence.append("two small components sit on one line at similar size")
    else:
        uncertainty.append("single-component region: one mark alone is weak text evidence")
    if stroke_runs >= MIN_STROKE_RUNS_ELONGATED:
        evidence.append(
            f"stroke density: up to {stroke_runs} separate ink runs cross the region "
            f"(glyph-like packing, mean stroke width {stroke_width} px)"
        )
    else:
        uncertainty.append(
            f"only {stroke_runs} ink run(s) cross the region; long straight marks "
            "can look like this without being text"
        )
    if height_consistency >= 0.7:
        evidence.append("component heights are consistent (shared type size)")
    else:
        uncertainty.append("component heights vary; grouping may mix marks of different origin")
    if alignment >= 0.7:
        evidence.append("components align on a shared baseline/axis")
    else:
        uncertainty.append("components deviate from a shared axis")
    uncertainty.append(
        "no OCR: letters and digits are inferred from geometry only, so the "
        "role (room label vs dimension) is a guess"
    )
    return evidence, uncertainty


# --- dedupe and mapping ------------------------------------------------


def _dedupe_identical(regions: list[TextRegion]) -> list[TextRegion]:
    """Collapse only byte-identical proposals of the two polarities.

    Overlapping proposals are independent evidence and are all kept: a
    proposal is never thrown away here for looking like another one.
    """
    seen: set[tuple[int, int, int, int]] = set()
    kept: list[TextRegion] = []
    for region in regions:
        if region.bbox in seen:
            continue
        seen.add(region.bbox)
        kept.append(region)
    return kept


def _to_original_space(region: TextRegion, scale_x: float, scale_y: float) -> TextRegion:
    """Re-express a region in original-image pixel space."""
    scale = (scale_x + scale_y) / 2.0

    def map_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = box
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        )

    return TextRegion(
        id=region.id,
        role=region.role,
        detection_method=region.detection_method,
        polarity=region.polarity,
        bbox=map_box(region.bbox),
        component_count=region.component_count,
        ink_pixels=region.ink_pixels,
        fill_ratio=region.fill_ratio,
        glyph_height=round(region.glyph_height * scale, 2),
        stroke_width=round(region.stroke_width * scale, 2),
        stroke_runs=region.stroke_runs,
        height_consistency=region.height_consistency,
        alignment=region.alignment,
        spacing_regularity=region.spacing_regularity,
        quality=region.quality,
        quality_evidence=region.quality_evidence,
        evidence=region.evidence,
        uncertainty=region.uncertainty,
        component_bboxes=[map_box(box) for box in region.component_bboxes],
    )


# --- mask, overlay and documents ---------------------------------------


def _build_mask(
    regions: list[TextRegion], width: int, height: int
) -> Image.Image:
    """Proposed text/annotation mask (255 = proposed region, filled bbox)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        x, y, w, h = region.bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return Image.fromarray(mask)


def _draw_overlay(base_rgb: np.ndarray, regions: list[TextRegion]) -> Image.Image:
    """Transparent highlights and boxes on a copy of the original image."""
    canvas = base_rgb.copy()
    color = np.array(HIGHLIGHT_COLOR, dtype=np.float32)
    for region in regions:
        x, y, w, h = region.bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = (
            (1.0 - HIGHLIGHT_ALPHA) * roi + HIGHLIGHT_ALPHA * color
        ).astype(np.uint8)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), HIGHLIGHT_COLOR, OVERLAY_BOX_THICKNESS)
        label = f"{region.id} {region.role} {region.quality:.2f}"
        _draw_label(canvas, label, (x0 + OVERLAY_LABEL_GAP_PX, y0 - OVERLAY_LABEL_GAP_PX))

    paired_roles = {}
    for region in regions:
        paired_roles[region.role] = paired_roles.get(region.role, 0) + 1
    rows = [
        "Step C2 - PROPOSED text/annotation regions (not verified; no OCR)",
        f"{len(regions)} regions  -  "
        + "  ·  ".join(f"{role}: {count}" for role, count in sorted(paired_roles.items())),
        "highlight = proposed region only; source pixels are preserved (nothing erased or inpainted)",
    ]
    _draw_banner(canvas, rows)
    return Image.fromarray(canvas)


def _draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_FONT_THICKNESS
    )
    x = max(0, min(origin[0], canvas.shape[1] - text_w - 1))
    y = max(text_h + 1, min(origin[1], canvas.shape[0] - baseline - 1))
    cv2.rectangle(canvas, (x - 1, y - text_h - 1), (x + text_w + 1, y + baseline), OVERLAY_LABEL_BG, -1)
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE,
        OVERLAY_TEXT_COLOR, OVERLAY_FONT_THICKNESS, cv2.LINE_AA,
    )


def _draw_banner(canvas: np.ndarray, rows: list[str]) -> None:
    pad = OVERLAY_LABEL_GAP_PX
    widths = [
        cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, 1)[0][0] for row in rows
    ]
    line_height = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, 1)[0][1] + pad
    box_w = min(canvas.shape[1], max(widths) + 2 * pad)
    box_h = line_height * len(rows) + pad
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), OVERLAY_BANNER_BG, -1)
    for index, row in enumerate(rows):
        cv2.putText(
            canvas, row, (pad, line_height * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE, OVERLAY_BANNER_TEXT, 1, cv2.LINE_AA,
        )


def _record(region: TextRegion) -> dict[str, Any]:
    x, y, w, h = region.bbox
    return {
        "id": region.id,
        "role": region.role,
        "role_is_proposed": True,
        "detection_method": region.detection_method,
        "polarity": region.polarity,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "dimensions": {
            "width": w,
            "height": h,
            "aspect_ratio": round(max(w, h) / max(1, min(w, h)), 3),
            "ink_pixels": region.ink_pixels,
            "fill_ratio": region.fill_ratio,
            "glyph_height_px": region.glyph_height,
            "stroke_width_px": region.stroke_width,
        },
        "quality": region.quality,
        "quality_evidence": region.quality_evidence,
        "component_count": region.component_count,
        "component_bboxes": [
            {"x": bx, "y": by, "width": bw, "height": bh}
            for bx, by, bw, bh in region.component_bboxes
        ],
        "evidence": region.evidence,
        "uncertainty": region.uncertainty,
    }


def _build_document(
    regions: list[TextRegion],
    original: Image.Image,
    normalized: Image.Image,
    mask: Image.Image,
) -> dict[str, Any]:
    """The ``text_regions.json`` document."""
    return {
        "phase": {"number": 3, "name": "text_regions", "step": "c2"},
        "status": "proposed (unverified): regions look text-like geometrically; "
                  "none is confirmed text and none is confirmed a room label or "
                  "dimension string",
        "image": {
            "width": original.width,
            "height": original.height,
            "normalized_width": normalized.width,
            "normalized_height": normalized.height,
            "coordinate_space": "original",
        },
        "mask": {
            "filename": TEXT_REGIONS_MASK_FILENAME,
            "width": mask.width,
            "height": mask.height,
            "meaning": "255 = proposed text/annotation region (filled bounding box); "
                       "source pixels are never erased or modified",
        },
        "ocr": {
            "used": False,
            "reason": "no OCR library or binary is available in the environment; "
                      "no heavyweight dependency was added",
        },
        "regions": [_record(region) for region in regions],
    }


def _build_report(
    regions: list[TextRegion],
    original: Image.Image,
    normalized: Image.Image,
    components: list[_Component],
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    """The ``text_detection_report.json`` document."""
    by_role: dict[str, int] = {}
    by_polarity: dict[str, int] = {}
    for region in regions:
        by_role[region.role] = by_role.get(region.role, 0) + 1
        by_polarity[region.polarity] = by_polarity.get(region.polarity, 0) + 1
    mask_pixels = sum(r.bbox[2] * r.bbox[3] for r in regions)
    return {
        "phase": {"number": 3, "name": "text_regions", "step": "c2"},
        "steps": {
            "c2": "proposed text/annotation region detection: two-polarity mark "
                  "extraction, connected components with stroke-density measures, "
                  "spatial grouping into lines; every output is a proposal and no "
                  "source pixel or wall candidate is changed",
        },
        "summary": {
            "total_regions": len(regions),
            "by_role": by_role,
            "by_polarity": by_polarity,
            "components_considered": len(components),
            "mean_quality": round(sum(r.quality for r in regions) / len(regions), 4) if regions else 0.0,
            "low_quality_regions": sum(1 for r in regions if r.quality < 0.5),
            "mask_coverage_fraction": round(mask_pixels / max(1, original.width * original.height), 4),
        },
        "input": {
            "original_width": original.width,
            "original_height": original.height,
            "normalized_width": normalized.width,
            "normalized_height": normalized.height,
            "coordinate_space": "original",
        },
        "parameters": {
            "dark_marks": f"adaptive threshold (block {ADAPTIVE_BLOCK_SIZE}, C {ADAPTIVE_C}), same method as Step A",
            "light_marks": "light pixels enclosed by dark marks (holes of the dark mask)",
            "min_component_area_px": MIN_COMPONENT_AREA_PX,
            "glyph_max_height_px": GLYPH_MAX_HEIGHT_PX,
            "glyph_max_width_px": GLYPH_MAX_WIDTH_PX,
            "glyph_max_aspect": GLYPH_MAX_ASPECT,
            "glyph_max_stroke_width_px": GLYPH_MAX_STROKE_WIDTH_PX,
            "min_stroke_runs_elongated": MIN_STROKE_RUNS_ELONGATED,
            "group_max_height_ratio": GROUP_MAX_HEIGHT_RATIO,
            "group_axis_tolerance": GROUP_AXIS_TOLERANCE,
            "group_coherence_spread_fraction": GROUP_COHERENCE_SPREAD_FRACTION,
            "region_gates": {
                "min_glyph_height_px": MIN_REGION_GLYPH_HEIGHT_PX,
                "min_alignment": MIN_REGION_ALIGNMENT,
                "min_height_consistency": MIN_REGION_HEIGHT_CONSISTENCY,
                "min_single_component_stroke_runs": MIN_REGION_SINGLE_STROKE_RUNS,
                "min_cross_axis_stroke_runs": MIN_REGION_CROSS_RUNS,
            },
            "role_room_label_min_height_px": ROLE_ROOM_LABEL_MIN_HEIGHT_PX,
            "quality_weights": QUALITY_WEIGHTS,
            "scale_from_normalized": [round(scale_x, 6), round(scale_y, 6)],
        },
        "constraints_respected": [
            "All source pixels preserved: nothing erased, inpainted or replaced.",
            "Output is explicitly a proposed text-region mask, not a verdict.",
            "Wall candidates are neither rejected nor filtered here.",
            "No production wall extraction, room detection, opening detection, UI or API changed.",
        ],
        "limitations": [
            "Every region is a proposal. Without OCR or labelled ground truth "
            "the detector can only say a region looks like text, never that it is.",
            "No OCR was used: no OCR library or binary is available in this "
            "environment and no heavyweight dependency was added, so letters "
            "and digits are inferred from geometry only.",
            "The role of a region (room_label_like, dimension_like, text_line) "
            "is a heuristic guess from glyph size and shape statistics.",
            "Anti-aliased text can merge into blobs through its halo; a region "
            "box may then be larger than the true text and grouping may join "
            "adjacent lines (label and dimension string).",
            "Low-contrast grey text can fragment into specks; fragments smaller "
            "than the component floor are not proposed at all.",
            "Dark text drawn on a mid-tone fill (such as the tan balcony fills) "
            "merges with the fill in the dark mask and can be missed entirely; "
            "light-on-dark detection only recovers light glyphs enclosed by dark "
            "marks (the upper-left rotated BALCONY label of real-time.png is one "
            "such miss).",
            "Rotated text is grouped along its reading axis, but curved or "
            "handwritten annotations (such as marker circles) are not text-like "
            "under these measures and are not proposed.",
            "Furniture rails and thin architectural strokes can pass the "
            "geometry filters; stroke density mitigates this but does not "
            "eliminate it.",
            "Single-mark regions are kept as low-evidence proposals and are "
            "listed with an explicit uncertainty reason.",
        ],
    }


# --- standalone runner -------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Generate the Step C2 artifacts for one floor plan.

    Run from the project root:

        python -m floorplanto3d.processing.text_regions IMAGE [OUT_DIR]
    """
    parser = argparse.ArgumentParser(description="Phase 2, Step C2 proposed text regions")
    parser.add_argument("image", type=Path, help="original floor plan image")
    parser.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=Path("artifacts/parse2d/phase_3_refined/c2_text_regions"),
        help="output directory (default: artifacts/parse2d/phase_3_refined/c2_text_regions)",
    )
    args = parser.parse_args(argv)

    normalization = normalize_image(args.image)
    result = detect_text_regions(normalization.normalized, normalization.original)
    outputs = save_artifacts(result, args.out_dir)
    for name, path in outputs.items():
        print(f"wrote {path}  ({name})")
    summary = result.report["summary"]
    print(
        f"{summary['total_regions']} proposed text regions "
        f"({summary['by_role'].get('room_label_like', 0)} room-label-like, "
        f"{summary['by_role'].get('dimension_like', 0)} dimension-like)"
    )


if __name__ == "__main__":
    main()
