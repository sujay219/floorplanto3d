"""Phase 2, Step C: wall-candidate identification.

A wall in a floor plan is a *thick stroke with two parallel edges*. This
step turns the Step A foreground mask into **geometric stroke candidates**
without deciding what anything is: every candidate is a proposal backed by
measurable evidence (paired parallel edges, thickness, continuity), never a
confirmed wall. Classification into wall / furniture / annotation is a
later step and is deliberately absent here.

The pipeline is:

1. Edge extraction. Canny on the Step A foreground mask gives the ink
   boundary — the two long faces of a thick stroke land here as two
   approximately parallel lines.
2. Edge consolidation. Nearly collinear Hough fragments are merged into one
   edge line each, so one physical stroke face stays one edge.
3. Edge pairing. Two approximately parallel edges at a plausible stroke
   distance with overlapping extent are merged into one stroke candidate
   whose centreline runs between them (this is where duplicate edges of a
   thick stroke collapse into a single candidate). Edges that find no mate
   are preserved as single-edge candidates — uncertain, never discarded.
4. Duplicate merging. Collinear candidates that overlap and agree on
   thickness are merged into one run and recorded with their sources.
5. Measurement. Length, orientation, thickness, straightness, thickness
   consistency, continuity and structural-mask support are measured for
   every candidate and combined into a confidence score with written
   evidence and uncertainty reasons.

The Step B structural masks (horizontal, vertical, combined) are recorded
as supporting evidence only. They are diagnostics and never drive
detection, matching the rule that they must not feed downstream stages.

Coordinates are reported in original-image pixel space; detection itself
runs in the normalized working space of the masks (Phase 1 preserves the
aspect ratio, so the mapping is a uniform scale in practice).
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

from floorplanto3d.models.geometry import classify_orientation
from floorplanto3d.processing.image import load_image, open_image
from floorplanto3d.processing.normalize import png_bytes
from floorplanto3d.processing.wall_detection import (
    COMBINED_STRUCTURES_FILENAME,
    FOREGROUND_MASK_FILENAME,
    HORIZONTAL_STRUCTURES_FILENAME,
    VERTICAL_STRUCTURES_FILENAME,
    detect_walls,
)

logger = logging.getLogger(__name__)

# Edge extraction. The Canny thresholds are fixed starting values on the
# binarized Step A mask (ink = 255), where edges are unambiguous boundaries.
CANNY_LOW = 50
CANNY_HIGH = 150

# Raw segment extraction (probabilistic Hough). Length and gap thresholds
# are fractions of the working image diagonal so nothing is tuned to one
# floor plan or one resolution; the floor keeps tiny synthetic tests sane.
# The vote threshold tracks the minimum segment length (votes scale with
# the number of collinear edge pixels), so the effective cut-off stays
# consistent at any resolution.
HOUGH_THRESHOLD_FLOOR = 12
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
MIN_SEGMENT_LENGTH_FRACTION = 0.012
MIN_SEGMENT_LENGTH_FLOOR_PX = 6
HOUGH_MAX_LINE_GAP_FRACTION = 0.004

# Edge consolidation: fragments merge when they agree in angle and sit on
# the same line. Gaps are bridged only when the gap itself carries ink
# evidence along the line — the probabilistic Hough pass leaves sampling
# holes on perfectly continuous edges, while a genuine interruption (door
# gap) has no ink between the fragments and stays split.
EDGE_MERGE_ANGLE_TOL_DEG = 4.0
EDGE_MERGE_DIST_FRACTION = 0.004
EDGE_MERGE_DIST_FLOOR_PX = 2.0
EDGE_MERGE_BRIDGE_FRACTION = 0.03
EDGE_MERGE_BRIDGE_COVERAGE = 0.5

# Edge pairing: two edges form one stroke candidate when they are
# approximately parallel, a plausible stroke width apart, and their
# extents overlap along the shared direction.
PAIR_ANGLE_TOL_DEG = 4.0
PAIR_MIN_DISTANCE_PX = 1.0
PAIR_MAX_DISTANCE_FRACTION = 0.03
PAIR_MIN_OVERLAP = 0.5

# Duplicate-candidate merging: collinear runs that overlap or nearly touch
# and agree on thickness are one stroke, reported once with merged_from.
CANDIDATE_MERGE_ANGLE_TOL_DEG = 4.0
CANDIDATE_MERGE_DIST_FRACTION = 0.006
CANDIDATE_MERGE_DIST_FLOOR_PX = 2.0
CANDIDATE_MERGE_GAP_FRACTION = 0.015
CANDIDATE_MERGE_THICKNESS_RATIO = 1.8
CANDIDATE_MERGE_THIN_FLOOR_PX = 3.0

# Measurement. Probing samples the foreground ink perpendicular to the
# centreline; continuity uses a band wide enough to also see hollow
# double-line strokes (ink on both faces counts as covered).
THICKNESS_PROBE_MAX_SPAN_FRACTION = 0.05
CONTINUITY_MAX_SAMPLES = 200
CONTINUITY_BAND_MARGIN_PX = 1.0
STRAIGHTNESS_TOLERANCE_FRACTION = 0.004
STRAIGHTNESS_TOLERANCE_FLOOR_PX = 2.0

# Confidence combines independent geometric signals only. It is confidence
# in "this is one coherent stroke-like structure", never in "this is a wall".
CONFIDENCE_WEIGHTS = {
    "pairing": 0.35,
    "continuity": 0.25,
    "straightness": 0.2,
    "thickness_consistency": 0.2,
}

# Overlay drawing (RGB). Outlines only, never filled, so the floor plan
# underneath stays readable. Colour codes the edge-pairing status only —
# no semantic classification is implied.
PAIRED_COLOR = (30, 120, 220)
UNPAIRED_COLOR = (240, 140, 50)
OVERLAY_TEXT_COLOR = (255, 255, 255)
OVERLAY_LEGEND_BG = (255, 255, 255)
OVERLAY_LEGEND_TEXT = (32, 32, 32)
LINE_THICKNESS_FRACTION = 0.002
LINE_THICKNESS_MIN_PX = 2
FONT_SCALE_FRACTION = 0.0002
FONT_SCALE_MIN = 0.03
LABEL_GAP_PX = 4

# Saved artifact filenames.
CANDIDATES_OVERLAY_FILENAME = "wall_candidates_overlay.png"
WALL_CANDIDATES_FILENAME = "wall_candidates.json"
CANDIDATE_REPORT_FILENAME = "candidate_report.json"


@dataclass
class WallCandidate:
    """One geometric stroke candidate, in original-image pixel space.

    A candidate is a *proposal*: its confidence is confidence in the
    geometric coherence of the stroke, not in any semantic reading.
    """

    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[int, int, int, int]
    length: float
    angle_deg: float
    orientation: str
    estimated_thickness: float
    thickness_consistency: float
    straightness: float
    continuity: float
    edge_pairing: dict[str, Any]
    axis_support: dict[str, float]
    confidence: float
    confidence_evidence: dict[str, float]
    evidence: list[str]
    uncertainty: list[str]
    merged_from: list[str] = field(default_factory=list)


@dataclass
class WallCandidateDetectionResult:
    """Outputs of Phase 2 (wall detection), Step C."""

    candidates: list[WallCandidate]
    overlay: Image.Image
    candidates_document: dict[str, Any]
    report: dict[str, Any]


@dataclass
class _Segment:
    """One raw Hough segment in working (mask) space."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def mid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def angle_mid(self) -> float:
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0


@dataclass
class _EdgeLine:
    """A merged ink-boundary line (one physical stroke face)."""

    id: str
    origin: tuple[float, float]
    ux: float
    uy: float
    t0: float
    t1: float
    points: list[tuple[float, float]]

    def point_at(self, t: float) -> tuple[float, float]:
        return (self.origin[0] + self.ux * t, self.origin[1] + self.uy * t)

    def project(self, point: tuple[float, float]) -> float:
        return (point[0] - self.origin[0]) * self.ux + (point[1] - self.origin[1]) * self.uy

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.uy, self.ux)) % 180.0

    @property
    def length(self) -> float:
        return self.t1 - self.t0


@dataclass
class _Stroke:
    """One stroke candidate before measurement, in working space."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    polygon: tuple[tuple[float, float], ...]
    thickness: float
    pairing: dict[str, Any]
    merged_from: list[str]


def detect_candidates(
    original: str | Path | bytes | bytearray | Image.Image,
    foreground_mask: Image.Image,
    horizontal: Image.Image,
    vertical: Image.Image,
    combined: Image.Image,
) -> WallCandidateDetectionResult:
    """Run Phase 2, Step C on the Step A/B outputs of one floor plan.

    Args:
        original: The original floor plan image, used purely as the visual
            reference the overlay is drawn on and the coordinate space of
            the report. Its pixels are never modified.
        foreground_mask: Step A's primary foreground mask (255 = ink).
        horizontal: Step B horizontal structures (evidence only).
        vertical: Step B vertical structures (evidence only).
        combined: Step B combined structures (evidence only).

    Returns:
        A :class:`WallCandidateDetectionResult` with every candidate
        preserved (uncertain ones included), the overlay, the
        ``wall_candidates.json`` document and the ``candidate_report.json``
        summary. Nothing here is classified: candidates are geometric
        proposals, not confirmed walls.
    """
    original_image = open_image(original)
    base_rgb, original_width, original_height, _ = load_image(original_image)
    foreground = _mask_array(foreground_mask)
    horizontal_array = _mask_array(horizontal)
    vertical_array = _mask_array(vertical)
    combined_array = _mask_array(combined)
    mask_height, mask_width = foreground.shape[:2]

    diagonal = math.hypot(mask_width, mask_height)
    edges = _edge_lines(foreground, diagonal)
    strokes = _pair_edges(edges, foreground, diagonal)
    strokes = _merge_strokes(strokes, diagonal)
    candidates = [
        _measure(
            stroke,
            foreground,
            horizontal_array,
            vertical_array,
            combined_array,
            diagonal,
            index,
        )
        for index, stroke in enumerate(strokes)
    ]
    candidates = _assign_ids(candidates)

    scale_x = original_width / max(1, mask_width)
    scale_y = original_height / max(1, mask_height)
    candidates = [_to_original_space(c, scale_x, scale_y) for c in candidates]

    overlay = _draw_overlay(base_rgb, candidates)
    candidates_document = _build_candidates_document(
        candidates, original_image, foreground_mask
    )
    report = _build_report(
        candidates, original_image, foreground_mask, diagonal, scale_x, scale_y
    )

    logger.info(
        "wall-candidate identification (Step C) complete: %d candidates "
        "(%d with paired edges, %d single-edge)",
        report["summary"]["total_candidates"],
        report["summary"]["paired_edge_candidates"],
        report["summary"]["single_edge_candidates"],
    )
    return WallCandidateDetectionResult(
        candidates=candidates,
        overlay=overlay,
        candidates_document=candidates_document,
        report=report,
    )


def save_artifacts(
    result: WallCandidateDetectionResult, directory: str | Path
) -> dict[str, Path]:
    """Write the Step C artifacts into ``directory``.

    Writes ``wall_candidates_overlay.png``, ``wall_candidates.json`` and
    ``candidate_report.json``, and returns the output paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "overlay": directory / CANDIDATES_OVERLAY_FILENAME,
        "candidates": directory / WALL_CANDIDATES_FILENAME,
        "report": directory / CANDIDATE_REPORT_FILENAME,
    }
    outputs["overlay"].write_bytes(png_bytes(result.overlay))
    outputs["candidates"].write_text(
        json.dumps(result.candidates_document, indent=2) + "\n", encoding="utf-8"
    )
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


# --- edge extraction and consolidation ---------------------------------


def _mask_array(mask: Image.Image) -> np.ndarray:
    """8-bit single-channel view of a Step A/B mask (255 = ink)."""
    return np.asarray(open_image(mask).convert("L"))


def _edge_lines(foreground: np.ndarray, diagonal: float) -> list[_EdgeLine]:
    """Merge the ink-boundary segments into one line per stroke face."""
    min_length = max(
        MIN_SEGMENT_LENGTH_FLOOR_PX, MIN_SEGMENT_LENGTH_FRACTION * diagonal
    )
    max_gap = HOUGH_MAX_LINE_GAP_FRACTION * diagonal
    canny = cv2.Canny(foreground, CANNY_LOW, CANNY_HIGH)
    raw = cv2.HoughLinesP(
        canny,
        HOUGH_RHO,
        HOUGH_THETA,
        max(HOUGH_THRESHOLD_FLOOR, round(min_length)),
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    segments = (
        [_Segment(*[float(v) for v in np.asarray(line).ravel()[:4]]) for line in raw]
        if raw is not None
        else []
    )
    segments.sort(key=lambda s: (-s.length, s.mid[1], s.mid[0], s.angle_mid))

    dist_tol = max(EDGE_MERGE_DIST_FLOOR_PX, EDGE_MERGE_DIST_FRACTION * diagonal)
    bridge_max = EDGE_MERGE_BRIDGE_FRACTION * diagonal
    lines: list[_EdgeLine] = []
    for segment in segments:
        segment_line = _new_edge_line(segment)
        best_index = -1
        best_key: tuple[float, float] | None = None
        for index, line in enumerate(lines):
            key = _merge_key(line, segment_line, dist_tol, bridge_max, canny)
            if key is None:
                continue
            if best_key is None or key < best_key:
                best_key, best_index = key, index
        if best_index >= 0:
            _absorb(lines[best_index], segment_line)
        else:
            lines.append(segment_line)
    lines = _consolidate_lines(lines, dist_tol, bridge_max, canny)

    lines.sort(key=lambda e: (round(e.angle_deg, 3), round(e.origin[1], 2), round(e.origin[0], 2), e.t0))
    for index, line in enumerate(lines, start=1):
        line.id = f"edge_{index:03d}"
    return lines


def _new_edge_line(segment: _Segment) -> _EdgeLine:
    angle = math.radians(segment.angle_mid)
    ux, uy = math.cos(angle), math.sin(angle)
    origin = (segment.x1, segment.y1)
    # The folded direction may oppose the segment's own direction, so the
    # extent must come from the signed projection of the far endpoint.
    t_end = (segment.x2 - segment.x1) * ux + (segment.y2 - segment.y1) * uy
    t0, t1 = sorted((0.0, t_end))
    return _EdgeLine(
        id="",
        origin=origin,
        ux=ux,
        uy=uy,
        t0=t0,
        t1=t1,
        points=[(segment.x1, segment.y1), (segment.x2, segment.y2)],
    )


def _consolidate_lines(
    lines: list[_EdgeLine], dist_tol: float, bridge_max: float, canny: np.ndarray
) -> list[_EdgeLine]:
    """Merge lines that only became consolidatable after growing.

    Absorbing a fragment can bring a line within reach of a neighbour it
    previously missed, so the pair-wise pass repeats until stable.
    """
    merged = list(lines)
    changed = True
    while changed:
        changed = False
        merged.sort(
            key=lambda e: (round(e.angle_deg, 3), round(e.origin[1], 2), round(e.origin[0], 2), e.t0)
        )
        for i in range(len(merged)):
            if changed:
                break
            for j in range(i + 1, len(merged)):
                if _merge_key(merged[i], merged[j], dist_tol, bridge_max, canny) is None:
                    continue
                _absorb(merged[i], merged[j])
                del merged[j]
                changed = True
                break
    return merged


def _merge_key(
    line: _EdgeLine, other: _EdgeLine, dist_tol: float, bridge_max: float, canny: np.ndarray
) -> tuple[float, float] | None:
    """(angle delta, perpendicular distance) if ``other`` merges, else None."""
    angle_delta = _angle_delta(line.angle_deg, other.angle_deg)
    if angle_delta > EDGE_MERGE_ANGLE_TOL_DEG:
        return None
    distance = max(
        _perpendicular_distance(line, other.point_at(other.t0)),
        _perpendicular_distance(line, other.point_at(other.t1)),
    )
    if distance > dist_tol:
        return None
    t_a = line.project(other.point_at(other.t0))
    t_b = line.project(other.point_at(other.t1))
    lo, hi = min(t_a, t_b), max(t_a, t_b)
    if hi < line.t0:
        gap_from, gap_to = hi, line.t0
    elif lo > line.t1:
        gap_from, gap_to = line.t1, lo
    else:
        gap_from, gap_to = 0.0, 0.0
    gap = gap_to - gap_from
    if gap > bridge_max:
        return None
    if gap > 0 and _line_ink_coverage(canny, line, gap_from, gap_to) < EDGE_MERGE_BRIDGE_COVERAGE:
        return None
    return (angle_delta, distance)


def _line_ink_coverage(
    canny: np.ndarray, line: _EdgeLine, t_from: float, t_to: float
) -> float:
    """Fraction of a line interval that carries edge ink within ~1 px."""
    height, width = canny.shape[:2]
    samples = max(1, int(math.ceil(t_to - t_from)))
    hits = 0
    for step in range(samples + 1):
        t = t_from + (t_to - t_from) * (step / samples if samples else 0.0)
        x, y = line.point_at(t)
        found = False
        for offset in (-1.0, 0.0, 1.0):
            px = int(round(x - line.uy * offset))
            py = int(round(y + line.ux * offset))
            if 0 <= px < width and 0 <= py < height and canny[py, px] > 0:
                found = True
                break
        if found:
            hits += 1
    return hits / (samples + 1)


def _absorb(line: _EdgeLine, other: _EdgeLine) -> None:
    for t in (other.t0, other.t1):
        projected = line.project(other.point_at(t))
        line.t0 = min(line.t0, projected)
        line.t1 = max(line.t1, projected)
    line.points.extend(other.points)


def _perpendicular_distance(line: _EdgeLine, point: tuple[float, float]) -> float:
    px = point[0] - line.origin[0]
    py = point[1] - line.origin[1]
    return abs(px * line.uy - py * line.ux)


def _angle_delta(a_deg: float, b_deg: float) -> float:
    delta = abs(a_deg - b_deg) % 180.0
    return min(delta, 180.0 - delta)


# --- edge pairing and duplicate merging --------------------------------


def _pair_edges(
    edges: list[_EdgeLine], foreground: np.ndarray, diagonal: float
) -> list[_Stroke]:
    """Pair parallel edges into strokes; unpaired edges stay single-edge."""
    max_distance = PAIR_MAX_DISTANCE_FRACTION * diagonal
    proposals: list[tuple[float, int, int, dict[str, Any]]] = []
    for i, a in enumerate(edges):
        for j in range(i + 1, len(edges)):
            b = edges[j]
            pairing = _pairing_evidence(a, b, max_distance)
            if pairing is not None:
                proposals.append((pairing["score"], i, j, pairing))
    proposals.sort(key=lambda p: (-p[0], p[1], p[2]))

    used: set[int] = set()
    strokes: list[_Stroke] = []
    for _score, i, j, pairing in proposals:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        strokes.append(_paired_stroke(edges[i], edges[j], pairing))
    for index, edge in enumerate(edges):
        if index in used:
            continue
        strokes.append(_single_edge_stroke(edge, foreground, diagonal))
    return strokes


def _pairing_evidence(
    a: _EdgeLine, b: _EdgeLine, max_distance: float
) -> dict[str, Any] | None:
    """The geometric evidence that two edges are faces of one stroke."""
    angle_delta = _angle_delta(a.angle_deg, b.angle_deg)
    if angle_delta > PAIR_ANGLE_TOL_DEG:
        return None
    distance = _perpendicular_distance(a, b.point_at(b.t0))
    if not PAIR_MIN_DISTANCE_PX <= distance <= max_distance:
        return None

    t_lo = max(a.t0, min(a.project(b.point_at(b.t0)), a.project(b.point_at(b.t1))))
    t_hi = min(a.t1, max(a.project(b.point_at(b.t0)), a.project(b.point_at(b.t1))))
    overlap = t_hi - t_lo
    shorter = min(a.length, b.length)
    if shorter <= 0 or overlap <= 0:
        return None
    overlap_fraction = overlap / shorter
    if overlap_fraction < PAIR_MIN_OVERLAP:
        return None

    score = overlap_fraction * (1.0 - angle_delta / PAIR_ANGLE_TOL_DEG)
    return {
        "paired": True,
        "edge_ids": [a.id, b.id],
        "angle_delta_deg": round(angle_delta, 3),
        "distance_px": round(distance, 2),
        "overlap_fraction": round(overlap_fraction, 4),
        "score": round(score, 4),
        "overlap": (t_lo, t_hi),
    }


def _paired_stroke(
    a: _EdgeLine, b: _EdgeLine, pairing: dict[str, Any]
) -> _Stroke:
    """Centreline between two paired edges over their overlap interval."""
    t_lo, t_hi = pairing["overlap"]
    a0, a1 = a.point_at(t_lo), a.point_at(t_hi)
    b0 = b.point_at(b.project(a0))
    b1 = b.point_at(b.project(a1))
    p0 = ((a0[0] + b0[0]) / 2.0, (a0[1] + b0[1]) / 2.0)
    p1 = ((a1[0] + b1[0]) / 2.0, (a1[1] + b1[1]) / 2.0)
    thickness = (math.dist(a0, b0) + math.dist(a1, b1)) / 2.0
    evidence = {
        key: value for key, value in pairing.items() if key != "overlap"
    }
    return _Stroke(
        p0=p0,
        p1=p1,
        polygon=(a0, a1, b1, b0),
        thickness=thickness,
        pairing=evidence,
        merged_from=[],
    )


def _single_edge_stroke(
    edge: _EdgeLine, foreground: np.ndarray, diagonal: float
) -> _Stroke:
    """A candidate from one edge with no parallel mate (kept, never dropped)."""
    p0, p1 = edge.point_at(edge.t0), edge.point_at(edge.t1)
    mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    thickness = _probe_ink_run(
        foreground, mid, edge.ux, edge.uy, THICKNESS_PROBE_MAX_SPAN_FRACTION * diagonal
    )
    half = max(0.5, thickness / 2.0)
    nx, ny = -edge.uy, edge.ux
    polygon = (
        (p0[0] + nx * half, p0[1] + ny * half),
        (p1[0] + nx * half, p1[1] + ny * half),
        (p1[0] - nx * half, p1[1] - ny * half),
        (p0[0] - nx * half, p0[1] - ny * half),
    )
    return _Stroke(
        p0=p0,
        p1=p1,
        polygon=polygon,
        thickness=thickness,
        pairing={
            "paired": False,
            "edge_ids": [edge.id],
            "angle_delta_deg": None,
            "distance_px": None,
            "overlap_fraction": None,
            "score": 0.0,
        },
        merged_from=[],
    )


def _merge_strokes(strokes: list[_Stroke], diagonal: float) -> list[_Stroke]:
    """Merge collinear duplicate candidates describing the same stroke."""
    dist_tol = max(
        CANDIDATE_MERGE_DIST_FLOOR_PX, CANDIDATE_MERGE_DIST_FRACTION * diagonal
    )
    gap_tol = CANDIDATE_MERGE_GAP_FRACTION * diagonal

    merged = list(strokes)
    changed = True
    while changed:
        changed = False
        merged.sort(key=_stroke_sort_key)
        for i in range(len(merged)):
            if changed:
                break
            for j in range(i + 1, len(merged)):
                combined = _try_merge_strokes(merged[i], merged[j], dist_tol, gap_tol)
                if combined is None:
                    continue
                merged[i] = combined
                del merged[j]
                changed = True
                break
    return merged


def _stroke_sort_key(stroke: _Stroke) -> tuple[float, float, float, float]:
    mid = ((stroke.p0[0] + stroke.p1[0]) / 2.0, (stroke.p0[1] + stroke.p1[1]) / 2.0)
    angle = math.degrees(math.atan2(stroke.p1[1] - stroke.p0[1], stroke.p1[0] - stroke.p0[0])) % 180.0
    return (round(mid[1], 2), round(mid[0], 2), round(angle, 3), -_stroke_length(stroke))


def _try_merge_strokes(
    a: _Stroke, b: _Stroke, dist_tol: float, gap_tol: float
) -> _Stroke | None:
    angle_a = math.degrees(math.atan2(a.p1[1] - a.p0[1], a.p1[0] - a.p0[0])) % 180.0
    angle_b = math.degrees(math.atan2(b.p1[1] - b.p0[1], b.p1[0] - b.p0[0])) % 180.0
    if _angle_delta(angle_a, angle_b) > CANDIDATE_MERGE_ANGLE_TOL_DEG:
        return None
    thin_a = a.thickness <= CANDIDATE_MERGE_THIN_FLOOR_PX
    thin_b = b.thickness <= CANDIDATE_MERGE_THIN_FLOOR_PX
    if not (thin_a and thin_b):
        ratio = max(a.thickness, b.thickness) / max(1e-6, min(a.thickness, b.thickness))
        if ratio > CANDIDATE_MERGE_THICKNESS_RATIO:
            return None

    ux = math.cos(math.radians(angle_a))
    uy = math.sin(math.radians(angle_a))
    a_ext = sorted((_project(a.p0, a.p0, ux, uy), _project(a.p1, a.p0, ux, uy)))
    b_ext = sorted((_project(b.p0, a.p0, ux, uy), _project(b.p1, a.p0, ux, uy)))
    gap = max(a_ext[0], b_ext[0]) - min(a_ext[1], b_ext[1])
    if gap > gap_tol:
        return None

    lateral = _perpendicular_distance(_line_of(a, ux, uy), b.p0)
    if lateral > dist_tol:
        return None

    lo = min(a_ext[0], b_ext[0])
    hi = max(a_ext[1], b_ext[1])
    p0 = (a.p0[0] + ux * lo, a.p0[1] + uy * lo)
    p1 = (a.p0[0] + ux * hi, a.p0[1] + uy * hi)
    half = max(0.5, (a.thickness + b.thickness) / 4.0)
    nx, ny = -uy, ux
    polygon = (
        (p0[0] + nx * half, p0[1] + ny * half),
        (p1[0] + nx * half, p1[1] + ny * half),
        (p1[0] - nx * half, p1[1] - ny * half),
        (p0[0] - nx * half, p0[1] - ny * half),
    )
    pairing = a.pairing if a.pairing["paired"] else b.pairing
    sources = (
        a.merged_from
        + b.merged_from
        + list(a.pairing["edge_ids"])
        + list(b.pairing["edge_ids"])
    )
    return _Stroke(
        p0=p0,
        p1=p1,
        polygon=polygon,
        thickness=(a.thickness + b.thickness) / 2.0,
        pairing=pairing,
        merged_from=sorted(set(sources)),
    )


def _line_of(stroke: _Stroke, ux: float, uy: float) -> _EdgeLine:
    return _EdgeLine(id="", origin=stroke.p0, ux=ux, uy=uy, t0=0.0, t1=_stroke_length(stroke), points=[])


def _project(point: tuple[float, float], origin: tuple[float, float], ux: float, uy: float) -> float:
    return (point[0] - origin[0]) * ux + (point[1] - origin[1]) * uy


def _stroke_length(stroke: _Stroke) -> float:
    return math.dist(stroke.p0, stroke.p1)


# --- measurement -------------------------------------------------------


def _measure(
    stroke: _Stroke,
    foreground: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    combined: np.ndarray,
    diagonal: float,
    index: int,
) -> WallCandidate:
    """Measure every documented feature of one candidate."""
    length = _stroke_length(stroke)
    dx, dy = stroke.p1[0] - stroke.p0[0], stroke.p1[1] - stroke.p0[1]
    angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
    orientation = classify_orientation(dx, dy)
    ux = dx / length if length > 0 else 1.0
    uy = dy / length if length > 0 else 0.0

    runs = _probe_runs(foreground, stroke, ux, uy, diagonal)
    probe_thickness = float(np.median(runs)) if runs else 0.0
    mean_run = float(np.mean(runs)) if runs else 0.0
    std_run = float(np.std(runs)) if runs else 0.0
    thickness_consistency = (
        max(0.0, 1.0 - std_run / max(1e-6, mean_run)) if runs else 0.0
    )
    estimated_thickness = (
        stroke.thickness if stroke.pairing["paired"] else probe_thickness
    )

    band = max(1.0, estimated_thickness / 2.0 + CONTINUITY_BAND_MARGIN_PX)
    continuity = _continuity(foreground, stroke, ux, uy, band)
    straightness = _straightness(foreground, stroke, ux, uy, band, diagonal)
    axis_support = {
        "horizontal": _band_coverage(horizontal, stroke, band),
        "vertical": _band_coverage(vertical, stroke, band),
        "combined": _band_coverage(combined, stroke, band),
    }

    pairing = dict(stroke.pairing)
    if pairing["paired"]:
        pairing["probed_thickness_px"] = round(probe_thickness, 2)
    confidence_evidence = {
        "pairing": pairing["score"],
        "continuity": round(continuity, 4),
        "straightness": round(straightness, 4),
        "thickness_consistency": round(thickness_consistency, 4),
    }
    confidence = round(
        sum(CONFIDENCE_WEIGHTS[name] * value for name, value in confidence_evidence.items()),
        4,
    )

    evidence, uncertainty = _explain(
        pairing, continuity, straightness, thickness_consistency, axis_support, stroke
    )
    return WallCandidate(
        id=f"cand_{index + 1:03d}",
        start=stroke.p0,
        end=stroke.p1,
        polygon=stroke.polygon,
        bbox=_bbox(stroke.polygon),
        length=length,
        angle_deg=angle_deg,
        orientation=orientation,
        estimated_thickness=estimated_thickness,
        thickness_consistency=round(thickness_consistency, 4),
        straightness=round(straightness, 4),
        continuity=round(continuity, 4),
        edge_pairing=pairing,
        axis_support={k: round(v, 4) for k, v in axis_support.items()},
        confidence=confidence,
        confidence_evidence=confidence_evidence,
        evidence=evidence,
        uncertainty=uncertainty,
        merged_from=stroke.merged_from,
    )


def _probe_runs(
    foreground: np.ndarray, stroke: _Stroke, ux: float, uy: float, diagonal: float
) -> list[float]:
    """Contiguous ink runs crossing the centreline at sampled points."""
    length = _stroke_length(stroke)
    samples = max(5, min(CONTINUITY_MAX_SAMPLES, int(length) + 1))
    max_span = THICKNESS_PROBE_MAX_SPAN_FRACTION * diagonal
    runs: list[float] = []
    for step in range(samples):
        t = step / (samples - 1) if samples > 1 else 0.5
        x = stroke.p0[0] + ux * length * t
        y = stroke.p0[1] + uy * length * t
        run = _probe_ink_run(foreground, (x, y), ux, uy, max_span)
        runs.append(run)
    return runs


def _probe_ink_run(
    foreground: np.ndarray,
    point: tuple[float, float],
    ux: float,
    uy: float,
    max_span: float,
) -> float:
    """Length in px of the contiguous ink run through ``point`` perpendicular
    to the (ux, uy) direction."""
    nx, ny = -uy, ux
    height, width = foreground.shape[:2]
    steps = int(max_span) + 1
    hit = 0.0
    for direction in (1.0, -1.0):
        distance = 0.0
        for step in range(1, steps + 1):
            x = int(round(point[0] + nx * step * direction))
            y = int(round(point[1] + ny * step * direction))
            if not (0 <= x < width and 0 <= y < height):
                break
            if foreground[y, x] == 0:
                break
            distance = float(step)
        hit += distance
    x0, y0 = int(round(point[0])), int(round(point[1]))
    if 0 <= x0 < width and 0 <= y0 < height and foreground[y0, x0] > 0:
        hit += 1.0
    return hit


def _continuity(
    foreground: np.ndarray,
    stroke: _Stroke,
    ux: float,
    uy: float,
    band: float,
) -> float:
    """Fraction of the centreline whose band contains ink.

    The band spans the estimated stroke width, so hollow double-line
    strokes count as covered while genuine interruptions (door gaps)
    lower the score. Interrupted candidates are preserved either way.
    """
    length = _stroke_length(stroke)
    if length <= 0:
        return 0.0
    samples = max(2, min(CONTINUITY_MAX_SAMPLES, int(length) + 1))
    covered = 0
    for step in range(samples):
        t = step / (samples - 1)
        x = stroke.p0[0] + ux * length * t
        y = stroke.p0[1] + uy * length * t
        if _band_has_ink(foreground, (x, y), ux, uy, band):
            covered += 1
    return covered / samples


def _band_has_ink(
    foreground: np.ndarray,
    point: tuple[float, float],
    ux: float,
    uy: float,
    band: float,
) -> bool:
    nx, ny = -uy, ux
    height, width = foreground.shape[:2]
    steps = int(math.ceil(band)) + 1
    for step in range(-steps, steps + 1):
        x = int(round(point[0] + nx * step))
        y = int(round(point[1] + ny * step))
        if 0 <= x < width and 0 <= y < height and foreground[y, x] > 0:
            return True
    return False


def _band_coverage(mask: np.ndarray, stroke: _Stroke, band: float) -> float:
    """Fraction of the centreline whose band overlaps a Step B mask."""
    array = mask
    length = _stroke_length(stroke)
    if length <= 0 or not np.any(array):
        return 0.0
    dx, dy = stroke.p1[0] - stroke.p0[0], stroke.p1[1] - stroke.p0[1]
    ux = dx / length
    uy = dy / length
    samples = max(2, min(CONTINUITY_MAX_SAMPLES, int(length) + 1))
    covered = 0
    for step in range(samples):
        t = step / (samples - 1)
        x = stroke.p0[0] + ux * length * t
        y = stroke.p0[1] + uy * length * t
        if _band_has_ink(array, (x, y), ux, uy, band):
            covered += 1
    return covered / samples


def _straightness(
    foreground: np.ndarray,
    stroke: _Stroke,
    ux: float,
    uy: float,
    band: float,
    diagonal: float,
) -> float:
    """How closely the ink itself follows a straight line.

    At each sample the ink's centroid across the band is compared with the
    fitted centreline. A constant offset (an edge-based centreline beside
    a thick stroke) is harmless; a wobbling or curved stroke produces
    varying offsets and scores lower.
    """
    tolerance = max(
        STRAIGHTNESS_TOLERANCE_FLOOR_PX, STRAIGHTNESS_TOLERANCE_FRACTION * diagonal
    )
    length = _stroke_length(stroke)
    if length <= 0:
        return 0.0
    nx, ny = -uy, ux
    height, width = foreground.shape[:2]
    samples = max(2, min(CONTINUITY_MAX_SAMPLES, int(length) + 1))
    span = int(math.ceil(band))
    offsets: list[float] = []
    for step in range(samples):
        t = step / (samples - 1)
        x = stroke.p0[0] + ux * length * t
        y = stroke.p0[1] + uy * length * t
        hits = [
            float(k)
            for k in range(-span, span + 1)
            if 0 <= int(round(x + nx * k)) < width
            and 0 <= int(round(y + ny * k)) < height
            and foreground[int(round(y + ny * k)), int(round(x + nx * k))] > 0
        ]
        if hits:
            offsets.append(sum(hits) / len(hits))
    if not offsets:
        return 0.0
    mean = sum(offsets) / len(offsets)
    variance = sum((value - mean) ** 2 for value in offsets) / len(offsets)
    return max(0.0, 1.0 - math.sqrt(variance) / tolerance)


def _explain(
    pairing: dict[str, Any],
    continuity: float,
    straightness: float,
    thickness_consistency: float,
    axis_support: dict[str, float],
    stroke: _Stroke,
) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    uncertainty: list[str] = []
    if pairing["paired"]:
        evidence.append(
            "two approximately parallel edges paired: "
            f"Δangle {pairing['angle_delta_deg']}°, distance {pairing['distance_px']} px, "
            f"overlap {pairing['overlap_fraction']}"
        )
    else:
        uncertainty.append(
            "no parallel edge pair: single-edge candidate (thin stroke or only one visible edge)"
        )
    if continuity >= 0.85:
        evidence.append(f"ink covers {round(continuity * 100)}% of the centreline band")
    elif continuity > 0:
        uncertainty.append(
            f"interrupted stroke: ink covers only {round(continuity * 100)}% of the centreline band"
        )
    else:
        uncertainty.append("no ink found along the centreline band")
    if straightness >= 0.8:
        evidence.append("supporting edges are straight along the centreline")
    else:
        uncertainty.append("supporting edges deviate from the fitted centreline")
    if thickness_consistency >= 0.7:
        evidence.append("thickness is consistent along the stroke")
    else:
        uncertainty.append("thickness varies along the stroke")
    for name in ("horizontal", "vertical"):
        if axis_support[name] > 0.0:
            evidence.append(
                f"Step B {name} structure mask overlaps {round(axis_support[name] * 100)}% "
                "of the centreline (supporting evidence only)"
            )
    if stroke.merged_from:
        evidence.append(
            f"merged {len(stroke.merged_from)} duplicate edge(s)/candidate(s): "
            + ", ".join(stroke.merged_from)
        )
    return evidence, uncertainty


# --- reporting and overlay --------------------------------------------


def _bbox(
    polygon: tuple[tuple[float, float], ...]
) -> tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs) - min(xs))),
        int(math.ceil(max(ys) - min(ys))),
    )


def _assign_ids(candidates: list[WallCandidate]) -> list[WallCandidate]:
    """Stable spatial ordering, then cand_001… assignment."""
    candidates.sort(
        key=lambda c: (
            round((c.start[1] + c.end[1]) / 2.0, 2),
            round((c.start[0] + c.end[0]) / 2.0, 2),
            round(c.angle_deg, 3),
            round(c.length, 2),
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate.id = f"cand_{index:03d}"
    return candidates


def _to_original_space(
    candidate: WallCandidate, scale_x: float, scale_y: float
) -> WallCandidate:
    """Re-express a candidate in original-image pixel space.

    Phase 1 preserves the aspect ratio, so the scale is uniform in
    practice; scalars use the mean scale, points are mapped per axis.
    """
    scale = (scale_x + scale_y) / 2.0

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] * scale_x, point[1] * scale_y)

    pairing = dict(candidate.edge_pairing)
    if pairing.get("distance_px") is not None:
        pairing["distance_px"] = round(pairing["distance_px"] * scale, 2)
    if pairing.get("probed_thickness_px") is not None:
        pairing["probed_thickness_px"] = round(
            pairing["probed_thickness_px"] * scale, 2
        )
    return WallCandidate(
        id=candidate.id,
        start=map_point(candidate.start),
        end=map_point(candidate.end),
        polygon=tuple(map_point(p) for p in candidate.polygon),
        bbox=_bbox(tuple(map_point(p) for p in candidate.polygon)),
        length=candidate.length * scale,
        angle_deg=candidate.angle_deg,
        orientation=candidate.orientation,
        estimated_thickness=candidate.estimated_thickness * scale,
        thickness_consistency=candidate.thickness_consistency,
        straightness=candidate.straightness,
        continuity=candidate.continuity,
        edge_pairing=pairing,
        axis_support=candidate.axis_support,
        confidence=candidate.confidence,
        confidence_evidence=candidate.confidence_evidence,
        evidence=candidate.evidence,
        uncertainty=candidate.uncertainty,
        merged_from=candidate.merged_from,
    )


def _record(candidate: WallCandidate) -> dict[str, Any]:
    """The ``wall_candidates.json`` entry for one candidate."""
    return {
        "id": candidate.id,
        "geometry": {
            "start": {"x": round(candidate.start[0], 2), "y": round(candidate.start[1], 2)},
            "end": {"x": round(candidate.end[0], 2), "y": round(candidate.end[1], 2)},
            "polygon": [{"x": round(x, 2), "y": round(y, 2)} for x, y in candidate.polygon],
            "bbox": {
                "x": candidate.bbox[0],
                "y": candidate.bbox[1],
                "width": candidate.bbox[2],
                "height": candidate.bbox[3],
            },
        },
        "length": round(candidate.length, 2),
        "angle_deg": round(candidate.angle_deg, 3),
        "orientation": candidate.orientation,
        "estimated_thickness": round(candidate.estimated_thickness, 2),
        "thickness_consistency": candidate.thickness_consistency,
        "straightness": candidate.straightness,
        "continuity": candidate.continuity,
        "edge_pairing": candidate.edge_pairing,
        "axis_support": candidate.axis_support,
        "confidence": candidate.confidence,
        "confidence_evidence": candidate.confidence_evidence,
        "evidence": candidate.evidence,
        "uncertainty": candidate.uncertainty,
        "merged_from": candidate.merged_from,
    }


def _build_candidates_document(
    candidates: list[WallCandidate],
    original: Image.Image,
    foreground_mask: Image.Image,
) -> dict[str, Any]:
    """The ``wall_candidates.json`` document."""
    return {
        "phase": {"number": 3, "name": "wall_candidates", "step": "c"},
        "image": {
            "width": original.width,
            "height": original.height,
            "normalized_width": foreground_mask.width,
            "normalized_height": foreground_mask.height,
            "coordinate_space": "original",
        },
        "candidates": [_record(candidate) for candidate in candidates],
    }


def _build_report(
    candidates: list[WallCandidate],
    original: Image.Image,
    foreground_mask: Image.Image,
    diagonal: float,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    """The ``candidate_report.json`` document."""
    by_orientation: dict[str, int] = {}
    for candidate in candidates:
        by_orientation[candidate.orientation] = (
            by_orientation.get(candidate.orientation, 0) + 1
        )
    paired = sum(1 for c in candidates if c.edge_pairing["paired"])
    return {
        "phase": {"number": 3, "name": "wall_candidates", "step": "c"},
        "steps": {
            "c": "geometric stroke-candidate identification: collinear edge "
            "consolidation, parallel-edge pairing, duplicate merging and "
            "per-candidate measurement; candidates are proposals only "
            "(nothing is classified and nothing is discarded)",
        },
        "summary": {
            "total_candidates": len(candidates),
            "paired_edge_candidates": paired,
            "single_edge_candidates": len(candidates) - paired,
            "merged_source_count": sum(len(c.merged_from) for c in candidates),
            "by_orientation": by_orientation,
            "mean_confidence": round(
                sum(c.confidence for c in candidates) / len(candidates), 4
            )
            if candidates
            else 0.0,
            "low_confidence_candidates": sum(1 for c in candidates if c.confidence < 0.5),
        },
        "input": {
            "original_width": original.width,
            "original_height": original.height,
            "normalized_width": foreground_mask.width,
            "normalized_height": foreground_mask.height,
            "coordinate_space": "original",
            "working_diagonal_px": round(diagonal, 2),
        },
        "parameters": {
            "canny": [CANNY_LOW, CANNY_HIGH],
            "hough_threshold": f"max({HOUGH_THRESHOLD_FLOOR}, min_segment_length)",
            "min_segment_length": f"{MIN_SEGMENT_LENGTH_FRACTION} * diagonal (floor {MIN_SEGMENT_LENGTH_FLOOR_PX} px)",
            "hough_max_line_gap": f"{HOUGH_MAX_LINE_GAP_FRACTION} * diagonal",
            "edge_merge_angle_tol_deg": EDGE_MERGE_ANGLE_TOL_DEG,
            "edge_merge_dist": f"{EDGE_MERGE_DIST_FRACTION} * diagonal (floor {EDGE_MERGE_DIST_FLOOR_PX} px)",
            "edge_merge_bridge": f"gap up to {EDGE_MERGE_BRIDGE_FRACTION} * diagonal merged only when ink coverage along the gap is >= {EDGE_MERGE_BRIDGE_COVERAGE}",
            "pair_angle_tol_deg": PAIR_ANGLE_TOL_DEG,
            "pair_distance_px": [PAIR_MIN_DISTANCE_PX, f"{PAIR_MAX_DISTANCE_FRACTION} * diagonal"],
            "pair_min_overlap": PAIR_MIN_OVERLAP,
            "candidate_merge_angle_tol_deg": CANDIDATE_MERGE_ANGLE_TOL_DEG,
            "candidate_merge_dist": f"{CANDIDATE_MERGE_DIST_FRACTION} * diagonal (floor {CANDIDATE_MERGE_DIST_FLOOR_PX} px)",
            "candidate_merge_gap": f"{CANDIDATE_MERGE_GAP_FRACTION} * diagonal",
            "candidate_merge_thickness_ratio": CANDIDATE_MERGE_THICKNESS_RATIO,
            "confidence_weights": CONFIDENCE_WEIGHTS,
            "scale_from_normalized": [round(scale_x, 6), round(scale_y, 6)],
        },
        "limitations": [
            "Candidates are geometric proposals; none of them is a confirmed "
            "wall and none is classified as furniture or annotation here.",
            "Detection works on straight strokes: curved walls, door swing "
            "arcs and hatching appear only through their straight fragments.",
            "Furniture edges, text baselines and dimension lines can and do "
            "produce candidates — that is expected at this step; the "
            "classification step decides their nature later.",
            "Collinear fragments are bridged only where the gap itself "
            "carries ink; a larger ink-free interruption (door gap) splits "
            "one physical stroke into several candidates.",
            "Confidence reflects geometric coherence only (edge pairing, "
            "continuity, straightness, thickness consistency); no semantic "
            "or accuracy claim is made without labelled ground truth.",
            "The Step B structural masks are recorded as supporting evidence "
            "only and never drive detection.",
        ],
    }


def _draw_overlay(base_rgb: np.ndarray, candidates: list[WallCandidate]) -> Image.Image:
    """Centrelines and stroke bands on a copy of the original image."""
    canvas = base_rgb.copy()
    height, width = canvas.shape[:2]
    thickness = max(LINE_THICKNESS_MIN_PX, round(LINE_THICKNESS_FRACTION * min(height, width)))
    font_scale = max(FONT_SCALE_MIN, FONT_SCALE_FRACTION * min(height, width))

    for candidate in candidates:
        color = PAIRED_COLOR if candidate.edge_pairing["paired"] else UNPAIRED_COLOR
        points = np.array(candidate.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, color, max(1, thickness // 2), cv2.LINE_AA)
        cv2.line(
            canvas,
            (round(candidate.start[0]), round(candidate.start[1])),
            (round(candidate.end[0]), round(candidate.end[1])),
            color,
            thickness,
            cv2.LINE_AA,
        )
        mid = (
            round((candidate.start[0] + candidate.end[0]) / 2.0) + LABEL_GAP_PX,
            round((candidate.start[1] + candidate.end[1]) / 2.0) - LABEL_GAP_PX,
        )
        _draw_label(canvas, candidate.id, mid, color, font_scale, max(1, thickness // 2))

    paired = sum(1 for c in candidates if c.edge_pairing["paired"])
    _draw_legend(
        canvas,
        [
            f"Step C wall candidates (proposals, not confirmed walls)  ·  {len(candidates)}",
            f"paired edges: {paired}   ·   single edge: {len(candidates) - paired}",
        ],
        font_scale,
    )
    return Image.fromarray(canvas)


def _draw_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x = max(0, min(origin[0], canvas.shape[1] - text_w - 1))
    y = max(text_h + 1, min(origin[1], canvas.shape[0] - baseline - 1))
    cv2.rectangle(
        canvas,
        (x - 1, y - text_h - 1),
        (x + text_w + 1, y + baseline),
        OVERLAY_LEGEND_BG,
        -1,
    )
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )


def _draw_legend(
    canvas: np.ndarray, rows: list[str], font_scale: float
) -> None:
    pad = LABEL_GAP_PX
    widths = [
        cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for row in rows
    ]
    line_height = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][1] + pad
    box_w = max(widths) + 2 * pad
    box_h = line_height * len(rows) + pad
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), OVERLAY_LEGEND_BG, -1)
    for index, row in enumerate(rows):
        cv2.putText(
            canvas,
            row,
            (pad, line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            OVERLAY_LEGEND_TEXT,
            1,
            cv2.LINE_AA,
        )


# --- standalone runner -------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Generate the Step C artifacts for one floor plan.

    Run from the project root:

        python -m floorplanto3d.processing.wall_candidates IMAGE [MASKS_DIR] [OUT_DIR]

    ``MASKS_DIR`` is an existing Step A/B artifact directory (for example
    ``artifacts/parse2d/phase_2/walls/actual``); when omitted the Step A/B
    masks are recomputed with the existing Phase 1 and wall-detection code.
    """
    parser = argparse.ArgumentParser(description="Phase 2, Step C wall candidates")
    parser.add_argument("image", type=Path, help="original floor plan image")
    parser.add_argument(
        "masks_dir",
        type=Path,
        nargs="?",
        default=None,
        help="existing Step A/B artifact directory (optional)",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=Path("artifacts/parse2d/phase_3"),
        help="output directory (default: artifacts/parse2d/phase_3)",
    )
    args = parser.parse_args(argv)

    if args.masks_dir is not None:
        foreground = open_image(args.masks_dir / FOREGROUND_MASK_FILENAME)
        horizontal = open_image(args.masks_dir / HORIZONTAL_STRUCTURES_FILENAME)
        vertical = open_image(args.masks_dir / VERTICAL_STRUCTURES_FILENAME)
        combined = open_image(args.masks_dir / COMBINED_STRUCTURES_FILENAME)
    else:
        from floorplanto3d.processing.normalize import normalize_image

        normalization = normalize_image(args.image)
        walls = detect_walls(normalization.normalized, normalization.original)
        foreground = walls.foreground_mask
        horizontal = walls.horizontal
        vertical = walls.vertical
        combined = walls.combined

    result = detect_candidates(args.image, foreground, horizontal, vertical, combined)
    outputs = save_artifacts(result, args.out_dir)
    for name, path in outputs.items():
        print(f"wrote {path}  ({name})")
    summary = result.report["summary"]
    print(
        f"{summary['total_candidates']} candidates "
        f"({summary['paired_edge_candidates']} paired, "
        f"{summary['single_edge_candidates']} single-edge)"
    )


if __name__ == "__main__":
    main()
