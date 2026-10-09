"""Wall extraction: stroke polygons to centrelines.

The pipeline is:

1. Find connected components of ink (each wall stroke is one blob).
2. Simplify each blob's contour into a polygon.
3. Treat every polygon edge as a candidate wall centreline.
4. Merge collinear candidates that belong to the same wall run.
5. Measure thickness by sampling ink across each centreline.

Because doors and windows are *gaps* in a wall rather than separate marks,
they are recovered later in :mod:`floorplanto3d.processing.openings` by
projecting the ink mask onto each wall.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from floorplanto3d.models.geometry import Point2D, classify_orientation
from floorplanto3d.models.wall import Wall

logger = logging.getLogger(__name__)


class CandidateDecision(StrEnum):
    """Internal wall-candidate decision used for diagnostics."""

    ACCEPTED_WALL = "accepted_wall"
    POSSIBLE_WALL = "possible_wall"
    REJECTED_NON_WALL = "rejected_non_wall"


@dataclass
class StrokeCandidate:
    """A candidate centreline straight from one contour edge."""

    start: Point2D
    end: Point2D
    stroke_id: int
    confidence: float = 1.0
    half_width: float = 1.0

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def angle_deg(self) -> float:
        return math.degrees(
            math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
        ) % 180.0

    @property
    def orientation(self) -> str:
        return classify_orientation(self.end.x - self.start.x, self.end.y - self.start.y)

    def key(self) -> tuple[int, int]:
        """Quantised orientation bucket for collinear grouping."""
        angle = self.angle_deg
        # Snap near-axis-aligned strokes so that rounding noise does not split
        # the same physical wall into several orientation groups.
        if angle < 12.5 or angle >= 172.5:
            angle = 0.0
        elif 77.5 <= angle < 102.5:
            angle = 90.0
        return (round(angle / 5.0), self.orientation == "vertical")


@dataclass
class CandidateDiagnostic:
    """Lightweight explanation of a candidate decision."""

    decision: CandidateDecision
    reason: str
    length: float
    thickness: float | None = None
    continuity: float = 0.0
    parallel_edge_score: float = 0.0
    connectivity: float = 0.0
    start: Point2D | None = None
    end: Point2D | None = None


@dataclass
class WallExtractionResult:
    """Walls plus the candidate-level diagnostics behind them."""

    walls: list[Wall]
    diagnostics: list[CandidateDiagnostic]
    raw_count: int
    normalized_count: int


def _estimate_stroke_width(binary: np.ndarray) -> float:
    """Robustly estimate stroke width via the distance transform.

    The plain median of the distance transform is biased low by the long thin
    tail produced where strokes meet at corners. Taking the 75th percentile
    instead tracks the body of the distribution, which is the wall width.
    """
    inverted = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    samples = inverted[inverted > 0]
    if samples.size == 0:
        return 1.0
    # Each pixel's distance to the nearest background is ~half its stroke width.
    return max(1.0, float(np.percentile(samples, 75)) * 2.0)


def _unit_vectors(angle_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return direction and left-normal unit vectors for an angle."""
    radians = math.radians(angle_deg)
    ux, uy = math.cos(radians), math.sin(radians)
    return (ux, uy), (-uy, ux)


def _project(point: Point2D, vector: tuple[float, float]) -> float:
    return point.x * vector[0] + point.y * vector[1]


def _axis_interval(segment: StrokeCandidate) -> tuple[float, float]:
    unit, _ = _unit_vectors(segment.angle_deg)
    a = _project(segment.start, unit)
    b = _project(segment.end, unit)
    return (min(a, b), max(a, b))


def _normal_offset(segment: StrokeCandidate, angle_deg: float | None = None) -> float:
    _, normal = _unit_vectors(segment.angle_deg if angle_deg is None else angle_deg)
    return (_project(segment.start, normal) + _project(segment.end, normal)) / 2.0


def _point_from_axis(
    along: float, normal_offset: float, angle_deg: float
) -> Point2D:
    unit, normal = _unit_vectors(angle_deg)
    return Point2D(
        x=unit[0] * along + normal[0] * normal_offset,
        y=unit[1] * along + normal[1] * normal_offset,
    )


def _collinear_merge(
    candidates: list[StrokeCandidate], tolerance: float, gap_tolerance: float
) -> list[StrokeCandidate]:
    """Merge segments that share a direction and lie on the same infinite line."""
    groups: dict[tuple[int, int], list[StrokeCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.key(), []).append(candidate)

    merged: list[StrokeCandidate] = []

    for group in groups.values():
        remaining = sorted(group, key=lambda segment: (_normal_offset(segment), *_axis_interval(segment)))
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                for other in list(remaining):
                    for member in cluster:
                        if _mergeable_collinear(member, other, tolerance, gap_tolerance):
                            cluster.append(other)
                            remaining.remove(other)
                            changed = True
                            break
                    else:
                        continue
                    break
            merged.append(_collapse_cluster(cluster, gap_tolerance))

    return merged


def _mergeable_collinear(
    a: StrokeCandidate, b: StrokeCandidate, tolerance: float, gap_tolerance: float
) -> bool:
    """Whether two segments are close enough on the same run to merge."""
    if not _on_same_line(a, b, tolerance):
        return False

    a0, a1 = _axis_interval(a)
    b0, b1 = _axis_interval(b)
    gap = max(a0, b0) - min(a1, b1)
    return gap <= gap_tolerance


def _on_same_line(a: StrokeCandidate, b: StrokeCandidate, tolerance: float) -> bool:
    """Whether two segments lie on the same infinite line within tolerance."""
    angle_diff = abs(a.angle_deg - b.angle_deg)
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    if angle_diff > 5.0:
        return False

    dx, dy = math.cos(math.radians(a.angle_deg)), math.sin(math.radians(a.angle_deg))
    distance = abs((b.start.x - a.start.x) * dy - (b.start.y - a.start.y) * dx)
    return distance <= tolerance


def _collapse_cluster(
    cluster: list[StrokeCandidate], gap_tolerance: float
) -> StrokeCandidate:
    """Reduce a collinear cluster to one segment spanning its extremes."""
    points: list[Point2D] = []
    for candidate in cluster:
        points.extend([candidate.start, candidate.end])

    best = (0.0, points[0], points[1])
    for i, p0 in enumerate(points):
        for p1 in points[i + 1 :]:
            distance = p0.distance_to(p1)
            if distance > best[0]:
                best = (distance, p0, p1)

    # Order endpoints along the dominant direction.
    start, end = best[1], best[2]
    if end.x < start.x or (end.x == start.x and end.y < start.y):
        start, end = end, start

    return StrokeCandidate(
        start=start,
        end=end,
        stroke_id=cluster[0].stroke_id,
        confidence=min(member.confidence for member in cluster),
        half_width=float(np.median([member.half_width for member in cluster])),
    )


def _dedupe_overlapping(
    segments: list[StrokeCandidate], overlap_ratio: float
) -> list[StrokeCandidate]:
    """Drop segments that are almost entirely contained in another."""
    kept: list[StrokeCandidate] = []

    for segment in sorted(segments, key=lambda s: s.length, reverse=True):
        redundant = False
        for existing in kept:
            shared = _shared_length(segment, existing)
            if shared / segment.length > overlap_ratio:
                redundant = True
                break
        if not redundant:
            kept.append(segment)

    return kept


def _shared_length(a: StrokeCandidate, b: StrokeCandidate) -> float:
    """Approximate length of the 1-D overlap between two collinear segments.

    Segments on different parallel lines share no length at all, so collinearity
    is checked first; without this, a wall and the opposite wall of the same
    room would be treated as duplicates of one another.
    """
    if not _on_same_line(a, b, 1.0):
        return 0.0

    a0, a1 = _axis_interval(a)
    b0, b1 = _axis_interval(b)
    return max(0.0, min(a1, b1) - max(a0, b0))


def measure_thickness(
    binary: np.ndarray, start: Point2D, end: Point2D, max_probe: float = 60.0
) -> float | None:
    """Measure wall thickness by probing perpendicular to the centreline.

    Walks outward from the centreline along the normal and records the extent
    of contiguous ink. Returns ``None`` when the probe finds no ink or the run
    is implausibly wide, which indicates the line is not on a wall stroke.
    """
    length = start.distance_to(end)
    if length < 1e-6:
        return None

    ux = (end.x - start.x) / length
    uy = (end.y - start.y) / length
    nx, ny = -uy, ux

    center = start.midpoint(end)
    height, width = binary.shape[:2]
    total: list[float] = []

    # Sample along the wall so we are not fooled by a local artefact.
    samples = max(3, min(int(length / 8), 25))
    for i in range(samples):
        t = (i + 0.5) / samples
        px = center.x + (t - 0.5) * (end.x - start.x)
        py = center.y + (t - 0.5) * (end.y - start.y)

        for sign in (1.0, -1.0):
            run = 0.0
            step = 1.0
            while run < max_probe:
                qx = int(round(px + sign * nx * (run + step)))
                qy = int(round(py + sign * ny * (run + step)))
                if not (0 <= qx < width and 0 <= qy < height):
                    break
                if binary[qy, qx] == 0:
                    break
                run += step
            if run > 0:
                total.append(run)

    if not total:
        return None

    thickness = float(np.median(total)) * 2.0
    # A wall thicker than this is almost certainly a fill or a furniture block.
    if thickness <= 0 or thickness > max_probe:
        return None
    return round(thickness, 2)


def detect_segments(
    binary: np.ndarray,
    *,
    min_length: float = 30.0,
    max_gap: float = 6.0,
    threshold: float = 60.0,
) -> list[StrokeCandidate]:
    """Detect axis-aligned wall runs with the probabilistic Hough transform.

    Floor plans are overwhelmingly made of horizontal and vertical strokes, and
    a Hough transform recovers them directly from the ink mask. This is far
    more reliable than reading centrelines off contour edges, because a
    connected building outline yields one contour that wraps every wall, and
    because a thick stroke's contour edges are its two *faces*, not its centre.

    Args:
        binary: Binary ink mask (255 = ink).
        min_length: Shortest run accepted as a wall, in pixels.
        max_gap: Gap bridged when joining collinear runs.
        threshold: Hough accumulator threshold, in pixels.

    Returns:
        Candidate segments as raw detections.
    """
    width = _estimate_stroke_width(binary)

    raw = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(threshold),
        minLineLength=int(min_length),
        maxLineGap=int(max_gap),
    )
    if raw is None:
        return []

    segments: list[StrokeCandidate] = []
    for entry in raw:
        x1, y1, x2, y2 = (float(v) for v in entry.reshape(-1)[:4])
        start = Point2D(x=x1, y=y1)
        end = Point2D(x=x2, y=y2)
        if start.distance_to(end) < min_length:
            continue
        # Keep the two faces of a thick stroke out; centring happens later.
        segments.append(
            StrokeCandidate(
                start=start, end=end, stroke_id=-1, half_width=width / 2.0
            )
        )

    logger.info("Hough transform produced %d raw segments", len(segments))
    return segments


def centre_on_stroke(
    binary: np.ndarray, segment: StrokeCandidate
) -> StrokeCandidate:
    """Shift a segment onto the middle of the ink it lies on.

    A Hough line locks onto a wall's edge. Stepping perpendicular until the
    local ink run is centred moves it onto the true centreline.
    """
    length = segment.length
    if length < 1e-6:
        return segment

    ux = (segment.end.x - segment.start.x) / length
    uy = (segment.end.y - segment.start.y) / length
    nx, ny = -uy, ux
    center = segment.start.midpoint(segment.end)

    best_symmetry = float("inf")
    limit = max(4.0, segment.half_width * 2.0 + 2.0)

    # Snap the answer to whole pixels: a 0.5px step lets a long wall accumulate
    # a visible slant, which then stops it closing against its neighbours.
    best_offset = 0.0
    offset = -limit
    while offset <= limit:
        left, right = _ink_runs(binary, center, nx * offset, ny * offset, ux, uy, length)
        if left > 0 and right > 0:
            symmetry = abs(left - right)
            if symmetry < best_symmetry:
                best_symmetry = symmetry
                best_offset = float(round(offset))
        offset += 0.5

    if best_symmetry == float("inf"):
        return segment

    return StrokeCandidate(
        start=Point2D(
            x=segment.start.x + nx * best_offset,
            y=segment.start.y + ny * best_offset,
        ),
        end=Point2D(
            x=segment.end.x + nx * best_offset,
            y=segment.end.y + ny * best_offset,
        ),
        stroke_id=segment.stroke_id,
        half_width=segment.half_width,
    )


def _ink_runs(
    binary: np.ndarray,
    center: Point2D,
    dx: float,
    dy: float,
    ux: float,
    uy: float,
    length: float,
    samples: int = 15,
    limit: float = 40.0,
) -> tuple[float, float]:
    """Measure contiguous ink either side of a line, returning (left, right)."""
    height, width = binary.shape[:2]
    left_runs: list[float] = []
    right_runs: list[float] = []

    for i in range(samples):
        t = (i + 0.5) / samples
        px = center.x + ux * (t - 0.5) * length + dx
        py = center.y + uy * (t - 0.5) * length + dy

        for sign, bucket in ((1.0, left_runs), (-1.0, right_runs)):
            run = 0.0
            while run < limit:
                qx = int(round(px + sign * (-uy) * run))
                qy = int(round(py + sign * ux * run))
                if not (0 <= qx < width and 0 <= qy < height):
                    break
                if binary[qy, qx] == 0:
                    break
                run += 1.0
            if run > 0:
                bucket.append(run)

    left = float(np.median(left_runs)) if left_runs else 0.0
    right = float(np.median(right_runs)) if right_runs else 0.0
    return left, right


def _snap_to_axis(segment: StrokeCandidate, tolerance: float = 8.0) -> StrokeCandidate:
    """Snap a near-axis-aligned segment onto its exact axis.

    A Hough segment that runs along a horizontal wall can still pick up a few
    pixels of drift over its length, and merging several such segments makes
    the drift worse. A wall that drifts never closes cleanly against its
    neighbours, so the rooms behind it silently fail to polygonise.
    """
    if segment.length < 1e-6:
        return segment

    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    orientation = classify_orientation(dx, dy)
    if orientation == "diagonal":
        return segment

    if orientation == "horizontal":
        # Small residual slope counts as noise, not as a real angle.
        if abs(dy) > abs(dx) * (tolerance / 100.0):
            return segment
        y = round((segment.start.y + segment.end.y) / 2.0)
        return StrokeCandidate(
            start=Point2D(x=segment.start.x, y=y),
            end=Point2D(x=segment.end.x, y=y),
            stroke_id=segment.stroke_id,
            confidence=segment.confidence,
            half_width=segment.half_width,
        )

    if abs(dx) > abs(dy) * (tolerance / 100.0):
        return segment
    x = round((segment.start.x + segment.end.x) / 2.0)
    return StrokeCandidate(
        start=Point2D(x=x, y=segment.start.y),
        end=Point2D(x=x, y=segment.end.y),
        stroke_id=segment.stroke_id,
        confidence=segment.confidence,
        half_width=segment.half_width,
    )


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, 180.0 - diff)


def _overlap_ratio(a: StrokeCandidate, b: StrokeCandidate) -> float:
    a0, a1 = _axis_interval(a)
    b0, b1 = _axis_interval(b)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = max(1e-6, min(a1 - a0, b1 - b0))
    return overlap / shorter


def _continuity_score(
    binary: np.ndarray, segment: StrokeCandidate, band: int = 1
) -> float:
    """Fraction of samples along a segment that still touch foreground ink."""
    length = segment.length
    if length < 1e-6:
        return 0.0

    height, width = binary.shape[:2]
    ux = (segment.end.x - segment.start.x) / length
    uy = (segment.end.y - segment.start.y) / length
    nx, ny = -uy, ux
    samples = max(12, min(int(length / 3), 160))
    hits = 0

    for index in range(samples):
        t = index / max(samples - 1, 1)
        x = segment.start.x + ux * length * t
        y = segment.start.y + uy * length * t
        found = False
        for offset in range(-band, band + 1):
            qx = int(round(x + nx * offset))
            qy = int(round(y + ny * offset))
            if 0 <= qx < width and 0 <= qy < height and binary[qy, qx] != 0:
                found = True
                break
        if found:
            hits += 1

    return hits / samples


def _band_fill_fraction(
    binary: np.ndarray,
    segment_a: StrokeCandidate,
    segment_b: StrokeCandidate,
    *,
    along_start: float | None = None,
    along_end: float | None = None,
) -> float:
    """Ink fill fraction in the strip between two parallel candidates."""
    angle = segment_a.angle_deg
    unit, _ = _unit_vectors(angle)
    n0 = _normal_offset(segment_a, angle)
    n1 = _normal_offset(segment_b, angle)
    if n1 < n0:
        n0, n1 = n1, n0

    a0, a1 = _axis_interval(segment_a)
    b0, b1 = _axis_interval(segment_b)
    start = max(a0, b0) if along_start is None else along_start
    end = min(a1, b1) if along_end is None else along_end
    if end <= start or n1 <= n0:
        return 0.0

    height, width = binary.shape[:2]
    along_samples = max(12, min(int(end - start), 120))
    across_samples = max(3, min(int(round(n1 - n0)) + 1, 40))
    ink = 0
    total = 0

    for i in range(along_samples):
        along = start + (end - start) * (i + 0.5) / along_samples
        for j in range(across_samples):
            normal = n0 + (n1 - n0) * (j + 0.5) / across_samples
            point = _point_from_axis(along, normal, angle)
            qx, qy = int(round(point.x)), int(round(point.y))
            if 0 <= qx < width and 0 <= qy < height:
                total += 1
                if binary[qy, qx] != 0:
                    ink += 1

    return ink / total if total else 0.0


def _component_fill_ratio(
    labels: np.ndarray, stats: np.ndarray, segment: StrokeCandidate
) -> float:
    """Fill ratio of the connected component most sampled by a segment."""
    length = segment.length
    if length < 1e-6:
        return 0.0

    height, width = labels.shape[:2]
    ux = (segment.end.x - segment.start.x) / length
    uy = (segment.end.y - segment.start.y) / length
    samples = max(8, min(int(length / 4), 80))
    counts: dict[int, int] = {}
    for index in range(samples):
        t = index / max(samples - 1, 1)
        qx = int(round(segment.start.x + ux * length * t))
        qy = int(round(segment.start.y + uy * length * t))
        if not (0 <= qx < width and 0 <= qy < height):
            continue
        label = int(labels[qy, qx])
        if label:
            counts[label] = counts.get(label, 0) + 1

    if not counts:
        return 0.0

    label = max(counts, key=counts.get)
    x, y, w, h, area = stats[label]
    box_area = max(1, int(w) * int(h))
    return float(area) / box_area


def _resolve_parallel_edges(
    binary: np.ndarray,
    segments: list[StrokeCandidate],
    *,
    min_overlap_ratio: float = 0.65,
    min_band_fill: float = 0.55,
) -> tuple[list[StrokeCandidate], list[StrokeCandidate], list[CandidateDiagnostic]]:
    """Collapse opposite faces of filled thick strokes into centrelines."""
    if not segments:
        return [], [], []

    stroke_width = _estimate_stroke_width(binary)
    min_separation = max(3.0, stroke_width * 0.35)
    max_separation = max(8.0, min(60.0, stroke_width * 4.0 + 8.0))
    used = [False] * len(segments)
    reconstructed: list[StrokeCandidate] = []
    diagnostics: list[CandidateDiagnostic] = []

    ordered = sorted(range(len(segments)), key=lambda i: segments[i].length, reverse=True)
    for i in ordered:
        if used[i]:
            continue
        segment = segments[i]
        best: tuple[float, int, float, float] | None = None

        for j in ordered:
            if i == j or used[j]:
                continue
            other = segments[j]
            if _angle_difference(segment.angle_deg, other.angle_deg) > 5.0:
                continue
            separation = abs(_normal_offset(segment) - _normal_offset(other, segment.angle_deg))
            if separation < min_separation or separation > max_separation:
                continue
            overlap = _overlap_ratio(segment, other)
            if overlap < min_overlap_ratio:
                continue
            fill = _band_fill_fraction(binary, segment, other)
            close_hollow_stroke = (
                separation <= max(16.0, stroke_width * 4.0)
                and overlap >= 0.85
                and fill >= 0.12
            )
            if fill < min_band_fill and not close_hollow_stroke:
                continue

            score = overlap * 0.55 + fill * 0.45
            if best is None or score > best[0]:
                best = (score, j, separation, fill)

        if best is None:
            continue

        _, j, separation, fill = best
        other = segments[j]
        used[i] = True
        used[j] = True

        angle = segment.angle_deg
        a0, a1 = _axis_interval(segment)
        b0, b1 = _axis_interval(other)
        along_start = max(a0, b0)
        along_end = min(a1, b1)
        center_offset = (
            _normal_offset(segment, angle) + _normal_offset(other, angle)
        ) / 2.0
        start = _point_from_axis(along_start, center_offset, angle)
        end = _point_from_axis(along_end, center_offset, angle)
        if end.x < start.x or (end.x == start.x and end.y < start.y):
            start, end = end, start

        reconstructed.append(
            StrokeCandidate(
                start=start,
                end=end,
                stroke_id=segment.stroke_id,
                confidence=min(1.0, max(0.0, best[0])),
                half_width=(separation if fill >= min_band_fill else separation + 2.0 * stroke_width)
                / 2.0,
            )
        )
        diagnostics.append(
            CandidateDiagnostic(
                decision=CandidateDecision.ACCEPTED_WALL,
                reason="parallel filled edge pair",
                length=start.distance_to(end),
                thickness=reconstructed[-1].half_width * 2.0,
                continuity=fill,
                parallel_edge_score=best[0],
            )
        )

    leftovers = [segment for index, segment in enumerate(segments) if not used[index]]
    return reconstructed, leftovers, diagnostics


def _validate_standalone_candidates(
    binary: np.ndarray,
    segments: list[StrokeCandidate],
    *,
    min_length: float,
    context_has_walls: bool = False,
    context_length_ratio: float = 0.30,
) -> tuple[list[StrokeCandidate], list[CandidateDiagnostic]]:
    """Keep only standalone runs that look like solid structural strokes."""
    if not segments:
        return [], []

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    image_diagonal = float(np.hypot(binary.shape[1], binary.shape[0]))
    min_structural_length = max(min_length, image_diagonal * 0.08)
    accepted: list[StrokeCandidate] = []
    diagnostics: list[CandidateDiagnostic] = []

    for segment in segments:
        thickness = measure_thickness(binary, segment.start, segment.end)
        band = max(1, int(round((thickness or 2.0) / 2.0)))
        continuity = _continuity_score(binary, segment, band=min(band, 4))
        component_fill = _component_fill_ratio(labels, stats, segment)

        solid_component = component_fill >= 0.35
        structural_context = (
            context_has_walls and segment.length >= image_diagonal * context_length_ratio
        )

        if (
            segment.length >= min_structural_length
            and thickness is not None
            and thickness >= 3.5
            and continuity >= 0.78
            and (solid_component or structural_context)
        ):
            accepted.append(segment)
            diagnostics.append(
                CandidateDiagnostic(
                    decision=CandidateDecision.POSSIBLE_WALL,
                    reason="solid standalone stroke",
                    length=segment.length,
                    thickness=thickness,
                    continuity=continuity,
                )
            )
            continue

        diagnostics.append(
            CandidateDiagnostic(
                decision=CandidateDecision.REJECTED_NON_WALL,
                reason="isolated thin or outline-like stroke",
                length=segment.length,
                thickness=thickness,
                continuity=continuity,
            )
        )

    return accepted, diagnostics


def _has_candidate_wall_network(
    binary: np.ndarray, segments: list[StrokeCandidate], min_length: float
) -> bool:
    """Whether raw candidates already form a plausible wall graph."""
    if len(segments) < 4:
        return False

    image_diagonal = float(np.hypot(binary.shape[1], binary.shape[0]))
    structural_min = max(min_length, image_diagonal * 0.12)
    long_segments = [segment for segment in segments if segment.length >= structural_min]
    if len(long_segments) < 4:
        return False

    horizontal = sum(1 for segment in long_segments if segment.orientation == "horizontal")
    vertical = sum(1 for segment in long_segments if segment.orientation == "vertical")
    diagonal = len(long_segments) - horizontal - vertical
    return (horizontal >= 2 and vertical >= 2) or (
        diagonal >= 1 and horizontal + vertical + diagonal >= 4
    )


def extract_walls_detailed(
    binary: np.ndarray,
    *,
    min_edge_length: float = 30.0,
    merge_tolerance: float = 4.0,
    gap_tolerance: float = 10.0,
    overlap_ratio: float = 0.80,
) -> WallExtractionResult:
    """Extract walls together with per-candidate diagnostics.

    Same pipeline as :func:`extract_walls`, but every candidate carries its
    decision, scores and rejection reason so callers can report why a segment
    was accepted or dropped.

    Returns:
        A :class:`WallExtractionResult`.
    """
    detected = detect_segments(binary, min_length=min_edge_length)
    if not detected:
        return WallExtractionResult(walls=[], diagnostics=[], raw_count=0, normalized_count=0)

    # Merge collinear runs *before* centring. A door or window splits one wall
    # into several Hough segments that each sit on their own fragment's edge;
    # centring them individually would leave the wall as a band several pixels
    # thick and would hide the opening from the gap detector.
    merged = _collinear_merge(detected, merge_tolerance, gap_tolerance)
    merged = [_snap_to_axis(segment) for segment in merged]

    paired, leftovers, pair_diagnostics = _resolve_parallel_edges(binary, merged)
    has_network_context = _has_candidate_wall_network(binary, merged, min_edge_length)
    context_length_ratio = 0.30 if paired else 0.12
    standalone, standalone_diagnostics = _validate_standalone_candidates(
        binary,
        leftovers,
        min_length=min_edge_length,
        context_has_walls=bool(paired) or has_network_context,
        context_length_ratio=context_length_ratio,
    )

    candidates = paired + [centre_on_stroke(binary, segment) for segment in standalone]

    deduped = _dedupe_overlapping(candidates, overlap_ratio)
    image_diagonal = float(np.hypot(binary.shape[1], binary.shape[0]))
    scale_aware_min_length = max(min_edge_length, image_diagonal * 0.08)
    deduped = [s for s in deduped if s.length >= scale_aware_min_length]

    # Order deterministically: vertical walls left to right, then horizontal.
    deduped.sort(
        key=lambda s: (s.orientation != "vertical", round(s.start.x), round(s.start.y))
    )

    walls: list[Wall] = []
    for index, segment in enumerate(deduped):
        thickness = measure_thickness(binary, segment.start, segment.end)
        if thickness is None and segment.half_width > 1.0:
            thickness = round(segment.half_width * 2.0, 2)
        walls.append(
            Wall(
                id=f"wall_{index + 1:03d}",
                start=segment.start,
                end=segment.end,
                thickness=thickness,
                confidence=segment.confidence,
            )
        )

    logger.info(
        "Extracted %d walls (median thickness %s, paired %d, standalone %d, rejected %d)",
        len(walls),
        _median_thickness(walls),
        len(pair_diagnostics),
        len(standalone),
        sum(
            1
            for diagnostic in standalone_diagnostics
            if diagnostic.decision == CandidateDecision.REJECTED_NON_WALL
        ),
    )
    return WallExtractionResult(
        walls=walls,
        diagnostics=pair_diagnostics + standalone_diagnostics,
        raw_count=len(detected),
        normalized_count=len(merged),
    )


def extract_walls(
    binary: np.ndarray,
    *,
    min_edge_length: float = 30.0,
    merge_tolerance: float = 4.0,
    gap_tolerance: float = 10.0,
    overlap_ratio: float = 0.80,
) -> list[Wall]:
    """Extract wall centrelines with measured thickness from a binary mask.

    Args:
        binary: Binary ink mask.
        min_edge_length: Minimum centreline length in pixels.
        merge_tolerance: Perpendicular tolerance for merging collinear runs.
        gap_tolerance: Smallest gap bridged when joining collinear runs.
        overlap_ratio: Fraction of a segment that must be shared to drop it.

    Returns:
        A list of :class:`Wall` objects with pixel coordinates.
    """
    return extract_walls_detailed(
        binary,
        min_edge_length=min_edge_length,
        merge_tolerance=merge_tolerance,
        gap_tolerance=gap_tolerance,
        overlap_ratio=overlap_ratio,
    ).walls


def _median_thickness(walls: list[Wall]) -> str:
    values = [w.thickness for w in walls if w.thickness]
    if not values:
        return "n/a"
    return f"{np.median(values):.1f}px"


def project_ink_profile(
    binary: np.ndarray, wall: Wall, samples: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Project the ink mask onto a wall's axis.

    Returns:
        ``(positions, has_ink)`` where ``positions`` are distances in pixels
        along the wall from ``wall.start`` and ``has_ink`` is a boolean array
        indicating wall material at each position. Door and window openings
        are exactly the runs where this is ``False``.
    """
    length = wall.length
    if length < 1e-6:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)

    if samples is None:
        samples = max(2, int(length))

    dx = (wall.end.x - wall.start.x) / length
    dy = (wall.end.y - wall.start.y) / length

    height, width = binary.shape[:2]
    positions = np.linspace(0.0, length, samples)
    has_ink = np.zeros(samples, dtype=bool)

    thickness = wall.thickness or 3.0
    # Search a band around the centreline, wide enough to tolerate imperfect
    # centrelines but tight enough not to pick up a parallel wall.
    band = max(2, int(round(thickness)))
    offsets = range(-band, band + 1)

    for index, position in enumerate(positions):
        cx = wall.start.x + dx * position
        cy = wall.start.y + dy * position
        nx, ny = -dy, dx

        found = False
        for offset in offsets:
            qx = int(round(cx + nx * offset))
            qy = int(round(cy + ny * offset))
            if 0 <= qx < width and 0 <= qy < height and binary[qy, qx] != 0:
                found = True
                break
        has_ink[index] = found

    return positions, has_ink
