"""Opening detection.

Doors and windows are gaps in a wall. Rather than trying to recognise the
symbol drawn on a plan (which varies by architect), this module projects the
ink mask onto each wall centreline and inspects where material is missing.

A missing run is classified by its measured length relative to the median
wall opening elsewhere in the same plan, falling back to the image scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.opening import Door, OpeningType, Window
from floorplanto3d.models.wall import Wall
from floorplanto3d.processing.walls import project_ink_profile

logger = logging.getLogger(__name__)

# In plan view a doorway is wider than a window is tall. Gaps therefore form two
# clusters separated by roughly this factor, and the largest gap is treated as
# the door reference.
DOOR_TO_WINDOW_LENGTH_RATIO = 1.2


@dataclass
class OpeningCandidate:
    """A measured gap along a wall, before classification."""

    wall_id: str
    start_distance: float
    end_distance: float
    gap_length: float

    @property
    def midpoint_distance(self) -> float:
        return (self.start_distance + self.end_distance) / 2.0


def find_gaps(
    binary: np.ndarray,
    walls: list[Wall],
    *,
    min_gap_length: float = 12.0,
    tolerance: float = 2.0,
) -> list[OpeningCandidate]:
    """Locate runs of missing ink along each wall.

    Args:
        binary: Binary ink mask.
        walls: Detected walls.
        min_gap_length: Ignore gaps shorter than this.
        tolerance: Permit this much ink noise before closing a gap.

    Returns:
        A list of :class:`OpeningCandidate` sorted by size, largest first.
    """
    candidates: list[OpeningCandidate] = []

    for wall in walls:
        positions, has_ink = project_ink_profile(binary, wall)
        if len(positions) == 0:
            continue

        # Convert the tolerance from pixels to samples once, then filter whole
        # runs: bridging sample-by-sample would leak across a genuine opening.
        sample = positions[1] - positions[0] if len(positions) > 1 else 1.0
        max_speckle = max(0, int(round(tolerance / max(sample, 1e-6))))
        smoothed = _drop_speckle(has_ink, max_speckle)

        for start_index, end_index in _false_runs(smoothed):
            gap_length = positions[end_index] - positions[start_index]
            if gap_length < min_gap_length:
                continue
            # A gap that consumes most of the wall means the wall itself simply
            # stops here; that is a detection artefact, not an opening.
            if gap_length > wall.length * 0.9:
                continue

            candidates.append(
                OpeningCandidate(
                    wall_id=wall.id,
                    start_distance=float(positions[start_index]),
                    end_distance=float(positions[end_index]),
                    gap_length=float(gap_length),
                )
            )

    candidates.sort(key=lambda c: c.gap_length, reverse=True)
    logger.info("Found %d opening gaps", len(candidates))
    return candidates


def _drop_speckle(has_ink: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Fill runs of missing ink shorter than ``max_gap_samples``.

    Filtering whole runs rather than individual samples is essential: bridging
    sample-by-sample would also bridge a genuine door gap and destroy it.
    """
    result = has_ink.copy()
    for start, end in _false_runs(result):
        if (end - start + 1) <= max_gap_samples:
            result[start : end + 1] = True
    return result


def _false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` index pairs of consecutive ``False``."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if not value and start is None:
            start = index
        elif value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def classify_openings(
    candidates: list[OpeningCandidate],
    walls: list[Wall],
    image_diagonal: float | None = None,
) -> tuple[list[Door], list[Window]]:
    """Split measured gaps into doors and windows.

    Every measured gap is classified; none are dropped. Thresholds come from
    the gaps of this same plan rather than from an absolute constant, so the
    result adapts to the drawing's own scale:

    * The widest gap is the door reference: a doorway is the largest gap a
      plan contains.
    * A gap counts as a window when it is at most ``1 / DOOR_TO_WINDOW_LENGTH_RATIO``
      of that reference.
    * Anything in between is still reported, as a window, so no information a
      3D engine needs to cut the wall is lost.

    ``image_diagonal`` is accepted for callers that have it but is not needed:
    plan-relative calibration is preferred over an assumed absolute size.
    """
    if not candidates:
        return [], []

    door_reference = max(candidate.gap_length for candidate in candidates)
    smallest = min(candidate.gap_length for candidate in candidates)

    # If every gap is essentially the same size, the plan gives no evidence to
    # separate doors from windows. Calling one a door and the rest windows
    # would be arbitrary, so they are all reported as doors: a doorway is the
    # more common break in a wall, and a mislabelled opening is far less
    # harmful to a 3D consumer than a mislabelled room.
    uniform = (door_reference - smallest) <= door_reference * 0.25
    window_threshold = door_reference / DOOR_TO_WINDOW_LENGTH_RATIO

    wall_by_id = {wall.id: wall for wall in walls}

    doors: list[Door] = []
    windows: list[Window] = []
    narrow = 0

    for candidate in candidates:
        wall = wall_by_id.get(candidate.wall_id)
        if wall is None:
            continue

        length = wall.length
        if length < 1e-6:
            continue
        ux = (wall.end.x - wall.start.x) / length
        uy = (wall.end.y - wall.start.y) / length

        start = Point2D(
            x=wall.start.x + ux * candidate.start_distance,
            y=wall.start.y + uy * candidate.start_distance,
        )
        end = Point2D(
            x=wall.start.x + ux * candidate.end_distance,
            y=wall.start.y + uy * candidate.end_distance,
        )

        if uniform or candidate.gap_length >= door_reference:
            doors.append(
                Door(
                    id=f"door_{len(doors) + 1:03d}",
                    wall_id=wall.id,
                    start=start,
                    end=end,
                )
            )
        else:
            # Everything else is a window: a gap narrow enough to be a window,
            # or one that is not window-shaped but is still a real break in
            # the wall that a 3D engine must cut.
            if candidate.gap_length < window_threshold:
                narrow += 1
            windows.append(
                Window(
                    id=f"window_{len(windows) + 1:03d}",
                    wall_id=wall.id,
                    start=start,
                    end=end,
                )
            )

    if narrow:
        logger.info("%d gaps narrower than window-sized, reported as windows", narrow)

    logger.info(
        "Classified openings: %d doors, %d windows (door reference %.1fpx)",
        len(doors),
        len(windows),
        door_reference,
    )
    return doors, windows


def extract_openings(
    binary: np.ndarray,
    walls: list[Wall],
    image_width: int,
    image_height: int,
    *,
    min_gap_length: float = 12.0,
) -> tuple[list[Door], list[Window]]:
    """Detect and classify doors and windows from wall gaps.

    Args:
        binary: Binary ink mask.
        walls: Detected walls.
        image_width: Source image width.
        image_height: Source image height.

    Returns:
        ``(doors, windows)``, each carrying the id of its host wall.
    """
    if not walls:
        return [], []

    candidates = find_gaps(binary, walls, min_gap_length=min_gap_length)
    diagonal = float(np.hypot(image_width, image_height))
    return classify_openings(candidates, walls, diagonal)


def openings_for_wall(openings: list[Door | Window], wall_id: str) -> list[Door | Window]:
    """Return the openings hosted by a given wall."""
    return [opening for opening in openings if opening.wall_id == wall_id]


__all__ = [
    "OpeningCandidate",
    "OpeningType",
    "classify_openings",
    "extract_openings",
    "find_gaps",
    "openings_for_wall",
]