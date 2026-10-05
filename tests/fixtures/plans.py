"""Synthetic floor plan fixtures.

Each generator draws a plan whose geometry is known exactly, so tests can
assert real accuracy rather than just that the pipeline produced output.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "blank_plan",
    "four_room_plan",
    "skewed_plan",
    "to_rgb",
    "two_room_plan",
    "window_plan",
]


def two_room_plan(
    width: int = 600, height: int = 400, thickness: int = 8, door_gap: int = 40
) -> np.ndarray:
    """Two rooms side by side separated by a wall containing a door gap.

    Ground truth: 5 walls, 2 rooms of 250x300px, 1 door 40px wide at y=200.
    """
    canvas = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (550, 350), 0, thickness)
    cv2.line(canvas, (300, 50), (300, 350), 0, thickness)

    gap_start = 180
    cv2.line(canvas, (300, gap_start), (300, gap_start + door_gap), 255, thickness)
    return canvas


def four_room_plan(width: int = 800, height: int = 600, thickness: int = 8) -> np.ndarray:
    """Four rooms in a 2x2 grid with a door punched in each internal wall.

    Ground truth: 6 walls, 4 rooms, 4 openings of 60px.
    """
    canvas = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 60), (740, 540), 0, thickness)
    cv2.line(canvas, (400, 60), (400, 540), 0, thickness)
    cv2.line(canvas, (60, 300), (740, 300), 0, thickness)

    cv2.line(canvas, (400, 150), (400, 210), 255, thickness)
    cv2.line(canvas, (400, 380), (400, 440), 255, thickness)
    cv2.line(canvas, (180, 300), (240, 300), 255, thickness)
    cv2.line(canvas, (560, 300), (620, 300), 255, thickness)
    return canvas


def window_plan(width: int = 600, height: int = 400, thickness: int = 8) -> np.ndarray:
    """One room with two short wall gaps and one long gap.

    Ground truth: 4 walls, 1 room, 1 door (~70px) and 2 windows (~30px).
    """
    canvas = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (80, 80), (520, 320), 0, thickness)

    cv2.line(canvas, (150, 80), (180, 80), 255, thickness)
    cv2.line(canvas, (380, 80), (410, 80), 255, thickness)
    cv2.line(canvas, (250, 320), (320, 320), 255, thickness)
    return canvas


def blank_plan(width: int = 400, height: int = 300) -> np.ndarray:
    """A blank page containing no geometry."""
    return np.full((height, width), 255, dtype=np.uint8)


def skewed_plan(
    width: int = 600, height: int = 400, thickness: int = 8, angle: float = 3.0
) -> np.ndarray:
    """A two-room plan rotated by a small angle, to exercise deskewing."""
    canvas = two_room_plan(width, height, thickness)
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = int(np.ceil(max(width, height) * 0.5))
    matrix[0, 2] += border - center[0]
    matrix[1, 2] += border - center[1]
    return cv2.warpAffine(
        canvas,
        matrix,
        (width + 2 * border, height + 2 * border),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert a grayscale fixture into the RGB array the library expects."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
