"""Regenerate the verification plans used in docs/api.md as example output."""

import cv2
import numpy as np


def two_room_plan(thickness: int = 8, door_gap: int = 40) -> np.ndarray:
    """Two rooms side by side, separated by a wall containing a door gap."""
    canvas = np.full((400, 600), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (550, 350), 0, thickness)
    cv2.line(canvas, (300, 50), (300, 350), 0, thickness)
    cv2.line(canvas, (300, 180), (300, 180 + door_gap), 255, thickness)
    return canvas


def four_room_plan(thickness: int = 8) -> np.ndarray:
    """Four rooms in a 2x2 grid with a door punched in each internal wall."""
    canvas = np.full((600, 800), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 60), (740, 540), 0, thickness)
    cv2.line(canvas, (400, 60), (400, 540), 0, thickness)
    cv2.line(canvas, (60, 300), (740, 300), 0, thickness)
    for x1, y1, x2, y2 in [
        (400, 150, 400, 210),
        (400, 380, 400, 440),
        (180, 300, 240, 300),
        (560, 300, 620, 300),
    ]:
        cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness)
    return canvas


def window_plan(thickness: int = 8) -> np.ndarray:
    """One room with two short wall gaps and one long gap."""
    canvas = np.full((400, 600), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (80, 80), (520, 320), 0, thickness)
    cv2.line(canvas, (150, 80), (180, 80), 255, thickness)
    cv2.line(canvas, (380, 80), (410, 80), 255, thickness)
    cv2.line(canvas, (250, 320), (320, 320), 255, thickness)
    return canvas


def blank_plan(width: int = 400, height: int = 300) -> np.ndarray:
    """A blank page containing no geometry."""
    return np.full((height, width), 255, dtype=np.uint8)


def skewed_plan(thickness: int = 8, angle: float = 3.0) -> np.ndarray:
    """A two-room plan rotated by a small angle, to exercise deskewing."""
    canvas = two_room_plan(thickness)
    height, width = canvas.shape
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
    """Convert a grayscale plan into the RGB array the library expects."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
