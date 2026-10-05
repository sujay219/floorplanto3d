"""Object detection using traditional CV."""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a detected object."""

    class_name: str  # "wall", "door", "window"
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


def detect_objects(
    image: np.ndarray,
    min_wall_area: int = 1000,
    min_opening_area: int = 100,
) -> list[Detection]:
    """Detect walls, doors, and windows in floor plan image.

    Uses contour analysis and size heuristics to classify objects.

    Args:
        image: Input RGB image
        min_wall_area: Minimum contour area for wall detection
        min_opening_area: Minimum contour area for opening detection

    Returns:
        List of Detection objects
    """
    # Preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple thresholding approaches and combine
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_opening_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Classify based on size and aspect ratio
        if area >= min_wall_area:
            # Likely a wall - check if it's elongated
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                detections.append(Detection(
                    class_name="wall",
                    x1=x, y1=y, x2=x + w, y2=y + h,
                ))
            elif aspect_ratio > 1.5:
                # Could be a long wall segment
                detections.append(Detection(
                    class_name="wall",
                    x1=x, y1=y, x2=x + w, y2=y + h,
                ))
        else:
            # Smaller contours - likely doors or windows
            # Doors are typically taller, windows wider
            if aspect_ratio < 0.6:  # Taller than wide
                detections.append(Detection(
                    class_name="door",
                    x1=x, y1=y, x2=x + w, y2=y + h,
                ))
            else:
                detections.append(Detection(
                    class_name="window",
                    x1=x, y1=y, x2=x + w, y2=y + h,
                ))

    return detections


def detect_walls_by_line_detection(image: np.ndarray) -> list[Detection]:
    """Detect walls using line detection (HoughLines).

    Alternative method using probabilistic Hough transform.

    Args:
        image: Input RGB image

    Returns:
        List of wall detections
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10,
    )

    detections = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detections.append(Detection(
                class_name="wall",
                x1=min(x1, x2),
                y1=min(y1, y2),
                x2=max(x1, x2),
                y2=max(y1, y2),
            ))

    return detections