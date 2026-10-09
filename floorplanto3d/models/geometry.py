"""Geometric primitives for floor plan representation."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel


class Point2D(BaseModel):
    """A 2D point.

    Coordinates are in the floor plan's declared unit of measure. When a
    FloorPlan carries no scale, ``units`` is ``"px"`` and these are image
    pixel coordinates.
    """

    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        """Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def midpoint(self, other: Point2D) -> Point2D:
        """Midpoint between this point and another."""
        return Point2D(x=(self.x + other.x) / 2, y=(self.y + other.y) / 2)

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(x=self.x - other.x, y=self.y - other.y)

    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, factor: float) -> Point2D:
        return Point2D(x=self.x * factor, y=self.y * factor)

    __rmul__ = __mul__


class Line2D(BaseModel):
    """A 2D line segment."""

    start: Point2D
    end: Point2D

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def midpoint(self) -> Point2D:
        return self.start.midpoint(self.end)

    @property
    def dx(self) -> float:
        return self.end.x - self.start.x

    @property
    def dy(self) -> float:
        return self.end.y - self.start.y

    @property
    def angle_deg(self) -> float:
        """Angle of the segment in degrees, normalised to [0, 180)."""
        return math.degrees(math.atan2(self.dy, self.dx)) % 180.0

    @property
    def unit_vector(self) -> Point2D:
        """Unit vector pointing from start to end."""
        length = self.length
        if length == 0:
            return Point2D(x=0.0, y=0.0)
        return Point2D(x=self.dx / length, y=self.dy / length)

    @property
    def normal_vector(self) -> Point2D:
        """Unit normal vector (left-hand side of the direction of travel)."""
        unit = self.unit_vector
        return Point2D(x=-unit.y, y=unit.x)

    @property
    def is_degenerate(self) -> bool:
        return self.length == 0

    def offset(self, distance: float) -> Line2D:
        """Return a parallel segment displaced by ``distance`` along the normal."""
        shift = self.normal_vector * distance
        return Line2D(start=self.start + shift, end=self.end + shift)

    def distance_to_point(self, point: Point2D) -> float:
        """Perpendicular distance from ``point`` to the infinite line."""
        if self.is_degenerate:
            return self.start.distance_to(point)
        numerator = abs(
            self.dy * point.x - self.dx * point.y + self.end.x * self.start.y - self.end.y * self.start.x
        )
        return numerator / self.length

    def closest_point(self, point: Point2D) -> Point2D:
        """Point on the segment closest to ``point`` (clamped to the segment)."""
        if self.is_degenerate:
            return self.start
        t = ((point.x - self.start.x) * self.dx + (point.y - self.start.y) * self.dy) / (self.length**2)
        t = max(0.0, min(1.0, t))
        return Point2D(x=self.start.x + t * self.dx, y=self.start.y + t * self.dy)


class BoundingBox(BaseModel):
    """Axis-aligned bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with another box."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def to_points(self) -> list[Point2D]:
        return [
            Point2D(x=self.x1, y=self.y1),
            Point2D(x=self.x2, y=self.y1),
            Point2D(x=self.x2, y=self.y2),
            Point2D(x=self.x1, y=self.y2),
        ]


class Orientation(StrEnum):
    """Coarse classification of a wall's direction."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"


def classify_orientation(dx: float, dy: float, tolerance: float = 0.1) -> str:
    """Classify a direction as horizontal, vertical or diagonal.

    ``tolerance`` is the ratio of the minor to major component below which
    the segment is considered axis aligned.
    """
    ax, ay = abs(dx), abs(dy)
    major = max(ax, ay)
    minor = min(ax, ay)
    if major == 0:
        return Orientation.DIAGONAL.value
    if minor / major <= tolerance:
        return (
            Orientation.HORIZONTAL.value if ax >= ay else Orientation.VERTICAL.value
        )
    return Orientation.DIAGONAL.value