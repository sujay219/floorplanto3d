"""Opening models (doors and windows).

An opening is a *gap* in a wall: a span of the wall centreline where no wall
ink was present. Openings are therefore expressed relative to their host wall,
which is what a 3D engine needs in order to cut a hole.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from floorplanto3d.models.geometry import Point2D


class OpeningType(StrEnum):
    DOOR = "door"
    WINDOW = "window"


class Opening(BaseModel):
    """A gap in a wall."""

    id: str
    type: OpeningType
    wall_id: str
    start: Point2D
    end: Point2D
    confidence: float | None = None

    @property
    def width(self) -> float:
        """Width of the gap measured along the wall."""
        return self.start.distance_to(self.end)

    @property
    def center(self) -> Point2D:
        return self.start.midpoint(self.end)

    @property
    def angle_deg(self) -> float:
        """Angle of the opening; matches the angle of its host wall."""
        from floorplanto3d.models.geometry import Line2D

        return Line2D(start=self.start, end=self.end).angle_deg


class Door(Opening):
    """A door opening."""

    type: OpeningType = OpeningType.DOOR
    swing_direction: str | None = None
    sill_height: float | None = None


class Window(Opening):
    """A window opening."""

    type: OpeningType = OpeningType.WINDOW
    sill_height: float | None = None
    head_height: float | None = None