"""Wall model."""

from __future__ import annotations

from pydantic import BaseModel

from floorplanto3d.models.geometry import Line2D, Point2D, classify_orientation


class Wall(BaseModel):
    """A wall, represented by its centreline.

    ``thickness`` is ``None`` when it could not be measured from the image.
    No thickness is ever fabricated.
    """

    id: str
    start: Point2D
    end: Point2D
    thickness: float | None = None
    confidence: float | None = None

    @property
    def centerline(self) -> Line2D:
        return Line2D(start=self.start, end=self.end)

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def angle_deg(self) -> float:
        return self.centerline.angle_deg

    @property
    def orientation(self) -> str:
        return classify_orientation(self.end.x - self.start.x, self.end.y - self.start.y)

    @property
    def is_horizontal(self) -> bool:
        return self.orientation == "horizontal"

    @property
    def is_vertical(self) -> bool:
        return self.orientation == "vertical"

    def face_lines(self, half_thickness: float) -> tuple[Line2D, Line2D]:
        """The two parallel boundary lines of a wall of the given thickness."""
        centerline = self.centerline
        return centerline.offset(half_thickness), centerline.offset(-half_thickness)