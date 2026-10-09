"""Room model."""

from __future__ import annotations

from pydantic import BaseModel

from floorplanto3d.models.geometry import Point2D


class Room(BaseModel):
    """A closed region enclosed by walls."""

    id: str
    name: str | None = None
    polygon: list[Point2D]
    wall_ids: list[str] = []
    area: float | None = None
    confidence: float | None = None

    @property
    def computed_area(self) -> float:
        """Area computed via the shoelace formula."""
        if len(self.polygon) < 3:
            return 0.0
        total = 0.0
        count = len(self.polygon)
        for i in range(count):
            j = (i + 1) % count
            total += self.polygon[i].x * self.polygon[j].y
            total -= self.polygon[j].x * self.polygon[i].y
        return abs(total) / 2.0

    @property
    def perimeter(self) -> float:
        """Perimeter of the polygon (implicitly closed)."""
        if len(self.polygon) < 2:
            return 0.0
        total = 0.0
        count = len(self.polygon)
        for i in range(count):
            j = (i + 1) % count
            total += self.polygon[i].distance_to(self.polygon[j])
        return total

    @property
    def is_closed(self) -> bool:
        """Whether the first and last vertices coincide (or fewer than 3)."""
        if len(self.polygon) < 3:
            return False
        return self.polygon[0].distance_to(self.polygon[-1]) < 1e-6